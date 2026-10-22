# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-prefill KV cache eviction (kvpress-style presses) for the V2 model runner.

A request opts in with ``SamplingParams(extra_args={"kvpress": {...}})``:
    press              "snapkv" (uniform) or "pyramidkv" (per layer)
    compression_ratio  fraction of the compressed span to evict (kvpress semantics)
    compress_len       leading prompt tokens to compress (default: the whole prompt)
    window_size        SnapKV observation window (default 64)
    kernel_size        SnapKV pooling kernel (default 5)
    beta               PyramidKV pyramid steepness (default 20)

How much can differ is set once per engine by ``VLLM_KVPRESS_PAGING``:
    shared  one KV cache group for all layers (vLLM default): category A only
    layer   one KV cache group per layer: categories A and B

The worker evicts in the forward pass where the compressed span ends and reports
the kept count of every KV cache group back to the scheduler, which frees the
emptied tail blocks (``ModelRunnerOutput.kvpress_num_kept``).
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm.v1.kvpress import triton_compact, triton_snapkv

PRESS_PAGING = {
    "snapkv": ("shared", "layer"),
    "pyramidkv": ("layer",),
}


def paging() -> str:
    return os.environ.get("VLLM_KVPRESS_PAGING", "shared")


def get_config(sampling_params: Any) -> dict | None:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    cfg = extra_args.get("kvpress")
    if not cfg or cfg.get("compression_ratio", 0.0) <= 0:
        return None
    press = cfg.get("press", "snapkv")
    if paging() not in PRESS_PAGING.get(press, ()):
        raise ValueError(
            f"kvpress press {press!r} needs VLLM_KVPRESS_PAGING in "
            f"{PRESS_PAGING.get(press)}, got {paging()!r}"
        )
    return cfg


def compress_len(cfg: dict, prompt_len: int) -> int:
    return min(cfg.get("compress_len", prompt_len), prompt_len)


def layer_budget(cfg: dict, k_len: int, layer_idx: int, num_layers: int) -> int:
    """KV pairs kept per head in one layer (kvpress rounding)."""
    ratio = cfg["compression_ratio"]
    if cfg.get("press", "snapkv") != "pyramidkv":
        return int(k_len * (1 - ratio))
    # kvpress PyramidKVPress.get_layer_budget
    window, beta = cfg.get("window_size", 64), cfg.get("beta", 20)
    max_capacity = window + k_len * (1 - ratio)
    min_num = (max_capacity - window) / beta
    max_num = (max_capacity - window) * 2 - min_num
    if max_num >= k_len - window:
        max_num = k_len - window
        min_num = (max_capacity - window) * 2 - max_num
    if not (k_len >= max_num >= min_num >= window):
        return max(1, round(k_len * (1 - ratio)))
    steps = (max_num - min_num) / (num_layers - 1) if num_layers > 1 else 0.0
    return round(max_num - layer_idx * steps)


# ---------------------------------------------------------------------------
# KV cache groups


def make_kv_cache_groups(kv_cache_spec: dict, mode: str) -> list:
    """One KV cache group per layer, so each layer has its own block table."""
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    assert mode == "layer"
    names = sorted(kv_cache_spec, key=extract_layer_index)
    return [KVCacheGroupSpec([n], kv_cache_spec[n]) for n in names]


@dataclass
class Layout:
    paging: str
    num_kv_heads: int
    num_groups: int
    layer_index: dict[str, int]  # attention layer name -> layer position
    layer_groups: list[list[int]]  # layer position -> its KV cache group ids

    @property
    def num_layers(self) -> int:
        return len(self.layer_groups)


_layout: Layout | None = None


def init_layout(kv_cache_config: Any, num_kv_heads: int) -> Layout:
    global _layout
    from vllm.model_executor.models.utils import extract_layer_index

    groups: dict[str, list[int]] = {}
    for g, group in enumerate(kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            groups[name] = [g]
    names = sorted(groups, key=extract_layer_index)
    _layout = Layout(
        paging=paging(),
        num_kv_heads=num_kv_heads,
        num_groups=len(kv_cache_config.kv_cache_groups),
        layer_index={n: i for i, n in enumerate(names)},
        layer_groups=[groups[n] for n in names],
    )
    return _layout


# ---------------------------------------------------------------------------
# Per-step plan


@dataclass
class PressEntry:
    """One request whose compressed span ends in the current step."""

    batch_idx: int  # row in the block tables / seq_lens of this step
    query_end: int  # index after the last query token of this request in the flat batch
    seq_len: int  # KV length to compress (== compress_len)
    cfg: dict


# Set by the model runner around a forward pass, read by attention backends.
_plan: list[PressEntry] = []
_kept: torch.Tensor | None = None  # (len(_plan), num_groups) kept count per group


def set_plan(plan: list[PressEntry], device: torch.device | None = None) -> None:
    global _plan, _kept
    _plan = plan
    _kept = None
    if plan:
        assert _layout is not None
        _kept = torch.zeros(
            len(plan), _layout.num_groups, dtype=torch.int32, device=device
        )


def get_plan() -> list[PressEntry]:
    return _plan


def take_kept() -> list[list[int]]:
    """Kept KV count per (plan entry, group) after the forward pass."""
    assert _kept is not None
    return _kept.tolist()


# ---------------------------------------------------------------------------
# Torch references (used in tests)


def snapkv_scores(
    queries: torch.Tensor, keys: torch.Tensor, window_size: int, kernel_size: int
) -> torch.Tensor:
    """Torch reference of triton_snapkv.snapkv_scores.

    SnapKV scores for one request, matching kvpress SnapKVPress.score.

    queries: (window_size, num_heads, head_size), post-RoPE.
    keys: (num_kv_heads, seq_len, head_size), post-RoPE.
    Returns (num_kv_heads, seq_len); the window gets max + 1 so it is always kept.
    """
    num_heads, (num_kv_heads, seq_len, head_size) = queries.shape[1], keys.shape
    groups = num_heads // num_kv_heads
    q = queries.permute(1, 0, 2)
    k = keys.repeat_interleave(groups, dim=0)
    attn = q @ k.transpose(1, 2) / math.sqrt(head_size)
    mask = torch.full_like(attn, float("-inf")).triu(diagonal=seq_len - window_size + 1)
    attn = F.softmax(attn + mask, dim=-1, dtype=torch.float32).to(queries.dtype)[
        ..., :-window_size
    ]
    scores = F.avg_pool1d(
        attn.mean(dim=1), kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    )
    scores = scores.view(num_kv_heads, groups, -1).mean(1)
    return F.pad(scores, (0, window_size), value=scores.max().item() + 1)


def compact(
    kv_cache: torch.Tensor, block_ids: torch.Tensor, seq_len: int, keep: torch.Tensor
) -> None:
    """Torch reference of triton_compact.compact.

    kv_cache: (num_blocks, num_kv_heads, block_size, C) with K and V packed in C.
    keep: (num_kv_heads, n_kept) token indices, sorted per head.
    """
    num_kv_heads, block_size = kv_cache.shape[1], kv_cache.shape[2]
    heads = torch.arange(num_kv_heads, device=kv_cache.device)[:, None]
    src = block_ids[keep // block_size], keep % block_size  # (H, k) each
    kept = kv_cache[
        src[0], heads, src[1]
    ]  # advanced indexing copies, so the write below is safe
    dst = torch.arange(keep.shape[1], device=kv_cache.device)
    kv_cache[
        block_ids[dst // block_size][None, :], heads, (dst % block_size)[None, :]
    ] = kept


# ---------------------------------------------------------------------------
# Eviction


def apply(
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
) -> None:
    """Score and compact the planned requests right after this layer's attention."""
    assert _layout is not None and _kept is not None
    layer_idx = _layout.layer_index[layer.layer_name]
    (group,) = _layout.layer_groups[layer_idx]
    for row, e in enumerate(_plan):
        block_ids = attn_metadata.block_table[e.batch_idx]
        window = e.cfg.get("window_size", 64)
        scores = triton_snapkv.snapkv_scores(
            query[e.query_end - window : e.query_end],
            kv_cache,
            block_ids,
            e.seq_len,
            e.cfg.get("kernel_size", 5),
        )
        budget = layer_budget(e.cfg, e.seq_len, layer_idx, _layout.num_layers)
        keep = scores.topk(budget, dim=-1).indices.sort(dim=-1).values
        triton_compact.compact(kv_cache, block_ids, keep)
        # With shared paging every layer writes the same group and budget
        _kept[row, group] = budget
