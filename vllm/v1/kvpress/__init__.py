# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-prefill KV cache eviction (kvpress-style presses) for the V2 model runner.

A request opts in with ``SamplingParams(extra_args={"kvpress": {...}})``:
    press              "snapkv" (only press for now)
    compression_ratio  fraction of the compressed span to evict (kvpress semantics)
    compress_len       leading prompt tokens to compress (default: the whole prompt)
    window_size        SnapKV observation window (default 64)
    kernel_size        SnapKV pooling kernel (default 5)

Category A only: every (layer, kv head) keeps the same number of tokens, so the standard
paged layout still works. Scheduler and worker derive the evicted count independently
from ``num_kept`` and must agree on it.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def get_config(sampling_params: Any) -> dict | None:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    cfg = extra_args.get("kvpress")
    if not cfg or cfg.get("compression_ratio", 0.0) <= 0:
        return None
    return cfg


def compress_len(cfg: dict, prompt_len: int) -> int:
    return min(cfg.get("compress_len", prompt_len), prompt_len)


def num_kept(cfg: dict, prompt_len: int) -> int:
    """Same rounding as kvpress ScorerPress.compress."""
    return int(compress_len(cfg, prompt_len) * (1 - cfg["compression_ratio"]))


def num_evicted(cfg: dict, prompt_len: int) -> int:
    return compress_len(cfg, prompt_len) - num_kept(cfg, prompt_len)


@dataclass
class PressEntry:
    """One request whose compressed span ends in the current step."""

    batch_idx: int  # row in attn_metadata.block_table / seq_lens
    query_end: int  # index after the last query token of this request in the flat batch
    seq_len: int  # KV length to compress (== compress_len)
    n_kept: int
    window_size: int
    kernel_size: int


# Set by the model runner around a forward pass, read by attention backends.
_plan: list[PressEntry] = []


def set_plan(plan: list[PressEntry]) -> None:
    global _plan
    _plan = plan


def get_plan() -> list[PressEntry]:
    return _plan


def snapkv_scores(
    queries: torch.Tensor, keys: torch.Tensor, window_size: int, kernel_size: int
) -> torch.Tensor:
    """SnapKV scores for one request, matching kvpress SnapKVPress.score.

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
    """Move each head's kept KV pairs to the front of the request's blocks.

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


def apply(
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
) -> None:
    """Score and compact the planned requests right after this layer's attention."""
    head_size = query.shape[-1]
    for e in _plan:
        block_ids = attn_metadata.block_table[e.batch_idx]
        n_blocks = math.ceil(e.seq_len / kv_cache.shape[2])
        keys = kv_cache[block_ids[:n_blocks], :, :, :head_size]
        keys = keys.permute(1, 0, 2, 3).reshape(kv_cache.shape[1], -1, head_size)[
            :, : e.seq_len
        ]
        scores = snapkv_scores(
            query[e.query_end - e.window_size : e.query_end],
            keys,
            e.window_size,
            e.kernel_size,
        )
        keep = scores.topk(e.n_kept, dim=-1).indices.sort(dim=-1).values
        compact(kv_cache, block_ids, e.seq_len, keep)
