# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-prefill KV cache eviction (kvpress-style presses) for the V2 model runner.

A request opts in with ``SamplingParams(extra_args={"kvpress": {...}})``:
    press              a key of PRESS_PAGING
    scorer             scorer of a wrapper press (adakv, chunk, block, chunkkv,
                       entropy_gated_chunkkv, per_layer), default snapkv
    compression_ratio  fraction of the compressed span to evict (kvpress semantics)
    compression_ratios per_layer: one ratio per layer
    compress_len       leading prompt tokens to compress (default: the whole prompt)
    window_size        SnapKV observation window (default 64)
    kernel_size        SnapKV pooling kernel (default 5)
    beta               PyramidKV pyramid steepness (default 20)
    alpha_safeguard    AdaKV minimum kept fraction per head (default 0.2)
    chunk_length, block_size, seed, n_sink, lag_size, lazy_threshold, ...
                       same names and defaults as the kvpress press

How much can differ is set once per engine by ``VLLM_KVPRESS_PAGING``:
    shared  one KV cache group for all layers (vLLM default): category A only
    layer   one KV cache group per layer: categories A and B
    head    one KV cache group per (layer, KV head): categories A, B and C

The worker evicts in the forward pass where the compressed span ends and reports
the kept count of every KV cache group back to the scheduler, which frees the
emptied tail blocks (``ModelRunnerOutput.kvpress_num_kept``).
"""

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from vllm.v1.kvpress import scorers, triton_compact, triton_window_attention
from vllm.v1.kvpress.scorers import query_window

_ANY = ("shared", "layer", "head")
PRESS_PAGING = {
    **dict.fromkeys(scorers.SCORERS, _ANY),
    "chunk": _ANY,
    "block": _ANY,
    "entropy_gated_chunkkv": _ANY,
    # Kept counts differ across layers
    "pyramidkv": ("layer", "head"),
    "per_layer": ("layer", "head"),
    "chunkkv": ("layer", "head"),
    "simlayerkv": ("layer", "head"),
    # Kept counts differ across heads
    "adakv": ("head",),
}
# Kept counts depend on the heads a rank holds
_SINGLE_RANK = ("adakv", "chunkkv", "simlayerkv")


def paging() -> str:
    return os.environ.get("VLLM_KVPRESS_PAGING", "shared")


def get_config(sampling_params: Any) -> dict | None:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    cfg = extra_args.get("kvpress")
    if not cfg:
        return None
    press = cfg.get("press", "snapkv")
    if not (
        (cfg.get("compression_ratio") or 0.0) > 0
        or any(cfg.get("compression_ratios") or ())
        or press == "simlayerkv"
    ):
        return None
    if paging() not in PRESS_PAGING.get(press, ()):
        raise ValueError(
            f"kvpress press {press!r} needs VLLM_KVPRESS_PAGING in "
            f"{PRESS_PAGING.get(press)}, got {paging()!r}"
        )
    name = scorers.scorer_name(cfg)
    if press in scorers.WRAPPERS and name not in (
        scorers.KV_SCORERS if press in ("chunk", "block") else scorers.SCORERS
    ):
        raise ValueError(f"kvpress press {press!r} cannot wrap scorer {name!r}")
    return cfg


def validate(sampling_params: Any, vllm_config: Any) -> None:
    """Reject a kvpress request on engine settings that silently break eviction
    (results/compat: their RULER scores collapse)."""
    if (cfg := get_config(sampling_params)) is None:
        return
    cache, backend = vllm_config.cache_config, vllm_config.attention_config.backend
    unsupported = {
        f"press={cfg.get('press')} with tensor parallelism (each rank selects its "
        "own heads, but the scheduler frees blocks by rank 0's kept counts)":
            cfg.get("press") in _SINGLE_RANK
            and vllm_config.parallel_config.tensor_parallel_size > 1,
        "prefix caching (hashes of compressed blocks no longer match their tokens)":
            cache.enable_prefix_caching,
        "async scheduling (the next step is scheduled before the kept counts arrive)":
            vllm_config.scheduler_config.async_scheduling,
        f"kv_cache_dtype={cache.cache_dtype} (kernels read bf16/fp16 caches)":
            cache.cache_dtype != "auto",
        f"attention backend {backend.name if backend else ''} (hooks are in "
        "FlashAttention only)": backend is not None and backend.name != "FLASH_ATTN",
    }
    if bad := [name for name, on in unsupported.items() if on]:
        raise ValueError("kvpress does not support " + ", ".join(bad))


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
    """One group per layer, or per (layer, KV head) with 1-head pages.

    With per-head paging only the head-0 group of a layer lists the layer name, so
    vLLM allocates one cache view and builds one attention metadata per layer; the
    other groups only own block tables and slot mappings.
    """
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    names = sorted(kv_cache_spec, key=extract_layer_index)
    if mode == "layer":
        return [KVCacheGroupSpec([n], kv_cache_spec[n]) for n in names]
    groups = []
    for n in names:
        spec = kv_cache_spec[n]
        head_spec = dataclasses.replace(spec, num_kv_heads=1)
        groups.append(KVCacheGroupSpec([n], head_spec))
        groups += [
            KVCacheGroupSpec([], head_spec) for _ in range(spec.num_kv_heads - 1)
        ]
    return groups


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

    mode = paging()
    heads = num_kv_heads if mode == "head" else 1
    groups: dict[str, list[int]] = {}
    for g, group in enumerate(kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            groups[name] = list(range(g, g + heads))
    names = sorted(groups, key=extract_layer_index)
    _layout = Layout(
        paging=mode,
        num_kv_heads=num_kv_heads,
        num_groups=len(kv_cache_config.kv_cache_groups),
        layer_index={n: i for i, n in enumerate(names)},
        layer_groups=[groups[n] for n in names],
    )
    return _layout


def per_head_paging() -> bool:
    return _layout is not None and _layout.paging == "head"


_attn_seq_lens: torch.Tensor | None = None


def init_attn_seq_lens(max_num_reqs: int, device: torch.device) -> None:
    global _attn_seq_lens
    assert _layout is not None
    shape = (_layout.num_groups, max_num_reqs)
    if _attn_seq_lens is None or _attn_seq_lens.shape != shape:
        _attn_seq_lens = torch.zeros(shape, dtype=torch.int32, device=device)


def attn_seq_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    """Copy of seq_lens with one row per KV cache group, for attention to subtract
    evicted pairs from. Always the same buffer: FULL CUDA graphs replay the
    addresses seen at capture, whose inputs are built outside the model runner
    (cudagraph_utils.prepare_inputs_to_capture)."""
    assert _attn_seq_lens is not None
    out = _attn_seq_lens[:, : seq_lens.shape[0]]
    out.copy_(seq_lens.expand_as(out))
    return out


# ---------------------------------------------------------------------------
# Per-step plan


@dataclass
class PressEntry:
    """One request whose compressed span ends in the current step."""

    batch_idx: int  # row in the block tables / seq_lens of this step
    query_end: int  # index after the last query token of this request in the flat batch
    seq_len: int  # KV length to compress (== compress_len)
    cfg: dict


@dataclass
class StepContext:
    """Per-layer inputs of this step for per-head paging, built once per step so
    each layer only indexes them (see per_head.py). A layer's attention runs over
    num_kv_heads * num_reqs sequences, sequence h * num_reqs + r holding request
    r's queries for KV head h. New K/V slots are token-major (token t, head h at
    t * num_kv_heads + h), the order of the K/V tensors, so writing them needs no
    reordering. Views of persistent buffers: FULL CUDA graphs replay the
    addresses seen at capture."""

    block_tables: torch.Tensor  # (num_layers, num_seqs, max_blocks)
    seq_lens: torch.Tensor  # (num_layers, num_seqs) KV length per sequence
    slot_mappings: torch.Tensor  # (num_layers, num_tokens * num_kv_heads)
    cu_seqlens_q: torch.Tensor  # (num_seqs + 1,)
    num_reqs: int
    num_tokens: int

    def request_block_tables(self, layer_idx: int, req_idx: int) -> torch.Tensor:
        """(num_kv_heads, max_blocks) block tables of one request in one layer."""
        table = self.block_tables[layer_idx]
        return table.view(-1, self.num_reqs, table.shape[-1])[:, req_idx]


# Persistent buffers behind StepContext (per-head paging only).
_step_buffers: dict[str, torch.Tensor] = {}


def init_step_buffers(
    max_num_reqs: int, max_num_tokens: int, max_blocks: int, device: torch.device
) -> None:
    assert _layout is not None
    groups, heads, rows = _layout.num_groups, _layout.num_kv_heads, max_num_reqs
    i32, i64 = torch.int32, torch.int64
    _step_buffers.update(
        block_tables=torch.zeros(groups * rows * max_blocks, dtype=i32, device=device),
        seq_lens=torch.zeros(groups * rows, dtype=i32, device=device),
        slot_mappings=torch.zeros(groups * max_num_tokens, dtype=i64, device=device),
        cu_seqlens_q=torch.zeros(heads * rows + 1, dtype=i32, device=device),
        head_offsets=torch.arange(heads, dtype=i32, device=device)[:, None],
    )


def make_step(
    block_tables: torch.Tensor,
    slot_mappings: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
) -> StepContext:
    """block_tables: (num_groups, >= num_reqs, max_blocks); seq_lens: (num_groups,
    >= num_reqs) or (>= num_reqs,) when no group has evicted anything;
    query_start_loc: (>= num_reqs + 1,).

    Under a FULL CUDA graph num_reqs and num_tokens are the padded sizes: padded
    requests have no queries except the last, which takes the padded tokens, and
    no KV (seq_len 0)."""
    assert _layout is not None
    layers, heads, rows = _layout.num_layers, _layout.num_kv_heads, num_reqs
    max_blocks = block_tables.shape[-1]
    seqs = heads * rows
    buf = _step_buffers
    step_block_tables = buf["block_tables"][: layers * seqs * max_blocks]
    step_block_tables.view(layers * heads, rows, max_blocks).copy_(
        block_tables[:, :rows]
    )
    step_seq_lens = buf["seq_lens"][: layers * seqs]
    step_seq_lens.view(layers * heads, rows).copy_(
        seq_lens[..., :rows].expand(layers * heads, -1)
    )
    # Token-major slots: (layer, head, token) -> (layer, token, head)
    step_slots = buf["slot_mappings"][: layers * num_tokens * heads]
    step_slots.view(layers, num_tokens, heads).copy_(
        slot_mappings[:, :num_tokens].view(layers, heads, num_tokens).transpose(1, 2)
    )
    # Sequence (h, r) holds request r's queries for KV head h at h * T + qsl[r]
    cu_seqlens_q = buf["cu_seqlens_q"][: seqs + 1]
    torch.add(
        query_start_loc[:rows][None, :],
        buf["head_offsets"] * num_tokens,
        out=cu_seqlens_q[:-1].view(heads, rows),
    )
    cu_seqlens_q[-1:].fill_(heads * num_tokens)
    return StepContext(
        block_tables=step_block_tables.view(layers, seqs, max_blocks),
        seq_lens=step_seq_lens.view(layers, seqs),
        slot_mappings=step_slots.view(layers, num_tokens * heads),
        cu_seqlens_q=cu_seqlens_q,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
    )


# Set by the model runner around a forward pass, read by attention backends.
_plan: list[PressEntry] = []
_kept: torch.Tensor | None = None  # (len(_plan), num_groups) kept count per group
_step: StepContext | None = None


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


def set_step(step: StepContext | None) -> None:
    global _step
    _step = step


# ---------------------------------------------------------------------------
# Torch references (used in tests)


def window_attention_scores(
    queries: torch.Tensor, keys: torch.Tensor, window_size: int, kernel_size: int
) -> torch.Tensor:
    """Torch reference of triton_window_attention.window_attention_scores.

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


def adakv_mask(scores: torch.Tensor, n_kept: int, alpha: float) -> torch.Tensor:
    """kvpress AdaKVPress: top-k across heads after a per-head safeguard."""
    num_heads, k_len = scores.shape
    n_safe = int(n_kept * alpha)
    if alpha > 0:
        n_safe = max(1, n_safe) if n_kept > 0 else 0
    scores = scores.scatter(
        -1, scores.topk(n_safe, dim=-1).indices, torch.finfo(scores.dtype).max
    )
    mask = torch.zeros(num_heads * k_len, dtype=torch.bool, device=scores.device)
    mask[scores.view(-1).topk(num_heads * n_kept).indices] = True
    return mask.view(num_heads, k_len)


def mask_to_keep(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(H, n) bool -> kept indices first, ascending, per head; and kept counts."""
    keep = torch.argsort((~mask).to(torch.int8), dim=-1, stable=True)
    return keep, mask.sum(dim=-1, dtype=torch.int32)


def select(
    cfg: dict,
    queries: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One request's kept KV pairs in one layer: (keep, n_kept) for compact."""
    assert _layout is not None
    press = cfg.get("press", "snapkv")
    if press == "per_layer":
        press = scorers.scorer_name(cfg)
        ratio = cfg["compression_ratios"][layer_idx]
        cfg = cfg | {"press": press, "compression_ratio": ratio}
    if press == "simlayerkv":
        return mask_to_keep(
            scorers.simlayerkv_mask(cfg, queries, kv_cache, block_ids, seq_len)
        )
    budget = layer_budget(cfg, seq_len, layer_idx, _layout.num_layers)
    if press in ("chunk", "block"):
        keys, values = scorers.read_kv(kv_cache, block_ids, seq_len)
        if press == "chunk":
            keep = scorers.chunk_keep(cfg, keys, values)
        else:
            keep = scorers.block_keep(cfg, keys, values, max(1, budget))
    else:
        s = scorers.scores(cfg, queries, kv_cache, block_ids, seq_len)
        if press == "adakv":
            return mask_to_keep(adakv_mask(s, budget, cfg.get("alpha_safeguard", 0.2)))
        if press == "chunkkv":
            return mask_to_keep(scorers.chunkkv_mask(cfg, s))
        if press == "entropy_gated_chunkkv":
            return mask_to_keep(scorers.entropy_gated_chunkkv_mask(cfg, s))
        keep = s.topk(budget, dim=-1).indices.sort(dim=-1).values
    return keep, torch.full_like(keep[:, 0], keep.shape[1], dtype=torch.int32)


def apply(
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
) -> None:
    """Score and compact the planned requests right after this layer's attention."""
    assert _layout is not None and _kept is not None
    layer_idx = _layout.layer_index[layer.layer_name]
    groups = _layout.layer_groups[layer_idx]
    num_kv_heads = _layout.num_kv_heads
    if per_head_paging():
        assert _step is not None
        # 1-head pages: every head reads the same cache view with its own blocks
        kv_cache = kv_cache.expand(-1, num_kv_heads, -1, -1)
    for row, e in enumerate(_plan):
        if per_head_paging():
            block_ids = _step.request_block_tables(layer_idx, e.batch_idx)
        else:
            block_ids = attn_metadata.block_table[e.batch_idx]
        window = query_window(e.cfg)
        keep, n_kept = select(
            e.cfg,
            query[e.query_end - window : e.query_end],
            kv_cache,
            block_ids,
            e.seq_len,
            layer_idx,
        )
        triton_compact.compact(kv_cache, block_ids, keep, n_kept)
        # Without per-head paging all heads of a layer share one group (and count)
        _kept[row, groups] = n_kept if len(groups) > 1 else n_kept[:1]
