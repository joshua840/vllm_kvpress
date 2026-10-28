# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention over per-head paged KV (VLLM_KVPRESS_PAGING=head).

Each (layer, KV head) is its own KV cache group with 1-head pages, its own block
table and its own KV length. A layer's attention runs as one varlen call over
num_kv_heads x num_reqs sequences, each an MQA problem with G = num_heads /
num_kv_heads query heads and one KV head (the head-as-sequence design of
KV-Compress). Query tokens are reordered head-major so each sequence is contiguous.
With AdaKV lengths head-major order is also faster in FlashAttention than
request-major (3.7 vs 4.4 ms per decode step, Qwen3-0.6B, 16 requests): a head's
KV length is similar across requests, so request-major order puts long sequences
next to each other. New K/V are written in their own token-major order. The per-step inputs (block tables, KV lengths, slots, query offsets) come
from kvpress.make_step; a layer only indexes its row of them.
"""

from typing import Any

import torch

from vllm.v1 import kvpress
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    reshape_and_cache_flash,
)


# id(kv_cache) -> (key_cache, value_cache). The cache is a tensor subclass, so each
# transpose / split goes through __torch_function__ (about 25 us); split it once.
_kv_views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _split_kv(
    kv_cache: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    views = _kv_views.get(id(kv_cache))
    if views is None or views[0].data_ptr() != kv_cache.data_ptr():
        views = kv_cache.transpose(1, 2).split(head_size, dim=-1)
        _kv_views[id(kv_cache)] = views
    return views


def _layer_index(layer: torch.nn.Module) -> tuple[int, int]:
    layout = kvpress._layout
    assert layout is not None
    return layout.layer_index[layer.layer_name], layout.num_kv_heads


def kv_cache_update(
    impl: Any,
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
) -> None:
    """Write each head's new K/V to the slots of its own group."""
    step = kvpress._step
    if step is None:
        return
    layer_idx, _ = _layer_index(layer)
    num_tokens = step.num_tokens
    head_size = key.shape[-1]
    # (T, H, D) -> (T * H, 1, D) in the token-major slot order: a view when the
    # tokens are contiguous (K after the QK norm), a copy otherwise (V, a slice of
    # the fused QKV projection)
    key = key[:num_tokens].reshape(-1, 1, head_size)
    value = value[:num_tokens].reshape(-1, 1, head_size)
    key_cache, value_cache = _split_kv(kv_cache, impl.head_size)
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        step.slot_mappings[layer_idx],
        impl.kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )


def forward(
    impl: Any,
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    output: torch.Tensor,
) -> torch.Tensor:
    step = kvpress._step
    assert step is not None
    layer_idx, num_kv_heads = _layer_index(layer)
    num_tokens = step.num_tokens
    group_size = impl.num_heads // num_kv_heads
    head_size = impl.head_size

    q = query[:num_tokens].view(num_tokens, num_kv_heads, group_size, head_size)
    q = q.transpose(0, 1).reshape(-1, group_size, head_size)
    out = torch.empty_like(q)
    key_cache, value_cache = _split_kv(kv_cache, head_size)
    flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=step.cu_seqlens_q,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=step.seq_lens[layer_idx],
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=True,
        block_table=step.block_tables[layer_idx],
        fa_version=impl.vllm_flash_attn_version,
    )
    output[:num_tokens].view(num_tokens, num_kv_heads, group_size, head_size).copy_(
        out.view(num_kv_heads, num_tokens, group_size, head_size).transpose(0, 1)
    )
    if kvpress.get_plan():
        kvpress.apply(layer, query, kv_cache, attn_metadata)
    return output
