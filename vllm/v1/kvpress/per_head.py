# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention over per-head paged KV (VLLM_KVPRESS_PAGING=head).

Each (layer, KV head) is its own KV cache group with 1-head pages, its own block
table and its own KV length. A layer's attention runs as one varlen call over
num_kv_heads x num_reqs sequences, each an MQA problem with G = num_heads /
num_kv_heads query heads and one KV head (the head-as-sequence design of
KV-Compress). Query tokens are reordered head-major so each sequence is contiguous.
The per-step inputs (block tables, KV lengths, slots, query offsets) come from
kvpress.make_step; a layer only slices its num_kv_heads groups out of them.
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


def _layer_groups(layer: torch.nn.Module) -> tuple[int, int]:
    layout = kvpress._layout
    assert layout is not None
    g0 = layout.layer_groups[layout.layer_index[layer.layer_name]][0]
    return g0, layout.num_kv_heads


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
    g0, num_kv_heads = _layer_groups(layer)
    num_tokens = step.num_tokens
    # (T, H, D) -> (H * T, 1, D), matching the head-major slot order. Copy even
    # when T == 1, where reshape returns a strided view that the cache op would
    # read as contiguous.
    key = key[:num_tokens].transpose(0, 1).contiguous().view(-1, 1, key.shape[-1])
    value = value[:num_tokens].transpose(0, 1).contiguous().view(-1, 1, value.shape[-1])
    key_cache, value_cache = _split_kv(kv_cache, impl.head_size)
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        step.slot_mappings[g0 : g0 + num_kv_heads].view(-1),
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
    g0, num_kv_heads = _layer_groups(layer)
    num_tokens = step.num_tokens
    group_size = impl.num_heads // num_kv_heads
    head_size = impl.head_size

    q = query[:num_tokens].view(num_tokens, num_kv_heads, group_size, head_size)
    q = q.transpose(0, 1).reshape(-1, group_size, head_size)
    key_cache, value_cache = _split_kv(kv_cache, head_size)
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=step.cu_seqlens_q,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=step.seq_lens[g0 : g0 + num_kv_heads].view(-1),
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=True,
        block_table=step.block_tables[g0 : g0 + num_kv_heads].flatten(0, 1),
        fa_version=impl.vllm_flash_attn_version,
    )
    output[:num_tokens].view(num_tokens, num_kv_heads, group_size, head_size).copy_(
        out.view(num_kv_heads, num_tokens, group_size, head_size).transpose(0, 1)
    )
    if kvpress.get_plan():
        kvpress.apply(layer, query, kv_cache, attn_metadata)
    return output
