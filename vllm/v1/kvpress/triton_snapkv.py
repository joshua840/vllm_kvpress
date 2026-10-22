# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SnapKV scoring that reads keys straight from the paged KV cache.

The torch reference gathers the keys, repeats them per query group, materializes the
(heads, window, seq_len) attention matrix in fp32 and reduces it. Here two kernels
stream key tiles from the cache instead, FlashAttention style:
  1. _softmax_stats_kernel: per (query head, window query, key split), the max and
     sum of exp over the visible keys (causal inside the window); splits are merged
     in torch.
  2. _window_mean_kernel: per (query head, key tile), recompute the logits, normalize
     with the stats and average over the window queries.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _load_keys(
    kv_ptr,
    block_ids_ptr,
    n,
    kv_head,
    stride_b,
    stride_h,
    stride_n,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    seq_len,
):
    n_mask = n < seq_len
    block = tl.load(block_ids_ptr + n // BLOCK_SIZE, mask=n_mask, other=0).to(tl.int64)
    offs = block * stride_b + kv_head * stride_h + (n % BLOCK_SIZE) * stride_n
    d = tl.arange(0, HEAD_DIM)
    return tl.load(kv_ptr + offs[:, None] + d[None, :], mask=n_mask[:, None], other=0.0)


@triton.jit
def _softmax_stats_kernel(
    q_ptr,  # (WINDOW, num_heads, HEAD_DIM), post-RoPE
    stride_qw,
    stride_qh,
    kv_ptr,  # (num_blocks, num_kv_heads, BLOCK_SIZE, C), K in C[:HEAD_DIM]
    stride_b,
    stride_h,
    stride_n,
    block_ids_ptr,  # (num_kv_heads, num_request_blocks)
    stride_bid,
    m_ptr,  # (num_heads, num_splits, WINDOW) fp32, partial max
    l_ptr,  # (num_heads, num_splits, WINDOW) fp32, partial sum of exp
    seq_len,
    scale,
    GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,  # keys per program (split-K, like flash-decoding)
):
    h = tl.program_id(0)
    split = tl.program_id(1)
    w = tl.arange(0, WINDOW)
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + w[:, None] * stride_qw + h * stride_qh + d[None, :])
    q_pos = seq_len - WINDOW + w
    m = tl.full([WINDOW], float("-inf"), tl.float32)
    row_sum = tl.zeros([WINDOW], tl.float32)
    for start in range(
        split * SPLIT_N, tl.minimum(seq_len, (split + 1) * SPLIT_N), BLOCK_N
    ):
        n = start + tl.arange(0, BLOCK_N)
        k = _load_keys(
            kv_ptr,
            block_ids_ptr + (h // GROUPS) * stride_bid,
            n,
            h // GROUPS,
            stride_b,
            stride_h,
            stride_n,
            BLOCK_SIZE,
            BLOCK_N,
            HEAD_DIM,
            seq_len,
        )
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(
            (n[None, :] <= q_pos[:, None]) & (n[None, :] < seq_len), s, float("-inf")
        )
        m_new = tl.maximum(m, tl.max(s, axis=1))
        # A split may see no visible key for a query (all -inf): keep row_sum at 0 there
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))
        p = tl.where(m_new[:, None] == float("-inf"), 0.0, tl.exp(s - m_new[:, None]))
        row_sum = row_sum * alpha + tl.sum(p, axis=1)
        m = m_new
    out = (h * tl.num_programs(1) + split) * WINDOW + w
    tl.store(m_ptr + out, m)
    tl.store(l_ptr + out, row_sum)


@triton.jit
def _window_mean_kernel(
    q_ptr,
    stride_qw,
    stride_qh,
    kv_ptr,
    stride_b,
    stride_h,
    stride_n,
    block_ids_ptr,
    stride_bid,
    m_ptr,
    l_ptr,
    out_ptr,  # (num_heads, seq_len - WINDOW) fp32
    seq_len,
    scale,
    GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    h = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_scored = (
        seq_len - WINDOW
    )  # keys before the window are visible to every window query
    w = tl.arange(0, WINDOW)
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + w[:, None] * stride_qw + h * stride_qh + d[None, :])
    k = _load_keys(
        kv_ptr,
        block_ids_ptr + (h // GROUPS) * stride_bid,
        n,
        h // GROUPS,
        stride_b,
        stride_h,
        stride_n,
        BLOCK_SIZE,
        BLOCK_N,
        HEAD_DIM,
        n_scored,
    )
    m = tl.load(m_ptr + h * WINDOW + w)
    row_sum = tl.load(l_ptr + h * WINDOW + w)
    p = tl.exp(tl.dot(q, tl.trans(k)) * scale - m[:, None]) / row_sum[:, None]
    tl.store(out_ptr + h * n_scored + n, tl.sum(p, axis=0) / WINDOW, mask=n < n_scored)


def snapkv_scores(
    queries: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
    kernel_size: int = 5,
) -> torch.Tensor:
    """SnapKV scores for one request, same output as kvpress SnapKVPress.score.

    queries: (window_size, num_heads, head_size), post-RoPE.
    kv_cache: (num_blocks, num_kv_heads, block_size, 2 * head_size); head stride 0
        for a per-head paged cache.
    block_ids: (num_request_blocks,) shared by all heads, or (num_kv_heads, ...).
    Returns (num_kv_heads, seq_len); the window gets max + 1 so it is always kept.
    """
    window, num_heads, head_dim = queries.shape
    num_kv_heads, block_size = kv_cache.shape[1], kv_cache.shape[2]
    groups = num_heads // num_kv_heads
    if block_ids.dim() == 1:
        block_ids = block_ids.expand(num_kv_heads, -1)
    n_scored = seq_len - window
    split_n = 512
    num_splits = triton.cdiv(seq_len, split_n)
    m = torch.empty(
        num_heads, num_splits, window, dtype=torch.float32, device=queries.device
    )
    row_sum = torch.empty_like(m)
    out = torch.empty(num_heads, n_scored, dtype=torch.float32, device=queries.device)
    args = (
        kv_cache,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_ids,
        block_ids.stride(0),
    )
    meta = dict(GROUPS=groups, BLOCK_SIZE=block_size, WINDOW=window, HEAD_DIM=head_dim)
    scale = head_dim**-0.5
    _softmax_stats_kernel[(num_heads, num_splits)](
        queries,
        queries.stride(0),
        queries.stride(1),
        *args,
        m,
        row_sum,
        seq_len,
        scale,
        BLOCK_N=64,
        SPLIT_N=split_n,
        **meta,
    )
    # Combine the splits: M = max_s m_s, L = sum_s l_s * exp(m_s - M)
    m_all = m.amax(dim=1)
    row_sum = (row_sum * torch.exp(m - m_all[:, None])).nan_to_num_(0.0).sum(dim=1)
    m = m_all.contiguous()
    _window_mean_kernel[(num_heads, triton.cdiv(n_scored, 64))](
        queries,
        queries.stride(0),
        queries.stride(1),
        *args,
        m,
        row_sum,
        out,
        seq_len,
        scale,
        BLOCK_N=64,
        **meta,
    )
    scores = F.avg_pool1d(
        out, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    )
    scores = scores.view(num_kv_heads, groups, n_scored).mean(1)
    return F.pad(scores, (0, window), value=scores.max().item() + 1)
