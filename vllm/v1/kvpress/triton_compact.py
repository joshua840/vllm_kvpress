# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place compaction of one request's paged KV cache with Triton."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compact_kernel(
    kv_ptr,  # (num_blocks, H, block_size, C)
    stride_b,
    stride_h,
    stride_n,
    block_ids_ptr,  # (H, num_request_blocks), row stride 0 when heads share blocks
    stride_bid,
    keep_ptr,  # (H, max_kept), sorted ascending per head
    n_kept_ptr,  # (H,) kept count per head
    max_kept,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,  # kept tokens per iteration
    C_TILE: tl.constexpr,  # channels per program
):
    """One program per (head, channel tile), walking the kept tokens in ascending order.

    In place is safe: keep[j] >= j, so slot i is only read as a source by tokens j <= i.
    The barrier makes every warp finish loading a tile (and so all earlier tiles)
    before any warp stores it, so every source is read before it is overwritten.
    """
    h = tl.program_id(0)
    c = tl.program_id(1) * C_TILE + tl.arange(0, C_TILE)
    head_ptr = kv_ptr + h * stride_h
    block_ids_ptr += h * stride_bid
    n_kept = tl.load(n_kept_ptr + h)
    for start in range(0, n_kept, TILE):
        i = start + tl.arange(0, TILE)
        mask = i < n_kept
        src = tl.load(keep_ptr + h * max_kept + i, mask=mask, other=0)
        src_block = tl.load(block_ids_ptr + src // BLOCK_SIZE, mask=mask, other=0)
        dst_block = tl.load(block_ids_ptr + i // BLOCK_SIZE, mask=mask, other=0)
        src_off = src_block.to(tl.int64) * stride_b + (src % BLOCK_SIZE) * stride_n
        dst_off = dst_block.to(tl.int64) * stride_b + (i % BLOCK_SIZE) * stride_n
        vals = tl.load(head_ptr + src_off[:, None] + c[None, :], mask=mask[:, None])
        tl.debug_barrier()
        tl.store(head_ptr + dst_off[:, None] + c[None, :], vals, mask=mask[:, None])


def compact(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    keep: torch.Tensor,
    n_kept: torch.Tensor | None = None,
) -> None:
    """Move each head's kept KV pairs to the front of the request's blocks, in place.

    kv_cache: (num_blocks, H, block_size, C), contiguous in C. With per-head paging
        it is a 1-head cache expanded to H heads (head stride 0).
    block_ids: (num_request_blocks,) block table row shared by all heads, or
        (H, num_request_blocks) with one row per head (per-head paging).
    keep: (H, max_kept) token indices, sorted ascending per head.
    n_kept: (H,) kept count per head, for ragged `keep` (default: all of `keep`).
    """
    num_heads, block_size, content = kv_cache.shape[1:]
    assert kv_cache.stride(3) == 1
    keep = keep.to(torch.int32).contiguous()
    if block_ids.dim() == 1:
        block_ids = block_ids.expand(num_heads, -1)
    if n_kept is None:
        n_kept = torch.full(
            (num_heads,), keep.shape[1], dtype=torch.int32, device=keep.device
        )
    # Tuned on an RTX A5000: 530-610 GB/s for 4k-16k tokens, 8 heads, C=256 (bf16).
    c_tile = 32
    _compact_kernel[(num_heads, triton.cdiv(content, c_tile))](
        kv_cache,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_ids,
        block_ids.stride(0),
        keep,
        n_kept.to(torch.int32),
        keep.shape[1],
        BLOCK_SIZE=block_size,
        TILE=128,
        C_TILE=c_tile,
        num_warps=8,
    )
