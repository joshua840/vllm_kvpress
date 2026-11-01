# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scores and kept sets of the kvpress presses computed from the paged KV cache.

Same results as kvpress for one request: tensors drop the batch dimension, so keys
and values are (num_kv_heads, seq_len, head_size) and scores (num_kv_heads, seq_len).
"""

import math

import torch
import torch.nn.functional as F

from vllm.v1.kvpress import triton_window_attention

EPSILON = 1e-12  # kvpress entropy_gated_chunkkv_press.py


def read_kv(
    kv_cache: torch.Tensor, block_ids: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy one request's keys and values out of the cache (see triton_compact)."""
    num_kv_heads, block_size = kv_cache.shape[1], kv_cache.shape[2]
    if block_ids.dim() == 1:
        block_ids = block_ids.expand(num_kv_heads, -1)
    pos = torch.arange(seq_len, device=kv_cache.device)
    heads = torch.arange(num_kv_heads, device=kv_cache.device)[:, None]
    kv = kv_cache[block_ids[:, pos // block_size].long(), heads, pos % block_size]
    return kv.chunk(2, dim=-1)


def random_scores(cfg: dict, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    generator = None
    if (seed := cfg.get("seed")) is not None:
        generator = torch.Generator(device=keys.device).manual_seed(seed)
    return torch.rand(
        *keys.shape[:-1], generator=generator, device=keys.device, dtype=keys.dtype
    )


def knorm_scores(cfg: dict, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    dtype = torch.promote_types(keys.dtype, torch.float32)
    return -torch.linalg.vector_norm(keys, dim=-1, dtype=dtype)


def streaming_llm_scores(
    cfg: dict, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    k_len, n_sink = keys.shape[1], cfg.get("n_sink", 4)
    n_pruned = k_len - int(k_len * (1 - cfg["compression_ratio"]))
    scores = torch.ones_like(keys[..., 0])
    scores[:, n_sink : n_sink + n_pruned] = 0
    return scores


def keydiff_scores(cfg: dict, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    anchor = F.normalize(keys, p=2, dim=-1).mean(dim=1, keepdim=True)
    return -F.cosine_similarity(keys, anchor, dim=-1)


def _lag_scores(states: torch.Tensor) -> torch.Tensor:
    """states: (num_kv_heads, num_partitions, lag_size, head_size)."""
    ref, v = states[:, 1:], states[:, :-1]
    min_r = ref.min(dim=-2, keepdim=True).values
    max_r = ref.max(dim=-2, keepdim=True).values
    return ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)


def lagkv_scores(cfg: dict, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    num_heads, q_len, head_size = keys.shape
    n_sink, lag = cfg.get("n_sink", 4), cfg.get("lag_size", 128)
    if q_len < n_sink + 2 * lag:
        scores = torch.ones(num_heads, q_len, dtype=keys.dtype, device=keys.device)
        if q_len > n_sink:
            ramp = torch.arange(q_len - n_sink, device=keys.device) / (q_len - n_sink)
            scores[:, n_sink:] = ramp.to(keys.dtype)
        return scores
    end = n_sink + ((q_len - n_sink) // lag) * lag
    key_score = _lag_scores(keys[:, n_sink:end].view(num_heads, -1, lag, head_size))
    value_score = _lag_scores(values[:, n_sink:end].view(num_heads, -1, lag, head_size))
    scores = (key_score + value_score) / 2
    if not cfg.get("cross_scoring", False):
        scores = (scores.argsort(dim=-1).argsort(dim=-1) / lag).to(keys.dtype)
    scores = scores.reshape(num_heads, -1)
    return F.pad(F.pad(scores, (n_sink, 0), value=1), (0, lag + q_len - end), value=1)


def _local_leverage(squared_norms: torch.Tensor, window: int) -> torch.Tensor:
    num_heads, n = squared_norms.shape
    x = F.pad(squared_norms, (0, (window - n % window) % window))
    x = x.reshape(num_heads, -1, window)
    return (x / x.sum(dim=-1, keepdim=True)).reshape(num_heads, -1)[:, :n]


def cur_scores(cfg: dict, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    window = cfg.get("local_window_size", 16)
    k2 = _local_leverage((keys**2).sum(dim=-1), window)
    v2 = _local_leverage((values**2).sum(dim=-1), window)
    scores = k2 * v2
    scores /= scores.sum(dim=-1, keepdim=True)
    scores[:, : cfg.get("num_sinks", 4)] = 1.0
    return scores


def dropkv_scores(
    cfg: dict, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    window, num_heads, head_dim = queries.shape
    num_kv_heads, seq_len = keys.shape[:2]
    groups = num_heads // num_kv_heads
    kernel_size, epsilon = cfg.get("kernel_size", 7), cfg.get("epsilon", 1e-6)
    q = queries.permute(1, 0, 2)
    k = keys.repeat_interleave(groups, dim=0)
    v = values.repeat_interleave(groups, dim=0)
    attn = q @ k.transpose(1, 2) / math.sqrt(head_dim)
    attn = attn + torch.full(
        (window, seq_len), float("-inf"), device=attn.device
    ).triu(diagonal=seq_len - window + 1)
    probs = F.softmax(attn, dim=-1, dtype=torch.float32).to(queries.dtype)
    output = (probs @ v).float()
    weights = (probs.float() / (1.0 - probs.float() + epsilon)).square()
    v = v.float()
    term_a = weights.sum(dim=-2) * v.square().sum(dim=-1)
    term_b = 2.0 * (v * (weights.transpose(-1, -2) @ output)).sum(dim=-1)
    term_c = (weights.transpose(-1, -2) @ output.square().sum(-1, keepdim=True))[..., 0]
    scores = (term_a - term_b + term_c).view(num_kv_heads, groups, seq_len).mean(1)
    scores = F.avg_pool1d(
        scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    )
    scores[:, -window:] = scores.amax(dim=-1, keepdim=True) + 1.0
    return scores


KV_SCORERS = {
    "random": random_scores,
    "knorm": knorm_scores,
    "streaming_llm": streaming_llm_scores,
    "keydiff": keydiff_scores,
    "lagkv": lagkv_scores,
    "cur": cur_scores,
}
WRAPPERS = ("adakv", "chunk", "block", "chunkkv", "entropy_gated_chunkkv", "per_layer")
SCORERS = (*KV_SCORERS, "snapkv", "tova", "dropkv")


def scorer_name(cfg: dict) -> str:
    press = cfg.get("press", "snapkv")
    return cfg.get("scorer", "snapkv") if press in WRAPPERS else press


def query_window(cfg: dict) -> int:
    """Last queries the press reads; the final prefill chunk must cover them."""
    name = scorer_name(cfg)
    if name in KV_SCORERS:
        return 0
    if name == "simlayerkv":
        return cfg.get("n_last", 1)
    return 1 if name == "tova" else cfg.get("window_size", 32 if name == "dropkv" else 64)


def scores(
    cfg: dict,
    queries: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """queries: the press's window, (query_window, num_heads, head_size), post-RoPE."""
    name = scorer_name(cfg)
    if name in ("snapkv", "pyramidkv"):
        return triton_window_attention.window_attention_scores(
            queries, kv_cache, block_ids, seq_len, cfg.get("kernel_size", 5)
        )
    if name == "tova":
        s = triton_window_attention.window_attention_scores(queries, kv_cache, block_ids, seq_len, 1)
        s = s[:, :-1].mean(0).expand(kv_cache.shape[1], -1)
        return F.pad(s, (0, 1), value=s.max().item() + 1)
    keys, values = read_kv(kv_cache, block_ids, seq_len)
    if name == "dropkv":
        return dropkv_scores(cfg, queries, keys, values)
    return KV_SCORERS[name](cfg, keys, values)


def chunk_keep(cfg: dict, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """kvpress ChunkPress: the same ratio inside every chunk."""
    score, length = KV_SCORERS[scorer_name(cfg)], cfg.get("chunk_length", 1024)
    keep = []
    for start in range(0, keys.shape[1], length):
        k, v = keys[:, start : start + length], values[:, start : start + length]
        n_kept = max(1, int(k.shape[1] * (1 - cfg["compression_ratio"])))
        keep.append(start + score(cfg, k, v).topk(n_kept, dim=-1).indices)
    return torch.cat(keep, dim=-1).sort(dim=-1).values


def block_keep(
    cfg: dict, keys: torch.Tensor, values: torch.Tensor, n_kept: int
) -> torch.Tensor:
    """kvpress BlockPress: rescore the kept set together with the next block."""
    score = KV_SCORERS[scorer_name(cfg)]
    num_heads, k_len, head_dim = keys.shape
    block_size = min(cfg.get("block_size", 128), k_len)
    kept = torch.arange(n_kept, device=keys.device).expand(num_heads, -1)
    for i in range(n_kept, k_len, block_size):
        block = torch.arange(i, min(i + block_size, k_len), device=keys.device)
        current = torch.cat([kept, block.expand(num_heads, -1)], dim=-1)
        index = current[..., None].expand(-1, -1, head_dim)
        s = score(cfg, keys.gather(1, index), values.gather(1, index))
        kept = current.gather(-1, s.topk(n_kept, dim=-1).indices)
    return kept.sort(dim=-1).values


def chunkkv_mask(cfg: dict, s: torch.Tensor) -> torch.Tensor:
    """kvpress ChunkKVPress: whole chunks, the same ones for every head. The kept
    count depends on whether the shorter last chunk is selected."""
    length = cfg.get("chunk_length", 20)
    chunk_scores = torch.stack([c.mean() for c in s.sum(dim=0).split(length)])
    n_chunks = max(1, int(len(chunk_scores) * (1 - cfg["compression_ratio"])))
    mask = torch.zeros_like(s, dtype=torch.bool)
    for c in chunk_scores.topk(n_chunks).indices.tolist():
        mask[:, c * length : (c + 1) * length] = True
    return mask


def _entropy(token_scores: torch.Tensor, n: int) -> torch.Tensor:
    p = token_scores / (token_scores.sum(dim=-1, keepdim=True) + EPSILON)
    h = -(p * (p + EPSILON).log()).sum(dim=-1)
    return (h / math.log(n)).clamp(0.0, 1.0)


def entropy_gated_chunkkv_mask(cfg: dict, s: torch.Tensor) -> torch.Tensor:
    """kvpress EntropyGatedChunkKVPress: spiky important chunks keep only their top
    tokens and the freed budget goes to further chunks."""
    length = cfg.get("chunk_length", 10)
    low_entropy_length = cfg.get("low_entropy_chunk_length", 4)
    token_scores = s.sum(dim=0).float()
    kv_len = token_scores.numel()
    budget = max(1, int(kv_len * (1 - cfg["compression_ratio"])))
    n_complete = kv_len // length
    full = token_scores[: n_complete * length].view(n_complete, length)
    chunk_scores = full.mean(dim=1)
    chunk_entropy = _entropy(full - full.amin(dim=1, keepdim=True).clamp(max=0), length)
    if (tail_len := kv_len % length) > 0:
        tail = token_scores[n_complete * length :]
        tail_entropy = torch.zeros(1, device=s.device)
        if tail_len > 1:
            rebased = tail - tail.min().clamp(max=0.0) if (tail < 0).any() else tail
            tail_entropy = _entropy(rebased, tail_len)[None]
        chunk_scores = torch.cat([chunk_scores, tail.mean()[None]])
        chunk_entropy = torch.cat([chunk_entropy, tail_entropy])
    high_score = (chunk_scores >= chunk_scores.median()).tolist()
    low_entropy = (chunk_entropy < chunk_entropy.median()).tolist()
    keep = torch.zeros(kv_len, dtype=torch.bool, device=s.device)
    for chunk in torch.argsort(chunk_scores, descending=True).tolist():
        if budget <= 0:
            break
        start, end = chunk * length, min((chunk + 1) * length, kv_len)
        n_kept = min(end - start, budget)
        if high_score[chunk] and low_entropy[chunk]:
            n_kept = min(n_kept, low_entropy_length)
        if n_kept == end - start:
            keep[start:end] = True
        else:
            keep[torch.topk(token_scores[start:end], n_kept).indices + start] = True
        budget -= n_kept
    if budget > 0:
        leftover = (~keep).nonzero()[:, 0]
        keep[leftover[token_scores[leftover].topk(budget).indices]] = True
    return keep.expand(s.shape[0], -1)


def simlayerkv_mask(
    cfg: dict,
    queries: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """kvpress SimLayerKVPress: a lazy layer keeps its first and last tokens only."""
    threshold = cfg.get("lazy_threshold", 1.0)
    n_last, n_recent = cfg.get("n_last", 1), cfg.get("n_recent", 1024)
    n_initial = cfg.get("n_initial", 4)
    mask = torch.ones(
        kv_cache.shape[1], seq_len, dtype=torch.bool, device=kv_cache.device
    )
    if threshold == 1.0 or seq_len <= n_initial + n_recent + n_last:
        return mask
    s = triton_window_attention.window_attention_scores(queries, kv_cache, block_ids, seq_len, 1)
    attn = s[:, :-n_last].mean(0)
    if (attn[:n_initial].sum() + attn[-n_recent:].sum()).item() > threshold:
        mask[:, n_initial : seq_len - n_recent + n_last] = False
    return mask
