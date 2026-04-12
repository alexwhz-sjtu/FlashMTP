"""
基于词表 embedding 的文本语义切分。

默认在「缝隙」两侧取固定长度 token 块，对块内 embedding 做均值池化后再算余弦不相似度
（TextTiling 思路：比较语义块与块之间的关系，而非只看相邻两个 token）。
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def consecutive_cosine_dissimilarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    embeddings: (seq_len, dim) 或 (1, seq_len, dim)
    返回 (seq_len - 1,)：第 i 项为 token i 与 token i+1 的余弦不相似度 1 - cos_sim。
    """
    if embeddings.dim() == 3:
        if embeddings.shape[0] != 1:
            raise ValueError("batch 维度仅支持 1")
        embeddings = embeddings[0]
    if embeddings.dim() != 2:
        raise ValueError("embeddings 应为 (seq, dim)")

    z = l2_normalize(embeddings.float(), dim=-1)
    sim = (z[:-1] * z[1:]).sum(dim=-1)
    return 1.0 - sim


def _gap_block_cosine_dissimilarity_single(
    emb: torch.Tensor,
    left_block_tokens: int,
    right_block_tokens: int,
) -> torch.Tensor:
    """
    emb: (n, d)。对每个缝隙 g（介于 token g 与 g+1 之间）：
    左块 = emb[max(0,g-L+1):g+1] 的均值，右块 = emb[g+1:min(n,g+1+R)] 的均值；
    两块向量 L2 归一化后算 1 - cos_sim。返回长度 n-1。
    """
    n, d = emb.shape
    if n < 2:
        return emb.new_zeros(0)

    L = max(1, int(left_block_tokens))
    R = max(1, int(right_block_tokens))
    device = emb.device
    g = torch.arange(n - 1, device=device, dtype=torch.long)
    sl = (g - L + 1).clamp_min(0)
    el = g + 1
    sr = g + 1
    er = torch.minimum(g + 1 + R, torch.full((n - 1,), n, device=device, dtype=torch.long))

    emb_f = emb.float()
    pad = torch.zeros(1, d, device=device, dtype=torch.float32)
    cum = torch.cat([pad, emb_f.cumsum(dim=0)], dim=0)
    sum_l = cum[el] - cum[sl]
    cnt_l = (el - sl).clamp_min(1).float().unsqueeze(-1)
    sum_r = cum[er] - cum[sr]
    cnt_r = (er - sr).clamp_min(1).float().unsqueeze(-1)

    mean_l = sum_l / cnt_l
    mean_r = sum_r / cnt_r
    zl = l2_normalize(mean_l, dim=-1)
    zr = l2_normalize(mean_r, dim=-1)
    sim = (zl * zr).sum(dim=-1)
    return 1.0 - sim.to(dtype=emb.dtype)


def gap_block_cosine_dissimilarity(
    embeddings: torch.Tensor,
    left_block_tokens: int = 32,
    right_block_tokens: int = 32,
    extra_scales: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    缝隙两侧「语义块」之间的余弦不相似度；可选 extra_scales 将多个 (L,R) 的结果取平均，多尺度更稳。
    embeddings: (seq, dim) 或 (1, seq, dim)
    """
    if embeddings.dim() == 3:
        if embeddings.shape[0] != 1:
            raise ValueError("batch 维度仅支持 1")
        embeddings = embeddings[0]
    if embeddings.dim() != 2:
        raise ValueError("embeddings 应为 (seq, dim)")

    parts = [
        _gap_block_cosine_dissimilarity_single(
            embeddings, left_block_tokens, right_block_tokens
        )
    ]
    for lr in extra_scales or ():
        parts.append(
            _gap_block_cosine_dissimilarity_single(embeddings, int(lr[0]), int(lr[1]))
        )
    return torch.stack(parts, dim=0).mean(dim=0)


def smooth_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    """奇数窗口移动平均；window<=1 则原样返回。"""
    if window <= 1 or x.numel() == 0:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    p = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad), mode="replicate")
    k = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    return torch.nn.functional.conv1d(p, k).squeeze()


def _local_maxima_indices(scores: torch.Tensor) -> List[int]:
    """一维向量上的严格局部极大值下标（不含平坦高原上的所有点，仅峰值邻域）。"""
    n = scores.numel()
    if n == 0:
        return []
    if n == 1:
        return [0]
    s = scores.cpu()
    out: List[int] = []
    for i in range(n):
        left = float(s[i - 1]) if i > 0 else float("-inf")
        right = float(s[i + 1]) if i < n - 1 else float("-inf")
        v = float(s[i])
        if v > left and v > right:
            out.append(i)
    return out


def boundaries_from_dissimilarity(
    dissimilarity: torch.Tensor,
    *,
    percentile: float = 85.0,
    min_segment_tokens: int = 2,
    smooth_window: int = 3,
    baseline_subtract_window: int = 0,
) -> List[int]:
    """
    在不相似度序列上选切分「缝隙」下标：返回 gap 索引 g，表示在 token g 与 token g+1 之间切一刀。

    - 先对 dissimilarity 做可选平滑，再取局部极大值；
    - 仅保留 >= 给定分位阈值的峰；
    - 任意两个断点 gap 下标之差的绝对值至少为 min_segment_tokens（保证中间段约有这么多 token）。
    """
    if dissimilarity.numel() == 0:
        return []
    base = dissimilarity.float().flatten()
    if baseline_subtract_window > 1:
        base = base - smooth_1d(base, baseline_subtract_window)
    x = smooth_1d(base, smooth_window)
    thr = torch.quantile(x, percentile / 100.0)
    candidates = [i for i in _local_maxima_indices(x) if float(x[i]) >= float(thr)]
    if not candidates:
        return []

    scores = [(float(x[i]), i) for i in candidates]
    scores.sort(key=lambda t: t[0], reverse=True)
    chosen: List[int] = []
    for _, idx in scores:
        if all(abs(idx - j) >= min_segment_tokens for j in chosen):
            chosen.append(idx)
    chosen.sort()
    return chosen


def split_input_ids_at_gaps(
    input_ids: Sequence[int], gap_indices: Iterable[int]
) -> List[List[int]]:
    """gap_indices 为缝隙位置 g（在 token[g] 与 token[g+1] 之间）。"""
    ids = list(input_ids)
    if not ids:
        return []
    gaps = sorted(set(g for g in gap_indices if 0 <= g < len(ids) - 1))
    if not gaps:
        return [ids]
    parts: List[List[int]] = []
    start = 0
    for g in gaps:
        parts.append(ids[start : g + 1])
        start = g + 1
    parts.append(ids[start:])
    return [p for p in parts if p]


def semantic_segment_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: nn.Module,
    *,
    device: Union[str, torch.device] = "cpu",
    add_special_tokens: bool = False,
    percentile: float = 85.0,
    min_segment_tokens: int = 2,
    smooth_window: int = 3,
    left_block_tokens: int = 32,
    right_block_tokens: int = 32,
    extra_block_scales: Optional[Sequence[Tuple[int, int]]] = None,
    adjacent_only: bool = False,
    baseline_subtract_window: int = 0,
) -> Tuple[List[str], torch.Tensor, List[int]]:
    """
    返回 (片段字符串列表, 每个缝隙的不相似度向量长度 n-1, 选中的 gap 下标列表)。

    默认在缝隙两侧各看 ``left_block_tokens`` / ``right_block_tokens`` 个 token 的**均值嵌入**，
    再算两块之间的余弦不相似度；``extra_block_scales`` 可传入更多 (L,R) 与主尺度取平均。
    ``adjacent_only=True`` 时退化为仅相邻两 token（与旧行为一致）。
    ``baseline_subtract_window``>1 时先从曲线中减去长窗移动平均，突出相对峰值（利于长文）。
    """
    enc = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    with torch.inference_mode():
        emb = embed_tokens(input_ids)
        if emb.dim() == 3:
            emb = emb[0]
        if adjacent_only:
            dissim = consecutive_cosine_dissimilarity(emb)
        else:
            dissim = gap_block_cosine_dissimilarity(
                emb,
                left_block_tokens=left_block_tokens,
                right_block_tokens=right_block_tokens,
                extra_scales=extra_block_scales,
            )
        gaps = boundaries_from_dissimilarity(
            dissim,
            percentile=percentile,
            min_segment_tokens=min_segment_tokens,
            smooth_window=smooth_window,
            baseline_subtract_window=baseline_subtract_window,
        )
    id_chunks = split_input_ids_at_gaps(input_ids[0].tolist(), gaps)
    strings = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in id_chunks]
    return strings, dissim.detach().cpu(), gaps
