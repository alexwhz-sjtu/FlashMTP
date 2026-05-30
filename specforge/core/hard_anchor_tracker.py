# coding=utf-8
"""Track low prefix-acceptance anchors and bias future sampling toward them."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch


class HardAnchorTracker:
    """EMA tracker of per-(sample, anchor_pos) prefix acceptance length.

    When the same training sample reappears, positions with consistently short
    acceptance get higher sampling weight (or reserved slots in mixture mode).
    """

    def __init__(
        self,
        ema_alpha: float = 0.2,
        threshold: float = 2.5,
        min_visits: int = 2,
        boost: float = 8.0,
        max_samples: int = 10000,
        mode: str = "weighted",
        hard_ratio: float = 0.3,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.threshold = threshold
        self.min_visits = min_visits
        self.boost = boost
        self.max_samples = max_samples
        self.mode = mode
        self.hard_ratio = hard_ratio
        # sample_key -> {anchor_pos: (ema_prefix_len, visit_count)}
        self._stats: OrderedDict[str, dict[int, Tuple[float, int]]] = OrderedDict()

    @staticmethod
    def sample_key(input_ids: torch.Tensor) -> str:
        """Stable fingerprint for one sequence (batch dim 0)."""
        row = input_ids[0].detach().cpu().tolist()
        digest = hashlib.md5(repr(row).encode()).hexdigest()
        return digest[:16]

    def _touch(self, sample_key: str) -> None:
        if sample_key in self._stats:
            self._stats.move_to_end(sample_key)
            return
        self._stats[sample_key] = {}
        if len(self._stats) > self.max_samples:
            self._stats.popitem(last=False)

    def update(
        self,
        sample_key: str,
        anchor_positions: torch.Tensor,
        prefix_lengths: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> None:
        """Record prefix lengths observed for anchors in this forward pass."""
        self._touch(sample_key)
        per_pos = self._stats[sample_key]
        pos_list = anchor_positions.tolist()
        plen_list = prefix_lengths.tolist()
        keep_list = block_keep_mask.tolist()
        for pos, plen, keep in zip(pos_list, plen_list, keep_list):
            if not keep:
                continue
            pos = int(pos)
            plen = float(plen)
            if pos not in per_pos:
                per_pos[pos] = (plen, 1)
                continue
            ema, cnt = per_pos[pos]
            ema = self.ema_alpha * plen + (1.0 - self.ema_alpha) * ema
            per_pos[pos] = (ema, cnt + 1)

    def _hard_positions(
        self, sample_key: str, valid: torch.Tensor
    ) -> List[Tuple[float, int]]:
        per_pos = self._stats.get(sample_key)
        if not per_pos:
            return []
        hard: List[Tuple[float, int]] = []
        valid_len = int(valid.numel())
        for pos, (ema, cnt) in per_pos.items():
            if pos >= valid_len or not bool(valid[pos].item()):
                continue
            if cnt >= self.min_visits and ema <= self.threshold:
                hard.append((ema, pos))
        hard.sort(key=lambda x: x[0])
        return hard

    def get_sampling_weights(
        self,
        sample_key: str,
        num_positions: int,
        valid: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        weights = torch.ones(num_positions, device=device)
        for _, pos in self._hard_positions(sample_key, valid):
            weights[pos] = self.boost
        return weights

    def select_hard_anchors(
        self,
        sample_key: str,
        valid: torch.Tensor,
        max_n: int,
    ) -> List[int]:
        hard = self._hard_positions(sample_key, valid)
        n_hard = min(len(hard), max(0, int(max_n * self.hard_ratio)))
        return [pos for _, pos in hard[:n_hard]]

    def count_tracked_hard(self, sample_key: str, valid: torch.Tensor) -> int:
        return len(self._hard_positions(sample_key, valid))
