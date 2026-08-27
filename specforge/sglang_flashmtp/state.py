from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class FlashMTPDraftInput(SpecInput):
    verified_id: torch.Tensor
    pivot_hidden: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> tuple[int, int]:
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        del has_been_filtered
        self.verified_id = self.verified_id[new_indices]
        self.pivot_hidden = self.pivot_hidden[new_indices]

    def merge_batch(self, spec_info: "FlashMTPDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.pivot_hidden = torch.cat(
            [self.pivot_hidden, spec_info.pivot_hidden], dim=0
        )


@dataclass
class FlashMTPDraftInputV2(DFlashDraftInputV2):
    """Overlap state; ``hidden_states`` is the per-request FlashMTP pivot."""

    @property
    def pivot_hidden(self) -> torch.Tensor:
        return self.hidden_states

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "FlashMTPDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 1), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 1), device=device, dtype=torch.int64),
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, 1, 1), device=device, dtype=torch.float16),
            verify_done=None,
        )


@dataclass
class FlashMTPVerifyInput(DFlashVerifyInput):
    """DFlash linear verification with FlashMTP pivot extraction."""

    def verify(self, *, batch, logits_output, page_size: int):
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("FlashMTP target verification returned no hidden states.")
        batch_size = batch.batch_size()
        hidden = hidden.view(batch_size, self.draft_token_num, -1)
        # DFlash's generic v1 verifier materializes every committed hidden row for
        # a cache-based draft. FlashMTP only needs one pivot row, so feed it a
        # one-column view and gather the real pivot entirely on GPU below.
        logits_output.hidden_states = hidden[..., :1].reshape(
            batch_size * self.draft_token_num, 1
        )
        result = super().verify(
            batch=batch, logits_output=logits_output, page_size=page_size
        )
        new_verified_id, commit_lens, _, accept_lens_cpu = result
        row = torch.arange(batch_size, device=hidden.device)
        pivot = hidden[row, commit_lens.to(torch.int64) - 1]
        return new_verified_id, commit_lens, pivot, accept_lens_cpu
