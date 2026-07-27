"""Low-rank serial heads for FlashMTP block prediction.

The expensive FlashMTP backbone produces all block-position hidden states in
parallel.  These heads reintroduce a light autoregressive dependency through
the previously generated token.  Training uses teacher-forced previous tokens;
inference samples the block from left to right.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")
MARKOV_OUTPUT_MODES = ("additive", "direct")


def _sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class FlashMTPMarkovHead(nn.Module):
    """Vanilla, gated, or recurrent low-rank vocabulary head."""

    def __init__(
        self,
        *,
        head_type: str,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.head_type = str(head_type).lower()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.hidden_size = int(hidden_size)

        if self.head_type not in MARKOV_HEAD_TYPES:
            raise ValueError(
                f"Unknown markov head type {self.head_type!r}; "
                f"expected one of {MARKOV_HEAD_TYPES}."
            )
        if self.markov_rank <= 0:
            raise ValueError(f"markov_rank must be positive, got {self.markov_rank}.")

        self.prev_token_embedding = nn.Embedding(self.vocab_size, self.markov_rank)
        self.output_proj = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

        self.gate_proj: Optional[nn.Linear] = None
        self.state_proj: Optional[nn.Linear] = None
        self.state_out_proj: Optional[nn.Linear] = None
        self.hidden_proj: Optional[nn.Linear] = None
        self.hidden_fuse_gate_proj: Optional[nn.Linear] = None
        if self.head_type == "gated":
            self.gate_proj = nn.Linear(
                self.hidden_size + self.markov_rank,
                self.markov_rank,
            )
            self.hidden_proj = nn.Linear(self.hidden_size, self.markov_rank, bias=False)
            self.hidden_fuse_gate_proj = nn.Linear(
                2 * self.markov_rank,
                self.markov_rank,
            )
        elif self.head_type == "rnn":
            self.state_proj = nn.Linear(2 * self.markov_rank, 2 * self.markov_rank)
            self.state_out_proj = nn.Linear(
                self.markov_rank,
                self.markov_rank,
                bias=False,
            )
            self.hidden_proj = nn.Linear(self.hidden_size, self.markov_rank, bias=False)
            self.hidden_fuse_gate_proj = nn.Linear(
                2 * self.markov_rank,
                self.markov_rank,
            )

    def project_logits(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project low-rank head states to full-vocabulary logits."""
        return self.output_proj(latent_states)

    def _hidden_latent_contribution(
        self,
        hidden_states: torch.Tensor,
        *,
        output_mode: str,
    ) -> Optional[torch.Tensor]:
        if output_mode != "direct" or self.hidden_proj is None:
            return None
        return self.hidden_proj(hidden_states)

    def _fuse_serial_and_hidden(
        self,
        serial_latent: torch.Tensor,
        hidden_latent: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_latent is None:
            return serial_latent
        assert self.hidden_fuse_gate_proj is not None
        fuse_inputs = torch.cat([serial_latent, hidden_latent], dim=-1)
        fuse_gate = torch.sigmoid(self.hidden_fuse_gate_proj(fuse_inputs))
        return fuse_gate * serial_latent + (1.0 - fuse_gate) * hidden_latent

    def _compute_step_latent(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor],
        output_mode: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prev_embeddings = self.prev_token_embedding(prev_token_ids.long())
        hidden_latent = self._hidden_latent_contribution(
            hidden_states,
            output_mode=output_mode,
        )

        if self.head_type == "vanilla":
            return prev_embeddings, None

        if self.head_type == "gated":
            assert self.gate_proj is not None
            gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_inputs))
            serial_latent = gate.to(prev_embeddings.dtype) * prev_embeddings
            return self._fuse_serial_and_hidden(serial_latent, hidden_latent), None

        assert self.state_proj is not None
        assert self.state_out_proj is not None
        if state is None:
            state = torch.zeros_like(prev_embeddings)
        mem_inputs = torch.cat([state, prev_embeddings], dim=-1)
        gate_raw, candidate_raw = self.state_proj(mem_inputs).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        serial_latent = torch.tanh(self.state_out_proj(new_state))
        return self._fuse_serial_and_hidden(serial_latent, hidden_latent), new_state

    def forward_teacher_forcing(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        output_mode: str = "additive",
    ) -> torch.Tensor:
        """Return low-rank states for teacher-forced block predictions.

        Args:
            hidden_states: ``[..., prediction_length, hidden_size]``.
            prev_token_ids: ``[..., prediction_length]``; entry ``k`` is the
                ground-truth token immediately preceding prediction ``k``.
        """
        if hidden_states.shape[:-1] != prev_token_ids.shape:
            raise ValueError(
                "hidden_states and prev_token_ids leading shapes must match, "
                f"got {tuple(hidden_states.shape)} and "
                f"{tuple(prev_token_ids.shape)}."
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.size(-1)}."
            )

        output_mode = str(output_mode).lower()
        if output_mode not in MARKOV_OUTPUT_MODES:
            raise ValueError(
                f"Unknown markov output mode {output_mode!r}; "
                f"expected one of {MARKOV_OUTPUT_MODES}."
            )

        prediction_length = hidden_states.size(-2)
        if self.head_type != "rnn":
            latent, _ = self._compute_step_latent(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_states,
                state=None,
                output_mode=output_mode,
            )
            return latent

        state = torch.zeros(
            *hidden_states.shape[:-2],
            self.markov_rank,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        outputs: list[torch.Tensor] = []
        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids[..., position],
                hidden_states=hidden_states[..., position, :],
                state=state,
                output_mode=output_mode,
            )
            outputs.append(latent.unsqueeze(-2))
        if not outputs:
            return hidden_states.new_empty(
                *hidden_states.shape[:-2], 0, self.markov_rank
            )
        return torch.cat(outputs, dim=-2)

    def sample_block_tokens(
        self,
        *,
        hidden_states: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        output_mode: str,
        base_logits: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Serially sample one FlashMTP prediction block.

        Returns sampled token IDs and the final logits actually used to sample
        them.  ``base_logits`` is required only in additive mode.
        """
        output_mode = str(output_mode).lower()
        if output_mode not in MARKOV_OUTPUT_MODES:
            raise ValueError(
                f"Unknown markov output mode {output_mode!r}; "
                f"expected one of {MARKOV_OUTPUT_MODES}."
            )
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, prediction_length, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if output_mode == "additive":
            if base_logits is None:
                raise ValueError("base_logits is required in additive mode.")
            if base_logits.shape[:2] != hidden_states.shape[:2]:
                raise ValueError(
                    "base_logits and hidden_states batch/position shapes must match."
                )

        batch_size, prediction_length = hidden_states.shape[:2]
        state = (
            hidden_states.new_zeros(batch_size, self.markov_rank)
            if self.head_type == "rnn"
            else None
        )
        prev_token_ids = first_prev_token_ids.long()
        sampled_tokens: list[torch.Tensor] = []
        final_logits: list[torch.Tensor] = []

        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_states[:, position, :],
                state=state,
                output_mode=output_mode,
            )
            step_logits = self.project_logits(latent)
            if output_mode == "additive":
                assert base_logits is not None
                step_logits = base_logits[:, position, :] + step_logits
            next_token_ids = _sample_tokens(step_logits, float(temperature))
            sampled_tokens.append(next_token_ids.unsqueeze(1))
            final_logits.append(step_logits.unsqueeze(1))
            prev_token_ids = next_token_ids

        if not sampled_tokens:
            return (
                torch.empty(
                    batch_size,
                    0,
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
                hidden_states.new_empty(batch_size, 0, self.vocab_size),
            )
        return torch.cat(sampled_tokens, dim=1), torch.cat(final_logits, dim=1)


__all__ = [
    "FlashMTPMarkovHead",
    "MARKOV_HEAD_TYPES",
    "MARKOV_OUTPUT_MODES",
]
