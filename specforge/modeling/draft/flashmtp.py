import time
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from ...utils import print_on_rank0
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def _cuda_sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


STOCHASTIC_VERIFICATION_MODES = ("match", "rejection")


def _validate_stochastic_verification_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in STOCHASTIC_VERIFICATION_MODES:
        raise ValueError(
            f"Unknown stochastic_verification_mode={mode!r}; expected one of "
            f"{STOCHASTIC_VERIFICATION_MODES}."
        )
    return mode


def _logits_to_probs(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1),
            num_classes=logits.shape[-1],
        ).float()
    return torch.softmax(logits.float() / float(temperature), dim=-1)


def _sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    return torch.multinomial(
        probs.reshape(-1, vocab_size), num_samples=1
    ).reshape(*probs.shape[:-1])


def _sample_residual(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(
        residual_mass > 1e-8,
        residual / residual_mass.clamp_min(1e-8),
        target_probs,
    )
    return _sample_from_probs(residual)


def rejection_sample_verify(
    *,
    proposed_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    """Verify one stochastic draft block using speculative rejection sampling.

    Returns the number of accepted draft tokens and the correction/bonus token.
    This first implementation intentionally supports batch size one, matching
    the sequential acceptance semantics used by DSpARK.
    """
    if temperature < 1e-5:
        raise ValueError("rejection sampling requires temperature > 0.")
    if proposed_tokens.ndim != 2 or proposed_tokens.shape[0] != 1:
        raise ValueError(
            "rejection sampling currently requires proposed_tokens shape [1, K]."
        )
    proposal_count = proposed_tokens.shape[1]
    if draft_logits.shape[:2] != (1, proposal_count):
        raise ValueError(
            "draft_logits must have shape [1, K, vocab] matching proposed_tokens."
        )
    if target_logits.shape[:2] != (1, proposal_count + 1):
        raise ValueError(
            "target_logits must have shape [1, K + 1, vocab]."
        )
    if draft_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError("draft and target logits must use the same vocabulary.")

    target_probs = _logits_to_probs(target_logits, temperature)
    if proposal_count == 0:
        return 0, _sample_from_probs(target_probs[:, 0, :])

    draft_probs = _logits_to_probs(draft_logits, temperature)
    token_index = proposed_tokens.unsqueeze(-1)
    selected_target = target_probs[:, :proposal_count, :].gather(
        dim=-1, index=token_index
    ).squeeze(-1)
    selected_draft = draft_probs.gather(
        dim=-1, index=token_index
    ).squeeze(-1)
    accept_probs = torch.minimum(
        torch.ones_like(selected_target),
        selected_target / selected_draft.clamp_min(1e-20),
    )
    accepted_prefix = (
        (torch.rand_like(accept_probs) < accept_probs)
        .to(torch.int64)
        .cumprod(dim=1)
    )
    accepted_count = int(accepted_prefix.sum().item())

    if accepted_count < proposal_count:
        next_token = _sample_residual(
            target_probs[:, accepted_count, :],
            draft_probs[:, accepted_count, :],
        )
    else:
        next_token = _sample_from_probs(
            target_probs[:, proposal_count, :]
        )
    return accepted_count, next_token


from .flashmtp_markov_head import (
    FlashMTPMarkovHead,
    MARKOV_HEAD_TYPES,
    MARKOV_OUTPUT_MODES,
    markov_output_uses_base_lm_head,
)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    k_len = k.size(-2)
    q_cos = cos[..., -q_len:, :]
    q_sin = sin[..., -q_len:, :]
    k_cos = cos[..., -k_len:, :]
    k_sin = sin[..., -k_len:, :]
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed


FLASHMTP_ARCHITECTURE_VERSION = "sliding_chs_first_token_window_v5"
FLASHMTP_ARCHITECTURE_VERSION_V4 = "sliding_chs_context_embedding_v4"
FLASHMTP_ARCHITECTURE_VERSION_V3 = "sliding_chs_query_embedding_v3"
SUPPORTED_FLASHMTP_ARCHITECTURE_VERSIONS = (
    FLASHMTP_ARCHITECTURE_VERSION,
    FLASHMTP_ARCHITECTURE_VERSION_V4,
    FLASHMTP_ARCHITECTURE_VERSION_V3,
)
def _infer_hs_embedding_offset(
    hidden_states: tuple | list, num_transformer_layers: int
) -> int:
    lt = len(hidden_states)
    if lt == num_transformer_layers:
        return 0
    if lt == num_transformer_layers + 1:
        return 1
    return 1 if lt > num_transformer_layers else 0


def build_target_layer_ids(
    num_transformer_layers: int, chs_num_layers: int
) -> list[int]:
    """Select exactly ``chs_num_layers`` layers, including the first and last."""
    L = num_transformer_layers
    S = int(chs_num_layers)
    if L < 2:
        raise ValueError(f"FlashMTP requires at least 2 target layers, got {L}.")
    if not 2 <= S <= L:
        raise ValueError(
            f"chs_num_layers must be in [2, {L}], got {chs_num_layers}."
        )
    picked: set[int] = {0, L - 1}
    n_middle = S - 2
    lo, hi = 1, L - 2
    if n_middle > 0 and lo <= hi:
        interior_span = hi - lo
        cap = hi - lo + 1
        n_take = min(n_middle, cap)
        if n_take == 1:
            picked.add((lo + hi) // 2)
        else:
            for i in range(n_take):
                if interior_span == 0:
                    idx = lo
                else:
                    idx = lo + int(round(i * interior_span / (n_take - 1)))
                picked.add(int(max(lo, min(hi, idx))))
    result = sorted(picked)
    if len(result) != S:
        raise RuntimeError(
            f"Failed to select exactly {S} CHS layers from target depth {L}: {result}"
        )
    return result


def gather_pivot_multilayer_inference(
    hidden_states: tuple | list,
    target_layer_ids: list[int],
    token_index: int,
    num_transformer_layers: int,
) -> torch.Tensor:
    """Return (B, 1, S, H) pivot features for inference."""
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    pieces: list[torch.Tensor] = []
    for layer_id in target_layer_ids:
        layer_h = hidden_states[layer_id + off]
        pieces.append(layer_h[:, token_index, :].unsqueeze(1))
    return torch.stack(pieces, dim=2)


def gather_hidden_layers_inference(
    hidden_states: tuple | list,
    layer_ids: list[int],
    num_transformer_layers: int,
) -> torch.Tensor:
    """Return selected target layers as ``(B, T, S, H)``."""
    off = _infer_hs_embedding_offset(hidden_states, num_transformer_layers)
    return torch.stack(
        [hidden_states[layer_id + off] for layer_id in layer_ids], dim=2
    )


class Qwen3FlashMTPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        cos, sin = position_embeddings

        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3FlashMTPDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlashMTPAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class FlashMTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3FlashMTPDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        flashmtp_config = getattr(config, "flashmtp_config", {}) or {}
        architecture_version = flashmtp_config.get("architecture_version")
        if architecture_version not in SUPPORTED_FLASHMTP_ARCHITECTURE_VERSIONS:
            raise ValueError(
                "Incompatible FlashMTP checkpoint/config architecture_version: expected one of "
                f"{SUPPORTED_FLASHMTP_ARCHITECTURE_VERSIONS}, got "
                f"{architecture_version!r}. Legacy linear/attention/prefix and "
                "left-shift checkpoints are not supported by the sliding-CHS model."
            )
        self.architecture_version = str(architecture_version)
        self.sliding_window_size = int(
            flashmtp_config.get("sliding_window_size", 32)
        )
        if self.sliding_window_size < 1:
            raise ValueError(
                f"sliding_window_size must be >= 1, got {self.sliding_window_size}."
            )
        configured_history_mode = str(
            flashmtp_config.pop("history_mode", "pivot_q")
        ).lower()
        flashmtp_config.pop("bwa_stride", None)
        if configured_history_mode in ("pivotq", "pivot-q"):
            configured_history_mode = "pivot_q"
        if configured_history_mode != "pivot_q":
            raise ValueError(
                "This FlashMTP implementation only supports pivot-Q token "
                f"windows; got legacy history_mode={configured_history_mode!r}."
            )
        self.chs_num_layers = int(flashmtp_config.get("chs_num_layers", 7))
        self.local_position = bool(flashmtp_config.get("local_position", False))
        leftover_draft_input_mode = flashmtp_config.pop("draft_input_mode", None)
        if leftover_draft_input_mode is not None and str(
            leftover_draft_input_mode
        ).lower() not in ("legacy",):
            raise ValueError(
                f"draft_input_mode={leftover_draft_input_mode!r} is no longer "
                "supported; the backbone always uses an unsupervised anchor "
                "query followed by MASK queries."
            )
        selected_layer_ids = build_target_layer_ids(
            config.num_target_layers, self.chs_num_layers
        )
        configured_target_ids = flashmtp_config.get("target_layer_ids")
        self.target_layer_ids = (
            list(configured_target_ids)
            if configured_target_ids is not None
            else selected_layer_ids
        )
        if self.target_layer_ids != selected_layer_ids:
            raise ValueError(
                "target_layer_ids must match the fixed first/last plus evenly "
                f"spaced selection {selected_layer_ids}, got {self.target_layer_ids}."
            )
        flashmtp_config.pop("history_layer_ids", None)

        self.chs_first_context = (
            self.architecture_version == FLASHMTP_ARCHITECTURE_VERSION
        )
        if self.architecture_version == FLASHMTP_ARCHITECTURE_VERSION_V3:
            self.pivot_query_embedding = bool(
                flashmtp_config.get("pivot_query_embedding", True)
            )
            self.include_token_embedding_chs = bool(
                flashmtp_config.get(
                    "include_token_embedding_chs",
                    not self.pivot_query_embedding,
                )
            )
        elif self.architecture_version == FLASHMTP_ARCHITECTURE_VERSION_V4:
            self.pivot_query_embedding = False
            self.include_token_embedding_chs = True
        else:
            # V5 supplies the explicit token window after the CHS hidden slots,
            # so the old duplicate pivot-token CHS slot is intentionally absent.
            self.pivot_query_embedding = False
            self.include_token_embedding_chs = False

        flashmtp_config["architecture_version"] = self.architecture_version
        flashmtp_config["sliding_window_size"] = self.sliding_window_size
        flashmtp_config["chs_num_layers"] = self.chs_num_layers
        flashmtp_config["include_token_embedding_chs"] = self.include_token_embedding_chs
        flashmtp_config["pivot_query_embedding"] = self.pivot_query_embedding
        flashmtp_config.pop("add_noise", None)
        flashmtp_config.pop("target_hidden_noise_ratio", None)
        flashmtp_config["local_position"] = self.local_position
        flashmtp_config.pop("draft_input_mode", None)
        flashmtp_config["target_layer_ids"] = self.target_layer_ids
        self.markov_head_type = str(
            flashmtp_config.get("markov_head_type", "none")
        ).lower()
        self.markov_output_mode = str(
            flashmtp_config.get("markov_output_mode", "additive")
        ).lower()
        configured_markov_rank = int(flashmtp_config.get("markov_rank", 0))
        self.markov_rank = (
            configured_markov_rank if self.markov_head_type != "none" else 0
        )
        if self.markov_head_type not in ("none", *MARKOV_HEAD_TYPES):
            raise ValueError(
                f"Unknown markov_head_type={self.markov_head_type!r}; expected "
                f"none or one of {MARKOV_HEAD_TYPES}."
            )
        if self.markov_output_mode not in MARKOV_OUTPUT_MODES:
            raise ValueError(
                f"Unknown markov_output_mode={self.markov_output_mode!r}; "
                f"expected one of {MARKOV_OUTPUT_MODES}."
            )
        if self.markov_head_type != "none" and self.markov_rank <= 0:
            raise ValueError(
                f"markov_rank must be positive when a Markov head is enabled, "
                f"got {self.markov_rank}."
            )
        if (
            self.markov_head_type == "none"
            and self.markov_output_mode == "direct"
        ):
            raise ValueError(
                f"markov_output_mode={self.markov_output_mode!r} requires a Markov head."
            )
        flashmtp_config["markov_head_type"] = self.markov_head_type
        flashmtp_config["markov_output_mode"] = self.markov_output_mode
        flashmtp_config["markov_rank"] = self.markov_rank
        self.markov_head = (
            None
            if self.markov_head_type == "none"
            else FlashMTPMarkovHead(
                head_type=self.markov_head_type,
                vocab_size=config.vocab_size,
                markov_rank=self.markov_rank,
                hidden_size=config.hidden_size,
                max_prediction_length=config.block_size - 1,
                markov_output_mode=self.markov_output_mode,
            )
        )
        self._compiled_serial_sampler_cache: dict[tuple[str, float], Callable] = {}
        config.flashmtp_config = flashmtp_config

        self.layers = nn.ModuleList(
            [
                Qwen3FlashMTPDecoderLayer(
                    config,
                    layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = flashmtp_config.get("mask_token_id", None)
        self._last_decode_stats = {}

        h = config.hidden_size
        self.layer_depth_embedding = nn.Embedding(config.num_target_layers, h)
        self.context_norm = Qwen3RMSNorm(h, eps=config.rms_norm_eps)
        print_on_rank0(
            f"FlashMTP: architecture_version={self.architecture_version}, "
            f"sliding_window_size={self.sliding_window_size}, "
            f"history_slots={self.history_slot_count}, "
            f"chs_num_layers={self.chs_num_layers}, "
            f"current_chs_layers={self.current_chs_slot_count}, "
            f"condition_slots={self.condition_slot_count}, "
            f"pivot_query_embedding={self.pivot_query_embedding}, "
            f"include_token_embedding_chs={self.include_token_embedding_chs}, "
            f"local_position={self.local_position}, "
            f"target_layer_ids={self.target_layer_ids}, "
            f"markov_head_type={self.markov_head_type}, "
            f"markov_output_mode={self.markov_output_mode}, "
            f"markov_rank={self.markov_rank}"
        )

        self.post_init()

    @property
    def chs_len_per_block(self) -> int:
        return self.condition_slot_count

    @property
    def condition_slot_count(self) -> int:
        """Physical current context slots in the CHS KV path."""
        if self.include_token_embedding_chs:
            return 1 + self.current_chs_slot_count
        return self.current_chs_slot_count

    @property
    def current_chs_slot_count(self) -> int:
        """Configured transformer CHS count; excludes the fixed token embedding."""
        return self.chs_num_layers

    @property
    def history_slot_count(self) -> int:
        return self.sliding_window_size - 1

    @property
    def window_query_count(self) -> int:
        """Unsupervised window query slots prepended to the draft block."""
        return self.history_slot_count

    @property
    def core_draft_query_length(self) -> int:
        """Anchor + MASK queries, excluding the optional window-as-query prefix."""
        return self.block_size + (1 if self.pivot_query_embedding else 0)

    @property
    def history_source_lookback(self) -> int:
        """Tokens needed before an anchor to cover the oldest history slot."""
        return self.history_slot_count

    @property
    def seed_rnn_from_predecessor(self) -> bool:
        """Prime rnn/rnn_easy serial state with ``embed(anchor-1)`` when W>1."""
        return (
            self.sliding_window_size > 1
            and self.markov_head_type in ("rnn", "rnn_easy")
        )

    def get_last_decode_stats(self) -> dict:
        return dict(self._last_decode_stats)

    def set_local_position(self, enabled: bool) -> None:
        """Switch draft-only RoPE between global and window-local positions."""
        self.local_position = bool(enabled)
        if self.config.flashmtp_config is None:
            self.config.flashmtp_config = {}
        self.config.flashmtp_config["local_position"] = self.local_position

    def build_block_position_ids(
        self,
        anchor_positions: torch.Tensor,
        history_position_ids: torch.Tensor,
        history_keep_mask: torch.Tensor,
        draft_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build flattened context/draft RoPE ids for independent draft blocks.

        Local mode subtracts the first valid history position while preserving
        real token gaps. Left-padding slots receive zero and remain masked. V5
        ``fuse``/``token`` returns context positions in
        ``[current CHS | history window]`` order. ``pivot_q`` keeps only CHS in
        context and prepends the window positions to the draft queries. Every
        CHS slot uses the pivot position ``anchor - 1``.
        """
        if anchor_positions.ndim != 2:
            raise ValueError(
                "anchor_positions must have shape (B,N), got "
                f"{tuple(anchor_positions.shape)}."
            )
        if history_position_ids.shape != history_keep_mask.shape:
            raise ValueError(
                "history_position_ids and history_keep_mask must have identical "
                f"shapes, got {tuple(history_position_ids.shape)} and "
                f"{tuple(history_keep_mask.shape)}."
            )
        if history_position_ids.shape[:2] != anchor_positions.shape:
            raise ValueError(
                "History position batch/block dimensions must match anchors."
            )

        draft_length = (
            self.core_draft_query_length if draft_length is None else int(draft_length)
        )
        if draft_length <= 0:
            raise ValueError(f"draft_length must be positive, got {draft_length}.")

        if self.local_position:
            pivot_positions = (anchor_positions - 1).clamp(min=0)
            if history_keep_mask.shape[-1] == 0:
                local_origins = pivot_positions
            else:
                has_history = history_keep_mask.any(dim=-1)
                sentinel = torch.iinfo(history_position_ids.dtype).max
                first_history_positions = torch.where(
                    history_keep_mask,
                    history_position_ids,
                    torch.full_like(history_position_ids, sentinel),
                ).amin(dim=-1)
                local_origins = torch.where(
                    has_history, first_history_positions, pivot_positions
                )
            history_positions = history_position_ids - local_origins.unsqueeze(-1)
            history_positions = torch.where(
                history_keep_mask,
                history_positions,
                torch.zeros_like(history_positions),
            )
            current_positions = (pivot_positions - local_origins).unsqueeze(-1).expand(
                -1, -1, self.condition_slot_count
            )
            draft_positions = self.build_draft_query_position_ids(
                anchor_positions,
                draft_length=draft_length,
                local_origins=local_origins,
            )
        else:
            history_positions = history_position_ids
            current_positions = (anchor_positions - 1).clamp(min=0).unsqueeze(-1)
            current_positions = current_positions.expand(
                -1, -1, self.condition_slot_count
            )
            draft_positions = self.build_draft_query_position_ids(
                anchor_positions,
                draft_length=draft_length,
            )

        context_positions = current_positions
        draft_positions = torch.cat([history_positions, draft_positions], dim=-1)
        return (
            context_positions.reshape(anchor_positions.shape[0], -1),
            draft_positions.reshape(anchor_positions.shape[0], -1),
        )

    def build_draft_query_position_ids(
        self,
        anchor_positions: torch.Tensor,
        draft_length: Optional[int] = None,
        local_origins: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """RoPE ids for core draft queries: known anchor ``a`` then ``a+1 .. a+B-1``."""
        draft_length = (
            self.core_draft_query_length if draft_length is None else int(draft_length)
        )
        if draft_length <= 0:
            raise ValueError(f"draft_length must be positive, got {draft_length}.")
        rest_start = anchor_positions
        if local_origins is not None:
            rest_start = rest_start - local_origins
        start_offset = -1 if self.pivot_query_embedding else 0
        rest_offsets = torch.arange(
            start_offset,
            start_offset + draft_length,
            device=anchor_positions.device,
            dtype=torch.long,
        ).view(1, 1, -1)
        return rest_start.unsqueeze(-1) + rest_offsets

    @property
    def draft_block_len(self) -> int:
        """Parallel draft slots per anchor (1 anchor + remaining MASK tokens)."""
        return self.block_size

    @property
    def unsupervised_query_count(self) -> int:
        """Context-only draft queries that are not supervised."""
        return self.window_query_count + (2 if self.pivot_query_embedding else 1)

    @property
    def draft_query_length(self) -> int:
        """Number of draft queries in one parallel block, including window Q."""
        return self.window_query_count + self.core_draft_query_length

    @property
    def proposal_length(self) -> int:
        """Draft tokens proposed after the anchor; total span is block_size."""
        return self.block_size - 1

    @property
    def max_verify_block_size(self) -> int:
        """Anchor-inclusive target verification window (equals config block_size)."""
        return self.proposal_length + 1

    def set_config_block_size(self, block_size: int) -> None:
        """Update the draft block size (known anchor plus MASK proposals)."""
        if int(block_size) <= 1:
            raise ValueError(
                "block_size must be > 1 so at least one MASK query exists."
            )
        self.block_size = int(block_size)
        self.config.block_size = int(block_size)

    def _prediction_hidden(self, block_hidden: torch.Tensor) -> torch.Tensor:
        """Return one hidden state for each proposed token."""
        return block_hidden[:, -self.proposal_length :, :]

    def _apply_chs_depth_embedding(
        self, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Depth-encode transformer CHS; optionally keep a token-embedding slot."""
        _, _, s_len, h = target_hidden.shape
        if s_len != self.condition_slot_count:
            raise ValueError(
                "Expected "
                f"{self.condition_slot_count} current CHS slots, got {s_len}."
            )
        depth_ids = torch.tensor(
            self.target_layer_ids, device=target_hidden.device, dtype=torch.long
        )
        depth_emb = self.layer_depth_embedding(depth_ids).view(
            1, 1, self.chs_num_layers, h
        )
        if self.include_token_embedding_chs:
            token_ctx = self.context_norm(target_hidden[:, :, :1, :])
            layer_ctx = self.context_norm(target_hidden[:, :, 1:, :] + depth_emb)
            return torch.cat([token_ctx, layer_ctx], dim=2)
        layer_ctx = self.context_norm(target_hidden + depth_emb)
        return layer_ctx

    def build_inference_query_embeddings(
        self,
        embed_tokens: nn.Module,
        draft_input_ids: torch.Tensor,
        pivot_token_ids: Optional[torch.Tensor] = None,
        window_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build draft queries for the configured layout."""
        anchor_embedding = embed_tokens(draft_input_ids)
        if self.pivot_query_embedding:
            if pivot_token_ids is None:
                raise ValueError(
                    "pivot_query_embedding=True requires pivot_token_ids for inference."
                )
            pivot_embedding = embed_tokens(pivot_token_ids)
            if pivot_embedding.ndim == 2:
                pivot_embedding = pivot_embedding.unsqueeze(1)
            queries = torch.cat([pivot_embedding, anchor_embedding], dim=1)
        else:
            queries = anchor_embedding
        if window_embeddings is None or window_embeddings.ndim != 3:
            raise ValueError(
                "window_embeddings with shape (B,T,H) are required."
            )
        queries = torch.cat([window_embeddings, queries], dim=1)
        return queries

    def inference_window_embeddings(
        self, recent_condition_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Window token embeddings prepended to draft queries."""
        if self.history_slot_count:
            return recent_condition_hidden[:, -self.history_slot_count :, :]
        return recent_condition_hidden[:, :0, :]

    def build_inference_current_chs(
        self,
        embed_tokens: nn.Module,
        target_hidden: torch.Tensor,
        pivot_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build current CHS context for inference."""
        if target_hidden.ndim != 4 or target_hidden.shape[2] != self.chs_num_layers:
            raise ValueError(
                f"target_hidden must have shape (B,N,{self.chs_num_layers},H), got "
                f"{tuple(target_hidden.shape)}."
            )
        if target_hidden.shape[1] != 1:
            raise ValueError("Inference current CHS expects exactly one anchor block.")
        if not self.include_token_embedding_chs:
            return target_hidden
        pivot_embedding = embed_tokens(pivot_token_ids)
        if pivot_embedding.ndim == 3 and pivot_embedding.shape[1] == 1:
            pivot_embedding = pivot_embedding.unsqueeze(2)
        elif pivot_embedding.ndim == 2:
            pivot_embedding = pivot_embedding.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(
                "pivot embedding must have shape (B,H) or (B,1,H), got "
                f"{tuple(pivot_embedding.shape)}."
            )
        return torch.cat([pivot_embedding, target_hidden], dim=2)

    def build_inference_context(
        self,
        recent_condition_hidden: torch.Tensor,
        current_target_hidden: torch.Tensor,
        anchor_position: int,
        draft_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build pivot-Q window positions for one draft pass."""
        if recent_condition_hidden.ndim != 3:
            raise ValueError(
                "recent_condition_hidden must have shape (B,T,H), got "
                f"{tuple(recent_condition_hidden.shape)}."
            )
        if current_target_hidden.shape[:3] != (
            recent_condition_hidden.shape[0],
            1,
            self.condition_slot_count,
        ):
            raise ValueError(
                "current_target_hidden must have shape (B,1,S,H) matching the "
                f"condition tensor; got {tuple(current_target_hidden.shape)}."
            )
        anchor_position = int(anchor_position)
        history_source = (
            recent_condition_hidden[:, -self.history_slot_count :, :]
            if self.history_slot_count
            else recent_condition_hidden[:, :0, :]
        )
        history_end = anchor_position
        history_len = history_source.shape[1]
        history_pos = torch.arange(
            history_end - history_len,
            history_end,
            device=recent_condition_hidden.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(
            recent_condition_hidden.shape[0], -1
        )
        history_keep = torch.ones(
            recent_condition_hidden.shape[0],
            1,
            history_len,
            dtype=torch.bool,
            device=recent_condition_hidden.device,
        )
        context_positions, draft_positions = self.build_block_position_ids(
            anchor_positions=torch.full(
                (recent_condition_hidden.shape[0], 1),
                anchor_position,
                dtype=torch.long,
                device=recent_condition_hidden.device,
            ),
            history_position_ids=history_pos.unsqueeze(1),
            history_keep_mask=history_keep,
            draft_length=draft_length,
        )
        return context_positions, draft_positions

    def initialize_inference_condition(
        self,
        pivot_index: int = -1,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Retain the bounded history source ending at the current pivot."""
        if token_embeddings is None or token_embeddings.ndim != 3:
            raise ValueError("token_embeddings with shape (B,T,H) are required.")
        condition_source = token_embeddings
        keep_length = self.history_slot_count
        seq_len = condition_source.shape[1]
        pivot_index = int(pivot_index)
        if pivot_index < 0:
            pivot_index += seq_len
        if not 0 <= pivot_index < seq_len:
            raise IndexError(
                f"pivot_index={pivot_index} is out of range for length {seq_len}."
            )
        through_pivot = condition_source[:, : pivot_index + 1, :]
        return (
            through_pivot[:, -keep_length:, :]
            if keep_length
            else through_pivot[:, :0, :]
        )

    def update_inference_condition(
        self,
        recent_condition_hidden: torch.Tensor,
        pivot_index: int,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rebuild the bounded condition tensor through the new pivot."""
        if token_embeddings is None or token_embeddings.ndim != 3:
            raise ValueError("token_embeddings with shape (B,T,H) are required.")
        condition_new = token_embeddings
        keep_length = self.history_slot_count
        pivot_index = int(pivot_index)
        if not 0 <= pivot_index < condition_new.shape[1]:
            raise IndexError(
                f"pivot_index={pivot_index} is out of range for target output "
                f"length {condition_new.shape[1]}."
            )
        through_new_pivot = condition_new[:, : pivot_index + 1, :]
        condition = torch.cat(
            [recent_condition_hidden, through_new_pivot], dim=1
        )
        return (
            condition[:, -keep_length:, :]
            if keep_length
            else condition[:, :0, :]
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        assert target_hidden is not None and target_hidden.ndim == 4
        target_hidden = self._apply_chs_depth_embedding(target_hidden)
        target_hidden = target_hidden.reshape(
            target_hidden.shape[0], -1, target_hidden.shape[-1]
        )
        noise_len = hidden_states.shape[1]
        if position_ids.shape[1] != noise_len:
            draft_pos = position_ids[:, -noise_len:]
        else:
            draft_pos = position_ids

        rotary_pos = (
            rotary_position_ids if rotary_position_ids is not None else draft_pos
        )
        total_len = rotary_pos.shape[1]
        dummy = hidden_states.new_zeros(
            hidden_states.shape[0], total_len, hidden_states.shape[-1]
        )
        position_embeddings = self.rotary_emb(dummy, rotary_pos)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=draft_pos,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def sample_draft_tokens(
        self,
        *,
        draft_hidden: torch.Tensor,
        lm_head: nn.Module,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        compile_serial_head: bool = False,
        initial_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample standard FlashMTP draft positions using configured head semantics."""
        if initial_prev_token_ids is not None and not self.seed_rnn_from_predecessor:
            raise ValueError(
                "initial_prev_token_ids require sliding_window_size > 1 with "
                "an rnn/rnn_easy serial head."
            )
        base_logits = None
        if self.markov_head is None or markov_output_uses_base_lm_head(
            self.markov_output_mode
        ):
            base_logits = lm_head(draft_hidden)
        if self.markov_head is None:
            assert base_logits is not None
            return sample(base_logits, temperature), base_logits
        if compile_serial_head:
            cache_key = (
                self.markov_output_mode,
                float(temperature),
                initial_prev_token_ids is not None,
            )
            compiled_sampler = self._compiled_serial_sampler_cache.get(cache_key)
            if compiled_sampler is None:
                markov_head = self.markov_head
                output_mode = self.markov_output_mode
                fixed_temperature = float(temperature)
                seed_from_predecessor = initial_prev_token_ids is not None
                if markov_output_uses_base_lm_head(output_mode):
                    def serial_sampler(
                        hidden_states: torch.Tensor,
                        previous_ids: torch.Tensor,
                        additive_logits: torch.Tensor,
                        initial_prev: Optional[torch.Tensor] = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return markov_head.sample_block_tokens(
                            hidden_states=hidden_states,
                            first_prev_token_ids=previous_ids,
                            output_mode=output_mode,
                            base_logits=additive_logits,
                            temperature=fixed_temperature,
                            initial_prev_token_ids=(
                                initial_prev if seed_from_predecessor else None
                            ),
                        )
                else:
                    def serial_sampler(
                        hidden_states: torch.Tensor,
                        previous_ids: torch.Tensor,
                        initial_prev: Optional[torch.Tensor] = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return markov_head.sample_block_tokens(
                            hidden_states=hidden_states,
                            first_prev_token_ids=previous_ids,
                            output_mode=output_mode,
                            base_logits=None,
                            temperature=fixed_temperature,
                            initial_prev_token_ids=(
                                initial_prev if seed_from_predecessor else None
                            ),
                        )
                compiled_sampler = torch.compile(
                    serial_sampler,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                self._compiled_serial_sampler_cache[cache_key] = compiled_sampler
            if base_logits is None:
                if initial_prev_token_ids is None:
                    return compiled_sampler(draft_hidden, first_prev_token_ids)
                return compiled_sampler(
                    draft_hidden, first_prev_token_ids, initial_prev_token_ids
                )
            if initial_prev_token_ids is None:
                return compiled_sampler(
                    draft_hidden, first_prev_token_ids, base_logits
                )
            return compiled_sampler(
                draft_hidden,
                first_prev_token_ids,
                base_logits,
                initial_prev_token_ids,
            )
        return self.markov_head.sample_block_tokens(
            hidden_states=draft_hidden,
            first_prev_token_ids=first_prev_token_ids,
            output_mode=self.markov_output_mode,
            base_logits=base_logits,
            temperature=temperature,
            initial_prev_token_ids=initial_prev_token_ids,
        )

    @staticmethod
    def _format_token_topk(
        logits: torch.Tensor,
        tokenizer,
        top_k: int,
    ) -> list[dict]:
        probs = torch.softmax(logits.float(), dim=-1)
        top_k = max(int(top_k), 1)
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        entries = []
        for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            entries.append(
                {
                    "id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "confidence": float(prob),
                }
            )
        return entries

    @staticmethod
    def _format_token_list(token_ids: torch.Tensor, tokenizer) -> list[dict]:
        return [
            {"id": int(tid), "token": tokenizer.decode([int(tid)])}
            for tid in token_ids.view(-1).tolist()
        ]

    @staticmethod
    def _serialize_topk_entries(entries: list[dict]) -> list[dict]:
        return [
            {
                "token_id": int(e["id"]),
                "token": e["token"],
                "confidence": float(e["confidence"]),
            }
            for e in entries
        ]

    @torch.inference_mode()
    def spec_generate_with_profile(
        self,
        target: nn.Module,
        tokenizer,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        top_k: int = 4,
        print_fn: Callable[..., None] = print,
        profile_records: Optional[list] = None,
    ) -> torch.LongTensor:
        """Legacy-aligned decode with optional JSONL-friendly profiling."""
        self.eval()
        self._last_decode_stats = {
            "accept_lengths": [],
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
        }
        if self.mask_token_id is None:
            raise ValueError(
                "FlashMTPDraftModel.mask_token_id is None. Set flashmtp_config['mask_token_id'] "
                "or draft_model.mask_token_id before spec_generate_with_profile()."
            )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        draft_block_len = self.draft_block_len
        proposal_length = self.proposal_length

        output_ids = torch.full(
            (1, max_length + proposal_length + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)
        past_key_values_target = DynamicCache()

        if target.device.type == "cuda":
            torch.cuda.synchronize(target.device)
        target_start = time.perf_counter()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        if target.device.type == "cuda":
            torch.cuda.synchronize(target.device)
        self._last_decode_stats["target_total_time"] += (
            time.perf_counter() - target_start
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        anchor_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
        print_fn(
            f"\n[Prefill anchor] pos={num_input_tokens} "
            f"token={self._format_token_list(anchor_token[0], tokenizer)}"
        )

        prefill_slot0_confidence: Optional[float] = None
        if profile_records is not None:
            logits_prefill = output.logits[0, -1].float()
            probs_prefill = torch.softmax(logits_prefill, dim=-1)
            aid = int(output_ids[0, num_input_tokens].item())
            prefill_slot0_confidence = float(probs_prefill[aid])

        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
        )
        recent_condition_hidden = self.initialize_inference_condition(
            token_embeddings=target.model.embed_tokens(input_ids),
        )

        start = input_ids.shape[1]
        while start < max_length:
            spec_step_idx = int(self._last_decode_stats["steps"])
            print_fn(f"\n[Spec step {spec_step_idx}] start={start}")
            draft_input_ids = output_ids[:, start : start + draft_block_len].clone()
            block_start_abs = int(start)
            slot0_tid = int(output_ids[0, start].item())

            pivot_token_ids = output_ids[:, start - 1 : start]
            noise_embedding = self.build_inference_query_embeddings(
                target.model.embed_tokens,
                draft_input_ids,
                pivot_token_ids=pivot_token_ids,
                window_embeddings=self.inference_window_embeddings(
                    recent_condition_hidden
                ),
            )
            current_target_hidden = self.build_inference_current_chs(
                target.model.embed_tokens,
                target_hidden,
                pivot_token_ids,
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            draft_start = time.perf_counter()
            ctx_pos_part, draft_block_pos = self.build_inference_context(
                recent_condition_hidden,
                current_target_hidden,
                start,
            )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_hidden = self(
                target_hidden=current_target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_block_pos,
                rotary_position_ids=full_rotary,
                is_causal=False,
            )
            draft_hidden = self._prediction_hidden(block_hidden)
            lm_head = target.lm_head
            sampled_draft_tokens, draft_logits = self.sample_draft_tokens(
                draft_hidden=draft_hidden,
                lm_head=lm_head,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=temperature,
                initial_prev_token_ids=(
                    pivot_token_ids.squeeze(1)
                    if self.seed_rnn_from_predecessor
                    else None
                ),
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            self._last_decode_stats["draft_total_time"] += (
                time.perf_counter() - draft_start
            )

            draft_topk_by_slot: dict[int, list[dict]] = {}
            for slot in range(1, proposal_length + 1):
                topk_entries = self._format_token_topk(
                    draft_logits[0, slot - 1], tokenizer, top_k
                )
                if profile_records is not None:
                    draft_topk_by_slot[slot] = self._serialize_topk_entries(
                        topk_entries
                    )
                print_fn(
                    f"  [Draft slot {slot} | abs_pos={start + slot}] "
                    f"top{top_k}={topk_entries}"
                )

            verify_output_ids = torch.cat(
                [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
            )

            draft_probs_on_logits: Optional[torch.Tensor] = None
            if profile_records is not None:
                draft_probs_on_logits = torch.softmax(draft_logits.float(), dim=-1)

            pending_profile_chunk: list[dict] = []
            if profile_records is not None:
                slot0_row = {
                    "abs_pos": block_start_abs,
                    "slot": 0,
                    "token_id": slot0_tid,
                    "token": tokenizer.decode([slot0_tid]),
                }
                if (
                    prefill_slot0_confidence is not None
                    and block_start_abs == num_input_tokens
                ):
                    slot0_row["confidence"] = round(prefill_slot0_confidence, 4)
                pending_profile_chunk.append(slot0_row)
                for s in sorted(draft_topk_by_slot):
                    cand_list = [
                        {
                            "id": int(c["token_id"]),
                            "t": c["token"],
                            "p": round(float(c["confidence"]), 3),
                        }
                        for c in draft_topk_by_slot[s]
                    ]
                    sampled_tid = int(verify_output_ids[0, s].item())
                    row = {
                        "abs_pos": block_start_abs + int(s),
                        "slot": int(s),
                        "top_k": int(top_k),
                        "candidates": cand_list,
                        "sampled_token_id": sampled_tid,
                        "sampled_token": tokenizer.decode([sampled_tid]),
                        "draft_p_sampled": round(
                            float(draft_probs_on_logits[0, s - 1, sampled_tid].item()),
                            4,
                        ),
                    }
                    pending_profile_chunk.append(row)

            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            target_start = time.perf_counter()
            verify_position_ids = position_ids[
                :, start : start + proposal_length + 1
            ]
            output = target(
                verify_output_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            if target.device.type == "cuda":
                torch.cuda.synchronize(target.device)
            self._last_decode_stats["target_total_time"] += (
                time.perf_counter() - target_start
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (verify_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            accepted_tokens = verify_output_ids[:, : acceptance_length + 1]
            print_fn(
                f"  [Accept] length={acceptance_length + 1} "
                f"tokens={self._format_token_list(accepted_tokens[0], tokenizer)}"
            )

            target_verify: list[dict] = []
            for target_pos in range(min(acceptance_length + 1, output.logits.shape[1])):
                role = "accepted" if target_pos < acceptance_length else "correction"
                target_token = posterior[:, target_pos : target_pos + 1]
                target_top3 = self._format_token_topk(
                    output.logits[0, target_pos], tokenizer, 3
                )
                if profile_records is not None:
                    tid = int(posterior[0, target_pos].item())
                    probs_tp = torch.softmax(
                        output.logits[0, target_pos].float(), dim=-1
                    )
                    target_verify.append(
                        {
                            "verify_step": int(target_pos),
                            "role": role,
                            "abs_pos": int(block_start_abs + target_pos + 1),
                            "chosen_token_id": tid,
                            "chosen_token": tokenizer.decode([tid]),
                            "target_p_chosen": round(float(probs_tp[tid].item()), 4),
                        }
                    )
                print_fn(
                    f"  [Target {role} pos={start + target_pos + 1}] "
                    f"token={self._format_token_list(target_token[0], tokenizer)} "
                    f"top3={target_top3}"
                )

            output_ids[:, start : start + acceptance_length + 1] = accepted_tokens
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
            recent_condition_hidden = self.update_inference_condition(
                recent_condition_hidden,
                pivot_index,
                token_embeddings=target.model.embed_tokens(verify_output_ids),
            )
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
            )
            accept_len_out = int(acceptance_length + 1)
            self._last_decode_stats["accept_lengths"].append(accept_len_out)
            if profile_records is not None:
                profile_records.append(
                    {
                        "block_start": block_start_abs,
                        "accept_length": accept_len_out,
                        "speculative_match_count": int(acceptance_length),
                        "target_verify": target_verify,
                    }
                )
                profile_records.extend(pending_profile_chunk)
            self._last_decode_stats["steps"] += 1
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_tensor
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        decode_timing_after_first_token: bool = False,
        verify_block_size: Optional[int] = None,
        stochastic_verification_mode: str = "match",
        compile_serial_head: bool = False,
    ):
        self.eval()
        stochastic_verification_mode = _validate_stochastic_verification_mode(
            stochastic_verification_mode
        )
        self._last_decode_stats = {
            "accept_lengths": [],
            "decode_wall_time": 0.0,
            "target_total_time": 0.0,
            "draft_total_time": 0.0,
            "steps": 0,
            "verification_mode": stochastic_verification_mode,
            "compile_serial_head": bool(compile_serial_head),
        }
        bsz = input_ids.shape[0]
        use_rejection_sampling = (
            temperature >= 1e-5
            and stochastic_verification_mode == "rejection"
        )
        if use_rejection_sampling and bsz != 1:
            raise ValueError(
                "stochastic rejection sampling currently requires batch size 1."
            )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        draft_block_len = self.draft_block_len
        proposal_length = self.proposal_length
        verify_block_size = (
            proposal_length + 1
            if verify_block_size is None
            else int(verify_block_size)
        )
        if not 1 <= verify_block_size <= proposal_length + 1:
            raise ValueError(
                f"verify_block_size must be in [1, {proposal_length + 1}], got "
                f"{verify_block_size}"
            )
        output_ids = torch.full(
            (bsz, max_length + proposal_length + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = (
            torch.arange(output_ids.shape[1], device=target.device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        past_key_values_target = DynamicCache()

        # Prefill stage (not included in decode wall time)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = gather_pivot_multilayer_inference(
            output.hidden_states,
            self.target_layer_ids,
            -1,
            self.config.num_target_layers,
        )
        recent_condition_hidden = self.initialize_inference_condition(
            token_embeddings=target.model.embed_tokens(input_ids),
        )

        # Decode stage: single cuda-synced wall clock (draft + target + bookkeeping)
        decode_start: float | None = (
            None if decode_timing_after_first_token else _cuda_sync_time(target.device)
        )
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            draft_input_ids = output_ids[:, start : start + draft_block_len].clone()
            pivot_token_ids = output_ids[:, start - 1 : start]
            noise_embedding = self.build_inference_query_embeddings(
                target.model.embed_tokens,
                draft_input_ids,
                pivot_token_ids=pivot_token_ids,
                window_embeddings=self.inference_window_embeddings(
                    recent_condition_hidden
                ),
            )
            current_target_hidden = self.build_inference_current_chs(
                target.model.embed_tokens,
                target_hidden,
                pivot_token_ids,
            )
            ctx_pos_part, draft_block_pos = self.build_inference_context(
                recent_condition_hidden,
                current_target_hidden,
                start,
            )
            full_rotary = torch.cat([ctx_pos_part, draft_block_pos], dim=-1)
            block_hidden = self(
                target_hidden=current_target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_block_pos,
                rotary_position_ids=full_rotary,
                is_causal=False,
            )
            draft_hidden = self._prediction_hidden(block_hidden)
            lm_head = target.lm_head
            draft_temperature = temperature if use_rejection_sampling else 0.0
            sampled_draft_tokens, draft_logits = self.sample_draft_tokens(
                draft_hidden=draft_hidden,
                lm_head=lm_head,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=draft_temperature,
                compile_serial_head=compile_serial_head,
                initial_prev_token_ids=(
                    pivot_token_ids.squeeze(1)
                    if self.seed_rnn_from_predecessor
                    else None
                ),
            )
            all_verify_output_ids = torch.cat(
                [draft_input_ids[:, :1], sampled_draft_tokens], dim=1
            )

            # The draft always consumes and predicts the full configured block.  Only
            # the requested prefix is sent to the target; the remaining draft tokens
            # are deliberately discarded before verification.
            verify_output_ids = all_verify_output_ids[:, :verify_block_size]
            verify_position_ids = position_ids[:, start : start + verify_block_size]
            output = target(
                verify_output_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            if use_rejection_sampling:
                proposal_count = verify_block_size - 1
                acceptance_length, next_token = rejection_sample_verify(
                    proposed_tokens=verify_output_ids[:, 1:],
                    draft_logits=draft_logits[:, :proposal_count, :],
                    target_logits=output.logits,
                    temperature=temperature,
                )
            else:
                posterior = sample(output.logits, temperature)
                acceptance_lengths_per_row = (
                    (verify_output_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)
                )
                acceptance_length = int(
                    acceptance_lengths_per_row.min().item()
                )
                next_token = posterior[:, acceptance_length]
            output_ids[:, start : start + acceptance_length + 1] = verify_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = next_token
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            pivot_index = min(acceptance_length, output.hidden_states[0].shape[1] - 1)
            recent_condition_hidden = self.update_inference_condition(
                recent_condition_hidden,
                pivot_index,
                token_embeddings=target.model.embed_tokens(verify_output_ids),
            )
            target_hidden = gather_pivot_multilayer_inference(
                output.hidden_states,
                self.target_layer_ids,
                pivot_index,
                self.config.num_target_layers,
            )
            acceptance_lengths.append(acceptance_length + 1)
            self._last_decode_stats["accept_lengths"].append(acceptance_length + 1)
            self._last_decode_stats["steps"] += 1

            if decode_timing_after_first_token and decode_start is None:
                decode_start = _cuda_sync_time(target.device)

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        if decode_start is None:
            decode_start = _cuda_sync_time(target.device)
        decode_wall_time = _cuda_sync_time(target.device) - decode_start
        self._last_decode_stats["decode_wall_time"] = decode_wall_time
        # Legacy fields: whole decode wall (no draft/target split)
        self._last_decode_stats["target_total_time"] = decode_wall_time
        self._last_decode_stats["draft_total_time"] = 0.0

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
