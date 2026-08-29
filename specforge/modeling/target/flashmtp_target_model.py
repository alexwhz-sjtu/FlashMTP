from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_group


@dataclass
class FlashMTPTargetOutput:
    hidden_states: Union[
        torch.Tensor, Tuple[torch.Tensor, ...], Dict[int, torch.Tensor]
    ]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]


@dataclass
class TempRolloutPrefillOutput:
    """Captured target states plus live true-sequence paged KV."""

    hidden_states: Union[
        torch.Tensor, Tuple[torch.Tensor, ...], Dict[int, torch.Tensor]
    ]
    handle: "TempRolloutPrefillHandle"


class TempRolloutPrefillHandle:
    """Own immutable true-prefix KV and private greedy-rollout branches."""

    def __init__(
        self,
        target_model: "SGLangFlashMTPTargetModel",
        parent_reqs: List[Any],
        true_token_ids: List[List[int]],
        prefix_indices: List[torch.Tensor],
        tree_cache: Any,
    ) -> None:
        # Keep this as a regular object: FSDP recursively rebuilds dataclass
        # kwargs and would otherwise try to cast the live SGLang state.
        self.target_model = target_model
        self.parent_reqs = parent_reqs
        self.true_token_ids = true_token_ids
        self.prefix_indices = prefix_indices
        self.tree_cache = tree_cache
        # Created on the first private-token extend, then reused in-place by
        # ScheduleBatch.prepare_for_decode for all remaining rollout positions.
        self.branch_batch: Optional[Any] = None
        self.branch_scatter_indices: List[Tuple[int, int]] = []
        self.branch_scatter_tensor: Optional[torch.Tensor] = None
        self.branch_true_prefix_lens: List[int] = []
        self.branch_num_generated = 0
        self.branch_initialized = False
        self.closed = False

    def extend_step(
        self,
        anchor_positions: torch.Tensor,
        generated_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Append one generated token per active branch and return its hidden."""
        if self.closed:
            raise RuntimeError("temp-rollout prefill handle is already closed.")
        return self.target_model._temp_rollout_extend_step(
            self, anchor_positions, generated_ids, active_mask
        )

    def close(self) -> None:
        if self.closed:
            return
        self.target_model._clear_memory_pools()
        self.branch_batch = None
        self.branch_scatter_indices.clear()
        self.branch_scatter_tensor = None
        self.branch_true_prefix_lens.clear()
        self.branch_initialized = False
        self.closed = True

    def __enter__(self) -> "TempRolloutPrefillHandle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def build_temp_rollout_branch_fill_ids(
    true_token_ids: List[int], anchor: int, generated_ids: List[int]
) -> Tuple[List[int], int]:
    """Build an independent ``true_prefix + private_generation`` branch."""
    prefix_len = int(anchor) + 1
    if prefix_len <= 0 or prefix_len > len(true_token_ids):
        raise ValueError(
            f"Invalid temp-rollout anchor {anchor} for true length "
            f"{len(true_token_ids)}."
        )
    if not generated_ids:
        raise ValueError("temp-rollout branch requires one generated token.")
    return true_token_ids[:prefix_len] + generated_ids, prefix_len


class FlashMTPTargetModel(ABC):
    """
    Abstract base class for FlashMTP target model backend.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "FlashMTPTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_flashmtp_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> FlashMTPTargetOutput:
        """Generate context hidden states for FlashMTP training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids

    def temp_rollout_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> TempRolloutPrefillOutput:
        raise NotImplementedError(
            "temp-rollout training currently requires the SGLang backend."
        )


class SGLangFlashMTPTargetModel(FlashMTPTargetModel):
    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangFlashMTPTargetModel":
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.server_args import ServerArgs

        from .sglang_backend import SGLangRunner, wrap_eagle3_logits_processors_in_module

        tp_size = dist.get_world_size(get_tp_group())
        dtype_arg = torch_dtype if torch_dtype is not None else "auto"
        # SGLang's ServerArgs evolves quickly (for example, 0.5.17 removed the
        # old piecewise-CUDA-graph fields used by 0.5.9).  The shared backend
        # config still carries those fields for older environments, so pass
        # only parameters supported by the installed SGLang runtime.
        supported_server_args = set(inspect.signature(ServerArgs).parameters)
        kwargs = {key: value for key, value in kwargs.items() if key in supported_server_args}
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype_arg,
            enable_return_hidden_states=True,  # Critical for FlashMTP
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)

        runner_kwargs = dict(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            server_args=server_args,
            nccl_port=None,
            is_draft_worker=False,
        )
        runner_base_parameters = inspect.signature(SGLangRunner.__mro__[1]).parameters
        if "ps" in runner_base_parameters:
            from sglang.srt.distributed.parallel_state_wrapper import ParallelState

            attn_cp_size = server_args.attn_cp_size
            attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
            attn_tp_size = server_args.tp_size // attn_cp_size // attn_dp_size
            tp_rank = dist.get_rank(get_tp_group())
            runner_kwargs["ps"] = ParallelState(
                tp_rank=tp_rank,
                tp_size=server_args.tp_size,
                pp_rank=0,
                pp_size=1,
                dp_rank=0,
                dp_size=server_args.dp_size,
                attn_tp_rank=tp_rank % attn_tp_size,
                attn_tp_size=attn_tp_size,
                attn_cp_rank=(tp_rank // attn_tp_size) % attn_cp_size,
                attn_cp_size=attn_cp_size,
                attn_dp_rank=tp_rank // (attn_tp_size * attn_cp_size),
                attn_dp_size=attn_dp_size,
                moe_ep_rank=moe_ep_rank,
                moe_ep_size=server_args.ep_size,
                moe_dp_rank=0,
                moe_dp_size=server_args.moe_dp_size,
                dcp_size=server_args.dcp_size,
                gpu_id=torch.cuda.current_device(),
            )
        else:
            runner_kwargs.update(
                tp_rank=dist.get_rank(get_tp_group()),
                tp_size=server_args.tp_size,
                moe_ep_rank=moe_ep_rank,
                moe_ep_size=server_args.ep_size,
                pp_rank=0,
                pp_size=1,
            )
        model_runner = SGLangRunner(**runner_kwargs)
        # SGLang >= 0.5.17 splits model loading from KV/request-pool
        # allocation.  The embedded training runner bypasses Scheduler, which
        # normally performs this second phase, so initialize the pools here.
        if model_runner.req_to_token_pool is None:
            model_runner.alloc_memory_pool()
        if not hasattr(model_runner, "attn_backend"):
            model_runner.init_attention_backends()
        wrap_eagle3_logits_processors_in_module(
            model_runner.model, return_full_logits=False
        )
        instance = cls(model_runner)
        # Default: capture all layers for FlashMTP
        instance.set_capture_layers([])
        return instance

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture.

        If layer_ids is None or empty, capture ALL layers (for FlashMTP mode).
        Note: SGLang's set_eagle3_layers_to_capture adds +1 offset to layer indices.
        """
        if layer_ids is None or len(layer_ids) == 0:
            # Capture all layers: range [0, num_hidden_layers)
            # SGLang will add +1 offset internally
            num_layers = getattr(self.model_runner.model_config, "num_hidden_layers", 36)
            layer_ids = list(range(num_layers))

        super().set_capture_layers(layer_ids)
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)
        else:
            self._install_generic_layer_capture(layer_ids)

    def _install_generic_layer_capture(self, layer_ids: List[int]) -> None:
        """Add intermediate-state capture to SGLang models missing EAGLE3 hooks.

        SGLang 0.5.9's Qwen3.5 implementation supports inference but does not
        yet expose ``set_eagle3_layers_to_capture``.  Its language backbone has
        the same residual convention as the other SGLang decoder models, so
        layer forward hooks can recover the post-layer hidden state without
        modifying the installed SGLang package.
        """
        top_model = self.model_runner.model
        language_model = getattr(top_model, "model", None)
        layers = getattr(language_model, "layers", None)
        if language_model is None or layers is None:
            raise NotImplementedError(
                f"{type(top_model).__name__} does not support intermediate-layer "
                "capture and has no hookable language-model layers."
            )

        num_layers = len(layers)
        invalid_ids = [idx for idx in layer_ids if idx < 0 or idx >= num_layers]
        if invalid_ids:
            raise ValueError(
                f"Invalid capture layer ids {invalid_ids} for {num_layers} layers."
            )

        # The normalized final state is already returned separately by the
        # logits processor.  Capture only non-final layers as auxiliary states.
        aux_layer_ids = [idx for idx in layer_ids if idx != num_layers - 1]
        language_model._flashmtp_capture_layer_ids = aux_layer_ids

        if not hasattr(language_model, "_flashmtp_capture_handles"):
            handles = []

            def make_hook(layer_idx):
                def capture_hook(_module, _inputs, output):
                    wanted = getattr(
                        language_model, "_flashmtp_capture_layer_ids", []
                    )
                    if layer_idx not in wanted:
                        return
                    if not isinstance(output, tuple) or len(output) < 2:
                        raise ValueError(
                            "Generic SGLang layer capture expects "
                            "(hidden_states, residual) output."
                        )
                    hidden, residual = output[:2]
                    state = hidden if residual is None else hidden + residual
                    language_model._flashmtp_captured[layer_idx] = state

                return capture_hook

            for layer_idx, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(make_hook(layer_idx)))
            language_model._flashmtp_capture_handles = handles

            original_forward = language_model.forward

            def forward_with_capture(this, *args, **kwargs):
                this._flashmtp_captured = {}
                final_hidden = original_forward(*args, **kwargs)
                wanted = this._flashmtp_capture_layer_ids
                missing = [idx for idx in wanted if idx not in this._flashmtp_captured]
                if missing:
                    raise RuntimeError(
                        f"Failed to capture SGLang hidden states for layers {missing}."
                    )
                aux_hidden = [this._flashmtp_captured[idx] for idx in wanted]
                return final_hidden, aux_hidden

            language_model.forward = MethodType(forward_with_capture, language_model)

    @staticmethod
    def _unpack_runner_output(runner_output):
        if isinstance(runner_output, tuple):
            return runner_output[0]
        if (
            hasattr(runner_output, "logits_output")
            and runner_output.logits_output is not None
        ):
            return runner_output.logits_output
        return runner_output

    def _aux_hidden_to_layer_tuple(
        self,
        aux_hidden: torch.Tensor,
        last_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Split SGLang concatenated aux hidden states into per-layer tensors."""
        hidden_size = self.model_runner.model_config.hidden_size
        bsz, seq_len, total_h = aux_hidden.shape
        if total_h % hidden_size != 0:
            raise ValueError(
                f"Unexpected aux_hidden_states shape {tuple(aux_hidden.shape)}; "
                f"last dim must be divisible by hidden_size={hidden_size}"
            )
        num_layers = total_h // hidden_size
        reshaped = aux_hidden.view(bsz, seq_len, num_layers, hidden_size)
        layers = [reshaped[:, :, layer_idx, :] for layer_idx in range(num_layers)]
        if last_hidden is not None:
            if last_hidden.shape != (bsz, seq_len, hidden_size):
                raise ValueError(
                    f"Unexpected last_hidden_states shape {tuple(last_hidden.shape)}; "
                    f"expected {(bsz, seq_len, hidden_size)}"
                )
            layers.append(last_hidden)
        return tuple(layers)

    def _new_tree_cache(self):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixCache

        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        return RadixCache(cache_params)

    def _clear_memory_pools(self) -> None:
        """Clear all embedded-runner request slots and KV after a microbatch."""
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def _set_logits_processor_output(self, *, return_last_hidden: bool) -> None:
        from .sglang_backend.utils import LogitsProcessorForEAGLE3

        for _, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.return_last_hidden_states = bool(return_last_hidden)
                module.return_logits = False

    def _prepare_mlp_sync(self, batch) -> None:
        """Populate SGLang's TP/DP metadata for an embedded batch."""
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    def _prepare_extend_batch(self, reqs, *, tree_cache):
        """Allocate request slots/KV and prepare a persistent extend batch."""
        from array import array

        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        # SGLang >= 0.5.17 represents the admitted prefill span explicitly.
        # The full Scheduler normally initializes these fields before creating
        # a ScheduleBatch; the embedded training path must do the same.
        for req in reqs:
            if getattr(req, "extend_range", None) is None and hasattr(
                req, "set_extend_range"
            ):
                req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._prepare_mlp_sync(batch)
        return batch

    def _forward_prepared_batch(self, batch, *, capture_full: bool):
        """Run one already prepared extend/decode batch and return raw output."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        self._set_logits_processor_output(return_last_hidden=True)

        capture_mode = CaptureHiddenMode.FULL if capture_full else CaptureHiddenMode.NULL
        if hasattr(batch, "get_model_worker_batch"):
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            forward_batch.capture_hidden_mode = capture_mode
        else:
            # SGLang >= 0.5.17 constructs ForwardBatch directly from the
            # scheduler batch and accepts capture mode as an explicit override.
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                capture_hidden_mode=capture_mode,
                return_hidden_states_before_norm=False,
            )

        return self._unpack_runner_output(self.model_runner.forward(forward_batch))

    @torch.no_grad()
    def _run_extend(self, reqs, *, tree_cache, capture_full: bool):
        """Run an embedded SGLang extend without releasing its paged KV."""
        batch = self._prepare_extend_batch(reqs, tree_cache=tree_cache)
        output = self._forward_prepared_batch(batch, capture_full=capture_full)
        extend_lens = [int(req.extend_input_len) for req in reqs]
        last_hidden_list = None
        if (
            hasattr(output, "last_hidden_states")
            and output.last_hidden_states is not None
        ):
            last_hidden_list = torch.split(
                output.last_hidden_states, extend_lens, dim=0
            )

        hidden_states_list = None
        if capture_full:
            if (
                hasattr(output, "aux_hidden_states")
                and output.aux_hidden_states is not None
            ):
                hidden_states_list = torch.split(
                    output.aux_hidden_states, extend_lens, dim=0
                )
            elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                hidden_states_list = torch.split(
                    output.hidden_states, extend_lens, dim=0
                )
            else:
                raise ValueError("SGLang prefill output does not contain hidden states.")

        return hidden_states_list, last_hidden_list

    @torch.no_grad
    def _extend(self, reqs):
        """Compatibility path used by baseline training; release KV immediately."""
        tree_cache = self._new_tree_cache()
        try:
            return self._run_extend(reqs, tree_cache=tree_cache, capture_full=True)
        finally:
            self._clear_memory_pools()

    @torch.no_grad()
    def temp_rollout_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> TempRolloutPrefillOutput:
        """Prefill true data once and retain its KV for independent anchors."""
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, seq].")
        if int(self.model_runner.server_args.page_size) != 1:
            raise ValueError(
                "temp-rollout arbitrary-anchor KV sharing requires SGLang "
                f"page_size=1, got {self.model_runner.server_args.page_size}."
            )

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs: List[Any] = []
        true_token_ids: List[List[int]] = []
        for batch_idx in range(input_ids.size(0)):
            valid_len = int(attention_mask[batch_idx].sum().item())
            ids = input_ids[batch_idx, :valid_len].tolist()
            req = Req(
                rid=f"temp-rollout-parent-{batch_idx}",
                origin_input_text="",
                origin_input_ids=ids,
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(ids)
            reqs.append(req)
            true_token_ids.append(ids)

        tree_cache = self._new_tree_cache()
        try:
            hidden_list, last_hidden_list = self._run_extend(
                reqs, tree_cache=tree_cache, capture_full=True
            )
            if hidden_list is None or last_hidden_list is None:
                raise ValueError(
                    "SGLang temp-rollout prefill did not return required states."
                )

            req_pool = self.model_runner.req_to_token_pool.req_to_token
            prefix_indices = [
                req_pool[req.req_pool_idx, : len(ids)].to(torch.int64).clone()
                for req, ids in zip(reqs, true_token_ids)
            ]
            padded_length = input_ids.size(1)

            def _pad_sequence(hidden: torch.Tensor) -> torch.Tensor:
                if hidden.size(0) > padded_length:
                    raise ValueError(
                        "SGLang returned more hidden positions than the padded input: "
                        f"{hidden.size(0)} > {padded_length}."
                    )
                if hidden.size(0) == padded_length:
                    return hidden
                return torch.nn.functional.pad(
                    hidden, (0, 0, 0, padded_length - hidden.size(0))
                )

            # Requests may have different true lengths. Pad only the returned
            # hidden tensors; their paged KV remains allocated at true length.
            batched_aux = torch.stack(
                [_pad_sequence(hidden) for hidden in hidden_list], dim=0
            )
            batched_last = torch.stack(
                [_pad_sequence(hidden) for hidden in last_hidden_list], dim=0
            )
            layer_tensors = self._aux_hidden_to_layer_tuple(
                batched_aux, batched_last
            )
            captured_ids = self.capture_layer_ids or []
            num_layers = getattr(
                self.model_runner.model_config, "num_hidden_layers", None
            )
            if num_layers is not None and len(captured_ids) < num_layers:
                hidden_states = {
                    layer_id: layer_tensors[idx]
                    for idx, layer_id in enumerate(captured_ids)
                }
            else:
                hidden_states = layer_tensors

            handle = TempRolloutPrefillHandle(
                target_model=self,
                parent_reqs=reqs,
                true_token_ids=true_token_ids,
                prefix_indices=prefix_indices,
                tree_cache=tree_cache,
            )
            return TempRolloutPrefillOutput(
                hidden_states=hidden_states, handle=handle
            )
        except Exception:
            self._clear_memory_pools()
            raise

    @torch.no_grad()
    def _temp_rollout_extend_step(
        self,
        handle: TempRolloutPrefillHandle,
        anchor_positions: torch.Tensor,
        generated_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Append one token with a persistent SGLang branch decode batch."""
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        if anchor_positions.ndim != 2 or generated_ids.ndim != 3:
            raise ValueError("anchors must be [B,N] and generated_ids [B,N,S].")
        if generated_ids.shape[:2] != anchor_positions.shape:
            raise ValueError("generated_ids batch/block dimensions must match anchors.")
        if active_mask.shape != anchor_positions.shape:
            raise ValueError("active_mask must match anchor_positions.")
        batch_size, num_blocks = anchor_positions.shape
        if batch_size != len(handle.true_token_ids):
            raise ValueError("temp-rollout branch batch does not match prefill.")
        if generated_ids.size(-1) < 1:
            raise ValueError("generated_ids must contain the newest token.")

        hidden_size = int(self.model_runner.model_config.hidden_size)
        result = torch.zeros(
            batch_size,
            num_blocks,
            hidden_size,
            dtype=torch.bfloat16,
            device=generated_ids.device,
        )
        expected_generation = handle.branch_num_generated + 1
        if generated_ids.size(-1) != expected_generation:
            raise RuntimeError(
                "temp-rollout persistent batch received a non-consecutive step: "
                f"expected {expected_generation} generated tokens, got "
                f"{generated_ids.size(-1)}."
            )

        # The first private token is admitted through one extend. This creates
        # the Req objects, request-pool rows, and ScheduleBatch that all later
        # positions reuse through prepare_for_decode.
        if not handle.branch_initialized:
            sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
            reqs: List[Any] = []
            scatter_indices: List[Tuple[int, int]] = []
            true_prefix_lens: List[int] = []
            anchors_cpu = anchor_positions.detach().cpu()
            generated_cpu = generated_ids.detach().cpu()
            active_cpu = active_mask.detach().cpu()
            for batch_idx in range(batch_size):
                true_ids = handle.true_token_ids[batch_idx]
                true_locs = handle.prefix_indices[batch_idx]
                for block_idx in range(num_blocks):
                    if not bool(active_cpu[batch_idx, block_idx]):
                        continue
                    anchor = int(anchors_cpu[batch_idx, block_idx])
                    branch_ids = generated_cpu[batch_idx, block_idx].tolist()
                    fill_ids, true_prefix_len = build_temp_rollout_branch_fill_ids(
                        true_ids, anchor, branch_ids
                    )
                    req = Req(
                        rid=f"temp-rollout-branch-{batch_idx}-{block_idx}",
                        origin_input_text="",
                        origin_input_ids=fill_ids,
                        sampling_params=sampling_params,
                    )
                    req.prefix_indices = true_locs[:true_prefix_len]
                    req.fill_ids = fill_ids
                    req.extend_input_len = 1
                    reqs.append(req)
                    scatter_indices.append((batch_idx, block_idx))
                    true_prefix_lens.append(true_prefix_len)

            handle.branch_initialized = True
            handle.branch_num_generated = 1
            handle.branch_scatter_indices = scatter_indices
            handle.branch_true_prefix_lens = true_prefix_lens
            if not reqs:
                return result
            handle.branch_scatter_tensor = torch.tensor(
                scatter_indices,
                dtype=torch.long,
                device=generated_ids.device,
            )
            batch = self._prepare_extend_batch(
                reqs, tree_cache=handle.tree_cache
            )
            handle.branch_batch = batch
            output = self._forward_prepared_batch(batch, capture_full=False)
        else:
            handle.branch_num_generated = generated_ids.size(-1)
            batch = handle.branch_batch
            if batch is None or not handle.branch_scatter_indices:
                return result

            assert handle.branch_scatter_tensor is not None
            scatter = handle.branch_scatter_tensor
            running_active = active_mask[scatter[:, 0], scatter[:, 1]]
            if not bool(running_active.all().item()):
                keep_indices = (
                    running_active.nonzero(as_tuple=False).flatten().cpu().tolist()
                )
                keep_set = set(keep_indices)
                req_pool = self.model_runner.req_to_token_pool
                kv_allocator = self.model_runner.token_to_kv_pool_allocator
                token_table = req_pool.req_to_token
                for index, req in enumerate(batch.reqs):
                    if index in keep_set:
                        continue
                    seq_len = int(batch.seq_lens_cpu[index])
                    true_prefix_len = handle.branch_true_prefix_lens[index]
                    private_pages = token_table[
                        req.req_pool_idx, true_prefix_len:seq_len
                    ].clone()
                    kv_allocator.free(private_pages)
                    req_pool.free(req)

                if not keep_indices:
                    handle.branch_batch = None
                    handle.branch_scatter_indices = []
                    handle.branch_scatter_tensor = None
                    handle.branch_true_prefix_lens = []
                    return result

                batch.filter_batch(keep_indices=keep_indices)
                handle.branch_scatter_indices = [
                    handle.branch_scatter_indices[index]
                    for index in keep_indices
                ]
                handle.branch_true_prefix_lens = [
                    handle.branch_true_prefix_lens[index]
                    for index in keep_indices
                ]
                handle.branch_scatter_tensor = scatter[
                    torch.tensor(
                        keep_indices, dtype=torch.long, device=scatter.device
                    )
                ]
                scatter = handle.branch_scatter_tensor

            # Decode consumes the newest token directly from GPU. No new Req,
            # request-slot allocation, page-table reconstruction, or token D2H.
            batch.output_ids = generated_ids[
                scatter[:, 0], scatter[:, 1], -1
            ].to(torch.long)
            batch.prepare_for_decode()
            self._prepare_mlp_sync(batch)
            output = self._forward_prepared_batch(batch, capture_full=False)

        if (
            not hasattr(output, "last_hidden_states")
            or output.last_hidden_states is None
        ):
            raise ValueError(
                "SGLang persistent temp-rollout did not return final hidden states."
            )
        last_hidden = output.last_hidden_states
        scatter = handle.branch_scatter_tensor
        assert scatter is not None
        if last_hidden.shape != (scatter.size(0), hidden_size):
            raise ValueError(
                "Unexpected persistent temp-rollout hidden shape: "
                f"expected {(scatter.size(0), hidden_size)}, "
                f"got {tuple(last_hidden.shape)}."
            )
        result[scatter[:, 0], scatter[:, 1]] = last_hidden
        return result

    @torch.no_grad()
    def generate_flashmtp_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> FlashMTPTargetOutput:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            valid_len = int(curr_attn.sum().item())
            true_ids = curr_ids.view(-1)[:valid_len].tolist()
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=true_ids,
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        hidden_states_list, last_hidden_list = self._extend(reqs)

        padded_length = input_ids.size(1)

        def _pad_sequence(hidden: torch.Tensor) -> torch.Tensor:
            if hidden.size(0) > padded_length:
                raise ValueError(
                    f"Target hidden length {hidden.size(0)} exceeds padded input "
                    f"length {padded_length}."
                )
            return torch.nn.functional.pad(
                hidden, (0, 0, 0, padded_length - hidden.size(0))
            )

        batched_aux = torch.stack(
            [_pad_sequence(h) for h in hidden_states_list], dim=0
        )
        batched_last = None
        if last_hidden_list is not None:
            batched_last = torch.stack(
                [_pad_sequence(h) for h in last_hidden_list], dim=0
            )

        hidden_size = self.model_runner.model_config.hidden_size
        num_transformer_layers = getattr(
            self.model_runner.model_config, "num_hidden_layers", None
        )
        if (
            batched_aux.ndim == 3
            and batched_aux.shape[-1] % hidden_size == 0
            and batched_aux.shape[-1] > hidden_size
        ):
            layer_tensors = self._aux_hidden_to_layer_tuple(
                batched_aux, batched_last
            )
            captured_ids = self.capture_layer_ids or []
            if (
                num_transformer_layers is not None
                and len(captured_ids) < num_transformer_layers
            ):
                # Partial capture: map absolute layer id -> hidden tensor.
                hidden_states = {
                    layer_id: layer_tensors[idx]
                    for idx, layer_id in enumerate(captured_ids)
                }
            else:
                hidden_states = layer_tensors
        else:
            raise ValueError(
                "SGLang returned single-layer hidden states; expected concatenated "
                "aux_hidden_states from all captured layers. Ensure "
                "wrap_eagle3_logits_processors_in_module is applied."
            )
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return FlashMTPTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class HFFlashMTPTargetModel(FlashMTPTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFFlashMTPTargetModel":

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model)

    @torch.no_grad()
    def generate_flashmtp_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> FlashMTPTargetOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        # Take all layers except embedding (index 0) and concat in feature dim
        hidden_states = outputs.hidden_states[1:]

        return FlashMTPTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


def get_flashmtp_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> FlashMTPTargetModel:
    if backend == "sglang":
        # Import sglang only when the sglang backend is requested.
        return SGLangFlashMTPTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFFlashMTPTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
