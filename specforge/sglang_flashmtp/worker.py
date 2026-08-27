from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import torch

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_accept_len_and_bonus,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func

from .config import parse_flashmtp_config, validate_target_compatibility
from .state import (
    FlashMTPDraftInput,
    FlashMTPDraftInputV2,
    FlashMTPVerifyInput,
)

logger = logging.getLogger(__name__)


@dataclass
class _DraftCudaGraph:
    graph: torch.cuda.CUDAGraph
    block_ids: torch.Tensor
    pivot_hidden: torch.Tensor
    draft_tokens: torch.Tensor
    hidden_states: torch.Tensor


def _as_int32_tensor(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.int32, non_blocking=True)
    return torch.tensor(value, device=device, dtype=torch.int32)


class _FlashMTPWorkerBase(DFlashWorker):
    """Shared FlashMTP worker pieces without constructing DFlash's draft runner."""

    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = torch.device(target_worker.device)
        self.page_size = int(server_args.page_size)
        self._logged_first_verify = False
        self._warned_sampling_fallback = False

        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.speculative_draft_model_path,
            model_revision=server_args.speculative_draft_model_revision,
            is_draft_model=True,
        )
        self.flashmtp_config = parse_flashmtp_config(draft_model_config.hf_config)
        validate_target_compatibility(
            self.flashmtp_config, self.model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is not None and int(
            server_args.speculative_num_draft_tokens
        ) != self.flashmtp_config.block_size:
            raise ValueError(
                "FlashMTP block size is fixed by training: "
                f"checkpoint={self.flashmtp_config.block_size}, "
                f"runtime={server_args.speculative_num_draft_tokens}."
            )
        self.block_size = self.flashmtp_config.block_size
        self.speculative_num_draft_tokens = self.block_size
        self._mask_token_id = self.flashmtp_config.mask_token_id

        # Load only weights/model. In particular, do not instantiate a draft
        # ModelRunner, attention backend, request table, allocator, or KV pool.
        load_config = deepcopy(self.model_runner.load_config)
        load_config.tp_rank = tp_rank
        loader = get_model_loader(load_config, draft_model_config)
        self.draft_model = loader.load_model(
            model_config=draft_model_config,
            device_config=DeviceConfig(str(self.device.type), gpu_id),
        ).eval()

        target_model = self.model_runner.model
        self.embed_module = target_model.get_input_embeddings()
        self.lm_head = getattr(target_model, "lm_head", None)
        if (
            self.lm_head is None
            or not hasattr(self.lm_head, "weight")
            or not hasattr(self.lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "FlashMTP requires the target's vocab-parallel lm_head with weight and shard_indices."
            )
        vocab_size = int(self.model_runner.model_config.vocab_size)
        if self._mask_token_id >= vocab_size:
            raise ValueError(
                f"FlashMTP mask_token_id={self._mask_token_id} is outside target vocab={vocab_size}."
            )

        self._block_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._block_ids: Optional[torch.Tensor] = None
        self._positions: Optional[torch.Tensor] = None
        self._draft_tokens: Optional[torch.Tensor] = None
        self._out_tokens: Optional[torch.Tensor] = None
        self._buffer_capacity = 0
        self._draft_graphs: dict[int, _DraftCudaGraph] = {}
        self._draft_graph_failed: set[int] = set()
        configured_graph_bs = getattr(server_args, "cuda_graph_bs", ()) or ()
        self._draft_graph_batch_sizes = {
            int(value) for value in configured_graph_bs if int(value) > 0
        }
        self._enable_draft_cuda_graph = (
            self.device.type == "cuda" and not server_args.disable_cuda_graph
        )

        # Buffers used by DFlashWorker's optimized vocab-parallel greedy sampler.
        self._draft_greedy_gathered_max_buf = None
        self._draft_greedy_gathered_ids_buf = None
        self._draft_greedy_gather_cap = 0
        self._draft_greedy_best_rank_buf = None
        self._draft_greedy_rank_index_buf = None
        self._draft_greedy_selected_ids_buf = None
        self._draft_greedy_index_cap = 0

        if tp_rank == 0:
            logger.info(
                "Initialized FlashMTP without draft KV cache: block=%d, context_layers=%d",
                self.block_size,
                self.flashmtp_config.num_context_tokens,
            )

    def _ensure_buffers(self, batch_size: int) -> None:
        if self._buffer_capacity >= batch_size:
            return
        capacity = max(batch_size, max(1, self._buffer_capacity * 2))
        shape = (capacity, self.block_size)
        self._block_ids = torch.empty(shape, device=self.device, dtype=torch.long)
        self._positions = torch.empty(shape, device=self.device, dtype=torch.int64)
        self._draft_tokens = torch.empty(shape, device=self.device, dtype=torch.long)
        self._out_tokens = torch.empty(shape, device=self.device, dtype=torch.long)
        self._buffer_capacity = capacity

    def _validate_request_mode(self, batch) -> None:
        sampling_info = batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise ValueError(
                "FlashMTP SGLang currently supports greedy decoding only "
                "(temperature=0, top_k=1)."
            )
        if bool(getattr(batch, "return_logprob", False)):
            raise ValueError("FlashMTP SGLang does not support return_logprob yet.")
        if bool(getattr(batch, "has_grammar", False)):
            raise ValueError("FlashMTP SGLang does not support grammar decoding yet.")

    def _reshape_captured(self, hidden: torch.Tensor) -> torch.Tensor:
        expected = (
            self.flashmtp_config.num_captured_tokens
            * self.flashmtp_config.hidden_size
        )
        if hidden.ndim != 2 or hidden.shape[-1] != expected:
            raise RuntimeError(
                "FlashMTP target hidden feature mismatch: "
                f"expected [N, {expected}], got {tuple(hidden.shape)}."
            )
        return hidden.view(
            hidden.shape[0],
            self.flashmtp_config.num_captured_tokens,
            self.flashmtp_config.hidden_size,
        )

    def _prepend_raw_embedding(
        self, captured: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.flashmtp_config.include_embedding_chs:
            return captured
        raw = self.embed_module(token_ids.long()).unsqueeze(1)
        return torch.cat([raw, captured], dim=1)

    def _prefill_pivot(
        self, hidden: torch.Tensor, extend_seq_lens, input_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden = self._reshape_captured(hidden)
        lengths = _as_int32_tensor(extend_seq_lens, hidden.device).to(torch.int64)
        last_indices = lengths.cumsum(0) - 1
        captured = hidden.index_select(0, last_indices)
        return self._prepend_raw_embedding(
            captured, input_ids.index_select(0, last_indices)
        )

    def _verify_pivot(
        self,
        hidden: torch.Tensor,
        draft_tokens: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> torch.Tensor:
        captured = self._reshape_captured(hidden)
        row = torch.arange(captured.shape[0], device=captured.device)
        raw_ids = draft_tokens[row, commit_lens.to(torch.int64) - 1]
        return self._prepend_raw_embedding(captured, raw_ids)

    def _draft_compute(
        self, block_ids: torch.Tensor, pivot_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(block_ids.shape[0])
        noise_embedding = self.embed_module(block_ids)
        hidden = self.draft_model(noise_embedding, pivot_hidden)
        prediction_hidden = hidden[:, 1:]
        if self.draft_model.markov_head is None:
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=prediction_hidden.reshape(-1, hidden.shape[-1]),
                lm_head=self.lm_head,
            ).view(batch_size, self.block_size - 1)
        else:
            markov = self.draft_model.markov_head
            state = markov.initial_state(prediction_hidden)
            prev_ids = block_ids[:, 0]
            sampled: list[torch.Tensor] = []
            for position in range(self.block_size - 1):
                latent, state = markov.step(
                    prev_token_ids=prev_ids,
                    hidden_states=prediction_hidden[:, position],
                    state=state,
                )
                prev_ids = self._greedy_sample_from_vocab_parallel_head(
                    hidden_states=latent,
                    lm_head=markov.output_proj,
                )
                sampled.append(prev_ids.unsqueeze(1))
            draft_next = torch.cat(sampled, dim=1)
        draft_tokens = torch.cat([block_ids[:, :1], draft_next], dim=1)
        return draft_tokens, hidden

    def _capture_draft_graph(self, batch_size: int) -> Optional[_DraftCudaGraph]:
        if batch_size in self._draft_graph_failed:
            return None
        static_block = torch.full(
            (batch_size, self.block_size),
            self._mask_token_id,
            device=self.device,
            dtype=torch.long,
        )
        static_pivot = torch.zeros(
            (
                batch_size,
                self.flashmtp_config.num_context_tokens,
                self.flashmtp_config.hidden_size,
            ),
            device=self.device,
            dtype=self.draft_model.norm.weight.dtype,
        )
        try:
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    self._draft_compute(static_block, static_pivot)
            torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                draft_tokens, hidden = self._draft_compute(
                    static_block, static_pivot
                )
            captured = _DraftCudaGraph(
                graph=graph,
                block_ids=static_block,
                pivot_hidden=static_pivot,
                draft_tokens=draft_tokens,
                hidden_states=hidden,
            )
            self._draft_graphs[batch_size] = captured
            if self.tp_rank == 0:
                logger.info("Captured FlashMTP draft CUDA graph for batch=%d", batch_size)
            return captured
        except Exception as exc:
            self._draft_graph_failed.add(batch_size)
            logger.warning(
                "FlashMTP draft CUDA graph capture failed for batch=%d; using eager: %s",
                batch_size,
                exc,
            )
            return None

    def _draft(
        self, verified_id: torch.Tensor, pivot_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(verified_id.shape[0])
        use_graph = (
            self._enable_draft_cuda_graph
            and batch_size in self._draft_graph_batch_sizes
        )
        captured = self._draft_graphs.get(batch_size)
        if use_graph and captured is None:
            captured = self._capture_draft_graph(batch_size)
        if captured is not None:
            captured.block_ids[:, 0].copy_(verified_id.to(torch.long))
            captured.pivot_hidden.copy_(pivot_hidden)
            captured.graph.replay()
            return captured.draft_tokens, captured.hidden_states

        self._ensure_buffers(batch_size)
        assert self._block_ids is not None
        assert self._draft_tokens is not None
        block_ids = self._block_ids[:batch_size]
        block_ids.fill_(self._mask_token_id)
        block_ids[:, 0].copy_(verified_id.to(torch.long))
        computed_tokens, hidden = self._draft_compute(block_ids, pivot_hidden)
        draft_tokens = self._draft_tokens[:batch_size]
        draft_tokens.copy_(computed_tokens)
        return draft_tokens, hidden

    def _global_positions(self, prefix_lens: torch.Tensor) -> torch.Tensor:
        batch_size = int(prefix_lens.shape[0])
        self._ensure_buffers(batch_size)
        assert self._positions is not None
        positions = self._positions[:batch_size]
        torch.add(prefix_lens.to(torch.int64)[:, None], self._block_offsets, out=positions)
        return positions


class FlashMTPWorker(_FlashMTPWorkerBase):
    """Spec-v1 continuous-batching worker."""

    def forward_batch_generation(
        self, batch: Union[ScheduleBatch, ModelWorkerBatch], **kwargs
    ) -> GenerationBatchResult:
        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)
        self._validate_request_mode(batch)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            worker_batch = batch.get_model_worker_batch()
            worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            result = self.target_worker.forward_batch_generation(worker_batch, **kwargs)
            hidden = result.logits_output.hidden_states
            if hidden is None or worker_batch.extend_seq_lens is None:
                raise RuntimeError("FlashMTP prefill requires target hidden states and extend lengths.")
            pivot = self._prefill_pivot(
                hidden, worker_batch.extend_seq_lens, worker_batch.input_ids
            )
            batch.spec_info = FlashMTPDraftInput(
                verified_id=result.next_token_ids.to(torch.int64), pivot_hidden=pivot
            )
            result.logits_output.hidden_states = None
            return result

        state = batch.spec_info
        if not isinstance(state, FlashMTPDraftInput):
            raise RuntimeError("FlashMTP decode is missing FlashMTPDraftInput state.")
        draft_tokens, _ = self._draft(state.verified_id, state.pivot_hidden)
        positions = self._global_positions(batch.seq_lens)
        verify = FlashMTPVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions.reshape(-1),
            draft_token_num=self.block_size,
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        verify.prepare_for_verify(
            batch, self.page_size, build_custom_mask=build_custom_mask
        )
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify
        batch.return_hidden_states = False

        worker_batch = batch.get_model_worker_batch()
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        result = self.target_worker.forward_batch_generation(
            worker_batch, is_verify=True, **kwargs
        )
        new_id, commit_lens, pivot_flat, accept_lens_cpu = verify.verify(
            batch=batch, logits_output=result.logits_output, page_size=self.page_size
        )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )
        batch.spec_info = FlashMTPDraftInput(
            verified_id=new_id,
            pivot_hidden=self._verify_pivot(
                pivot_flat, draft_tokens, commit_lens
            ),
        )
        batch.forward_mode = ForwardMode.DECODE
        accepted = sum(accept_lens_cpu)
        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=new_id,
            num_accepted_tokens=accepted,
            accept_length_per_req_cpu=accept_lens_cpu,
            can_run_cuda_graph=result.can_run_cuda_graph,
        )


class FlashMTPWorkerV2(_FlashMTPWorkerBase):
    """Spec-v2 overlap worker; pivot tensors travel through FutureMap."""

    def _next_state(
        self,
        verified_id: torch.Tensor,
        seq_lens: torch.Tensor,
        pivot: torch.Tensor,
    ) -> FlashMTPDraftInputV2:
        batch_size = int(verified_id.shape[0])
        return FlashMTPDraftInputV2(
            topk_p=torch.ones((batch_size, 1), device=self.device, dtype=torch.float32),
            topk_index=torch.zeros((batch_size, 1), device=self.device, dtype=torch.int64),
            verified_id=verified_id.to(torch.int32),
            new_seq_lens=seq_lens.to(torch.int32),
            hidden_states=pivot,
        )

    def _record_done(self, state: FlashMTPDraftInputV2) -> None:
        event = torch.get_device_module(self.device).Event()
        event.record()
        state.verify_done = event

    def forward_batch_generation(
        self, batch: ModelWorkerBatch, **kwargs
    ) -> GenerationBatchResult:
        self._validate_request_mode(batch)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            result = self.target_worker.forward_batch_generation(batch, **kwargs)
            hidden = result.logits_output.hidden_states
            if hidden is None or batch.extend_seq_lens is None:
                raise RuntimeError("FlashMTP overlap prefill requires target hidden states.")
            pivot = self._prefill_pivot(
                hidden, batch.extend_seq_lens, batch.input_ids
            )
            state = self._next_state(result.next_token_ids, batch.seq_lens, pivot)
            self._record_done(state)
            result.next_draft_input = state
            result.logits_output.hidden_states = None
            return result

        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            state = FlashMTPDraftInputV2.create_idle_input(self.device)
            self._record_done(state)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=state,
                can_run_cuda_graph=False,
            )

        state = batch.spec_info
        if not isinstance(state, FlashMTPDraftInputV2):
            raise RuntimeError("FlashMTP overlap decode is missing FlashMTPDraftInputV2.")
        batch.seq_lens.record_stream(torch.get_device_module(self.device).current_stream())
        batch_size = len(batch.seq_lens)
        draft_tokens, _ = self._draft(state.verified_id, state.pivot_hidden)
        positions = self._global_positions(batch.seq_lens)
        end_offset = batch.seq_lens + self.block_size
        out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=end_offset,
            batch_size=batch_size,
            draft_token_num=self.block_size,
            device=self.device,
        )
        verify = FlashMTPVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions.reshape(-1),
            draft_token_num=self.block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = verify.draft_token
        batch.out_cache_loc = out_cache_loc
        batch.spec_info = verify
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        target_result = self.target_worker.forward_batch_generation(
            batch, is_verify=True, **kwargs
        )
        logits = target_result.logits_output
        if batch.sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits.next_token_logits,
                sampling_info=batch.sampling_info,
                draft_token_num=self.block_size,
            )
        target_predict = torch.argmax(logits.next_token_logits, dim=-1).view(
            batch_size, self.block_size
        )
        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=draft_tokens, target_predict=target_predict
        )
        commit_lens = accept_len.to(torch.int32) + 1
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        assert self._out_tokens is not None
        out_tokens = self._out_tokens[:batch_size]
        if self.block_size > 1:
            out_tokens[:, : self.block_size - 1].copy_(draft_tokens[:, 1:])
        out_tokens[:, self.block_size - 1].zero_()
        out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])

        hidden = logits.hidden_states
        if hidden is None:
            raise RuntimeError("FlashMTP overlap verify returned no target hidden states.")
        hidden = hidden.view(batch_size, self.block_size, -1)
        row = torch.arange(batch_size, device=self.device)
        pivot_flat = hidden[row, commit_lens.to(torch.int64) - 1]
        pivot = self._verify_pivot(pivot_flat, draft_tokens, commit_lens)
        logits.hidden_states = None
        next_state = self._next_state(
            bonus, batch.seq_lens + commit_lens.to(batch.seq_lens.dtype), pivot
        )
        self._record_done(next_state)
        return GenerationBatchResult(
            logits_output=logits,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_result.can_run_cuda_graph,
            next_draft_input=next_state,
        )
