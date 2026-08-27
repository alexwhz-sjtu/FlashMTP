from __future__ import annotations

import importlib.metadata
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .config import is_flashmtp_config, parse_flashmtp_config

logger = logging.getLogger(__name__)

SUPPORTED_SGLANG_VERSION = "0.5.6.post2"
_INSTALLED = False
_ORIGINAL_PARSE = None
_ORIGINAL_CREATE_WORKER = None
_ORIGINAL_QWEN2_FORWARD = None
_ORIGINAL_QWEN35_FORWARD = None
_ORIGINAL_QWEN35_SET_DFLASH = None
_ORIGINAL_INIT_DISAGGREGATION = None
_ORIGINAL_NORMALIZE_REQUEST = None


def _validate_sglang_runtime() -> str:
    """Fail early when Python resolves SGLang from an incompatible env."""
    actual = importlib.metadata.version("sglang")
    if actual != SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            "FlashMTP's SGLang adapter is pinned to "
            f"sglang=={SUPPORTED_SGLANG_VERSION}; found {actual}. "
            f"Python executable: {sys.executable}."
        )

    import sglang

    module_path = Path(inspect.getfile(sglang)).resolve()
    interpreter_prefix = Path(sys.prefix).resolve()
    try:
        module_path.relative_to(interpreter_prefix)
    except ValueError as exc:
        raise RuntimeError(
            "FlashMTP imported SGLang from a different Python environment: "
            f"{module_path}. Current interpreter: {sys.executable} "
            f"(prefix {interpreter_prefix}). Unset the stale PYTHONPATH and "
            "launch with the mtp-sglang Python executable."
        ) from exc

    from sglang.srt.models.qwen3 import Qwen3ForCausalLM
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    supported_models = (Qwen3ForCausalLM, Qwen3_5ForCausalLM)
    missing = [
        model_cls for model_cls in supported_models
        if not callable(getattr(model_cls, "set_dflash_layers_to_capture", None))
    ]
    if missing:
        model_path = Path(inspect.getfile(missing[0])).resolve()
        raise RuntimeError(
            "The active SGLang Qwen implementation lacks DFlash hidden-state "
            "capture support required by FlashMTP: "
            f"{model_path}. Current interpreter: {sys.executable}. Use the "
            "patched sglang==0.5.6.post2 installation in mtp-sglang."
        )
    return actual


def _qwen35_set_dflash_layers(self, layers_to_capture):
    """Accept Qwen3-VL's +1 capture positions, including the final sentinel."""
    self.layers_to_capture = list(layers_to_capture)
    num_layers = len(self.layers)
    for layer_id in self.layers_to_capture:
        if 0 <= int(layer_id) < num_layers:
            setattr(self.layers[int(layer_id)], "_is_layer_to_capture", True)
        elif int(layer_id) != num_layers:
            raise ValueError(
                f"Invalid Qwen3.5 DFlash capture position {layer_id}; "
                f"expected [0, {num_layers}]."
            )


def _qwen35_forward_with_final_capture(self, *args, **kwargs):
    """Append Qwen3.5's normalized final hidden for target layer L-1."""
    assert _ORIGINAL_QWEN35_FORWARD is not None
    output = _ORIGINAL_QWEN35_FORWARD(self, *args, **kwargs)
    final_sentinel = int(self.config.num_hidden_layers)
    if final_sentinel not in self.layers_to_capture or not self.pp_group.is_last_rank:
        return output
    if isinstance(output, tuple):
        hidden_states, aux_hidden_states = output
        return hidden_states, [*aux_hidden_states, hidden_states]
    return output, [output]


def _parse_dflash_compatible_config(*, draft_hf_config: Any):
    """Expose FlashMTP capture metadata through SGLang's DFlash protocol."""
    if not is_flashmtp_config(draft_hf_config):
        assert _ORIGINAL_PARSE is not None
        return _ORIGINAL_PARSE(draft_hf_config=draft_hf_config)

    from sglang.srt.speculative.dflash_utils import DFlashDraftConfig

    cfg = parse_flashmtp_config(draft_hf_config)
    return DFlashDraftConfig(
        # Zero is deliberate: FlashMTP has no draft KV cache. The explicit target
        # layer ids below are sufficient for target hidden-state capture.
        num_hidden_layers=0,
        num_target_layers=cfg.num_target_layers,
        block_size=cfg.block_size,
        target_layer_ids=list(cfg.target_layer_ids),
        mask_token="<|MASK|>",
        mask_token_id=cfg.mask_token_id,
    )


def _create_flashmtp_worker(self, server_args):
    if self.is_dflash() and os.environ.get("SGLANG_FLASHMTP_ACTIVE") == "1":
        if server_args.disable_overlap_schedule:
            from .worker import FlashMTPWorker

            return FlashMTPWorker
        from .worker import FlashMTPWorkerV2

        return FlashMTPWorkerV2
    assert _ORIGINAL_CREATE_WORKER is not None
    return _ORIGINAL_CREATE_WORKER(self, server_args)


def _qwen2_forward_with_final_capture(self, *args, **kwargs):
    """Make SGLang's HF-style layer-35 capture include final normalized hidden."""
    assert _ORIGINAL_QWEN2_FORWARD is not None
    output = _ORIGINAL_QWEN2_FORWARD(self, *args, **kwargs)
    final_sentinel = int(self.config.num_hidden_layers)
    if final_sentinel not in self.layers_to_capture or not self.pp_group.is_last_rank:
        return output
    if isinstance(output, tuple):
        hidden_states, aux_hidden_states = output
        return hidden_states, [*aux_hidden_states, hidden_states]
    # This also handles checkpoints that capture only the final target layer.
    return output, [output]


def _init_disaggregation_without_draft_kv(self):
    """Tell scheduler initialization that FlashMTP has no transferable draft KV."""
    assert _ORIGINAL_INIT_DISAGGREGATION is not None
    draft_worker = self.draft_worker
    self.draft_worker = None
    try:
        return _ORIGINAL_INIT_DISAGGREGATION(self)
    finally:
        self.draft_worker = draft_worker


def _normalize_flashmtp_request(self):
    """Reject unsupported modes at HTTP admission instead of crashing a worker."""
    assert _ORIGINAL_NORMALIZE_REQUEST is not None
    result = _ORIGINAL_NORMALIZE_REQUEST(self)
    params = self.sampling_params
    params_list = params if isinstance(params, list) else [params]
    for item in params_list:
        if float(item.get("temperature", 1.0)) != 0.0:
            raise ValueError("FlashMTP requires temperature=0 (greedy decoding).")
        grammar_keys = ("json_schema", "regex", "ebnf", "structural_tag")
        if any(item.get(key) for key in grammar_keys):
            raise ValueError("FlashMTP does not support grammar decoding yet.")
    return_logprob = self.return_logprob
    if isinstance(return_logprob, list):
        has_logprob = any(bool(value) for value in return_logprob)
    else:
        has_logprob = bool(return_logprob)
    if has_logprob:
        raise ValueError("FlashMTP does not support return_logprob yet.")
    return result


def install() -> None:
    """Install the repository-local adapter in the current SGLang process."""
    global _INSTALLED, _ORIGINAL_PARSE, _ORIGINAL_CREATE_WORKER
    global _ORIGINAL_QWEN2_FORWARD
    global _ORIGINAL_QWEN35_FORWARD, _ORIGINAL_QWEN35_SET_DFLASH
    global _ORIGINAL_INIT_DISAGGREGATION
    global _ORIGINAL_NORMALIZE_REQUEST
    if _INSTALLED:
        return

    actual = _validate_sglang_runtime()

    from sglang.srt.model_executor import model_runner
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.models.qwen2 import Qwen2Model
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.speculative import dflash_utils
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from .model import FlashMTPDraftModel

    _ORIGINAL_PARSE = dflash_utils.parse_dflash_draft_config
    dflash_utils.parse_dflash_draft_config = _parse_dflash_compatible_config
    # model_runner imported the parser by value, so patch that binding as well.
    model_runner.parse_dflash_draft_config = _parse_dflash_compatible_config

    ModelRegistry.models["FlashMTPDraftModel"] = FlashMTPDraftModel
    _ORIGINAL_CREATE_WORKER = SpeculativeAlgorithm.create_worker
    SpeculativeAlgorithm.create_worker = _create_flashmtp_worker
    _ORIGINAL_QWEN2_FORWARD = Qwen2Model.forward
    Qwen2Model.forward = _qwen2_forward_with_final_capture
    _ORIGINAL_QWEN35_FORWARD = Qwen3_5ForCausalLM.forward
    _ORIGINAL_QWEN35_SET_DFLASH = Qwen3_5ForCausalLM.set_dflash_layers_to_capture
    Qwen3_5ForCausalLM.forward = _qwen35_forward_with_final_capture
    Qwen3_5ForCausalLM.set_dflash_layers_to_capture = _qwen35_set_dflash_layers
    _ORIGINAL_INIT_DISAGGREGATION = Scheduler.init_disaggregation
    Scheduler.init_disaggregation = _init_disaggregation_without_draft_kv
    _ORIGINAL_NORMALIZE_REQUEST = GenerateReqInput.normalize_batch_and_arguments
    GenerateReqInput.normalize_batch_and_arguments = _normalize_flashmtp_request
    _INSTALLED = True
    logger.info("Installed repository-local FlashMTP adapter for SGLang %s", actual)
