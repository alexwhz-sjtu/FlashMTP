from .base import Eagle3DraftModel
from .dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)
from .flashmtp import (
    FlashMTPDraftModel,
    build_ablation_target_layer_ids,
    build_target_layer_ids as build_flashmtp_target_layer_ids,
    gather_pivot_multilayer_inference,
)
from .llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "Eagle3DraftModel",
    "DFlashDraftModel",
    "FlashMTPDraftModel",
    "LlamaForCausalLMEagle3",
    "build_target_layer_ids",
    "build_flashmtp_target_layer_ids",
    "build_ablation_target_layer_ids",
    "gather_pivot_multilayer_inference",
    "extract_context_feature",
    "sample",
]
