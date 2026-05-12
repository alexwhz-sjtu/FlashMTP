from .base import Eagle3DraftModel
from .dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)
from .flashmtp import (
    FlashMTPDraftModel,
    build_target_layer_ids as build_flashmtp_target_layer_ids,
    extract_context_features_at_positions,
)
from .llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "Eagle3DraftModel",
    "DFlashDraftModel",
    "FlashMTPDraftModel",
    "LlamaForCausalLMEagle3",
    "build_target_layer_ids",
    "build_flashmtp_target_layer_ids",
    "extract_context_feature",
    "extract_context_features_at_positions",
    "sample",
]
