"""FlashMTP Target Model for online hidden states generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_device_mesh, get_tp_group


@dataclass
class FlashMTPTargetOutput:
    """Output format for FlashMTP training."""
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class FlashMTPTargetModel(ABC):
    """Abstract base class for FlashMTP target model backend."""

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
        pass

    @abstractmethod
    def generate_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> FlashMTPTargetOutput:
        pass


class HFFlashMTPTargetModel(FlashMTPTargetModel):
    """HuggingFace backend for FlashMTP target model."""

    def __init__(self, model: nn.Module):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "HFFlashMTPTargetModel":
        tp_size = get_tp_group().size()

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {"device_map": device}

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **device_kwargs,
            **kwargs,
        )
        model.eval()
        return cls(model)

    @torch.no_grad()
    def generate_hidden_states(
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
        hidden_states = outputs.hidden_states
        return FlashMTPTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


def get_flashmtp_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "hf",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> FlashMTPTargetModel:
    if backend == "hf":
        return HFFlashMTPTargetModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")