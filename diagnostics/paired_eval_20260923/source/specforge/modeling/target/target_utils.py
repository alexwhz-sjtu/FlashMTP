import gc
import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open

from specforge.modeling.config_utils import is_qwen35_model_type, load_text_model_config

DEFAULT_EMBED_KEYS = (
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
)
DEFAULT_LM_HEAD_KEYS = (
    "lm_head.weight",
    "model.language_model.lm_head.weight",
    "language_model.lm_head.weight",
)


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Handles safetensors slicing and Weight Tying correctly.
    """

    def __init__(self, config, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.config = config
        factory_kwargs = {"dtype": dtype} if dtype is not None else {}

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            **factory_kwargs,
        )

        tie_weights = getattr(config, "tie_word_embeddings", False)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device="meta" if tie_weights else None,
            dtype=dtype,
        )
        if tie_weights:
            # Avoid allocating a second full-vocabulary matrix while loading
            # tied checkpoints such as Qwen3.5.
            self.lm_head.weight = self.embed_tokens.weight

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config
        config = load_text_model_config(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        instance = cls(config, dtype=dtype)

        # 2. Resolve Model Path
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
                )
            except Exception as e:
                print(f"Warning: Snapshot download failed or path check failed: {e}")

        # 3. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 4. Load Weights
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)

        # 5. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self,
        model_path: str,
        embed_key: Optional[str],
        lm_head_key: Optional[str],
        tie_weights: bool,
    ):
        index_files = sorted(glob.glob(os.path.join(model_path, "*.index.json")))
        weight_map = {}
        files_to_load = {}
        target_file = None
        available_keys: set[str] = set()

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            available_keys = set(weight_map)
        else:
            safetensors = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
            bins = sorted(glob.glob(os.path.join(model_path, "*.bin")))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")
            if target_file.endswith(".safetensors"):
                with safe_open(target_file, framework="pt") as f:
                    available_keys = set(f.keys())

        source_model_type = getattr(self.config, "flashmtp_source_model_type", None)
        embed_candidates = list(DEFAULT_EMBED_KEYS)
        if is_qwen35_model_type(source_model_type):
            embed_candidates.insert(0, "model.language_model.embed_tokens.weight")
        embed_key = self._resolve_weight_key(
            requested=embed_key,
            candidates=embed_candidates,
            available_keys=available_keys,
            kind="embedding",
        )
        if not tie_weights:
            lm_head_key = self._resolve_weight_key(
                requested=lm_head_key,
                candidates=list(DEFAULT_LM_HEAD_KEYS),
                available_keys=available_keys,
                kind="LM head",
            )

        if index_files:
            files_to_load[embed_key] = weight_map[embed_key]
            if not tie_weights and lm_head_key is not None:
                files_to_load[lm_head_key] = weight_map[lm_head_key]
        else:
            filename = os.path.basename(target_file)
            files_to_load[embed_key] = filename
            if not tie_weights and lm_head_key is not None:
                files_to_load[lm_head_key] = filename

        loaded_keys = set()

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            loaded_keys.update(
                self._load_file_content(file_path, keys, embed_key, lm_head_key)
            )

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        if embed_key not in loaded_keys:
            raise RuntimeError("Failed to load embeddings.")
        if not tie_weights and lm_head_key not in loaded_keys:
            print(
                "Warning: LM Head weights were not found (and tie_weights is False). Head is random."
            )

    @staticmethod
    def _resolve_weight_key(
        *,
        requested: Optional[str],
        candidates: list[str],
        available_keys: set[str],
        kind: str,
    ) -> str:
        if requested is not None:
            if available_keys and requested not in available_keys:
                raise ValueError(f"{kind} key '{requested}' not found in checkpoint.")
            return requested
        if available_keys:
            for candidate in candidates:
                if candidate in available_keys:
                    return candidate
            raise ValueError(
                f"Could not find a supported {kind} key in checkpoint. Tried: "
                + ", ".join(candidates)
            )
        # Legacy .bin checkpoints cannot be inspected cheaply. Preserve the
        # architecture-specific first candidate and validate it after loading.
        return candidates[0]

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: Optional[str],
    ) -> set[str]:
        """Helper to load specific keys from a file"""
        print(f"Loading {keys_to_extract} from {os.path.basename(file_path)}...")

        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            print(
                f"Warning: Loading .bin file {os.path.basename(file_path)} into RAM. Convert to safetensors for efficiency."
            )
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
                print(" -> Loaded Embeddings")
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)
                    print(" -> Loaded LM Head")
                else:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected {self.lm_head.weight.shape}, got {tensor.shape}"
                    )
        return set(state_dict_part)
