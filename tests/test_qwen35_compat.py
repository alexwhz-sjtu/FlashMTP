import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from specforge.modeling.config_utils import load_text_model_config
from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    FlashMTPDraftModel,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


def _write_qwen35_config(path: Path, *, vocab_size: int = 32, hidden_size: int = 16):
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": hidden_size // 4,
            "hidden_act": "silu",
            "max_position_embeddings": 4096,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "eos_token_id": vocab_size - 1,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1_000_000,
                "partial_rotary_factor": 0.25,
            },
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


class Qwen35CompatibilityTest(unittest.TestCase):
    def test_composite_config_becomes_dense_qwen3_draft_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            _write_qwen35_config(model_path)

            config = load_text_model_config(str(model_path))

        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.flashmtp_source_model_type, "qwen3_5")
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.vocab_size, 32)
        self.assertEqual(config.rope_theta, 1_000_000)
        self.assertEqual(config.layer_types, ["full_attention"] * 4)

    def test_converted_config_constructs_flashmtp_draft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            _write_qwen35_config(model_path)
            config = load_text_model_config(str(model_path))

        config.num_hidden_layers = 1
        config.num_target_layers = 4
        config.block_size = 4
        config.layer_types = ["full_attention"]
        config.flashmtp_config = {
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "markov_head_type": "none",
            "markov_output_mode": "additive",
        }
        model = FlashMTPDraftModel(config)

        self.assertEqual(model.config.hidden_size, 16)
        self.assertEqual(model.target_layer_ids, [0, 3])
        self.assertEqual(len(model.layers), 1)

    def test_qwen35_embedding_key_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            _write_qwen35_config(model_path, vocab_size=12, hidden_size=8)
            expected = torch.arange(96, dtype=torch.float32).reshape(12, 8)
            filename = "model.safetensors"
            save_file(
                {"model.language_model.embed_tokens.weight": expected},
                str(model_path / filename),
            )
            index = {
                "metadata": {},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": filename,
                },
            }
            (model_path / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            components = TargetEmbeddingsAndHead.from_pretrained(
                str(model_path), device="cpu", dtype=torch.float32
            )

        torch.testing.assert_close(components.embed_tokens.weight, expected)
        self.assertIs(components.lm_head.weight, components.embed_tokens.weight)


if __name__ == "__main__":
    unittest.main()
