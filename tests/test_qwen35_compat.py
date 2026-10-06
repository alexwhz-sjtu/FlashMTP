import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.dlite_training import build_draft_config
from specforge.modeling.config_utils import load_text_model_config
from specforge.modeling.draft.dlite import DLiteDraftModel


def _write_qwen35_config(path: Path) -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 63,
            "rope_parameters": {"rope_theta": 1_000_000},
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


class Qwen35CompatibilityTest(unittest.TestCase):
    def test_qwen35_config_becomes_dense_qwen3_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_qwen35_config(Path(tmpdir))
            config = load_text_model_config(tmpdir)

        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.dlite_source_model_type, "qwen3_5")
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.layer_types, ["full_attention"] * 32)

    def test_explicit_target_layers_override_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_qwen35_config(Path(tmpdir))
            args = Namespace(
                target_model_path=tmpdir,
                trust_remote_code=False,
                num_draft_layers=2,
                block_size=4,
                target_layer_ids="0,1,3,7,11,15,19,23,27,29,30,31",
                chs_num_layers=7,
                dlite_version="dlite_v2",
                sequential_head="rnn",
                sequential_rank=8,
                mask_token_id=63,
                swa_window_size=8,
            )
            config = build_draft_config(args, model_role="swa_teacher")
            model = DLiteDraftModel(config)

        self.assertEqual(args.chs_num_layers, 12)
        self.assertEqual(
            model.target_layer_ids,
            [0, 1, 3, 7, 11, 15, 19, 23, 27, 29, 30, 31],
        )
        self.assertEqual(model.chs_num_layers, 12)


if __name__ == "__main__":
    unittest.main()
