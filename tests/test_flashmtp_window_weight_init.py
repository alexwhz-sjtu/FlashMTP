import tempfile
import unittest

import torch
from transformers import Qwen3Config

from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    LEGACY_PIVOTQ_STUDENT_ARCHITECTURE,
    FlashMTPDraftModel,
    adapt_checkpoint_state_dict,
    plan_checkpoint_weight_init,
)


def _small_config(window: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    config.num_target_layers = 4
    config.block_size = 4
    config.flashmtp_config = {
        "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
        "sliding_window_size": window,
        "chs_num_layers": 2,
        "target_layer_ids": [0, 3],
        "local_position": True,
        "include_token_embedding_chs": False,
        "pivot_query_embedding": False,
        "markov_head_type": "rnn_easy",
        "markov_output_mode": "direct",
        "markov_rank": 4,
        "backbone_conv_mode": "none",
    }
    return config


class WindowWeightInitTest(unittest.TestCase):
    def test_same_architecture_window_change_copies_every_weight(self) -> None:
        source = FlashMTPDraftModel(_small_config(4))
        target = FlashMTPDraftModel(_small_config(1))
        self.assertEqual(source.history_slot_count, 3)
        self.assertEqual(target.history_slot_count, 0)
        plan = plan_checkpoint_weight_init(
            target,
            source.config.flashmtp_config,
            load_weights_only=True,
        )
        self.assertTrue(plan["window_changed"])
        self.assertEqual(plan["source_window"], 4)
        adapted, dropped = adapt_checkpoint_state_dict(
            target,
            source.state_dict(),
            source_architecture=plan["source_architecture"],
        )
        self.assertEqual(dropped, [])
        target.load_state_dict(adapted, strict=True)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)

    def test_window_change_without_weights_only_is_rejected(self) -> None:
        source = FlashMTPDraftModel(_small_config(4))
        target = FlashMTPDraftModel(_small_config(1))
        with self.assertRaisesRegex(ValueError, "load-weights-only"):
            plan_checkpoint_weight_init(
                target,
                source.config.flashmtp_config,
                load_weights_only=False,
            )

    def test_legacy_student_drops_history_fusion_and_keeps_the_rest(self) -> None:
        source = FlashMTPDraftModel(_small_config(5))
        target = FlashMTPDraftModel(_small_config(1))
        legacy_config = {
            "architecture_version": LEGACY_PIVOTQ_STUDENT_ARCHITECTURE,
            "anchor_group_size": 6,
            "swa_window_size": 512,
            "model_role": "pivot_q_student",
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
            "markov_head_type": "rnn_easy",
            "markov_output_mode": "direct",
            "markov_rank": 4,
        }
        plan = plan_checkpoint_weight_init(
            target, legacy_config, load_weights_only=True
        )
        self.assertEqual(plan["source_window_name"], "anchor_group_size")
        self.assertEqual(plan["source_window"], 6)
        state = source.state_dict()
        state["history_fuse.weight"] = torch.zeros(16, 48)
        state["history_norm.weight"] = torch.zeros(16)
        state["unexpected.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            adapt_checkpoint_state_dict(
                target,
                state,
                source_architecture=LEGACY_PIVOTQ_STUDENT_ARCHITECTURE,
            )
        del state["unexpected.weight"]
        adapted, dropped = adapt_checkpoint_state_dict(
            target,
            state,
            source_architecture=LEGACY_PIVOTQ_STUDENT_ARCHITECTURE,
        )
        self.assertEqual(
            dropped, ["history_fuse.weight", "history_norm.weight"]
        )
        target.load_state_dict(adapted, strict=True)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)

    def test_saved_wider_window_checkpoint_reloads_at_window_one(self) -> None:
        source = FlashMTPDraftModel(_small_config(4))
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            source.save_pretrained(checkpoint_dir)
            from specforge.modeling.draft.flashmtp import (
                load_checkpoint_state_dict,
                read_checkpoint_flashmtp_config,
            )

            _, flashmtp_config = read_checkpoint_flashmtp_config(checkpoint_dir)
            target = FlashMTPDraftModel(_small_config(1))
            plan = plan_checkpoint_weight_init(
                target, flashmtp_config, load_weights_only=True
            )
            adapted, dropped = adapt_checkpoint_state_dict(
                target,
                load_checkpoint_state_dict(checkpoint_dir),
                source_architecture=plan["source_architecture"],
            )
        self.assertEqual(dropped, [])
        target.load_state_dict(adapted, strict=True)
        self.assertEqual(target.sliding_window_size, 1)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
