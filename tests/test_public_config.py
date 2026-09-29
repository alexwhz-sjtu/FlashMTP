import unittest

try:
    import torch
    from torch import nn
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"Transformers runtime is unavailable: {exc}") from exc

from specforge.modeling.draft.dlite import DLiteDraftModel, build_target_layer_ids
from specforge.core.dlite import OnlineDLiteModel


class PublicConfigTest(unittest.TestCase):
    def test_student_uses_minimal_dlite_config(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        config.num_target_layers = 4
        config.block_size = 4
        config.dlite_config = {
            "architecture_version": "dlite_v1",
            "model_role": "pivot_q_student",
            "chs_num_layers": 2,
            "target_layer_ids": build_target_layer_ids(4, 2),
            "sequential_head": "rnn",
            "sequential_rank": 8,
            "mask_token_id": 31,
        }
        model = DLiteDraftModel(config)
        self.assertTrue(model.is_student)
        self.assertEqual(model.sequential_head_type, "rnn")
        self.assertEqual(model.draft_query_length, config.block_size)
        self.assertNotIn("anchor_group_size", config.dlite_config)
        self.assertNotIn("sequential_output_mode", config.dlite_config)
        self.assertNotIn("mask_embedding_mode", config.dlite_config)

        embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        online = OnlineDLiteModel(
            draft_model=model,
            target_lm_head=nn.Linear(config.hidden_size, config.vocab_size),
            target_embed_tokens=embed_tokens,
            mask_token_id=31,
            block_size=config.block_size,
        )
        input_ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
        hidden_states = {
            layer_id: torch.randn(1, input_ids.size(1), config.hidden_size)
            for layer_id in model.target_layer_ids
        }
        prepared = online.prepare_batch(
            input_ids,
            hidden_states,
            torch.ones_like(input_ids),
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.tensor([[True]]),
        )
        self.assertEqual(prepared.query_embeddings.shape, (1, 1, 4, 16))
        torch.testing.assert_close(
            prepared.query_embeddings[0, 0, 0], embed_tokens(input_ids[0, 2])
        )
        torch.testing.assert_close(
            prepared.query_embeddings[0, 0, 1:],
            embed_tokens(torch.tensor([31, 31, 31])),
        )
        self.assertEqual(prepared.token_position_ids.tolist(), [[[2]]])
        self.assertFalse(hasattr(prepared, "initial_prev_token_ids"))


if __name__ == "__main__":
    unittest.main()
