import unittest
from types import SimpleNamespace

import torch

from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    create_flashmtp_block_mask,
    create_flashmtp_shared_context_mask,
    prepare_shared_target_hidden,
    prepare_target_hidden,
)


class SharedTargetHiddenTest(unittest.TestCase):
    def setUp(self):
        self.batch = 2
        self.seq_len = 5
        self.hidden_size = 3
        self.num_layers = 4
        self.layer_ids = [0, 3]
        self.layers = tuple(
            torch.full(
                (self.batch, self.seq_len, self.hidden_size),
                float(layer_id),
            )
            + torch.arange(self.seq_len).view(1, -1, 1) * 10
            for layer_id in range(self.num_layers)
        )

    def _assert_position_major(self, actual):
        self.assertEqual(
            actual.shape,
            (self.batch, self.seq_len, len(self.layer_ids), self.hidden_size),
        )
        for position in range(self.seq_len):
            for slot, layer_id in enumerate(self.layer_ids):
                torch.testing.assert_close(
                    actual[:, position, slot],
                    self.layers[layer_id][:, position],
                )

    def test_tuple_without_embedding(self):
        actual = prepare_shared_target_hidden(
            self.layers, self.layer_ids, self.num_layers
        )
        self._assert_position_major(actual)

    def test_tuple_with_embedding_prefix(self):
        embedding = torch.full_like(self.layers[0], -1.0)
        actual = prepare_shared_target_hidden(
            (embedding,) + self.layers,
            self.layer_ids,
            self.num_layers,
        )
        self._assert_position_major(actual)

    def test_layer_dict(self):
        actual = prepare_shared_target_hidden(
            {layer_id: self.layers[layer_id] for layer_id in self.layer_ids},
            self.layer_ids,
            self.num_layers,
        )
        self._assert_position_major(actual)


class SharedContextCompatibilityTest(unittest.TestCase):
    class _Draft(torch.nn.Module):
        def __init__(self, pivot_fuse_mode, context_window_size):
            super().__init__()
            self.pivot_fuse_mode = pivot_fuse_mode
            self.context_window_size = context_window_size
            self.target_layer_ids = [0, 3]
            self.config = SimpleNamespace(num_target_layers=4)

    @staticmethod
    def _wrapper(pivot_fuse_mode, context_window_size, add_noise=False):
        return OnlineFlashMTPModel(
            draft_model=SharedContextCompatibilityTest._Draft(
                pivot_fuse_mode, context_window_size
            ),
            target_lm_head=torch.nn.Linear(4, 8, bias=False),
            target_embed_tokens=torch.nn.Embedding(8, 4),
            mask_token_id=7,
            add_noise=add_noise,
        )

    def test_shared_path_is_limited_to_swa_without_independent_noise(self):
        self.assertTrue(
            self._wrapper("prefix_condition", 3).uses_shared_training_context
        )
        self.assertFalse(
            self._wrapper("prefix_condition", 1).uses_shared_training_context
        )
        self.assertFalse(
            self._wrapper(
                "prefix_condition", 3, add_noise=True
            ).uses_shared_training_context
        )
        self.assertFalse(self._wrapper("linear_fuse", 1).uses_shared_training_context)


@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention test requires CUDA")
class SharedContextFlexAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")

    @staticmethod
    def _mask_value(mask, batch, query, key):
        device = mask.kv_num_blocks.device
        return bool(
            mask.mask_mod(
                torch.tensor(batch, device=device),
                torch.tensor(0, device=device),
                torch.tensor(query, device=device),
                torch.tensor(key, device=device),
            ).item()
        )

    def test_shared_mask_preserves_visible_source_keys(self):
        anchors = torch.tensor([[1, 5, 0]], device=self.device)
        keep = torch.tensor([[True, True, False]], device=self.device)
        seq_len, num_layers, window, block_size = 8, 2, 3, 2
        n_blocks = anchors.shape[1]
        legacy_chs = window * num_layers

        offsets = torch.arange(1 - window, 1, device=self.device).view(1, 1, window)
        context_valid = (
            (anchors.unsqueeze(-1) - 1 + offsets >= 0)
            .unsqueeze(-1)
            .expand(-1, -1, -1, num_layers)
        )
        context_valid = context_valid.reshape(1, n_blocks, legacy_chs)

        legacy_mask = create_flashmtp_block_mask(
            anchors,
            keep,
            legacy_chs,
            block_size,
            self.device,
            context_valid,
        )
        shared_mask = create_flashmtp_shared_context_mask(
            anchors,
            keep,
            seq_len,
            num_layers,
            window,
            block_size,
            self.device,
        )

        legacy_context_len = n_blocks * legacy_chs
        shared_context_len = seq_len * num_layers
        q_len = n_blocks * block_size
        for query in range(q_len):
            legacy_visible = set()
            for key in range(legacy_context_len + q_len):
                if not self._mask_value(legacy_mask, 0, query, key):
                    continue
                if key < legacy_context_len:
                    context_block = key // legacy_chs
                    slot = key % legacy_chs
                    source_position = (
                        int(anchors[0, context_block]) - window + slot // num_layers
                    )
                    legacy_visible.add(("context", source_position, slot % num_layers))
                else:
                    draft_slot = key - legacy_context_len
                    legacy_visible.add(
                        ("draft", draft_slot // block_size, draft_slot % block_size)
                    )

            shared_visible = set()
            for key in range(shared_context_len + q_len):
                if not self._mask_value(shared_mask, 0, query, key):
                    continue
                if key < shared_context_len:
                    shared_visible.add(("context", key // num_layers, key % num_layers))
                else:
                    draft_slot = key - shared_context_len
                    shared_visible.add(
                        ("draft", draft_slot // block_size, draft_slot % block_size)
                    )
            self.assertEqual(legacy_visible, shared_visible)

    def test_tiny_draft_forward_matches_legacy_layout(self):
        from transformers import Qwen3Config

        from specforge.modeling.draft.flashmtp import FlashMTPDraftModel

        torch.manual_seed(0)
        batch, seq_len, hidden_size = 1, 8, 64
        num_target_layers, num_context_layers = 4, 2
        window, block_size = 3, 2
        layer_ids = [0, 3]

        config = Qwen3Config(
            vocab_size=128,
            hidden_size=hidden_size,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
        config.num_target_layers = num_target_layers
        config.block_size = block_size
        config.flashmtp_config = {
            "pivot_fuse_mode": "prefix_condition",
            "context_window_size": window,
            "target_layer_ids": layer_ids,
            "num_middle_layers_n": 0,
        }
        config._attn_implementation = "flex_attention"
        model = FlashMTPDraftModel(config).to(self.device).eval()

        hidden_states = tuple(
            torch.randn(batch, seq_len, hidden_size, device=self.device)
            for _ in range(num_target_layers)
        )
        anchors = torch.tensor([[2, 6]], device=self.device)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        shared_hidden = prepare_shared_target_hidden(
            hidden_states, layer_ids, num_target_layers
        )
        legacy_hidden = prepare_target_hidden(
            hidden_states, anchors, layer_ids, num_target_layers, window
        )

        noise = torch.randn(
            batch, anchors.shape[1] * block_size, hidden_size, device=self.device
        )
        draft_positions = (
            anchors.unsqueeze(-1)
            + torch.arange(block_size, device=self.device).view(1, 1, -1)
        ).reshape(batch, -1)
        shared_context_positions = (
            torch.arange(seq_len, device=self.device)
            .view(1, seq_len, 1)
            .expand(batch, -1, num_context_layers)
            .reshape(batch, -1)
        )
        legacy_context_positions = model.context_position_ids(anchors - 1)

        shared_mask = create_flashmtp_shared_context_mask(
            anchors,
            keep,
            seq_len,
            num_context_layers,
            window,
            block_size,
            self.device,
        )
        offsets = torch.arange(1 - window, 1, device=self.device).view(1, 1, window)
        context_valid = (
            (anchors.unsqueeze(-1) - 1 + offsets >= 0)
            .unsqueeze(-1)
            .expand(-1, -1, -1, num_context_layers)
        )
        context_valid = context_valid.reshape(
            batch, anchors.shape[1], window * num_context_layers
        )
        legacy_mask = create_flashmtp_block_mask(
            anchors,
            keep,
            window * num_context_layers,
            block_size,
            self.device,
            context_valid,
        )

        with torch.no_grad():
            shared_output = model(
                position_ids=draft_positions,
                noise_embedding=noise,
                target_hidden=shared_hidden,
                attention_mask=shared_mask,
                rotary_position_ids=torch.cat(
                    [shared_context_positions, draft_positions], dim=-1
                ),
            )
            legacy_output = model(
                position_ids=draft_positions,
                noise_embedding=noise,
                target_hidden=legacy_hidden,
                attention_mask=legacy_mask,
                rotary_position_ids=torch.cat(
                    [legacy_context_positions, draft_positions], dim=-1
                ),
            )
        torch.testing.assert_close(shared_output, legacy_output, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
