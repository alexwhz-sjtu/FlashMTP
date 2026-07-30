import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config

from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    prepare_target_prediction_hidden,
)
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel
from specforge.modeling.draft.flashmtp_markov_head import FlashMTPMarkovHead


class FlashMTPMarkovHeadTest(unittest.TestCase):
    def test_target_prediction_hidden_uses_causal_predecessor_positions(self) -> None:
        last_hidden = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)
        hidden_states = {
            0: torch.zeros_like(last_hidden),
            1: torch.zeros_like(last_hidden),
            2: torch.zeros_like(last_hidden),
            3: last_hidden,
        }
        anchors = torch.tensor([[1, 3], [2, 4]])

        gathered = prepare_target_prediction_hidden(
            hidden_states=hidden_states,
            anchor_positions=anchors,
            block_size=4,
            num_transformer_layers=4,
        )

        expected_positions = anchors.unsqueeze(-1) + torch.arange(3)
        expected = torch.gather(
            last_hidden.unsqueeze(1).expand(-1, 2, -1, -1),
            2,
            expected_positions.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        self.assertTrue(torch.equal(gathered, expected))

    def _assert_teacher_forcing_matches_serial(
        self, head_type: str, output_mode: str
    ) -> None:
        torch.manual_seed(7)
        batch_size, prediction_length = 2, 4
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type=head_type,
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            markov_output_mode=output_mode,
        )
        hidden = torch.randn(
            batch_size, prediction_length, hidden_size, requires_grad=True
        )
        base_logits = torch.randn(
            batch_size, prediction_length, vocab_size, requires_grad=True
        )
        first_prev = torch.tensor([2, 3])

        sampled, serial_logits = head.sample_block_tokens(
            hidden_states=hidden,
            first_prev_token_ids=first_prev,
            output_mode=output_mode,
            base_logits=base_logits if output_mode == "additive" else None,
            temperature=0.0,
        )
        teacher_prev = torch.cat([first_prev.unsqueeze(1), sampled[:, :-1]], dim=1)
        teacher_latent = head.forward_teacher_forcing(
            hidden_states=hidden,
            prev_token_ids=teacher_prev,
            output_mode=output_mode,
        )
        teacher_logits = head.project_logits(teacher_latent)
        if output_mode == "additive":
            teacher_logits = base_logits + teacher_logits

        self.assertEqual(tuple(teacher_latent.shape), (2, 4, rank))
        self.assertEqual(tuple(sampled.shape), (2, 4))
        self.assertTrue(
            torch.allclose(serial_logits, teacher_logits, atol=1e-6, rtol=1e-6)
        )

        teacher_logits.float().square().mean().backward()
        self.assertIsNotNone(head.prev_token_embedding.weight.grad)
        self.assertIsNotNone(head.output_proj.weight.grad)
        if head_type != "vanilla":
            if head_type in ("gated", "mlp") or (
                head_type in ("rnn", "rnn_easy") and output_mode == "direct"
            ):
                self.assertIsNotNone(hidden.grad)
            else:
                self.assertIsNone(hidden.grad)
        if head_type in ("gated", "mlp") or (
            head_type in ("rnn", "rnn_easy") and output_mode == "direct"
        ):
            self.assertIsNotNone(head.hidden_proj)
            if output_mode == "direct" or head_type == "mlp":
                self.assertIsNotNone(head.hidden_proj.weight.grad)
            else:
                self.assertIsNone(head.hidden_proj.weight.grad)

    def test_direct_hidden_latent_changes_gated_and_rnn_outputs(self) -> None:
        torch.manual_seed(11)
        hidden_size, vocab_size, rank = 12, 23, 5
        hidden = torch.randn(1, 3, hidden_size)
        prev_token_ids = torch.tensor([[1, 2, 3]])

        for head_type in ("gated", "rnn"):
            with self.subTest(head_type=head_type):
                head = FlashMTPMarkovHead(
                    head_type=head_type,
                    vocab_size=vocab_size,
                    markov_rank=rank,
                    hidden_size=hidden_size,
                )
                direct_latent = head.forward_teacher_forcing(
                    hidden_states=hidden,
                    prev_token_ids=prev_token_ids,
                    output_mode="direct",
                )
                additive_latent = head.forward_teacher_forcing(
                    hidden_states=hidden,
                    prev_token_ids=prev_token_ids,
                    output_mode="additive",
                )
                self.assertFalse(torch.allclose(direct_latent, additive_latent))

    def test_all_head_and_output_modes(self) -> None:
        for head_type in ("vanilla", "gated", "rnn", "rnn_easy", "mlp"):
            for output_mode in ("additive", "direct"):
                with self.subTest(head_type=head_type, output_mode=output_mode):
                    self._assert_teacher_forcing_matches_serial(head_type, output_mode)

    def test_rnn_easy_uses_state_without_state_out_proj(self) -> None:
        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=23,
            markov_rank=5,
            hidden_size=12,
            markov_output_mode="direct",
        )
        self.assertIsNone(head.state_out_proj)
        self.assertIsNone(head.hidden_fuse_gate_proj)
        self.assertIsNotNone(head.state_hidden_mlp)
        self.assertEqual(head.state_hidden_mlp.in_features, 10)
        self.assertEqual(head.state_hidden_mlp.out_features, 5)

        torch.manual_seed(9)
        hidden = torch.randn(2, 3, 12, requires_grad=True)
        prev_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        latent = head.forward_teacher_forcing(
            hidden_states=hidden,
            prev_token_ids=prev_token_ids,
            output_mode="direct",
        )
        self.assertEqual(tuple(latent.shape), (2, 3, 5))
        latent.sum().backward()
        self.assertIsNotNone(head.state_hidden_mlp.weight.grad)
        self.assertIsNotNone(head.hidden_proj.weight.grad)

    def test_rnn_easy_state_update_does_not_depend_on_hidden(self) -> None:
        torch.manual_seed(3)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
        )
        state = torch.zeros(2, rank, requires_grad=False)
        prev_token_ids = torch.tensor([4, 5])
        hidden = torch.randn(2, hidden_size, requires_grad=True)

        _, new_state = head._compute_step_latent(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
            state=state,
            output_mode="direct",
        )
        grads = torch.autograd.grad(
            new_state.sum(),
            hidden,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertTrue(
            grads is None or torch.allclose(grads, torch.zeros_like(hidden))
        )

        latent, _ = head._compute_step_latent(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
            state=state,
            output_mode="direct",
        )
        latent_grads = torch.autograd.grad(latent.sum(), hidden)[0]
        self.assertIsNotNone(latent_grads)
        self.assertFalse(torch.allclose(latent_grads, torch.zeros_like(hidden)))

    def test_mlp_state_tracks_tokens_without_hidden(self) -> None:
        torch.manual_seed(5)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="mlp",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            markov_output_mode="direct",
        )
        state = torch.zeros(2, rank, requires_grad=False)
        prev_token_ids = torch.tensor([4, 5])
        hidden = torch.randn(2, hidden_size, requires_grad=True)

        latent, new_state = head._compute_step_latent(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
            state=state,
            output_mode="direct",
        )
        state_grads = torch.autograd.grad(
            new_state.sum(),
            hidden,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertTrue(
            state_grads is None
            or torch.allclose(state_grads, torch.zeros_like(hidden))
        )
        latent_grads = torch.autograd.grad(latent.sum(), hidden)[0]
        self.assertFalse(torch.allclose(latent_grads, torch.zeros_like(hidden)))

    def test_rnn_h_output_mode_is_removed(self) -> None:
        with self.assertRaises(ValueError):
            FlashMTPMarkovHead(
                head_type="rnn",
                vocab_size=23,
                markov_rank=5,
                hidden_size=12,
                markov_output_mode="rnn_h",
            )

    def test_mlp_model_sampling_skips_base_lm_head_in_direct_mode(self) -> None:
        class FailingLMHead(nn.Module):
            def forward(self, hidden_states):
                raise AssertionError("direct mode must not call the base LM head")

        config = Qwen3Config(
            vocab_size=29,
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
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "mlp",
            "markov_output_mode": "direct",
            "markov_rank": 5,
        }
        model = FlashMTPDraftModel(config)
        sampled, logits = model.sample_draft_tokens(
            draft_hidden=torch.randn(2, 3, 16),
            lm_head=FailingLMHead(),
            first_prev_token_ids=torch.tensor([1, 2]),
        )
        self.assertEqual(tuple(sampled.shape), (2, 3))
        self.assertEqual(tuple(logits.shape), (2, 3, 29))

    def test_rnn_state_update_does_not_depend_on_hidden(self) -> None:
        torch.manual_seed(3)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="rnn",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
        )
        state = torch.zeros(2, rank, requires_grad=False)
        prev_token_ids = torch.tensor([4, 5])
        hidden = torch.randn(2, hidden_size, requires_grad=True)

        _, new_state = head._compute_step_latent(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
            state=state,
            output_mode="direct",
        )
        grads = torch.autograd.grad(
            new_state.sum(),
            hidden,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertTrue(
            grads is None or torch.allclose(grads, torch.zeros_like(hidden))
        )

        latent, _ = head._compute_step_latent(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
            state=state,
            output_mode="direct",
        )
        latent_grads = torch.autograd.grad(latent.sum(), hidden)[0]
        self.assertIsNotNone(latent_grads)
        self.assertFalse(torch.allclose(latent_grads, torch.zeros_like(hidden)))

    def test_model_config_round_trip(self) -> None:
        config = Qwen3Config(
            vocab_size=29,
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
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "mlp",
            "markov_output_mode": "direct",
            "markov_rank": 7,
        }
        model = FlashMTPDraftModel(config)
        self.assertEqual(model.markov_head_type, "mlp")
        self.assertEqual(model.markov_output_mode, "direct")
        self.assertEqual(model.markov_rank, 7)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            loaded = FlashMTPDraftModel.from_pretrained(checkpoint_dir)
        self.assertEqual(loaded.markov_head_type, "mlp")
        self.assertEqual(loaded.markov_output_mode, "direct")
        self.assertEqual(loaded.markov_rank, 7)
        self.assertIsNotNone(loaded.markov_head)

    def test_direct_model_sampling_skips_base_lm_head(self) -> None:
        class FailingLMHead(nn.Module):
            def forward(self, hidden_states):
                raise AssertionError("direct mode must not call the base LM head")

        config = Qwen3Config(
            vocab_size=29,
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
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "gated",
            "markov_output_mode": "direct",
            "markov_rank": 5,
        }
        model = FlashMTPDraftModel(config)
        sampled, logits = model.sample_draft_tokens(
            draft_hidden=torch.randn(2, 3, 16),
            lm_head=FailingLMHead(),
            first_prev_token_ids=torch.tensor([1, 2]),
        )
        self.assertEqual(tuple(sampled.shape), (2, 3))
        self.assertEqual(tuple(logits.shape), (2, 3, 29))

    def test_chunked_teacher_forcing_loss(self) -> None:
        config = Qwen3Config(
            vocab_size=31,
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
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "mlp",
            "markov_output_mode": "direct",
            "markov_rank": 6,
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            num_anchors=2,
            ce_chunk_size=3,
            final_ce_weight=0.3,
            tv_loss_weight=0.7,
        )
        prediction_hidden = torch.randn(2, 2, 3, 16, requires_grad=True)
        target_prediction_hidden = torch.randn(
            2, 2, 3, 16, requires_grad=True
        )
        prev_token_ids = torch.randint(0, 31, (2, 2, 3))
        labels = torch.randint(0, 31, (2, 2, 3))
        weight_mask = torch.tensor(
            [
                [[1.0, 0.5, 0.25], [1.0, 0.5, 0.0]],
                [[1.0, 0.5, 0.25], [1.0, 0.0, 0.0]],
            ]
        )
        binary_eval_mask = weight_mask > 0
        block_keep_mask = torch.ones(2, 2, dtype=torch.bool)

        loss, accuracy, prefix_acc, final_ce_loss, base_ce_loss, tv_loss = (
            wrapper._chunked_weighted_ce_and_metrics(
                prediction_hidden=prediction_hidden,
                prev_token_ids=prev_token_ids,
                labels=labels,
                weight_mask=weight_mask,
                binary_eval_mask=binary_eval_mask,
                block_keep_mask=block_keep_mask,
                target_prediction_hidden=target_prediction_hidden,
            )
        )
        markov_latent = draft_model.markov_head.forward_teacher_forcing(
            hidden_states=prediction_hidden,
            prev_token_ids=prev_token_ids,
            output_mode="direct",
        )
        draft_logits = draft_model.markov_head.project_logits(markov_latent)
        target_logits = wrapper.lm_head(target_prediction_hidden)
        manual_tv = (
            (
                F.softmax(draft_logits, dim=-1)
                - F.softmax(target_logits, dim=-1)
            )
            .abs()
            .sum(dim=-1)
            .mul(weight_mask)
            .sum()
            / (binary_eval_mask.sum() + 1e-6)
        )
        manual_ce = (
            F.cross_entropy(
                draft_logits.reshape(-1, draft_logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            )
            .view_as(labels)
            .mul(weight_mask)
            .sum()
            / (weight_mask.sum() + 1e-6)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(final_ce_loss, manual_ce))
        self.assertEqual(float(base_ce_loss), 0.0)
        self.assertTrue(torch.allclose(tv_loss, manual_tv))
        self.assertTrue(
            torch.allclose(loss, 0.3 * manual_ce + 0.7 * manual_tv)
        )
        self.assertTrue(0.0 <= float(accuracy) <= 1.0)
        self.assertTrue(1.0 <= float(prefix_acc) <= 4.0)
        loss.backward()
        self.assertIsNotNone(prediction_hidden.grad)
        self.assertIsNotNone(target_prediction_hidden.grad)
        self.assertIsNotNone(draft_model.markov_head.output_proj.weight.grad)
        self.assertIsNotNone(draft_model.markov_head.hidden_proj.weight.grad)
        self.assertIsNotNone(draft_model.markov_head.mlp_state_proj.weight.grad)

    def test_base_lm_ce_auxiliary_loss(self) -> None:
        config = Qwen3Config(
            vocab_size=31,
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
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "rnn",
            "markov_output_mode": "direct",
            "markov_rank": 6,
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            num_anchors=2,
            ce_chunk_size=3,
            base_lm_ce_weight=0.5,
            base_lm_ce_decay_gamma=2.0,
            tv_loss_weight=0.0,
        )
        prediction_hidden = torch.randn(2, 2, 3, 16, requires_grad=True)
        prev_token_ids = torch.randint(0, 31, (2, 2, 3))
        labels = torch.randint(0, 31, (2, 2, 3))
        weight_mask = torch.ones(2, 2, 3)
        base_weight_mask = torch.tensor(
            [[[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]], [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]]]
        )
        binary_eval_mask = torch.ones(2, 2, 3, dtype=torch.bool)
        block_keep_mask = torch.ones(2, 2, dtype=torch.bool)

        loss, _, _, final_ce_loss, base_ce_loss, tv_loss = (
            wrapper._chunked_weighted_ce_and_metrics(
                prediction_hidden=prediction_hidden,
                prev_token_ids=prev_token_ids,
                labels=labels,
                weight_mask=weight_mask,
                binary_eval_mask=binary_eval_mask,
                block_keep_mask=block_keep_mask,
                base_weight_mask=base_weight_mask,
            )
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(final_ce_loss))
        self.assertTrue(torch.isfinite(base_ce_loss))
        self.assertGreater(float(base_ce_loss), 0.0)
        self.assertEqual(float(tv_loss), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction_hidden.grad)


if __name__ == "__main__":
    unittest.main()
