import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config

from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    add_noise_to_target_hidden,
    prepare_target_hidden,
    prepare_target_prediction_hidden,
)
from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
    rejection_sample_verify,
)
from specforge.modeling.draft.flashmtp_markov_head import FlashMTPMarkovHead


class FlashMTPMarkovHeadTest(unittest.TestCase):
    def test_training_tensor_prep_does_not_lookup_fsdp_sharded_embedding(self) -> None:
        class ShardedEmbedding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(31 * 16), requires_grad=False)

            def forward(self, input_ids):
                raise AssertionError("embedding lookup must happen inside FSDP forward")

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
            "pivot_fuse_mode": "prefix_condition",
            "include_embedding_chs": True,
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=ShardedEmbedding(),
            mask_token_id=30,
            block_size=4,
            num_anchors=1,
            tv_loss_weight=0.0,
        )
        hidden_states = {
            0: torch.randn(1, 8, 16),
            3: torch.randn(1, 8, 16),
        }
        with mock.patch.object(
            wrapper,
            "_sample_anchor_positions",
            return_value=(torch.tensor([[2]]), torch.tensor([[True]])),
        ):
            _, _, target_hidden, _ = wrapper.prepare_training_tensors(
                input_ids=torch.arange(8).view(1, 8),
                hidden_states=hidden_states,
                loss_mask=torch.ones(1, 8),
            )

        # Only configured CHS layers are prepared outside FSDP. The fixed
        # embedding slot is added later, after FSDP unshards the table.
        self.assertEqual(tuple(target_hidden.shape), (1, 1, 2, 16))

    def test_embedding_chs_slot_is_not_perturbed_by_hidden_noise(self) -> None:
        target_hidden = torch.zeros(2, 3, 4, 5)
        noisy = add_noise_to_target_hidden(
            target_hidden, noise_ratio=0.1, preserve_first_slot=True
        )
        self.assertTrue(torch.equal(noisy[:, :, 0], target_hidden[:, :, 0]))
        self.assertFalse(torch.equal(noisy[:, :, 1:], target_hidden[:, :, 1:]))

    def test_training_chs_prepends_anchor_predecessor_embedding(self) -> None:
        embeddings = torch.arange(1 * 6 * 3, dtype=torch.float32).view(1, 6, 3)
        layer0 = embeddings + 100
        layer2 = embeddings + 300
        anchors = torch.tensor([[2, 5]])

        gathered = prepare_target_hidden(
            hidden_states={0: layer0, 2: layer2},
            anchor_positions=anchors,
            target_layer_ids=[0, 2],
            num_transformer_layers=3,
            input_embeddings=embeddings,
            include_embedding_chs=True,
        )

        self.assertEqual(tuple(gathered.shape), (1, 2, 3, 3))
        self.assertTrue(torch.equal(gathered[:, :, 0], embeddings[:, [1, 4]]))
        self.assertTrue(torch.equal(gathered[:, :, 1], layer0[:, [1, 4]]))
        self.assertTrue(torch.equal(gathered[:, :, 2], layer2[:, [1, 4]]))

    def test_inference_chs_prepends_embedding_without_counting_it_as_layer(self) -> None:
        embedding = torch.full((1, 2, 3), 7.0)
        layer0 = torch.full((1, 2, 3), 11.0)
        layer1 = torch.full((1, 2, 3), 13.0)

        gathered = gather_pivot_multilayer_inference(
            hidden_states=(embedding, layer0, layer1),
            target_layer_ids=[0, 1],
            token_index=-1,
            num_transformer_layers=2,
            include_embedding_chs=True,
        )

        self.assertEqual(tuple(gathered.shape), (1, 1, 3, 3))
        self.assertTrue(torch.equal(gathered[0, 0, 0], embedding[0, -1]))
        self.assertTrue(torch.equal(gathered[0, 0, 1], layer0[0, -1]))
        self.assertTrue(torch.equal(gathered[0, 0, 2], layer1[0, -1]))

    def test_prefix_embedding_slot_has_no_depth_encoding(self) -> None:
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
            "pivot_fuse_mode": "prefix_condition",
            "include_embedding_chs": True,
        }
        model = FlashMTPDraftModel(config)
        target_hidden = torch.randn(1, 1, 3, 16)
        with torch.no_grad():
            model.layer_depth_embedding.weight.fill_(2.0)
            expected_embedding = model.hidden_norm(target_hidden[:, :, 0]).squeeze(1)
            expected_layer = model.hidden_norm(
                target_hidden[:, :, 1] + 2.0
            ).squeeze(1)
            fused = model._fuse_target_hidden(target_hidden)

        self.assertEqual(model.target_layer_ids, [0, 3])
        self.assertEqual(model.chs_len_per_block, 3)
        self.assertTrue(torch.allclose(fused[:, 0], expected_embedding))
        self.assertTrue(torch.allclose(fused[:, 1], expected_layer))

    def test_embedding_prefix_uses_same_rotary_position_as_chs(self) -> None:
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
            "pivot_fuse_mode": "prefix_condition",
            "include_embedding_chs": True,
            "left_shift": True,
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            num_anchors=1,
            tv_loss_weight=0.0,
        )
        output_hidden = torch.zeros(1, 3, 16)
        fake_result = tuple(torch.zeros(()) for _ in range(6))
        expected_predecessor_embedding = (
            wrapper.embed_tokens.weight[1].detach().clone()
        )
        with (
            mock.patch.object(
                draft_model, "forward", return_value=output_hidden
            ) as draft_forward,
            mock.patch(
                "specforge.core.flashmtp.create_flashmtp_block_mask",
                return_value=None,
            ) as mask_mock,
            mock.patch.object(
                wrapper,
                "_chunked_weighted_ce_and_metrics",
                return_value=fake_result,
            ),
        ):
            wrapper(
                input_ids=torch.arange(8).view(1, 8),
                loss_mask=torch.ones(1, 8),
                anchor_positions=torch.tensor([[2]]),
                block_keep_mask=torch.tensor([[True]]),
                target_hidden=torch.zeros(1, 1, 2, 16),
            )

        rotary_ids = draft_forward.call_args.kwargs["rotary_position_ids"]
        self.assertTrue(torch.equal(rotary_ids, torch.tensor([[1, 1, 1, 2, 3, 4]])))
        forwarded_chs = draft_forward.call_args.kwargs["target_hidden"]
        self.assertEqual(tuple(forwarded_chs.shape), (1, 1, 3, 16))
        self.assertTrue(
            torch.equal(forwarded_chs[0, 0, 0], expected_predecessor_embedding)
        )
        self.assertEqual(mask_mock.call_args.kwargs["chs_len_per_block"], 3)

    def test_rejection_sampling_accepts_identical_distributions(self) -> None:
        draft_logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]])
        target_logits = torch.cat(
            [draft_logits, torch.tensor([[[0.0, 0.0, 3.0]]])],
            dim=1,
        )
        proposed_tokens = torch.tensor([[0, 1]])

        accepted, bonus = rejection_sample_verify(
            proposed_tokens=proposed_tokens,
            draft_logits=draft_logits,
            target_logits=target_logits,
            temperature=1.0,
        )

        self.assertEqual(accepted, 2)
        self.assertEqual(tuple(bonus.shape), (1,))

    def test_rejection_sampling_uses_residual_on_rejection(self) -> None:
        draft_logits = torch.tensor([[[100.0, -100.0, -100.0]]])
        target_logits = torch.tensor([[[-100.0, 100.0, -100.0], [0.0, 0.0, 100.0]]])

        accepted, correction = rejection_sample_verify(
            proposed_tokens=torch.tensor([[0]]),
            draft_logits=draft_logits,
            target_logits=target_logits,
            temperature=1.0,
        )

        self.assertEqual(accepted, 0)
        self.assertEqual(correction.item(), 1)

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

        left_shifted = prepare_target_prediction_hidden(
            hidden_states=hidden_states,
            anchor_positions=anchors,
            block_size=4,
            num_transformer_layers=4,
            left_shift=True,
        )
        left_shifted_positions = anchors.unsqueeze(-1) + torch.arange(3)
        left_shifted_expected = torch.gather(
            last_hidden.unsqueeze(1).expand(-1, 2, -1, -1),
            2,
            left_shifted_positions.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        self.assertTrue(torch.equal(left_shifted, left_shifted_expected))

    def test_left_shift_target_prediction_hidden_uses_total_span(self) -> None:
        last_hidden = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)
        hidden_states = {3: last_hidden}
        anchors = torch.tensor([[1, 3], [2, 4]])
        gathered = prepare_target_prediction_hidden(
            hidden_states=hidden_states,
            anchor_positions=anchors,
            block_size=4,
            num_transformer_layers=4,
            left_shift=True,
        )
        self.assertEqual(tuple(gathered.shape), (2, 2, 3, 3))

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
            if head_type in ("rnn", "rnn_easy") and output_mode == "direct":
                self.assertIsNotNone(hidden.grad)
            else:
                self.assertIsNone(hidden.grad)
        if head_type in ("rnn", "rnn_easy") and output_mode == "direct":
            self.assertIsNotNone(head.hidden_proj)
            self.assertIsNotNone(head.hidden_proj.weight.grad)

    def test_direct_hidden_latent_changes_rnn_outputs(self) -> None:
        torch.manual_seed(11)
        hidden_size, vocab_size, rank = 12, 23, 5
        hidden = torch.randn(1, 3, hidden_size)
        prev_token_ids = torch.tensor([[1, 2, 3]])

        for head_type in ("rnn",):
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
        for head_type in ("vanilla", "rnn", "rnn_easy"):
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

    def test_rnn_h_output_mode_is_removed(self) -> None:
        with self.assertRaises(ValueError):
            FlashMTPMarkovHead(
                head_type="rnn",
                vocab_size=23,
                markov_rank=5,
                hidden_size=12,
                markov_output_mode="rnn_h",
            )

    def test_rnn_easy_model_sampling_skips_base_lm_head_in_direct_mode(self) -> None:
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
            "markov_head_type": "rnn_easy",
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
            "markov_head_type": "rnn_easy",
            "markov_output_mode": "direct",
            "markov_rank": 7,
            "left_shift": True,
        }
        model = FlashMTPDraftModel(config)
        self.assertEqual(model.markov_head_type, "rnn_easy")
        self.assertEqual(model.markov_output_mode, "direct")
        self.assertEqual(model.markov_rank, 7)
        self.assertTrue(model.left_shift)
        self.assertEqual(model.proposal_length, 3)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            loaded = FlashMTPDraftModel.from_pretrained(checkpoint_dir)
        self.assertEqual(loaded.markov_head_type, "rnn_easy")
        self.assertEqual(loaded.markov_output_mode, "direct")
        self.assertEqual(loaded.markov_rank, 7)
        self.assertTrue(loaded.left_shift)
        self.assertEqual(loaded.proposal_length, 3)
        self.assertIsNotNone(loaded.markov_head)
        self.assertFalse(loaded.include_embedding_chs)
        self.assertEqual(loaded.fc.in_features, 2 * config.hidden_size)

    def test_prediction_hidden_legacy_skips_slot_zero(self) -> None:
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
        config.block_size = 8
        config.flashmtp_config = {
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "none",
            "markov_output_mode": "additive",
            "left_shift": False,
        }
        legacy_model = FlashMTPDraftModel(config)
        block_hidden = torch.randn(2, 8, 16)
        legacy_hidden = legacy_model._prediction_hidden(block_hidden)
        self.assertEqual(legacy_hidden.shape, (2, 7, 16))

        config.flashmtp_config["left_shift"] = True
        left_shift_model = FlashMTPDraftModel(config)
        left_shift_hidden = left_shift_model._prediction_hidden(block_hidden)
        self.assertEqual(left_shift_hidden.shape, (2, 7, 16))

    def test_legacy_left_shift_defaults_false_without_config_key(self) -> None:
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
        config.block_size = 8
        config.flashmtp_config = {
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "none",
            "markov_output_mode": "additive",
        }
        legacy_model = FlashMTPDraftModel(config)
        self.assertFalse(legacy_model.left_shift)
        self.assertEqual(legacy_model.draft_block_len, 8)
        self.assertEqual(legacy_model.proposal_length, 7)
        self.assertEqual(legacy_model.max_verify_block_size, 8)

    def test_left_shift_decode_block_sizes(self) -> None:
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
        config.block_size = 8
        config.flashmtp_config = {
            "target_layer_ids": [0, 3],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "none",
            "markov_output_mode": "additive",
            "left_shift": True,
        }
        model = FlashMTPDraftModel(config)
        self.assertTrue(model.left_shift)
        self.assertEqual(model.draft_block_len, 7)
        self.assertEqual(model.proposal_length, 7)
        self.assertEqual(model.max_verify_block_size, 8)

    def test_left_shift_training_alignment(self) -> None:
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
            "markov_head_type": "none",
            "markov_output_mode": "additive",
            "left_shift": True,
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            num_anchors=1,
            tv_loss_weight=0.0,
        )
        output_hidden = torch.arange(3 * 16, dtype=torch.float32).view(1, 3, 16)
        fake_result = tuple(torch.zeros(()) for _ in range(6))
        with (
            mock.patch.object(draft_model, "forward", return_value=output_hidden),
            mock.patch(
                "specforge.core.flashmtp.create_flashmtp_block_mask",
                return_value=None,
            ),
            mock.patch.object(
                wrapper,
                "_chunked_weighted_ce_and_metrics",
                return_value=fake_result,
            ) as loss_mock,
        ):
            wrapper(
                input_ids=torch.arange(8).view(1, 8),
                loss_mask=torch.ones(1, 8),
                anchor_positions=torch.tensor([[1]]),
                block_keep_mask=torch.tensor([[True]]),
                target_hidden=torch.zeros(1, 1, 2, 16),
            )

        call = loss_mock.call_args.kwargs
        self.assertTrue(
            torch.equal(call["prediction_hidden"], output_hidden.view(1, 1, 3, 16))
        )
        self.assertTrue(
            torch.equal(call["prev_token_ids"], torch.tensor([[[1, 2, 3]]]))
        )
        self.assertTrue(torch.equal(call["labels"], torch.tensor([[[2, 3, 4]]])))
        self.assertTrue(torch.equal(call["weight_mask"], torch.ones(1, 1, 3)))

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
            "markov_head_type": "rnn",
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

    def test_compiled_serial_sampler_is_cached(self) -> None:
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
            "markov_head_type": "rnn",
            "markov_output_mode": "direct",
            "markov_rank": 5,
        }
        model = FlashMTPDraftModel(config)
        hidden = torch.randn(1, 3, 16)
        previous = torch.tensor([1])

        with mock.patch(
            "torch.compile", side_effect=lambda fn, **kwargs: fn
        ) as compile_mock:
            first = model.sample_draft_tokens(
                draft_hidden=hidden,
                lm_head=FailingLMHead(),
                first_prev_token_ids=previous,
                compile_serial_head=True,
            )
            second = model.sample_draft_tokens(
                draft_hidden=hidden,
                lm_head=FailingLMHead(),
                first_prev_token_ids=previous,
                compile_serial_head=True,
            )

        compile_mock.assert_called_once()
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

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
            "markov_head_type": "rnn_easy",
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
        target_prediction_hidden = torch.randn(2, 2, 3, 16, requires_grad=True)
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
            F.softmax(draft_logits, dim=-1) - F.softmax(target_logits, dim=-1)
        ).abs().sum(dim=-1).mul(weight_mask).sum() / (weight_mask.sum() + 1e-6)
        manual_ce = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels).mul(weight_mask).sum() / (weight_mask.sum() + 1e-6)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(final_ce_loss, manual_ce))
        self.assertEqual(float(base_ce_loss), 0.0)
        self.assertTrue(torch.allclose(tv_loss, manual_tv))
        self.assertTrue(torch.allclose(loss, 0.3 * manual_ce + 0.7 * manual_tv))
        self.assertTrue(0.0 <= float(accuracy) <= 1.0)
        self.assertTrue(1.0 <= float(prefix_acc) <= 4.0)
        loss.backward()
        self.assertIsNotNone(prediction_hidden.grad)
        self.assertIsNotNone(target_prediction_hidden.grad)
        self.assertIsNotNone(draft_model.markov_head.output_proj.weight.grad)
        self.assertIsNotNone(draft_model.markov_head.hidden_proj.weight.grad)
        self.assertIsNotNone(draft_model.markov_head.state_proj.weight.grad)

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

    def test_masked_nan_block_does_not_contaminate_loss_or_gradients(self) -> None:
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
            "markov_head_type": "none",
            "markov_output_mode": "additive",
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            num_anchors=2,
            tv_loss_weight=0.0,
        )
        prediction_hidden = torch.randn(1, 2, 3, 16, requires_grad=True)
        with torch.no_grad():
            prediction_hidden[:, 1].fill_(float("nan"))
        labels = torch.randint(0, 31, (1, 2, 3))
        weight_mask = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
        binary_eval_mask = weight_mask > 0

        loss, *_ = wrapper._chunked_weighted_ce_and_metrics(
            prediction_hidden=prediction_hidden,
            prev_token_ids=torch.zeros_like(labels),
            labels=labels,
            weight_mask=weight_mask,
            binary_eval_mask=binary_eval_mask,
            block_keep_mask=torch.tensor([[True, False]]),
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction_hidden.grad).all())
        self.assertTrue(torch.isfinite(wrapper.lm_head.weight.grad).all())
        self.assertTrue(
            torch.equal(
                prediction_hidden.grad[:, 1],
                torch.zeros_like(prediction_hidden.grad[:, 1]),
            )
        )

    def test_illegal_supervised_label_is_rejected(self) -> None:
        config = Qwen3Config(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
        )
        config.num_target_layers = 2
        config.block_size = 2
        config.flashmtp_config = {
            "target_layer_ids": [0, 1],
            "pivot_fuse_mode": "linear_fuse",
            "markov_head_type": "none",
            "markov_output_mode": "additive",
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(8, 11, bias=False),
            target_embed_tokens=nn.Embedding(11, 8),
            mask_token_id=10,
            block_size=2,
            tv_loss_weight=0.0,
        )

        with self.assertRaisesRegex(ValueError, "within the output vocabulary"):
            wrapper._chunked_weighted_ce_and_metrics(
                prediction_hidden=torch.randn(1, 1, 1, 8),
                prev_token_ids=torch.zeros(1, 1, 1, dtype=torch.long),
                labels=torch.tensor([[[11]]]),
                weight_mask=torch.ones(1, 1, 1),
                binary_eval_mask=torch.ones(1, 1, 1, dtype=torch.bool),
                block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
