import tempfile
import unittest
import warnings
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config

from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    create_flashmtp_block_mask,
    gather_sliding_history,
    prepare_target_prediction_logits,
)
from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    FlashMTPDraftModel,
    build_target_layer_ids,
    rejection_sample_verify,
)
from specforge.modeling.draft.flashmtp_markov_head import (
    FlashMTPMarkovHead,
    migrate_legacy_rnn_easy_direct_state_dict,
)


class FlashMTPMarkovHeadTest(unittest.TestCase):
    def test_sliding_layer_selection_and_architecture_validation(self) -> None:
        self.assertEqual(build_target_layer_ids(8, 5), [0, 1, 3, 6, 7])
        with self.assertRaises(ValueError):
            build_target_layer_ids(4, 1)

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
        config.flashmtp_config = {"target_layer_ids": [0, 3]}
        with self.assertRaisesRegex(ValueError, "architecture_version"):
            FlashMTPDraftModel(config)

    def test_pivot_q_is_the_only_history_layout(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
        }
        model = FlashMTPDraftModel(config)
        self.assertEqual(model.history_source_lookback, 3)
        self.assertEqual(model.window_query_count, 3)
        self.assertEqual(model.unsupervised_query_count, 4)
        self.assertEqual(model.core_draft_query_length, 4)
        self.assertEqual(model.draft_query_length, 7)
        self.assertEqual(model.chs_len_per_block, 2)
        self.assertFalse(hasattr(model, "history_fuse"))
        self.assertNotIn("history_mode", model.config.flashmtp_config)

        config.flashmtp_config["history_mode"] = "token"
        with self.assertRaisesRegex(ValueError, "only supports pivot-Q"):
            FlashMTPDraftModel(config)

    def test_gather_sliding_history_left_pads_short_windows(self) -> None:
        fused = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
        anchors = torch.tensor([[1, 3, 5]])
        history, keep, positions = gather_sliding_history(fused, anchors, 4)

        self.assertEqual(tuple(history.shape), (1, 3, 3, 1))
        self.assertTrue(
            torch.equal(
                keep,
                torch.tensor(
                    [[[False, False, False], [False, True, True], [True, True, True]]]
                ),
            )
        )
        self.assertTrue(torch.equal(history[0, 1, :, 0], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(positions[0, 2], torch.tensor([1, 2, 3])))

        empty, empty_keep, empty_pos = gather_sliding_history(fused, anchors, 1)
        self.assertEqual(tuple(empty.shape), (1, 3, 0, 1))
        self.assertEqual(empty_keep.numel(), 0)
        self.assertEqual(empty_pos.numel(), 0)

        token_history, token_keep, token_positions = gather_sliding_history(
            fused, torch.tensor([[3]]), 4, include_pivot=True
        )
        self.assertTrue(
            torch.equal(
                token_history[0, 0, :, 0], torch.tensor([0.0, 1.0, 2.0])
            )
        )
        self.assertTrue(token_keep.all())
        self.assertTrue(torch.equal(token_positions, torch.tensor([[[0, 1, 2]]])))

    def test_inference_chs_does_not_duplicate_token_embedding(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 1,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
        }
        model = FlashMTPDraftModel(config)
        embed = nn.Embedding(29, 16)
        with torch.no_grad():
            embed.weight.copy_(
                torch.arange(29, dtype=torch.float32).view(-1, 1).expand(-1, 16)
            )
        draft_ids = torch.tensor([[5, 28, 28, 28]])
        pivot_ids = torch.tensor([[3]])
        noise = model.build_inference_query_embeddings(
            embed, draft_ids, window_embeddings=torch.empty(1, 0, 16)
        )
        target_hidden = torch.randn(1, 1, 2, 16)
        current_chs = model.build_inference_current_chs(
            embed, target_hidden, pivot_ids
        )
        self.assertEqual(tuple(noise.shape), (1, 4, 16))
        self.assertTrue(torch.equal(noise[0, 0], embed.weight[5]))
        self.assertEqual(tuple(current_chs.shape), (1, 1, 2, 16))
        self.assertTrue(torch.equal(current_chs, target_hidden))
        self.assertEqual(model.unsupervised_query_count, 1)
        self.assertEqual(model.draft_query_length, 4)

    def test_inference_condition_uses_token_embeddings(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
        }
        model = FlashMTPDraftModel(config)
        embeddings = (
            torch.arange(5, dtype=torch.float32)
            .view(1, 5, 1)
            .expand(-1, -1, 16)
        )
        condition = model.initialize_inference_condition(
            token_embeddings=embeddings
        )
        self.assertTrue(
            torch.equal(condition[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))
        )

        new_embeddings = (
            torch.arange(10, 12, dtype=torch.float32)
            .view(1, 2, 1)
            .expand(-1, -1, 16)
        )
        condition = model.update_inference_condition(
            condition,
            pivot_index=0,
            token_embeddings=new_embeddings,
        )
        self.assertTrue(
            torch.equal(condition[0, :, 0], torch.tensor([3.0, 4.0, 10.0]))
        )

    def test_training_history_uses_embedding_table(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
            "local_position": True,
        }
        draft_model = FlashMTPDraftModel(config)
        embed_tokens = nn.Embedding(29, 16)
        with torch.no_grad():
            embed_tokens.weight.copy_(
                torch.arange(29, dtype=torch.float32).view(-1, 1).expand(-1, 16)
            )
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 29, bias=False),
            target_embed_tokens=embed_tokens,
            mask_token_id=28,
            block_size=4,
            tv_loss_weight=0.0,
        )
        input_ids = torch.arange(8).view(1, 8)
        loss_mask = torch.zeros(1, 8)
        loss_mask[:, 4:] = 1

        packed, starts, lengths = wrapper._prepare_history_sources(
            input_ids, loss_mask
        )

        self.assertTrue(torch.equal(starts, torch.tensor([1])))
        self.assertTrue(torch.equal(lengths, torch.tensor([7])))
        self.assertEqual(tuple(packed.shape), (1, 7, 16))
        self.assertTrue(torch.equal(packed[0, :, 0], torch.arange(1.0, 8.0)))

    def test_pivot_q_moves_window_embeddings_onto_queries(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
        }
        model = FlashMTPDraftModel(config)
        token_history = torch.arange(3, dtype=torch.float32).view(1, 3, 1).expand(
            -1, -1, 16
        ).clone()
        current = torch.randn(1, 1, 2, 16)
        context_positions, draft_positions = model.build_inference_context(
            token_history, current, anchor_position=3
        )

        self.assertTrue(torch.equal(context_positions, torch.tensor([[2, 2]])))
        self.assertTrue(
            torch.equal(draft_positions, torch.tensor([[0, 1, 2, 3, 4, 5, 6]]))
        )

        embed = nn.Embedding(29, 16)
        with torch.no_grad():
            embed.weight.copy_(
                torch.arange(29, dtype=torch.float32).view(-1, 1).expand(-1, 16)
            )
        queries = model.build_inference_query_embeddings(
            embed,
            torch.tensor([[5, 28, 28, 28]]),
            window_embeddings=token_history,
        )
        self.assertEqual(tuple(queries.shape), (1, 7, 16))
        self.assertTrue(torch.equal(queries[0, :3], token_history[0]))
        self.assertTrue(torch.equal(queries[0, 3], embed.weight[5]))
        self.assertTrue(torch.equal(queries[0, 4], embed.weight[28]))
        self.assertEqual(tuple(model._prediction_hidden(queries).shape), (1, 3, 16))
        self.assertTrue(torch.equal(model._prediction_hidden(queries)[0, 0], queries[0, 4]))
        output = model(
            position_ids=draft_positions,
            rotary_position_ids=torch.cat(
                [context_positions, draft_positions], dim=-1
            ),
            noise_embedding=queries,
            target_hidden=current,
            is_causal=False,
        )
        self.assertEqual(tuple(output.shape), (1, 7, 16))

        model.set_local_position(True)
        local_context_positions, local_draft_positions = (
            model.build_inference_context(
                token_history, current, anchor_position=10
            )
        )
        self.assertTrue(torch.equal(local_context_positions, torch.tensor([[2, 2]])))
        self.assertTrue(
            torch.equal(local_draft_positions, torch.tensor([[0, 1, 2, 3, 4, 5, 6]]))
        )

    def test_pivot_q_training_puts_window_on_query_not_kv(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
            "local_position": True,
        }
        draft_model = FlashMTPDraftModel(config)
        embed_tokens = nn.Embedding(29, 16)
        with torch.no_grad():
            embed_tokens.weight.copy_(
                torch.arange(29, dtype=torch.float32).view(-1, 1).expand(-1, 16)
            )
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(16, 29, bias=False),
            target_embed_tokens=embed_tokens,
            mask_token_id=28,
            block_size=4,
            num_anchors=1,
            tv_loss_weight=0.0,
        )
        packed, starts, lengths = wrapper._prepare_history_sources(
            torch.arange(8).view(1, 8), torch.ones(1, 8)
        )
        self.assertTrue(torch.equal(starts, torch.tensor([0])))
        self.assertEqual(tuple(packed.shape), (1, 8, 16))

        output_hidden = torch.arange(7 * 16, dtype=torch.float32).view(1, 7, 16)
        fake_result = tuple(torch.zeros(()) for _ in range(6))
        with (
            mock.patch.object(
                draft_model, "forward", return_value=output_hidden
            ) as draft_forward_mock,
            mock.patch(
                "specforge.core.flashmtp.create_flashmtp_block_mask",
                return_value=None,
            ) as mask_mock,
            mock.patch.object(
                wrapper,
                "_chunked_weighted_ce_and_metrics",
                return_value=fake_result,
            ) as loss_mock,
        ):
            wrapper(
                input_ids=torch.arange(8).view(1, 8),
                loss_mask=torch.ones(1, 8),
                anchor_positions=torch.tensor([[3]]),
                block_keep_mask=torch.tensor([[True]]),
                target_hidden=torch.zeros(1, 1, 2, 16),
                history_hidden_states=packed,
                history_start_positions=starts,
                history_source_lengths=lengths,
            )

        draft_call = draft_forward_mock.call_args.kwargs
        mask_kwargs = mask_mock.call_args.kwargs
        call = loss_mock.call_args.kwargs
        self.assertEqual(mask_kwargs["block_size"], 7)
        self.assertEqual(mask_kwargs["chs_len_per_block"], 2)
        self.assertEqual(tuple(mask_kwargs["context_keep_mask"].shape), (1, 1, 2))
        self.assertIsNotNone(mask_kwargs["draft_keep_mask"])
        self.assertTrue(mask_kwargs["draft_keep_mask"].all())
        noise = draft_call["noise_embedding"]
        self.assertEqual(tuple(noise.shape), (1, 7, 16))
        self.assertTrue(torch.equal(noise[0, 0], embed_tokens.weight[0]))
        self.assertTrue(torch.equal(noise[0, 1], embed_tokens.weight[1]))
        self.assertTrue(torch.equal(noise[0, 2], embed_tokens.weight[2]))
        self.assertTrue(torch.equal(noise[0, 3], embed_tokens.weight[3]))
        self.assertTrue(torch.equal(noise[0, 4], embed_tokens.weight[28]))
        self.assertTrue(
            torch.equal(
                draft_call["rotary_position_ids"],
                torch.tensor([[2, 2, 0, 1, 2, 3, 4, 5, 6]]),
            )
        )
        self.assertTrue(
            torch.equal(
                call["prediction_hidden"], output_hidden[:, 4:].view(1, 1, 3, 16)
            )
        )

    def test_pivot_q_masks_padded_window_queries_as_kv(self) -> None:
        with mock.patch(
            "specforge.core.flashmtp.compile_friendly_create_block_mask",
            side_effect=lambda mask_mod, **_: mask_mod,
        ):
            draft_keep = torch.tensor(
                [[[False, False, True, True, True, True, True]]]
            )
            packed_mod = create_flashmtp_block_mask(
                anchor_positions=torch.tensor([[3]]),
                block_keep_mask=torch.tensor([[True]]),
                context_keep_mask=torch.ones(1, 1, 2, dtype=torch.bool),
                chs_len_per_block=2,
                block_size=7,
                device=torch.device("cpu"),
                draft_keep_mask=draft_keep,
            )

        def visible(q_idx: int, kv_idx: int) -> bool:
            return bool(
                packed_mod(
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(q_idx),
                    torch.tensor(kv_idx),
                )
            )

        # KV: [CHS (2) | window+draft Q (7)]
        self.assertTrue(visible(6, 0))
        self.assertTrue(visible(6, 1))
        self.assertFalse(visible(6, 2))
        self.assertFalse(visible(6, 3))
        self.assertTrue(visible(6, 4))
        for draft_kv in range(5, 9):
            self.assertTrue(visible(6, draft_kv))

    def test_training_masks_keep_each_draft_block_bidirectional(self) -> None:
        with mock.patch(
            "specforge.core.flashmtp.compile_friendly_create_block_mask",
            side_effect=lambda mask_mod, **_: mask_mod,
        ):
            packed_mod = create_flashmtp_block_mask(
                anchor_positions=torch.tensor([[5, 20]]),
                block_keep_mask=torch.tensor([[True, True]]),
                context_keep_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                chs_len_per_block=2,
                block_size=5,
                device=torch.device("cpu"),
            )

        def packed_visible(q_idx: int, kv_idx: int) -> bool:
            return bool(
                packed_mod(
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(q_idx),
                    torch.tensor(kv_idx),
                )
            )

        # Packed KV: [CHS_0 (2) | CHS_1 (2) | Block_0 (5) | Block_1 (5)]
        for q_idx in (0, 1, 4):
            self.assertTrue(packed_visible(q_idx, 0))
            self.assertTrue(packed_visible(q_idx, 1))
            self.assertFalse(packed_visible(q_idx, 2))
            self.assertFalse(packed_visible(q_idx, 3))
            for draft_kv in range(4, 9):
                self.assertTrue(packed_visible(q_idx, draft_kv))
            self.assertFalse(packed_visible(q_idx, 9))

    def test_local_block_positions_skip_left_padding_and_support_w1(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
            "local_position": True,
        }
        model = FlashMTPDraftModel(config)
        anchors = torch.tensor([[1, 3]])
        history_global = torch.tensor([[[0, 0, 0], [0, 1, 2]]])
        history_keep = torch.tensor(
            [[[False, False, False], [False, True, True]]]
        )
        context_pos, draft_pos = model.build_block_position_ids(
            anchors, history_global, history_keep, draft_length=4
        )
        self.assertTrue(
            torch.equal(
                context_pos,
                torch.tensor([[0, 0, 1, 1]]),
            )
        )
        self.assertTrue(
            torch.equal(
                draft_pos,
                torch.tensor([[0, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 5]]),
            )
        )

        config.flashmtp_config["sliding_window_size"] = 1
        w1_model = FlashMTPDraftModel(config)
        empty_pos = torch.empty(1, 1, 0, dtype=torch.long)
        empty_keep = torch.empty(1, 1, 0, dtype=torch.bool)
        context_pos, draft_pos = w1_model.build_block_position_ids(
            torch.tensor([[9]]), empty_pos, empty_keep, draft_length=4
        )
        self.assertTrue(torch.equal(context_pos, torch.tensor([[0, 0]])))
        self.assertTrue(torch.equal(draft_pos, torch.tensor([[1, 2, 3, 4]])))

    def test_rejection_sampling_accepts_identical_distributions(self) -> None:
        draft_logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]
        )
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
        target_logits = torch.tensor(
            [[[-100.0, 100.0, -100.0], [0.0, 0.0, 100.0]]]
        )

        accepted, correction = rejection_sample_verify(
            proposed_tokens=torch.tensor([[0]]),
            draft_logits=draft_logits,
            target_logits=target_logits,
            temperature=1.0,
        )

        self.assertEqual(accepted, 0)
        self.assertEqual(correction.item(), 1)

    def test_target_prediction_logits_use_causal_predecessor_positions(self) -> None:
        prefill_logits = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)
        anchors = torch.tensor([[1, 3], [2, 4]])

        gathered = prepare_target_prediction_logits(
            target_logits=prefill_logits,
            anchor_positions=anchors,
            block_size=4,
        )

        expected_positions = anchors.unsqueeze(-1) + torch.arange(3)
        expected = torch.gather(
            prefill_logits.unsqueeze(1).expand(-1, 2, -1, -1),
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
            max_prediction_length=prediction_length,
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
        if head_type == "vanilla":
            self.assertIsNone(hidden.grad)
        elif head_type == "gated":
            self.assertIsNotNone(hidden.grad)
            assert head.gate_proj is not None
            self.assertIsNotNone(head.gate_proj.weight.grad)
        elif head_type in ("rnn", "rnn_easy") and output_mode == "direct":
            self.assertIsNotNone(hidden.grad)
        else:
            self.assertIsNone(hidden.grad)
        if head_type == "rnn" and output_mode == "direct":
            self.assertIsNotNone(head.hidden_proj)
            self.assertIsNotNone(head.hidden_proj.weight.grad)
        if head_type == "rnn_easy" and output_mode == "direct":
            self.assertIsNone(head.hidden_proj)
            assert head.state_hidden_mlp is not None
            self.assertIsNotNone(head.state_hidden_mlp.weight.grad)

    def test_direct_hidden_latent_changes_rnn_outputs(self) -> None:
        torch.manual_seed(11)
        hidden_size, vocab_size, rank = 12, 23, 5
        hidden = torch.randn(1, 3, hidden_size)
        prev_token_ids = torch.tensor([[1, 2, 3]])

        for head_type in ("rnn", "rnn_easy"):
            with self.subTest(head_type=head_type):
                head = FlashMTPMarkovHead(
                    head_type=head_type,
                    vocab_size=vocab_size,
                    markov_rank=rank,
                    hidden_size=hidden_size,
                    max_prediction_length=3,
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
        for head_type in ("vanilla", "gated", "rnn", "rnn_easy"):
            output_modes = ("additive",) if head_type == "gated" else ("additive", "direct")
            for output_mode in output_modes:
                with self.subTest(head_type=head_type, output_mode=output_mode):
                    self._assert_teacher_forcing_matches_serial(head_type, output_mode)

    def test_gated_rejects_direct_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "additive"):
            FlashMTPMarkovHead(
                head_type="gated",
                vocab_size=23,
                markov_rank=5,
                hidden_size=12,
                max_prediction_length=4,
                markov_output_mode="direct",
            )
        head = FlashMTPMarkovHead(
            head_type="gated",
            vocab_size=23,
            markov_rank=5,
            hidden_size=12,
            max_prediction_length=4,
            markov_output_mode="additive",
        )
        with self.assertRaisesRegex(ValueError, "additive"):
            head.sample_block_tokens(
                hidden_states=torch.randn(1, 2, 12),
                first_prev_token_ids=torch.tensor([1]),
                output_mode="direct",
                base_logits=None,
            )

    def test_vanilla_additive_head_is_position_agnostic(self) -> None:
        head = FlashMTPMarkovHead(
            head_type="vanilla",
            vocab_size=23,
            markov_rank=5,
            hidden_size=12,
            max_prediction_length=3,
            markov_output_mode="additive",
        )
        latent = head.forward_teacher_forcing(
            hidden_states=torch.randn(2, 3, 12),
            prev_token_ids=torch.full((2, 3), 4),
            output_mode="additive",
        )
        self.assertTrue(torch.equal(latent[:, :1, :].expand_as(latent), latent))

    def test_rnn_easy_uses_state_without_state_out_proj(self) -> None:
        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=23,
            markov_rank=5,
            hidden_size=12,
            max_prediction_length=3,
            markov_output_mode="direct",
        )
        self.assertIsNone(head.state_out_proj)
        self.assertIsNone(head.hidden_fuse_gate_proj)
        self.assertIsNone(head.hidden_proj)
        self.assertIsNotNone(head.state_hidden_mlp)
        self.assertEqual(head.state_hidden_mlp.in_features, 17)
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
        self.assertIsNotNone(hidden.grad)

    def test_rnn_easy_state_update_does_not_depend_on_hidden(self) -> None:
        torch.manual_seed(3)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            max_prediction_length=3,
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

    def test_migrate_legacy_rnn_easy_direct_is_numerically_exact(self) -> None:
        torch.manual_seed(21)
        hidden_size, vocab_size, rank = 12, 23, 5
        hidden = torch.randn(2, hidden_size)
        prev_token_ids = torch.tensor([4, 5])
        state = torch.zeros(2, rank)

        legacy_hidden_proj = nn.Linear(hidden_size, rank, bias=False)
        legacy_state_hidden_mlp = nn.Linear(2 * rank, rank)
        legacy_state_proj = nn.Linear(2 * rank, 2 * rank)
        prev_embeddings = nn.Embedding(vocab_size, rank)

        mem_inputs = torch.cat([state, prev_embeddings(prev_token_ids)], dim=-1)
        gate_raw, candidate_raw = legacy_state_proj(mem_inputs).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        hidden_latent = legacy_hidden_proj(hidden)
        legacy_latent = legacy_state_hidden_mlp(
            torch.cat([new_state, hidden_latent], dim=-1)
        )

        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            max_prediction_length=3,
            markov_output_mode="direct",
        )
        head.state_proj.load_state_dict(legacy_state_proj.state_dict())
        head.prev_token_embedding.load_state_dict(prev_embeddings.state_dict())
        legacy_state = {
            "hidden_proj.weight": legacy_hidden_proj.weight.detach().clone(),
            "state_hidden_mlp.weight": legacy_state_hidden_mlp.weight.detach().clone(),
            "state_hidden_mlp.bias": legacy_state_hidden_mlp.bias.detach().clone(),
        }
        migrated = migrate_legacy_rnn_easy_direct_state_dict(
            legacy_state,
            markov_rank=rank,
            hidden_size=hidden_size,
        )
        self.assertTrue(migrated)
        self.assertNotIn("hidden_proj.weight", legacy_state)
        head.state_hidden_mlp.load_state_dict(
            {
                "weight": legacy_state["state_hidden_mlp.weight"],
                "bias": legacy_state["state_hidden_mlp.bias"],
            }
        )
        migrated_latent, _ = head._compute_step_latent(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
            state=state,
            output_mode="direct",
        )
        self.assertTrue(torch.allclose(legacy_latent, migrated_latent, atol=1e-6, rtol=1e-6))

    def test_rnn_easy_load_state_dict_migrates_legacy_weights(self) -> None:
        torch.manual_seed(22)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            max_prediction_length=3,
            markov_output_mode="direct",
        )
        legacy_state = {
            "hidden_proj.weight": torch.randn(rank, hidden_size),
            "state_hidden_mlp.weight": torch.randn(rank, 2 * rank),
            "state_hidden_mlp.bias": torch.randn(rank),
            "state_proj.weight": torch.randn(2 * rank, 2 * rank),
            "state_proj.bias": torch.randn(2 * rank),
            "prev_token_embedding.weight": torch.randn(vocab_size, rank),
            "output_proj.weight": torch.randn(vocab_size, rank),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            incompatible = head.load_state_dict(legacy_state, strict=False)
        self.assertTrue(any("Migrated legacy rnn_easy" in str(w.message) for w in caught))
        self.assertNotIn("hidden_proj.weight", incompatible.unexpected_keys)
        self.assertEqual(
            tuple(head.state_hidden_mlp.weight.shape),
            (rank, rank + hidden_size),
        )

    def test_rnn_h_output_mode_is_removed(self) -> None:
        with self.assertRaises(ValueError):
            FlashMTPMarkovHead(
                head_type="rnn",
                vocab_size=23,
                markov_rank=5,
                hidden_size=12,
                max_prediction_length=3,
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
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

    def test_rnn_seeds_state_from_anchor_predecessor_when_window_gt_one(self) -> None:
        torch.manual_seed(11)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="rnn_easy",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            max_prediction_length=3,
        )
        hidden_states = torch.randn(2, 3, hidden_size)
        prev_token_ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
        initial_prev = torch.tensor([1, 2])

        zero_latent = head.forward_teacher_forcing(
            hidden_states=hidden_states,
            prev_token_ids=prev_token_ids,
            output_mode="direct",
        )
        seeded_latent = head.forward_teacher_forcing(
            hidden_states=hidden_states,
            prev_token_ids=prev_token_ids,
            output_mode="direct",
            initial_prev_token_ids=initial_prev,
        )
        self.assertFalse(torch.allclose(zero_latent, seeded_latent))

        zero_sampled, zero_logits = head.sample_block_tokens(
            hidden_states=hidden_states,
            first_prev_token_ids=prev_token_ids[:, 0],
            output_mode="direct",
        )
        seeded_sampled, seeded_logits = head.sample_block_tokens(
            hidden_states=hidden_states,
            first_prev_token_ids=prev_token_ids[:, 0],
            output_mode="direct",
            initial_prev_token_ids=initial_prev,
        )
        self.assertFalse(torch.allclose(zero_logits, seeded_logits))

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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "target_layer_ids": [0, 3],
            "markov_head_type": "rnn_easy",
            "markov_output_mode": "direct",
            "markov_rank": 7,
        }
        model = FlashMTPDraftModel(config)
        self.assertTrue(model.seed_rnn_from_predecessor)

    def test_rnn_state_update_does_not_depend_on_hidden(self) -> None:
        torch.manual_seed(3)
        hidden_size, vocab_size, rank = 12, 23, 5
        head = FlashMTPMarkovHead(
            head_type="rnn",
            vocab_size=vocab_size,
            markov_rank=rank,
            hidden_size=hidden_size,
            max_prediction_length=3,
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "local_position": True,
            "markov_head_type": "rnn_easy",
            "markov_output_mode": "direct",
            "markov_rank": 7,
            "add_noise": True,
            "target_hidden_noise_ratio": 0.1,
        }
        model = FlashMTPDraftModel(config)
        self.assertEqual(model.markov_head_type, "rnn_easy")
        self.assertEqual(model.markov_output_mode, "direct")
        self.assertEqual(model.markov_rank, 7)
        self.assertEqual(model.proposal_length, 3)
        self.assertEqual(model.sliding_window_size, 4)
        self.assertEqual(model.chs_num_layers, 2)
        self.assertEqual(model.current_chs_slot_count, 2)
        self.assertEqual(model.condition_slot_count, 2)
        self.assertEqual(model.chs_len_per_block, 2)
        self.assertEqual(model.draft_query_length, 7)
        self.assertFalse(model.config.flashmtp_config["include_token_embedding_chs"])
        self.assertFalse(model.config.flashmtp_config["pivot_query_embedding"])
        self.assertNotIn("add_noise", model.config.flashmtp_config)
        self.assertNotIn("target_hidden_noise_ratio", model.config.flashmtp_config)
        self.assertTrue(model.local_position)
        self.assertEqual(model.markov_head.max_prediction_length, 3)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model.save_pretrained(checkpoint_dir)
            loaded = FlashMTPDraftModel.from_pretrained(checkpoint_dir)
        self.assertEqual(loaded.markov_head_type, "rnn_easy")
        self.assertEqual(loaded.markov_output_mode, "direct")
        self.assertEqual(loaded.markov_rank, 7)
        self.assertEqual(loaded.proposal_length, 3)
        self.assertTrue(loaded.local_position)
        self.assertTrue(loaded.config.flashmtp_config["local_position"])
        self.assertEqual(loaded.config.flashmtp_config["architecture_version"], FLASHMTP_ARCHITECTURE_VERSION)
        self.assertFalse(
            loaded.config.flashmtp_config["include_token_embedding_chs"]
        )
        self.assertFalse(loaded.config.flashmtp_config["pivot_query_embedding"])
        self.assertIsNotNone(loaded.markov_head)

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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "markov_head_type": "none",
            "markov_output_mode": "additive",
        }
        legacy_model = FlashMTPDraftModel(config)
        block_hidden = torch.randn(2, 9, 16)
        legacy_hidden = legacy_model._prediction_hidden(block_hidden)
        self.assertEqual(legacy_hidden.shape, (2, 7, 16))

    def test_removed_anchor_kv_mode_is_rejected(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "draft_input_mode": "anchor_kv",
        }
        with self.assertRaisesRegex(ValueError, "draft_input_mode"):
            FlashMTPDraftModel(config)

        config.flashmtp_config["draft_input_mode"] = "legacy"
        model = FlashMTPDraftModel(config)
        self.assertEqual(model.unsupervised_query_count, 4)
        self.assertEqual(model.draft_query_length, 7)
        self.assertNotIn("draft_input_mode", model.config.flashmtp_config)

    def test_legacy_decode_block_sizes(self) -> None:
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
            "markov_head_type": "none",
            "markov_output_mode": "additive",
        }
        legacy_model = FlashMTPDraftModel(config)
        self.assertEqual(legacy_model.draft_block_len, 8)
        self.assertEqual(legacy_model.proposal_length, 7)
        self.assertEqual(legacy_model.max_verify_block_size, 8)

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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
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
        target_prediction_logits = torch.randn(2, 2, 3, 31)
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

        with mock.patch.object(
            wrapper.lm_head,
            "forward",
            side_effect=AssertionError("TV must reuse target prefill logits"),
        ):
            loss, accuracy, prefix_acc, final_ce_loss, base_ce_loss, tv_loss = (
                wrapper._chunked_weighted_ce_and_metrics(
                    prediction_hidden=prediction_hidden,
                    prev_token_ids=prev_token_ids,
                    labels=labels,
                    weight_mask=weight_mask,
                    binary_eval_mask=binary_eval_mask,
                    block_keep_mask=block_keep_mask,
                    target_prediction_logits=target_prediction_logits,
                )
            )
        markov_latent = draft_model.markov_head.forward_teacher_forcing(
            hidden_states=prediction_hidden,
            prev_token_ids=prev_token_ids,
            output_mode="direct",
        )
        draft_logits = draft_model.markov_head.project_logits(markov_latent)
        manual_tv = (
            (
                F.softmax(draft_logits, dim=-1)
                - F.softmax(target_prediction_logits, dim=-1)
            )
            .abs()
            .sum(dim=-1)
            .mul(weight_mask)
            .sum()
            / (weight_mask.sum() + 1e-6)
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
        self.assertIsNotNone(draft_model.markov_head.output_proj.weight.grad)
        self.assertIsNotNone(draft_model.markov_head.state_hidden_mlp.weight.grad)
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
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 4,
            "chs_num_layers": 2,
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

    def test_tv_prefill_logits_allow_synthetic_mask_vocab_row(self) -> None:
        config = Qwen3Config(
            vocab_size=12,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        config.num_target_layers = 2
        config.block_size = 3
        config.flashmtp_config = {
            "target_layer_ids": [0, 1],
            "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
            "sliding_window_size": 1,
            "chs_num_layers": 2,
            "markov_head_type": "rnn_easy",
            "markov_output_mode": "direct",
            "markov_rank": 4,
        }
        draft_model = FlashMTPDraftModel(config)
        wrapper = OnlineFlashMTPModel(
            draft_model=draft_model,
            target_lm_head=nn.Linear(8, 12, bias=False),
            target_embed_tokens=nn.Embedding(12, 8),
            mask_token_id=11,
            block_size=3,
            tv_loss_weight=1.0,
        )
        prediction_hidden = torch.randn(1, 1, 2, 8, requires_grad=True)
        target_prediction_logits = torch.randn(1, 1, 2, 11)
        token_ids = torch.randint(0, 11, (1, 1, 2))
        weights = torch.ones(1, 1, 2)

        result = wrapper._chunked_weighted_ce_and_metrics(
            prediction_hidden=prediction_hidden,
            prev_token_ids=token_ids,
            labels=token_ids,
            weight_mask=weights,
            binary_eval_mask=weights.bool(),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            target_prediction_logits=target_prediction_logits,
        )

        self.assertTrue(torch.isfinite(result[5]))


if __name__ == "__main__":
    unittest.main()
