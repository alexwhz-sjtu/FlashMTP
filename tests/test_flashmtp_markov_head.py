import copy
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from scripts.train_flashmtp_two_stage import (
    SHARED_BACKBONE_MODULES,
    _copy_partial_shared_backbone,
    _copy_serial_head,
    _copy_shared_backbone,
    _cosine_transition_scales,
    _evenly_spaced_teacher_layer_ids,
    _resolve_student_init_mode,
    _set_student_stage1_trainable,
    _set_student_stage2_trainable,
)
from scripts import flashmtp_training
from scripts.flashmtp_training import resume_cursor
from specforge.core.flashmtp import (
    OnlineFlashMTPModel,
    PreparedFlashMTPBatch,
    _make_flex_mask,
    compute_stage1_distillation_loss,
    gather_target_prefill_logits,
    gather_token_group,
)
from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    FlashMTPDraftModel,
)
from specforge.optimizer import BF16Optimizer
from specforge.data.utils import DataCollatorWithPadding


def make_model(
    role="pivot_q_student",
    *,
    block_size=4,
    g=3,
    w=4,
    num_draft_layers=1,
    markov_head_type="vanilla",
    markov_output_mode="additive",
):
    config = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_draft_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    config.num_target_layers = 4
    config.block_size = block_size
    config.layer_types = ["full_attention"] * num_draft_layers
    config._attn_implementation = "eager"
    config.flashmtp_config = {
        "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
        "model_role": role,
        "swa_window_size": w,
        "anchor_group_size": g,
        "chs_num_layers": 2,
        "markov_head_type": markov_head_type,
        "markov_output_mode": markov_output_mode,
        "markov_rank": 4,
    }
    return FlashMTPDraftModel(config)


class CountingHead(nn.Linear):
    def __init__(self):
        super().__init__(16, 31, bias=False)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return super().forward(value)


class CurrentFlashMTPArchitectureTest(unittest.TestCase):
    def test_training_collator_matches_v2_dynamic_batch_padding(self):
        with mock.patch(
            "specforge.data.utils.torch.distributed.get_world_size", return_value=1
        ):
            collator = DataCollatorWithPadding()

        short = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "loss_mask": torch.ones(1, 3, dtype=torch.long),
        }
        long = {
            "input_ids": torch.tensor([[4, 5, 6, 7, 8]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "loss_mask": torch.ones(1, 5, dtype=torch.long),
        }

        batch = collator([short, long])

        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertTrue(
            torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3, 0, 0]))
        )
        self.assertTrue(
            torch.equal(batch["attention_mask"][0], torch.tensor([1, 1, 1, 0, 0]))
        )
        self.assertTrue(
            torch.equal(batch["loss_mask"][0], torch.tensor([1, 1, 1, 0, 0]))
        )

    def test_training_dataloader_does_not_force_max_length_padding(self):
        args = SimpleNamespace(
            block_size=4,
            batch_size=2,
            dataloader_num_workers=0,
            max_length=10240,
        )
        dataset = mock.Mock()
        dataset.filter.return_value = dataset
        sentinel = [object()]

        with (
            mock.patch.object(
                flashmtp_training, "prepare_dp_dataloaders", return_value=sentinel
            ) as prepare,
            mock.patch.object(flashmtp_training, "get_dp_group", return_value="dp"),
        ):
            result = flashmtp_training._prepare_dataloader(
                args, dataset, train_data_path="/data/train.jsonl"
            )

        self.assertIs(result, sentinel)
        self.assertNotIn("pad_to_length", prepare.call_args.kwargs)

    def test_dataset_filter_requires_a_trainable_anchor(self):
        from scripts.flashmtp_training import _has_valid_anchor_supervision

        # Eight supervised tokens satisfy the old count-only filter, but the
        # only adjacent pair is too close to the end for a full block.
        tail_only = torch.zeros(16)
        tail_only[[1, 3, 5, 7, 9, 11, 14, 15]] = 1
        self.assertFalse(
            _has_valid_anchor_supervision(
                {"loss_mask": tail_only}, block_size=4
            )
        )

        trainable = tail_only.clone()
        trainable[4] = 1
        self.assertTrue(
            _has_valid_anchor_supervision(
                {"loss_mask": trainable}, block_size=4
            )
        )

    def test_two_stage_dataset_preprocessing_starts_concurrently(self):
        args = SimpleNamespace(
            stage1_train_data_path="/data/stage1.jsonl",
            stage2_train_data_path="/data/stage2.jsonl",
            stage1_build_dataset_num_proc=3,
            stage2_build_dataset_num_proc=5,
            build_dataset_num_proc=8,
        )
        rendezvous = threading.Barrier(2)
        calls = []

        def build(_args, _tokenizer, **kwargs):
            calls.append((kwargs["cache_namespace"], kwargs["num_proc"]))
            rendezvous.wait(timeout=2)
            return kwargs["cache_namespace"]

        def prepare(_args, dataset, *, train_data_path):
            return dataset, train_data_path

        with (
            mock.patch.object(flashmtp_training.dist, "get_rank", return_value=0),
            mock.patch.object(
                flashmtp_training.dist, "get_world_size", return_value=1
            ),
            mock.patch.object(flashmtp_training.dist, "barrier"),
            mock.patch.object(
                flashmtp_training, "_build_processed_dataset", side_effect=build
            ),
            mock.patch.object(
                flashmtp_training, "_prepare_dataloader", side_effect=prepare
            ),
            mock.patch.object(flashmtp_training, "print_on_rank0"),
        ):
            stage1, stage2 = flashmtp_training.build_two_stage_dataloaders(
                args, object()
            )

        self.assertCountEqual(calls, [("stage1", 3), ("stage2", 5)])
        self.assertEqual(stage1, ("stage1", "/data/stage1.jsonl"))
        self.assertEqual(stage2, ("stage2", "/data/stage2.jsonl"))

    def test_stage_total_steps_carries_accumulation_across_epochs(self):
        from scripts.flashmtp_training import stage_total_steps

        dataloader = [None] * 3
        self.assertEqual(stage_total_steps(dataloader, epochs=2, accumulation_steps=2), 3)
        self.assertEqual(stage_total_steps(dataloader, epochs=2, accumulation_steps=4), 2)

    def test_tp_rank_batch_selection_recurses_and_owns_storage(self):
        from scripts.flashmtp_training import select_tp_rank_batch

        source = torch.arange(24).view(3, 2, 4)
        selected = select_tp_rank_batch(
            {"dict": source, "tuple": (source + 1,)}, tp_rank=1
        )
        self.assertTrue(torch.equal(selected["dict"], source[1:2]))
        self.assertTrue(torch.equal(selected["tuple"][0], source[1:2] + 1))
        self.assertNotEqual(
            selected["dict"].untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )

    def test_stage_resume_cursor_preserves_monotonic_global_step(self):
        state = {
            "training_stage": "stage2",
            "stage_epoch": 3,
            "next_batch_in_epoch": 17,
            "stage_step": 211,
            "global_step": 509,
        }
        self.assertEqual(resume_cursor(state, "stage2"), (3, 17, 211, 509))
        with self.assertRaisesRegex(ValueError, "Expected a 'stage1' checkpoint"):
            resume_cursor(state, "stage1")

    def test_only_current_architecture_loads(self):
        model = make_model()
        self.assertTrue(model.is_student)
        config = copy.deepcopy(model.config)
        config.flashmtp_config["architecture_version"] = "sliding_chs_first_token_window_v5"
        with self.assertRaisesRegex(ValueError, "Historical checkpoints"):
            FlashMTPDraftModel(config)

    def test_teacher_and_student_positions(self):
        anchor = torch.tensor([[5]])
        token_pos = torch.tensor([[[3, 4, 5]]])
        keep = torch.ones_like(token_pos, dtype=torch.bool)
        teacher = make_model("swa_teacher")
        teacher_ctx, teacher_q = teacher.build_block_position_ids(anchor, token_pos, keep)
        self.assertTrue(torch.equal(teacher_ctx, torch.tensor([[1, 2, 3, 4, 4]])))
        self.assertTrue(torch.equal(teacher_q, torch.tensor([[3, 4, 5, 6, 7, 8]])))
        student = make_model("pivot_q_student")
        student_ctx, student_q = student.build_block_position_ids(anchor, token_pos, keep)
        self.assertTrue(torch.equal(student_ctx, torch.tensor([[1, 1]])))
        self.assertTrue(torch.equal(student_q, torch.tensor([[0, 1, 2, 3, 4, 5]])))

    def test_short_context_is_left_padded_and_masked(self):
        ids = torch.tensor([[10, 11, 12]])
        gathered, keep, positions = gather_token_group(
            ids, torch.tensor([[1]]), 4, fill_token_id=30
        )
        self.assertTrue(torch.equal(gathered, torch.tensor([[[30, 30, 10, 11]]])))
        self.assertTrue(torch.equal(keep, torch.tensor([[[False, False, True, True]]])))
        student = make_model(g=4)
        context, query = student.build_block_position_ids(
            torch.tensor([[1]]), positions, keep
        )
        self.assertTrue(torch.equal(context, torch.tensor([[0, 0]])))
        self.assertTrue(torch.equal(query, torch.tensor([[0, 0, 0, 1, 2, 3, 4]])))

    def test_invalid_block_keeps_finite_chs_fallback(self):
        with mock.patch(
            "specforge.core.flashmtp.compile_friendly_create_block_mask",
            side_effect=lambda mask_mod, **_: mask_mod,
        ):
            mask_mod = _make_flex_mask(
                model_role="pivot_q_student",
                anchor_positions=torch.tensor([[4, 0]]),
                block_keep_mask=torch.tensor([[True, False]]),
                token_keep_mask=torch.ones(1, 2, 2, dtype=torch.bool),
                seq_len=8,
                swa_window_size=4,
                chs_slots=2,
                query_len=3,
                device=torch.device("cpu"),
            )

        def visible(q_idx, kv_idx):
            return bool(
                mask_mod(
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(q_idx),
                    torch.tensor(kv_idx),
                )
            )

        # KV: [CHS_0 (2) | CHS_1 (2) | Q_0 (3) | Q_1 (3)].
        self.assertFalse(visible(3, 0))
        self.assertFalse(visible(3, 1))
        self.assertTrue(visible(3, 2))
        self.assertTrue(visible(3, 3))
        self.assertFalse(visible(3, 7))

    def test_anchor_sampling_requires_next_supervised_label(self):
        model = make_model(block_size=2)
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=2,
            num_anchors=8,
        )
        loss_mask = torch.zeros(1, 8)
        loss_mask[0, [2, 4, 5, 6]] = 1

        anchors, keep = wrapper.sample_anchor_positions(8, loss_mask)

        valid_anchors = anchors[keep]
        self.assertEqual(valid_anchors.numel(), 2)
        self.assertTrue(torch.isin(valid_anchors, torch.tensor([4, 5])).all())
        self.assertFalse((valid_anchors == 2).any())

    def test_anchor_sampling_keeps_sparse_rows_after_position_sort(self):
        model = make_model(block_size=2)
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=2,
            num_anchors=8,
        )
        loss_mask = torch.zeros(2, 12)
        loss_mask[0, 8:10] = 1
        loss_mask[1, 2:10] = 1

        anchors, keep = wrapper.sample_anchor_positions(12, loss_mask)

        self.assertEqual(int(keep[0].sum()), 1)
        self.assertEqual(anchors[0, keep[0]].tolist(), [8])
        self.assertEqual(int(keep[1].sum()), 7)
        for row in range(loss_mask.size(0)):
            selected = anchors[row, keep[row]]
            self.assertTrue((loss_mask[row, selected] > 0.5).all())
            self.assertTrue((loss_mask[row, selected + 1] > 0.5).all())

    def test_inference_mask_embedding_does_not_expand_output_vocab(self):
        model = make_model()
        model.mask_token_id = 31
        embedding = nn.Embedding(31, 16)
        token_group = torch.tensor([[1, 2, 3]])
        draft_ids = torch.tensor([[3, 31, 31, 31]])
        query = model.build_inference_query_embeddings(
            embedding, draft_ids, token_group_ids=token_group
        )
        expected_mask = embedding.weight.mean(dim=0)
        self.assertEqual(query.shape, (1, 6, 16))
        self.assertTrue(torch.allclose(query[0, -1], expected_mask))

    def test_inference_vocab_row_mask_uses_exact_embedding(self):
        model = make_model()
        model.mask_token_id = 30
        model.mask_embedding_mode = "vocab_row"
        embedding = nn.Embedding(31, 16)
        token_group = torch.tensor([[1, 2, 3]])
        draft_ids = torch.tensor([[3, 30, 30, 30]])

        query = model.build_inference_query_embeddings(
            embedding, draft_ids, token_group_ids=token_group
        )

        self.assertTrue(torch.equal(query[0, -1], embedding.weight[30]))

    def test_inference_vocab_row_mask_rejects_oov_embedding(self):
        model = make_model()
        model.mask_token_id = 31
        model.mask_embedding_mode = "vocab_row"
        with self.assertRaisesRegex(ValueError, "requires vocab_row"):
            model.build_inference_query_embeddings(
                nn.Embedding(31, 16),
                torch.tensor([[3, 31, 31, 31]]),
                token_group_ids=torch.tensor([[1, 2, 3]]),
            )

    def test_target_component_builder_passes_constructed_target(self):
        args = object()
        drafts = [object()]
        target = object()
        tokenizer = object()
        components = object()
        with mock.patch.object(
            flashmtp_training, "build_target_model", return_value=target
        ), mock.patch.object(
            flashmtp_training,
            "resolve_tokenizer_and_components",
            return_value=(tokenizer, components, 17),
        ) as resolve:
            result = flashmtp_training.build_target_and_components(args, drafts)

        self.assertEqual(result, (target, tokenizer, components, 17))
        resolve.assert_called_once_with(args, drafts, target=target)

    def test_prediction_hidden_is_last_b_minus_one_slots(self):
        for role in ("swa_teacher", "pivot_q_student"):
            model = make_model(role)
            hidden = torch.arange(model.draft_query_length * 16).view(
                1, model.draft_query_length, 16
            )
            self.assertTrue(torch.equal(model._prediction_hidden(hidden), hidden[:, -3:]))

    def test_teacher_shared_history_rope_is_not_duplicated_per_anchor(self):
        model = make_model("swa_teacher")
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
        )
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[5]]),
            block_keep_mask=torch.tensor([[True]]),
            target_hidden=torch.randn(1, 1, 2, 16),
            shared_fused_history=torch.randn(1, 7, 16),
            query_embeddings=torch.randn(1, 1, 6, 16),
            token_keep_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            token_position_ids=torch.tensor([[[3, 4, 5]]]),
            labels=torch.zeros(1, 1, 3, dtype=torch.long),
            prev_token_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            raw_weight_mask=torch.ones(1, 1, 3),
            binary_eval_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            initial_prev_token_ids=None,
        )
        captured = {}

        def fake_forward(**kwargs):
            captured["rotary"] = kwargs["rotary_position_ids"]
            return torch.zeros(1, 6, 16)

        with mock.patch("specforge.core.flashmtp._make_flex_mask", return_value=None):
            with mock.patch.object(model, "forward", side_effect=fake_forward):
                wrapper.forward_backbone(batch, seq_len=7)
        # one shared 7-token history + two CHS slots + six Q slots
        self.assertEqual(captured["rotary"].shape, (1, 15))
        self.assertTrue(torch.equal(captured["rotary"][0, :7], torch.arange(7)))

    def test_prefill_logits_use_causal_predecessor_positions(self):
        logits = torch.arange(1 * 8 * 5).view(1, 8, 5)
        selected = gather_target_prefill_logits(logits, torch.tensor([[2, 4]]), 4)
        self.assertTrue(torch.equal(selected[0, 0], logits[0, 2:5]))
        self.assertTrue(torch.equal(selected[0, 1], logits[0, 4:7]))

    def test_prefill_gather_matches_projecting_selected_hidden(self):
        head = nn.Linear(16, 31, bias=False)
        hidden = torch.randn(1, 8, 16)
        anchors = torch.tensor([[2, 4]])
        full_logits = head(hidden)
        gathered_logits = gather_target_prefill_logits(full_logits, anchors, 4)
        positions = anchors.unsqueeze(-1) + torch.arange(3).view(1, 1, -1)
        selected_hidden = torch.gather(
            hidden.unsqueeze(1).expand(-1, 2, -1, -1),
            2,
            positions.unsqueeze(-1).expand(-1, -1, -1, 16),
        )
        self.assertTrue(torch.allclose(gathered_logits, head(selected_hidden)))

    def test_two_stages_share_one_combined_scheduler(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        all_parameters = list(model.parameters())
        for parameter in model[1].parameters():
            parameter.requires_grad_(False)
        optimizer = BF16Optimizer(
            model,
            parameters=all_parameters,
            lr=1e-3,
            total_steps=30,
            warmup_ratio=0.1,
        )

        self.assertEqual(len(optimizer.model_params), len(all_parameters))
        self.assertEqual(optimizer.scheduler.warmup_epochs, 3)
        self.assertEqual(optimizer.scheduler.after_scheduler.T_max, 27)

    def test_supervised_logits_are_projected_once(self):
        model = make_model()
        head = CountingHead()
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=head,
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            attention_backend="flex_attention",
            final_ce_weight=1.0,
            tv_loss_weight=1.0,
            base_lm_ce_weight=1.0,
        )
        hidden = torch.randn(1, 1, 3, 16, requires_grad=True)
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.tensor([[True]]),
            target_hidden=torch.empty(1, 1, 2, 16),
            shared_fused_history=None,
            query_embeddings=torch.empty(1, 1, 6, 16),
            token_keep_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            token_position_ids=torch.tensor([[[0, 1, 2]]]),
            labels=torch.tensor([[[3, 4, 5]]]),
            prev_token_ids=torch.tensor([[[2, 3, 4]]]),
            raw_weight_mask=torch.ones(1, 1, 3),
            binary_eval_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            initial_prev_token_ids=None,
        )
        target_logits = torch.randn(1, 1, 3, 31)
        output = wrapper.compute_supervised_loss(hidden, batch, target_logits)
        self.assertEqual(head.calls, 1)
        output.loss.backward()
        self.assertIsNotNone(hidden.grad)

    def test_masked_nan_block_does_not_contaminate_supervised_loss(self):
        model = make_model()
        head = nn.Linear(16, 31, bias=False)
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=head,
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
            final_ce_weight=1.0,
            tv_loss_weight=1.0,
            base_lm_ce_weight=1.0,
        )
        hidden = torch.randn(1, 2, 3, 16, requires_grad=True)
        target_logits = torch.randn(1, 2, 3, 31)
        with torch.no_grad():
            hidden[:, 1].fill_(float("nan"))
            target_logits[:, 1].fill_(float("nan"))
        weights = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[2, 0]]),
            block_keep_mask=torch.tensor([[True, False]]),
            target_hidden=torch.empty(1, 2, 2, 16),
            shared_fused_history=None,
            query_embeddings=torch.empty(1, 2, 6, 16),
            token_keep_mask=torch.ones(1, 2, 3, dtype=torch.bool),
            token_position_ids=torch.zeros(1, 2, 3, dtype=torch.long),
            labels=torch.randint(0, 31, (1, 2, 3)),
            prev_token_ids=torch.zeros(1, 2, 3, dtype=torch.long),
            raw_weight_mask=weights,
            binary_eval_mask=weights.bool(),
            initial_prev_token_ids=None,
        )

        output = wrapper.compute_supervised_loss(hidden, batch, target_logits)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.loss.dtype, torch.float32)
        output.loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertTrue(torch.equal(hidden.grad[:, 1], torch.zeros_like(hidden.grad[:, 1])))
        for parameter in model.markov_head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_illegal_supervised_label_is_rejected(self):
        model = make_model()
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
        )
        weights = torch.ones(1, 1, 3)
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            target_hidden=torch.empty(1, 1, 2, 16),
            shared_fused_history=None,
            query_embeddings=torch.empty(1, 1, 6, 16),
            token_keep_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            token_position_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            labels=torch.tensor([[[31, 1, 2]]]),
            prev_token_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            raw_weight_mask=weights,
            binary_eval_mask=weights.bool(),
            initial_prev_token_ids=None,
        )

        with self.assertRaisesRegex(ValueError, "within the output vocabulary"):
            wrapper.compute_supervised_loss(
                torch.randn(1, 1, 3, 16), batch, torch.randn(1, 1, 3, 31)
            )

    def test_empty_supervision_is_rejected(self):
        model = make_model()
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
        )
        weights = torch.zeros(1, 1, 3)
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[0]]),
            block_keep_mask=torch.zeros(1, 1, dtype=torch.bool),
            target_hidden=torch.empty(1, 1, 2, 16),
            shared_fused_history=None,
            query_embeddings=torch.empty(1, 1, 6, 16),
            token_keep_mask=torch.zeros(1, 1, 3, dtype=torch.bool),
            token_position_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            labels=torch.zeros(1, 1, 3, dtype=torch.long),
            prev_token_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            raw_weight_mask=weights,
            binary_eval_mask=weights.bool(),
            initial_prev_token_ids=None,
        )

        with self.assertRaisesRegex(ValueError, "no supervised label positions"):
            wrapper.compute_supervised_loss(
                torch.randn(1, 1, 3, 16), batch, torch.randn(1, 1, 3, 31)
            )

    def test_stage1_teacher_is_detached(self):
        student = torch.randn(1, 2, 3, 16, requires_grad=True)
        teacher = torch.randn(1, 2, 3, 16, requires_grad=True)
        student_logits = torch.randn(6, 31, requires_grad=True)
        teacher_logits = torch.randn(6, 31, requires_grad=True)
        loss, _, _, _ = compute_stage1_distillation_loss(
            student_hidden=student,
            teacher_hidden=teacher,
            student_serial_logits=student_logits,
            teacher_serial_logits=teacher_logits,
            raw_weight_mask=torch.ones(1, 2, 3),
            labels=torch.zeros(1, 2, 3, dtype=torch.long),
            kl_weight=1.0,
            hidden_weight=1.0,
            smooth_l1_beta=1.0,
            loss_decay_gamma=2.0,
        )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_stage1_forward_kl_is_zero_for_matching_distributions(self):
        student = torch.randn(1, 1, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student)
        logits = torch.randn(3, 31)

        loss, kl_loss, hidden_loss, _ = compute_stage1_distillation_loss(
            student_hidden=student,
            teacher_hidden=teacher,
            student_serial_logits=logits.clone().requires_grad_(),
            teacher_serial_logits=logits.clone(),
            raw_weight_mask=torch.ones(1, 1, 3),
            labels=torch.zeros(1, 1, 3, dtype=torch.long),
            kl_weight=2.0,
            hidden_weight=0.0,
            smooth_l1_beta=1.0,
            loss_decay_gamma=None,
        )

        self.assertAlmostEqual(kl_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertEqual(hidden_loss.item(), 0.0)

    def test_frozen_direct_serial_head_backpropagates_only_to_hidden(self):
        model = make_model(
            markov_head_type="rnn_easy",
            markov_output_mode="direct",
        )
        model.markov_head.requires_grad_(False)
        wrapper = OnlineFlashMTPModel(
            draft_model=model,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
        )
        hidden = torch.randn(1, 1, 3, 16, requires_grad=True)
        weights = torch.ones(1, 1, 3)
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            target_hidden=torch.empty(1, 1, 2, 16),
            shared_fused_history=None,
            query_embeddings=torch.empty(1, 1, 6, 16),
            token_keep_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            token_position_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            labels=torch.zeros(1, 1, 3, dtype=torch.long),
            prev_token_ids=torch.tensor([[[1, 2, 3]]]),
            raw_weight_mask=weights,
            binary_eval_mask=weights.bool(),
            initial_prev_token_ids=torch.tensor([[1]]),
        )

        wrapper.compute_serial_logits(hidden, batch).sum().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertGreater(hidden.grad.abs().sum().item(), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.markov_head.parameters())
        )

    def test_stage1_masked_nan_does_not_contaminate_loss(self):
        student = torch.randn(1, 2, 3, 16, requires_grad=True)
        teacher = torch.randn(1, 2, 3, 16, requires_grad=True)
        with torch.no_grad():
            student[:, 1].fill_(float("nan"))
            teacher[:, 1].fill_(float("nan"))
        weights = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
        student_logits = torch.randn(3, 31, requires_grad=True)
        teacher_logits = torch.randn(3, 31, requires_grad=True)

        loss, kl_loss, hidden_loss, _ = compute_stage1_distillation_loss(
            student_hidden=student,
            teacher_hidden=teacher,
            student_serial_logits=student_logits,
            teacher_serial_logits=teacher_logits,
            raw_weight_mask=weights,
            labels=torch.zeros(1, 2, 3, dtype=torch.long),
            kl_weight=1.0,
            hidden_weight=1.0,
            smooth_l1_beta=1.0,
            loss_decay_gamma=2.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(kl_loss))
        self.assertTrue(torch.isfinite(hidden_loss))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertTrue(torch.isfinite(student_logits.grad).all())
        self.assertTrue(torch.equal(student.grad[:, 1], torch.zeros_like(student.grad[:, 1])))
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_stage1_zero_hidden_weight_skips_smooth_l1_and_reports_prefix(self):
        student_logits = torch.tensor(
            [[[[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]],
              [[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]]],
        ).reshape(6, 3).requires_grad_()
        teacher_logits = torch.zeros_like(student_logits, requires_grad=True)
        student = torch.randn(1, 2, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        labels = torch.tensor([[[1, 2, 2], [1, 2, 2]]])

        with mock.patch(
            "specforge.core.flashmtp.F.smooth_l1_loss",
            side_effect=AssertionError("SmoothL1 should be skipped"),
        ):
            loss, kl_loss, hidden_loss, prefix_acc = (
                compute_stage1_distillation_loss(
                    student_hidden=student,
                    teacher_hidden=teacher,
                    student_serial_logits=student_logits,
                    teacher_serial_logits=teacher_logits,
                    raw_weight_mask=torch.ones(1, 2, 3),
                    labels=labels,
                    kl_weight=1.0,
                    hidden_weight=0.0,
                    smooth_l1_beta=1.0,
                    loss_decay_gamma=None,
                )
            )

        self.assertTrue(torch.equal(loss, kl_loss))
        self.assertEqual(hidden_loss.item(), 0.0)
        self.assertEqual(prefix_acc.item(), 2.5)
        loss.backward()
        self.assertIsNone(student.grad)
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_serial_head_copy_does_not_touch_backbone(self):
        teacher = make_model("swa_teacher")
        student = make_model("pivot_q_student")
        before = student.layers[0].self_attn.q_proj.weight.detach().clone()
        with torch.no_grad():
            for parameter in teacher.markov_head.parameters():
                parameter.fill_(0.25)
        _copy_serial_head(teacher, student)
        self.assertTrue(torch.equal(before, student.layers[0].self_attn.q_proj.weight))
        for parameter in student.markov_head.parameters():
            self.assertTrue(torch.all(parameter == 0.25))

    def test_stage1_freezes_serial_head_and_stage2_unfreezes_it(self):
        student = make_model("pivot_q_student")

        _set_student_stage1_trainable(student)
        self.assertTrue(any(parameter.requires_grad for parameter in student.layers.parameters()))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in student.markov_head.parameters()
            )
        )

        _set_student_stage2_trainable(student)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in student.markov_head.parameters()
            )
        )

    def test_transition_cosine_scales_cover_both_endpoints(self):
        self.assertEqual(_cosine_transition_scales(0, 5), (1.0, 0.0))
        stage1_scale, stage2_scale = _cosine_transition_scales(2, 5)
        self.assertAlmostEqual(stage1_scale, 0.5)
        self.assertAlmostEqual(stage2_scale, 0.5)
        self.assertEqual(_cosine_transition_scales(4, 5), (0.0, 1.0))
        self.assertEqual(_cosine_transition_scales(0, 1), (0.5, 0.5))

        stage1_scales = [
            _cosine_transition_scales(index, 5)[0] for index in range(5)
        ]
        self.assertTrue(
            all(
                left >= right
                for left, right in zip(stage1_scales, stage1_scales[1:])
            )
        )

    def test_transition_stage1_kl_trains_serial_head(self):
        student = make_model(
            "pivot_q_student",
            markov_head_type="rnn_easy",
            markov_output_mode="direct",
        )
        _set_student_stage2_trainable(student)
        wrapper = OnlineFlashMTPModel(
            draft_model=student,
            target_lm_head=nn.Linear(16, 31, bias=False),
            target_embed_tokens=nn.Embedding(31, 16),
            mask_token_id=30,
            block_size=4,
        )
        hidden = torch.randn(1, 1, 3, 16, requires_grad=True)
        weights = torch.ones(1, 1, 3)
        batch = PreparedFlashMTPBatch(
            anchor_positions=torch.tensor([[2]]),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            target_hidden=torch.empty(1, 1, 2, 16),
            shared_fused_history=None,
            query_embeddings=torch.empty(1, 1, 6, 16),
            token_keep_mask=torch.ones(1, 1, 3, dtype=torch.bool),
            token_position_ids=torch.zeros(1, 1, 3, dtype=torch.long),
            labels=torch.zeros(1, 1, 3, dtype=torch.long),
            prev_token_ids=torch.tensor([[[1, 2, 3]]]),
            raw_weight_mask=weights,
            binary_eval_mask=weights.bool(),
            initial_prev_token_ids=torch.tensor([[1]]),
        )
        student_logits = wrapper.compute_serial_logits(hidden, batch)
        loss, _, _, _ = compute_stage1_distillation_loss(
            student_hidden=hidden,
            teacher_hidden=torch.randn_like(hidden),
            student_serial_logits=student_logits,
            teacher_serial_logits=torch.randn_like(student_logits),
            raw_weight_mask=weights,
            labels=batch.labels,
            kl_weight=1.0,
            hidden_weight=0.0,
            smooth_l1_beta=1.0,
            loss_decay_gamma=None,
        )
        loss.backward()

        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in student.markov_head.parameters()
            )
        )

    def test_shared_init_copies_only_parallel_backbone(self):
        teacher = make_model("swa_teacher")
        student = make_model("pivot_q_student")
        with torch.no_grad():
            for module_name in SHARED_BACKBONE_MODULES:
                for parameter in getattr(teacher, module_name).parameters():
                    parameter.fill_(0.25)
            for parameter in teacher.markov_head.parameters():
                parameter.fill_(0.5)
            for parameter in teacher.history_fuse.parameters():
                parameter.fill_(0.75)
        serial_before = {
            name: value.detach().clone()
            for name, value in student.markov_head.state_dict().items()
        }
        history_before = {
            name: value.detach().clone()
            for name, value in student.history_fuse.state_dict().items()
        }

        _copy_shared_backbone(teacher, student)

        for module_name in SHARED_BACKBONE_MODULES:
            for parameter in getattr(student, module_name).parameters():
                self.assertTrue(torch.all(parameter == 0.25))
        for name, value in student.markov_head.state_dict().items():
            self.assertTrue(torch.equal(value, serial_before[name]))
        for name, value in student.history_fuse.state_dict().items():
            self.assertTrue(torch.equal(value, history_before[name]))

    def test_shared_partial_evenly_copies_teacher_backbone_layers(self):
        teacher = make_model("swa_teacher", num_draft_layers=5)
        student = make_model("pivot_q_student", num_draft_layers=3)
        with torch.no_grad():
            for layer_id, layer in enumerate(teacher.layers):
                for parameter in layer.parameters():
                    parameter.fill_(float(layer_id + 1))
            for module_name in SHARED_BACKBONE_MODULES[1:]:
                for parameter in getattr(teacher, module_name).parameters():
                    parameter.fill_(0.25)
            for parameter in teacher.markov_head.parameters():
                parameter.fill_(0.5)
        serial_before = {
            name: value.detach().clone()
            for name, value in student.markov_head.state_dict().items()
        }

        copied_ids = _copy_partial_shared_backbone(teacher, student)

        self.assertEqual(copied_ids, [0, 2, 4])
        for student_layer, expected in zip(
            student.layers, (1.0, 3.0, 5.0), strict=True
        ):
            for parameter in student_layer.parameters():
                self.assertTrue(torch.all(parameter == expected))
        for module_name in SHARED_BACKBONE_MODULES[1:]:
            for parameter in getattr(student, module_name).parameters():
                self.assertTrue(torch.all(parameter == 0.25))
        for name, value in student.markov_head.state_dict().items():
            self.assertTrue(torch.equal(value, serial_before[name]))

    def test_shared_partial_requires_deeper_teacher(self):
        with self.assertRaisesRegex(ValueError, "teacher draft depth"):
            _evenly_spaced_teacher_layer_ids(3, 3)
        with self.assertRaisesRegex(ValueError, "teacher draft depth"):
            _copy_partial_shared_backbone(
                make_model("swa_teacher", num_draft_layers=2),
                make_model("pivot_q_student", num_draft_layers=3),
            )

    def test_serial_head_copy_allows_different_backbone_depths(self):
        teacher = make_model("swa_teacher", num_draft_layers=5)
        student = make_model("pivot_q_student", num_draft_layers=3)
        with torch.no_grad():
            for parameter in teacher.markov_head.parameters():
                parameter.fill_(0.125)

        _copy_serial_head(teacher, student)

        for parameter in student.markov_head.parameters():
            self.assertTrue(torch.all(parameter == 0.125))

    def test_student_init_mode_is_checkpoint_stable(self):
        self.assertEqual(_resolve_student_init_mode(None, None), "scratch")
        self.assertEqual(_resolve_student_init_mode("shared_init", None), "shared_init")
        state = {"student_init_mode": "shared_init"}
        self.assertEqual(_resolve_student_init_mode(None, state), "shared_init")
        with self.assertRaisesRegex(ValueError, "must match"):
            _resolve_student_init_mode("scratch", state)
        inconsistent = {
            "student_init_mode": "shared_init",
            "shared_backbone_inherited": False,
        }
        with self.assertRaisesRegex(ValueError, "metadata is inconsistent"):
            _resolve_student_init_mode(None, inconsistent)
        partial_state = {
            "student_init_mode": "shared_partial",
            "shared_backbone_inherited": True,
        }
        self.assertEqual(
            _resolve_student_init_mode(None, partial_state), "shared_partial"
        )


if __name__ == "__main__":
    unittest.main()
