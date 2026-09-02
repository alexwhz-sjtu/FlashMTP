import unittest
from types import MethodType, SimpleNamespace

import torch

from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.modeling.target.flashmtp_target_model import (
    SGLangFlashMTPTargetModel,
    TempRolloutPrefillHandle,
    build_temp_rollout_branch_fill_ids,
)


class _IdentityLMHead(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(vocab_size), requires_grad=False)
        self.batch_sizes = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(hidden_states.size(0))
        return hidden_states @ self.weight.T


class _FakeRolloutModel:
    _temp_rollout_greedy = OnlineFlashMTPModel._temp_rollout_greedy
    _lm_head_module = OnlineFlashMTPModel._lm_head_module
    _needs_target_distribution_hidden = (
        OnlineFlashMTPModel._needs_target_distribution_hidden
    )

    def __init__(self, *, prediction_length: int, eos_token_id=None):
        self.block_size = prediction_length + 1
        self.tv_loss_weight = 1.0
        self.final_forward_kl_weight = 0.0
        self.base_lm_forward_kl_weight = 0.0
        self.draft_model = SimpleNamespace(markov_head=object())
        self.temp_rollout_projection_chunk_size = 1
        self.target_vocab_size = 6
        self.eos_token_id = eos_token_id
        self.lm_head = _IdentityLMHead(self.target_vocab_size)


class _FakeRolloutContext:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.calls = []

    def extend_step(self, anchors, generated_ids, active_mask):
        self.calls.append(
            (
                anchors.clone(),
                generated_ids.clone(),
                active_mask.clone(),
            )
        )
        newest = generated_ids[..., -1]
        next_token = (newest + 1) % self.vocab_size
        hidden = torch.nn.functional.one_hot(
            next_token, num_classes=self.vocab_size
        ).float()
        return torch.where(active_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))


class TempRolloutTest(unittest.TestCase):
    def test_branch_fill_uses_only_its_true_prefix(self):
        true_ids = [10, 11, 12, 13, 14, 15]
        fill_a, prefix_a = build_temp_rollout_branch_fill_ids(
            true_ids, 1, [101, 102]
        )
        fill_b, prefix_b = build_temp_rollout_branch_fill_ids(
            true_ids, 4, [201]
        )
        self.assertEqual(fill_a, [10, 11, 101, 102])
        self.assertEqual(prefix_a, 2)
        self.assertEqual(fill_b, [10, 11, 12, 13, 14, 201])
        self.assertEqual(prefix_b, 5)
        self.assertNotIn(101, fill_b)

    def test_anchor_parallel_autoregressive_greedy(self):
        model = _FakeRolloutModel(prediction_length=3)
        model.temp_rollout_projection_chunk_size = 0
        context = _FakeRolloutContext(model.target_vocab_size)
        anchors = torch.tensor([[3, 9]])
        keep = torch.tensor([[True, True]])
        first_hidden = torch.tensor(
            [[[0, 5, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0]]],
            dtype=torch.float32,
        )

        labels, validity, predecessor_hidden = model._temp_rollout_greedy(
            anchor_positions=anchors,
            block_keep_mask=keep,
            target_anchor_hidden=first_hidden,
            rollout_context=context,
        )

        self.assertEqual(labels.tolist(), [[[1, 2, 3], [2, 3, 4]]])
        self.assertTrue(validity.all())
        self.assertEqual(tuple(predecessor_hidden.shape), (1, 2, 3, 6))
        self.assertEqual(len(context.calls), 2)
        # Both anchors are extended together at each step, but with private IDs.
        self.assertEqual(context.calls[0][1].tolist(), [[[1], [2]]])
        self.assertEqual(context.calls[1][1].tolist(), [[[1, 2], [2, 3]]])
        self.assertEqual(model.lm_head.batch_sizes, [2, 2, 2])

    def test_eos_is_valid_and_later_positions_are_masked(self):
        model = _FakeRolloutModel(prediction_length=3, eos_token_id=2)
        model.temp_rollout_projection_chunk_size = 0
        context = _FakeRolloutContext(model.target_vocab_size)
        anchors = torch.tensor([[3, 9]])
        keep = torch.tensor([[True, True]])
        first_hidden = torch.tensor(
            [[[0, 5, 0, 0, 0, 0], [0, 0, 7, 0, 0, 0]]],
            dtype=torch.float32,
        )

        labels, validity, _ = model._temp_rollout_greedy(
            anchor_positions=anchors,
            block_keep_mask=keep,
            target_anchor_hidden=first_hidden,
            rollout_context=context,
        )

        self.assertEqual(labels[0, 0, :2].tolist(), [1, 2])
        self.assertEqual(validity.tolist(), [[[True, True, False], [True, False, False]]])
        self.assertEqual(context.calls[0][2].tolist(), [[True, False]])
        self.assertFalse(context.calls[1][2].any())

    def test_persistent_decode_reuses_reqs_and_compacts_eos_branches(self):
        class FakeReqPool:
            def __init__(self):
                self.req_to_token = torch.zeros(4, 32, dtype=torch.long)
                self.free_count = 0

            def free(self, req):
                self.free_count += 1
                req.req_pool_idx = None

        class FakeKVAllocator:
            def __init__(self):
                self.size = 1024
                self.freed = []

            def free(self, indices):
                self.freed.append(indices.clone())

        class FakeBatch:
            def __init__(self, reqs, seq_lens, req_pool, allocate_page):
                self.reqs = reqs
                self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.long)
                self.req_pool = req_pool
                self.allocate_page = allocate_page
                self.output_ids = None
                self.decode_calls = 0

            def prepare_for_decode(self):
                self.decode_calls += 1
                for index, req in enumerate(self.reqs):
                    seq_len = int(self.seq_lens_cpu[index])
                    self.req_pool.req_to_token[req.req_pool_idx, seq_len] = (
                        self.allocate_page()
                    )
                self.seq_lens_cpu += 1

            def filter_batch(self, keep_indices):
                self.reqs = [self.reqs[index] for index in keep_indices]
                self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]

        target = object.__new__(SGLangFlashMTPTargetModel)
        req_pool = FakeReqPool()
        kv_allocator = FakeKVAllocator()
        target.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=4),
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=kv_allocator,
        )
        seen_prefixes = []
        next_kv = 100
        prepare_count = 0
        forwarded_req_ids = []

        def allocate_page():
            nonlocal next_kv
            page = next_kv
            next_kv += 1
            return page

        def fake_prepare_extend(this, reqs, *, tree_cache):
            nonlocal prepare_count
            prepare_count += 1
            round_prefixes = []
            seq_lens = []
            for slot, req in enumerate(reqs, start=1):
                req.req_pool_idx = slot
                prefix = req.prefix_indices.clone()
                round_prefixes.append(prefix.tolist())
                req_pool.req_to_token[slot, : len(prefix)] = prefix
                req_pool.req_to_token[slot, len(prefix)] = allocate_page()
                seq_lens.append(len(prefix) + 1)
            seen_prefixes.append(round_prefixes)
            return FakeBatch(reqs, seq_lens, req_pool, allocate_page)

        def fake_forward(this, batch, *, capture_full):
            forwarded_req_ids.append([id(req) for req in batch.reqs])
            return SimpleNamespace(
                last_hidden_states=torch.ones(
                    len(batch.reqs), 4, dtype=torch.bfloat16
                )
            )

        target._prepare_extend_batch = MethodType(fake_prepare_extend, target)
        target._forward_prepared_batch = MethodType(fake_forward, target)
        target._prepare_mlp_sync = MethodType(lambda this, batch: None, target)
        handle = TempRolloutPrefillHandle(
            target_model=target,
            parent_reqs=[],
            true_token_ids=[[10, 11, 12, 13, 14, 15]],
            prefix_indices=[torch.tensor([1, 2, 3, 4, 5, 6])],
            tree_cache=object(),
        )
        anchors = torch.tensor([[1, 3]])

        first_hidden = target._temp_rollout_extend_step(
            handle,
            anchors,
            torch.tensor([[[21], [31]]]),
            torch.tensor([[True, True]]),
        )
        self.assertEqual(tuple(first_hidden.shape), (1, 2, 4))
        self.assertEqual(prepare_count, 1)
        self.assertEqual(req_pool.free_count, 0)
        self.assertEqual(seen_prefixes[0], [[1, 2], [1, 2, 3, 4]])
        persistent_req_ids = forwarded_req_ids[0]

        target._temp_rollout_extend_step(
            handle,
            anchors,
            torch.tensor([[[21, 22], [31, 32]]]),
            torch.tensor([[True, True]]),
        )
        self.assertEqual(prepare_count, 1)
        self.assertEqual(req_pool.free_count, 0)
        self.assertEqual(handle.branch_batch.decode_calls, 1)
        self.assertEqual(forwarded_req_ids[1], persistent_req_ids)

        target._temp_rollout_extend_step(
            handle,
            anchors,
            torch.tensor([[[21, 22, 23], [31, 32, 33]]]),
            torch.tensor([[True, False]]),
        )
        self.assertEqual(prepare_count, 1)
        self.assertEqual(req_pool.free_count, 1)
        self.assertEqual(len(kv_allocator.freed), 1)
        # Removed branch owned z1 and z2 pages; its true prefix was shared.
        self.assertEqual(kv_allocator.freed[0].numel(), 2)
        self.assertEqual(forwarded_req_ids[2], [persistent_req_ids[0]])
        self.assertEqual(handle.branch_batch.decode_calls, 2)


if __name__ == "__main__":
    unittest.main()
