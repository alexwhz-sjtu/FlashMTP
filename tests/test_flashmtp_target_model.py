import unittest
from unittest import mock

import torch
from torch import nn

from specforge.modeling.target.flashmtp_target_model import (
    HFFlashMTPTargetModel,
    SGLangFlashMTPTargetModel,
)
from specforge.modeling.target.target_utils import (
    SGLangTPEmbeddingAdapter,
    SGLangTPLMHeadAdapter,
)


class _Output:
    def __init__(self, hidden_states, logits=None):
        self.hidden_states = hidden_states
        self.logits = logits


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        hidden = torch.zeros(input_ids.size(0), input_ids.size(1), 4)
        return _Output((hidden, hidden + 1, hidden + 2))


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        output = self.model(input_ids, **kwargs)
        output.logits = torch.zeros(input_ids.size(0), input_ids.size(1), 7)
        return output


class _FakeVocabParallelEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(4, 2)
        self.org_vocab_size = 4
        self.embedding_dim = 2


class _FakeVocabParallelLMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32).view(2, 2))
        self.embedding_dim = 2
        self.org_vocab_size = 4

    def get_sharded_to_full_mapping(self):
        return None


class FlashMTPTargetModelCaptureTest(unittest.TestCase):
    def test_sglang_mask_uses_the_in_vocab_embedding_row(self):
        embedding = _FakeVocabParallelEmbedding()
        with torch.no_grad():
            embedding.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            )

        def all_gather(outputs, value, group=None):
            outputs[0].copy_(value)

        with mock.patch("torch.distributed.get_rank", return_value=0), mock.patch(
            "torch.distributed.get_world_size", return_value=1
        ), mock.patch("torch.distributed.all_gather", side_effect=all_gather):
            adapter = SGLangTPEmbeddingAdapter(embedding, object(), mask_token_id=3)
            output = adapter(torch.tensor([[1, 3]]))

        self.assertEqual(adapter.num_embeddings, 4)
        self.assertTrue(torch.equal(output[0, 0], embedding.weight[1]))
        self.assertTrue(torch.equal(output[0, 1], embedding.weight[3]))
        self.assertFalse(
            torch.allclose(output[0, 1], embedding.weight.detach().mean(dim=0))
        )

    def test_sglang_vocab_row_mode_rejects_oov_mask(self):
        embedding = _FakeVocabParallelEmbedding()
        with mock.patch("torch.distributed.get_rank", return_value=0), mock.patch(
            "torch.distributed.get_world_size", return_value=1
        ):
            with self.assertRaisesRegex(ValueError, "existing SGLang embedding row"):
                SGLangTPEmbeddingAdapter(embedding, object(), mask_token_id=4)

    def test_sglang_tp_lm_head_pads_uneven_rank_rows(self):
        lm_head = _FakeVocabParallelLMHead()
        observed = {}

        def gather_counts(outputs, value, group=None):
            outputs[0].fill_(1)
            outputs[1].fill_(2)

        def gather_hidden(value, group=None):
            observed["padded_hidden"] = value.detach().clone()
            return (value, value + 1)

        def exchange_vocab(received, local_logits, group=None):
            return tuple(local_logits)

        with mock.patch("torch.distributed.get_rank", return_value=0), mock.patch(
            "torch.distributed.get_world_size", return_value=2
        ), mock.patch(
            "torch.distributed.all_gather", side_effect=gather_counts
        ), mock.patch(
            "torch.distributed.nn.functional.all_gather", side_effect=gather_hidden
        ), mock.patch(
            "torch.distributed.nn.functional.all_to_all", side_effect=exchange_vocab
        ):
            adapter = SGLangTPLMHeadAdapter(lm_head, object())
            output = adapter(torch.tensor([[3.0, 5.0]]))

        self.assertEqual(tuple(observed["padded_hidden"].shape), (2, 2))
        self.assertTrue(torch.equal(observed["padded_hidden"][1], torch.zeros(2)))
        self.assertEqual(tuple(output.shape), (1, 4))

    def test_hf_stage1_skips_causal_lm_logits_projection(self):
        causal_lm = _CausalLM()
        target = HFFlashMTPTargetModel(causal_lm)
        target.set_capture_layers([0, 1])
        ids = torch.ones(1, 3, dtype=torch.long)
        output = target.generate_flashmtp_data(
            ids, torch.ones_like(ids), torch.ones_like(ids), return_logits=False
        )
        self.assertIsNone(output.logits)
        self.assertEqual(causal_lm.calls, 0)
        self.assertEqual(causal_lm.model.calls, 1)
        self.assertEqual(set(output.hidden_states), {0, 1})

    def test_split_aux_and_last_capture_ids_partial(self):
        captured_ids = [0, 1, 17, 18, 34, 35]
        aux_ids, last_ids = SGLangFlashMTPTargetModel._split_aux_and_last_capture_ids(
            captured_ids=captured_ids,
            num_aux_layers=5,
            num_transformer_layers=36,
            has_last_hidden=True,
        )
        self.assertEqual(aux_ids, [0, 1, 17, 18, 34])
        self.assertEqual(last_ids, [35])

    def test_split_aux_and_last_capture_ids_without_final_layer(self):
        captured_ids = [0, 1, 17, 18]
        aux_ids, last_ids = SGLangFlashMTPTargetModel._split_aux_and_last_capture_ids(
            captured_ids=captured_ids,
            num_aux_layers=4,
            num_transformer_layers=36,
            has_last_hidden=True,
        )
        self.assertEqual(aux_ids, captured_ids)
        self.assertEqual(last_ids, [])

    def test_split_aux_and_last_capture_ids_full_capture(self):
        aux_ids, last_ids = SGLangFlashMTPTargetModel._split_aux_and_last_capture_ids(
            captured_ids=[],
            num_aux_layers=35,
            num_transformer_layers=36,
            has_last_hidden=True,
        )
        self.assertEqual(aux_ids, list(range(35)))
        self.assertEqual(last_ids, [35])


if __name__ == "__main__":
    unittest.main()
