import unittest

try:
    import torch
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"PyTorch runtime is unavailable: {exc}") from exc

from specforge.modeling.draft.sequential_head import (
    DLiteSequentialHead,
    SEQUENTIAL_HEAD_TYPES,
)


class SequentialHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.head = DLiteSequentialHead(
            head_type="rnn",
            vocab_size=17,
            sequential_rank=5,
            hidden_size=7,
            max_prediction_length=4,
        )

    def test_only_rnn_is_public(self):
        self.assertEqual(SEQUENTIAL_HEAD_TYPES, ("rnn",))
        with self.assertRaises(ValueError):
            DLiteSequentialHead(
                head_type="rnn_easy",
                vocab_size=17,
                sequential_rank=5,
                hidden_size=7,
                max_prediction_length=4,
            )

    def test_teacher_forcing_shapes(self):
        hidden = torch.randn(2, 3, 7)
        previous = torch.randint(0, 17, (2, 3))
        latent = self.head.forward_teacher_forcing(
            hidden_states=hidden,
            prev_token_ids=previous,
        )
        self.assertEqual(latent.shape, (2, 3, 5))
        self.assertEqual(self.head.project_logits(latent).shape, (2, 3, 17))

    def test_greedy_sampling_shapes(self):
        tokens, logits = self.head.sample_block_tokens(
            hidden_states=torch.randn(2, 4, 7),
            first_prev_token_ids=torch.tensor([1, 2]),
        )
        self.assertEqual(tokens.shape, (2, 4))
        self.assertEqual(logits.shape, (2, 4, 17))


if __name__ == "__main__":
    unittest.main()
