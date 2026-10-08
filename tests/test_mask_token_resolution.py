import unittest

from scripts.dlite.dlite_training import _select_mask_token_id


class MaskTokenResolutionTest(unittest.TestCase):
    def test_explicit_override_has_highest_priority(self):
        token_id, source = _select_mask_token_id(
            explicit_id=7,
            configured_ids=[8],
            tokenizer_mask_id=9,
            used_token_ids=set(range(10)),
            vocab_size=16,
        )
        self.assertEqual(token_id, 7)
        self.assertIn("explicit", source)

    def test_checkpoint_precedes_tokenizer(self):
        token_id, source = _select_mask_token_id(
            explicit_id=None,
            configured_ids=[11, 11],
            tokenizer_mask_id=12,
            used_token_ids=set(range(10)),
            vocab_size=16,
        )
        self.assertEqual(token_id, 11)
        self.assertIn("checkpoint", source)

    def test_uses_native_mask_then_first_unused_embedding_row(self):
        native_id, native_source = _select_mask_token_id(
            explicit_id=None,
            configured_ids=[],
            tokenizer_mask_id=5,
            used_token_ids={0, 1, 2, 3, 4, 5},
            vocab_size=8,
        )
        auto_id, auto_source = _select_mask_token_id(
            explicit_id=None,
            configured_ids=[],
            tokenizer_mask_id=None,
            used_token_ids={0, 1, 2, 3, 4, 5},
            vocab_size=8,
        )
        self.assertEqual((native_id, auto_id), (5, 6))
        self.assertIn("tokenizer.mask_token_id", native_source)
        self.assertIn("unused", auto_source)

    def test_rejects_conflicting_checkpoints_and_full_vocabularies(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            _select_mask_token_id(
                explicit_id=None,
                configured_ids=[6, 7],
                tokenizer_mask_id=None,
                used_token_ids=set(),
                vocab_size=8,
            )
        with self.assertRaisesRegex(ValueError, "No tokenizer-unused row"):
            _select_mask_token_id(
                explicit_id=None,
                configured_ids=[],
                tokenizer_mask_id=None,
                used_token_ids=set(range(8)),
                vocab_size=8,
            )


if __name__ == "__main__":
    unittest.main()
