import unittest

from specforge.modeling.target.flashmtp_target_model import SGLangFlashMTPTargetModel


class FlashMTPTargetModelCaptureTest(unittest.TestCase):
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
