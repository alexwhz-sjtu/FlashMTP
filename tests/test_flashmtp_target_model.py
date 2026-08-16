import unittest
from types import SimpleNamespace

import torch
from torch import nn

from specforge.modeling.target.flashmtp_target_model import (
    HFFlashMTPTargetModel,
    SGLangFlashMTPTargetModel,
)


class FlashMTPTargetModelCaptureTest(unittest.TestCase):
    def test_hf_target_output_keeps_prefill_logits(self):
        logits = torch.randn(1, 4, 11)
        layer_hidden = torch.randn(1, 4, 7)

        class FakeCausalLM(nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(
                    logits=logits,
                    hidden_states=(torch.zeros_like(layer_hidden), layer_hidden),
                )

        target = HFFlashMTPTargetModel(FakeCausalLM())
        input_ids = torch.ones(1, 4, dtype=torch.long)
        mask = torch.ones_like(input_ids)
        output = target.generate_flashmtp_data(input_ids, mask, mask)

        self.assertIs(output.logits, logits)
        self.assertEqual(output.hidden_states, (layer_hidden,))

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
