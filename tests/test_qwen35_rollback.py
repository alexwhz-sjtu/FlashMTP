"""CPU float32 checks against independent token-by-token Qwen3.5 execution."""
import copy
import unittest
from unittest import mock

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from evaluation.qwen35_target import install_qwen35_rollback


class Qwen35RollbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(912)
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            layer_types=['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]},
        )
        config._attn_implementation = 'eager'
        with mock.patch.object(hf, 'FusedRMSNormGated', None):
            cls.model = install_qwen35_rollback(Qwen3_5ForCausalLM(config).float().eval())
        for layer in cls.model.model.layers:
            if layer.layer_type == 'linear_attention':
                m = layer.linear_attn
                m.causal_conv1d_fn = None
                m.causal_conv1d_update = hf.torch_causal_conv1d_update
                m.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
                m.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule

    @torch.inference_mode()
    def test_all_prefixes_against_serial_with_repeated_rollback(self):
        for batch in (1, 2):
            reference = self.model.make_inference_cache()
            prompt = torch.randint(0, 128, (batch, 11))
            self.model(prompt, past_key_values=reference, use_cache=True)
            tested = copy.deepcopy(reference)
            for accepted in (0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 7, 1):
                with self.subTest(batch=batch, accepted=accepted):
                    tokens = torch.randint(0, 128, (batch, 8))
                    start = reference.get_seq_length()
                    block = self.model(tokens, past_key_values=tested, use_cache=True).logits
                    for index in range(accepted):
                        serial = self.model(tokens[:, index:index+1], past_key_values=reference, use_cache=True).logits
                        torch.testing.assert_close(block[:, index:index+1], serial, atol=1e-5, rtol=1e-4)
                    tested.crop(start + accepted)
                    self.assertEqual(tested.get_seq_length(), reference.get_seq_length())
                    for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
                        for a, b in zip(getattr(tested, name), getattr(reference, name)):
                            if a is None:
                                self.assertIsNone(b)
                            else:
                                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                    self.assertFalse(tested.pending)

    @torch.inference_mode()
    def test_cannot_crop_outside_saved_block(self):
        cache = self.model.make_inference_cache()
        self.model(torch.ones((1, 5), dtype=torch.long), past_key_values=cache, use_cache=True)
        with self.assertRaises(ValueError):
            cache.crop(4)
        self.model(torch.ones((1, 8), dtype=torch.long), past_key_values=cache, use_cache=True)
        for length in (-1, 4, 14):
            with self.assertRaises(ValueError):
                cache.crop(length)


if __name__ == '__main__':
    unittest.main()
