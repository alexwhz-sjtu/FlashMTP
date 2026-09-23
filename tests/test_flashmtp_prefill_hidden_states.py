import unittest
from unittest import mock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
)


class FlashMTPPrefillHiddenStatesTest(unittest.TestCase):
    @torch.no_grad()
    def test_each_acceptance_length_keeps_training_layer_token_and_cache_alignment(self):
        from specforge.core.flashmtp import prepare_target_hidden
        from specforge.modeling.target.flashmtp_target_model import HFFlashMTPTargetModel

        torch.manual_seed(42)
        target_config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, attention_dropout=0.0,
        )
        target_config._attn_implementation = "eager"
        target = Qwen3ForCausalLM(target_config).eval()
        prompt = torch.tensor([[2, 5, 7, 11, 13]])
        reference = prompt.clone()
        # A deterministic target-greedy trajectory long enough for eight rounds.
        for _ in range(44):
            logits = target(reference, use_cache=False).logits
            reference = torch.cat([reference, logits[:, -1].argmax(-1)[:, None]], -1)
        used_tokens = set(reference.flatten().tolist())
        mask_token_id = next(i for i in range(32) if i not in used_tokens)
        train_output = HFFlashMTPTargetModel(target).generate_flashmtp_data(
            reference, torch.ones_like(reference), torch.ones_like(reference)
        )

        for conv_enabled in (False, True):
            with self.subTest(conv_enabled=conv_enabled):
                config = Qwen3Config(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=8, attention_dropout=0.0,
                )
                config._attn_implementation = "eager"
                config.num_target_layers = 3
                config.block_size = 8
                config.flashmtp_config = {
                    "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
                    "sliding_window_size": 1, "chs_num_layers": 3,
                    "target_layer_ids": [0, 1, 2], "mask_token_id": mask_token_id,
                    "backbone_conv_enabled": conv_enabled,
                    "conv_kernel_size": 2, "conv_group_size": 4,
                    "markov_head_type": "none",
                }
                draft = FlashMTPDraftModel(config).eval()
                records, cache_lengths = [], []
                schedule = list(range(8))
                expected_anchors = []
                a = prompt.shape[1]
                for k in schedule:
                    expected_anchors.append(a)
                    a += k + 1

                def capture_draft(_module, _args, kwargs):
                    records.append({key: kwargs[key].clone() for key in (
                        "target_hidden", "position_ids", "rotary_position_ids"
                    )})

                def capture_target(_module, _args, kwargs):
                    cache_lengths.append(kwargs["past_key_values"].get_seq_length())

                def proposals(**kwargs):
                    r = len(records) - 1
                    anchor, k = expected_anchors[r], schedule[r]
                    self.assertTrue(torch.equal(
                        kwargs["first_prev_token_ids"], reference[:, anchor]
                    ))
                    tokens = reference[:, anchor+1:anchor+8].clone()
                    if k < 7:
                        # The first wrong proposal enforces exactly k accepted drafts.
                        correct = tokens[0, k].item()
                        tokens[0, k] = next(i for i in range(32)
                                            if i not in (correct, mask_token_id))
                    return tokens, torch.zeros(1, 7, 32)

                handles = [draft.register_forward_pre_hook(capture_draft, with_kwargs=True),
                           target.register_forward_pre_hook(capture_target, with_kwargs=True)]
                try:
                    with mock.patch.object(draft, "sample_draft_tokens", side_effect=proposals):
                        generated = draft.spec_generate(
                            target=target, input_ids=prompt, max_new_tokens=36,
                            temperature=0.0, stop_token_ids=None, verify_block_size=8,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(draft.get_last_decode_stats()["accept_lengths"], list(range(1, 9)))
                self.assertEqual(cache_lengths, [0] + expected_anchors)
                torch.testing.assert_close(generated, reference[:, :prompt.shape[1]+36])
                self.assertEqual(len(records), 8)
                for r, anchor in enumerate(expected_anchors):
                    expected = prepare_target_hidden(
                        train_output.hidden_states, torch.tensor([[anchor]]), [0, 1, 2], 3
                    )
                    torch.testing.assert_close(records[r]["target_hidden"], expected, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(records[r]["position_ids"], torch.arange(anchor, anchor+8)[None])
                    torch.testing.assert_close(records[r]["rotary_position_ids"][:, :3], torch.full((1,3), anchor-1))

    @torch.no_grad()
    def test_prefill_chs_matches_training_hidden_states_including_final_norm(self):
        torch.manual_seed(42)
        target_config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, attention_dropout=0.0,
        )
        target_config._attn_implementation = "eager"
        target = Qwen3ForCausalLM(target_config).eval()
        input_ids = torch.tensor([[2, 5, 7, 11, 13]])
        expected_output = target(input_ids, output_hidden_states=True, use_cache=False)
        expected = gather_pivot_multilayer_inference(
            expected_output.hidden_states, [0, 1], -1, 2
        )

        for conv_enabled in (False, True):
            with self.subTest(conv_enabled=conv_enabled):
                config = Qwen3Config(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=8, attention_dropout=0.0,
                )
                config._attn_implementation = "eager"
                config.num_target_layers = 2
                config.block_size = 4
                config.flashmtp_config = {
                    "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
                    "sliding_window_size": 1, "chs_num_layers": 2,
                    "target_layer_ids": [0, 1], "mask_token_id": 31,
                    "backbone_conv_enabled": conv_enabled,
                    "conv_kernel_size": 2, "conv_group_size": 4,
                    "markov_head_type": "none",
                }
                draft = FlashMTPDraftModel(config).eval()
                captured = []

                def capture(_module, _args, kwargs):
                    captured.append(kwargs["target_hidden"].clone())

                handle = draft.register_forward_pre_hook(capture, with_kwargs=True)
                try:
                    draft.spec_generate(
                        target=target, input_ids=input_ids,
                        max_new_tokens=1, temperature=0.0, stop_token_ids=None,
                    )
                finally:
                    handle.remove()
                self.assertEqual(len(captured), 1)
                torch.testing.assert_close(captured[0], expected)


if __name__ == "__main__":
    unittest.main()
