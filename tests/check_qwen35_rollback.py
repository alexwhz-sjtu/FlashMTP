"""GPU integration checks for Qwen3.5 cached verification and rollback."""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.model_loading import load_flashmtp_benchmark_models
from evaluation.benchmark import target_generate, flashmtp_generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-name-or-path', required=True)
    p.add_argument('--draft-name-or-path', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    args.block_size = None
    target, draft, tokenizer, _ = load_flashmtp_benchmark_models(args, torch.device('cuda'))
    records = []
    with torch.inference_mode():
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'What is 12 times 13? Explain briefly.'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        base = target.make_inference_cache()
        target(ids, past_key_values=base, use_cache=True, logits_to_keep=1)
        continuation = tokenizer('The answer is 156 because twelve times thirteen equals one hundred and fifty six.', return_tensors='pt').input_ids[:, :8].cuda()
        assert continuation.shape[1] == 8
        # Identical block math, forced every possible acceptance length, then compare
        # to a fresh cache extended only by the retained prefix. This detects stale
        # conv/recurrent state even when next-token argmax happens to be unchanged.
        for accepted in range(9):
            tested = copy.deepcopy(base)
            target(continuation, past_key_values=tested, use_cache=True)
            tested.crop(ids.shape[1] + accepted)
            expected = copy.deepcopy(base)
            if accepted:
                target(continuation[:, :accepted], past_key_values=expected, use_cache=True)
            state_errors = {}
            for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
                errors = []
                for a, b in zip(getattr(tested, name), getattr(expected, name)):
                    if a is None:
                        assert b is None
                        continue
                    assert a.shape == b.shape
                    errors.append((a.float() - b.float()).abs().max().item())
                    # BF16 block-size-dependent projection rounding is expected.
                    torch.testing.assert_close(a, b, atol=0.06, rtol=0.04)
                state_errors[name] = max(errors, default=0)
            probe = continuation[:, :1]
            a = target(probe, past_key_values=tested, use_cache=True).logits
            b = target(probe, past_key_values=expected, use_cache=True).logits
            assert torch.equal(a.argmax(-1), b.argmax(-1)), f'rollback next-token mismatch at {accepted}'
            records.append({'accepted': accepted, 'max_state_errors': state_errors, 'next_token_equal': True, 'max_logit_error': (a.float()-b.float()).abs().max().item()})
            print(json.dumps(records[-1]), flush=True)
        for text in ['What is 12 times 13? Explain briefly.', 'Write a Python function that reverses a string.', '用一句话解释什么是光合作用。']:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            stop = [tokenizer.eos_token_id]
            ref = target_generate(target, ids, 64, stop, temperature=0)
            spec = flashmtp_generate(draft, target, ids, 64, draft.block_size, 8, stop, temperature=0)
            equal = torch.equal(ref.output_ids, spec.output_ids)
            item = {'prompt': text, 'greedy_equal': equal, 'baseline_tokens': ref.num_output_tokens, 'spec_tokens': spec.num_output_tokens, 'acceptance_lengths': spec.acceptance_lengths}
            records.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            if not equal:
                print('BASELINE:', tokenizer.decode(ref.output_ids[0]), flush=True)
                print('SPEC:', tokenizer.decode(spec.output_ids[0]), flush=True)
            assert equal, 'greedy output differs from autoregressive baseline'
    Path(args.output).write_text(json.dumps({'passed': True, 'checks': records}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
