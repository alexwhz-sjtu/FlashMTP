"""Compare production cached teacher decoding with training path over many steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import load_benchmark_dataset, select_max_samples
from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.modeling.draft.flashmtp import (
    FlashMTPDraftModel,
    gather_pivot_multilayer_inference,
    sample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda:0")

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()
    draft_train = FlashMTPDraftModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flex_attention"
    ).to(device).eval()
    draft_inf = FlashMTPDraftModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    item = select_max_samples(load_benchmark_dataset("gsm8k"), 1)[0]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["turns"][0]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        reference_ids = target.generate(
            prompt_ids, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        reference = target(reference_ids, output_hidden_states=True, use_cache=False)
        loss_mask = torch.ones_like(reference_ids, dtype=torch.float32)
        online = OnlineFlashMTPModel(
            draft_model=draft_train,
            target_lm_head=target.lm_head,
            target_embed_tokens=target.model.embed_tokens,
            mask_token_id=draft_train.mask_token_id,
            block_size=draft_train.block_size,
            num_anchors=1,
        )

        cache = DynamicCache()
        positions = torch.arange(reference_ids.shape[1], device=device).unsqueeze(0)
        out = target(
            prompt_ids, position_ids=positions[:, : prompt_ids.shape[1]],
            past_key_values=cache, use_cache=True, logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids = torch.full_like(reference_ids, draft_inf.mask_token_id)
        start = prompt_ids.shape[1]
        output_ids[:, :start] = prompt_ids
        output_ids[:, start:start+1] = sample(out.logits, 0.0)
        target_hidden = gather_pivot_multilayer_inference(
            out.hidden_states, draft_inf.target_layer_ids, -1,
            draft_inf.config.num_target_layers,
        )
        recent = draft_inf.initialize_inference_condition(out.hidden_states)
        train_prefixes: list[int] = []
        inference_prefixes: list[int] = []
        proposal_matches: list[bool] = []

        for step in range(args.steps):
            if start + draft_inf.block_size >= reference_ids.shape[1]:
                break
            token_group_start = max(0, start - draft_inf.anchor_group_size + 1)
            token_group = output_ids[:, token_group_start:start+1]
            draft_input = torch.full(
                (1, draft_inf.block_size), draft_inf.mask_token_id,
                device=device, dtype=torch.long,
            )
            draft_input[:, 0] = output_ids[:, start]
            noise = draft_inf.build_inference_query_embeddings(
                target.model.embed_tokens, draft_input, token_group_ids=token_group
            )
            current = draft_inf.build_inference_current_chs(
                target.model.embed_tokens, target_hidden, output_ids[:, start-1:start]
            )
            history, ctx_pos, q_pos = draft_inf.build_inference_context(
                recent, current, start, token_group_length=token_group.shape[1]
            )
            inf_block = draft_inf(
                target_hidden=current, history_hidden=history,
                noise_embedding=noise, position_ids=q_pos,
                rotary_position_ids=torch.cat([ctx_pos, q_pos], dim=-1),
                is_causal=False,
            )
            inf_hidden = draft_inf._prediction_hidden(inf_block)

            anchor = torch.tensor([[start]], device=device)
            keep = torch.ones_like(anchor, dtype=torch.bool)
            train_batch = online.prepare_batch(
                reference_ids, reference.hidden_states, loss_mask,
                anchor_positions=anchor, block_keep_mask=keep,
            )
            train_hidden = online.forward_backbone(
                train_batch, seq_len=reference_ids.shape[1]
            )[:, 0]
            delta = (inf_hidden.float() - train_hidden.float()).abs()
            cosine = torch.nn.functional.cosine_similarity(
                inf_hidden.float().flatten(), train_hidden.float().flatten(), dim=0
            )

            proposals, _ = draft_inf.sample_draft_tokens(
                draft_hidden=inf_hidden, lm_head=target.lm_head,
                first_prev_token_ids=draft_input[:, 0], temperature=0.0,
                initial_prev_token_ids=output_ids[:, start-1],
            )
            train_proposals, _ = draft_train.sample_draft_tokens(
                draft_hidden=train_hidden,
                lm_head=target.lm_head,
                first_prev_token_ids=train_batch.prev_token_ids[:, 0, 0],
                temperature=0.0,
                initial_prev_token_ids=train_batch.initial_prev_token_ids[:, 0],
            )
            train_correct = train_proposals.eq(train_batch.labels[:, 0])
            train_prefix = int(train_correct.cumprod(1).sum().item()) + 1
            proposal_match = bool(torch.equal(train_proposals, proposals))
            verify_ids = torch.cat([draft_input[:, :1], proposals], dim=1)
            verify = target(
                verify_ids, position_ids=positions[:, start:start+draft_inf.block_size],
                past_key_values=cache, use_cache=True, output_hidden_states=True,
            )
            posterior = sample(verify.logits, 0.0)
            accepted = int(
                (verify_ids[:, 1:] == posterior[:, :-1]).cumprod(1).sum().item()
            )
            next_token = posterior[:, accepted]
            train_prefixes.append(train_prefix)
            inference_prefixes.append(accepted + 1)
            proposal_matches.append(proposal_match)
            output_ids[:, start:start+accepted+1] = verify_ids[:, :accepted+1]
            output_ids[:, start+accepted+1] = next_token
            print(
                f"step={step} anchor={start} train_prefix={train_prefix} "
                f"inference_prefix={accepted+1} proposals_equal={proposal_match} "
                f"prefix_matches_reference={bool(torch.equal(output_ids[:, :start+accepted+2], reference_ids[:, :start+accepted+2]))} "
                f"max_abs={delta.max().item():.6f} mean_abs={delta.mean().item():.6f} "
                f"cos={cosine.item():.8f}"
            )
            start += accepted + 1
            cache.crop(start)
            pivot_index = accepted
            recent = draft_inf.update_inference_condition(
                recent, verify.hidden_states, pivot_index
            )
            target_hidden = gather_pivot_multilayer_inference(
                verify.hidden_states, draft_inf.target_layer_ids, pivot_index,
                draft_inf.config.num_target_layers,
            )

        print(
            f"summary steps={len(train_prefixes)} "
            f"train_prefix_mean={sum(train_prefixes) / max(len(train_prefixes), 1):.6f} "
            f"inference_prefix_mean={sum(inference_prefixes) / max(len(inference_prefixes), 1):.6f} "
            f"proposal_exact_rate={sum(proposal_matches) / max(len(proposal_matches), 1):.6f}"
        )


if __name__ == "__main__":
    main()
