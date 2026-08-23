"""Measure teacher-forced versus free-running draft prefixes on target-greedy text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmark import load_benchmark_dataset, select_max_samples
from specforge.core.flashmtp import OnlineFlashMTPModel, gather_target_prefill_logits
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


def prefix_lengths(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (predictions == labels).cumprod(dim=-1).sum(dim=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument(
        "--training-jsonl-stdin",
        action="store_true",
        help="Read one training JSONL record from stdin instead of target-greedy generation.",
    )
    parser.add_argument(
        "--tokenized-jsonl-stdin",
        action="store_true",
        help="Read one JSON object containing input_ids and loss_mask from stdin.",
    )
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    draft = FlashMTPDraftModel.from_pretrained(
        args.draft,
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if args.tokenized_jsonl_stdin:
        record = json.loads(sys.stdin.readline())
        sequence = torch.tensor(record["input_ids"], device=device).view(1, -1)
        loss_mask = torch.tensor(record["loss_mask"], device=device).float().view(1, -1)
        valid_indices = loss_mask[0].nonzero(as_tuple=False).flatten()
        prompt_length = int(valid_indices[0].item()) if valid_indices.numel() else 0
        prompt_ids = sequence[:, :prompt_length]
    elif args.training_jsonl_stdin:
        record = json.loads(sys.stdin.readline())
        conversations = record["conversations"]
        processed = preprocess_conversations(
            tokenizer, [conversations], TEMPLATE_REGISTRY.get("qwen"), max_length=2048
        )
        sequence = processed["input_ids"][0].to(device)
        loss_mask = processed["loss_mask"][0].float().to(device)
        valid_indices = loss_mask[0].nonzero(as_tuple=False).flatten()
        prompt_length = int(valid_indices[0].item()) if valid_indices.numel() else 0
        prompt_ids = sequence[:, :prompt_length]
    else:
        item = select_max_samples(load_benchmark_dataset(args.dataset), 1)[0]
        messages = [{"role": "user", "content": item["turns"][0]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        if not args.training_jsonl_stdin and not args.tokenized_jsonl_stdin:
            sequence = target.generate(
                prompt_ids,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        target_output = target(sequence, output_hidden_states=True, use_cache=False)

        first_anchor = prompt_ids.shape[1]
        last_anchor = sequence.shape[1] - draft.block_size
        count = min(args.anchors, max(last_anchor - first_anchor + 1, 0))
        if count < 1:
            raise ValueError("Generated sequence is too short for an anchor block")
        anchors_1d = torch.linspace(
            first_anchor,
            last_anchor,
            steps=count,
            device=device,
        ).round().long().unique(sorted=True)
        anchors = anchors_1d.unsqueeze(0)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        if not args.training_jsonl_stdin and not args.tokenized_jsonl_stdin:
            loss_mask = torch.ones_like(sequence, dtype=torch.float32)

        online = OnlineFlashMTPModel(
            draft_model=draft,
            target_lm_head=target.lm_head,
            target_embed_tokens=target.model.embed_tokens,
            mask_token_id=draft.mask_token_id,
            block_size=draft.block_size,
            num_anchors=anchors.shape[1],
        )
        batch = online.prepare_batch(
            sequence,
            target_output.hidden_states,
            loss_mask,
            anchor_positions=anchors,
            block_keep_mask=keep,
        )
        hidden = online.forward_backbone(batch, seq_len=sequence.shape[1])
        markov = draft.markov_head
        assert markov is not None

        teacher_latent = markov.forward_teacher_forcing(
            hidden_states=hidden,
            prev_token_ids=batch.prev_token_ids,
            output_mode=draft.markov_output_mode,
            initial_prev_token_ids=batch.initial_prev_token_ids,
        )
        teacher_predictions = markov.project_logits(teacher_latent).argmax(dim=-1)

        n = hidden.shape[1]
        free_predictions, _ = draft.sample_draft_tokens(
            draft_hidden=hidden.reshape(n, draft.proposal_length, -1),
            lm_head=target.lm_head,
            first_prev_token_ids=batch.prev_token_ids[0, :, 0],
            temperature=0.0,
            initial_prev_token_ids=batch.initial_prev_token_ids[0],
        )
        labels = batch.labels[0]
        teacher_prefix = prefix_lengths(teacher_predictions[0], labels)
        free_prefix = prefix_lengths(free_predictions, labels)
        target_predictions = gather_target_prefill_logits(
            target_output.logits, anchors, draft.block_size
        )[0].argmax(dim=-1)
        target_prefix = prefix_lengths(target_predictions, labels)
        draft_vs_target_prefix = prefix_lengths(free_predictions, target_predictions)
        base_predictions = target.lm_head(hidden[0]).argmax(dim=-1)
        base_prefix = prefix_lengths(base_predictions, labels)
        additive_predictions = (
            target.lm_head(hidden[0]).float()
            + markov.project_logits(teacher_latent[0]).float()
        ).argmax(dim=-1)
        additive_prefix = prefix_lengths(additive_predictions, labels)

    print(
        f"dataset={args.dataset} prompt_tokens={prompt_ids.shape[1]} "
        f"sequence_tokens={sequence.shape[1]} anchors={n}"
    )
    print("teacher_forced_position_accuracy", (teacher_predictions[0] == labels).float().mean(dim=0).cpu().tolist())
    print("free_running_position_accuracy", (free_predictions == labels).float().mean(dim=0).cpu().tolist())
    print(f"teacher_forced_prefix_mean={teacher_prefix.float().mean().item():.6f}")
    print(f"free_running_prefix_mean={free_prefix.float().mean().item():.6f}")
    print(f"target_vs_labels_prefix_mean={target_prefix.float().mean().item():.6f}")
    print(
        f"draft_vs_target_prefix_mean="
        f"{draft_vs_target_prefix.float().mean().item():.6f}"
    )
    print(f"base_only_prefix_mean={base_prefix.float().mean().item():.6f}")
    print(f"base_plus_markov_prefix_mean={additive_prefix.float().mean().item():.6f}")
    with torch.inference_mode():
        for alpha in (0.01, 0.03, 0.1, 0.3, 1.0):
            blended = (
                markov.project_logits(teacher_latent[0]).float()
                + alpha * target.lm_head(hidden[0]).float()
            ).argmax(dim=-1)
            blended_prefix = prefix_lengths(blended, labels)
            print(
                f"direct_plus_{alpha:g}_base_prefix_mean="
                f"{blended_prefix.float().mean().item():.6f}"
            )
    print("teacher_forced_prefixes", teacher_prefix.cpu().tolist())
    print("free_running_prefixes", free_prefix.cpu().tolist())


if __name__ == "__main__":
    main()
