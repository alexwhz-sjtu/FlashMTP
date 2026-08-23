"""Check that rolling teacher fuse history matches a full target forward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()
    draft = FlashMTPDraftModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    ids = tokenizer.encode(
        "Teacher rolling history alignment check. " * 100,
        return_tensors="pt",
    ).to(device)[:, :320]
    prefill_len = 141
    chunk_sizes = [1, 4, 2, 7, 3, 8, 1, 6, 5, 2]

    with torch.inference_mode():
        full = target(ids, output_hidden_states=True, use_cache=False)
        full_fused = draft.fuse_target_output_history(full.hidden_states)

        cache = DynamicCache()
        out = target(
            ids[:, :prefill_len],
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        recent = draft.initialize_inference_condition(out.hidden_states)
        end = prefill_len
        rows = []

        def check(label: str) -> None:
            expected = full_fused[:, max(0, end - draft.swa_window_size) : end]
            delta = (recent.float() - expected.float()).abs()
            history, positions, _ = draft.build_inference_context(
                recent,
                torch.empty(
                    1, 1, draft.chs_num_layers, draft.config.hidden_size,
                    device=device, dtype=torch.bfloat16,
                ),
                end,
            )
            expected_pos = torch.arange(
                end - history.shape[2] - 1, end - 1, device=device
            )
            rows.append(
                (label, end, recent.shape[1], history.shape[2],
                 delta.max().item(), delta.mean().item(),
                 bool(torch.equal(positions[0, : history.shape[2]], expected_pos)))
            )

        check("prefill")
        for step, size in enumerate(chunk_sizes, 1):
            chunk = ids[:, end : end + size]
            if chunk.shape[1] == 0:
                break
            out = target(
                chunk,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            end += chunk.shape[1]
            recent = draft.update_inference_condition(
                recent, out.hidden_states, chunk.shape[1] - 1
            )
            check(f"step{step}")

    print(
        f"swa_window_size={draft.swa_window_size} "
        f"fuse_slots_used={draft.fuse_slot_count} chs_slots={draft.chs_num_layers}"
    )
    for label, pos, kept, used, max_abs, mean_abs, pos_ok in rows:
        print(
            f"{label}: end={pos} retained={kept} passed_to_draft={used} "
            f"max_abs={max_abs:.8f} mean_abs={mean_abs:.8f} positions_ok={pos_ok}"
        )


if __name__ == "__main__":
    main()
