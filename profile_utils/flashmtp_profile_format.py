"""Compact line formatting for ``spec_generate_with_profile`` records."""

from __future__ import annotations

from typing import Iterator


def iter_profile_blocks(records: list[dict]) -> Iterator[tuple[dict, list[dict]]]:
    """Yield ``(block_summary, slot_rows)`` from flat ``profile_records``."""
    i = 0
    n = len(records)
    while i < n:
        row = records[i]
        if "block_start" in row and "accept_length" in row:
            block = row
            slots: list[dict] = []
            i += 1
            while i < n and "slot" in records[i] and "block_start" not in records[i]:
                slots.append(records[i])
                i += 1
            yield block, slots
        else:
            i += 1


def _fmt_topk(candidates: list[dict]) -> str:
    parts = []
    for c in candidates:
        tok = c.get("t", c.get("token", ""))
        p = c.get("p", c.get("confidence"))
        if p is not None:
            parts.append(f"{tok!r}({float(p):.3f})")
        else:
            parts.append(f"{tok!r}")
    return ", ".join(parts)


def compact_profile_token_lines(records: list[dict]) -> list[str]:
    """One line per spec step header, slot, or target verify; ``*`` on last accepted slot."""
    lines: list[str] = []
    for step_idx, (block, slots) in enumerate(iter_profile_blocks(records)):
        accept_len = int(block["accept_length"])
        last_accept_slot = max(accept_len - 1, 0)
        lines.append(f"step{step_idx} accept_length={accept_len}")

        for slot_row in sorted(slots, key=lambda r: int(r["slot"])):
            slot = int(slot_row["slot"])
            star = "*" if slot == last_accept_slot else ""
            if slot == 0:
                tok = slot_row.get("token", "")
                conf = slot_row.get("confidence")
                conf_s = f" conf={conf:.4f}" if conf is not None else ""
                lines.append(
                    f"step{step_idx} slot{slot} anchor {tok!r}{conf_s}{star}"
                )
                continue

            topk = slot_row.get("candidates", [])
            k = len(topk) if topk else int(slot_row.get("top_k", 0))
            topk_s = _fmt_topk(topk) if topk else ""
            lines.append(f"step{step_idx} slot{slot} draft top{k}={topk_s}{star}")

        for tv in block.get("target_verify") or []:
            role = tv.get("role", "?")
            tok = tv.get("chosen_token", "")
            p = tv.get("target_p_chosen")
            verify_step = tv.get("verify_step")
            p_s = f" p={p}" if p is not None else ""
            lines.append(
                f"step{step_idx} target step={verify_step} {role} {tok!r}{p_s}"
            )

    return lines
