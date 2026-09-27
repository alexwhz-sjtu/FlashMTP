import torch

from specforge.modeling.draft.flashmtp import FlashMTPDraftModel


checkpoint = (
    "/data/wanghanzhen/FlashMTP_v2swa/cache/models/Qwen3-8B/"
    "Flashmtp_v2.3_qwen3_8b_ep3"
)
torch.manual_seed(42)
model = FlashMTPDraftModel.from_pretrained(
    checkpoint, attn_implementation="flash_attention_2", dtype=torch.bfloat16
).cuda().eval()
query_length = model.draft_query_length
context_length = model.condition_slot_count
noise = torch.randn(1, query_length, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
context = torch.randn(1, 1, context_length, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
positions = torch.arange(query_length, device="cuda").unsqueeze(0)
rotary_positions = torch.cat(
    [torch.zeros(1, context_length, dtype=torch.long, device="cuda"), positions],
    dim=1,
)


def run(backend):
    model.config._attn_implementation = backend
    with torch.inference_mode():
        result = model(
            target_hidden=context,
            noise_embedding=noise,
            position_ids=positions,
            rotary_position_ids=rotary_positions,
            is_causal=False,
        )
    return result.float()


fa2 = run("flash_attention_2")
print("fa2_finite", bool(torch.isfinite(fa2).all().item()))
print("fa2_abs_max", float(fa2.abs().max().item()))
for backend in ("flex_attention", "sdpa", "eager"):
    try:
        other = run(backend)
        diff = (fa2 - other).abs()
        print(
            backend,
            "finite", bool(torch.isfinite(other).all().item()),
            "mean_abs_diff", float(diff.mean().item()),
            "max_abs_diff", float(diff.max().item()),
            "relative_l2", float(torch.linalg.vector_norm(diff).item() / torch.linalg.vector_norm(fa2).item()),
        )
    except Exception as exc:
        print(backend, type(exc).__name__, str(exc)[:300])
