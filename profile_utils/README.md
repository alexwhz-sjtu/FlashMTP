# FlashMTP profile_utils

## Modes (`spec_profile.py`)

| Mode | Output | Description |
|------|--------|-------------|
| `jsonl` (default) | `profile_utils/log/spec_profile_<dataset>_n<N>.jsonl` | Per-block accept length + per-slot draft top-k (reuses `spec_generate_with_profile`). |
| `profile_time` | `profile_utils/log/spec_profile_time_<dataset>.json` | CUDA-event GPU ms: **target verify avg** and **draft forward avg** per batch size. |
| `profile_token` | `profile_utils/log/spec_profile_token_<dataset>_n<N>.json` | Per-sample compact JSON (`lines`: one string per slot/step; no abs_pos / sampled_token / block_start). |

## Examples

From **FlashMTP_v1.1** root:

```bash
# JSONL (existing behavior)
torchrun --nproc_per_node=1 profile_utils/spec_profile.py \
  --draft-name-or-path /path/to/draft \
  --dataset alpaca --max-samples 2 --profile-mode jsonl

# Token-level log
torchrun --nproc_per_node=1 profile_utils/spec_profile.py \
  --draft-name-or-path /path/to/draft \
  --dataset alpaca --max-samples 1 --profile-mode profile_token \
  --max-new-tokens 64

# Timing sweep (single GPU, no torchrun required)
CUDA_VISIBLE_DEVICES=3 python profile_utils/spec_profile.py \
  --draft-name-or-path /data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP_v1.1/cache/models/flashmtp_qz_prefix_condition_fuse_middle_16_feature_sample_900000_think_off_nlayers5_block_12_gamma_8_maxlen4096_epochs8_tlmh0_lp1 \
  --dataset alpaca \
  --profile-mode profile_time \
  --batch-sizes 1,32,64,128 --max-new-tokens 512 --temperature 0
```

`profile_time` replicates the same prompt on the batch dimension (greedy). Long-context datasets typically use `--batch-sizes 1` only.

## Modules

- `flashmtp_cuda_profile.py` — CUDA-event loop aligned with `spec_generate` (see `dflash/profile_utils/dflash_cuda_profile.py`).
- `flashmtp_profile_format.py` — formats `profile_records` from `spec_generate_with_profile`.
