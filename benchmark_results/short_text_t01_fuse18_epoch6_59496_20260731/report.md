# Short-text benchmark: fuse18 epoch 6 step 59496

## Settings

- Target: `Qwen3-8B`
- Draft checkpoint: `flashmtp_h100_prefix_condition_fuse18_sample_pb_80k_nlayers5_block_16_mhrnn_easy_direct_r512_ce1.0_tv0.0_wb_0.2_bgemma_21_maxlen4096_epochs6_Qwen3-8B/epoch_6_step_59496`
- Maximum samples per dataset: 50 (AIME25 has 30 samples)
- Maximum new tokens: 512
- Batch size: 1
- Draft/verify block: 16/16
- Temperature 0: greedy match verification
- Temperature 1: stochastic rejection verification
- `compile_serial_head=false`
- GPUs: idle H800 GPUs 1–7

## Results

Speedup is token-weighted baseline decode seconds/token divided by FlashMTP decode seconds/token. Accept is the mean number of committed tokens per speculative step.

| Dataset | T=0 speedup | T=0 accept | T=1 speedup | T=1 accept | T=1 speedup delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 3.69x | 4.83 | 3.17x | 4.40 | -14.1% |
| Alpaca | 1.86x | 2.45 | 1.63x | 2.26 | -12.4% |
| MT-Bench | 1.99x | 2.61 | 1.73x | 2.40 | -13.1% |
| Math500 | 3.73x | 4.91 | 3.04x | 4.23 | -18.5% |
| MBPP | 2.94x | 3.86 | 2.49x | 3.47 | -15.3% |
| AIME25 | 2.95x | 3.88 | 2.41x | 3.35 | -18.3% |
| LiveCodeBench | 2.65x | 3.48 | 2.34x | 3.25 | -11.7% |
| **Macro mean** | **2.83x** | **3.72** | **2.40x** | **3.34** | **-15.1%** |

## Validation

- All 14 dataset/temperature jobs completed successfully.
- Every log contains one top-level `Overall` result block.
- No traceback, CUDA OOM, or runtime error was found.
- GPU 0 was left untouched; the benchmark used GPUs 1–7.
- MT-Bench contains two turns per selected prompt, so 50 samples produce 100 measured turns.
