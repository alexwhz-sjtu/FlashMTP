# QZ Sample Two-Model Benchmark Report

**Run:** `qz_sample_two_models_20260731_151459`  
**Duration:** 2026-07-31 15:14:59 → 16:42:33 (~88 min)  
**Status:** 28/28 completed, 0 failed

## Configuration

| Setting | Value |
| --- | --- |
| Target model | `/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B` |
| GPUs | 4, 5, 6, 7 (one worker per GPU, no overlap) |
| max_new_tokens | 512 |
| max_samples | 50 |
| block_size / verify_block | 16 / 16 |
| T=0 verification | `match` (greedy) |
| T=1 verification | `rejection` (`--stochastic-verification-mode rejection`) |
| compile_serial_head | false |

## Models

1. **ce0.02_tv1.0** — `flashmtp_qz_sample_80000_think_off_nlayers5_block_16_mhrnn_direct_r512_ce0.02_tv1.0_maxlen4096_epochs6_Qwen3-8B`
2. **ce0.1_tv1.0_wb0.04** — `flashmtp_qz_sample_80000_think_off_nlayers5_block_16_mhrnn_direct_r512_ce0.1_tv1.0_wb0.04_maxlen4096_epochs6_Qwen3-8B`

## Results (speedup × / accept length)

Each cell: **speedup** / **accept length**

### Model: ce0.02_tv1.0

| Benchmark | T=0 speedup | T=0 accept | T=1 speedup | T=1 accept |
| --- | ---: | ---: | ---: | ---: |
| alpaca | 1.74× | 2.35 | 1.61× | 2.30 |
| mt-bench | 1.49× | 2.00 | 1.35× | 1.93 |
| gsm8k | 3.53× | 4.74 | 3.23× | 4.56 |
| math500 | 3.61× | 4.85 | 3.12× | 4.43 |
| aime25 | 2.88× | 3.88 | 2.50× | 3.55 |
| mbpp | 2.80× | 3.79 | 2.55× | 3.63 |
| livecodebench | 2.58× | 3.46 | 2.37× | 3.35 |
| **Macro mean** | **2.66×** | **3.58** | **2.39×** | **3.39** |

### Model: ce0.1_tv1.0_wb0.04

| Benchmark | T=0 speedup | T=0 accept | T=1 speedup | T=1 accept |
| --- | ---: | ---: | ---: | ---: |
| alpaca | 1.83× | 2.47 | 1.69× | 2.40 |
| mt-bench | 1.49× | 2.01 | 1.35× | 1.90 |
| gsm8k | 3.64× | 4.88 | 3.25× | 4.60 |
| math500 | 3.74× | 5.06 | 3.22× | 4.58 |
| aime25 | 2.94× | 3.95 | 2.58× | 3.67 |
| mbpp | 2.95× | 3.98 | 2.59× | 3.69 |
| livecodebench | 2.66× | 3.58 | 2.43× | 3.44 |
| **Macro mean** | **2.75×** | **3.71** | **2.44×** | **3.47** |

## Head-to-head (ce0.1 vs ce0.02)

| Benchmark | T=0 Δ speedup | T=1 Δ speedup |
| --- | ---: | ---: |
| alpaca | +0.09× | +0.08× |
| mt-bench | 0.00× | 0.00× |
| gsm8k | +0.11× | +0.02× |
| math500 | +0.13× | +0.10× |
| aime25 | +0.06× | +0.08× |
| mbpp | +0.15× | +0.04× |
| livecodebench | +0.08× | +0.06× |
| **Macro mean** | **+0.09×** | **+0.05×** |

**ce0.1_tv1.0_wb0.04** wins on macro speedup at both temperatures (+3.4% at T=0, +2.1% at T=1) with higher acceptance length across all benchmarks.

## Job schedule

4 GPU workers (round-robin: `gpu = (dataset_index + combo_index) % 4`):

| GPU | Jobs (sequential queue) |
| --- | --- |
| 4 | ce0.02 T=0: alpaca, aime25 · ce0.02 T=1: math500 · ce0.1 T=0: gsm8k, livecodebench · ce0.1 T=1: mt-bench, mbpp |
| 5 | ce0.02 T=0: mt-bench, mbpp · ce0.02 T=1: alpaca, aime25 · ce0.1 T=0: math500 · ce0.1 T=1: gsm8k, livecodebench |
| 6 | ce0.02 T=0: gsm8k, livecodebench · ce0.02 T=1: mt-bench, mbpp · ce0.1 T=0: alpaca, aime25 · ce0.1 T=1: math500 |
| 7 | ce0.02 T=0: math500 · ce0.02 T=1: gsm8k, livecodebench · ce0.1 T=0: mt-bench, mbpp · ce0.1 T=1: alpaca, aime25 |

## Command used

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
GPU_LIST=4,5,6,7 bash scripts/run_qz_sample_two_model_benchmarks.sh
```

Per-job example:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python evaluation/benchmark.py \
  --model-name-or-path /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
  --draft-name-or-path /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2/cache/models/flashmtp_qz_sample_80000_think_off_nlayers5_block_16_mhrnn_direct_r512_ce0.1_tv1.0_wb0.04_maxlen4096_epochs6_Qwen3-8B \
  --max-new-tokens 512 --max-samples 50 --dataset gsm8k \
  --batch-size 1 --block-size 16 --verify-block 16 \
  --temperature 1 --stochastic-verification-mode rejection
```

## Artifacts

- Summary CSV: `summary.csv`
- Manifest: `manifest.tsv`
- Logs: `logs/<model>/temperature_<T>/<dataset>.log`
- Worker logs: `workers/gpu_{4,5,6,7}.log`
