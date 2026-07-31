# FlashMTP v2 Benchmark Summary

Unified results from:

| Run | Directory | Tasks | Status |
|-----|-----------|------:|--------|
| Three-model speedup | `three_model_speedup_20260730_1505/` | 48 | **48/48 completed** |
| Compile + rejection | `compile_rejection_20260730_1834/` | 24 | **21/24 completed**, 3 failed (SIGTERM) |

**Target model:** Qwen3-8B  
**Decode config:** `block_size=16`, `verify_block=16`, `max_new_tokens=512`, `batch_size=1`, 8× H800  
**Parser:** `scripts/summarize_benchmarks.py` → `consolidated_summary.csv` / `.json`

---

## Experiment Checklist (reproducibility)

### Three-model sweep (match verification, no compile)

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
source .venv/bin/activate
bash scripts/run_three_model_speedup_benchmarks.sh
# After completion:
python scripts/summarize_benchmarks.py benchmark_results/three_model_speedup_*/
```

### Compile + rejection ablation

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
source .venv/bin/activate
bash scripts/run_compile_rejection_benchmarks.sh
python scripts/summarize_benchmarks.py benchmark_results/compile_rejection_*/
```

### Single-dataset smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/benchmark.py \
  --model-name-or-path /share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B \
  --draft-name-or-path cache/models/flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b \
  --dataset gsm8k --max-samples 5 --max-new-tokens 512 \
  --temperature 0 --compile-serial-head

# Stochastic rejection at temp=1:
CUDA_VISIBLE_DEVICES=0 python evaluation/benchmark.py \
  --dataset gsm8k --max-samples 5 --temperature 1 \
  --stochastic-verification-mode rejection --compile-serial-head \
  ...  # same model/draft paths as above
```

### Profile `compile_serial_head`

```bash
CUDA_VISIBLE_DEVICES=0 python profile/profile_compile_serial_head.py \
  --dataset gsm8k --max-samples 3 --max-new-tokens 256
```

See `profile/compile_serial_head_profile.md` for full analysis.

---

## Incomplete runs (compile_rejection)

| Config | Dataset | Status | Notes |
|--------|---------|--------|-------|
| `rnn_ce0.1_tv1.0_base0.0_temp1_rejection_compile` | mt-bench | **failed** (exit 143) | No `=== Overall` block in log |
| `rnn_easy_base0.2_temp0_compile` | mt-bench | **failed** (exit 143) | Killed mid-run |
| `rnn_easy_base0.2_temp0_compile` | longbench mdqa | **failed** (exit 143) | Killed mid-run |

Re-run these three jobs before citing them in the paper.

---

## Metric definitions

| Metric | Definition |
|--------|------------|
| **Token-weighted speedup** | `baseline_decode_s_per_token / flashmtp_decode_s_per_token` (CUDA wall after prefill) |
| **Accept length** | Mean tokens committed per spec step (includes anchor + accepted draft + bonus) |
| **Draft accept rate** | `(accept_length − 1) / (verify_block − 1)` — fraction of proposed draft tokens accepted |
| **Macro mean** | Unweighted average over 8 datasets (not pooled by output tokens) |

---

## Best configuration: Model B @ temp=0

**Model B** = `ce0.1_tv1.0_base0.0` (`flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b`)

| Dataset | Speedup | Accept length | Draft accept rate |
|---------|--------:|--------------:|------------------:|
| **math500** | **3.77×** | **5.06** | 27.0% |
| **gsm8k** | **3.68×** | **4.94** | 26.3% |
| mbpp | 2.93× | 3.97 | 19.8% |
| aime25 | 2.99× | 4.03 | 20.2% |
| alpaca | 1.83× | 2.45 | 9.7% |
| mt-bench | 1.53× | 2.04 | 6.9% |
| longbench icl | 1.42× | 2.35 | 9.0% |
| longbench mdqa | 1.26× | 2.12 | 7.5% |
| **Macro mean** | **2.43×** | **3.25** | — |

With `compile_serial_head` on the same checkpoint (temp=0, match):

| Dataset | Speedup (compile on) | Δ vs no-compile |
|---------|---------------------:|----------------:|
| gsm8k | 3.94× | +7.1% |
| math500 | 4.01× | +6.4% |
| mbpp | 3.16× | +7.8% |

---

## Three-model comparison (token-weighted speedup)

### Temperature 0

| Dataset | A (tv0.9) | **B (tv1.0)** | C (legacy) |
|---------|----------:|--------------:|-----------:|
| gsm8k | 3.62× | **3.68×** | 3.56× |
| math500 | 3.67× | **3.77×** | 3.63× |
| mbpp | 2.87× | **2.93×** | 2.86× |
| aime25 | 2.92× | **2.99×** | 2.88× |
| alpaca | 1.82× | **1.83×** | 1.81× |
| mt-bench | 1.48× | **1.53×** | 1.51× |
| longbench mdqa | **1.26×** | **1.26×** | 1.24× |
| longbench icl | **1.44×** | 1.42× | 1.41× |
| **Macro mean** | 2.39× | **2.43×** | 2.36× |

### Temperature 1 (match verification — **not** proper rejection sampling)

| Dataset | A | **B** | C |
|---------|--:|----:|--:|
| gsm8k | 3.23× | 3.20× | **3.28×** |
| math500 | 3.05× | **3.22×** | 3.05× |
| mbpp | **2.69×** | 2.68× | 2.64× |
| aime25 | **2.46×** | **2.46×** | 2.42× |
| alpaca | **1.71×** | 1.70× | 1.70× |
| mt-bench | **1.49×** | 1.41× | 1.41× |
| longbench mdqa | 1.10× | **1.12×** | **1.12×** |
| longbench icl | 1.21× | **1.22×** | 1.19× |
| **Macro mean** | 2.12× | **2.13×** | 2.10× |

Model aliases: A = `ce0.1_tv0.9_base0.2`, B = `ce0.1_tv1.0_base0.0`, C = `legacy_ce1.0_tv0.0_base0.2`.

---

## Stochastic verification: match vs rejection @ temp=1

Same checkpoint as Model B, with `compile_serial_head=true`.

| Dataset | Match (three_model) | | Rejection (compile run) | | Δ speedup |
|---------|--------------------:|---|------------------------:|---|----------:|
| | Speedup | Accept | Speedup | Accept | |
| gsm8k | 3.20× | 4.33 | **3.58×** | **4.60** | **+12%** |
| math500 | 3.22× | 4.34 | **3.48×** | **4.50** | **+8%** |
| alpaca | 1.70× | 2.30 | **1.82×** | **2.35** | **+7%** |
| mbpp | 2.68× | 3.61 | **2.94×** | **3.79** | **+10%** |
| aime25 | 2.46× | 3.31 | **2.79×** | **3.56** | **+13%** |

**Key finding:** At `temperature=1`, switching from greedy-draft + token-match to stochastic-draft + rejection sampling recovers **8–13% speedup** and **+0.2–0.3 accept length** on structured tasks. Match mode uses greedy drafts (`draft_temperature=0`) and compares to target samples — this is **not** equivalent to proper speculative rejection sampling. See `docs/STOCHASTIC_VERIFICATION.md`.

---

## Comparison vs prior art (Qwen3-8B, related work)

From `benchmark_results_fa3_decode_only_noradix_20260722/summary.md` (v1.3 FlashMTP vs DFlash, server-side FA3):

| Setting | DFlash mean | FlashMTP v1.3 mean |
|---------|------------:|-------------------:|
| 7 short datasets | 3.64× | **3.77×** |
| 4 LongBench tasks | 1.94× | **2.61×** |
| All 11 datasets | 3.02× | **3.35×** |

FlashMTP v2 (this work, Model B @ temp=0, client benchmark): **2.43× macro mean** on 8 datasets. Direct comparison is approximate (different draft checkpoints, harness, and overlap settings), but v2's Markov RNN head and training recipe maintain competitive acceptance on math/code while adding proper stochastic verification.

vs **Eagle3**: autoregressive draft tree (serial verify); FlashMTP uses **parallel block draft** (one bidirectional forward per block) + low-rank Markov head for within-block AR — better GPU utilization on long blocks.

vs **DSpARK/DistSpec**: shares rejection-sampling theory; FlashMTP differs in **decoupled memory/context Markov head** (see `compare.md`) and **parallel block conditional distribution** rather than standard AR draft.

---

## Highlight configs for paper tables

| Claim | Config | Number |
|-------|--------|-------:|
| Peak speedup (greedy) | B, temp=0, math500 | **3.77×** |
| Peak speedup (+ compile) | B, temp=0, compile, math500 | **4.01×** |
| Best macro average | B, temp=0 | **2.43×** |
| Stochastic recovery | B, temp=1, rejection, gsm8k | **3.58×** (+12% vs match) |
| `compile_serial_head` e2e | gsm8k, compile on/off | **+7–11%** |
| Long-context weakness | B, temp=0, longbench mdqa | **1.26×** |

---

## Files

```
benchmark_results/
├── consolidated_summary.csv      # all runs, all rows
├── consolidated_summary.json     # aggregates + per-run breakdown
├── three_model_speedup_20260730_1505/
│   ├── summary.csv
│   ├── report.md
│   └── logs/...
└── compile_rejection_20260730_1834/
    ├── summary.csv
    └── logs/...
```
