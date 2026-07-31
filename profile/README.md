# FlashMTP_v2 Profiling Artifacts

Index of profiling scripts and results under `profile/`.

## `compile_serial_head` study

**Question:** Why does compiling only the Markov serial head improve end-to-end decode when target verify is the bottleneck?

**Answer (short):** Serial head is ~6–9% of per-step GPU time but speeds up ~1.7–1.9× with `torch.compile`. Amdahl gives ~2.6–4.4% per-step gain, matching micro-benchmarks; e2e gains of ~5–11% follow at unchanged acceptance (temp=0). See full analysis:

→ **[`compile_serial_head_profile.md`](compile_serial_head_profile.md)** (includes 中文摘要)

### Scripts

| File | Purpose |
|------|---------|
| [`profile_compile_serial_head.py`](profile_compile_serial_head.py) | Micro step breakdown + controlled e2e (compile on/off) |
| [`summarize_compile_profile.py`](summarize_compile_profile.py) | Print summary from a timing JSON |

### Per-dataset raw JSON

| Dataset | JSON | Console log |
|---------|------|-------------|
| gsm8k | [`gsm8k/compile_serial_head_timing.json`](gsm8k/compile_serial_head_timing.json) | [`gsm8k/run.log`](gsm8k/run.log) |
| math500 | [`math500/compile_serial_head_timing.json`](math500/compile_serial_head_timing.json) | [`math500/run.log`](math500/run.log) |
| mt-bench | [`mt-bench/compile_serial_head_timing.json`](mt-bench/compile_serial_head_timing.json) | [`mt-bench_run.log`](mt-bench_run.log) |
| longbench mdqa | [`longbench_mdqa/compile_serial_head_timing.json`](longbench_mdqa/compile_serial_head_timing.json) | [`longbench_mdqa_run.log`](longbench_mdqa_run.log) |

Dataset key for longbench: `longbench_v2_64000_32000_multi_document_qa`.

### Benchmark logs referenced (50 samples, temp=0)

| Condition | Path |
|-----------|------|
| compile_off | `benchmark_results/three_model_speedup_20260730_1505/logs/ce0.1_tv1.0_base0.0/temperature_0/` |
| compile_on | `benchmark_results/compile_rejection_20260730_1834/logs/rnn_ce0.1_tv1.0_base0.0_temp0_compile/` |

### Config (all profile runs)

- **Target:** Qwen3-8B
- **Draft:** `flashmtp_v2_mhrnn_direct_r512_ce0.1_tv1.0_wb_0.0_bgemma_21_qwen3_8b`
- **block_size=16**, **verify_block=16**, **temperature=0**
- **GPU:** H800 (CUDA device 3 for mt-bench / longbench; device 0 for gsm8k / math500)

### Quick reproduce

```bash
cd /share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=3 python profile/profile_compile_serial_head.py \
  --dataset gsm8k --max-samples 3 --max-new-tokens 256 \
  --output-dir profile/gsm8k
python profile/summarize_compile_profile.py profile/gsm8k/compile_serial_head_timing.json
```

## Other profiling tools (elsewhere in repo)

- `scripts/profile_spec_step_breakdown.py` — per-step component breakdown
- `scripts/profile_markov_head_timing.py` — Markov head mode comparison
- `profile_utils/spec_profile.py` — e2e profile modes
