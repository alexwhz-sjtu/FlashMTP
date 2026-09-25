# Qwen3.5-4B FlashMTP evaluation

Use `.venv-qwen35-eval/bin/python`, not the training `.venv/bin/python`.
`bash scripts/setup_qwen35_eval.sh` creates the separate environment, reuses the
base Torch 2.9.1/CUDA 12.8 and FlashAttention installation, and installs Transformers
5.3.0, flash-linear-attention/fla-core 0.5.2, and causal-conv1d 1.7.0 locally.
The training environment is unchanged. The adapter is tested against these pinned
versions; other Transformers cache interfaces need revalidation.

## Implementation

`evaluation/model_loading.py` loads Qwen3.5's text-only causal model from the
original multimodal checkpoint. A direct loading check found no missing,
unexpected, or mismatched weights. Token embeddings and the target LM head are
loaded from the original target checkpoint.

`evaluation/qwen35_target.py` installs two text-inference adaptations:

- Convert the three-axis mRoPE positions passed to FlashAttention's packed-sequence
  detector into the text axis. Rotary embeddings themselves remain unchanged.
- Continue cached multi-token GatedDeltaNet blocks from the existing convolution
  and recurrent states. Upstream 5.3's multi-token branch assumes prefill. Retain
  the block's recurrence inputs and initial state; when proposals are rejected,
  restore the convolution window and replay only the accepted delta-recurrence
  prefix. Full attention KV is cropped to the same accepted length. All rollback
  work is included in the speculative decode timer.

Prefill and the single-token autoregressive baseline use the upstream linear
attention implementation. No full-prefix replay or target replacement is used.
This is an eager PyTorch/Transformers comparison, not an SGLang/vLLM serving
throughput measurement. BF16 arithmetic may have block-size-dependent rounding;
cache consistency and greedy output agreement are checked separately.

## Validation

Run `tests/check_qwen35_rollback.py` with `--model-name-or-path`,
`--draft-name-or-path`, and `--output`. It checks all accepted prefix lengths 0–8,
compares all four cache state families to a freshly computed retained prefix,
checks the next token, and compares 64-token greedy generation for math, Python,
and Chinese prompts to the autoregressive baseline. The tested ep3 pairing passed.
The original environment's two prefill/CHS regression tests also passed.

## Benchmark

`bash scripts/run_qwen35_ep3_benchmarks.sh` runs the matrix in `benchmark.md`
sequentially on GPU 0 (override `GPU`), with max-new-tokens=512, temperature=0,
batch-size=1, checkpoint block-size=8 and default per-dataset sample counts.
The six LongBench shards retain the evaluator's default 10-sample cap.
No input truncation is added. The script resumes completed dataset JSON files;
use a new `RUN_ROOT` for a fresh comparison after code or environment changes.
Ensure the chosen GPU is idle before launching.

Each dataset has its full log and a JSON containing arguments, package versions,
aggregate metrics, individual turn timing/token counts, acceptance lengths, and
greedy output agreement. Mean acceptance length includes the anchor token
(range 1–8). Token-weighted speedup is baseline seconds/token divided by
speculative seconds/token, with prefill excluded from both decode measurements.
These benchmarks do not grade task accuracy or code pass@1.

The CPU float32 regression test (`tests/test_qwen35_rollback.py`) also checks
all prefix lengths against independent token-by-token execution, repeated
rollback, batches 1 and 2, and invalid crop boundaries. Both tests pass.
Run `scripts/summarize_qwen35_eval.py <run-directory>` to validate and aggregate
completed dataset JSON files into summary.json and summary.csv.

## Numerical follow-up (2026-09-24)

The initial full/gsm8k.json run completed 128 samples (mean acceptance 5.4695,
decode speedup 4.1923x), but only 72 complete greedy sequences matched. Those
results precede the fused-convolution correction and are retained as diagnostics,
not the final benchmark. The remaining initial matrix was stopped intentionally.

The cached block convolution now uses causal_conv1d_fn, matching the upstream
fused convolution/SiLU accumulation instead of introducing an intermediate BF16
rounding. On mismatching GSM8K indices 5–9, three complete outputs (7, 8, 9)
now match. The two remaining first differences occur at equal or near-equal
BF16 logits (reference top-2 gaps 0 or 0.125); see divergence_v2.json. CPU
float32 independent serial/cache tests and GPU all-prefix rollback checks pass.
This supports numerical sensitivity as a cause for the inspected residual
cases; it is not proof that every future mismatch has the same cause.
PyTorch documents that batched and slice computations need not be bitwise equal:
https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html

The corrected matrix writes to
benchmark_results/Flashmtp_v2_qwen3.5_4b_ep3_20260924/full_v2.
Always report its observed complete-sequence agreement alongside speed and
acceptance. Do not claim universal bitwise equivalence from short smoke tests.
