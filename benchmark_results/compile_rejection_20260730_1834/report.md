# Compile + rejection sampling benchmark

Configuration: Qwen3-8B target, batch size 1, draft/verify block 16,
up to 50 samples per dataset and 512 generated tokens. All three runs use
`--compile-serial-head`.

Each cell below is `token-weighted speedup / average acceptance length`.

| Dataset | rnn TV1 base0, T=0 | rnn TV1 base0, T=1 rejection | rnn_easy base0.2, T=0 |
|---|---:|---:|---:|
| alpaca | 1.91x / 2.46 | 1.82x / 2.35 | 1.95x / 2.43 |
| gsm8k | 3.94x / 4.95 | 3.58x / 4.60 | 3.92x / 4.92 |
| mbpp | 3.16x / 3.97 | 2.94x / 3.79 | 3.09x / 3.86 |
| aime25 | 3.21x / 4.03 | 2.79x / 3.56 | 3.14x / 3.93 |
| math500 | 4.01x / 5.05 | 3.48x / 4.50 | 3.90x / 4.88 |
| mt-bench | 1.65x / 2.04 | 1.58x / 1.98 | 1.60x / 2.00 |
| multi-document QA | 1.34x / 2.12 | 1.25x / 2.03 | 1.30x / 2.08 |
| in-context learning | 1.46x / 2.35 | 1.34x / 2.16 | 1.44x / 2.35 |
| Macro average | **2.59x / 3.37** | **2.35x / 3.12** | **2.54x / 3.31** |

## Comparisons with the previous non-compile benchmark

- At temperature 0, compiling the same rnn TV1/base0 checkpoint improved
  speedup on every dataset. The mean relative improvement was **6.3%**,
  while mean acceptance changed by only **+0.001**. This isolates the compile
  benefit and confirms greedy decoding semantics were preserved.
- At temperature 1, rejection + compile improved speedup over the previous
  match + eager run on every dataset. The mean relative improvement was
  **10.5%**, and mean acceptance increased by **0.163 token per step**.
  This is a combined comparison, so it does not isolate rejection from compile.
- Within the new rnn TV1/base0 runs, temperature 1 rejection has lower
  acceptance than temperature 0 greedy decoding (3.12 versus 3.37 macro
  average), but it preserves the target model's temperature-1 distribution.

Raw results are in `summary.csv`; all 24 jobs completed successfully.
