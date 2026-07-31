# Three-model FlashMTP speedup benchmark

## Configuration

- Target: `/share/dai-sys/wanghanzhen/models/Qwen/Qwen3-8B`
- Maximum new tokens: 512
- Requested samples per dataset: 50
- Batch size: 1
- Draft block size: 16
- Verify block size: 16
- Temperatures: 0 and 1
- Hardware: 8 × NVIDIA H800; at most one benchmark process per GPU
- Completed tasks: 48/48
- Failed tasks: 0

Actual dataset sizes below 50:

- AIME25: 30
- LongBench v2 Multi-Document QA: 33
- LongBench v2 In-Context Learning: 6

Model aliases:

- **A — ce0.1_tv0.9_base0.2**: final CE 0.1, TV 0.9, Base CE 0.2
- **B — ce0.1_tv1.0_base0.0**: final CE 0.1, TV 1.0, Base CE 0.0
- **C — legacy_ce1.0_tv0.0_base0.2**: checkpoint before TV loss, Base CE 0.2

## Token-weighted speedup

### Temperature 0

| Dataset | A | B | C |
|---|---:|---:|---:|
| Alpaca | 1.82x | **1.83x** | 1.81x |
| GSM8K | 3.62x | **3.68x** | 3.56x |
| MBPP | 2.87x | **2.93x** | 2.86x |
| AIME25 | 2.92x | **2.99x** | 2.88x |
| Math500 | 3.67x | **3.77x** | 3.63x |
| MT-Bench | 1.48x | **1.53x** | 1.51x |
| LongBench Multi-Document QA | **1.26x** | **1.26x** | 1.24x |
| LongBench In-Context Learning | **1.44x** | 1.42x | 1.41x |
| **Macro mean** | 2.385x | **2.426x** | 2.362x |

### Temperature 1

| Dataset | A | B | C |
|---|---:|---:|---:|
| Alpaca | **1.71x** | 1.70x | 1.70x |
| GSM8K | 3.23x | 3.20x | **3.28x** |
| MBPP | **2.69x** | 2.68x | 2.64x |
| AIME25 | **2.46x** | **2.46x** | 2.42x |
| Math500 | 3.05x | **3.22x** | 3.05x |
| MT-Bench | **1.49x** | 1.41x | 1.41x |
| LongBench Multi-Document QA | 1.10x | **1.12x** | **1.12x** |
| LongBench In-Context Learning | 1.21x | **1.22x** | 1.19x |
| **Macro mean** | 2.118x | **2.126x** | 2.101x |

## Aggregate comparison

The following values are unweighted macro averages over the listed datasets,
not a recomputation from pooled output tokens across datasets.

| Model | Temperature 0 | Temperature 1 | All 16 settings | Mean acceptance length |
|---|---:|---:|---:|---:|
| A — CE 0.1 / TV 0.9 / Base 0.2 | 2.385x | 2.118x | 2.251x | 3.131 |
| B — CE 0.1 / TV 1.0 / Base 0.0 | **2.426x** | **2.126x** | **2.276x** | **3.164** |
| C — legacy / Base 0.2 | 2.362x | 2.101x | 2.232x | 3.099 |

Model B has the highest macro-average speedup:

- About 1.1% above model A across all 16 settings.
- About 2.0% above model C across all 16 settings.
- Its advantage is clearer at temperature 0 than at temperature 1.

See `summary.csv` for per-task decode times, unweighted speedup, acceptance
length, elapsed time, exact checkpoint path, and log path.
