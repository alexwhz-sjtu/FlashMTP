# rnn_easy versus rnn serial-head compile profile

Hardware: one idle NVIDIA H800 (physical GPU 1).

The profile loads the real checkpoint weights and measures only serial-head
block sampling. It excludes the FlashMTP backbone, target verification and
acceptance behavior.

Common shape:

- batch size: 1
- prediction length: 15
- hidden size: 4096
- Markov rank: 512
- vocabulary size: 151936
- output mode: direct
- 20 warmup iterations
- 5 repeats of 200 timed iterations

## Steady-state latency

| Temperature | Head | Eager ms/block | Compile ms/block | Compile speedup | Latency reduction |
|---:|---|---:|---:|---:|---:|
| 0 | rnn_easy | 2.061 | 1.526 | 1.35x | 25.9% |
| 0 | rnn | 2.892 | 1.602 | 1.81x | 44.6% |
| 1 | rnn_easy | 3.738 | 2.189 | 1.71x | 41.4% |
| 1 | rnn | 4.666 | 2.241 | 2.08x | 52.0% |

## Structural comparison

At temperature 0:

- Eager rnn_easy is 1.40x faster than eager rnn (28.7% lower latency).
- Compiled rnn_easy is 1.05x faster than compiled rnn (4.7% lower latency).

At temperature 1:

- Eager rnn_easy is 1.25x faster than eager rnn (19.9% lower latency).
- Compiled rnn_easy is 1.02x faster than compiled rnn (2.3% lower latency).

Compile therefore removes most of the runtime difference between the two
structures. rnn benefits more because it contains the additional
`state_out_proj`, `hidden_fuse_gate_proj`, sigmoid and weighted fusion at every
serial position. rnn_easy remains slightly faster after compilation.

Observed first compiled-call wall times were 6.05 s and 14.67 s for rnn_easy
at temperatures 0 and 1. The rnn checkpoint ran second and reused compiler
artifacts, so its first-call times are not directly comparable.

Raw measurements are in `results.json`.
