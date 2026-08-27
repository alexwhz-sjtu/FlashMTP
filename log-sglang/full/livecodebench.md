# FlashMTP SGLang Benchmark

## Settings

- dataset: `livecodebench`
- target: `/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-35B-A3B`
- draft: `/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2/cache/models/flashmtp_qwen35_35b_a3b_pb80k_tp4_prefix_fuse16_nlayers5_block8_rnn_easy_direct_r512_ep8/epoch_8_step_76888`
- overlap: `True`
- TP: `1`
- attention backends: `fa3`
- context length: `None`
- yarn: `False`
- radix cache: `True`
- timing scope: `first model decode schedule -> final decode/verify result processed`
- excluded timing: `target prefill, HTTP/client, tokenization`
- throughput weighting: `sum(output_tokens) / sum(model_decode_time)`
- acceptance weighting: `sum(verify_ct * request_accept_length) / sum(verify_ct)`
- first full batch excluded as warmup: `true`

## Backend: `fa3`

| metric | 1 |
| --- | --- |
| baseline decode-only token-weighted output tok/s | 188.02 |
| FlashMTP decode-only token-weighted output tok/s | 339.66 |
| baseline measured output tokens | 18,197 |
| FlashMTP measured output tokens | 18,178 |
| baseline measured wall time (s) | 100.829 |
| FlashMTP measured wall time (s) | 58.616 |
| baseline measured model decode time (s) | 96.785 |
| FlashMTP measured model decode time (s) | 53.518 |
| speedup | 1.807 |
| FlashMTP globally verification-weighted accept length | 3.669 |
| baseline mean latency (s) | 2.016 |
| FlashMTP mean latency (s) | 1.172 |
| FlashMTP p99 latency (s) | 2.139 |
| baseline incremental GPU memory (GiB) | 72.30 |
| FlashMTP incremental GPU memory (GiB) | 74.75 |

