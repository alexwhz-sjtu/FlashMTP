# FlashMTP SGLang Benchmark

## Settings

- dataset: `longbench_v2_64000_32000_code_repo`
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
| baseline decode-only token-weighted output tok/s | 171.32 |
| FlashMTP decode-only token-weighted output tok/s | 230.33 |
| baseline measured output tokens | 1,536 |
| FlashMTP measured output tokens | 1,536 |
| baseline measured wall time (s) | 13.579 |
| FlashMTP measured wall time (s) | 11.564 |
| baseline measured model decode time (s) | 8.966 |
| FlashMTP measured model decode time (s) | 6.669 |
| speedup | 1.344 |
| FlashMTP globally verification-weighted accept length | 2.560 |
| baseline mean latency (s) | 4.526 |
| FlashMTP mean latency (s) | 3.855 |
| FlashMTP p99 latency (s) | 4.877 |
| baseline incremental GPU memory (GiB) | 72.30 |
| FlashMTP incremental GPU memory (GiB) | 74.75 |

