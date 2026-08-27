# FlashMTP SGLang Benchmark

## Settings

- dataset: `longbench_v2_64000_32000_in_context_learning`
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
| baseline decode-only token-weighted output tok/s | 174.31 |
| FlashMTP decode-only token-weighted output tok/s | 255.73 |
| baseline measured output tokens | 2,919 |
| FlashMTP measured output tokens | 2,958 |
| baseline measured wall time (s) | 25.021 |
| FlashMTP measured wall time (s) | 20.322 |
| baseline measured model decode time (s) | 16.746 |
| FlashMTP measured model decode time (s) | 11.567 |
| speedup | 1.467 |
| FlashMTP globally verification-weighted accept length | 2.844 |
| baseline mean latency (s) | 4.170 |
| FlashMTP mean latency (s) | 3.387 |
| FlashMTP p99 latency (s) | 4.319 |
| baseline incremental GPU memory (GiB) | 72.30 |
| FlashMTP incremental GPU memory (GiB) | 74.75 |

