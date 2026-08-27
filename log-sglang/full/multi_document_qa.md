# FlashMTP SGLang Benchmark

## Settings

- dataset: `longbench_v2_64000_32000_multi_document_qa`
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
| baseline decode-only token-weighted output tok/s | 172.84 |
| FlashMTP decode-only token-weighted output tok/s | 209.24 |
| baseline measured output tokens | 15,897 |
| FlashMTP measured output tokens | 15,856 |
| baseline measured wall time (s) | 150.428 |
| FlashMTP measured wall time (s) | 136.649 |
| baseline measured model decode time (s) | 91.975 |
| FlashMTP measured model decode time (s) | 75.778 |
| speedup | 1.211 |
| FlashMTP globally verification-weighted accept length | 2.350 |
| baseline mean latency (s) | 4.558 |
| FlashMTP mean latency (s) | 4.141 |
| FlashMTP p99 latency (s) | 5.299 |
| baseline incremental GPU memory (GiB) | 72.30 |
| FlashMTP incremental GPU memory (GiB) | 74.75 |

