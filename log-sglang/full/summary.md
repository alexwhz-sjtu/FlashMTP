# FlashMTP Qwen3.5-35B-A3B SGLang Benchmark

- target: `/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-35B-A3B`
- draft: `/share/dai-sys/wanghanzhen/projects/MTP/FlashMTP_v2/cache/models/flashmtp_qwen35_35b_a3b_pb80k_tp4_prefix_fuse16_nlayers5_block8_rnn_easy_direct_r512_ep8/epoch_8_step_76888`
- runtime: `sglang==0.5.6.post2`, H800, BF16, FA3, TP=1, concurrency=1, spec-v2 overlap
- sampling: seed 0 without replacement; `max_samples=50` is a cap
- generation: greedy, `max_new_tokens=512`; one warmup request per runner is excluded
- input: no input-token cap and no truncation
- speedup: FlashMTP decode-only output tokens/s divided by baseline decode-only output tokens/s
- accept length: globally weighted by target verification count

| dataset | samples | accept length | speedup |
| --- | ---: | ---: | ---: |
| alpaca | 50 | 2.505 | 1.256x |
| gsm8k | 50 | 4.962 | 2.368x |
| math500 | 50 | 4.847 | 2.328x |
| mbpp | 50 | 3.981 | 1.954x |
| livecodebench | 50 | 3.669 | 1.807x |
| humaneval | 50 | 3.979 | 1.933x |
| mt-bench | 50 | 2.854 | 1.418x |
| aime25 | 30 | 4.139 | 2.019x |
| longbench_v2_64000_32000_single_document_qa | 16 | 2.371 | 1.216x |
| longbench_v2_64000_32000_multi_document_qa | 33 | 2.350 | 1.211x |
| longbench_v2_64000_32000_long_dialogue | 7 | 4.623 | 2.370x |
| longbench_v2_64000_32000_structured_data | 1 | 2.393 | 1.246x |
| longbench_v2_64000_32000_in_context_learning | 6 | 2.844 | 1.467x |
| longbench_v2_64000_32000_code_repo | 3 | 2.560 | 1.344x |

## Long-context configuration check

The target text config and SGLang both report a context length of 262,144. Its
native RoPE parameters are `rope_type=default`, `rope_theta=10000000`, and
`partial_rotary_factor=0.25`. The measured LongBench v2 prompts contain
33,209--64,806 tokens before generation, so no YaRN or context override was
applied.

All 14 benchmark pairs completed without OOM or traceback, and all GPUs were
idle after the run.
