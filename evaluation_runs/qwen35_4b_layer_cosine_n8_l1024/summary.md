# Target layer input/output cosine similarity

- Model: `/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-4B`
- Dataset: `/share/dai-sys/wanghanzhen/projects/MTP/training_data/generated/qwen3.5-4b/qwen_3.5_4b_math_code_aug_think_off_math_code_chat_aug1_temp1_maxnew4096.jsonl`
- Samples: 8
- Maximum sequence length: 1024
- Layer IDs are 0-based.
- `loss_mean` covers supervised assistant tokens only.

## Group summary

- Full-attention layers, loss-token mean: 0.878987
- Full-attention layers excluding the final layer, loss-token mean: 0.895373
- Linear-attention layers excluding layer 0, loss-token mean: 0.918214

Layer 0 compares token embeddings with the first block output; all later layers compare the previous block output with the current block output.

## Per-layer results

| layer | type | all-token mean | std | loss-token mean | loss tokens |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 0 | linear_attention | 0.123826 | 0.043881 | 0.121729 | 4378 |
| 1 | linear_attention | 0.936639 | 0.031333 | 0.940258 | 4378 |
| 2 | linear_attention | 0.889939 | 0.053370 | 0.899469 | 4378 |
| 3 | full_attention | 0.894421 | 0.040890 | 0.900114 | 4378 |
| 4 | linear_attention | 0.919479 | 0.033687 | 0.923866 | 4378 |
| 5 | linear_attention | 0.901141 | 0.033659 | 0.906858 | 4378 |
| 6 | linear_attention | 0.880866 | 0.035553 | 0.888584 | 4378 |
| 7 | full_attention | 0.886632 | 0.035613 | 0.891211 | 4378 |
| 8 | linear_attention | 0.911472 | 0.037591 | 0.915459 | 4378 |
| 9 | linear_attention | 0.920615 | 0.031562 | 0.923589 | 4378 |
| 10 | linear_attention | 0.920401 | 0.028004 | 0.924107 | 4378 |
| 11 | full_attention | 0.910608 | 0.027067 | 0.911153 | 4378 |
| 12 | linear_attention | 0.927421 | 0.022870 | 0.928171 | 4378 |
| 13 | linear_attention | 0.934170 | 0.019855 | 0.934461 | 4378 |
| 14 | linear_attention | 0.925340 | 0.019710 | 0.926075 | 4378 |
| 15 | full_attention | 0.907246 | 0.028948 | 0.903473 | 4378 |
| 16 | linear_attention | 0.915868 | 0.022258 | 0.912882 | 4378 |
| 17 | linear_attention | 0.907426 | 0.025433 | 0.903984 | 4378 |
| 18 | linear_attention | 0.868216 | 0.041806 | 0.863316 | 4378 |
| 19 | full_attention | 0.844106 | 0.044573 | 0.837688 | 4378 |
| 20 | linear_attention | 0.910081 | 0.027243 | 0.908603 | 4378 |
| 21 | linear_attention | 0.918377 | 0.024546 | 0.916948 | 4378 |
| 22 | linear_attention | 0.908747 | 0.026520 | 0.908737 | 4378 |
| 23 | full_attention | 0.914181 | 0.027372 | 0.911319 | 4378 |
| 24 | linear_attention | 0.939683 | 0.022720 | 0.938730 | 4378 |
| 25 | linear_attention | 0.948793 | 0.021622 | 0.948355 | 4378 |
| 26 | linear_attention | 0.935691 | 0.030054 | 0.936362 | 4378 |
| 27 | full_attention | 0.914965 | 0.033207 | 0.912651 | 4378 |
| 28 | linear_attention | 0.935339 | 0.026962 | 0.934156 | 4378 |
| 29 | linear_attention | 0.936244 | 0.031192 | 0.935457 | 4378 |
| 30 | linear_attention | 0.901532 | 0.048577 | 0.900497 | 4378 |
| 31 | full_attention | 0.748209 | 0.070863 | 0.764288 | 4378 |
