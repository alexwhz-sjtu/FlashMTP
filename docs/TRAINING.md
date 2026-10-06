# DLite training

## Common model settings

The launchers accept these environment variables and forward the corresponding
CLI options:

| Environment variable | CLI option | Default |
| --- | --- | --- |
| `TARGET_MODEL` | `--target-model-path` | required |
| `TARGET_MODEL_BACKEND` | `--target-model-backend` | `hf` |
| `TRAIN_DATA_PATH` | `--train-data-path` | required |
| `DLITE_VERSION` | `--dlite-version` | `dlite_v2` |
| `BLOCK_SIZE` | `--block-size` | `8` |
| `NUM_DRAFT_LAYERS` | `--num-draft-layers` | `5` |
| `SWA_WINDOW_SIZE` | `--swa-window-size` | `32` |
| `CHS_NUM_LAYERS` | `--chs-num-layers` | `7` |
| `TARGET_LAYER_IDS` | `--target-layer-ids` | `0,1,3,7,11,15,19,23,27,29,30,31` |
| `SEQUENTIAL_HEAD` | `--sequential-head` | `rnn` |
| `SEQUENTIAL_RANK` | `--sequential-rank` | `256` |
| `MASK_TOKEN_ID` | `--mask-token-id` | automatically selected for the target model |

`sglang` remains available for target prefill and exposes the existing SGLang
memory, tensor-parallel, and draft-sharding options. The draft model uses
FlexAttention during training; inference may use FlashAttention 2 or SDPA.

When `TARGET_LAYER_IDS` is non-empty, its unique, increasing, zero-based IDs
are used directly and `CHS_NUM_LAYERS` is ignored. Set `TARGET_LAYER_IDS=` to
restore automatic evenly spaced selection by count. Qwen3.5 targets are loaded
through SGLang with FA3 while their DLite draft remains a dense Qwen3-style
backbone with matching vocabulary and hidden dimensions.

When `MASK_TOKEN_ID` is omitted, training preserves a value stored in a loaded
draft checkpoint, otherwise uses the tokenizer's native mask token, or selects
the first target embedding row not occupied by the tokenizer. An explicit
`MASK_TOKEN_ID` remains available as an override.

## Teacher

Run `scripts/run_training_dlite_teacher.sh`. Its loss can combine final-token CE,
TV, and base-LM CE using `FINAL_CE_WEIGHT`, `TV_LOSS_WEIGHT`, and
`BASE_LM_CE_WEIGHT`. CE labels use the target model's greedy prefill tokens,
while the `rnn` sequential head updates its state with ground-truth previous
tokens from the training sequence.

`dlite_v2` uses
`Q=[embed(anchor-1), embed(anchor), MASK x (block_size-1)]`. The added real
token uses the target model's input embedding row. CHS remains the selected
target-layer hidden states at `anchor-1`: for students both CHS and the new
query use local position 0; for teachers both use absolute position
`anchor-1`. Before predicting `anchor+1`, the sequential RNN state is primed
with the `anchor-1` token and then updated with the anchor token.

`dlite_v1` checkpoints remain loadable for inference and retain their original
`Q=[embed(anchor), MASK x (block_size-1)]` layout and zero-initialized RNN
state. Set `DLITE_VERSION=dlite_v1` only when intentionally training that
legacy layout.

Teacher, Stage 1, transition, and Stage 2 training backpropagate additive loss
numerators. At each optimizer boundary, gradients are normalized by the sum of
effective loss weights across the complete gradient-accumulation window and all
ranks in the FSDP data-parallel process group. A final partial window uses its
actual accumulated denominator rather than treating missing microbatches as
zero-weight rank averages.

## Student

Run `scripts/run_training_dlite_two_stage.sh` and provide a trained teacher with
`TEACHER_DRAFT_PATH`.

- A new student backbone is always initialized from scratch.
- Stage 1 uses only `STAGE1_KL_WEIGHT`; it has no auxiliary regression or
  true-label CE term.
- Stage 1 and Stage 2 share one processed dataloader built from
  `TRAIN_DATA_PATH`.
- The teacher sequential head is copied into the student, then frozen for
  Stage 1. The student backbone learns to match the teacher logits.
- Stage 2 unfreezes the student and uses `STAGE2_FINAL_CE_WEIGHT`,
  `STAGE2_TV_WEIGHT`, and `STAGE2_BASE_CE_WEIGHT`.

Resume with `RESUME_FROM`. For fresh two-stage training,
`TEACHER_DRAFT_PATH`, `STAGE1_EPOCHS`, `STAGE2_EPOCHS`, and `LEARNING_RATE` are
required.
