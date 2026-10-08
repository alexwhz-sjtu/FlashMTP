# DLite training

## DFlash-family training

`DFlashDraftModel`, `DFlash2DraftModel`, and `DSparkDraftModel` use independent
direct-training entrypoints while sharing the target capture, FSDP, offline
`regen_full`, checkpoint, and disaggregated execution infrastructure.

```bash
TARGET_MODEL=/path/to/Qwen3-4B \
TRAIN_DATA=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/output \
bash scripts/dflash_family/run_training_dflash.sh

# The other launchers are run_training_dflash2.sh and run_training_dspark.sh.
```

All three accept `--init-from`, `--resume-from`, `--train-hidden-states-path`,
and the common distributed options. Set `--disaggregate` together with the
target/draft rank topology to transport full target context hidden states
without transporting vocabulary logits. DFlash uses hard-label CE; DFlash2
adds the top-k selector objective; DSpark combines CE, distribution L1, and
confidence loss. DSpark uses anchor-inclusive blocks: output slot zero is not
supervised and slot one predicts the token after the anchor.

The default generated architectures are five draft layers with block size 16
for DFlash/DFlash2 and block size 7 for DSpark. DFlash2 defaults to grouped
convolution `(kernel=2, group=16)` and selector `(rank=256, top-k=16)`. DSpark
defaults to a rank-256 vanilla Markov head with Markov-conditioned confidence.

Benchmarks use `evaluation/run_benchmark_dflash.sh`,
`evaluation/run_benchmark_dflash2.sh`, or `evaluation/run_benchmark_dspark.sh`.

## Common model settings

The launchers accept these environment variables and forward the corresponding
CLI options:

| Environment variable | CLI option | Default |
| --- | --- | --- |
| `TARGET_MODEL` | `--target-model-path` | required |
| `TARGET_MODEL_BACKEND` | `--target-model-backend` | `hf` |
| `TRAIN_DATA_PATH` | `--train-data-path` | online mode |
| `TRAIN_HIDDEN_STATES_PATH` | `--train-hidden-states-path` | offline mode |
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

Set exactly one training-data variable. `TRAIN_DATA_PATH` tokenizes JSONL and
runs target prefill online. `TRAIN_HIDDEN_STATES_PATH` reads a schema-v2
`regen_full` cache and loads only the target embedding and LM head. Offline
target logits are projected from cached final-norm hidden states at sampled
anchor positions. Offline mode requires `TP_SIZE=1` and
`SHARD_DRAFT_BY_TP=0`; torchrun workers then operate as data-parallel ranks.

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

Run `scripts/dlite/run_training_dlite_teacher.sh`. Its loss can combine final-token CE,
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

Run `scripts/dlite/run_training_dlite_two_stage.sh` and provide a trained teacher with
`TEACHER_DRAFT_PATH`.

- A new student backbone is always initialized from scratch.
- Stage 1 uses only `STAGE1_KL_WEIGHT`; it has no auxiliary regression or
  true-label CE term.
- Stage 1 and Stage 2 share one dataloader built from the selected online JSONL
  or offline `regen_full` cache.
- The teacher sequential head is copied into the student, then frozen for
  Stage 1. The student backbone learns to match the teacher logits.
- Stage 2 unfreezes the student and uses `STAGE2_FINAL_CE_WEIGHT`,
  `STAGE2_TV_WEIGHT`, and `STAGE2_BASE_CE_WEIGHT`.

Resume with `RESUME_FROM`. For fresh two-stage training,
`TEACHER_DRAFT_PATH`, `STAGE1_EPOCHS`, `STAGE2_EPOCHS`, and `LEARNING_RATE` are
required.

## Disaggregated target/draft execution

Set `DISAGGREGATE=1` for online JSONL training in any of the teacher, SFT, or
two-stage launchers. The launcher forwards:

| Environment variable | CLI option | Meaning |
| --- | --- | --- |
| `RANK_TARGET_PER_NODE` | `--target-ranks-per-node` | target ranks on each node |
| `RANK_DRAFT_PER_NODE` | `--draft-ranks-per-node` | draft FSDP ranks on each node |
| `TARGET_TP_SIZE` | `--target-tp-size` | TP size of each independent target replica |
| `SGLANG_EP_SIZE` | `--sglang-ep-size` | EP size inside each target replica |
| `NODE_BATCH_SIZE` | `--node-batch-size` | total samples produced per node and step |
| `DRAFT_MICRO_BATCH_SIZE` | `--draft-micro-batch-size` | microbatch on each draft rank |
| `PIPELINE_DEPTH` | `--pipeline-depth` | in-flight packet slots, at least 2 |
| `PROFILE` | `--profile` | collect bridge timing when set to `1` |

`RANK_TARGET_PER_NODE` must be divisible by `TARGET_TP_SIZE`, and
`TARGET_TP_SIZE` must be divisible by `SGLANG_EP_SIZE`. `NODE_BATCH_SIZE` must
be divisible by both the target-replica count and `RANK_DRAFT_PER_NODE`.
TP/EP greater than one requires the SGLang backend. Pipeline depth is excluded
from checkpoint identity and may be changed on resume; topology, batch sizes,
model/layer configuration, and the training-data identity must match.

All process groups are created before ranks split into target and draft roles.
Target ranks retain the backend's sharded model head, but the training path does
not compute or transmit its vocabulary logits. Each draft rank instead loads a
full frozen target embedding and LM head, outside FSDP and the optimizer, and
projects the transmitted final-norm hidden states locally. Stage 1 packets omit
prediction logits entirely; transition, teacher, and SFT/Stage 2 compute them
only where their losses require them.

Disaggregated execution accepts `TRAIN_DATA_PATH` only. It is incompatible
with `TRAIN_HIDDEN_STATES_PATH` and `SHARD_DRAFT_BY_TP=1`; `regen_full` keeps
using the ordinary offline DP/FSDP path.
