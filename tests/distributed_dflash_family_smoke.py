"""Two-rank FSDP smoke test for DFlash-family wrappers."""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import Qwen3Config

from specforge.core.dflash_family import OnlineDFlashModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    method = {
        "block_size": 4,
        "target_layer_ids": [1],
        "mask_token_id": 31,
        "attention_mode": "gqa",
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 8,
        "selector_top_k": 3,
    }
    config = Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=16,
        max_position_embeddings=64,
        vocab_size=32,
        block_size=4,
        layer_types=["full_attention"],
        dflash_config=method,
    )
    config._attn_implementation = "flex_attention"
    draft = DFlash2DraftModel(config).cuda().to(torch.bfloat16)
    embed = nn.Embedding(32, 64, device="cuda", dtype=torch.bfloat16)
    head = nn.Linear(64, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    online = OnlineDFlashModel(
        draft,
        head,
        embed,
        mask_token_id=31,
        block_size=4,
        attention_backend="flex_attention",
        num_anchors=4,
        objective_chunk_blocks=2,
        selector_loss_alpha=1.0,
    )
    wrapped = FSDP(
        online,
        ignored_modules=[head, embed],
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    torch.manual_seed(100 + dist.get_rank())
    ids = torch.randint(0, 32, (1, 16), device="cuda")
    hidden = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)
    loss_mask = torch.ones(1, 16, device="cuda")
    _, _, metrics = wrapped(
        input_ids=ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        collect_detailed_metrics=False,
    )
    numerator, denominator = metrics["loss_terms"]
    (numerator / denominator.clamp_min(1)).backward()
    grad_count = torch.tensor(
        sum(parameter.grad is not None for parameter in draft.parameters()),
        device="cuda",
    )
    dist.all_reduce(grad_count, op=dist.ReduceOp.MIN)
    if not int(grad_count.item()):
        raise RuntimeError("DFlash2 FSDP smoke produced no gradients")
    if dist.get_rank() == 0:
        print("distributed dflash2 FSDP smoke passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
