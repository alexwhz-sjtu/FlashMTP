"""Validate that target-only SGLang initialization reuses pre-created TP/EP groups."""

import argparse

import sglang.srt.distributed.parallel_state as parallel_state
import torch
import torch.distributed as dist

from specforge.distributed import destroy_distributed, init_disaggregated
from specforge.modeling.target.sglang_backend.patch import (
    init_distributed_environment,
    initialize_model_parallel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ranks", type=int, default=4)
    parser.add_argument("--draft-ranks", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--target-ep", type=int, default=2)
    args = parser.parse_args()
    topology = init_disaggregated(
        timeout=5,
        target_ranks_per_node=args.target_ranks,
        draft_ranks_per_node=args.draft_ranks,
        target_tp_size=args.target_tp,
        target_ep_size=args.target_ep,
    )
    original_new_group = dist.new_group

    def reject_late_group(*_args, **_kwargs):
        raise AssertionError("SGLang created a process group after role divergence")

    # From here onward draft ranks do not enter SGLang.  Any target-only
    # new_group call would deadlock in production, so make the smoke test fail
    # immediately instead of relying on a timeout.
    dist.new_group = reject_late_group
    if topology.is_target:
        init_distributed_environment(local_rank=topology.local_rank)
        initialize_model_parallel(
            tensor_model_parallel_size=args.target_tp,
            expert_model_parallel_size=args.target_ep,
            pipeline_model_parallel_size=1,
        )
        assert parallel_state._WORLD.world_size == args.target_tp
        assert parallel_state._TP.world_size == args.target_tp
        assert parallel_state._MOE_EP.world_size == args.target_ep
        assert parallel_state._MOE_TP.world_size == args.target_tp // args.target_ep
        assert parallel_state._PP.world_size == 1
        tp_value = torch.ones((), device="cuda")
        parallel_state._TP.all_reduce(tp_value)
        assert int(tp_value.item()) == args.target_tp
        ep_value = torch.ones((), device="cuda")
        parallel_state._MOE_EP.all_reduce(ep_value)
        assert int(ep_value.item()) == args.target_ep
    dist.new_group = original_new_group
    dist.barrier()
    destroy_distributed()


if __name__ == "__main__":
    main()
