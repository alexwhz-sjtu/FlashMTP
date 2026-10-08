import argparse

import pytest
import torch

from scripts.dlite.dlite_disaggregate import (
    _select_packet_layers,
    aligned_anchor_count,
    topology_identity,
    validate_resume_model,
    validate_resume_topology,
)
from scripts.dlite.dlite_training import add_common_args, validate_common_args
from specforge.disaggregate import (
    DraftBatchPacket,
    DraftPacketSpec,
    build_node_routes,
    copy_packet_slice,
)
from specforge.distributed import disaggregated_group_plan


def test_three_tp2_ep2_targets_and_two_drafts_group_plan():
    plan = disaggregated_group_plan(
        nnodes=1,
        target_ranks_per_node=6,
        draft_ranks_per_node=2,
        target_tp_size=2,
        target_ep_size=2,
    )
    assert plan["tp"] == [[0, 1], [2, 3], [4, 5]]
    assert plan["moe_ep"] == [[0, 1], [2, 3], [4, 5]]
    assert plan["moe_tp"] == [[0], [1], [2], [3], [4], [5]]
    assert plan["singleton"] == [[0], [1], [2], [3], [4], [5]]
    assert plan["bridge"] == [[0, 2, 4, 6, 7]]
    assert plan["draft"] == [[6, 7]]


def test_three_producers_route_six_samples_to_two_drafts():
    routes = build_node_routes(producers=3, drafts=2, node_batch_size=6)
    assert [(r.producer, r.draft, r.size) for r in routes] == [
        (0, 0, 2),
        (1, 0, 1),
        (1, 1, 1),
        (2, 1, 2),
    ]


def test_packet_contains_history_and_prediction_hidden_but_no_logits():
    spec = DraftPacketSpec(
        batch_size=2,
        max_length=8,
        num_anchors=4,
        num_target_layers=3,
        hidden_size=16,
        prediction_length=7,
        num_history_layers=3,
        include_target_prediction_hidden=True,
    )
    source = DraftBatchPacket.empty(spec, device="cpu")
    destination = DraftBatchPacket.empty(spec, device="cpu")
    source.target_history_hidden.fill_(5)
    source.target_prediction_hidden.fill_(7)
    copy_packet_slice(
        source,
        destination,
        source_start=0,
        source_end=2,
        destination_start=0,
        destination_end=2,
    )
    assert torch.equal(source.target_history_hidden, destination.target_history_hidden)
    assert torch.equal(
        source.target_prediction_hidden, destination.target_prediction_hidden
    )
    assert not hasattr(source, "target_logits")


def test_packet_layer_union_is_reordered_for_each_draft():
    spec = DraftPacketSpec(1, 4, 2, 3, 1, 2)
    packet = DraftBatchPacket.empty(spec, device="cpu")
    packet.target_hidden[..., 0, :].fill_(2)
    packet.target_hidden[..., 1, :].fill_(7)
    packet.target_hidden[..., 2, :].fill_(11)
    selected = _select_packet_layers(packet, [2, 7, 11], [11, 2])
    assert selected.shape == (1, 2, 2, 1)
    assert torch.equal(selected[..., 0, :], torch.full((1, 2, 1), 11.0))
    assert torch.equal(selected[..., 1, :], torch.full((1, 2, 1), 2.0))


def test_anchor_width_is_fixed_and_flex_aligned():
    count = aligned_anchor_count(17, query_length=8, chs_slots=7)
    assert count >= 17
    assert (count * 8) % 128 == 0
    assert (count * 15) % 128 == 0


def _parse_common(*extra):
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--target-model-path",
            "target",
            "--train-data-path",
            "train.jsonl",
            "--output-dir",
            "out",
            *extra,
        ]
    )
    validate_common_args(parser, args)
    return args


def test_disaggregate_cli_accepts_tp2_ep2_topology():
    args = _parse_common(
        "--target-model-backend",
        "sglang",
        "--disaggregate",
        "--target-ranks-per-node",
        "6",
        "--draft-ranks-per-node",
        "2",
        "--target-tp-size",
        "2",
        "--sglang-ep-size",
        "2",
        "--node-batch-size",
        "6",
    )
    assert args.target_batch_size == 2
    assert args.draft_micro_batch_size == 3


def test_disaggregate_rejects_offline_cache():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--target-model-path",
            "target",
            "--train-hidden-states-path",
            "cache",
            "--output-dir",
            "out",
            "--disaggregate",
            "--target-ranks-per-node",
            "1",
            "--draft-ranks-per-node",
            "1",
            "--node-batch-size",
            "2",
        ]
    )
    with pytest.raises(SystemExit):
        validate_common_args(parser, args)


def test_resume_topology_allows_pipeline_depth_change_only():
    topology = type(
        "Topology",
        (),
        {"nnodes": 1, "draft_global_ranks": [6, 7]},
    )()
    args = type(
        "Args",
        (),
        {
            "target_ranks_per_node": 6,
            "draft_ranks_per_node": 2,
            "target_tp_size": 2,
            "sglang_ep_size": 2,
            "node_batch_size": 6,
        },
    )()
    identity = topology_identity(args, topology)
    validate_resume_topology({**identity, "pipeline_depth": 9}, identity)
    with pytest.raises(ValueError, match="target_tp_size"):
        validate_resume_topology({**identity, "target_tp_size": 1}, identity)


def test_resume_model_identity_rejects_layer_mismatch():
    draft = type(
        "Draft",
        (),
        {
            "architecture_version": "dlite_v2",
            "model_role": "pivot_q_student",
            "target_layer_ids": [0, 3, 7],
            "config": type("Config", (), {"num_target_layers": 8, "hidden_size": 16})(),
        },
    )()
    state = {
        "architecture_version": "dlite_v2",
        "model_role": "pivot_q_student",
        "target_layer_ids": [0, 7],
        "num_target_layers": 8,
        "target_hidden_size": 16,
    }
    with pytest.raises(ValueError, match="target_layer_ids"):
        validate_resume_model(state, draft)
