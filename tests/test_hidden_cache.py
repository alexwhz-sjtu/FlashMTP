import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dlite import dlite_training
from scripts.dlite.dlite_training import (
    add_common_args,
    project_cached_target_logits,
    validate_common_args,
)
from specforge.core.dlite import gather_target_prefill_logits
from specforge.data.hidden_cache import (
    HiddenCacheError,
    RegenFullCollator,
    RegenFullDataset,
    load_hidden_cache_manifest,
)
from specforge.modeling.target.dlite_target_model import SGLangDLiteTargetModel

PREPARE_PATH = ROOT / "scripts" / "data" / "prepare_hidden_states.py"
SPEC = importlib.util.spec_from_file_location("prepare_hidden_states_v2", PREPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def _serialized(payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _payload(source_index, length, offset=0):
    layer_zero = torch.arange(length * 4, dtype=torch.float32).reshape(length, 4)
    final = layer_zero + 100 + offset
    return {
        "schema_version": 2,
        "sample_index": source_index,
        "input_ids": torch.arange(length, dtype=torch.long) + offset,
        "attention_mask": torch.ones(length, dtype=torch.long),
        "loss_mask": torch.ones(length, dtype=torch.float32),
        "hidden_states": {0: layer_zero + offset},
        "final_hidden_state": final,
    }


def _write_cache(tmp_path, *, compress=False):
    root = tmp_path / ("compressed" if compress else "plain")
    identity = {"path": "source.jsonl", "sha256": "source"}
    writer = prepare.TarShardWriter(
        root / "shards", "build", 0, compress, "final", identity
    )
    payloads = [_payload(4, 5), _payload(8, 3, offset=10)]
    for ordinal, payload in enumerate(payloads):
        writer.add_sample(
            payload["sample_index"], _serialized(payload), ordinal=ordinal
        )
    extension = ".tar.gz" if compress else ".tar"
    committed = writer.commit(
        {"success_start": 0, "success_end": 1},
        final_filename=f"shard_00000000_00000001{extension}",
    )
    checksum = hashlib.sha256(committed.path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "save_mode": "regen_full",
        "complete": True,
        "target_model": "mock/model",
        "num_hidden_layers": 3,
        "hidden_size": 4,
        "max_length": 8,
        "requested_hidden_layer_ids": [0, 2],
        "stored_hidden_layer_ids": [0, 2],
        "final_norm_layer_id": 2,
        "successful_samples": 2,
        "shards": [
            {
                "path": str(committed.path.relative_to(root)),
                "sample_count": 2,
                "source_indices": [4, 8],
                "success_start": 0,
                "success_end": 1,
                "sha256": checksum,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def test_sample_payload_saves_final_norm_once():
    final = torch.full((1, 3, 4), 9.0)
    output = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        loss_mask=torch.ones((1, 3)),
        hidden_states={0: torch.zeros((1, 3, 4)), 2: final},
    )

    encoded = prepare.sample_payload(output, 0, 7, [0, 2], 2)
    payload = torch.load(io.BytesIO(encoded), weights_only=True)

    assert set(payload["hidden_states"]) == {0}
    assert torch.equal(payload["final_hidden_state"], final[0])
    assert payload["schema_version"] == 2


def test_sglang_final_only_capture_has_no_duplicate_aux_layer():
    aux, final = SGLangDLiteTargetModel._split_aux_and_last_capture_ids(
        [2], num_aux_layers=0, num_transformer_layers=3, has_last_hidden=True
    )
    assert aux == []
    assert final == [2]


@pytest.mark.parametrize("compress", [False, True])
def test_regen_full_dataset_and_collator(compress, tmp_path):
    root, _ = _write_cache(tmp_path, compress=compress)
    manifest = load_hidden_cache_manifest(
        root,
        target_model="mock/model",
        num_hidden_layers=3,
        hidden_size=4,
        required_layer_ids={0, 2},
        max_length=6,
    )
    dataset = RegenFullDataset(root, manifest, required_layer_ids={0, 2}, max_length=6)

    first, second = dataset[0], dataset[1]
    assert set(first["hidden_states"]) == {0, 2}
    assert (
        first["hidden_states"][2].data_ptr() == first["final_hidden_state"].data_ptr()
    )
    batch = RegenFullCollator(pad_token_id=99)([first, second])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["hidden_states"][0].shape == (2, 5, 4)
    assert batch["final_hidden_state"].shape == (2, 5, 4)
    assert batch["input_ids"][1, 3:].tolist() == [99, 99]

    dataset.close()
    worker_batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=2,
                num_workers=2,
                collate_fn=RegenFullCollator(pad_token_id=99),
            )
        )
    )
    assert worker_batch["sample_index"].tolist() == [4, 8]


def test_manifest_rejects_missing_required_layer_and_checksum(tmp_path):
    root, manifest = _write_cache(tmp_path)
    with pytest.raises(HiddenCacheError, match="missing required hidden layers"):
        load_hidden_cache_manifest(
            root,
            target_model="mock/model",
            num_hidden_layers=3,
            hidden_size=4,
            required_layer_ids={0, 1, 2},
            max_length=8,
        )

    shard = root / manifest["shards"][0]["path"]
    with shard.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(HiddenCacheError, match="checksum mismatch"):
        load_hidden_cache_manifest(
            root,
            target_model="mock/model",
            num_hidden_layers=3,
            hidden_size=4,
            required_layer_ids={0, 2},
            max_length=8,
        )


def test_cached_final_hidden_projection_matches_full_logits():
    torch.manual_seed(1)
    final_hidden = torch.randn(2, 7, 4)
    lm_head = torch.nn.Linear(4, 11, bias=False)
    anchors = torch.tensor([[1, 3], [0, 2]])

    expected = gather_target_prefill_logits(lm_head(final_hidden), anchors, 3)
    actual = project_cached_target_logits(final_hidden, anchors, 3, lm_head)

    torch.testing.assert_close(actual, expected)


def test_training_data_arguments_are_exclusive():
    import argparse

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    common = [
        "--target-model-path",
        "mock/model",
        "--output-dir",
        "output",
    ]
    args = parser.parse_args(common + ["--train-hidden-states-path", "cache"])
    validate_common_args(parser, args)

    both = parser.parse_args(
        common
        + [
            "--train-data-path",
            "data.jsonl",
            "--train-hidden-states-path",
            "cache",
        ]
    )
    with pytest.raises(SystemExit):
        validate_common_args(parser, both)

    tp_offline = parser.parse_args(
        common
        + [
            "--train-hidden-states-path",
            "cache",
            "--tp-size",
            "2",
        ]
    )
    with pytest.raises(SystemExit):
        validate_common_args(parser, tp_offline)


def test_offline_resource_build_skips_target_transformer(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("target transformer must not be constructed")

    observed = {}

    def fake_components(args, drafts, target=None, *, standalone_components=False):
        observed["target"] = target
        observed["standalone"] = standalone_components
        return "tokenizer", "components", 7

    monkeypatch.setattr(dlite_training, "build_target_model", fail_if_called)
    monkeypatch.setattr(
        dlite_training, "resolve_tokenizer_and_components", fake_components
    )
    args = SimpleNamespace(train_hidden_states_path="cache")

    target, tokenizer, components, mask_id = dlite_training.build_target_and_components(
        args, [object()]
    )

    assert target is None
    assert (tokenizer, components, mask_id) == ("tokenizer", "components", 7)
    assert observed == {"target": None, "standalone": True}
