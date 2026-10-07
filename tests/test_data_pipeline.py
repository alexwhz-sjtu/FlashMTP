import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / "data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


regen = load_script("regenerate_train_data")
hidden = load_script("prepare_hidden_states")


def write_standard_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def regen_args(input_path: Path, output_path: Path, resume=False):
    return argparse.Namespace(
        model="test-model",
        input_file_path=str(input_path),
        output_file_path=str(output_path),
        server_address=["localhost:30000"],
        num_samples=None,
        concurrency=2,
        temperature=0.7,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        max_tokens=16,
        request_timeout=5.0,
        enable_thinking=False,
        is_reasoning_model=False,
        is_gpt_oss=False,
        resume=resume,
    )


def standard_record(record_id):
    return {
        "id": record_id,
        "conversations": [{"role": "user", "content": f"prompt {record_id}"}],
        "source": "unit",
        "category": None,
    }


def test_default_save_mode_paths():
    regen_path = regen.default_output_path(
        SimpleNamespace(
            input_file_path="/datasets/example.jsonl",
            num_samples=None,
            enable_thinking=False,
            model="org/Model-8B",
        )
    )
    assert regen_path.parent == Path("cache/data/regen_token_only")

    full_path = hidden.default_output_path(
        str(regen_path),
        "org/Model-8B",
    )
    assert full_path.parent == Path("cache/data/regen_full")
    assert full_path.name.endswith("_Model-8B")


def test_regen_failures_are_separate_and_resume_is_id_based(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_standard_jsonl(input_path, [standard_record(0), standard_record(1)])
    calls = []

    monkeypatch.setattr(regen, "check_server", lambda args, server: (True, ""))

    def fake_regenerate(args, server, record):
        calls.append(record["id"])
        if record["id"] == 1:
            return {"id": 1, "source": "unit", "category": None, "error": "failed"}
        return {
            **record,
            "conversations": record["conversations"]
            + [{"role": "assistant", "content": "answer"}],
        }

    monkeypatch.setattr(regen, "regenerate_record", fake_regenerate)
    summary = regen.regenerate(regen_args(input_path, output_path))
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert calls == [0, 1]

    calls.clear()
    write_standard_jsonl(
        input_path, [standard_record(0), standard_record(1), standard_record(2)]
    )
    summary = regen.regenerate(regen_args(input_path, output_path, resume=True))
    assert calls == [2]
    assert summary["already_processed"] == 2


def test_regen_rejects_duplicate_ids(tmp_path, monkeypatch):
    input_path = tmp_path / "duplicates.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_standard_jsonl(input_path, [standard_record("x"), standard_record("x")])
    monkeypatch.setattr(regen, "check_server", lambda args, server: (True, ""))
    monkeypatch.setattr(regen, "regenerate_record", lambda args, server, record: record)

    with pytest.raises(regen.RegenerationError, match="duplicate input id"):
        regen.regenerate(regen_args(input_path, output_path))


def test_regenerate_record_replaces_assistant_turns_and_keeps_thinking(monkeypatch):
    prompts = []

    class Completions:
        def create(self, **kwargs):
            prompts.append(list(kwargs["messages"]))
            number = len(prompts)
            message = SimpleNamespace(
                content=f"new answer {number}", reasoning_content=f"thought {number}"
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    args = regen_args(Path("input.jsonl"), Path("output.jsonl"))
    args.is_reasoning_model = True
    record = standard_record(7)
    record["conversations"] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "second"},
    ]

    result = regen.regenerate_record(args, "http://server", record)

    assert [message["content"] for message in result["conversations"]] == [
        "system",
        "first",
        "new answer 1",
        "second",
        "new answer 2",
    ]
    assert result["conversations"][2]["thinking"] == "thought 1"
    assert prompts[1][2]["content"] == "new answer 1"


def make_identity(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    return hidden.input_fingerprint(source)


def write_staging_rank(root, identity, rank, source_indices, failed, shard_size=512):
    sequence = 0
    writer = hidden.TarShardWriter(
        root / "staging" / f"rank_{rank:05d}",
        "stage",
        sequence,
        False,
        "staging",
        identity,
    )
    success_count = 0
    for source_index in source_indices:
        if source_index in failed:
            writer.add_failure(
                {
                    "source_index": source_index,
                    "stage": "target",
                    "error_type": "SyntheticError",
                    "error": "synthetic failure",
                }
            )
            continue
        writer.add_sample(source_index, f"payload-{source_index}".encode())
        success_count += 1
        if success_count == shard_size:
            writer.commit({"rank": rank})
            sequence += 1
            writer = hidden.TarShardWriter(
                root / "staging" / f"rank_{rank:05d}",
                "stage",
                sequence,
                False,
                "staging",
                identity,
            )
            success_count = 0
    if writer.consumed_start is not None:
        writer.commit({"rank": rank, "final_local": True})
    else:
        writer.abort()
    hidden.write_rank_errors(root, rank, hidden.discover_staging(root, identity, rank))


def test_success_shards_fill_to_512_across_failures_and_ranks(tmp_path):
    root = tmp_path / "cache"
    identity = make_identity(tmp_path)
    failures = {3, 514, 1025}
    write_staging_rank(root, identity, 0, range(0, 550), failures)
    write_staging_rank(root, identity, 1, range(550, 1100), failures)

    manifest = hidden.finalize_staging(
        root,
        identity,
        shard_size=512,
        compress=False,
        run_metadata={"target_model": "mock"},
    )

    assert manifest["successful_samples"] == 1097
    assert manifest["failed_samples"] == 3
    assert [item["sample_count"] for item in manifest["shards"]] == [512, 512, 73]
    assert [Path(item["path"]).name for item in manifest["shards"]] == [
        "shard_00000000_00000511.tar",
        "shard_00000512_00001023.tar",
        "shard_00001024_00001096.tar",
    ]
    all_source_indices = [
        source_index
        for item in manifest["shards"]
        for source_index in item["source_indices"]
    ]
    assert not failures.intersection(all_source_indices)
    assert all_source_indices[:5] == [0, 1, 2, 4, 5]
    assert not (root / "staging").exists()
    assert len((root / "errors.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    for item in manifest["shards"]:
        shard_path = root / item["path"]
        assert hidden.sha256_file(shard_path) == item["sha256"]
        assert (
            hidden.read_shard_metadata(shard_path)["sample_count"]
            == item["sample_count"]
        )


def test_repair_staging_discards_corrupt_shard_and_later_work(tmp_path):
    root = tmp_path / "cache"
    identity = make_identity(tmp_path)
    write_staging_rank(root, identity, 0, range(6), set(), shard_size=3)
    paths = sorted((root / "staging" / "rank_00000").glob("stage_*.tar"))
    paths[1].write_bytes(b"corrupt")

    valid = hidden.repair_staging(root, identity, rank=0, expected_start=0)

    assert len(valid) == 1
    assert valid[0].metadata["consumed_end"] == 2
    assert not paths[1].exists()


def test_final_shard_checksum_controls_resume_prefix(tmp_path):
    root = tmp_path / "cache"
    identity = make_identity(tmp_path)
    write_staging_rank(root, identity, 0, range(1024), set())
    hidden.finalize_staging(root, identity, 512, False, {"target_model": "mock"})
    valid = hidden.valid_final_shards(root, 512, identity)
    assert len(valid) == 2

    checksum = valid[1].path.with_name(f"{valid[1].path.name}.sha256")
    checksum.write_text("wrong\n", encoding="utf-8")
    assert len(hidden.valid_final_shards(root, 512, identity)) == 1


def test_contiguous_dp_ranges_cover_every_sample_once():
    ranges = [hidden.contiguous_range(10, rank, 3) for rank in range(3)]
    assert ranges == [(0, 4), (4, 7), (7, 10)]
    assert [index for start, end in ranges for index in range(start, end)] == list(
        range(10)
    )
