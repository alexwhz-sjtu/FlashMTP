import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / "scripts" / "data" / "convert_dataset.py"
SPEC = importlib.util.spec_from_file_location("convert_dataset", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def source_args(input_path: str, **overrides):
    values = {
        "input": input_path,
        "input_format": "auto",
        "records_path": None,
        "hf_config": None,
        "hf_filter": None,
        "split": "train",
    }
    values.update(overrides)
    return values


def convert_args(input_path: str, adapter: Path, output: Path, **overrides):
    values = {
        **source_args(input_path),
        "adapter": str(adapter),
        "output": str(output),
        "turn_mode": "multi",
        "limit": None,
        "skip_invalid": False,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_json_adapter_converts_text_and_drops_extra_fields(tmp_path):
    input_path = write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {"question": " First? ", "topic": "math", "unused": "drop"},
            {"question": "Second?", "topic": "code", "unused": 3},
        ],
    )
    adapter = write_json(
        tmp_path / "adapter.json",
        {
            "prompt": {"path": "question", "kind": "text"},
            "category": {"path": "topic"},
        },
    )
    output = tmp_path / "converted.jsonl"

    summary = converter.convert_records(convert_args(str(input_path), adapter, output))

    assert summary == {
        "input": str(input_path),
        "output": str(output),
        "processed": 2,
        "written": 2,
        "skipped": 0,
        "skip_reasons": {},
    }
    assert read_jsonl(output) == [
        {
            "id": 0,
            "conversations": [{"role": "user", "content": "First?"}],
            "source": "questions",
            "category": "math",
        },
        {
            "id": 1,
            "conversations": [{"role": "user", "content": "Second?"}],
            "source": "questions",
            "category": "code",
        },
    ]


@pytest.mark.parametrize(
    ("turn_mode", "expected"),
    [
        (
            "multi",
            [
                {"role": "system", "content": "Be useful"},
                {"role": "user", "content": "One"},
                {"role": "user", "content": "Two"},
            ],
        ),
        (
            "first",
            [
                {"role": "system", "content": "Be useful"},
                {"role": "user", "content": "One"},
            ],
        ),
    ],
)
def test_messages_role_mapping_and_turn_modes(tmp_path, turn_mode, expected):
    input_path = write_jsonl(
        tmp_path / "chat.jsonl",
        [
            {
                "payload": {
                    "dialog": [
                        {"from": "system", "value": "Be useful"},
                        {"from": "human", "value": "One"},
                        {"from": "gpt", "value": "Old answer"},
                        {"from": "tool", "value": "Old tool output"},
                        {"from": "human", "value": "Two"},
                    ]
                },
                "origin": "chat-source",
            }
        ],
    )
    adapter = write_json(
        tmp_path / "chat_adapter.json",
        {
            "prompt": {
                "path": "payload.dialog",
                "kind": "messages",
                "role_path": "from",
                "content_path": "value",
                "role_map": {"human": "user", "gpt": "assistant"},
            },
            "source": {"path": "origin"},
        },
    )
    output = tmp_path / f"{turn_mode}.jsonl"

    converter.convert_records(
        convert_args(str(input_path), adapter, output, turn_mode=turn_mode)
    )

    assert read_jsonl(output)[0]["conversations"] == expected


def test_json_records_path_and_limit(tmp_path):
    input_path = write_json(
        tmp_path / "wrapped.json",
        {"payload": {"rows": [{"prompt": "a"}, {"prompt": "b"}]}, "ignored": []},
    )
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "limited.jsonl"

    args = convert_args(
        str(input_path), adapter, output, records_path="payload.rows", limit=1
    )
    summary = converter.convert_records(args)

    assert summary["processed"] == 1
    assert len(read_jsonl(output)) == 1


def test_python_adapter(tmp_path):
    input_path = write_jsonl(
        tmp_path / "parts.jsonl", [{"instruction": "Add", "input": "two numbers"}]
    )
    adapter = tmp_path / "parts_adapter.py"
    adapter.write_text(
        "def extract(row):\n"
        "    return {'prompt': row['instruction'] + ': ' + row['input'], "
        "'source': 'parts', 'category': None}\n",
        encoding="utf-8",
    )
    output = tmp_path / "parts_out.jsonl"

    converter.convert_records(convert_args(str(input_path), adapter, output))

    assert read_jsonl(output)[0]["conversations"][0]["content"] == "Add: two numbers"


def test_skip_invalid_assigns_contiguous_ids(tmp_path):
    input_path = write_jsonl(
        tmp_path / "mixed.jsonl",
        [{"prompt": "ok"}, {"prompt": "  "}, {"prompt": "also ok"}],
    )
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "mixed_out.jsonl"

    summary = converter.convert_records(
        convert_args(str(input_path), adapter, output, skip_invalid=True)
    )

    assert summary["processed"] == 3
    assert summary["written"] == 2
    assert summary["skipped"] == 1
    assert [row["id"] for row in read_jsonl(output)] == [0, 1]


def test_strict_failure_is_atomic(tmp_path):
    input_path = write_jsonl(tmp_path / "bad.jsonl", [{"prompt": "ok"}, {"prompt": ""}])
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "must_not_exist.jsonl"

    with pytest.raises(converter.ConversionError, match="record 1"):
        converter.convert_records(convert_args(str(input_path), adapter, output))

    assert not output.exists()
    assert not list(tmp_path.glob(".must_not_exist.jsonl.*.tmp"))


def test_malformed_jsonl_fails_without_output(tmp_path):
    input_path = tmp_path / "malformed.jsonl"
    input_path.write_text('{"prompt": "ok"}\nnot-json\n', encoding="utf-8")
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "malformed_out.jsonl"

    with pytest.raises(converter.ConversionError, match="invalid JSON"):
        converter.convert_records(convert_args(str(input_path), adapter, output))

    assert not output.exists()


def test_existing_output_requires_overwrite(tmp_path):
    input_path = write_jsonl(tmp_path / "input.jsonl", [{"prompt": "new"}])
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "output.jsonl"
    output.write_text("old", encoding="utf-8")

    with pytest.raises(converter.ConversionError, match="output already exists"):
        converter.convert_records(convert_args(str(input_path), adapter, output))
    assert output.read_text(encoding="utf-8") == "old"

    converter.convert_records(
        convert_args(str(input_path), adapter, output, overwrite=True)
    )
    assert read_jsonl(output)[0]["conversations"][0]["content"] == "new"


def test_inspect_reports_fields_and_prompt_candidates(tmp_path, capsys):
    input_path = write_jsonl(
        tmp_path / "inspect.jsonl", [{"payload": {"question": "Why?"}, "source": "x"}]
    )
    args = argparse.Namespace(**source_args(str(input_path)), rows=5)

    result = converter.inspect_records(args)
    capsys.readouterr()

    assert result["rows_inspected"] == 1
    assert "payload.question" in result["prompt_candidates"]
    assert any(
        field["path"] == "payload.question" for field in result["rows"][0]["fields"]
    )


def test_hf_source_uses_streaming_and_dataset_id(tmp_path, monkeypatch):
    calls = []

    def fake_load(dataset_id, hf_config, split, hf_filters):
        calls.append((dataset_id, hf_config, split, hf_filters))
        return iter([{"prompt": "from hf"}])

    monkeypatch.setattr(converter, "_load_hf_dataset", fake_load)
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "hf.jsonl"
    args = convert_args(
        "org/dataset",
        adapter,
        output,
        input_format="hf",
        hf_config="subset",
        split="validation",
        hf_filter=[
            ["language", "==", "Chinese"],
            ["model", "==", "gpt-4-0314"],
        ],
    )

    converter.convert_records(args)

    assert calls == [
        (
            "org/dataset",
            "subset",
            "validation",
            [("language", "==", "Chinese"), ("model", "==", "gpt-4-0314")],
        )
    ]
    assert read_jsonl(output)[0]["source"] == "org/dataset"


def test_hf_filter_parses_json_values_and_validates_operators():
    assert converter.normalize_hf_filters(
        [["score", ">=", "0.5"], ["language", "in", '["Chinese", "English"]']]
    ) == [("score", ">=", 0.5), ("language", "in", ["Chinese", "English"])]

    with pytest.raises(converter.ConversionError, match="unsupported"):
        converter.normalize_hf_filters([["language", "contains", "Chinese"]])


def test_hf_filter_is_rejected_for_local_input(tmp_path):
    input_path = write_jsonl(tmp_path / "input.jsonl", [{"prompt": "hello"}])
    filters = converter.normalize_hf_filters([["language", "==", "Chinese"]])

    with pytest.raises(converter.ConversionError, match="only applies"):
        list(
            converter.iter_records(
                str(input_path), "auto", None, None, "train", filters
            )
        )


def test_parquet_is_read_in_batches(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    input_path = tmp_path / "input.parquet"
    parquet.write_table(pyarrow.table({"prompt": ["p1", "p2"]}), input_path)
    adapter = write_json(
        tmp_path / "adapter.json", {"prompt": {"path": "prompt", "kind": "text"}}
    )
    output = tmp_path / "parquet.jsonl"

    converter.convert_records(convert_args(str(input_path), adapter, output))

    assert [row["id"] for row in read_jsonl(output)] == [0, 1]


def test_unmapped_role_and_multimodal_content_are_invalid(tmp_path):
    adapter_path = write_json(
        tmp_path / "adapter.json",
        {"prompt": {"path": "messages", "kind": "messages"}},
    )
    extract = converter.load_adapter(str(adapter_path))

    with pytest.raises(converter.ConversionError, match="unmapped role"):
        extract({"messages": [{"role": "critic", "content": "x"}]})

    with pytest.raises(converter.ConversionError, match="multimodal"):
        converter.normalize_extracted(
            {"messages": [{"role": "user", "content": [{"type": "text"}]}]},
            "fallback",
            "multi",
        )


def test_empty_and_discarded_messages_do_not_reject_valid_prompt():
    normalized = converter.normalize_extracted(
        {
            "messages": [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "   "},
                {"role": "assistant", "content": [{"type": "image"}]},
                {"role": "user", "content": "later prompt"},
            ]
        },
        "fallback",
        "first",
    )

    assert normalized["conversations"] == [
        {"role": "user", "content": "first prompt"}
    ]


def test_messages_with_only_empty_users_are_invalid():
    with pytest.raises(converter.ConversionError, match="no non-empty user turn"):
        converter.normalize_extracted(
            {"messages": [{"role": "user", "content": ""}]},
            "fallback",
            "first",
        )
