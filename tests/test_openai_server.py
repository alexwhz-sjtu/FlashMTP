import json

from evaluation.openai_server import _completion_chunks, parse_tool_calls


def test_parse_qwen_tool_calls() -> None:
    content, calls = parse_tool_calls(
        'I will inspect it.\n<tool_call>\n{"name":"read_file","arguments":{"path":"a.txt"}}\n</tool_call>'
    )
    assert content == "I will inspect it."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}


def test_stream_has_tool_call_and_done() -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
        }
    ]
    chunks = list(
        _completion_chunks(
            "chatcmpl-1",
            "local",
            1,
            None,
            calls,
            "tool_calls",
            {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            True,
        )
    )
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert any('"finish_reason": "tool_calls"' in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
