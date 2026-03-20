"""Tests for bud.stages.parse."""

import json
import os
import time
from pathlib import Path
import pytest

from bud.stages.parse import parse_conversations_file, parse_all, _extract_blocks


SAMPLE_CONVERSATION = [
    {
        "uuid": "conv-1",
        "name": "Test Conversation",
        "summary": "A test",
        "created_at": "2024-01-01T00:00:00Z",
        "chat_messages": [
            {
                "uuid": "msg-1",
                "sender": "human",
                "text": "Hello",
                "content": [{"type": "text", "text": "Hello"}],
                "created_at": "2024-01-01T00:00:01Z",
            },
            {
                "uuid": "msg-2",
                "sender": "assistant",
                "text": "Hi there",
                "content": [{"type": "text", "text": "Hi there"}],
                "created_at": "2024-01-01T00:00:02Z",
            },
        ],
    }
]


def test_parse_conversations_file_returns_list(tmp_path):
    f = tmp_path / "conversations_001.json"
    f.write_text(json.dumps(SAMPLE_CONVERSATION))
    result = parse_conversations_file(f, memory_context=None)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["id"] == "conv-1"
    assert result[0]["conversation_name"] == "Test Conversation"
    assert len(result[0]["turns"]) == 2


def test_parse_extracts_turn_text(tmp_path):
    f = tmp_path / "conversations_001.json"
    f.write_text(json.dumps(SAMPLE_CONVERSATION))
    result = parse_conversations_file(f, memory_context=None)
    assert result[0]["turns"][0]["text"] == "Hello"
    assert result[0]["turns"][1]["text"] == "Hi there"


def test_parse_includes_memory_context(tmp_path):
    f = tmp_path / "conversations_001.json"
    f.write_text(json.dumps(SAMPLE_CONVERSATION))
    result = parse_conversations_file(f, memory_context="some memory")
    assert result[0]["memory_context"] == "some memory"


def test_parse_returns_empty_for_non_list(tmp_path):
    f = tmp_path / "conversations_bad.json"
    f.write_text(json.dumps({"not": "a list"}))
    result = parse_conversations_file(f, memory_context=None)
    assert result == []


def test_parse_skips_conversations_with_no_messages(tmp_path):
    empty_msg = [{"uuid": "c1", "name": "empty", "chat_messages": []}]
    f = tmp_path / "conversations_empty.json"
    f.write_text(json.dumps(empty_msg))
    result = parse_conversations_file(f, memory_context=None)
    assert result == []


def test_extract_blocks_text():
    content = [{"type": "text", "text": "hello"}]
    text, thinking, truncated, dropped = _extract_blocks(content, "fallback")
    assert text == "hello"
    assert thinking == ""
    assert truncated is False
    assert dropped == 0


def test_extract_blocks_drops_tool_use():
    content = [
        {"type": "tool_use", "name": "some_tool"},
        {"type": "text", "text": "response"},
    ]
    text, thinking, truncated, dropped = _extract_blocks(content, "fallback")
    assert text == "response"
    assert dropped == 1


def test_extract_blocks_uses_fallback_when_no_text():
    content = [{"type": "tool_use", "name": "tool"}]
    text, thinking, truncated, dropped = _extract_blocks(content, "fallback text")
    assert text == "fallback text"


def test_extract_blocks_thinking():
    content = [
        {"type": "thinking", "thinking": "inner thought"},
        {"type": "text", "text": "response"},
    ]
    text, thinking, truncated, dropped = _extract_blocks(content, "")
    assert thinking == "inner thought"
    assert text == "response"


# ---------------------------------------------------------------------------
# parse_all tests
# ---------------------------------------------------------------------------

def test_parse_all_basic(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "parsed"

    (data_dir / "conversations_001.json").write_text(json.dumps(SAMPLE_CONVERSATION))

    total = parse_all(data_dir, out_dir)
    assert total == 1

    jsonl = out_dir / "conversations_001.jsonl"
    assert jsonl.exists()
    lines = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["id"] == "conv-1"


def test_parse_all_skips_up_to_date(tmp_path):
    """parse_all without force=True should skip files whose JSONL is newer."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "parsed"

    src = data_dir / "conversations_001.json"
    src.write_text(json.dumps(SAMPLE_CONVERSATION))

    # First parse
    parse_all(data_dir, out_dir)
    jsonl = out_dir / "conversations_001.jsonl"
    first_mtime = jsonl.stat().st_mtime

    # Ensure filesystem timestamp granularity
    time.sleep(0.05)

    # Second parse — should skip (output is newer than source)
    total = parse_all(data_dir, out_dir)
    assert total == 1  # still counted
    assert jsonl.stat().st_mtime == first_mtime  # file untouched


def test_parse_all_force_reparses(tmp_path):
    """parse_all with force=True should re-parse even if output exists."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "parsed"

    src = data_dir / "conversations_001.json"
    src.write_text(json.dumps(SAMPLE_CONVERSATION))

    parse_all(data_dir, out_dir)
    jsonl = out_dir / "conversations_001.jsonl"
    first_mtime = jsonl.stat().st_mtime

    time.sleep(0.05)

    total = parse_all(data_dir, out_dir, force=True)
    assert total == 1
    assert jsonl.stat().st_mtime > first_mtime  # file was rewritten


def test_parse_all_reparses_when_source_newer(tmp_path):
    """parse_all should re-parse when the source JSON is newer than the output."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "parsed"

    src = data_dir / "conversations_001.json"
    src.write_text(json.dumps(SAMPLE_CONVERSATION))

    parse_all(data_dir, out_dir)
    jsonl = out_dir / "conversations_001.jsonl"
    first_mtime = jsonl.stat().st_mtime

    time.sleep(0.05)

    # Touch the source so it's newer
    src.write_text(json.dumps(SAMPLE_CONVERSATION))

    total = parse_all(data_dir, out_dir)
    assert total == 1
    assert jsonl.stat().st_mtime > first_mtime  # re-parsed
