"""Tests for bud.mcp_logger."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import bud.mcp_logger as logger_module


def test_logger_creates_session_directory(tmp_path, monkeypatch):
    """Test that MCPLogger creates the sessions directory."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()

    assert logger_module.SESSIONS_DIR.exists()
    assert logger_module.SESSIONS_DIR.is_dir()


def test_logger_creates_session_file(tmp_path, monkeypatch):
    """Test that start_session creates the session JSONL file."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("test-session-123")

    assert (logger_module.SESSIONS_DIR / "test-session-123.jsonl").exists()


def test_logger_logs_tool_call(tmp_path, monkeypatch):
    """Test that log_tool_call writes entries to the JSONL file."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("session-456")
    logger.log_tool_call("bud_recall", {"query": "test"}, {"results": []}, 100)

    log_file = logger_module.SESSIONS_DIR / "session-456.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["tool_name"] == "bud_recall"
    assert entry["duration_ms"] == 100
    assert entry["session_id"] == "session-456"
    assert "timestamp" in entry
    assert entry["parameters"] == {"query": "test"}
    assert entry["result"] == {"results": []}


def test_logger_generates_session_id_if_not_provided(tmp_path, monkeypatch):
    """Test that a session ID is generated if not provided."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    session_id = logger.start_session()

    assert session_id is not None
    assert session_id.startswith("mcp-")
    assert len(session_id) == 16  # mcp- + 12 hex chars


def test_logger_loads_session_id_from_persisted_file(tmp_path, monkeypatch):
    """Test that session ID is loaded from .current_session file."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    # Pre-create a session file
    persisted_id = "mcp-preloaded-123"
    (tmp_path / "mcp_logs").mkdir(parents=True)
    (tmp_path / "mcp_logs" / ".current_session").write_text(persisted_id)

    logger = logger_module.MCPLogger()
    session_id = logger.start_session()

    assert session_id == persisted_id


def test_logger_persists_session_id(tmp_path, monkeypatch):
    """Test that session ID is persisted to .current_session file."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("session-789")

    persisted = (tmp_path / "mcp_logs" / ".current_session").read_text().strip()
    assert persisted == "session-789"


def test_logger_ends_session_clears_state(tmp_path, monkeypatch):
    """Test that end_session clears internal state."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("session-999")
    logger.end_session()

    assert logger._current_session_id is None
    assert logger._session_file is None


def test_logger_multiple_tool_calls_append_to_file(tmp_path, monkeypatch):
    """Test that multiple tool calls append to the same file."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("session-multi")
    logger.log_tool_call("tool1", {"a": 1}, {}, 10)
    logger.log_tool_call("tool2", {"b": 2}, {}, 20)

    log_file = logger_module.SESSIONS_DIR / "session-multi.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    entry2 = json.loads(lines[1])

    assert entry1["tool_name"] == "tool1"
    assert entry2["tool_name"] == "tool2"


def test_logger_with_custom_session_id(tmp_path, monkeypatch):
    """Test logging with a custom session ID."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("my-custom-session")
    logger.log_tool_call("test_tool", {}, {}, 50.5)

    log_file = logger_module.SESSIONS_DIR / "my-custom-session.jsonl"
    entry = json.loads(log_file.read_text().strip())
    assert entry["session_id"] == "my-custom-session"
    assert entry["duration_ms"] == 50.5


def test_global_logger_instance_exists():
    """Test that the global logger instance is created."""
    assert hasattr(logger_module, '_logger')
    assert isinstance(logger_module._logger, logger_module.MCPLogger)


def test_convenience_functions_exist():
    """Test that convenience functions are available."""
    assert callable(logger_module.get_logger)
    assert callable(logger_module.start_logging_session)
    assert callable(logger_module.log_tool_call)
    assert callable(logger_module.end_logging_session)


def test_get_logger_returns_global_instance():
    """Test that get_logger returns the global instance."""
    assert logger_module.get_logger() is logger_module._logger


def test_timestamp_format_is_utc_isoformat(tmp_path, monkeypatch):
    """Test that timestamps are in UTC ISO format."""
    monkeypatch.setattr(logger_module, 'SESSIONS_DIR', tmp_path / "mcp_logs")
    monkeypatch.setattr(logger_module, 'CURRENT_SESSION_FILE', tmp_path / "mcp_logs" / ".current_session")

    logger = logger_module.MCPLogger()
    logger.start_session("session-time")
    logger.log_tool_call("bud_recall", {}, {}, 100)

    log_file = logger_module.SESSIONS_DIR / "session-time.jsonl"
    entry = json.loads(log_file.read_text().strip())

    # Should be ISO format with timezone
    timestamp = entry["timestamp"]
    assert "T" in timestamp  # ISO format separator
    assert timestamp.endswith("Z") or "+" in timestamp or "-" in timestamp  # Has timezone info
