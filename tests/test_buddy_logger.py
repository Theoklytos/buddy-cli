"""Tests for buddy.plugins.logger — LoggerPlugin."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buddy.core.events import Event, EventType
from buddy.core.session import Message, Session
from buddy.plugins.logger import LoggerPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tmp_path: Path) -> Session:
    """Return a minimal Session with a predictable session_id."""
    session = Session(session_id="test1234")
    session.plugin_registry = MagicMock()
    return session


def _make_plugin_with_session(tmp_path: Path) -> tuple[LoggerPlugin, Session]:
    """Create and set up a LoggerPlugin with a real temp log dir."""
    plugin = LoggerPlugin()
    session = _make_session(tmp_path)
    log_dir_patch = tmp_path / "buddy_logs"

    with patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir_patch):
        plugin.setup(session)

    return plugin, session


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------

class TestLoggerSetup:
    def test_setup_creates_log_file(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        assert plugin.log_file.exists()

    def test_initial_log_file_is_valid_json(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        data = json.loads(plugin.log_file.read_text())
        assert data["session_id"] == session.session_id

    def test_initial_log_has_empty_exchanges(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        data = json.loads(plugin.log_file.read_text())
        assert data["exchanges"] == []

    def test_initial_log_contains_model(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        data = json.loads(plugin.log_file.read_text())
        assert data["model"] == session.model

    def test_initial_log_contains_config_keys(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        data = json.loads(plugin.log_file.read_text())
        cfg = data["config"]
        assert "system_prompt" in cfg
        assert "temperature" in cfg
        assert "max_tokens" in cfg
        assert "context_depth" in cfg

    def test_exchanges_list_starts_empty(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        assert plugin._exchanges == []


# ---------------------------------------------------------------------------
# on_event() tests
# ---------------------------------------------------------------------------

class TestLoggerOnEvent:
    def test_non_assistant_event_is_ignored(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        plugin.on_event(Event(type=EventType.USER_MESSAGE))
        assert plugin._exchanges == []

    def test_assistant_message_appends_exchange(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        msg = Message(
            role="assistant",
            content="Hello!",
            raw_request={"messages": []},
            raw_response={"content": "Hello!"},
        )
        session.history.append(msg)
        plugin.on_event(Event(type=EventType.ASSISTANT_MESSAGE))
        assert len(plugin._exchanges) == 1

    def test_exchange_has_correct_structure(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        msg = Message(
            role="assistant",
            content="Hi",
            raw_request={"messages": [{"role": "user", "content": "hey"}]},
            raw_response={"content": "Hi"},
        )
        session.history.append(msg)
        plugin.on_event(Event(type=EventType.ASSISTANT_MESSAGE))
        exchange = plugin._exchanges[0]
        assert exchange["turn"] == 1
        assert "timestamp" in exchange
        assert exchange["request"] == msg.raw_request
        assert exchange["response"] == msg.raw_response

    def test_turn_number_increments(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        for i in range(3):
            msg = Message(role="assistant", content=f"msg{i}")
            session.history.append(msg)
            plugin.on_event(Event(type=EventType.ASSISTANT_MESSAGE))
        turns = [e["turn"] for e in plugin._exchanges]
        assert turns == [1, 2, 3]

    def test_log_file_updated_after_event(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        msg = Message(role="assistant", content="Updated")
        session.history.append(msg)
        plugin.on_event(Event(type=EventType.ASSISTANT_MESSAGE))
        data = json.loads(plugin.log_file.read_text())
        assert len(data["exchanges"]) == 1
        assert data["exchanges"][0]["turn"] == 1

    def test_multiple_events_written_to_file(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        for i in range(5):
            msg = Message(role="assistant", content=f"msg{i}")
            session.history.append(msg)
            plugin.on_event(Event(type=EventType.ASSISTANT_MESSAGE))
        data = json.loads(plugin.log_file.read_text())
        assert len(data["exchanges"]) == 5


# ---------------------------------------------------------------------------
# cmd_save() tests
# ---------------------------------------------------------------------------

class TestLoggerCmdSave:
    def test_save_no_args_returns_path(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        result = plugin.cmd_save("", session)
        assert str(plugin.log_file) in result

    def test_save_with_tag_creates_copy(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        result = plugin.cmd_save("mytag", session)
        tagged = plugin.log_dir / f"{session.session_id}_mytag.json"
        assert tagged.exists()
        assert str(tagged) in result

    def test_save_with_path_creates_copy(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        dest = tmp_path / "exports" / "session.json"
        result = plugin.cmd_save(str(dest), session)
        assert dest.exists()
        assert str(dest) in result

    def test_save_preserves_content(self, tmp_path):
        plugin, session = _make_plugin_with_session(tmp_path)
        dest = tmp_path / "copy.json"
        plugin.cmd_save(str(dest), session)
        original = json.loads(plugin.log_file.read_text())
        copy = json.loads(dest.read_text())
        assert original == copy
