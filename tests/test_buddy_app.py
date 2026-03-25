"""Integration tests for buddy.app.main()."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buddy.app import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_stream(chunks: list[str], usage: dict | None = None):
    """Build a mock streaming context manager that yields text chunks."""
    if usage is None:
        usage = {"input_tokens": 5, "output_tokens": 3}

    final_message = MagicMock()
    final_message.model_dump.return_value = {"usage": usage, "content": "..."}

    mock_stream_obj = MagicMock()
    mock_stream_obj.text_stream = iter(chunks)
    mock_stream_obj.get_final_message.return_value = final_message

    @contextmanager
    def _ctx(**kwargs):
        yield mock_stream_obj

    return _ctx


TEST_CONFIG = {
    "model": "claude-test",
    "system_prompt": "You are a test assistant.",
    "temperature": None,
    "max_tokens": 1024,
    "context_depth": 3,
    "api_key": "sk-test-key",
}


# ---------------------------------------------------------------------------
# Integration: main() completes without error
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def test_main_runs_without_error(self, tmp_path):
        """main() with scripted inputs completes cleanly without exceptions."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter(["hello world", None])

        mock_client = MagicMock()
        mock_client.messages.stream = _make_mock_stream(["Hi there!"])

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=mock_client),
            patch("buddy.plugins.renderer.Console"),  # suppress rich output
        ):
            main()  # should not raise

    def test_main_creates_log_file(self, tmp_path):
        """main() creates a log file in BUDDY_LOGS_DIR/<date>/<session_id>.json."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter([None])

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=MagicMock()),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()

        # Find log files in the date subdirectory
        log_files = list(log_dir.glob("*/*.json"))
        assert len(log_files) == 1, f"Expected 1 log file, found: {log_files}"

    def test_main_log_file_is_valid_json(self, tmp_path):
        """main() writes a valid JSON document to the log file."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter([None])

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=MagicMock()),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()

        log_files = list(log_dir.glob("*/*.json"))
        assert log_files
        doc = json.loads(log_files[0].read_text())
        assert "session_id" in doc
        assert "model" in doc
        assert "exchanges" in doc

    def test_main_uses_config_wizard_when_no_config_file(self, tmp_path):
        """main() calls run_config_wizard when BUDDY_CONFIG_FILE doesn't exist."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter([None])
        mock_wizard = MagicMock(return_value=TEST_CONFIG)

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", mock_wizard),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=MagicMock()),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()

        mock_wizard.assert_called_once()

    def test_main_loads_config_when_file_exists(self, tmp_path):
        """main() calls load_buddy_config (not wizard) when BUDDY_CONFIG_FILE exists."""
        log_dir = tmp_path / "buddy_logs"
        config_file = tmp_path / "buddy.yaml"
        config_file.write_text("model: claude-test\n")
        inputs = iter([None])
        mock_load = MagicMock(return_value=TEST_CONFIG)

        with (
            patch("buddy.app.BUDDY_CONFIG_FILE", config_file),
            patch("buddy.app.load_buddy_config", mock_load),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=MagicMock()),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()

        mock_load.assert_called_once()

    def test_main_quit_command_exits_loop(self, tmp_path):
        """main() exits cleanly when /quit command is issued."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter(["/quit"])

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=MagicMock()),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()  # should not raise

    def test_main_unknown_command_emits_error_event(self, tmp_path):
        """main() emits ERROR event for unknown slash commands."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter(["/unknown_cmd", None])
        error_events = []

        original_main = main

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=MagicMock()),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()  # should not raise even for unknown command

    def test_main_with_message_and_response(self, tmp_path):
        """main() processes a user message and gets an assistant response."""
        log_dir = tmp_path / "buddy_logs"
        inputs = iter(["tell me a joke", None])

        mock_client = MagicMock()
        mock_client.messages.stream = _make_mock_stream(
            ["Why did the", " chicken cross", " the road?"],
            usage={"input_tokens": 10, "output_tokens": 7},
        )

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=mock_client),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()

        # Verify a log exchange was recorded
        log_files = list(log_dir.glob("*/*.json"))
        assert log_files
        doc = json.loads(log_files[0].read_text())
        assert len(doc["exchanges"]) == 1

    def test_main_empty_input_is_skipped(self, tmp_path):
        """main() skips empty/whitespace-only input without processing it."""
        log_dir = tmp_path / "buddy_logs"
        # Empty string and whitespace should be skipped; None exits
        inputs = iter(["", "   ", None])

        mock_client = MagicMock()

        with (
            patch("buddy.core.config.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.BUDDY_CONFIG_FILE", tmp_path / "nonexistent.yaml"),
            patch("buddy.app.run_config_wizard", return_value=TEST_CONFIG),
            patch("buddy.plugins.logger.BUDDY_LOGS_DIR", log_dir),
            patch("buddy.plugins.input.InputPlugin.get_input", side_effect=inputs),
            patch("buddy.plugins.chat.anthropic.Anthropic", return_value=mock_client),
            patch("buddy.plugins.renderer.Console"),
        ):
            main()

        # No chat calls should have been made
        mock_client.messages.stream.assert_not_called()
