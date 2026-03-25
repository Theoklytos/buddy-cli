"""Tests for buddy.plugins.renderer and buddy.plugins.input."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from buddy.core.events import Event, EventType
from buddy.core.session import Session
from buddy.plugins.renderer import RendererPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_session() -> Session:
    return Session()


def make_plugin_with_mock_console() -> tuple[RendererPlugin, MagicMock]:
    plugin = RendererPlugin()
    session = make_session()
    with patch("buddy.plugins.renderer.Console") as MockConsole:
        mock_console = MagicMock()
        MockConsole.return_value = mock_console
        plugin.setup(session)
    return plugin, mock_console


# ---------------------------------------------------------------------------
# RendererPlugin — instantiation & setup
# ---------------------------------------------------------------------------


class TestRendererPluginSetup:
    def test_name(self):
        assert RendererPlugin.name == "renderer"

    def test_commands_empty(self):
        assert RendererPlugin.commands == {}

    def test_setup_stores_session(self):
        plugin, _ = make_plugin_with_mock_console()
        assert plugin.session is not None

    def test_setup_creates_console(self):
        plugin, mock_console = make_plugin_with_mock_console()
        assert plugin.console is mock_console

    def test_stream_start_is_none_after_setup(self):
        plugin, _ = make_plugin_with_mock_console()
        assert plugin._stream_start is None


# ---------------------------------------------------------------------------
# SESSION_START event
# ---------------------------------------------------------------------------


class TestSessionStartEvent:
    def test_prints_welcome_panel(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.SESSION_START))
        assert mock_console.print.called
        args, _ = mock_console.print.call_args
        assert isinstance(args[0], Panel)

    def test_welcome_panel_contains_version(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.SESSION_START))
        args, _ = mock_console.print.call_args
        panel = args[0]
        assert "v0.1.0" in str(panel.renderable)

    def test_welcome_panel_contains_keybindings(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.SESSION_START))
        args, _ = mock_console.print.call_args
        panel = args[0]
        rendered = str(panel.renderable)
        assert "Ctrl+X" in rendered
        assert "Ctrl+Q" in rendered


# ---------------------------------------------------------------------------
# ASSISTANT_TOKENS event
# ---------------------------------------------------------------------------


class TestAssistantTokensEvent:
    def test_sets_stream_start_on_first_token(self):
        plugin, mock_console = make_plugin_with_mock_console()
        assert plugin._stream_start is None
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1000.0
            with patch("builtins.print"):
                plugin.on_event(
                    Event(type=EventType.ASSISTANT_TOKENS, payload={"delta": "Hello"})
                )
        assert plugin._stream_start == 1000.0

    def test_does_not_reset_stream_start_on_subsequent_tokens(self):
        plugin, mock_console = make_plugin_with_mock_console()
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.side_effect = [1000.0, 1001.0]
            with patch("builtins.print"):
                plugin.on_event(
                    Event(type=EventType.ASSISTANT_TOKENS, payload={"delta": "Hello"})
                )
                plugin.on_event(
                    Event(type=EventType.ASSISTANT_TOKENS, payload={"delta": " world"})
                )
        assert plugin._stream_start == 1000.0

    def test_prints_delta_without_newline(self):
        plugin, _ = make_plugin_with_mock_console()
        plugin._stream_start = 999.0  # already set
        with patch("builtins.print") as mock_print:
            plugin.on_event(
                Event(type=EventType.ASSISTANT_TOKENS, payload={"delta": "abc"})
            )
        mock_print.assert_called_with("abc", end="", flush=True)

    def test_prints_blank_line_on_first_token(self):
        plugin, _ = make_plugin_with_mock_console()
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 50.0
            with patch("builtins.print") as mock_print:
                plugin.on_event(
                    Event(type=EventType.ASSISTANT_TOKENS, payload={"delta": "Hi"})
                )
        # First call should be print() (empty line), second call the delta
        calls = mock_print.call_args_list
        assert calls[0] == call()
        assert calls[1] == call("Hi", end="", flush=True)


# ---------------------------------------------------------------------------
# ASSISTANT_MESSAGE event
# ---------------------------------------------------------------------------


class TestAssistantMessageEvent:
    def _make_message_event(self, text="Hello **world**", model="claude-test", usage=None):
        if usage is None:
            usage = {"input_tokens": 10, "output_tokens": 20}
        return Event(
            type=EventType.ASSISTANT_MESSAGE,
            payload={"text": text, "model": model, "usage": usage},
        )

    def test_prints_panel_with_markdown(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin._stream_start = 1000.0
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1001.2
            with patch("builtins.print"):
                plugin.on_event(self._make_message_event())

        # console.print called at least twice: panel + footer
        assert mock_console.print.call_count >= 2
        first_call_arg = mock_console.print.call_args_list[0][0][0]
        assert isinstance(first_call_arg, Panel)

    def test_panel_has_blue_border(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin._stream_start = 1000.0
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1001.0
            with patch("builtins.print"):
                plugin.on_event(self._make_message_event())

        panel = mock_console.print.call_args_list[0][0][0]
        assert panel.border_style == "blue"

    def test_panel_title_is_buddy(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin._stream_start = 1000.0
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1001.0
            with patch("builtins.print"):
                plugin.on_event(self._make_message_event())

        panel = mock_console.print.call_args_list[0][0][0]
        assert panel.title == "buddy"

    def test_footer_contains_model_and_tokens(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin._stream_start = 1000.0
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1001.5
            with patch("builtins.print"):
                plugin.on_event(
                    self._make_message_event(
                        model="claude-sonnet-4-20250514",
                        usage={"input_tokens": 127, "output_tokens": 245},
                    )
                )

        footer_call = mock_console.print.call_args_list[1]
        footer_text = footer_call[0][0]
        assert "claude-sonnet-4-20250514" in footer_text
        assert "127 in" in footer_text
        assert "245 out" in footer_text
        assert "1.5s" in footer_text

    def test_resets_stream_start_to_none(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin._stream_start = 1000.0
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1001.0
            with patch("builtins.print"):
                plugin.on_event(self._make_message_event())
        assert plugin._stream_start is None

    def test_missing_usage_uses_zero_counts(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin._stream_start = 1000.0
        with patch("buddy.plugins.renderer.time") as mock_time:
            mock_time.time.return_value = 1000.5
            with patch("builtins.print"):
                plugin.on_event(
                    Event(
                        type=EventType.ASSISTANT_MESSAGE,
                        payload={"text": "hi", "model": "m"},
                    )
                )
        footer_text = mock_console.print.call_args_list[1][0][0]
        assert "0 in" in footer_text
        assert "0 out" in footer_text


# ---------------------------------------------------------------------------
# COMMAND_EXECUTED event
# ---------------------------------------------------------------------------


class TestCommandExecutedEvent:
    def test_prints_result_when_present(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(
            Event(type=EventType.COMMAND_EXECUTED, payload={"result": "done!"})
        )
        mock_console.print.assert_called_once_with("done!")

    def test_no_print_when_result_is_none(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(
            Event(type=EventType.COMMAND_EXECUTED, payload={"result": None})
        )
        mock_console.print.assert_not_called()

    def test_no_print_when_result_key_absent(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.COMMAND_EXECUTED, payload={}))
        mock_console.print.assert_not_called()


# ---------------------------------------------------------------------------
# ERROR event
# ---------------------------------------------------------------------------


class TestErrorEvent:
    def test_prints_error_panel(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(
            Event(type=EventType.ERROR, payload={"message": "Something went wrong"})
        )
        assert mock_console.print.called
        args, _ = mock_console.print.call_args
        assert isinstance(args[0], Panel)

    def test_error_panel_has_red_border(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(
            Event(type=EventType.ERROR, payload={"message": "boom"})
        )
        panel = mock_console.print.call_args[0][0]
        assert panel.border_style == "red"

    def test_error_panel_title_is_error(self):
        from rich.panel import Panel

        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(
            Event(type=EventType.ERROR, payload={"message": "boom"})
        )
        panel = mock_console.print.call_args[0][0]
        assert panel.title == "Error"

    def test_error_panel_contains_message(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(
            Event(type=EventType.ERROR, payload={"message": "disk full"})
        )
        panel = mock_console.print.call_args[0][0]
        assert "disk full" in panel.renderable

    def test_error_fallback_when_no_message_key(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.ERROR, payload={"code": 42}))
        assert mock_console.print.called


# ---------------------------------------------------------------------------
# Unhandled event types — no crash
# ---------------------------------------------------------------------------


class TestUnhandledEvents:
    def test_unknown_event_type_does_not_raise(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.USER_MESSAGE, payload={"text": "hello"}))
        mock_console.print.assert_not_called()

    def test_session_end_does_not_raise(self):
        plugin, mock_console = make_plugin_with_mock_console()
        plugin.on_event(Event(type=EventType.SESSION_END))
        mock_console.print.assert_not_called()


# ---------------------------------------------------------------------------
# InputPlugin — basic instantiation (prompt_toolkit is hard to unit-test)
# ---------------------------------------------------------------------------


class TestInputPluginInstantiation:
    def test_import(self):
        from buddy.plugins.input import InputPlugin  # noqa: F401

    def test_name(self):
        from buddy.plugins.input import InputPlugin

        assert InputPlugin.name == "input"

    def test_commands_empty(self):
        from buddy.plugins.input import InputPlugin

        assert InputPlugin.commands == {}

    def test_instantiates_without_error(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        assert plugin is not None

    def test_setup_creates_prompt_session(self):
        from buddy.plugins.input import InputPlugin
        from prompt_toolkit import PromptSession

        plugin = InputPlugin()
        session = make_session()
        with patch("buddy.plugins.input.PromptSession") as MockPS:
            mock_ps = MagicMock()
            MockPS.return_value = mock_ps
            plugin.setup(session)

        assert plugin.prompt_session is mock_ps

    def test_setup_stores_session(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        session = make_session()
        with patch("buddy.plugins.input.PromptSession"):
            plugin.setup(session)
        assert plugin.session is session

    def test_get_input_returns_none_on_eoferror(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        plugin.prompt_session = MagicMock()
        plugin.prompt_session.prompt.side_effect = EOFError
        assert plugin.get_input() is None

    def test_get_input_returns_none_on_keyboard_interrupt(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        plugin.prompt_session = MagicMock()
        plugin.prompt_session.prompt.side_effect = KeyboardInterrupt
        assert plugin.get_input() is None

    def test_get_input_returns_none_on_system_exit(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        plugin.prompt_session = MagicMock()
        plugin.prompt_session.prompt.side_effect = SystemExit
        assert plugin.get_input() is None

    def test_get_input_returns_text(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        plugin.prompt_session = MagicMock()
        plugin.prompt_session.prompt.return_value = "hello world"
        assert plugin.get_input() == "hello world"

    def test_get_input_prompts_with_correct_string(self):
        from buddy.plugins.input import InputPlugin

        plugin = InputPlugin()
        plugin.prompt_session = MagicMock()
        plugin.prompt_session.prompt.return_value = "x"
        plugin.get_input()
        plugin.prompt_session.prompt.assert_called_once_with("You > ")
