"""Tests for buddy.plugins.chat.ChatPlugin."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

import pytest

from buddy.core.events import Event, EventBus, EventType
from buddy.core.session import Session
from buddy.plugins.chat import ChatPlugin


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_session(**overrides) -> Session:
    """Return a Session wired to a real EventBus with default config."""
    session = Session(
        model="claude-test-model",
        system_prompt="You are a test assistant.",
        temperature=None,
        max_tokens=1024,
        context_depth=3,
        config={},
        event_bus=EventBus(),
    )
    for k, v in overrides.items():
        setattr(session, k, v)
    return session


def _make_plugin(session: Session | None = None) -> tuple[ChatPlugin, Session]:
    """Return a ChatPlugin set up against a session."""
    if session is None:
        session = _make_session()
    plugin = ChatPlugin()
    with patch("buddy.plugins.chat.anthropic.Anthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        plugin.setup(session)
    return plugin, session


def _mock_stream(chunks: list[str], usage: dict | None = None):
    """Build a mock stream context manager that yields text chunks."""
    if usage is None:
        usage = {"input_tokens": 10, "output_tokens": 5}

    final_message = MagicMock()
    final_message.model_dump.return_value = {"usage": usage, "content": "..."}

    mock_stream_obj = MagicMock()
    mock_stream_obj.text_stream = iter(chunks)
    mock_stream_obj.get_final_message.return_value = final_message

    @contextmanager
    def _ctx(**kwargs):
        yield mock_stream_obj

    return _ctx, mock_stream_obj, final_message


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    def test_stores_session_reference(self):
        session = _make_session()
        plugin = ChatPlugin()
        with patch("buddy.plugins.chat.anthropic.Anthropic"):
            plugin.setup(session)
        assert plugin.session is session

    def test_resolve_api_key_called_with_session_config(self):
        session = _make_session()
        session.config = {"api_key": "sk-test"}
        plugin = ChatPlugin()
        with (
            patch("buddy.plugins.chat.resolve_api_key", return_value="sk-test") as mock_resolve,
            patch("buddy.plugins.chat.anthropic.Anthropic"),
        ):
            plugin.setup(session)
        mock_resolve.assert_called_once_with(session.config)

    def test_anthropic_client_created_with_api_key(self):
        session = _make_session()
        plugin = ChatPlugin()
        with (
            patch("buddy.plugins.chat.resolve_api_key", return_value="sk-abc"),
            patch("buddy.plugins.chat.anthropic.Anthropic") as mock_cls,
        ):
            plugin.setup(session)
        mock_cls.assert_called_once_with(api_key="sk-abc")


# ---------------------------------------------------------------------------
# on_event() routing
# ---------------------------------------------------------------------------

class TestOnEvent:
    def test_user_message_event_triggers_handler(self):
        plugin, session = _make_plugin()
        plugin._handle_user_message = MagicMock()
        event = Event(EventType.USER_MESSAGE, {"text": "hello"})
        plugin.on_event(event)
        plugin._handle_user_message.assert_called_once_with("hello")

    def test_other_events_ignored(self):
        plugin, session = _make_plugin()
        plugin._handle_user_message = MagicMock()
        for etype in [
            EventType.SESSION_START,
            EventType.SESSION_END,
            EventType.ASSISTANT_MESSAGE,
            EventType.ERROR,
        ]:
            plugin.on_event(Event(etype, {}))
        plugin._handle_user_message.assert_not_called()


# ---------------------------------------------------------------------------
# Request envelope construction
# ---------------------------------------------------------------------------

class TestRequestEnvelope:
    def test_envelope_uses_session_fields(self):
        session = _make_session(model="claude-haiku", max_tokens=512, temperature=None)
        plugin, session = _make_plugin(session)

        captured_kwargs = {}

        @contextmanager
        def fake_stream(**kwargs):
            captured_kwargs.update(kwargs)
            mock = MagicMock()
            mock.text_stream = iter(["hi"])
            final = MagicMock()
            final.model_dump.return_value = {"usage": {}}
            mock.get_final_message.return_value = final
            yield mock

        plugin.client.messages.stream = fake_stream
        plugin._handle_user_message("test")

        assert captured_kwargs["model"] == "claude-haiku"
        assert captured_kwargs["max_tokens"] == 512
        assert captured_kwargs["system"] == session.system_prompt
        assert "temperature" not in captured_kwargs

    def test_temperature_included_when_set(self):
        session = _make_session(temperature=0.7)
        plugin, session = _make_plugin(session)

        captured_kwargs = {}

        @contextmanager
        def fake_stream(**kwargs):
            captured_kwargs.update(kwargs)
            mock = MagicMock()
            mock.text_stream = iter([])
            final = MagicMock()
            final.model_dump.return_value = {"usage": {}}
            mock.get_final_message.return_value = final
            yield mock

        plugin.client.messages.stream = fake_stream
        plugin._handle_user_message("test")

        assert captured_kwargs["temperature"] == 0.7

    def test_messages_reflect_context_window(self):
        session = _make_session(context_depth=2)
        plugin, session = _make_plugin(session)

        # Seed history with 6 messages (3 pairs); window=2 pairs → last 4
        for i in range(3):
            session.add_message("user", f"u{i}")
            session.add_message("assistant", f"a{i}")

        captured_kwargs = {}

        @contextmanager
        def fake_stream(**kwargs):
            captured_kwargs.update(kwargs)
            mock = MagicMock()
            mock.text_stream = iter([])
            final = MagicMock()
            final.model_dump.return_value = {"usage": {}}
            mock.get_final_message.return_value = final
            yield mock

        plugin.client.messages.stream = fake_stream
        plugin._handle_user_message("new question")

        # context_depth=2 → last 4 messages from history before the new one,
        # then the new user message is prepended; actual window is sliced
        # after adding the new message
        msgs = captured_kwargs["messages"]
        # The new user message is the last one
        assert msgs[-1] == {"role": "user", "content": "new question"}


# ---------------------------------------------------------------------------
# Streaming event emission
# ---------------------------------------------------------------------------

class TestStreamingEvents:
    def test_assistant_token_events_emitted_per_chunk(self):
        plugin, session = _make_plugin()
        chunks = ["Hello", ", ", "world", "!"]
        ctx, _, _ = _mock_stream(chunks)
        plugin.client.messages.stream = ctx

        received: list[Event] = []
        session.event_bus.subscribe(EventType.ASSISTANT_TOKENS, received.append)

        plugin._handle_user_message("hi")

        assert len(received) == 4
        assert received[0].payload["delta"] == "Hello"
        assert received[0].payload["accumulated"] == "Hello"
        assert received[3].payload["delta"] == "!"
        assert received[3].payload["accumulated"] == "Hello, world!"

    def test_assistant_message_event_emitted_after_stream(self):
        plugin, session = _make_plugin()
        usage = {"input_tokens": 20, "output_tokens": 8}
        ctx, _, _ = _mock_stream(["full response"], usage=usage)
        plugin.client.messages.stream = ctx

        received: list[Event] = []
        session.event_bus.subscribe(EventType.ASSISTANT_MESSAGE, received.append)

        plugin._handle_user_message("hi")

        assert len(received) == 1
        ev = received[0]
        assert ev.payload["text"] == "full response"
        assert ev.payload["usage"] == {"input_tokens": 20, "output_tokens": 8}

    def test_message_added_to_session_history(self):
        plugin, session = _make_plugin()
        ctx, _, _ = _mock_stream(["response text"])
        plugin.client.messages.stream = ctx

        plugin._handle_user_message("question")

        # History: [user, assistant]
        assert len(session.history) == 2
        assert session.history[0].role == "user"
        assert session.history[0].content == "question"
        assert session.history[1].role == "assistant"
        assert session.history[1].content == "response text"

    def test_raw_request_and_response_stored_on_message(self):
        plugin, session = _make_plugin()
        raw_resp = {"usage": {"input_tokens": 1, "output_tokens": 2}, "id": "msg_123"}
        ctx, stream_obj, final = _mock_stream(["ok"])
        final.model_dump.return_value = raw_resp
        plugin.client.messages.stream = ctx

        plugin._handle_user_message("ping")

        assistant_msg = session.history[-1]
        assert assistant_msg.raw_response == raw_resp
        assert assistant_msg.raw_request is not None
        assert assistant_msg.raw_request["model"] == session.model


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_error_event_emitted_on_exception(self):
        plugin, session = _make_plugin()

        @contextmanager
        def exploding_stream(**kwargs):
            raise RuntimeError("network error")
            yield  # make it a generator

        plugin.client.messages.stream = exploding_stream

        received: list[Event] = []
        session.event_bus.subscribe(EventType.ERROR, received.append)

        plugin._handle_user_message("hi")

        assert len(received) == 1
        assert received[0].payload["context"] == "chat"
        assert isinstance(received[0].payload["error"], RuntimeError)

    def test_no_assistant_message_emitted_on_error(self):
        plugin, session = _make_plugin()

        @contextmanager
        def exploding_stream(**kwargs):
            raise ValueError("bad")
            yield

        plugin.client.messages.stream = exploding_stream

        received: list[Event] = []
        session.event_bus.subscribe(EventType.ASSISTANT_MESSAGE, received.append)

        plugin._handle_user_message("hi")

        assert received == []

    def test_history_not_updated_with_assistant_msg_on_error(self):
        plugin, session = _make_plugin()

        @contextmanager
        def exploding_stream(**kwargs):
            raise OSError("timeout")
            yield

        plugin.client.messages.stream = exploding_stream

        plugin._handle_user_message("hi")

        # Only the user message should be in history
        assert len(session.history) == 1
        assert session.history[0].role == "user"


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

class TestCmdModel:
    def test_returns_current_model_when_no_args(self):
        plugin, session = _make_plugin()
        session.model = "claude-opus"
        result = plugin.cmd_model("", session)
        assert "claude-opus" in result

    def test_sets_model_and_returns_confirmation(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_model("claude-haiku", session)
        assert session.model == "claude-haiku"
        assert "claude-haiku" in result

    def test_strips_whitespace_from_args(self):
        plugin, session = _make_plugin()
        plugin.cmd_model("  claude-haiku  ", session)
        assert session.model == "claude-haiku"


class TestCmdSystem:
    def test_returns_current_prompt_when_no_args(self):
        plugin, session = _make_plugin()
        session.system_prompt = "Be concise."
        result = plugin.cmd_system("", session)
        assert "Be concise." in result

    def test_sets_system_prompt_and_returns_confirmation(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_system("Be very helpful.", session)
        assert session.system_prompt == "Be very helpful."
        assert result is not None


class TestCmdTemp:
    def test_returns_current_temp_when_no_args_none(self):
        plugin, session = _make_plugin()
        session.temperature = None
        result = plugin.cmd_temp("", session)
        assert "auto" in result

    def test_returns_current_temp_when_no_args_float(self):
        plugin, session = _make_plugin()
        session.temperature = 0.5
        result = plugin.cmd_temp("", session)
        assert "0.5" in result

    def test_sets_valid_temperature(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_temp("0.8", session)
        assert session.temperature == 0.8
        assert "0.8" in result

    def test_rejects_out_of_range_temperature(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_temp("3.0", session)
        assert session.temperature is None  # unchanged
        assert result is not None

    def test_rejects_non_numeric_temperature(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_temp("hot", session)
        assert result is not None
        assert session.temperature is None

    def test_boundary_values_accepted(self):
        plugin, session = _make_plugin()
        plugin.cmd_temp("0.0", session)
        assert session.temperature == 0.0
        plugin.cmd_temp("2.0", session)
        assert session.temperature == 2.0


class TestCmdClear:
    def test_clears_history(self):
        plugin, session = _make_plugin()
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")
        result = plugin.cmd_clear("", session)
        assert session.history == []
        assert result is not None

    def test_returns_confirmation_string(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_clear("", session)
        assert isinstance(result, str)
        assert len(result) > 0


class TestCmdHistory:
    def test_empty_history(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_history("", session)
        assert result is not None
        assert "No history" in result or result == "No history."

    def test_lists_all_messages(self):
        plugin, session = _make_plugin()
        session.add_message("user", "First question")
        session.add_message("assistant", "First answer")
        result = plugin.cmd_history("", session)
        assert "user" in result
        assert "assistant" in result
        assert "First question" in result
        assert "First answer" in result

    def test_truncates_long_content(self):
        plugin, session = _make_plugin()
        long_text = "x" * 200
        session.add_message("user", long_text)
        result = plugin.cmd_history("", session)
        # Should not contain the full 200 chars untruncated
        assert len(result) < 300  # rough guard


class TestCmdRemember:
    def test_calls_stream_with_full_history_and_modified_prompt(self):
        plugin, session = _make_plugin()
        session.system_prompt = "Be helpful."
        session.add_message("user", "hi")
        session.add_message("assistant", "hello")

        plugin._stream_with_messages = MagicMock()
        plugin.cmd_remember("", session)

        plugin._stream_with_messages.assert_called_once()
        kwargs = plugin._stream_with_messages.call_args
        messages_arg = kwargs[1]["messages"] if kwargs[1] else kwargs[0][0]
        system_arg = kwargs[1]["system_prompt"] if kwargs[1] else kwargs[0][1]

        assert "Be helpful." in system_arg
        assert "Context restoration" in system_arg

    def test_returns_none(self):
        plugin, session = _make_plugin()
        plugin._stream_with_messages = MagicMock()
        result = plugin.cmd_remember("", session)
        assert result is None


class TestCmdDepth:
    def test_returns_current_depth_when_no_args(self):
        plugin, session = _make_plugin()
        session.context_depth = 7
        result = plugin.cmd_depth("", session)
        assert "7" in result

    def test_sets_valid_depth(self):
        plugin, session = _make_plugin()
        result = plugin.cmd_depth("10", session)
        assert session.context_depth == 10
        assert "10" in result

    def test_rejects_zero(self):
        plugin, session = _make_plugin()
        original = session.context_depth
        result = plugin.cmd_depth("0", session)
        assert session.context_depth == original
        assert result is not None

    def test_rejects_negative(self):
        plugin, session = _make_plugin()
        original = session.context_depth
        result = plugin.cmd_depth("-1", session)
        assert session.context_depth == original

    def test_rejects_non_integer(self):
        plugin, session = _make_plugin()
        original = session.context_depth
        result = plugin.cmd_depth("abc", session)
        assert session.context_depth == original
        assert result is not None


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

class TestPluginMetadata:
    def test_name(self):
        assert ChatPlugin.name == "chat"

    def test_commands_keys(self):
        expected = {"/model", "/system", "/temp", "/clear", "/history", "/remember", "/depth"}
        assert set(ChatPlugin.commands.keys()) == expected

    def test_command_method_names_exist(self):
        plugin = ChatPlugin()
        for method_name in ChatPlugin.commands.values():
            assert hasattr(plugin, method_name), f"Missing method: {method_name}"
