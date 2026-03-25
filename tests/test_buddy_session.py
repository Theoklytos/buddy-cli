"""Tests for buddy.core.session."""

from datetime import datetime

import pytest

from buddy.core.session import Message, Session


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

class TestMessage:
    def test_required_fields(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_timestamp_defaults_to_now(self):
        before = datetime.now()
        msg = Message(role="user", content="hi")
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_timestamps_are_independent(self):
        m1 = Message(role="user", content="a")
        m2 = Message(role="assistant", content="b")
        # Both are datetimes; they may be equal but should not share state
        assert isinstance(m1.timestamp, datetime)
        assert isinstance(m2.timestamp, datetime)

    def test_raw_fields_default_to_none(self):
        msg = Message(role="user", content="hi")
        assert msg.raw_request is None
        assert msg.raw_response is None

    def test_explicit_raw_fields(self):
        req = {"model": "claude"}
        resp = {"id": "123"}
        msg = Message(role="assistant", content="ok", raw_request=req, raw_response=resp)
        assert msg.raw_request == req
        assert msg.raw_response == resp

    def test_explicit_timestamp(self):
        ts = datetime(2025, 1, 1, 12, 0, 0)
        msg = Message(role="user", content="hi", timestamp=ts)
        assert msg.timestamp == ts


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSessionDefaults:
    def test_session_id_is_8_chars(self):
        s = Session()
        assert len(s.session_id) == 8

    def test_session_ids_are_unique(self):
        ids = {Session().session_id for _ in range(50)}
        assert len(ids) == 50

    def test_created_at_is_datetime(self):
        before = datetime.now()
        s = Session()
        after = datetime.now()
        assert before <= s.created_at <= after

    def test_default_model(self):
        s = Session()
        assert s.model == "claude-sonnet-4-20250514"

    def test_default_system_prompt(self):
        s = Session()
        assert s.system_prompt == "You are a helpful assistant."

    def test_default_temperature_is_none(self):
        s = Session()
        assert s.temperature is None

    def test_default_max_tokens(self):
        s = Session()
        assert s.max_tokens == 4096

    def test_default_context_depth(self):
        s = Session()
        assert s.context_depth == 5

    def test_default_history_is_empty(self):
        s = Session()
        assert s.history == []

    def test_history_not_shared_between_instances(self):
        s1 = Session()
        s2 = Session()
        s1.history.append(Message(role="user", content="hi"))
        assert s2.history == []

    def test_default_event_bus_is_none(self):
        s = Session()
        assert s.event_bus is None

    def test_default_plugin_registry_is_none(self):
        s = Session()
        assert s.plugin_registry is None

    def test_default_config_is_empty_dict(self):
        s = Session()
        assert s.config == {}

    def test_config_not_shared_between_instances(self):
        s1 = Session()
        s2 = Session()
        s1.config["key"] = "value"
        assert "key" not in s2.config


# ---------------------------------------------------------------------------
# Session.add_message tests
# ---------------------------------------------------------------------------

class TestSessionAddMessage:
    def test_add_message_appends_to_history(self):
        s = Session()
        s.add_message("user", "hello")
        assert len(s.history) == 1

    def test_add_message_returns_message(self):
        s = Session()
        msg = s.add_message("user", "hello")
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_add_message_with_kwargs(self):
        s = Session()
        req = {"model": "claude"}
        msg = s.add_message("user", "hi", raw_request=req)
        assert msg.raw_request == req

    def test_add_multiple_messages(self):
        s = Session()
        s.add_message("user", "hello")
        s.add_message("assistant", "hi back")
        assert len(s.history) == 2
        assert s.history[0].role == "user"
        assert s.history[1].role == "assistant"


# ---------------------------------------------------------------------------
# Session.get_context_messages tests
# ---------------------------------------------------------------------------

class TestSessionGetContextMessages:
    def _add_n_messages(self, session, n):
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(role, f"msg {i}")

    def test_empty_history_returns_empty(self):
        s = Session()
        assert s.get_context_messages() == []

    def test_returns_last_context_depth_times_two(self):
        s = Session(context_depth=3)
        self._add_n_messages(s, 10)
        ctx = s.get_context_messages()
        assert len(ctx) == 6  # 3 * 2

    def test_returns_all_messages_when_fewer_than_window(self):
        s = Session(context_depth=5)
        self._add_n_messages(s, 4)
        ctx = s.get_context_messages()
        assert len(ctx) == 4

    def test_context_messages_are_dicts(self):
        s = Session()
        s.add_message("user", "test")
        ctx = s.get_context_messages()
        assert ctx[0] == {"role": "user", "content": "test"}

    def test_context_messages_are_the_most_recent(self):
        s = Session(context_depth=2)
        for i in range(6):
            s.add_message("user", f"msg {i}")
        ctx = s.get_context_messages()
        contents = [m["content"] for m in ctx]
        assert contents == ["msg 2", "msg 3", "msg 4", "msg 5"]

    def test_context_messages_do_not_include_raw_fields(self):
        s = Session()
        s.add_message("user", "hi", raw_request={"x": 1})
        ctx = s.get_context_messages()
        assert set(ctx[0].keys()) == {"role", "content"}


# ---------------------------------------------------------------------------
# Session.get_full_history_messages tests
# ---------------------------------------------------------------------------

class TestSessionGetFullHistoryMessages:
    def test_empty_history(self):
        s = Session()
        assert s.get_full_history_messages() == []

    def test_returns_all_messages(self):
        s = Session(context_depth=2)
        for i in range(10):
            s.add_message("user", f"msg {i}")
        full = s.get_full_history_messages()
        assert len(full) == 10

    def test_format_is_role_content_dict(self):
        s = Session()
        s.add_message("assistant", "reply")
        full = s.get_full_history_messages()
        assert full[0] == {"role": "assistant", "content": "reply"}


# ---------------------------------------------------------------------------
# Session.clear_history tests
# ---------------------------------------------------------------------------

class TestSessionClearHistory:
    def test_clear_empties_history(self):
        s = Session()
        s.add_message("user", "hi")
        s.add_message("assistant", "hey")
        s.clear_history()
        assert s.history == []

    def test_clear_on_empty_history_does_not_raise(self):
        s = Session()
        s.clear_history()  # should not raise
        assert s.history == []

    def test_can_add_messages_after_clear(self):
        s = Session()
        s.add_message("user", "first")
        s.clear_history()
        s.add_message("user", "second")
        assert len(s.history) == 1
        assert s.history[0].content == "second"
