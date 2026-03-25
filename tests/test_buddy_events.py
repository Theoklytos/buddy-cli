"""Tests for buddy.core.events."""

import logging

import pytest

from buddy.core.events import Event, EventBus, EventType


class TestEventType:
    def test_all_members_exist(self):
        expected = {
            "SESSION_START",
            "SESSION_END",
            "USER_MESSAGE",
            "ASSISTANT_TOKENS",
            "ASSISTANT_MESSAGE",
            "COMMAND_EXECUTED",
            "ERROR",
        }
        assert {e.name for e in EventType} == expected

    def test_members_have_int_values(self):
        for member in EventType:
            assert isinstance(member.value, int)

    def test_members_are_unique(self):
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))


class TestEvent:
    def test_default_payload_is_empty_dict(self):
        event = Event(type=EventType.SESSION_START)
        assert event.payload == {}

    def test_payload_not_shared_between_instances(self):
        e1 = Event(type=EventType.SESSION_START)
        e2 = Event(type=EventType.SESSION_END)
        e1.payload["key"] = "value"
        assert "key" not in e2.payload

    def test_explicit_payload(self):
        event = Event(type=EventType.ERROR, payload={"msg": "boom"})
        assert event.payload == {"msg": "boom"}

    def test_type_attribute(self):
        event = Event(type=EventType.USER_MESSAGE)
        assert event.type is EventType.USER_MESSAGE


class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.SESSION_START, lambda e: received.append(e))

        event = Event(type=EventType.SESSION_START)
        bus.emit(event)

        assert received == [event]

    def test_emit_with_no_subscribers_does_not_raise(self):
        bus = EventBus()
        bus.emit(Event(type=EventType.SESSION_END))  # should not raise

    def test_multiple_handlers_called_in_order(self):
        bus = EventBus()
        order = []
        bus.subscribe(EventType.USER_MESSAGE, lambda e: order.append(1))
        bus.subscribe(EventType.USER_MESSAGE, lambda e: order.append(2))
        bus.subscribe(EventType.USER_MESSAGE, lambda e: order.append(3))

        bus.emit(Event(type=EventType.USER_MESSAGE))
        assert order == [1, 2, 3]

    def test_handler_for_different_type_not_called(self):
        bus = EventBus()
        called = []
        bus.subscribe(EventType.SESSION_START, lambda e: called.append(e))

        bus.emit(Event(type=EventType.SESSION_END))
        assert called == []

    def test_failing_handler_does_not_propagate(self, caplog):
        bus = EventBus()

        def bad_handler(e):
            raise RuntimeError("handler failure")

        good_called = []
        bus.subscribe(EventType.ERROR, bad_handler)
        bus.subscribe(EventType.ERROR, lambda e: good_called.append(e))

        with caplog.at_level(logging.ERROR, logger="buddy.core.events"):
            bus.emit(Event(type=EventType.ERROR))

        # The good handler after the bad one should still be called.
        assert len(good_called) == 1
        # An error should have been logged.
        assert any("handler failure" in r.message or "handler failure" in str(r.exc_info) for r in caplog.records)

    def test_failing_handler_logged(self, caplog):
        bus = EventBus()

        def raising_handler(e):
            raise ValueError("oops")

        bus.subscribe(EventType.COMMAND_EXECUTED, raising_handler)

        with caplog.at_level(logging.ERROR, logger="buddy.core.events"):
            bus.emit(Event(type=EventType.COMMAND_EXECUTED))

        assert len(caplog.records) >= 1

    def test_independent_event_types(self):
        bus = EventBus()
        start_calls = []
        end_calls = []
        bus.subscribe(EventType.SESSION_START, lambda e: start_calls.append(e))
        bus.subscribe(EventType.SESSION_END, lambda e: end_calls.append(e))

        bus.emit(Event(type=EventType.SESSION_START))
        assert len(start_calls) == 1
        assert len(end_calls) == 0

        bus.emit(Event(type=EventType.SESSION_END))
        assert len(start_calls) == 1
        assert len(end_calls) == 1

    def test_subscribe_same_handler_multiple_times(self):
        bus = EventBus()
        count = []
        handler = lambda e: count.append(1)
        bus.subscribe(EventType.ASSISTANT_MESSAGE, handler)
        bus.subscribe(EventType.ASSISTANT_MESSAGE, handler)

        bus.emit(Event(type=EventType.ASSISTANT_MESSAGE))
        assert len(count) == 2
