"""Tests for buddy.core.plugin."""

import pytest

from buddy.core.events import Event, EventType
from buddy.core.plugin import Plugin, PluginRegistry


# ---------------------------------------------------------------------------
# Concrete plugin implementations for testing
# ---------------------------------------------------------------------------

class NoOpPlugin(Plugin):
    name = "noop"


class GreetPlugin(Plugin):
    name = "greet"
    commands = {"/hello": "do_hello", "/bye": "do_bye"}

    def __init__(self):
        self.setup_called_with = None
        self.teardown_called = False
        self.received_events = []

    def do_hello(self):
        return "hello!"

    def do_bye(self):
        return "goodbye!"

    def setup(self, session):
        self.setup_called_with = session

    def teardown(self):
        self.teardown_called = True

    def on_event(self, event):
        self.received_events.append(event)


class InfoPlugin(Plugin):
    name = "info"
    commands = {"/info": "do_info"}

    def do_info(self):
        return "info"


class ConflictPlugin(Plugin):
    name = "conflict"
    commands = {"/hello": "conflict_hello"}  # conflicts with GreetPlugin


# ---------------------------------------------------------------------------
# Plugin ABC tests
# ---------------------------------------------------------------------------

class TestPluginDefaults:
    def test_default_name(self):
        assert NoOpPlugin.name == "noop"

    def test_default_commands(self):
        assert NoOpPlugin.commands == {}

    def test_setup_is_noop(self):
        p = NoOpPlugin()
        p.setup(object())  # should not raise

    def test_teardown_is_noop(self):
        p = NoOpPlugin()
        p.teardown()  # should not raise

    def test_on_event_is_noop(self):
        p = NoOpPlugin()
        p.on_event(Event(type=EventType.SESSION_START))  # should not raise


# ---------------------------------------------------------------------------
# PluginRegistry tests
# ---------------------------------------------------------------------------

class TestPluginRegistryRegister:
    def test_register_plugin(self):
        reg = PluginRegistry()
        p = GreetPlugin()
        reg.register(p)
        assert p in reg._plugins

    def test_commands_populated_after_register(self):
        reg = PluginRegistry()
        p = GreetPlugin()
        reg.register(p)
        assert "/hello" in reg._commands
        assert "/bye" in reg._commands

    def test_command_maps_to_correct_plugin_and_method(self):
        reg = PluginRegistry()
        p = GreetPlugin()
        reg.register(p)
        plugin, method_name = reg._commands["/hello"]
        assert plugin is p
        assert method_name == "do_hello"

    def test_register_duplicate_command_raises(self):
        reg = PluginRegistry()
        reg.register(GreetPlugin())
        with pytest.raises(ValueError, match="/hello"):
            reg.register(ConflictPlugin())

    def test_second_plugin_not_added_on_duplicate(self):
        reg = PluginRegistry()
        reg.register(GreetPlugin())
        conflict = ConflictPlugin()
        with pytest.raises(ValueError):
            reg.register(conflict)
        assert conflict not in reg._plugins

    def test_register_multiple_plugins(self):
        reg = PluginRegistry()
        g = GreetPlugin()
        i = InfoPlugin()
        reg.register(g)
        reg.register(i)
        assert len(reg._plugins) == 2
        assert "/info" in reg._commands


class TestPluginRegistrySetupTeardown:
    def test_setup_all_calls_setup_on_each_plugin(self):
        reg = PluginRegistry()
        p1 = GreetPlugin()
        p2 = GreetPlugin()
        p2.name = "greet2"
        # Override commands to avoid duplicate
        p2.commands = {}
        reg.register(p1)
        reg.register(p2)

        sentinel = object()
        reg.setup_all(sentinel)

        assert p1.setup_called_with is sentinel
        assert p2.setup_called_with is sentinel

    def test_teardown_all_calls_teardown_on_each_plugin(self):
        reg = PluginRegistry()
        p = GreetPlugin()
        reg.register(p)
        reg.teardown_all()
        assert p.teardown_called is True


class TestPluginRegistryGetCommand:
    def test_get_existing_command(self):
        reg = PluginRegistry()
        p = GreetPlugin()
        reg.register(p)
        result = reg.get_command("/hello")
        assert result is not None
        plugin, method_name = result
        assert plugin is p
        assert method_name == "do_hello"

    def test_get_nonexistent_command_returns_none(self):
        reg = PluginRegistry()
        reg.register(GreetPlugin())
        assert reg.get_command("/nonexistent") is None

    def test_get_all_commands_returns_copy(self):
        reg = PluginRegistry()
        reg.register(GreetPlugin())
        cmds = reg.get_all_commands()
        assert "/hello" in cmds
        assert "/bye" in cmds
        # Mutating returned dict does not affect registry
        cmds["/mutated"] = None
        assert "/mutated" not in reg._commands


class TestPluginRegistryBroadcast:
    def test_broadcast_calls_on_event_for_all_plugins(self):
        reg = PluginRegistry()
        p1 = GreetPlugin()
        p2 = GreetPlugin()
        p2.name = "greet2"
        p2.commands = {}
        reg.register(p1)
        reg.register(p2)

        event = Event(type=EventType.USER_MESSAGE, payload={"text": "hi"})
        reg.broadcast(event)

        assert p1.received_events == [event]
        assert p2.received_events == [event]

    def test_broadcast_with_no_plugins_does_not_raise(self):
        reg = PluginRegistry()
        reg.broadcast(Event(type=EventType.SESSION_START))  # should not raise
