"""Tests for buddy.plugins.help — HelpPlugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from buddy.core.plugin import Plugin, PluginRegistry
from buddy.core.session import Session
from buddy.plugins.help import HelpPlugin


# ---------------------------------------------------------------------------
# Minimal stub plugin used across tests
# ---------------------------------------------------------------------------

class StubPlugin(Plugin):
    name = "stub"
    commands = {"/foo": "do_foo", "/bar": "do_bar"}

    def do_foo(self, args, session):
        """Perform the foo action."""

    def do_bar(self, args, session):
        """Perform the bar action. It does things."""


class NoDocPlugin(Plugin):
    name = "nodoc"
    commands = {"/nodoc": "do_nodoc"}

    def do_nodoc(self, args, session):
        pass  # intentionally no docstring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_session_with_plugins(*extra_plugins) -> tuple[HelpPlugin, Session]:
    """Build a PluginRegistry with HelpPlugin + extras, call setup_all, return plugin and session."""
    registry = PluginRegistry()
    help_plugin = HelpPlugin()
    registry.register(help_plugin)
    for p in extra_plugins:
        registry.register(p)

    session = Session()
    session.plugin_registry = registry
    registry.setup_all(session)
    return help_plugin, session


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------

class TestHelpSetup:
    def test_setup_stores_session(self):
        plugin = HelpPlugin()
        session = Session()
        session.plugin_registry = PluginRegistry()
        session.plugin_registry.register(plugin)
        plugin.setup(session)
        assert plugin.session is session


# ---------------------------------------------------------------------------
# cmd_help() — no args
# ---------------------------------------------------------------------------

class TestHelpCmdNoArgs:
    def test_returns_string(self):
        plugin, session = _setup_session_with_plugins()
        result = plugin.cmd_help("", session)
        assert isinstance(result, str)

    def test_header_present(self):
        plugin, session = _setup_session_with_plugins()
        result = plugin.cmd_help("", session)
        assert "Available commands" in result

    def test_lists_registered_command(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("", session)
        assert "/foo" in result
        assert "/bar" in result

    def test_includes_help_command(self):
        plugin, session = _setup_session_with_plugins()
        result = plugin.cmd_help("", session)
        assert "/help" in result

    def test_docstring_first_line_used(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("", session)
        assert "Perform the foo action." in result

    def test_commands_sorted_alphabetically(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("", session)
        bar_pos = result.index("/bar")
        foo_pos = result.index("/foo")
        help_pos = result.index("/help")
        # Alphabetical order: /bar < /foo < /help
        assert bar_pos < foo_pos < help_pos

    def test_footer_hint_present(self):
        plugin, session = _setup_session_with_plugins()
        result = plugin.cmd_help("", session)
        assert "Ctrl+X" in result or "ctrl+x" in result.lower()

    def test_no_docstring_shows_fallback(self):
        plugin, session = _setup_session_with_plugins(NoDocPlugin())
        result = plugin.cmd_help("", session)
        assert "No description available." in result


# ---------------------------------------------------------------------------
# cmd_help() — specific command lookup
# ---------------------------------------------------------------------------

class TestHelpCmdWithArgs:
    def test_specific_command_returns_description(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("/foo", session)
        assert "Perform the foo action." in result

    def test_specific_command_without_slash(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("foo", session)
        assert "Perform the foo action." in result

    def test_unknown_command_returns_error(self):
        plugin, session = _setup_session_with_plugins()
        result = plugin.cmd_help("/unknown", session)
        assert "Unknown command" in result

    def test_specific_command_shows_full_docstring(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("/bar", session)
        # Multi-line docstring should be present
        assert "Perform the bar action." in result

    def test_command_name_highlighted_in_specific_lookup(self):
        plugin, session = _setup_session_with_plugins(StubPlugin())
        result = plugin.cmd_help("/foo", session)
        assert "/foo" in result
