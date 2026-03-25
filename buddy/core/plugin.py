from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from buddy.core.events import Event
    from buddy.core.session import Session


class Plugin(ABC):
    name: str = "unnamed"
    commands: dict[str, str] = {}

    def setup(self, session: "Session") -> None:
        pass

    def teardown(self) -> None:
        pass

    def on_event(self, event: "Event") -> None:
        pass


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._commands: dict[str, tuple[Plugin, str]] = {}

    def register(self, plugin: Plugin) -> None:
        for cmd_name in plugin.commands:
            if cmd_name in self._commands:
                raise ValueError(
                    f"Duplicate command '{cmd_name}': already registered by "
                    f"'{self._commands[cmd_name][0].name}'"
                )
        self._plugins.append(plugin)
        for cmd_name, method_name in plugin.commands.items():
            self._commands[cmd_name] = (plugin, method_name)

    def setup_all(self, session: Any) -> None:
        for plugin in self._plugins:
            plugin.setup(session)

    def teardown_all(self) -> None:
        for plugin in self._plugins:
            plugin.teardown()

    def get_command(self, name: str) -> tuple[Plugin, str] | None:
        return self._commands.get(name)

    def get_all_commands(self) -> dict[str, tuple[Plugin, str]]:
        return dict(self._commands)

    def get_plugins(self) -> list[Plugin]:
        return list(self._plugins)

    def get_plugin_by_type(self, cls: type) -> Plugin | None:
        return next((p for p in self._plugins if isinstance(p, cls)), None)

    def broadcast(self, event: Any) -> None:
        for plugin in self._plugins:
            plugin.on_event(event)
