"""Main entry point for the buddy CLI command."""
from __future__ import annotations

import sys

from buddy.core.events import EventBus, Event, EventType
from buddy.core.plugin import PluginRegistry
from buddy.core.session import Session
from buddy.core.config import load_buddy_config, BUDDY_CONFIG_FILE, run_config_wizard
from buddy.plugins.chat import ChatPlugin
from buddy.plugins.renderer import RendererPlugin
from buddy.plugins.logger import LoggerPlugin
from buddy.plugins.help import HelpPlugin
from buddy.plugins.input import InputPlugin
from buddy.plugins.config_plugin import ConfigPlugin


def main() -> None:
    # 1. Load or create config
    if not BUDDY_CONFIG_FILE.exists():
        config = run_config_wizard()
    else:
        config = load_buddy_config()

    # 2. Build session
    event_bus = EventBus()
    registry = PluginRegistry()
    session = Session(
        model=config.get("model", "claude-sonnet-4-20250514"),
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens", 4096),
        context_depth=config.get("context_depth", 5),
        event_bus=event_bus,
        plugin_registry=registry,
        config=config,
    )

    # 3. Register plugins (order matters — renderer before chat so it receives streaming events)
    for plugin_cls in [InputPlugin, RendererPlugin, ChatPlugin, LoggerPlugin, HelpPlugin, ConfigPlugin]:
        plugin = plugin_cls()
        registry.register(plugin)

    # 4. Setup all plugins
    registry.setup_all(session)

    # 5. Wire event bus — subscribe all plugins to receive events through the bus.
    # This ensures that when event_bus.emit() is called from anywhere (main loop or
    # from inside a plugin like ChatPlugin), all plugins receive the event.
    for plugin in registry._plugins:
        for event_type in EventType:
            event_bus.subscribe(event_type, plugin.on_event)

    # 6. Emit session start
    event_bus.emit(Event(EventType.SESSION_START))

    # 7. Main loop
    input_plugin = next(p for p in registry._plugins if isinstance(p, InputPlugin))
    try:
        while True:
            text = input_plugin.get_input()
            if text is None:
                break
            text = text.strip()
            if not text:
                continue

            # Slash commands
            if text.startswith("/"):
                cmd_name = text.split()[0]
                args = text[len(cmd_name):].strip()

                if cmd_name == "/quit":
                    break

                result = registry.get_command(cmd_name)
                if result:
                    plugin, method_name = result
                    output = getattr(plugin, method_name)(args, session)
                    event_bus.emit(Event(EventType.COMMAND_EXECUTED, {
                        "name": cmd_name, "args": args, "result": output
                    }))
                else:
                    event_bus.emit(Event(EventType.ERROR, {
                        "error": f"Unknown command: {cmd_name}", "context": "command"
                    }))
                continue

            # Regular message
            event_bus.emit(Event(EventType.USER_MESSAGE, {"text": text}))
    finally:
        event_bus.emit(Event(EventType.SESSION_END))
        registry.teardown_all()


if __name__ == "__main__":
    main()
