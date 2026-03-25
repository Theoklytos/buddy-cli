from __future__ import annotations

from buddy.core.plugin import Plugin
from buddy.core.events import EventType


class HelpPlugin(Plugin):
    """Provides /help command to display available commands and their descriptions."""

    name = "help"
    commands = {"/help": "cmd_help"}

    def setup(self, session) -> None:
        """Store session reference."""
        self.session = session

    def cmd_help(self, args: str, session) -> str:
        """Show available commands, or details for a specific command.

        Usage: /help [command]
        """
        all_commands = session.plugin_registry.get_all_commands()

        if args.strip():
            # Look up a specific command
            target = args.strip()
            if not target.startswith("/"):
                target = f"/{target}"
            entry = all_commands.get(target)
            if entry is None:
                return f"[yellow]Unknown command:[/yellow] {target}"
            plugin_instance, method_name = entry
            method = getattr(plugin_instance, method_name, None)
            doc = (method.__doc__ or "No description available.").strip()
            return f"[bold cyan]{target}[/bold cyan]\n\n{doc}"

        # Build full help listing
        lines = ["[bold]Available commands:[/bold]", ""]
        for cmd_name in sorted(all_commands):
            plugin_instance, method_name = all_commands[cmd_name]
            method = getattr(plugin_instance, method_name, None)
            if method and method.__doc__:
                first_line = method.__doc__.strip().splitlines()[0]
            else:
                first_line = "No description available."
            lines.append(f"  [cyan]{cmd_name}[/cyan]    {first_line}")

        lines.append("")
        lines.append("[dim]Enter: newline  |  Ctrl+X: send  |  Ctrl+Q: quit[/dim]")
        return "\n".join(lines)
