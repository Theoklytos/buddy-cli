from __future__ import annotations

from buddy.core.plugin import Plugin
from buddy.core.config import list_buddy_profiles, load_buddy_profile


class ConfigPlugin(Plugin):
    """Provides /config and /profile commands to inspect and change session settings."""

    name = "config"
    commands = {"/config": "cmd_config", "/profile": "cmd_profile"}

    def setup(self, session) -> None:
        """Store session reference."""
        self.session = session

    def cmd_config(self, args: str, session) -> str:
        """Display current session configuration settings.

        Usage: /config
        """
        system_prompt = session.system_prompt or ""
        truncated = (
            system_prompt[:50] + "…" if len(system_prompt) > 50 else system_prompt
        )
        temperature_display = (
            str(session.temperature) if session.temperature is not None else "auto"
        )
        lines = [
            "[bold]Current session config:[/bold]",
            "",
            f"  [cyan]model[/cyan]          {session.model}",
            f"  [cyan]system_prompt[/cyan]  {truncated}",
            f"  [cyan]temperature[/cyan]    {temperature_display}",
            f"  [cyan]max_tokens[/cyan]     {session.max_tokens}",
            f"  [cyan]context_depth[/cyan]  {session.context_depth}",
        ]
        return "\n".join(lines)

    def cmd_profile(self, args: str, session) -> str:
        """List available profiles or load a named profile.

        Usage: /profile          — list available profiles
               /profile <name>   — apply named profile to current session
        """
        if not args.strip():
            profiles = list_buddy_profiles()
            if not profiles:
                return "[yellow]No profiles found.[/yellow] Create a YAML file in ~/.config/bud/buddy_profiles/"
            lines = ["[bold]Available profiles:[/bold]", ""]
            for p in profiles:
                lines.append(f"  [cyan]{p}[/cyan]")
            return "\n".join(lines)

        name = args.strip()
        try:
            profile = load_buddy_profile(name)
        except FileNotFoundError as exc:
            return f"[red]Profile not found:[/red] {name}"

        # Apply settings from profile to session
        if "model" in profile:
            session.model = profile["model"]
        if "system_prompt" in profile:
            session.system_prompt = profile["system_prompt"]
        if "temperature" in profile:
            session.temperature = profile["temperature"]
        if "max_tokens" in profile:
            session.max_tokens = profile["max_tokens"]
        if "context_depth" in profile:
            session.context_depth = profile["context_depth"]

        applied = ", ".join(profile.keys()) if profile else "(empty)"
        return f"[green]Profile '{name}' applied.[/green] Updated: {applied}"
