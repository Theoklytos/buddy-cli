from __future__ import annotations

import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from buddy.core.events import EventType
from buddy.core.plugin import Plugin


class RendererPlugin(Plugin):
    name = "renderer"
    commands: dict = {}

    def setup(self, session) -> None:
        self.session = session
        self.console = Console()
        self._stream_start: float | None = None

    def on_event(self, event) -> None:
        if event.type == EventType.SESSION_START:
            self.console.print(
                Panel(
                    "buddy v0.1.0\n\n"
                    "[dim]Enter[/dim]: newline   "
                    "[dim]Ctrl+X[/dim]: send   "
                    "[dim]Ctrl+Q[/dim]: quit",
                    title="Welcome",
                    border_style="green",
                )
            )

        elif event.type == EventType.ASSISTANT_TOKENS:
            if self._stream_start is None:
                self._stream_start = time.time()
                print()
            print(event.payload["delta"], end="", flush=True)

        elif event.type == EventType.ASSISTANT_MESSAGE:
            elapsed = time.time() - self._stream_start if self._stream_start is not None else 0.0
            print()  # end the raw stream line
            self.console.print(
                Panel(
                    Markdown(event.payload["text"]),
                    border_style="blue",
                    title="buddy",
                )
            )
            usage = event.payload.get("usage", {})
            model = event.payload.get("model", "")
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self.console.print(
                f"[dim]── {model} · {input_tokens} in / {output_tokens} out · {elapsed:.1f}s ──[/dim]"
            )
            self._stream_start = None

        elif event.type == EventType.COMMAND_EXECUTED:
            result = event.payload.get("result")
            if result is not None:
                self.console.print(result)

        elif event.type == EventType.ERROR:
            message = event.payload.get("message", str(event.payload))
            self.console.print(
                Panel(message, border_style="red", title="Error")
            )
