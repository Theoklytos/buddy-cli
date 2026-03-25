from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from buddy.core.plugin import Plugin


class InputPlugin(Plugin):
    name = "input"
    commands: dict = {}

    def setup(self, session) -> None:
        self.session = session

        self.bindings = KeyBindings()

        @self.bindings.add("c-x")
        def validate_and_handle(event):
            """Submit the current buffer."""
            event.current_buffer.validate_and_handle()

        @self.bindings.add("c-q")
        def quit_app(event):
            raise SystemExit

        self.prompt_session: PromptSession = PromptSession(
            multiline=True,
            key_bindings=self.bindings,
            prompt_continuation="  ... ",
        )

    def get_input(self) -> str | None:
        try:
            return self.prompt_session.prompt("You > ")
        except (EOFError, KeyboardInterrupt, SystemExit):
            return None
