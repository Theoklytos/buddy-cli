from __future__ import annotations

import anthropic

from buddy.core.plugin import Plugin
from buddy.core.events import Event, EventType
from buddy.core.config import resolve_api_key


class ChatPlugin(Plugin):
    name = "chat"
    commands = {
        "/model": "cmd_model",
        "/system": "cmd_system",
        "/temp": "cmd_temp",
        "/clear": "cmd_clear",
        "/history": "cmd_history",
        "/remember": "cmd_remember",
        "/depth": "cmd_depth",
    }

    def setup(self, session) -> None:
        self.session = session
        api_key = resolve_api_key(session.config)
        if not api_key:
            raise RuntimeError(
                "No Anthropic API key found. Set it in buddy.yaml, "
                "ANTHROPIC_API_KEY env var, or bud's config.yaml under llm.api_key."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def on_event(self, event: Event) -> None:
        if event.type == EventType.USER_MESSAGE:
            self._handle_user_message(event.payload["text"])

    def _handle_user_message(self, text: str) -> None:
        self.session.add_message("user", text)
        request_envelope = {
            "model": self.session.model,
            "max_tokens": self.session.max_tokens,
            "system": self.session.system_prompt,
            "messages": self.session.get_context_messages(),
        }
        if self.session.temperature is not None:
            request_envelope["temperature"] = self.session.temperature
        self._stream_with_messages(
            messages=request_envelope["messages"],
            system_prompt=request_envelope["system"],
            request_envelope=request_envelope,
        )

    def _stream_with_messages(
        self,
        messages: list[dict],
        system_prompt: str,
        request_envelope: dict | None = None,
    ) -> None:
        if request_envelope is None:
            request_envelope = {
                "model": self.session.model,
                "max_tokens": self.session.max_tokens,
                "system": system_prompt,
                "messages": messages,
            }
            if self.session.temperature is not None:
                request_envelope["temperature"] = self.session.temperature

        accumulated = ""
        raw_response = None
        try:
            with self.client.messages.stream(**request_envelope) as stream:
                for text_chunk in stream.text_stream:
                    accumulated += text_chunk
                    self.session.event_bus.emit(Event(
                        EventType.ASSISTANT_TOKENS,
                        {"delta": text_chunk, "accumulated": accumulated},
                    ))
                raw_response = stream.get_final_message().model_dump()
        except Exception as e:
            self.session.event_bus.emit(Event(
                EventType.ERROR, {"error": e, "context": "chat"}
            ))
            return

        msg = self.session.add_message("assistant", accumulated)
        msg.raw_request = request_envelope
        msg.raw_response = raw_response

        self.session.event_bus.emit(Event(
            EventType.ASSISTANT_MESSAGE,
            {
                "text": accumulated,
                "model": self.session.model,
                "usage": raw_response.get("usage", {}),
                "raw_response": raw_response,
            },
        ))

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------

    def cmd_model(self, args: str, session) -> str | None:
        args = args.strip()
        if not args:
            return f"Current model: {session.model}"
        session.model = args
        return f"Model set to: {session.model}"

    def cmd_system(self, args: str, session) -> str | None:
        args = args.strip()
        if not args:
            return f"Current system prompt: {session.system_prompt}"
        session.system_prompt = args
        return f"System prompt updated."

    def cmd_temp(self, args: str, session) -> str | None:
        args = args.strip()
        if not args:
            current = session.temperature
            return f"Current temperature: {'auto' if current is None else current}"
        if args.lower() in ("none", "auto", ""):
            session.temperature = None
            return "Temperature set to: auto"
        try:
            value = float(args)
        except ValueError:
            return f"Invalid temperature: {args!r}. Must be a float between 0.0 and 2.0."
        if not (0.0 <= value <= 2.0):
            return f"Invalid temperature: {value}. Must be between 0.0 and 2.0."
        session.temperature = value
        return f"Temperature set to: {value}"

    def cmd_clear(self, args: str, session) -> str | None:
        session.clear_history()
        return "History cleared."

    def cmd_history(self, args: str, session) -> str | None:
        if not session.history:
            return "No history."
        lines = []
        for i, msg in enumerate(session.history, 1):
            snippet = msg.content[:80]
            if len(msg.content) > 80:
                snippet += "…"
            lines.append(f"{i}. [{msg.role}] {snippet}")
        return "\n".join(lines)

    def cmd_remember(self, args: str, session) -> None:
        restoration_prompt = (
            session.system_prompt
            + "\n\n[Context restoration: the full conversation history follows.]"
        )
        self._stream_with_messages(
            messages=session.get_full_history_messages(),
            system_prompt=restoration_prompt,
        )
        return None

    def cmd_depth(self, args: str, session) -> str | None:
        args = args.strip()
        if not args:
            return f"Current context depth: {session.context_depth}"
        try:
            value = int(args)
        except ValueError:
            return f"Invalid depth: {args!r}. Must be a positive integer."
        if value <= 0:
            return f"Invalid depth: {value}. Must be greater than 0."
        session.context_depth = value
        return f"Context depth set to: {value}"
