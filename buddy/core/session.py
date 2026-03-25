from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    raw_request: dict | None = None
    raw_response: dict | None = None


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = "You are a helpful assistant."
    temperature: float | None = None
    max_tokens: int = 4096
    context_depth: int = 5
    history: list[Message] = field(default_factory=list)
    event_bus: Any = None
    plugin_registry: Any = None
    config: dict = field(default_factory=dict)

    def get_context_messages(self) -> list[dict[str, str]]:
        window = self.history[-(self.context_depth * 2):]
        return [{"role": m.role, "content": m.content} for m in window]

    def get_full_history_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.history.append(msg)
        return msg

    def clear_history(self) -> None:
        self.history = []
