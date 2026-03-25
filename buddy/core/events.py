from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    SESSION_START = auto()
    SESSION_END = auto()
    USER_MESSAGE = auto()
    ASSISTANT_TOKENS = auto()
    ASSISTANT_MESSAGE = auto()
    COMMAND_EXECUTED = auto()
    ERROR = auto()


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable]] = {}

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def emit(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, []):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Error in event handler %r for event %s", handler, event.type
                )
