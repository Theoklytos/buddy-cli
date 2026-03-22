"""MCP logging middleware - records all tool calls for audit/replay."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Logs stored separately from Bud data
SESSIONS_DIR = Path.home() / "mcp_logs"
CURRENT_SESSION_FILE = SESSIONS_DIR / ".current_session"


class MCPLogger:
    """Logger for MCP tool calls - stores per-session JSONL logs."""

    def __init__(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._current_session_id: Optional[str] = None
        self._session_file: Optional[Path] = None

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start or resume a logging session.

        Args:
            session_id: Optional explicit session ID. If not provided,
                       loads from .current_session or generates new.

        Returns:
            The session ID being used.
        """
        if session_id is None:
            # Try to resume from .current_session
            if CURRENT_SESSION_FILE.exists():
                try:
                    session_id = CURRENT_SESSION_FILE.read_text().strip()
                except Exception:
                    session_id = None

        if session_id is None:
            session_id = f"mcp-{uuid.uuid4().hex[:12]}"

        self._current_session_id = session_id
        self._session_file = SESSIONS_DIR / f"{session_id}.jsonl"

        # Persist current session ID
        CURRENT_SESSION_FILE.write_text(session_id)

        # Create the session file if it doesn't exist
        if not self._session_file.exists():
            self._session_file.touch()

        return session_id

    def log_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        duration_ms: float
    ) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the MCP tool called
            parameters: The parameters passed to the tool
            result: The result returned by the tool
            duration_ms: Execution time in milliseconds
        """
        if self._session_file is None:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._current_session_id,
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "duration_ms": round(duration_ms, 2),
        }

        with open(self._session_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def end_session(self) -> None:
        """End the current logging session."""
        self._current_session_id = None
        self._session_file = None


# Global logger instance
_logger = MCPLogger()


def get_logger() -> MCPLogger:
    """Get the global MCP logger instance."""
    return _logger


def start_logging_session(session_id: Optional[str] = None) -> str:
    """Start or resume a logging session (convenience function)."""
    return _logger.start_session(session_id)


def log_tool_call(
    tool_name: str,
    parameters: dict[str, Any],
    result: dict[str, Any],
    duration_ms: float
) -> None:
    """Log a tool call (convenience function)."""
    _logger.log_tool_call(tool_name, parameters, result, duration_ms)


def end_logging_session() -> None:
    """End the current logging session (convenience function)."""
    _logger.end_session()
