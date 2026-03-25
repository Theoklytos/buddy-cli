from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path

from buddy.core.plugin import Plugin
from buddy.core.events import EventType
from buddy.core.config import BUDDY_LOGS_DIR


class LoggerPlugin(Plugin):
    """Persists session exchanges to a JSON log file."""

    name = "logger"
    commands = {"/save": "cmd_save"}

    def setup(self, session) -> None:
        """Initialise log directory and file, write empty log skeleton."""
        self.session = session
        self.log_dir = BUDDY_LOGS_DIR / date.today().isoformat()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{session.session_id}.json"
        self._exchanges: list[dict] = []
        self._write_full()

    def on_event(self, event) -> None:
        """Append a new exchange entry whenever an assistant message arrives."""
        if event.type != EventType.ASSISTANT_MESSAGE:
            return

        assistant_msg = self.session.history[-1]
        exchange = {
            "turn": len(self._exchanges) + 1,
            "timestamp": datetime.now().isoformat(),
            "request": assistant_msg.raw_request,
            "response": assistant_msg.raw_response,
        }
        self._exchanges.append(exchange)
        self._write_full()

    def _write_full(self) -> None:
        """Write the complete session document (including all exchanges) to disk."""
        doc = {
            "session_id": self.session.session_id,
            "created_at": self.session.created_at.isoformat(),
            "model": self.session.model,
            "config": {
                "system_prompt": self.session.system_prompt,
                "temperature": self.session.temperature,
                "max_tokens": self.session.max_tokens,
                "context_depth": self.session.context_depth,
            },
            "exchanges": self._exchanges,
        }
        self.log_file.write_text(json.dumps(doc, indent=2, default=str))

    def cmd_save(self, args: str, session) -> str:
        """Copy the current log to a custom path or tag it with args as a label.

        Usage: /save [label_or_path]
        If no args, confirms the current log file location.
        If args looks like a path (contains / or \\), copy to that path.
        Otherwise treat args as a tag and copy to a sibling file with the tag appended.
        """
        if not args.strip():
            return f"[green]Session log:[/green] {self.log_file}"

        args = args.strip()
        # Determine destination
        if "/" in args or "\\" in args:
            dest = Path(args).expanduser()
        else:
            # Tag-based copy: same directory, session_id + tag
            dest = self.log_dir / f"{session.session_id}_{args}.json"

        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(self.log_file, dest)
        return f"[green]Log saved to:[/green] {dest}"
