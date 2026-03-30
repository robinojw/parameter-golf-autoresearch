# agents/shared.py
"""Shared communication layer for experiment and research agents.

File-based message passing using JSONL queues. Each message is a single
JSON line with a timestamp, priority, optional source experiment, and
natural language content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Queue file paths
# ---------------------------------------------------------------------------

RESEARCH_QUEUE_PATH = Path("research_queue.jsonl")
RESEARCH_RESULTS_PATH = Path("research_results.jsonl")
PROMOTION_QUEUE_PATH = Path("promotion_queue.jsonl")


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Message:
    message: str
    priority: str = "normal"
    source_experiment: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "priority": self.priority,
            "source_experiment": self.source_experiment,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        return cls(
            message=d["message"],
            priority=d.get("priority", "normal"),
            source_experiment=d.get("source_experiment", ""),
            timestamp=d.get("timestamp", _now_iso()),
        )


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

def append_message(path: Path, msg: Message) -> None:
    """Append a message as a single JSON line. Creates file if missing."""
    with open(path, "a") as f:
        f.write(json.dumps(msg.to_dict()) + "\n")


def read_messages_since(path: Path, since: float = 0.0) -> list[Message]:
    """Read messages from a JSONL queue, filtering by timestamp.

    Args:
        path: Path to the JSONL file.
        since: Unix timestamp. Only messages after this time are returned.
               Use 0.0 to return all messages.

    Returns:
        List of Message objects, ordered as they appear in the file.
    """
    if not path.exists():
        return []

    messages: list[Message] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts_str = d.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                if ts > since:
                    messages.append(Message.from_dict(d))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return messages
