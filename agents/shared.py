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
RESEARCH_ACK_PATH = Path("research_ack.jsonl")
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


def read_unfulfilled(path: Path) -> list[dict]:
    """Return queue entries that have not been marked fulfilled.

    Each entry is the raw dict from the JSONL line. An entry is considered
    fulfilled if it contains ``"fulfilled": true``.
    """
    if not path.exists():
        return []

    entries: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if not d.get("fulfilled", False):
                    entries.append(d)
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


def mark_fulfilled(path: Path, fulfilled_timestamps: list[str]) -> None:
    """Mark queue entries as fulfilled by rewriting the file.

    Any entry whose timestamp is in *fulfilled_timestamps* gets a
    ``"fulfilled": true`` field added.
    """
    if not path.exists() or not fulfilled_timestamps:
        return

    ts_set = set(fulfilled_timestamps)
    lines: list[str] = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
                if d.get("timestamp", "") in ts_set:
                    d["fulfilled"] = True
                lines.append(json.dumps(d))
            except (json.JSONDecodeError, KeyError):
                lines.append(stripped)

    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")


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


# ---------------------------------------------------------------------------
# Research acknowledgment
# ---------------------------------------------------------------------------

def ack_research_result(
    result_timestamp: str,
    action: str,
    experiment_id: str = "",
    ack_path: Path = RESEARCH_ACK_PATH,
) -> None:
    """Record that the experiment agent consumed a research finding.

    Args:
        result_timestamp: The timestamp of the research_results entry being acked.
        action: What the agent did with it (e.g. "adopted", "rejected", "deferred").
        experiment_id: Optional experiment name that used this finding.
    """
    entry = {
        "ack_timestamp": _now_iso(),
        "result_timestamp": result_timestamp,
        "action": action,
        "experiment_id": experiment_id,
    }
    with open(ack_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_acked_timestamps(ack_path: Path = RESEARCH_ACK_PATH) -> set[str]:
    """Return the set of research result timestamps that have been acknowledged."""
    if not ack_path.exists():
        return set()
    acked: set[str] = set()
    with open(ack_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                acked.add(d.get("result_timestamp", ""))
            except (json.JSONDecodeError, KeyError):
                continue
    return acked


def get_unacked_results(
    results_path: Path = RESEARCH_RESULTS_PATH,
    ack_path: Path = RESEARCH_ACK_PATH,
) -> list[Message]:
    """Return research results that haven't been acknowledged by the experiment agent."""
    acked = get_acked_timestamps(ack_path)
    all_messages = read_messages_since(results_path, since=0.0)
    return [m for m in all_messages if m.timestamp not in acked]
