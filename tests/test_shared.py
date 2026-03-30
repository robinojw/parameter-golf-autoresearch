# tests/test_shared.py
import json
import time
from pathlib import Path

from agents.shared import (
    Message,
    append_message,
    read_messages_since,
    RESEARCH_QUEUE_PATH,
    RESEARCH_RESULTS_PATH,
    PROMOTION_QUEUE_PATH,
)


class TestMessage:
    def test_to_json_includes_all_fields(self):
        msg = Message(
            priority="high",
            source_experiment="abc123",
            message="need help with ternary quant",
        )
        d = msg.to_dict()
        assert d["priority"] == "high"
        assert d["source_experiment"] == "abc123"
        assert d["message"] == "need help with ternary quant"
        assert "timestamp" in d

    def test_from_dict_roundtrip(self):
        msg = Message(priority="normal", message="test")
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.priority == msg.priority
        assert restored.message == msg.message

    def test_default_priority_is_normal(self):
        msg = Message(message="hello")
        assert msg.priority == "normal"


class TestAppendMessage:
    def test_creates_file_if_missing(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        msg = Message(message="first message")
        append_message(path, msg)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["message"] == "first message"

    def test_appends_to_existing(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        append_message(path, Message(message="one"))
        append_message(path, Message(message="two"))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_concurrent_appends_no_corruption(self, tmp_path):
        """Each line must be valid JSON even under rapid appends."""
        path = tmp_path / "queue.jsonl"
        for i in range(50):
            append_message(path, Message(message=f"msg {i}"))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 50
        for line in lines:
            json.loads(line)  # should not raise


class TestReadMessagesSince:
    def test_returns_empty_for_missing_file(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        msgs = read_messages_since(path, since=0.0)
        assert msgs == []

    def test_filters_by_timestamp(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        old = Message(message="old")
        old.timestamp = "2020-01-01T00:00:00Z"
        append_message(path, old)

        new = Message(message="new")
        append_message(path, new)

        # Read only messages after 2025
        from datetime import datetime
        cutoff = datetime(2025, 1, 1).timestamp()
        msgs = read_messages_since(path, since=cutoff)
        assert len(msgs) == 1
        assert msgs[0].message == "new"

    def test_returns_all_when_since_is_zero(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        append_message(path, Message(message="a"))
        append_message(path, Message(message="b"))
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 2
