# tests/test_shared.py
import json
import time
from pathlib import Path

from agents.shared import (
    Message,
    append_message,
    read_messages_since,
    read_unfulfilled,
    mark_fulfilled,
    ack_research_result,
    get_acked_timestamps,
    get_unacked_results,
    RESEARCH_QUEUE_PATH,
    RESEARCH_RESULTS_PATH,
    RESEARCH_ACK_PATH,
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


class TestFulfillment:
    def test_read_unfulfilled_returns_all_when_none_fulfilled(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        append_message(path, Message(message="request 1"))
        append_message(path, Message(message="request 2"))
        unfulfilled = read_unfulfilled(path)
        assert len(unfulfilled) == 2

    def test_read_unfulfilled_filters_fulfilled(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        # Write one normal and one fulfilled entry
        with open(path, "w") as f:
            f.write(json.dumps({"timestamp": "2025-01-01T00:00:00Z", "message": "req1"}) + "\n")
            f.write(json.dumps({"timestamp": "2025-01-02T00:00:00Z", "message": "req2", "fulfilled": True}) + "\n")
        unfulfilled = read_unfulfilled(path)
        assert len(unfulfilled) == 1
        assert unfulfilled[0]["message"] == "req1"

    def test_mark_fulfilled_marks_by_timestamp(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"timestamp": "2025-01-01T00:00:00Z", "message": "req1"}) + "\n")
            f.write(json.dumps({"timestamp": "2025-01-02T00:00:00Z", "message": "req2"}) + "\n")
        mark_fulfilled(path, ["2025-01-01T00:00:00Z"])
        unfulfilled = read_unfulfilled(path)
        assert len(unfulfilled) == 1
        assert unfulfilled[0]["message"] == "req2"

    def test_mark_fulfilled_no_file_is_noop(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        # Should not raise
        mark_fulfilled(path, ["any"])

    def test_mark_fulfilled_empty_list_is_noop(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        path.write_text('{"timestamp": "2025-01-01T00:00:00Z", "message": "x"}\n')
        mark_fulfilled(path, [])
        unfulfilled = read_unfulfilled(path)
        assert len(unfulfilled) == 1

    def test_read_unfulfilled_empty_file(self, tmp_path):
        path = tmp_path / "queue.jsonl"
        assert read_unfulfilled(path) == []


class TestAcknowledgment:
    def test_ack_creates_file(self, tmp_path):
        ack_path = tmp_path / "ack.jsonl"
        ack_research_result("2025-01-01T00:00:00Z", "adopted", "exp1", ack_path=ack_path)
        assert ack_path.exists()
        line = json.loads(ack_path.read_text().strip())
        assert line["result_timestamp"] == "2025-01-01T00:00:00Z"
        assert line["action"] == "adopted"
        assert line["experiment_id"] == "exp1"

    def test_get_acked_timestamps(self, tmp_path):
        ack_path = tmp_path / "ack.jsonl"
        ack_research_result("ts1", "adopted", ack_path=ack_path)
        ack_research_result("ts2", "rejected", ack_path=ack_path)
        acked = get_acked_timestamps(ack_path=ack_path)
        assert acked == {"ts1", "ts2"}

    def test_get_acked_timestamps_empty(self, tmp_path):
        ack_path = tmp_path / "ack.jsonl"
        assert get_acked_timestamps(ack_path=ack_path) == set()

    def test_get_unacked_results(self, tmp_path):
        results_path = tmp_path / "results.jsonl"
        ack_path = tmp_path / "ack.jsonl"
        # Add 3 research results
        msg1 = Message(message="finding 1", timestamp="2025-01-01T00:00:00Z")
        msg2 = Message(message="finding 2", timestamp="2025-01-02T00:00:00Z")
        msg3 = Message(message="finding 3", timestamp="2025-01-03T00:00:00Z")
        append_message(results_path, msg1)
        append_message(results_path, msg2)
        append_message(results_path, msg3)
        # Ack one of them
        ack_research_result("2025-01-02T00:00:00Z", "adopted", ack_path=ack_path)
        # Should return 2 unacked
        unacked = get_unacked_results(results_path=results_path, ack_path=ack_path)
        assert len(unacked) == 2
        assert {m.timestamp for m in unacked} == {"2025-01-01T00:00:00Z", "2025-01-03T00:00:00Z"}
