# tests/test_orchestrate_supervisor.py
"""Integration tests for the orchestrator's non-agent functionality.

Does NOT test actual Claude Code agent spawning (requires API key).
Tests promotion queue reading, result logging, and budget gating.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from orchestrate import (
    _read_pending_promotions,
    _clear_promotion_queue,
    _append_result,
    parse_run_log,
)


class TestPromotionQueue:
    def test_reads_empty_queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr("orchestrate._PROMOTION_QUEUE", tmp_path / "promo.jsonl")
        assert _read_pending_promotions() == []

    def test_reads_valid_requests(self, tmp_path, monkeypatch):
        queue_path = tmp_path / "promo.jsonl"
        queue_path.write_text(
            json.dumps({"source_experiment": "abc", "message": "promote this"}) + "\n"
            + json.dumps({"source_experiment": "def", "message": "and this"}) + "\n"
        )
        monkeypatch.setattr("orchestrate._PROMOTION_QUEUE", queue_path)
        requests = _read_pending_promotions()
        assert len(requests) == 2
        assert requests[0]["source_experiment"] == "abc"

    def test_clear_queue(self, tmp_path, monkeypatch):
        queue_path = tmp_path / "promo.jsonl"
        queue_path.write_text('{"msg": "test"}\n')
        monkeypatch.setattr("orchestrate._PROMOTION_QUEUE", queue_path)
        _clear_promotion_queue()
        assert queue_path.read_text() == ""


class TestAppendResult:
    def test_creates_tsv_with_header(self, tmp_path, monkeypatch):
        tsv_path = tmp_path / "results.tsv"
        monkeypatch.setattr("orchestrate._RESULTS_TSV", tsv_path)
        _append_result("run1", "runpod", {"val_bpb": 1.15}, 3.50)
        content = tsv_path.read_text()
        assert "commit\ttier\tval_bpb" in content
        assert "run1\trunpod\t1.15" in content

    def test_appends_to_existing(self, tmp_path, monkeypatch):
        tsv_path = tmp_path / "results.tsv"
        tsv_path.write_text("commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n")
        monkeypatch.setattr("orchestrate._RESULTS_TSV", tsv_path)
        _append_result("run1", "local", {}, 0.0)
        _append_result("run2", "runpod", {"val_bpb": 1.12}, 3.50)
        lines = tsv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 results


class TestParseRunLog:
    def test_extracts_metrics(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("val_bpb: 1.1234\nval_loss: 0.789\nartifact_bytes: 15000000\ntraining_seconds: 598.5\n")
        result = parse_run_log(str(log))
        assert result["val_bpb"] == 1.1234
        assert result["artifact_bytes"] == 15000000.0
        assert result["training_seconds"] == 598.5

    def test_handles_missing_file(self, tmp_path):
        result = parse_run_log(str(tmp_path / "nope.log"))
        assert result == {}
