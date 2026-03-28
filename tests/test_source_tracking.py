"""Tests for source_item column in results.tsv and get_source_yield."""

import json
from pathlib import Path
from unittest import mock

from research.experiments import (
    _ExperimentRow,
    _parse_single_row,
    get_source_yield,
)


class TestExperimentRowSourceItem:
    def test_parse_row_with_source_item(self):
        raw = {
            "commit": "abc1234", "tier": "local", "val_bpb": "1.15",
            "artifact_bytes": "15000000", "memory_gb": "8", "status": "keep",
            "promoted": "no", "cost_usd": "0", "description": "try int5 QAT",
            "source_item": "arxiv:2401.12345",
        }
        row = _parse_single_row(raw)
        assert row.source_item == "arxiv:2401.12345"

    def test_parse_row_without_source_item(self):
        raw = {
            "commit": "abc1234", "tier": "local", "val_bpb": "1.15",
            "artifact_bytes": "15000000", "memory_gb": "8", "status": "keep",
            "promoted": "no", "cost_usd": "0", "description": "try int5 QAT",
        }
        row = _parse_single_row(raw)
        assert row.source_item == ""

    def test_parse_row_with_empty_source_item(self):
        raw = {
            "commit": "abc1234", "tier": "local", "val_bpb": "1.15",
            "artifact_bytes": "", "memory_gb": "", "status": "keep",
            "promoted": "no", "cost_usd": "0", "description": "original idea",
            "source_item": "",
        }
        row = _parse_single_row(raw)
        assert row.source_item == ""


class TestGetSourceYield:
    def _write_results_tsv(self, path: Path, rows: list[dict]) -> None:
        header = "commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n"
        path.write_text(header)
        with open(path, "a") as f:
            for r in rows:
                cols = [
                    r.get("commit", ""), r.get("tier", "local"),
                    str(r.get("val_bpb", "")), str(r.get("artifact_bytes", "")),
                    r.get("memory_gb", ""), r.get("status", "keep"),
                    r.get("promoted", "no"), str(r.get("cost_usd", "0")),
                    r.get("description", ""), r.get("source_item", ""),
                ]
                f.write("\t".join(cols) + "\n")

    def _write_graded_cache(self, path: Path, items: list[dict]) -> None:
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_counts_per_source(self, tmp_path):
        results = tmp_path / "results.tsv"
        graded = tmp_path / "graded_cache.jsonl"
        self._write_results_tsv(results, [
            {"commit": "a", "description": "try QAT", "source_item": "arxiv:111"},
            {"commit": "b", "description": "try EMA", "source_item": "arxiv:222"},
            {"commit": "c", "description": "muon opt", "source_item": "s2:333"},
            {"commit": "d", "description": "original", "source_item": ""},
        ])
        self._write_graded_cache(graded, [
            {"id": "arxiv:111", "source": "arxiv", "score": 12},
            {"id": "arxiv:222", "source": "arxiv", "score": 8},
            {"id": "s2:333", "source": "semantic_scholar", "score": 11},
        ])
        with mock.patch("research.experiments.RESULTS_TSV_PATH", results), \
             mock.patch("research.experiments.GRADED_CACHE_PATH", graded):
            yield_data = get_source_yield()
        assert yield_data["arxiv"]["items_tried"] == 2
        assert yield_data["semantic_scholar"]["items_tried"] == 1

    def test_empty_results_returns_empty_dict(self, tmp_path):
        results = tmp_path / "results.tsv"
        graded = tmp_path / "graded_cache.jsonl"
        results.write_text("commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n")
        graded.write_text("")
        with mock.patch("research.experiments.RESULTS_TSV_PATH", results), \
             mock.patch("research.experiments.GRADED_CACHE_PATH", graded):
            yield_data = get_source_yield()
        assert yield_data == {}
