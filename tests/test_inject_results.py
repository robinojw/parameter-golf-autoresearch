# tests/test_inject_results.py
import json
from pathlib import Path
from unittest.mock import patch

from agents.shared import Message, read_messages_since


class TestAppendToResearchResults:
    def test_creates_results_file(self, tmp_path):
        from research.inject import append_to_research_results
        path = tmp_path / "research_results.jsonl"
        append_to_research_results(
            "Found new int4 technique from arxiv",
            priority="high",
            results_path=path,
        )
        assert path.exists()
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 1
        assert "int4" in msgs[0].message
        assert msgs[0].priority == "high"

    def test_appends_multiple(self, tmp_path):
        from research.inject import append_to_research_results
        path = tmp_path / "research_results.jsonl"
        append_to_research_results("first", results_path=path)
        append_to_research_results("second", results_path=path)
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 2
