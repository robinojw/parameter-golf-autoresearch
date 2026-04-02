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


class TestDeduplication:
    def test_exact_duplicate_is_dropped(self, tmp_path):
        from research.inject import append_to_research_results
        path = tmp_path / "research_results.jsonl"
        msg = "NorMuon variance reduction technique for optimizer convergence improvement"
        append_to_research_results(msg, results_path=path)
        append_to_research_results(msg, results_path=path)
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 1

    def test_near_duplicate_is_dropped(self, tmp_path):
        from research.inject import append_to_research_results
        path = tmp_path / "research_results.jsonl"
        append_to_research_results(
            "NorMuon variance reduction technique for optimizer convergence",
            results_path=path,
        )
        # Same idea, slightly reworded — should be dropped
        append_to_research_results(
            "NorMuon variance reduction approach for improved optimizer convergence",
            results_path=path,
        )
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 1

    def test_different_topics_both_kept(self, tmp_path):
        from research.inject import append_to_research_results
        path = tmp_path / "research_results.jsonl"
        append_to_research_results(
            "NorMuon variance reduction for optimizers",
            results_path=path,
        )
        append_to_research_results(
            "GPTQ full Hessian quantization with Cholesky decomposition",
            results_path=path,
        )
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 2

    def test_empty_file_no_dedup(self, tmp_path):
        from research.inject import append_to_research_results
        path = tmp_path / "research_results.jsonl"
        # First message should always be appended
        append_to_research_results("any message", results_path=path)
        msgs = read_messages_since(path, since=0.0)
        assert len(msgs) == 1


class TestKeywordExtraction:
    def test_extract_keywords_filters_stopwords(self):
        from research.inject import _extract_keywords
        kw = _extract_keywords("The quick brown fox is very fast")
        assert "the" not in kw
        assert "very" not in kw
        assert "quick" in kw
        assert "brown" in kw
        assert "fox" in kw
        assert "fast" in kw

    def test_extract_keywords_filters_short_tokens(self):
        from research.inject import _extract_keywords
        kw = _extract_keywords("AI is OK but ML works")
        # "AI", "is", "OK", "ML" are all 2 chars -> filtered
        assert "ai" not in kw
        assert "works" in kw

    def test_is_duplicate_below_threshold(self, tmp_path):
        from research.inject import _is_duplicate
        import json
        path = tmp_path / "results.jsonl"
        # Write an existing entry about optimizers
        entry = {"timestamp": "2025-01-01T00:00:00Z", "message": "optimizer convergence NorMuon variance reduction"}
        path.write_text(json.dumps(entry) + "\n")
        # Check with a completely different topic
        assert not _is_duplicate("GPTQ quantization Cholesky full Hessian decomposition", path)
