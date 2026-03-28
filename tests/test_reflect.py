"""Tests for research.reflect — reflection cycle and strategy.md management."""

import json
from pathlib import Path

from research.reflect import (
    _build_reflection_prompt,
    _parse_reflection_response,
    _write_strategy_md,
    _read_strategy_md,
    _should_reflect,
    MAX_STRATEGY_ENTRIES,
)


class TestShouldReflect:
    def test_returns_true_when_no_strategy_file(self, tmp_path):
        strategy = tmp_path / "strategy.md"
        results = tmp_path / "results.tsv"
        results.write_text(
            "commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n"
            "abc\tlocal\t1.15\t\t\tkeep\tno\t0\ttry X\t\n"
        )
        assert _should_reflect(strategy_path=strategy, results_path=results) is True

    def test_returns_false_when_no_new_experiments(self, tmp_path):
        strategy = tmp_path / "strategy.md"
        strategy.write_text("---\nlast_reflection: 2099-01-01T00:00:00Z\n---\n\n## Entry\nContent\n")
        results = tmp_path / "results.tsv"
        results.write_text(
            "commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n"
        )
        assert _should_reflect(strategy_path=strategy, results_path=results) is False

    def test_returns_true_when_results_exist_and_no_prior_reflection(self, tmp_path):
        strategy = tmp_path / "strategy.md"
        results = tmp_path / "results.tsv"
        results.write_text(
            "commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n"
            "abc\tlocal\t1.15\t\t\tkeep\tno\t0\ttry X\t\n"
        )
        assert _should_reflect(strategy_path=strategy, results_path=results) is True


class TestBuildReflectionPrompt:
    def test_includes_experiments(self):
        experiments = [
            {"description": "try QAT", "val_bpb": 1.15, "status": "keep", "tier": "local"},
            {"description": "try Mamba", "val_bpb": 0.0, "status": "crash", "tier": "local"},
        ]
        prompt = _build_reflection_prompt(
            recent_experiments=experiments, current_sota=1.1194,
            previous_strategy="No prior strategy.", technique_map=None,
        )
        assert "try QAT" in prompt
        assert "try Mamba" in prompt
        assert "1.1194" in prompt


class TestParseReflectionResponse:
    def test_parses_valid_json(self):
        response = json.dumps({
            "failure_patterns": ["quantization below int6 always crashes"],
            "exhausted_dimensions": ["extreme quantization"],
            "promising_dimensions": ["test-time training"],
            "working_hypothesis": "TTT is the path forward",
            "recommended_next": [{"idea": "add TTT", "rationale": "untried", "estimated_impact": "2%"}],
            "technique_updates": [{"node": "int4_QAT", "status": "dead_end", "parent": "int6_QAT", "relation": "refinement"}],
        })
        result = _parse_reflection_response(response)
        assert result["failure_patterns"] == ["quantization below int6 always crashes"]
        assert len(result["technique_updates"]) == 1

    def test_handles_markdown_fenced_json(self):
        response = "```json\n" + json.dumps({
            "failure_patterns": [], "exhausted_dimensions": [], "promising_dimensions": [],
            "working_hypothesis": "test", "recommended_next": [], "technique_updates": [],
        }) + "\n```"
        result = _parse_reflection_response(response)
        assert result["working_hypothesis"] == "test"


class TestWriteStrategyMd:
    def test_creates_new_file(self, tmp_path):
        path = tmp_path / "strategy.md"
        entry = {
            "failure_patterns": ["pattern A"], "exhausted_dimensions": ["dim X"],
            "promising_dimensions": ["dim Y"], "working_hypothesis": "hypothesis Z",
            "recommended_next": [{"idea": "do thing", "rationale": "because", "estimated_impact": "1%"}],
        }
        _write_strategy_md(entry, strategy_path=path)
        content = path.read_text()
        assert "last_reflection:" in content
        assert "pattern A" in content
        assert "hypothesis Z" in content

    def test_prepends_to_existing_and_caps_entries(self, tmp_path):
        path = tmp_path / "strategy.md"
        for i in range(MAX_STRATEGY_ENTRIES):
            _write_strategy_md({
                "failure_patterns": [f"old pattern {i}"], "exhausted_dimensions": [],
                "promising_dimensions": [], "working_hypothesis": f"old hypothesis {i}",
                "recommended_next": [],
            }, strategy_path=path)
        _write_strategy_md({
            "failure_patterns": ["newest pattern"], "exhausted_dimensions": [],
            "promising_dimensions": [], "working_hypothesis": "newest hypothesis",
            "recommended_next": [],
        }, strategy_path=path)
        content = path.read_text()
        assert "newest pattern" in content
        assert content.count("## 20") <= MAX_STRATEGY_ENTRIES


class TestReadStrategyMd:
    def test_returns_empty_string_when_no_file(self, tmp_path):
        assert _read_strategy_md(strategy_path=tmp_path / "strategy.md") == ""

    def test_returns_most_recent_entry(self, tmp_path):
        path = tmp_path / "strategy.md"
        _write_strategy_md({
            "failure_patterns": ["the pattern"], "exhausted_dimensions": [],
            "promising_dimensions": ["dim A"], "working_hypothesis": "the hypothesis",
            "recommended_next": [{"idea": "X", "rationale": "Y", "estimated_impact": "Z"}],
        }, strategy_path=path)
        text = _read_strategy_md(strategy_path=path)
        assert "the pattern" in text
        assert "the hypothesis" in text
