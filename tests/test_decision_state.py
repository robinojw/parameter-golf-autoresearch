# tests/test_decision_state.py
"""Tests for the decision-state handoff artifact generator."""

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Run all tests in a temp dir with mocked file paths."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "agents.decision_state._BUDGET_PATH", tmp_path / "budget.json"
    )
    monkeypatch.setattr(
        "agents.decision_state._TECHNIQUE_MAP_PATH", tmp_path / "technique_map.json"
    )
    monkeypatch.setattr(
        "agents.decision_state._DECISION_STATE_PATH", tmp_path / "decision_state.md"
    )
    monkeypatch.setattr(
        "agents.shared.RESEARCH_RESULTS_PATH", tmp_path / "research_results.jsonl"
    )
    monkeypatch.setattr(
        "agents.shared.RESEARCH_ACK_PATH", tmp_path / "research_ack.jsonl"
    )
    monkeypatch.setattr(
        "research.experiments.RESULTS_TSV_PATH", tmp_path / "results.tsv"
    )


class TestGenerateDecisionState:
    def test_produces_markdown_with_best_bpb(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agents.decision_state.get_current_best_bpb", lambda: 1.1147
        )
        monkeypatch.setattr(
            "agents.decision_state.get_proven_techniques",
            lambda: ["EMA", "BigramHash"],
        )
        monkeypatch.setattr(
            "agents.decision_state.get_failed_experiments", lambda: []
        )

        from agents.decision_state import generate_decision_state

        md = generate_decision_state()
        assert "# Decision State" in md
        assert "1.1147" in md
        assert "EMA" in md
        assert "BigramHash" in md

    def test_includes_dead_ends_from_technique_map(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agents.decision_state.get_current_best_bpb", lambda: 1.15
        )
        monkeypatch.setattr(
            "agents.decision_state.get_proven_techniques", lambda: []
        )
        monkeypatch.setattr(
            "agents.decision_state.get_failed_experiments", lambda: []
        )

        technique_map = {
            "nodes": {
                "P2_focal_loss": {"status": "dead_end", "best_bpb": 1.23},
                "EMA": {"status": "proven", "best_bpb": 1.11},
            },
            "edges": [],
        }
        (tmp_path / "technique_map.json").write_text(json.dumps(technique_map))

        from agents.decision_state import generate_decision_state

        md = generate_decision_state()
        assert "Dead Ends" in md
        assert "P2_focal_loss" in md
        assert "EMA" not in md.split("Dead Ends")[1].split("##")[0]  # EMA not in dead ends section

    def test_includes_budget_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agents.decision_state.get_current_best_bpb", lambda: 1.15
        )
        monkeypatch.setattr(
            "agents.decision_state.get_proven_techniques", lambda: []
        )
        monkeypatch.setattr(
            "agents.decision_state.get_failed_experiments", lambda: []
        )

        budget = {
            "total_credits": 200,
            "spent": 79.83,
            "min_reserve": 50,
            "runs": [],
        }
        (tmp_path / "budget.json").write_text(json.dumps(budget))

        from agents.decision_state import generate_decision_state

        md = generate_decision_state()
        assert "$79.83" in md
        assert "$200.00" in md


class TestWriteDecisionState:
    def test_writes_file_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agents.decision_state.get_current_best_bpb", lambda: 1.15
        )
        monkeypatch.setattr(
            "agents.decision_state.get_proven_techniques", lambda: []
        )
        monkeypatch.setattr(
            "agents.decision_state.get_failed_experiments", lambda: []
        )

        from agents.decision_state import write_decision_state

        path = write_decision_state(output_path=tmp_path / "ds.md")
        assert path.exists()
        content = path.read_text()
        assert "Decision State" in content
