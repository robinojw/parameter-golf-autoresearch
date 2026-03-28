"""Tests for compute.tournament — tournament hypothesis testing."""

import json
import shutil
from pathlib import Path

from compute.tournament import (
    _apply_diff_to_copy,
    _parse_run_log,
    _rank_candidates,
    TournamentConfig,
)


class TestApplyDiffToCopy:
    def test_copies_file_and_applies_simple_diff(self, tmp_path):
        source = tmp_path / "train_gpt_mlx.py"
        source.write_text("line1\nline2\nline3\n")
        dest_dir = tmp_path / "candidate_a"
        diff = (
            "--- a/train_gpt_mlx.py\n"
            "+++ b/train_gpt_mlx.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+modified_line2\n"
            " line3\n"
        )
        result = _apply_diff_to_copy(source, dest_dir, diff)
        assert result.exists()
        content = result.read_text()
        assert "modified_line2" in content

    def test_falls_back_to_plain_copy_on_bad_diff(self, tmp_path):
        source = tmp_path / "train_gpt_mlx.py"
        source.write_text("original content\n")
        dest_dir = tmp_path / "candidate_b"
        result = _apply_diff_to_copy(source, dest_dir, "this is not a valid diff")
        assert result.exists()
        assert "original content" in result.read_text()


class TestParseRunLog:
    def test_extracts_val_bpb(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("epoch 1\nval_bpb:           1.1832\nartifact_bytes:    15000000\n")
        result = _parse_run_log(log)
        assert result["val_bpb"] == 1.1832
        assert result["artifact_bytes"] == 15000000.0

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        assert _parse_run_log(tmp_path / "nonexistent.log") == {}

    def test_returns_empty_dict_for_no_metrics(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("training started\nno metrics here\n")
        assert "val_bpb" not in _parse_run_log(log)


class TestRankCandidates:
    def test_ranks_by_val_bpb_ascending(self):
        candidates = [
            {"name": "a", "val_bpb": 1.19},
            {"name": "b", "val_bpb": 1.17},
            {"name": "c", "val_bpb": 1.18},
        ]
        ranked = _rank_candidates(candidates)
        assert ranked[0]["name"] == "b"
        assert ranked[1]["name"] == "c"
        assert ranked[2]["name"] == "a"

    def test_handles_missing_val_bpb(self):
        candidates = [
            {"name": "a", "val_bpb": 1.19},
            {"name": "b", "val_bpb": None},
            {"name": "c", "val_bpb": 1.18},
        ]
        ranked = _rank_candidates(candidates)
        assert ranked[-1]["name"] == "b"

    def test_empty_list(self):
        assert _rank_candidates([]) == []


class TestTournamentConfig:
    def test_defaults(self):
        cfg = TournamentConfig()
        assert cfg.candidates == 4
        assert cfg.survivors == 2
        assert cfg.elim_iterations == 100
        assert cfg.full_iterations == 500
        assert cfg.parallel == 1
        assert cfg.cooldown == 3
