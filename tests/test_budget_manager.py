# tests/test_budget_manager.py
"""Tests for BudgetManager: rate limiting, best_bpb tracking, persistence."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_budget(tmp_path, monkeypatch):
    """Ensure BudgetManager reads/writes to a temp dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("compute.budget.BUDGET_FILE", tmp_path / "budget.json")
    monkeypatch.setattr(
        "compute.dashboard.DashboardPusher.push_budget", lambda *a, **kw: None
    )


class TestRateLimiting:
    def test_no_runs_not_rate_limited(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        ok, msg = bm.can_submit()
        assert ok is True

    def test_recent_run_is_rate_limited(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        # Simulate a run started 10 minutes ago
        recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        bm.runs = [{"started_at": recent_ts}]
        ok, msg = bm.can_submit()
        assert ok is False
        assert "Rate limited" in msg

    def test_old_run_not_rate_limited(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        # Simulate a run started 2 hours ago
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        bm.runs = [{"started_at": old_ts}]
        ok, msg = bm.can_submit()
        assert ok is True

    def test_missing_timestamp_not_rate_limited(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        bm.runs = [{"started_at": ""}]
        ok, msg = bm.can_submit()
        assert ok is True


class TestBestBpbTracking:
    def test_update_best_bpb_first_run(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        assert bm.best_h100_bpb == 0.0
        bm.update_best_bpb(1.15)
        assert bm.best_h100_bpb == 1.15

    def test_update_best_bpb_improves(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        bm.update_best_bpb(1.15)
        bm.update_best_bpb(1.10)
        assert bm.best_h100_bpb == 1.10

    def test_update_best_bpb_does_not_regress(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        bm.update_best_bpb(1.10)
        bm.update_best_bpb(1.20)
        assert bm.best_h100_bpb == 1.10

    def test_update_best_bpb_ignores_zero(self):
        from compute.budget import BudgetManager

        bm = BudgetManager(total_credits=200)
        bm.update_best_bpb(1.10)
        bm.update_best_bpb(0.0)
        assert bm.best_h100_bpb == 1.10

    def test_best_bpb_persists(self, tmp_path, monkeypatch):
        from compute.budget import BudgetManager, BUDGET_FILE

        bm = BudgetManager(total_credits=200)
        bm.update_best_bpb(1.12)
        # Reload
        bm2 = BudgetManager(total_credits=200)
        assert bm2.best_h100_bpb == 1.12
