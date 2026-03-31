"""Tests for orchestrate._handle_promotion with preflight and typed errors."""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestHandlePromotionPreflight:
    """Preflight verification blocks promotion before any pod is launched."""

    def test_preflight_syntax_error_blocks_promotion(self, tmp_path, monkeypatch):
        # Write a broken script
        script = tmp_path / "train_gpt.py"
        script.write_text("def broken(\n")
        monkeypatch.chdir(tmp_path)

        # Mock budget, threshold, and bpb — these are lazily imported from their
        # source modules, so we patch at the source.
        monkeypatch.setattr(
            "orchestrate._make_budget_manager",
            lambda: MagicMock(can_submit=MagicMock(return_value=(True, "ok"))),
        )
        monkeypatch.setattr(
            "research.experiments.get_current_best_bpb", lambda: 1.15
        )
        monkeypatch.setattr(
            "compute.threshold.compute_promotion_threshold", lambda *a, **kw: 0.99
        )

        # Track whether create_pod is called
        mock_client = MagicMock()
        import compute.runpod_client as rc_mod
        original_rc = rc_mod.RunPodClient
        monkeypatch.setattr(rc_mod, "RunPodClient", lambda **kw: mock_client)
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        from orchestrate import _handle_promotion

        # Should not raise — preflight error is caught and logged
        _handle_promotion({"source_experiment": "abc123", "message": "test"})

        # create_pod should never have been called (preflight blocked it)
        mock_client.create_pod.assert_not_called()

    def test_preflight_passes_allows_pod_launch(self, tmp_path, monkeypatch):
        """When preflight passes, create_pod is called."""
        script = tmp_path / "train_gpt.py"
        script.write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runpod_results").mkdir()

        monkeypatch.setattr(
            "orchestrate._make_budget_manager",
            lambda: MagicMock(
                can_submit=MagicMock(return_value=(True, "ok")),
                record_run=MagicMock(return_value=1.50),
            ),
        )
        monkeypatch.setattr(
            "research.experiments.get_current_best_bpb", lambda: 1.15
        )
        monkeypatch.setattr(
            "compute.threshold.compute_promotion_threshold", lambda *a, **kw: 0.99
        )
        monkeypatch.setattr("orchestrate._append_result", lambda *a, **kw: None)
        monkeypatch.setattr("orchestrate.parse_run_log", lambda *a: {"val_bpb": 1.10})

        mock_client = MagicMock()
        mock_client.create_pod.return_value = "pod-xyz"
        mock_client.wait_for_ready.return_value = "root@1.2.3.4 -p 17445"

        import compute.runpod_client as rc_mod
        monkeypatch.setattr(rc_mod, "RunPodClient", lambda **kw: mock_client)

        monkeypatch.setattr("compute.sync.push_to_pod", lambda *a, **kw: None)
        monkeypatch.setattr(
            "compute.sync.run_remote_training", lambda *a, **kw: 0
        )
        monkeypatch.setattr("compute.sync.pull_from_pod", lambda *a, **kw: None)

        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        from orchestrate import _handle_promotion
        _handle_promotion({"source_experiment": "abc123", "message": "test"})

        mock_client.create_pod.assert_called_once()


class TestHandlePromotionErrorHandling:
    """Typed errors result in pod termination and structured logging."""

    def test_ssh_error_terminates_pod(self, tmp_path, monkeypatch):
        from compute.runpod_client import SSHConnectionError

        script = tmp_path / "train_gpt.py"
        script.write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "runpod_results").mkdir()

        monkeypatch.setattr(
            "orchestrate._make_budget_manager",
            lambda: MagicMock(
                can_submit=MagicMock(return_value=(True, "ok")),
                record_run=MagicMock(return_value=1.50),
            ),
        )
        monkeypatch.setattr(
            "research.experiments.get_current_best_bpb", lambda: 1.15
        )
        monkeypatch.setattr(
            "compute.threshold.compute_promotion_threshold", lambda *a, **kw: 0.99
        )
        monkeypatch.setattr("orchestrate._append_result", lambda *a, **kw: None)

        mock_client = MagicMock()
        mock_client.create_pod.return_value = "pod-xyz"
        mock_client.wait_for_ready.side_effect = SSHConnectionError(
            "SSH failed", pod_id="pod-xyz"
        )

        import compute.runpod_client as rc_mod
        monkeypatch.setattr(rc_mod, "RunPodClient", lambda **kw: mock_client)
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        from orchestrate import _handle_promotion
        _handle_promotion({"source_experiment": "abc123", "message": "test"})

        # Pod should be terminated (called from both except and finally blocks)
        assert mock_client.terminate_pod.call_count >= 1
        mock_client.terminate_pod.assert_any_call("pod-xyz")
