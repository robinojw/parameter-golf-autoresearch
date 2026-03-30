# tests/test_micro_run.py
import os
import shutil
import textwrap

import pytest

from research.tools.micro_run import MicroRunResult, run_micro_experiment

# MLX tests need a Python interpreter with mlx installed.
# Set MLX_PYTHON env var to point to it, or these tests will be skipped.
_mlx_python = os.environ.get("MLX_PYTHON", "")
_has_mlx = _mlx_python and shutil.which(_mlx_python)

requires_mlx = pytest.mark.skipif(
    not _has_mlx,
    reason="MLX_PYTHON not set or not found (set to a Python with mlx installed)",
)


class TestMicroRunResult:
    def test_default_fields(self):
        r = MicroRunResult(
            status="pass", iterations=50,
            initial_loss=9.0, final_loss=7.5,
            loss_decreased=True, ms_per_iter=300.0,
            artifact_bytes=12000000, error="",
        )
        assert r.status == "pass"
        assert r.loss_decreased is True


class TestRunMicroExperiment:
    @requires_mlx
    def test_empty_diff_runs_baseline(self):
        """Empty diff = unmodified train_gpt_mlx.py. Should complete without crash."""
        result = run_micro_experiment(diff="", iterations=10)
        assert result.status in ("pass", "no_signal")
        assert result.iterations == 10
        assert result.initial_loss > 0
        assert result.final_loss > 0
        assert result.ms_per_iter > 0
        assert result.error == ""

    @requires_mlx
    def test_syntax_error_diff_crashes(self):
        """A diff that introduces a syntax error should return crash status."""
        bad_diff = textwrap.dedent("""\
            --- a/train_gpt_mlx.py
            +++ b/train_gpt_mlx.py
            @@ -1,3 +1,3 @@
            -import math
            +import math; this is not valid python syntax !!!
             import os
             import time
        """)
        result = run_micro_experiment(diff=bad_diff, iterations=10)
        assert result.status == "crash"
        assert result.error != ""

    @requires_mlx
    def test_timeout_returns_crash(self):
        """A 1-second timeout on a real run should trigger timeout."""
        result = run_micro_experiment(diff="", iterations=1000, timeout=1)
        assert result.status == "crash"
        assert "timeout" in result.error.lower() or result.iterations < 1000

    @requires_mlx
    def test_result_has_artifact_bytes(self):
        """Baseline run should report artifact size."""
        result = run_micro_experiment(diff="", iterations=10)
        assert result.artifact_bytes >= 0
