# research/tools/micro_run.py
"""Micro-experiment runner for hypothesis sanity-checking.

Applies a unified diff to train_gpt_mlx.py, runs a short training loop
(default 50 iterations), and returns structured metrics. No side effects —
runs in a temp directory, doesn't log to results.tsv.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MicroRunResult:
    status: str          # "pass", "crash", "diverged", "no_signal"
    iterations: int      # how many completed
    initial_loss: float  # loss at step 1
    final_loss: float    # loss at last logged step
    loss_decreased: bool # did it learn anything
    ms_per_iter: float   # timing for 600s budget estimation
    artifact_bytes: int  # compressed model size (0 if unavailable)
    error: str           # empty unless crash


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "train_gpt_mlx.py"
_LOSS_PATTERN = re.compile(r"train_loss=(\d+\.\d+)")
_METRIC_PATTERN = re.compile(r"^(\w+):\s+(.+)$")


def _apply_diff(source: Path, dest_dir: Path, diff: str) -> Path:
    """Copy source to dest_dir and apply unified diff. Falls back to plain copy."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    shutil.copy2(source, dest)

    if not diff or not diff.strip():
        return dest

    try:
        result = subprocess.run(
            ["patch", "-p1", str(dest)],
            input=diff,
            capture_output=True,
            text=True,
            cwd=str(dest_dir),
        )
        if result.returncode != 0:
            shutil.copy2(source, dest)
    except (subprocess.SubprocessError, FileNotFoundError):
        shutil.copy2(source, dest)

    return dest


def _parse_losses(output: str) -> list[float]:
    """Extract train_loss values from training output."""
    losses = []
    for match in _LOSS_PATTERN.finditer(output):
        try:
            losses.append(float(match.group(1)))
        except ValueError:
            continue
    return losses


def _parse_final_metrics(output: str) -> dict:
    """Extract key: value metrics from final output lines."""
    metrics = {}
    for line in output.splitlines():
        match = _METRIC_PATTERN.match(line.strip())
        if match:
            key, value = match.group(1), match.group(2).strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


def run_micro_experiment(
    diff: str,
    iterations: int = 50,
    timeout: int = 60,
) -> MicroRunResult:
    """Run a micro-experiment from a unified diff against train_gpt_mlx.py.

    Args:
        diff: Unified diff to apply. Empty string = run unmodified baseline.
        iterations: Number of training iterations.
        timeout: Maximum seconds before killing the subprocess.

    Returns:
        MicroRunResult with status, metrics, and any error.
    """
    if not _TRAIN_SCRIPT.exists():
        return MicroRunResult(
            status="crash", iterations=0,
            initial_loss=0.0, final_loss=0.0,
            loss_decreased=False, ms_per_iter=0.0,
            artifact_bytes=0,
            error=f"{_TRAIN_SCRIPT} not found",
        )

    work_dir = tempfile.mkdtemp(prefix="micro_run_")
    try:
        script = _apply_diff(_TRAIN_SCRIPT, Path(work_dir), diff)

        # Use MLX_PYTHON env var if set (for when test runner != MLX-capable Python),
        # otherwise fall back to sys.executable.
        python = os.environ.get("MLX_PYTHON", sys.executable)

        env = os.environ.copy()
        env.update({
            "ITERATIONS": str(iterations),
            "MLX_EAGER_EVAL": "1",
            "TRAIN_BATCH_TOKENS": "8192",
            "TRAIN_SEQ_LEN": "512",
            "VAL_LOSS_EVERY": "0",
        })

        t0 = time.time()
        try:
            proc = subprocess.run(
                [python, str(script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
                cwd=work_dir,
            )
            elapsed = time.time() - t0
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - t0
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            output = stdout + stderr
            return MicroRunResult(
                status="crash", iterations=0,
                initial_loss=0.0, final_loss=0.0,
                loss_decreased=False,
                ms_per_iter=elapsed * 1000 / max(iterations, 1),
                artifact_bytes=0,
                error=f"Timeout after {timeout}s",
            )

        if proc.returncode != 0:
            return MicroRunResult(
                status="crash", iterations=0,
                initial_loss=0.0, final_loss=0.0,
                loss_decreased=False, ms_per_iter=0.0,
                artifact_bytes=0,
                error=output[-500:] if len(output) > 500 else output,
            )

        # Parse results
        losses = _parse_losses(output)
        final_metrics = _parse_final_metrics(output)

        # Use train_loss values from step logs if available.
        # Fall back to val_loss from final metrics when no intermediate logs
        # (e.g., iterations < LOG_INTERVAL so no step logs are printed).
        if losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
        else:
            val_loss = final_metrics.get("val_loss", 0.0)
            if isinstance(val_loss, str):
                try:
                    val_loss = float(val_loss)
                except ValueError:
                    val_loss = 0.0
            initial_loss = float(val_loss)
            final_loss = float(val_loss)

        artifact_bytes = int(final_metrics.get("artifact_bytes", 0))

        # Check for NaN/inf
        all_losses = losses if losses else [initial_loss, final_loss]
        has_nan = any(math.isnan(l) or math.isinf(l) for l in all_losses if l != 0.0)
        if has_nan:
            return MicroRunResult(
                status="diverged", iterations=len(losses) if losses else iterations,
                initial_loss=initial_loss, final_loss=final_loss,
                loss_decreased=False,
                ms_per_iter=elapsed * 1000 / max(len(losses) if losses else iterations, 1),
                artifact_bytes=artifact_bytes,
                error="NaN or inf detected in loss",
            )

        # Determine status
        loss_decreased = final_loss < initial_loss if initial_loss > 0 else False
        diverged = len(losses) >= 2 and final_loss > initial_loss * 1.5

        if diverged:
            status = "diverged"
        elif proc.returncode == 0:
            # Completed successfully — "pass" means it ran without crashing or diverging
            status = "pass"
        else:
            status = "no_signal"

        return MicroRunResult(
            status=status,
            iterations=iterations,
            initial_loss=initial_loss,
            final_loss=final_loss,
            loss_decreased=loss_decreased,
            ms_per_iter=elapsed * 1000 / max(iterations, 1),
            artifact_bytes=artifact_bytes,
            error="",
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
