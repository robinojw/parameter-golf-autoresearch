"""Canonical run configuration and validation for H100 experiments.

Every env var that affects training quality is declared here with its
expected value, acceptable range, and validation logic. This prevents
silent environment divergence — the #1 cause of bpb regression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    """Expected configuration for an H100 run. Validated pre-flight and post-flight."""

    gpu_count: int = 8
    gpu_type: str = "NVIDIA H100 80GB HBM3"
    max_wallclock_seconds: float = 600.0
    train_shards_expected: int = 80
    loader_mode_expected: str = "coprime"
    fa3_required: bool = True
    compressor_expected: str = "brotli"
    max_artifact_bytes: int = 16_000_000
    min_steps_expected: int = 6000
    max_step_avg_ms: float = 95.0
    min_step_avg_ms: float = 70.0
    bpb_range: tuple[float, float] = (1.00, 1.25)


_WARN = "WARN"
_FAIL = "FAIL"
_OK = "OK"


@dataclass
class CheckResult:
    status: str  # OK, WARN, FAIL
    check: str
    message: str
    value: Any = None
    expected: Any = None


def validate_pre_flight(config: RunConfig) -> list[CheckResult]:
    """Validate configuration before creating a pod. Returns list of check results."""
    import os

    results: list[CheckResult] = []

    gpu_count = int(os.getenv("RUNPOD_GPU_COUNT", "8"))
    if gpu_count != config.gpu_count:
        results.append(
            CheckResult(
                _FAIL,
                "gpu_count",
                f"RUNPOD_GPU_COUNT={gpu_count}, expected {config.gpu_count}",
                gpu_count,
                config.gpu_count,
            )
        )
    else:
        results.append(CheckResult(_OK, "gpu_count", f"GPU count: {gpu_count}"))

    use_http = os.getenv("RUNPOD_USE_HTTP", "0")
    if use_http not in ("1", "true", "yes"):
        results.append(
            CheckResult(
                _WARN,
                "use_http",
                f"RUNPOD_USE_HTTP={use_http} — using SSH flow (HTTP recommended)",
                use_http,
                "1",
            )
        )
    else:
        results.append(CheckResult(_OK, "use_http", "HTTP flow enabled"))

    github_token = os.getenv("GITHUB_TOKEN", "")
    if not github_token:
        results.append(CheckResult(_FAIL, "github_token", "GITHUB_TOKEN not set"))
    else:
        results.append(CheckResult(_OK, "github_token", "GITHUB_TOKEN set"))

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        results.append(
            CheckResult(
                _WARN,
                "hf_token",
                "HF_TOKEN not set — data downloads will be slower",
            )
        )
    else:
        results.append(CheckResult(_OK, "hf_token", "HF_TOKEN set"))

    script = Path("train_gpt.py")
    if not script.exists():
        results.append(CheckResult(_FAIL, "train_script", "train_gpt.py not found"))
    else:
        code_bytes = script.stat().st_size
        if code_bytes > 200_000:
            results.append(
                CheckResult(
                    _WARN,
                    "code_size",
                    f"train_gpt.py is {code_bytes:,} bytes — watch artifact budget",
                    code_bytes,
                    200_000,
                )
            )
        else:
            results.append(CheckResult(_OK, "code_size", f"Code: {code_bytes:,} bytes"))

    return results


def validate_post_flight(config: RunConfig, run_log_path: str) -> list[CheckResult]:
    """Validate run.log after training completes. Catches silent environment issues."""
    results: list[CheckResult] = []
    log_path = Path(run_log_path)

    if not log_path.exists():
        results.append(
            CheckResult(_FAIL, "run_log", "run.log not found — pod startup failed")
        )
        return results

    log_text = log_path.read_text(errors="replace")

    # FA3 detection
    if "FA3 OK" in log_text or "flash_attn_3_func" in log_text:
        results.append(CheckResult(_OK, "flash_attn", "FA3 active"))
    elif "FA3 install failed" in log_text:
        results.append(
            CheckResult(
                _FAIL if config.fa3_required else _WARN,
                "flash_attn",
                "FA3 install failed — running FA2 (~18% slower, ~1000 fewer steps)",
            )
        )
    else:
        results.append(CheckResult(_WARN, "flash_attn", "FA3 status unknown from logs"))

    # Shard count
    shard_match = re.search(r"train_shards:(\d+)", log_text)
    if shard_match:
        shards = int(shard_match.group(1))
        if shards < config.train_shards_expected:
            results.append(
                CheckResult(
                    _FAIL,
                    "train_shards",
                    f"Only {shards} shards (expected {config.train_shards_expected})",
                    shards,
                    config.train_shards_expected,
                )
            )
        else:
            results.append(CheckResult(_OK, "train_shards", f"Train shards: {shards}"))
    else:
        results.append(
            CheckResult(_WARN, "train_shards", "Shard count not found in log")
        )

    # World size
    ws_match = re.search(r"world_size:(\d+)", log_text)
    if ws_match:
        ws = int(ws_match.group(1))
        if ws != config.gpu_count:
            results.append(
                CheckResult(
                    _FAIL,
                    "world_size",
                    f"world_size={ws}, expected {config.gpu_count}",
                    ws,
                    config.gpu_count,
                )
            )
        else:
            results.append(CheckResult(_OK, "world_size", f"World size: {ws}"))

    # Step average (detects FA3 vs FA2)
    step_match = re.search(r"step_avg:([\d.]+)ms", log_text)
    if step_match:
        step_avg = float(step_match.group(1))
        if step_avg > config.max_step_avg_ms:
            results.append(
                CheckResult(
                    _WARN,
                    "step_avg",
                    f"step_avg={step_avg:.1f}ms > {config.max_step_avg_ms}ms — likely FA2 not FA3",
                    step_avg,
                    config.max_step_avg_ms,
                )
            )
        elif step_avg < config.min_step_avg_ms:
            results.append(
                CheckResult(
                    _WARN,
                    "step_avg",
                    f"step_avg={step_avg:.1f}ms — unusually fast, verify correctness",
                    step_avg,
                    config.min_step_avg_ms,
                )
            )
        else:
            results.append(CheckResult(_OK, "step_avg", f"Step avg: {step_avg:.1f}ms"))

    # Total steps
    step_counts = re.findall(r"step:(\d+)/\d+", log_text)
    if step_counts:
        max_step = max(int(s) for s in step_counts)
        if max_step < config.min_steps_expected:
            results.append(
                CheckResult(
                    _WARN,
                    "total_steps",
                    f"Only {max_step} steps (expected ≥{config.min_steps_expected})",
                    max_step,
                    config.min_steps_expected,
                )
            )
        else:
            results.append(CheckResult(_OK, "total_steps", f"Total steps: {max_step}"))

    # Compression algorithm
    if (
        "compressor:brotli" in log_text
        or "_COMPRESSOR=brotli" in log_text
        or "final_model.int6.br" in log_text
    ):
        results.append(CheckResult(_OK, "compressor", "Using brotli"))
    elif "compressor:zstd" in log_text or "final_model.int6.zst" in log_text:
        results.append(
            CheckResult(_WARN, "compressor", "Using zstd (brotli saves ~4.8MB)")
        )
    elif "compressor:zlib" in log_text:
        results.append(
            CheckResult(
                _FAIL,
                "compressor",
                "Using zlib — artifact will be ~4.8MB larger than brotli",
            )
        )

    # Loader mode
    loader_match = re.search(r"loader:(coprime|sequential)", log_text)
    if loader_match:
        mode = loader_match.group(1)
        if mode != config.loader_mode_expected:
            results.append(
                CheckResult(
                    _WARN,
                    "loader_mode",
                    f"Loader: {mode} (expected {config.loader_mode_expected})",
                    mode,
                    config.loader_mode_expected,
                )
            )
        else:
            results.append(CheckResult(_OK, "loader_mode", f"Loader: {mode}"))

    # Final val_bpb
    bpb_matches = re.findall(r"val_bpb:([\d.]+)", log_text)
    if bpb_matches:
        final_bpb = float(bpb_matches[-1])
        lo, hi = config.bpb_range
        if final_bpb < lo or final_bpb > hi:
            results.append(
                CheckResult(
                    _WARN,
                    "val_bpb",
                    f"val_bpb={final_bpb:.4f} outside expected range [{lo}, {hi}]",
                    final_bpb,
                    config.bpb_range,
                )
            )
        else:
            results.append(CheckResult(_OK, "val_bpb", f"val_bpb: {final_bpb:.4f}"))

    # Artifact size
    artifact_match = re.search(r"artifact.*?(\d{5,})\s*bytes", log_text)
    if not artifact_match:
        artifact_match = re.search(r"file_bytes:(\d+)", log_text)
    if artifact_match:
        artifact_bytes = int(artifact_match.group(1))
        if artifact_bytes > config.max_artifact_bytes:
            results.append(
                CheckResult(
                    _FAIL,
                    "artifact_size",
                    f"Artifact {artifact_bytes:,} > {config.max_artifact_bytes:,} (OVER LIMIT)",
                    artifact_bytes,
                    config.max_artifact_bytes,
                )
            )
        else:
            results.append(
                CheckResult(_OK, "artifact_size", f"Artifact: {artifact_bytes:,} bytes")
            )

    return results


def format_checks(checks: list[CheckResult]) -> str:
    """Format check results for logging."""
    lines = []
    fails = sum(1 for c in checks if c.status == _FAIL)
    warns = sum(1 for c in checks if c.status == _WARN)
    oks = sum(1 for c in checks if c.status == _OK)

    for c in checks:
        icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}[c.status]
        lines.append(f"  {icon} {c.check}: {c.message}")

    summary = f"  {oks} passed, {warns} warnings, {fails} failures"
    lines.append(summary)
    return "\n".join(lines)


def has_failures(checks: list[CheckResult]) -> bool:
    return any(c.status == _FAIL for c in checks)
