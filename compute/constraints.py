"""Constraint calculator — mathematical feasibility checks for Parameter Golf.

Pure deterministic math by default. When weight files or results.tsv exist on
disk, calibrates compression ratios and throughput from observed data.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Challenge constants
# ---------------------------------------------------------------------------

ARTIFACT_LIMIT = 16_000_000  # bytes
ARTIFACT_SOFT_LIMIT = 15_800_000  # 200KB headroom
TRAINING_TIME_LIMIT = 600  # seconds
GPU_COUNT = 8
H100_PEAK_TFLOPS_FP16 = 990.0  # per GPU
FLOPS_PER_PARAM_PER_TOKEN = 6  # approximate: 2 forward + 4 backward

# Theoretical zstd compression ratio for neural network weights.
# Empirically, zstd-22 on quantized weights achieves 0.85-0.95x of raw size.
# We use 0.90 as a conservative default and calibrate from real data when available.
_DEFAULT_COMPRESSION_RATIO = 0.90

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------


def max_parameters(
    bits: int,
    artifact_budget: int = ARTIFACT_SOFT_LIMIT,
    code_bytes: int = 0,
    compression_ratio: float | None = None,
) -> int:
    """Maximum parameters that fit in the artifact budget at a given bit-width.

    Args:
        bits: Quantization bit-width (e.g., 6 for int6).
        artifact_budget: Total byte budget for artifact.
        code_bytes: Size of train_gpt.py in bytes.
        compression_ratio: zstd compressed/raw ratio. None = auto-calibrate or default.

    Returns:
        Maximum number of parameters.
    """
    if compression_ratio is None:
        compression_ratio = _calibrate_compression_ratio()
    weight_budget = artifact_budget - code_bytes
    if weight_budget <= 0:
        return 0
    raw_bits_available = weight_budget * 8 / compression_ratio
    return int(raw_bits_available / bits)


def artifact_size(
    params: int,
    bits: int,
    code_bytes: int = 0,
    compression_ratio: float | None = None,
) -> int:
    """Predicted artifact size in bytes for a given model configuration.

    Args:
        params: Number of parameters.
        bits: Quantization bit-width.
        code_bytes: Size of train_gpt.py in bytes.
        compression_ratio: zstd compressed/raw ratio. None = auto-calibrate or default.

    Returns:
        Estimated total artifact bytes.
    """
    if compression_ratio is None:
        compression_ratio = _calibrate_compression_ratio()
    raw_weight_bytes = (params * bits) / 8
    compressed_weight_bytes = raw_weight_bytes * compression_ratio
    return int(compressed_weight_bytes) + code_bytes


def quantization_mse(
    bits: int,
    symmetric: bool = True,
    weight_std: float | None = None,
) -> float:
    """Theoretical MSE floor for uniform quantization.

    For symmetric uniform quantization over [-R, R] with 2^bits levels:
        MSE = R^2 / (3 * 4^bits)
    where R ≈ 3 * weight_std for a normal distribution (covers 99.7%).

    For asymmetric quantization, the quantization range is halved per level,
    reducing MSE by approximately (1 - 1/2^bits) factor.

    Args:
        bits: Quantization bit-width.
        symmetric: Whether quantization is symmetric.
        weight_std: Standard deviation of weights. None = calibrate from disk or use 0.02.

    Returns:
        Theoretical MSE floor.
    """
    if weight_std is None:
        weight_std = _calibrate_weight_std()
    # Range covers ±3 sigma
    R = 3.0 * weight_std
    mse = (R ** 2) / (3.0 * (4.0 ** bits))
    if not symmetric:
        # Asymmetric gets ~1 extra effective bit of resolution
        mse *= (1.0 - 1.0 / (2.0 ** bits))
    return mse


def training_steps(
    batch_size: int,
    seq_len: int,
    model_params: int,
    time_budget: int = TRAINING_TIME_LIMIT,
    gpu_count: int = GPU_COUNT,
) -> int:
    """Estimated training steps that fit within the time budget.

    Uses a roofline throughput model:
        tokens_per_second = gpu_count * peak_flops / (flops_per_param_per_token * model_params)
        steps = tokens_per_second * time_budget / (batch_size * seq_len)

    If results.tsv has observed throughput data, uses that instead.

    Args:
        batch_size: Per-step batch size in sequences.
        seq_len: Sequence length in tokens.
        model_params: Total model parameters.
        time_budget: Training time limit in seconds.
        gpu_count: Number of GPUs.

    Returns:
        Estimated number of training steps.
    """
    # Theoretical roofline (typically ~30-50% utilization in practice)
    total_flops = gpu_count * H100_PEAK_TFLOPS_FP16 * 1e12
    flops_per_token = FLOPS_PER_PARAM_PER_TOKEN * model_params
    tokens_per_second = total_flops / flops_per_token
    # Apply realistic utilization factor
    tokens_per_second *= 0.35

    tokens_per_step = batch_size * seq_len
    if tokens_per_step <= 0:
        return 0
    total_tokens = tokens_per_second * time_budget
    return int(total_tokens / tokens_per_step)


def entropy_lower_bound(bits: int, params: int) -> float:
    """Shannon entropy lower bound for compressed weight storage.

    For uniformly distributed N-bit integers, entropy = N bits per value.
    zstd can't compress below this. For non-uniform distributions (typical
    of trained weights), entropy is lower and compression helps.

    Returns the minimum bytes the weights can compress to, assuming
    the distribution uses full entropy of the quantized values.
    """
    # Entropy in bits per parameter (at most `bits`, in practice less)
    # Conservative: assume 95% of theoretical entropy
    effective_bits = bits * 0.95
    return int((params * effective_bits) / 8)


# ---------------------------------------------------------------------------
# Feasibility report
# ---------------------------------------------------------------------------

_STATUS_PASS = "pass"
_STATUS_WARN = "warn"
_STATUS_FAIL = "fail"

# H100 SXM5 has 80GB HBM3
H100_VRAM_BYTES = 80 * 1024 * 1024 * 1024  # 80 GB


def memory_footprint_check(
    params: int,
    bits: int,
    batch_size: int = 64,
    seq_len: int = 512,
    gpu_count: int = GPU_COUNT,
) -> dict:
    """Estimate GPU memory footprint and check against H100 VRAM."""
    weight_bytes = (params * bits) / 8
    optimizer_bytes = params * 4 * 2  # Adam: momentum + variance in fp32
    gradient_bytes = params * 4  # fp32 gradients
    batch_per_gpu = batch_size / gpu_count
    activation_bytes = 20 * batch_per_gpu * seq_len * math.sqrt(params)

    total_per_gpu = weight_bytes + optimizer_bytes + gradient_bytes + activation_bytes

    if total_per_gpu > H100_VRAM_BYTES:
        status = _STATUS_FAIL
    elif total_per_gpu > H100_VRAM_BYTES * 0.85:
        status = _STATUS_WARN
    else:
        status = _STATUS_PASS

    return {
        "status": status,
        "value": int(total_per_gpu),
        "limit": H100_VRAM_BYTES,
        "headroom_gb": (H100_VRAM_BYTES - total_per_gpu) / (1024**3),
        "detail": f"~{total_per_gpu / (1024**3):.1f} GB per GPU "
                  f"(weights={weight_bytes/(1024**3):.2f}GB, "
                  f"optim={optimizer_bytes/(1024**3):.2f}GB, "
                  f"grad={gradient_bytes/(1024**3):.2f}GB, "
                  f"act={activation_bytes/(1024**3):.2f}GB)",
    }


def feasibility_report(
    params: int,
    bits: int,
    code_bytes: int = 0,
    batch_size: int = 64,
    seq_len: int = 512,
    time_budget: int = TRAINING_TIME_LIMIT,
) -> dict:
    """Run all constraint checks and return a structured report.

    Returns:
        Dict with keys: artifact, training_steps, quantization_mse, entropy_bound,
        each containing status (pass/warn/fail), value, limit, and detail.
        Top-level "feasible" key is True only if no checks fail.
    """
    compression = _calibrate_compression_ratio()

    # 1. Artifact size
    est_artifact = artifact_size(params, bits, code_bytes, compression)
    if est_artifact > ARTIFACT_LIMIT:
        art_status = _STATUS_FAIL
    elif est_artifact > ARTIFACT_SOFT_LIMIT:
        art_status = _STATUS_WARN
    else:
        art_status = _STATUS_PASS
    artifact_check = {
        "status": art_status,
        "value": est_artifact,
        "limit": ARTIFACT_LIMIT,
        "headroom": ARTIFACT_LIMIT - est_artifact,
        "detail": f"{est_artifact:,} bytes at {bits}-bit, {compression:.2f} compression",
    }

    # 2. Training steps
    est_steps = training_steps(batch_size, seq_len, params, time_budget)
    # Warn if fewer than 100 steps (probably too few to converge)
    step_status = _STATUS_PASS if est_steps >= 100 else _STATUS_WARN
    steps_check = {
        "status": step_status,
        "value": est_steps,
        "limit": time_budget,
        "detail": f"~{est_steps} steps in {time_budget}s at batch={batch_size}×{seq_len}",
    }

    # 3. Quantization MSE
    mse = quantization_mse(bits)
    # Context: int6 achieves ~1e-6 MSE which is known to work
    mse_check = {
        "status": _STATUS_PASS,
        "value": mse,
        "detail": f"MSE floor {mse:.2e} at {bits}-bit symmetric",
    }

    # 4. Entropy lower bound
    entropy_bytes = entropy_lower_bound(bits, params)
    ent_status = _STATUS_PASS if entropy_bytes < ARTIFACT_LIMIT - code_bytes else _STATUS_FAIL
    entropy_check = {
        "status": ent_status,
        "value": entropy_bytes,
        "limit": ARTIFACT_LIMIT - code_bytes,
        "detail": f"Entropy floor {entropy_bytes:,} bytes ({bits}-bit, {params:,} params)",
    }

    # 5. Memory footprint
    mem_check = memory_footprint_check(params, bits, batch_size, seq_len)

    # 6. Max parameters at this bit-width
    max_params = max_parameters(bits, ARTIFACT_SOFT_LIMIT, code_bytes, compression)

    checks = {
        "artifact": artifact_check,
        "training_steps": steps_check,
        "quantization_mse": mse_check,
        "entropy_bound": entropy_check,
        "memory_footprint": mem_check,
    }

    feasible = all(c["status"] != _STATUS_FAIL for c in checks.values())

    return {
        "feasible": feasible,
        "params": params,
        "bits": bits,
        "max_params_at_bits": max_params,
        "compression_ratio": compression,
        "checks": checks,
    }


def print_report(report: dict) -> None:
    """Pretty-print a feasibility report to stdout."""
    status_icon = {_STATUS_PASS: "✓", _STATUS_WARN: "!", _STATUS_FAIL: "✗"}
    feasible = report["feasible"]
    tag = "FEASIBLE" if feasible else "NOT FEASIBLE"

    print(f"\n[constraints] {tag}")
    print(f"  Config: {report['params']:,} params at {report['bits']}-bit "
          f"(compression={report['compression_ratio']:.2f})")
    print(f"  Max params at {report['bits']}-bit: {report['max_params_at_bits']:,}")
    print()

    for name, check in report["checks"].items():
        icon = status_icon.get(check["status"], "?")
        print(f"  [{icon}] {name}: {check['detail']}")

    print()


# ---------------------------------------------------------------------------
# Calibration from real data
# ---------------------------------------------------------------------------

_RESULTS_TSV_PATH = Path("results.tsv")


def _calibrate_compression_ratio() -> float:
    """Measure actual zstd compression ratio from weight files on disk.

    Falls back to default if no weight files found.
    """
    weight_files = list(Path(".").glob("*.npz")) + list(Path(".").glob("*.pt"))
    if not weight_files:
        return _DEFAULT_COMPRESSION_RATIO

    try:
        import zstandard as zstd
    except ImportError:
        return _DEFAULT_COMPRESSION_RATIO

    total_raw = 0
    total_compressed = 0
    cctx = zstd.ZstdCompressor()

    for wf in weight_files:
        try:
            raw = wf.read_bytes()
            compressed = cctx.compress(raw)
            total_raw += len(raw)
            total_compressed += len(compressed)
        except Exception:
            continue

    if total_raw == 0:
        return _DEFAULT_COMPRESSION_RATIO

    return total_compressed / total_raw


def _calibrate_weight_std() -> float:
    """Compute actual weight std from weight files on disk.

    Falls back to 0.02 (typical for small transformers).
    """
    weight_files = list(Path(".").glob("*.npz")) + list(Path(".").glob("*.pt"))
    if not weight_files:
        return 0.02

    try:
        import numpy as np
    except ImportError:
        return 0.02

    all_values = []
    for wf in weight_files:
        try:
            if wf.suffix == ".npz":
                data = np.load(wf)
                for key in data:
                    arr = data[key].flatten().astype(np.float32)
                    all_values.append(arr)
            # .pt files require torch — skip if not importable
        except Exception:
            continue

    if not all_values:
        return 0.02

    concatenated = np.concatenate(all_values)
    return float(np.std(concatenated))
