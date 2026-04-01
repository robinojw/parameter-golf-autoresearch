"""Extract operational run specifications from competition PR descriptions and logs.

When baselining a SOTA submission, we need more than the script — we need
the full operational context: packages, env vars, expected step times,
shard counts, etc. This module extracts that from PR descriptions, README
files, and submission.json files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunSpec:
    """Operational specification for reproducing a competition submission."""

    pr_number: int | None = None
    author: str = ""
    reported_bpb: float | None = None
    reported_steps: int | None = None
    reported_step_avg_ms: float | None = None
    reported_artifact_bytes: int | None = None

    seeds: list[int] = field(default_factory=list)
    gpu_count: int = 8
    gpu_type: str = "H100 SXM"
    wallclock_seconds: float = 600.0

    packages: list[str] = field(default_factory=list)
    quantization: str = ""
    compression: str = ""
    gptq_clip_range: int | None = None
    loader_mode: str = ""
    train_shards: int | None = None
    tokenizer: str = "fineweb_1024_bpe"
    vocab_size: int = 1024

    env_vars: dict[str, str] = field(default_factory=dict)
    techniques: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def extract_run_spec_from_pr(pr_body: str, pr_title: str = "") -> RunSpec:
    """Extract a RunSpec from a GitHub PR description."""
    spec = RunSpec()
    text = f"{pr_title}\n{pr_body}"

    bpb_match = re.search(r"val[_\s]bpb[:\s]*(\d+\.\d+)", text, re.IGNORECASE)
    if bpb_match:
        spec.reported_bpb = float(bpb_match.group(1))

    steps_match = re.search(r"(\d{4,})\s*steps", text)
    if steps_match:
        spec.reported_steps = int(steps_match.group(1))

    step_avg_match = re.search(r"(\d+\.?\d*)\s*ms/step", text)
    if step_avg_match:
        spec.reported_step_avg_ms = float(step_avg_match.group(1))

    artifact_match = re.search(
        r"(\d[\d,]+)\s*bytes.*(?:artifact|size|~\d+\.?\d*\s*MB)", text
    )
    if artifact_match:
        spec.reported_artifact_bytes = int(artifact_match.group(1).replace(",", ""))
    mb_match = re.search(r"~?(\d+\.?\d*)\s*MB", text)
    if mb_match and spec.reported_artifact_bytes is None:
        spec.reported_artifact_bytes = int(float(mb_match.group(1)) * 1_000_000)

    seed_matches = re.findall(r"seed[:\s]*(\d+)", text, re.IGNORECASE)
    if seed_matches:
        spec.seeds = [int(s) for s in seed_matches]

    pr_match = re.search(r"PR\s*#?(\d+)", pr_title)
    if pr_match:
        spec.pr_number = int(pr_match.group(1))

    if re.search(r"FA3|flash.?attn.?3|flash.attention.3", text, re.IGNORECASE):
        spec.packages.append("flash-attn-3 (hopper)")
    if re.search(r"brotli", text, re.IGNORECASE):
        spec.packages.append("brotli")
        spec.compression = "brotli"
    if re.search(r"zstd|zstandard", text, re.IGNORECASE):
        if not spec.compression:
            spec.compression = "zstd"
        spec.packages.append("zstandard")

    if re.search(r"GPTQ|gptq|Full.Hessian", text, re.IGNORECASE):
        if re.search(r"naive|no.GPTQ", text, re.IGNORECASE):
            spec.quantization = "naive_int6"
        else:
            spec.quantization = "gptq"
        clip_match = re.search(r"clip.?range[=:\s]*(\d+)", text, re.IGNORECASE)
        if clip_match:
            spec.gptq_clip_range = int(clip_match.group(1))
    elif re.search(r"int6", text, re.IGNORECASE):
        spec.quantization = "int6"
    elif re.search(r"int5", text, re.IGNORECASE):
        spec.quantization = "int5"

    if re.search(r"coprime", text, re.IGNORECASE):
        spec.loader_mode = "coprime"
    shard_match = re.search(r"(\d+)\s*(?:train\s*)?shards", text, re.IGNORECASE)
    if shard_match:
        spec.train_shards = int(shard_match.group(1))

    technique_patterns = [
        (r"XSA.?all|XSA.?11", "XSA-all"),
        (r"MuonEq|muon.?eq", "MuonEq"),
        (r"NorMuon|nor.?muon", "NorMuon"),
        (r"Turbo.?Muon", "Turbo-Muon"),
        (r"Parallel.?Muon", "Parallel-Muon"),
        (r"EngramLite|engram", "EngramLite"),
        (r"BigramHash|bigram.?hash", "BigramHash"),
        (r"SLOT", "SLOT"),
        (r"EGGROLL|eggroll", "EGGROLL"),
        (r"TTT|test.?time.?train", "TTT"),
        (r"LoRA.?TTT", "LoRA-TTT"),
        (r"P2.?loss|focal.?loss", "P2-loss"),
        (r"ResidLambdas", "ResidLambdas"),
        (r"SmearGate", "SmearGate"),
        (r"LeakyReLU", "LeakyReLU²"),
        (r"OrthoInit", "OrthoInit"),
        (r"SWA|swa", "SWA"),
        (r"EMA|ema", "EMA"),
        (r"Late.?QAT", "Late-QAT"),
    ]
    for pattern, name in technique_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            spec.techniques.append(name)

    return spec


def extract_run_spec_from_submission_json(json_str: str) -> RunSpec:
    """Extract a RunSpec from a submission.json file."""
    data = json.loads(json_str)
    spec = RunSpec()
    spec.author = data.get("author", "")
    spec.reported_bpb = data.get("val_bpb")
    spec.seeds = data.get("seeds", [])

    seed_results = data.get("seed_results", {})
    for seed_data in seed_results.values():
        if "steps" in seed_data:
            spec.reported_steps = seed_data["steps"]
        if "step_avg_ms" in seed_data:
            spec.reported_step_avg_ms = seed_data["step_avg_ms"]
        if "artifact_bytes" in seed_data:
            spec.reported_artifact_bytes = seed_data["artifact_bytes"]
        break

    return spec


def run_spec_to_dict(spec: RunSpec) -> dict[str, Any]:
    """Serialize RunSpec for storage in research_results.jsonl."""
    return {
        "pr_number": spec.pr_number,
        "author": spec.author,
        "reported_bpb": spec.reported_bpb,
        "reported_steps": spec.reported_steps,
        "reported_step_avg_ms": spec.reported_step_avg_ms,
        "reported_artifact_bytes": spec.reported_artifact_bytes,
        "seeds": spec.seeds,
        "gpu_count": spec.gpu_count,
        "packages": spec.packages,
        "quantization": spec.quantization,
        "compression": spec.compression,
        "gptq_clip_range": spec.gptq_clip_range,
        "loader_mode": spec.loader_mode,
        "train_shards": spec.train_shards,
        "techniques": spec.techniques,
        "notes": spec.notes,
    }


def format_run_spec_for_agent(spec: RunSpec) -> str:
    """Format a RunSpec as human-readable text for the experiment agent."""
    lines = []
    if spec.pr_number:
        lines.append(f"PR #{spec.pr_number} ({spec.author})")
    if spec.reported_bpb:
        lines.append(f"  Reported val_bpb: {spec.reported_bpb}")
    if spec.reported_steps and spec.reported_step_avg_ms:
        lines.append(
            f"  Steps: {spec.reported_steps} @ {spec.reported_step_avg_ms}ms/step"
        )
    if spec.reported_artifact_bytes:
        lines.append(
            f"  Artifact: {spec.reported_artifact_bytes:,} bytes "
            f"({spec.reported_artifact_bytes / 1_000_000:.1f} MB)"
        )
    if spec.seeds:
        lines.append(f"  Seeds tested: {spec.seeds}")
    if spec.packages:
        lines.append(f"  Required packages: {', '.join(spec.packages)}")
    if spec.quantization:
        quant_info = spec.quantization
        if spec.gptq_clip_range:
            quant_info += f" (clip_range={spec.gptq_clip_range})"
        lines.append(f"  Quantization: {quant_info}")
    if spec.compression:
        lines.append(f"  Compression: {spec.compression}")
    if spec.loader_mode:
        lines.append(f"  Loader: {spec.loader_mode}")
    if spec.train_shards:
        lines.append(f"  Train shards: {spec.train_shards}")
    if spec.techniques:
        lines.append(f"  Techniques: {', '.join(spec.techniques)}")
    if spec.notes:
        for note in spec.notes:
            lines.append(f"  Note: {note}")

    lines.append("")
    lines.append("  POST-FLIGHT VALIDATION (check these after your run):")
    if spec.reported_step_avg_ms:
        lines.append(
            f"    step_avg_ms should be ≤ {spec.reported_step_avg_ms + 5:.0f}ms"
        )
    if spec.reported_steps:
        lines.append(f"    total_steps should be ≥ {int(spec.reported_steps * 0.95)}")
    if spec.reported_bpb:
        lines.append(f"    val_bpb should be ≤ {spec.reported_bpb + 0.01:.4f}")

    return "\n".join(lines)


def fetch_and_extract_pr_spec(
    pr_number: int, repo: str = "openai/parameter-golf"
) -> RunSpec:
    """Fetch a PR from GitHub and extract its RunSpec. Requires GITHUB_TOKEN."""
    import os

    import requests

    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    owner, repo_name = repo.split("/")
    resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    pr_data = resp.json()

    title = pr_data.get("title", "")
    body = pr_data.get("body", "") or ""
    author = (pr_data.get("user") or {}).get("login", "")

    spec = extract_run_spec_from_pr(body, title)
    spec.pr_number = pr_number
    spec.author = author

    files_resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}/files",
        headers=headers,
        params={"per_page": 100},
        timeout=30,
    )
    if files_resp.status_code == 200:
        for f in files_resp.json():
            filename = f.get("filename", "")
            if filename.endswith("submission.json"):
                raw_url = f.get("raw_url", "")
                if raw_url:
                    try:
                        content = requests.get(
                            raw_url, headers=headers, timeout=10
                        ).text
                        sub_spec = extract_run_spec_from_submission_json(content)
                        if sub_spec.reported_bpb and not spec.reported_bpb:
                            spec.reported_bpb = sub_spec.reported_bpb
                        if sub_spec.reported_steps and not spec.reported_steps:
                            spec.reported_steps = sub_spec.reported_steps
                        if (
                            sub_spec.reported_step_avg_ms
                            and not spec.reported_step_avg_ms
                        ):
                            spec.reported_step_avg_ms = sub_spec.reported_step_avg_ms
                        if (
                            sub_spec.reported_artifact_bytes
                            and not spec.reported_artifact_bytes
                        ):
                            spec.reported_artifact_bytes = (
                                sub_spec.reported_artifact_bytes
                            )
                        if sub_spec.seeds and not spec.seeds:
                            spec.seeds = sub_spec.seeds
                        if sub_spec.author and not spec.author:
                            spec.author = sub_spec.author
                    except Exception:
                        pass

            if filename.endswith("README.md") and "records/" in filename:
                raw_url = f.get("raw_url", "")
                if raw_url:
                    try:
                        readme = requests.get(raw_url, headers=headers, timeout=10).text
                        readme_spec = extract_run_spec_from_pr(readme)
                        if readme_spec.reported_steps and not spec.reported_steps:
                            spec.reported_steps = readme_spec.reported_steps
                        if (
                            readme_spec.reported_step_avg_ms
                            and not spec.reported_step_avg_ms
                        ):
                            spec.reported_step_avg_ms = readme_spec.reported_step_avg_ms
                        spec.packages = list(set(spec.packages + readme_spec.packages))
                        spec.techniques = list(
                            set(spec.techniques + readme_spec.techniques)
                        )
                    except Exception:
                        pass

    return spec
