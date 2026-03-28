"""Critic pre-commit gate — deterministic + LLM checks before committing experiments."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARTIFACT_BLOCK_LIMIT = 15_800_000
_ARTIFACT_WARN_LIMIT = 15_500_000
_DIFF_WARN_LINES = 100
_SIMILARITY_WARN_THRESHOLD = 2  # keyword overlaps to trigger warn
_LLM_DIFF_TRUNCATE = 3000
_MAX_FAILED_IN_PROMPT = 5

_KEYWORDS: list[str] = [
    "int3",
    "int4",
    "int5",
    "int6",
    "int8",
    "qat",
    "quantization",
    "ternary",
    "binary",
    "mamba",
    "state space",
    "ssm",
    "rope",
    "rotary",
    "ema",
    "swa",
    "muon",
    "adam",
    "optimizer",
    "bigram",
    "tokenizer",
    "triton",
    "kernel",
    "smeargate",
    "orthoinit",
    "test-time",
    "ttt",
    "diffusion",
    "jepa",
    "weight sharing",
    "low-rank",
    "lora",
]

_VERDICT_ORDER = {"pass": 0, "warn": 1, "block": 2}

# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def check_artifact_size(artifact_bytes: int) -> dict:
    """Block if > 15,800,000 bytes, warn if > 15,500,000 bytes, pass otherwise."""
    if artifact_bytes > _ARTIFACT_BLOCK_LIMIT:
        return {
            "check": "artifact_size",
            "result": "block",
            "detail": f"artifact_bytes={artifact_bytes} exceeds soft limit {_ARTIFACT_BLOCK_LIMIT}",
        }
    if artifact_bytes > _ARTIFACT_WARN_LIMIT:
        return {
            "check": "artifact_size",
            "result": "warn",
            "detail": f"artifact_bytes={artifact_bytes} close to limit ({_ARTIFACT_WARN_LIMIT}–{_ARTIFACT_BLOCK_LIMIT})",
        }
    return {
        "check": "artifact_size",
        "result": "pass",
        "detail": f"artifact_bytes={artifact_bytes}",
    }


def check_diff_size(lines_changed: int) -> dict:
    """Warn if diff exceeds 100 lines changed."""
    if lines_changed > _DIFF_WARN_LINES:
        return {
            "check": "diff_size",
            "result": "warn",
            "detail": f"lines_changed={lines_changed} exceeds {_DIFF_WARN_LINES}",
        }
    return {
        "check": "diff_size",
        "result": "pass",
        "detail": f"lines_changed={lines_changed}",
    }


def _extract_keywords(text: str) -> set[str]:
    """Return set of known technique keywords found in text (case-insensitive)."""
    lower = text.lower()
    found: set[str] = set()
    for kw in _KEYWORDS:
        # Match whole word or phrase; use word boundary for single-word keywords
        if " " in kw:
            if kw in lower:
                found.add(kw)
        else:
            # Use \b for single-word keywords to avoid partial matches
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                found.add(kw)
    return found


def check_similarity_to_failed(
    diff_summary: str, failed_experiments: list[dict]
) -> dict:
    """Warn if diff_summary shares >= 2 technique keywords with any failed experiment."""
    if not failed_experiments:
        return {
            "check": "similarity_to_failed",
            "result": "pass",
            "detail": "no failed experiments to compare against",
        }

    diff_keywords = _extract_keywords(diff_summary)

    for exp in failed_experiments:
        exp_desc = exp.get("description", "")
        exp_keywords = _extract_keywords(exp_desc)
        overlap = diff_keywords & exp_keywords
        if len(overlap) >= _SIMILARITY_WARN_THRESHOLD:
            return {
                "check": "similarity_to_failed",
                "result": "warn",
                "detail": (
                    f"overlaps with failed experiment '{exp_desc}' "
                    f"on keywords: {sorted(overlap)}"
                ),
            }

    return {
        "check": "similarity_to_failed",
        "result": "pass",
        "detail": f"no significant keyword overlap with {len(failed_experiments)} failed experiments",
    }


def merge_verdicts(checks: list[dict]) -> str:
    """Return the most severe verdict across all checks (pass < warn < block)."""
    worst = "pass"
    for check in checks:
        verdict = check.get("result", "pass")
        if _VERDICT_ORDER.get(verdict, 0) > _VERDICT_ORDER.get(worst, 0):
            worst = verdict
    return worst


# ---------------------------------------------------------------------------
# Git / artifact helpers
# ---------------------------------------------------------------------------


def _get_git_diff() -> str:
    """Run git diff HEAD -- train_gpt_mlx.py and return stdout."""
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", "train_gpt_mlx.py"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def _count_diff_lines(diff_text: str) -> int:
    """Count lines starting with + or - (excluding +++ and --- headers)."""
    count = 0
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def _get_artifact_bytes() -> Optional[int]:
    """Run measure_artifact.py and parse the artifact_bytes output line."""
    try:
        result = subprocess.run(
            ["python", "measure_artifact.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in result.stdout.splitlines():
            if line.startswith("artifact_bytes:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# LLM critic
# ---------------------------------------------------------------------------

_CRITIC_PROMPT_TEMPLATE = """You are a pre-commit critic for a competitive ML experiment.

## CHALLENGE CONSTRAINTS (hard — violations = score 0):
- Artifact: train_gpt.py code bytes + zstd-compressed weights ≤ 16,000,000 bytes
- Training: ≤ 600 seconds on 8×H100 SXM
- No network calls or external downloads during evaluation
- No validation data access during training
- Metric: val_bpb on FineWeb — lower is better

## LAST 5 FAILED EXPERIMENTS:
{failed_section}

## DIFF (truncated to {diff_len} chars):
```
{diff}
```

Evaluate this diff for:
1. Constraint violations (artifact size, training time, no external deps)
2. Similarity to past failures (same technique repackaged)
3. Artifact size impact (will this likely grow the artifact?)

Return ONLY a JSON object with these exact keys:
- "verdict": "pass" | "warn" | "block"
- "reasons": list of strings explaining your verdict
- "similar_to_failed": list of failed experiment descriptions this resembles
- "artifact_impact": "neutral" | "grow" | "shrink"

Do NOT return markdown fences or any text outside the JSON object."""


def _run_llm_critic(diff_text: str, failed: list[dict]) -> Optional[dict]:
    """Call the LLM harness with a critic prompt. Returns parsed JSON or None on failure."""
    try:
        from research.grade import _detect_harness, _run_claude, _run_opencode
    except ImportError:
        return None

    truncated_diff = diff_text[:_LLM_DIFF_TRUNCATE]

    failed_lines = []
    for exp in failed[-_MAX_FAILED_IN_PROMPT:]:
        desc = exp.get("description", "")
        val_bpb = exp.get("val_bpb", "")
        failed_lines.append(f"- {desc} (val_bpb={val_bpb})")
    failed_section = "\n".join(failed_lines) if failed_lines else "(none)"

    prompt = _CRITIC_PROMPT_TEMPLATE.format(
        failed_section=failed_section,
        diff_len=len(truncated_diff),
        diff=truncated_diff,
    )

    try:
        harness = _detect_harness()
        if harness == "opencode":
            raw = _run_opencode(prompt)
        else:
            raw = _run_claude(prompt)
    except Exception as exc:
        print(f"[critic] LLM harness failed: {exc}")
        return None

    # Parse the JSON response
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_critique() -> str:
    """Run all checks (deterministic + LLM), print summary, return overall verdict."""
    from research.experiments import get_failed_experiments

    checks: list[dict] = []

    # 1. Git diff
    diff_text = _get_git_diff()
    lines_changed = _count_diff_lines(diff_text)
    checks.append(check_diff_size(lines_changed))

    # 2. Artifact size
    artifact_bytes = _get_artifact_bytes()
    if artifact_bytes is not None:
        checks.append(check_artifact_size(artifact_bytes))
    else:
        print("[critic] could not measure artifact bytes — skipping size check")

    # 3. Similarity to failed experiments
    failed = get_failed_experiments()
    diff_summary = diff_text[:500]  # use first 500 chars as summary for keyword matching
    checks.append(check_similarity_to_failed(diff_summary, failed))

    # 4. LLM critic (optional — skip gracefully on failure)
    llm_result = _run_llm_critic(diff_text, failed)
    if llm_result is not None:
        llm_verdict = llm_result.get("verdict", "pass")
        llm_reasons = llm_result.get("reasons", [])
        llm_detail = "; ".join(llm_reasons) if llm_reasons else ""
        checks.append({"check": "llm_critic", "result": llm_verdict, "detail": llm_detail})
        similar = llm_result.get("similar_to_failed", [])
        if similar:
            print(f"[critic] LLM flagged similarity to: {similar}")
        artifact_impact = llm_result.get("artifact_impact", "neutral")
        if artifact_impact != "neutral":
            print(f"[critic] LLM artifact impact: {artifact_impact}")

    # Print summary
    print("\n[critic] Pre-commit gate results:")
    for chk in checks:
        icon = {"pass": "✓", "warn": "!", "block": "✗"}.get(chk["result"], "?")
        print(f"  [{icon}] {chk['check']}: {chk['result']} — {chk['detail']}")

    overall = merge_verdicts(checks)
    print(f"\n[critic] Overall verdict: {overall.upper()}")
    return overall
