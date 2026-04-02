"""Tournament hypothesis testing — Tier 1 local MLX tournament runner."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_TOURNAMENT_RUNS_DIR = Path("tournament_runs")
_TOURNAMENT_RESULTS_DIR = Path("tournament_results")

_METRIC_KEYS = ("val_bpb", "val_loss", "artifact_bytes", "training_seconds")
_METRIC_PATTERN = re.compile(r"^(\w+):\s+([\d.]+)\s*$")

_TRAIN_TIMEOUT = 600
_TRAIN_BATCH_TOKENS = "8192"
_VAL_LOSS_EVERY = "0"
_TRAIN_SEQ_LEN = "512"
_MLX_EAGER_EVAL = "1"


@dataclass
class TournamentConfig:
    """Configuration for a tournament run."""

    candidates: int = 4
    survivors: int = 2
    elim_iterations: int = 100
    full_iterations: int = 500
    parallel: int = 1
    cooldown: int = 3
    auto_promote: bool = False
    prompt: str = ""


def _apply_diff_to_copy(source_file: Path, dest_dir: Path, diff: str) -> Path:
    """Copy source_file into dest_dir and apply unified diff.

    Falls back to plain copy if patch fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / source_file.name
    shutil.copy2(source_file, dest_file)

    if not diff or not diff.strip():
        return dest_file

    # Write diff to a temp file and apply with patch
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False
    ) as tmp:
        tmp.write(diff)
        patch_file = tmp.name

    try:
        result = subprocess.run(
            ["patch", "-p1", str(dest_file)],
            input=diff,
            capture_output=True,
            text=True,
            cwd=str(dest_dir),
        )
        if result.returncode != 0:
            # Restore original on failure
            shutil.copy2(source_file, dest_file)
    except (subprocess.SubprocessError, FileNotFoundError):
        # patch not available or failed — keep plain copy
        shutil.copy2(source_file, dest_file)
    finally:
        try:
            os.unlink(patch_file)
        except OSError:
            pass

    return dest_file


def _parse_run_log(log_path: Path) -> dict:
    """Parse metrics from a training run log file.

    Looks for lines of the form: ``key:   value``
    Returns a dict with any of val_bpb, val_loss, artifact_bytes, training_seconds.
    """
    if not log_path.exists():
        return {}

    metrics: dict = {}
    with open(log_path, "r") as fh:
        for line in fh:
            match = _METRIC_PATTERN.match(line.strip())
            if match:
                key, value = match.group(1), match.group(2)
                if key in _METRIC_KEYS:
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass
    return metrics


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    """Sort candidates by val_bpb ascending. None values sort last."""
    if not candidates:
        return []

    def _sort_key(c: dict) -> float:
        v = c.get("val_bpb")
        if v is None:
            return float("inf")
        return float(v)

    return sorted(candidates, key=_sort_key)


def _run_single_candidate(
    script_path: Path,
    run_id: str,
    iterations: int,
    log_dir: Path,
) -> dict:
    """Run MLX training as a subprocess and return parsed metrics."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "ITERATIONS": str(iterations),
            "TRAIN_BATCH_TOKENS": _TRAIN_BATCH_TOKENS,
            "VAL_LOSS_EVERY": _VAL_LOSS_EVERY,
            "TRAIN_SEQ_LEN": _TRAIN_SEQ_LEN,
            "MLX_EAGER_EVAL": _MLX_EAGER_EVAL,
        }
    )

    try:
        with open(log_file, "w") as log_fh:
            result = subprocess.run(
                ["python3", str(script_path)],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=_TRAIN_TIMEOUT,
            )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        returncode = -1

    metrics = _parse_run_log(log_file)
    metrics["run_id"] = run_id
    metrics["returncode"] = returncode
    metrics["log_file"] = str(log_file)
    return metrics


def _build_tournament_prompt(config: TournamentConfig, program_md_path: Path) -> str:
    """Build the LLM prompt for generating tournament candidates."""
    program_context = ""
    if program_md_path.exists():
        program_context = program_md_path.read_text()[:6000]

    n = config.candidates
    user_prompt = config.prompt or "Generate diverse hypothesis variants to improve val_bpb."

    return f"""You are an autonomous ML research agent competing in the Parameter Golf challenge.

## CHALLENGE CONSTRAINTS (hard — violations = disqualified)
- Artifact: train_gpt.py code bytes + zstd-compressed weights ≤ 16,000,000 bytes
- Training: ≤ 600 seconds on 8×H100 SXM
- No network calls or external downloads during evaluation
- No validation data access during training
- Metric: val_bpb on FineWeb — lower is better

## PROGRAM CONTEXT
{program_context}

## YOUR TASK
{user_prompt}

Generate exactly {n} candidate modifications to train_gpt_mlx.py. Each candidate should
represent a distinct hypothesis for improving val_bpb.

Return a JSON array with exactly {n} objects. Each object must have:
- "name": snake_case identifier (e.g. "reduce_lr_warmup")
- "hypothesis": 1-2 sentence explanation of why this might improve val_bpb
- "diff": unified diff (--- a/train_gpt_mlx.py / +++ b/train_gpt_mlx.py) of the change

Return ONLY the JSON array, no markdown fences, no explanation.
"""


def _generate_candidates(
    config: TournamentConfig, program_md_path: Path
) -> list[dict]:
    """Call LLM harness to generate N candidate modifications."""
    from research.grade import _detect_harness, _run_claude, _run_opencode

    prompt = _build_tournament_prompt(config, program_md_path)

    harness = _detect_harness()
    if harness == "opencode":
        raw = _run_opencode(prompt)
    else:
        raw = _run_claude(prompt)

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates", parsed.get("items", [parsed]))
    if not isinstance(parsed, list):
        parsed = [parsed]

    return parsed[: config.candidates]


def _print_results_table(label: str, ranked: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'#':<4} {'Name':<30} {'val_bpb':<12} {'Status'}")
    print(f"  {'-'*52}")
    for i, c in enumerate(ranked):
        bpb = c.get("val_bpb")
        bpb_str = f"{bpb:.4f}" if bpb is not None else "N/A"
        status = "WINNER" if i == 0 else ""
        print(f"  {i+1:<4} {c.get('name','?'):<30} {bpb_str:<12} {status}")
    print(f"{'='*60}\n")


def run_tournament(config: TournamentConfig, source_file: Path) -> dict:
    """Run a full tournament: generate candidates, elimination, final round.

    Returns a dict with winner, elimination results, final results, all_results.
    """
    timestamp = int(time.time())
    run_dir = _TOURNAMENT_RUNS_DIR / f"tournament_{timestamp}"
    results_dir = _TOURNAMENT_RESULTS_DIR / f"tournament_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    program_md_path = Path("program.md")

    # --- Generate candidates ---
    print(f"[tournament] Generating {config.candidates} candidates...")
    try:
        raw_candidates = _generate_candidates(config, program_md_path)
    except Exception as exc:
        print(f"[tournament] Failed to generate candidates: {exc}")
        raw_candidates = []

    # Prepare candidate scripts
    candidates = []
    for i, cand in enumerate(raw_candidates):
        name = cand.get("name", f"candidate_{i}")
        diff = cand.get("diff", "")
        cand_dir = run_dir / name
        try:
            script = _apply_diff_to_copy(source_file, cand_dir, diff)
        except Exception as exc:
            print(f"[tournament] Failed to prepare {name}: {exc}")
            continue
        candidates.append(
            {
                "name": name,
                "hypothesis": cand.get("hypothesis", ""),
                "script": script,
                "dir": cand_dir,
            }
        )

    if not candidates:
        print("[tournament] No candidates prepared — aborting.")
        return {"winner": None, "elimination": [], "final": [], "all_results": []}

    # --- Elimination round ---
    print(
        f"[tournament] Elimination round: {len(candidates)} candidates, "
        f"{config.elim_iterations} iterations each"
    )
    elim_results = []
    for idx, cand in enumerate(candidates):
        if idx > 0 and config.cooldown > 0:
            print(f"[tournament] Cooling down {config.cooldown}s...")
            time.sleep(config.cooldown)

        run_id = f"elim_{cand['name']}_{timestamp}"
        log_dir = run_dir / f"elim_{cand['name']}"
        print(f"[tournament]   Running {cand['name']}...")
        metrics = _run_single_candidate(
            cand["script"], run_id, config.elim_iterations, log_dir
        )
        result = {
            "name": cand["name"],
            "hypothesis": cand["hypothesis"],
            "val_bpb": metrics.get("val_bpb"),
            "val_loss": metrics.get("val_loss"),
            "artifact_bytes": metrics.get("artifact_bytes"),
            "training_seconds": metrics.get("training_seconds"),
            "returncode": metrics.get("returncode"),
            "log_file": metrics.get("log_file"),
            "script": str(cand["script"]),
            "round": "elimination",
        }
        elim_results.append(result)

    ranked_elim = _rank_candidates(elim_results)
    _print_results_table("ELIMINATION ROUND", ranked_elim)

    # --- Select survivors ---
    survivors = ranked_elim[: config.survivors]
    print(
        f"[tournament] {len(survivors)} survivors advance to final: "
        + ", ".join(s["name"] for s in survivors)
    )

    # --- Final round ---
    print(
        f"[tournament] Final round: {len(survivors)} survivors, "
        f"{config.full_iterations} iterations each"
    )
    final_results = []
    for idx, survivor in enumerate(survivors):
        if idx > 0 and config.cooldown > 0:
            print(f"[tournament] Cooling down {config.cooldown}s...")
            time.sleep(config.cooldown)

        # Find the original script for this survivor
        orig = next((c for c in candidates if c["name"] == survivor["name"]), None)
        if orig is None:
            continue

        run_id = f"final_{survivor['name']}_{timestamp}"
        log_dir = run_dir / f"final_{survivor['name']}"
        print(f"[tournament]   Running {survivor['name']}...")
        metrics = _run_single_candidate(
            orig["script"], run_id, config.full_iterations, log_dir
        )
        result = {
            "name": survivor["name"],
            "hypothesis": survivor["hypothesis"],
            "val_bpb": metrics.get("val_bpb"),
            "val_loss": metrics.get("val_loss"),
            "artifact_bytes": metrics.get("artifact_bytes"),
            "training_seconds": metrics.get("training_seconds"),
            "returncode": metrics.get("returncode"),
            "log_file": metrics.get("log_file"),
            "script": str(orig["script"]),
            "round": "final",
        }
        final_results.append(result)

    ranked_final = _rank_candidates(final_results)
    _print_results_table("FINAL ROUND", ranked_final)

    winner = ranked_final[0] if ranked_final else (ranked_elim[0] if ranked_elim else None)
    if winner:
        print(f"[tournament] WINNER: {winner['name']} (val_bpb={winner.get('val_bpb')})")

    # --- Archive results ---
    all_results = elim_results + final_results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as fh:
        json.dump(
            {
                "winner": winner,
                "elimination": ranked_elim,
                "final": ranked_final,
                "all_results": all_results,
            },
            fh,
            indent=2,
        )
    print(f"[tournament] Results archived to {results_file}")

    return {
        "winner": winner,
        "elimination": ranked_elim,
        "final": ranked_final,
        "all_results": all_results,
    }
