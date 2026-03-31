"""Parameter Golf Autoresearch — Process Supervisor.

Thin orchestrator that spawns two Claude Code (Opus 4.6) agents as subprocesses:
  1. Experiment agent — designs hypotheses, runs local experiments, requests promotions
  2. Research agent — continuously discovers, grades, and synthesizes research

Also owns RunPod instance lifecycle (create/delete via API, execute via SSH)
and polls promotion_queue.jsonl for Tier 2 promotion requests.

No LLM logic lives here — all intelligence is in the agents.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from compute.dashboard import DashboardPusher

_dashboard = DashboardPusher()


def _print_stream_line(line: str, prefix: str) -> None:
    """Parse a stream-json line and print human-readable output."""
    line = line.strip()
    if not line:
        return
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        print(f"[{prefix}] {line}", flush=True)
        return

    msg_type = obj.get("type", "")

    if msg_type == "assistant":
        # Content blocks are nested in message.content
        msg = obj.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block["text"].strip()
                if text:
                    # Truncate very long text blocks
                    display = text[:500] + ("..." if len(text) > 500 else "")
                    print(f"[{prefix}] {display}", flush=True)
                    _push_activity(prefix, "text", display)
            elif btype == "tool_use":
                tool_name = block.get("name", "?")
                inp = block.get("input", {})
                # Extract the most useful detail from the tool input
                detail = inp.get("command", inp.get("file_path", inp.get("pattern", "")))
                detail = str(detail)[:150]
                print(f"[{prefix}] [{tool_name}] {detail}", flush=True)
                _push_activity(prefix, tool_name, detail)

    elif msg_type == "user":
        # Tool results coming back
        msg = obj.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_text = str(block.get("content", ""))[:200]
                if result_text and "error" in result_text.lower():
                    print(f"[{prefix}] [error] {result_text}", flush=True)

    elif msg_type == "result":
        cost = obj.get("cost_usd", 0)
        duration = obj.get("duration_ms", 0)
        duration_s = duration / 1000 if duration else 0
        print(f"[{prefix}] [done] cost=${cost:.4f} duration={duration_s:.1f}s", flush=True)
        _push_activity(prefix, "done", f"cost=${cost:.4f} duration={duration_s:.1f}s")


def _push_activity(agent: str, action: str, detail: str) -> None:
    """Push an activity event to the dashboard (fire-and-forget)."""
    try:
        _dashboard._post("activity", {
            "agent": agent,
            "action": action,
            "detail": detail[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_TOTAL_CREDITS = "TOTAL_COMPUTE_CREDITS"
_ENV_MIN_RESERVE = "RUNPOD_MIN_RESERVE"
_DEFAULT_CREDITS = "500"
_DEFAULT_RESERVE = "50"
_KEY_VAL_BPB = "val_bpb"
_KEY_ARTIFACT_BYTES = "artifact_bytes"
_COMMIT_SHORT_LEN = 7
_LOG_KEYS = (_KEY_VAL_BPB, "val_loss", _KEY_ARTIFACT_BYTES, "training_seconds")

_EXPERIMENT_AGENT_PROMPT = Path("agents/experiment_agent.md")
_RESEARCH_AGENT_PROMPT = Path("agents/research_agent.md")
_PROMOTION_QUEUE = Path("promotion_queue.jsonl")
_RESULTS_TSV = Path("results.tsv")

_POLL_INTERVAL_SECONDS = 10
_AGENT_HEALTH_CHECK_SECONDS = 10
_MAX_RESTART_ATTEMPTS = 5
_RESTART_BACKOFF_SECONDS = 5
_AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def _make_budget_manager():
    from compute.budget import BudgetManager
    return BudgetManager(
        total_credits=float(os.getenv(_ENV_TOTAL_CREDITS, _DEFAULT_CREDITS)),
        min_reserve=float(os.getenv(_ENV_MIN_RESERVE, _DEFAULT_RESERVE)),
    )


# ---------------------------------------------------------------------------
# Agent process management
# ---------------------------------------------------------------------------


def _launch_agent(prompt_path: Path, name: str) -> subprocess.Popen:
    """Launch a Claude Code instance with the given system prompt."""
    if not prompt_path.exists():
        print(f"[orchestrate] ERROR: {prompt_path} not found")
        sys.exit(1)

    prompt_content = prompt_path.read_text()

    # Ensure agents use the project venv
    venv_dir = Path(__file__).parent / ".venv"
    venv_note = ""
    if venv_dir.exists():
        venv_note = (
            f"\n\nIMPORTANT: Always use the project venv for Python: "
            f"{venv_dir / 'bin' / 'python'}\n"
            f"Run all python commands as: {venv_dir / 'bin' / 'python'} <script>\n"
            f"Or activate first: source {venv_dir / 'bin' / 'activate'}\n"
        )

    cmd = [
        "claude",
        "-p", prompt_content + venv_note,
        "--output-format", "stream-json",
        "--verbose",
        "--model", _AGENT_MODEL,
        "--permission-mode", "bypassPermissions",
    ]

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stdout_log = logs_dir / f"{name}_{ts}_stdout.log"
    stderr_log = logs_dir / f"{name}_{ts}_stderr.log"

    stdout_fh = open(stdout_log, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,  # line-buffered
    )
    proc._log_files = (stdout_fh, stdout_log)  # type: ignore[attr-defined]
    print(f"[orchestrate] Launched {name} (PID {proc.pid})")
    print(f"[orchestrate]   log -> {stdout_log}")

    # Tail the log in a background thread so output appears in this console
    import threading

    def _tail(path: Path, prefix: str):
        """Follow a stream-json log file and print text content with a prefix."""
        with open(path, "r") as f:
            while proc.poll() is None:
                line = f.readline()
                if line:
                    _print_stream_line(line, prefix)
                else:
                    time.sleep(0.1)
            # Drain remaining lines after process exits
            for line in f:
                _print_stream_line(line, prefix)

    t = threading.Thread(target=_tail, args=(stdout_log, name), daemon=True)
    t.start()

    return proc


def _check_agent_alive(proc: subprocess.Popen) -> bool:
    """Check if agent subprocess is still running. Returns (alive, clean_exit)."""
    if proc.poll() is None:
        return True
    rc = proc.returncode
    stdout_fh, log_path = proc._log_files  # type: ignore[attr-defined]
    stdout_fh.close()
    # Store exit code so supervisor can distinguish clean exit from crash
    proc._exit_code = rc  # type: ignore[attr-defined]
    if rc == 0:
        print(f"[orchestrate] Agent completed cycle (log: {log_path})")
    else:
        print(f"[orchestrate] Agent crashed with code {rc} (log: {log_path})")
    return False


def _terminate_agent(proc: subprocess.Popen, name: str, timeout: int = 10) -> None:
    """Gracefully terminate an agent, then force kill if needed."""
    if proc.poll() is not None:
        return
    print(f"[orchestrate] Terminating {name} (PID {proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[orchestrate] Force killing {name} (PID {proc.pid})")
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# RunPod promotion handling
# ---------------------------------------------------------------------------


def _read_pending_promotions() -> list[dict]:
    """Read unprocessed promotion requests from the queue."""
    if not _PROMOTION_QUEUE.exists():
        return []
    requests = []
    with open(_PROMOTION_QUEUE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                requests.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return requests


def _clear_promotion_queue() -> None:
    """Clear the promotion queue after processing."""
    if _PROMOTION_QUEUE.exists():
        _PROMOTION_QUEUE.write_text("")


def _handle_promotion(request: dict) -> None:
    """Process a single promotion request: budget check, threshold, launch RunPod."""
    from compute.runpod_client import RunPodClient
    from compute import sync
    from compute.threshold import compute_promotion_threshold, check_adaptive_fallback
    from research.experiments import get_current_best_bpb

    commit_hash = request.get("source_experiment", "unknown")
    message = request.get("message", "")
    print(f"[orchestrate] Processing promotion: {commit_hash} — {message[:80]}")

    # 1. Budget check
    budget = _make_budget_manager()
    allowed, reason = budget.can_submit()
    if not allowed:
        print(f"[orchestrate] BLOCKED by budget: {reason}")
        return

    # 2. Threshold check
    current_bpb = get_current_best_bpb()
    threshold = compute_promotion_threshold(current_bpb, sota=current_bpb)

    candidate_bpb = request.get("candidate_bpb")
    if candidate_bpb is not None:
        candidate_bpb = float(candidate_bpb)
        required_bpb = current_bpb * threshold
        if candidate_bpb >= required_bpb:
            # Try adaptive fallback
            from research.experiments import _read_rows
            rows = _read_rows()
            fallback_window = int(os.getenv("PROMOTION_FALLBACK_WINDOW", "10"))
            row_dicts = [
                {"tier": r.tier, "status": r.status, "val_bpb": r.val_bpb}
                for r in rows
            ]
            relaxed = check_adaptive_fallback(
                row_dicts, current_bpb, threshold, window=fallback_window
            )
            if relaxed and candidate_bpb < current_bpb * relaxed:
                print(f"[orchestrate] Adaptive fallback: accepting {candidate_bpb:.4f}")
            else:
                print(f"[orchestrate] BLOCKED by threshold: {candidate_bpb:.4f} >= {required_bpb:.4f}")
                return

    # 3. Launch RunPod
    run_id = f"runpod_{commit_hash[:_COMMIT_SHORT_LEN]}_{time.strftime('%m%d%H%M')}"
    client = RunPodClient(
        api_key=os.environ["RUNPOD_API_KEY"],
        template_id=os.getenv("RUNPOD_TEMPLATE_ID", "y5cejece4j"),
    )

    # Save a snapshot of the script being submitted
    run_dir = Path("runpod_results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2("train_gpt.py", run_dir / "train_gpt.py")

    pod_id = client.launch_pod()
    try:
        ssh = client.wait_for_ready(pod_id)
        sync.push_to_pod(ssh, ["train_gpt.py", "data"])
        t0 = time.time()
        exit_code = sync.run_remote_training(ssh, run_id=run_id)
        duration = time.time() - t0
        sync.pull_from_pod(ssh, [f"logs/{run_id}.txt", "run.log"], local_dir=str(run_dir), optional=True)
        if exit_code == 0:
            sync.pull_from_pod(ssh, ["model.zst", "model.bin"], local_dir=str(run_dir), optional=True)
    finally:
        client.terminate_pod(pod_id)

    if exit_code != 0:
        print(f"[orchestrate] Training failed with exit code {exit_code}")
        cost = budget.record_run(run_id, duration)
        _append_result(run_id, "runpod", {}, cost, status="crash")
        return

    result = parse_run_log(str(run_dir / "run.log"))
    cost = budget.record_run(run_id, duration)
    _append_result(run_id, "runpod", result, cost)
    print(f"[orchestrate] Done. val_bpb={result.get(_KEY_VAL_BPB)} cost=${cost:.2f}")


# ---------------------------------------------------------------------------
# Results & logging
# ---------------------------------------------------------------------------


def parse_run_log(log_path: str) -> dict:
    result: dict = {}
    path = Path(log_path)
    if not path.exists():
        return result
    text = path.read_text()
    for line in text.splitlines():
        for key in _LOG_KEYS:
            if not line.strip().startswith(f"{key}:"):
                continue
            try:
                result[key] = float(line.split(":", 1)[1].strip())
            except ValueError:
                continue
    return result


def _append_result(
    run_id: str, tier: str, result: dict, cost: float,
    status: str = "keep", source_item: str = "",
) -> None:
    if not _RESULTS_TSV.exists():
        _RESULTS_TSV.write_text(
            "commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n"
        )
    with open(_RESULTS_TSV, "a") as f:
        val_bpb = result.get(_KEY_VAL_BPB, "")
        artifact_bytes = result.get(_KEY_ARTIFACT_BYTES, "")
        f.write(
            f"{run_id}\t{tier}\t{val_bpb}\t{artifact_bytes}\t\t{status}\tyes\t{cost:.2f}\t\t{source_item}\n"
        )
    _dashboard.push_experiment({
        "id": run_id,
        "tier": tier,
        "val_bpb": result.get(_KEY_VAL_BPB),
        "artifact_bytes": result.get(_KEY_ARTIFACT_BYTES),
        "memory_gb": None,
        "status": status,
        "promoted": tier == "runpod",
        "cost_usd": cost,
        "description": "",
        "source_item": source_item,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })


def print_budget_status() -> None:
    budget = _make_budget_manager()
    status = budget.status()
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Legacy CLI commands (preserved for direct use)
# ---------------------------------------------------------------------------


def promote_to_runpod(commit_hash: str, dry_run: bool = False) -> None:
    """Direct promotion (bypass queue). Kept for manual --promote usage."""
    if dry_run:
        print("[dry-run] Would submit to RunPod. No credits spent.")
        return
    _handle_promotion({
        "source_experiment": commit_hash,
        "message": f"Manual promotion of {commit_hash}",
    })


# ---------------------------------------------------------------------------
# Main supervisor loop
# ---------------------------------------------------------------------------


def _read_sota_bpb() -> float:
    """Read current SOTA bpb from program.md or return 0."""
    try:
        text = Path("program.md").read_text()
        for line in text.splitlines():
            if "sota" in line.lower() and "bpb" in line.lower():
                import re
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    return float(match.group(1))
    except Exception:
        pass
    return 0.0


def _read_pipeline_counts() -> dict:
    """Count lines in each pipeline cache file."""
    def _count_lines(path: str) -> int:
        try:
            return sum(1 for _ in open(path))
        except FileNotFoundError:
            return 0
    return {
        "fetched": _count_lines("raw_cache.jsonl"),
        "graded": _count_lines("graded_cache.jsonl"),
        "verified": _count_lines("verified_cache.jsonl"),
        "injected": _count_lines("research_results.jsonl"),
    }


def _run_supervisor() -> None:
    """Main loop: spawn agents, monitor health, poll promotion queue."""
    experiment_proc = _launch_agent(_EXPERIMENT_AGENT_PROMPT, "experiment-agent")
    research_proc = _launch_agent(_RESEARCH_AGENT_PROMPT, "research-agent")

    experiment_restarts = 0
    research_restarts = 0

    def _cleanup(signum=None, frame=None):
        print("\n[orchestrate] Shutting down...")
        _terminate_agent(experiment_proc, "experiment-agent")
        _terminate_agent(research_proc, "research-agent")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    atexit.register(lambda: _cleanup())

    print("[orchestrate] Supervisor running. Polling promotion queue every "
          f"{_POLL_INTERVAL_SECONDS}s.")

    _heartbeat_counter = 0

    while True:
        # Health check: restart agents that have exited
        if not _check_agent_alive(experiment_proc):
            rc = getattr(experiment_proc, '_exit_code', -1)
            if rc != 0:
                experiment_restarts += 1
            if experiment_restarts > _MAX_RESTART_ATTEMPTS:
                print("[orchestrate] Experiment agent exceeded max crash restarts. Stopping.")
                _cleanup()
            label = "cycle complete" if rc == 0 else f"crashed ({experiment_restarts}/{_MAX_RESTART_ATTEMPTS})"
            print(f"[orchestrate] Restarting experiment-agent ({label})...")
            time.sleep(_RESTART_BACKOFF_SECONDS if rc != 0 else 5)
            experiment_proc = _launch_agent(_EXPERIMENT_AGENT_PROMPT, "experiment-agent")

        if not _check_agent_alive(research_proc):
            rc = getattr(research_proc, '_exit_code', -1)
            if rc != 0:
                research_restarts += 1
            if research_restarts > _MAX_RESTART_ATTEMPTS:
                print("[orchestrate] Research agent exceeded max crash restarts. Stopping.")
                _cleanup()
            label = "cycle complete" if rc == 0 else f"crashed ({research_restarts}/{_MAX_RESTART_ATTEMPTS})"
            print(f"[orchestrate] Restarting research-agent ({label})...")
            time.sleep(_RESTART_BACKOFF_SECONDS if rc != 0 else 5)
            research_proc = _launch_agent(_RESEARCH_AGENT_PROMPT, "research-agent")

        _heartbeat_counter += 1
        if _heartbeat_counter % 10 == 0:
            _dashboard.push_heartbeat(
                statuses=[
                    {
                        "agent": "experiment",
                        "status": "running" if _check_agent_alive(experiment_proc) else "crashed",
                        "last_activity": datetime.now(timezone.utc).isoformat(),
                        "restart_count": experiment_restarts,
                    },
                    {
                        "agent": "research",
                        "status": "running" if _check_agent_alive(research_proc) else "crashed",
                        "last_activity": datetime.now(timezone.utc).isoformat(),
                        "restart_count": research_restarts,
                    },
                ],
                sota_bpb=_read_sota_bpb(),
                pipeline_counts=_read_pipeline_counts(),
            )

        # Poll promotion queue
        promotions = _read_pending_promotions()
        if promotions:
            _clear_promotion_queue()
            for req in promotions:
                try:
                    _handle_promotion(req)
                except Exception as exc:
                    print(f"[orchestrate] Promotion failed: {exc}")

        time.sleep(_POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter Golf Autoresearch — Process Supervisor"
    )
    parser.add_argument("--run-tag", type=str, help="Tag for this run session")
    _store_true = "store_true"

    # Direct commands (no agent needed)
    parser.add_argument("--promote", type=str, metavar="COMMIT_HASH",
                        help="Manually promote a commit to RunPod")
    parser.add_argument("--dry-run", action=_store_true)
    parser.add_argument("--budget-status", action=_store_true)
    parser.add_argument("--threshold-status", action=_store_true)
    parser.add_argument("--check-constraints", action=_store_true)
    parser.add_argument("--critique", action=_store_true)
    parser.add_argument("--tournament", action=_store_true)
    parser.add_argument("--tournament-prompt", type=str, default="")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--survivors", type=int, default=2)
    parser.add_argument("--elim-iterations", type=int, default=100)
    parser.add_argument("--full-iterations", type=int, default=500)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--cooldown", type=int, default=3)
    parser.add_argument("--auto-promote", action=_store_true)
    parser.add_argument("--params", type=int, default=20_000_000)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--code-bytes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)

    # Research commands (still useful for manual runs)
    parser.add_argument("--refresh", action=_store_true,
                        help="Run a full research refresh (all sources)")
    parser.add_argument("--refresh-fast", action=_store_true,
                        help="Run a fast refresh (GitHub + Tavily)")
    parser.add_argument("--refresh-hours", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=12)

    args = parser.parse_args()

    if args.budget_status:
        print_budget_status()
    elif args.promote:
        promote_to_runpod(args.promote, dry_run=args.dry_run)
    elif args.critique:
        from research.critic import run_critique
        run_critique()
    elif args.tournament:
        from compute.tournament import run_tournament, TournamentConfig
        config = TournamentConfig(
            candidates=args.candidates,
            survivors=args.survivors,
            elim_iterations=args.elim_iterations,
            full_iterations=args.full_iterations,
            parallel=args.parallel,
            cooldown=args.cooldown,
            auto_promote=args.auto_promote,
            prompt=args.tournament_prompt,
        )
        result = run_tournament(config)
        winner = result.get("winner")
        if winner and args.auto_promote and winner.get("val_bpb"):
            print("[tournament] Auto-promote: checking threshold...")
            _append_result(
                f"tournament_{winner['name']}",
                "tournament_final",
                {"val_bpb": winner["val_bpb"]},
                0.0,
            )
    elif args.threshold_status:
        from compute.threshold import compute_promotion_threshold
        from research.experiments import get_current_best_bpb
        current = get_current_best_bpb()
        threshold = compute_promotion_threshold(current, sota=current)
        required_bpb = current * threshold
        print(f"  Current best: {current:.4f} bpb")
        print(f"  Threshold ratio: {threshold:.4f} ({1-threshold:.2%} improvement required)")
        print(f"  Candidate must beat: {required_bpb:.4f} bpb")
    elif args.check_constraints:
        from compute.constraints import feasibility_report, print_report
        report = feasibility_report(
            params=args.params,
            bits=args.bits,
            code_bytes=args.code_bytes,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        print_report(report)
    elif args.refresh:
        import asyncio
        since_hours = int(os.getenv("SINCE_HOURS", "48"))

        async def _do_refresh():
            from research.fetch import fetch_all
            from research.grade import grade_items, _load_graded_ids
            from research.inject import inject_into_program_md
            from research.verify import run_verification_cycle
            from research.reflect import run_reflection_cycle, bootstrap_technique_map

            new_items = await fetch_all(since_hours=since_hours)
            already_graded = _load_graded_ids()
            ungraded = [i for i in new_items if i.id not in already_graded]
            if ungraded:
                grade_items(ungraded)
            await run_verification_cycle()
            inject_into_program_md(top_n=args.top_n)
            bootstrap_technique_map()
            await run_reflection_cycle()
            print(f"[refresh] Done. {len(ungraded)} new items graded.")

        asyncio.run(_do_refresh())
    elif args.refresh_fast:
        import asyncio
        since_hours = int(os.getenv("SINCE_HOURS", "48"))

        async def _do_fast():
            from research.fetch import fetch_fast
            from research.grade import grade_items, _load_graded_ids
            from research.inject import inject_into_program_md

            new_items = await fetch_fast(since_hours=since_hours)
            already_graded = _load_graded_ids()
            ungraded = [i for i in new_items if i.id not in already_graded]
            if ungraded:
                grade_items(ungraded)
            inject_into_program_md(top_n=args.top_n)
            print(f"[refresh:fast] Done. {len(ungraded)} new items graded.")

        asyncio.run(_do_fast())
    else:
        # Default: run the supervisor
        _run_supervisor()


if __name__ == "__main__":
    main()
