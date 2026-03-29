import argparse
import asyncio
import os
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ENV_TOTAL_CREDITS = "TOTAL_COMPUTE_CREDITS"
_ENV_MIN_RESERVE = "RUNPOD_MIN_RESERVE"
_DEFAULT_CREDITS = "500"
_DEFAULT_RESERVE = "50"
_KEY_VAL_BPB = "val_bpb"
_KEY_ARTIFACT_BYTES = "artifact_bytes"
_COMMIT_SHORT_LEN = 7
_LOW_BUDGET_FACTOR = 2
_DEFAULT_REFRESH_HOURS = 6
_DEFAULT_FAST_REFRESH_HOURS = 2
_DEFAULT_SLOW_REFRESH_HOURS = 12
_DEFAULT_TOP_N = 12
_SECONDS_PER_HOUR = 3600
_BREAKING_NEWS_INTERVAL_HOURS = 1
_LOG_KEYS = (_KEY_VAL_BPB, "val_loss", _KEY_ARTIFACT_BYTES, "training_seconds")


def _make_budget_manager():
    from compute.budget import BudgetManager

    return BudgetManager(
        total_credits=float(os.getenv(_ENV_TOTAL_CREDITS, _DEFAULT_CREDITS)),
        min_reserve=float(os.getenv(_ENV_MIN_RESERVE, _DEFAULT_RESERVE)),
    )


async def refresh_research(since_hours: int, top_n: int) -> None:
    """Full refresh: all sources + grade + verify + reflect + inject."""
    from research.fetch import fetch_all
    from research.grade import grade_items, _load_graded_ids
    from research.inject import inject_into_program_md
    from research.verify import run_verification_cycle

    new_items = await fetch_all(since_hours=since_hours)
    already_graded = _load_graded_ids()
    ungraded = [i for i in new_items if i.id not in already_graded]
    if ungraded:
        grade_items(ungraded)
    verified = await run_verification_cycle()
    verified_count = len(verified)
    inject_into_program_md(top_n=top_n)
    from research.reflect import run_reflection_cycle, bootstrap_technique_map

    bootstrap_technique_map()
    reflection = await run_reflection_cycle()
    reflection_msg = "reflected" if reflection else "skipped reflection"
    print(
        f"[research] {len(ungraded)} new items graded, "
        f"{verified_count} verified, {reflection_msg}. program.md updated."
    )


async def refresh_fast_sources(since_hours: int, top_n: int) -> None:
    """Fast refresh: GitHub + Tavily only, then grade + inject. No verify/reflect."""
    from research.fetch import fetch_fast
    from research.grade import grade_items, _load_graded_ids
    from research.inject import inject_into_program_md

    new_items = await fetch_fast(since_hours=since_hours)
    already_graded = _load_graded_ids()
    ungraded = [i for i in new_items if i.id not in already_graded]
    if ungraded:
        grade_items(ungraded)
    inject_into_program_md(top_n=top_n)
    print(f"[research:fast] {len(ungraded)} new items graded. program.md updated.")


def _check_low_budget(remaining: float, min_reserve: float) -> bool:
    if remaining >= min_reserve * _LOW_BUDGET_FACTOR:
        return True
    confirm = input(
        f"[budget] WARNING: only ${remaining:.2f} remaining. Submit? [y/N] "
    )
    return confirm.lower() == "y"


def promote_to_runpod(commit_hash: str, dry_run: bool = False) -> None:
    from compute.runpod_client import RunPodClient
    from compute import sync

    budget = _make_budget_manager()

    allowed, reason = budget.can_submit()
    if not allowed:
        print(f"[budget] BLOCKED: {reason}")
        return

    remaining = budget.status()["remaining"]
    if not _check_low_budget(remaining, budget.min_reserve):
        return

    # Threshold check
    from compute.threshold import compute_promotion_threshold, check_adaptive_fallback
    from research.experiments import get_current_best_bpb, _read_rows

    current_bpb = get_current_best_bpb()
    threshold = compute_promotion_threshold(current_bpb, sota=current_bpb)

    rows = _read_rows()
    candidate_row = None
    for row in reversed(rows):
        if row.commit.startswith(commit_hash[:_COMMIT_SHORT_LEN]):
            candidate_row = row
            break

    if candidate_row and candidate_row.val_bpb > 0:
        candidate_bpb = candidate_row.val_bpb
        passes_threshold = candidate_bpb < current_bpb * threshold

        if not passes_threshold:
            fallback_window = int(os.getenv("PROMOTION_FALLBACK_WINDOW", "10"))
            row_dicts = [
                {"tier": r.tier, "status": r.status, "val_bpb": r.val_bpb}
                for r in rows
            ]
            relaxed = check_adaptive_fallback(
                row_dicts, current_bpb, threshold, window=fallback_window
            )
            if relaxed and candidate_bpb < current_bpb * relaxed:
                print(f"[threshold] Adaptive fallback: accepting {candidate_bpb:.4f} "
                      f"(relaxed threshold {relaxed:.4f})")
            else:
                required_bpb = current_bpb * threshold
                print(f"[threshold] BLOCKED: candidate val_bpb={candidate_bpb:.4f} "
                      f"does not meet threshold {required_bpb:.4f} "
                      f"(current={current_bpb:.4f}, required improvement={1-threshold:.2%})")
                return

    if dry_run:
        print("[dry-run] Would submit to RunPod. No credits spent.")
        return

    run_id = f"runpod_{commit_hash[:_COMMIT_SHORT_LEN]}_{time.strftime('%m%d%H%M')}"
    client = RunPodClient(
        api_key=os.environ["RUNPOD_API_KEY"],
        template_id=os.getenv("RUNPOD_TEMPLATE_ID", "y5cejece4j"),
    )

    pod_id = client.launch_pod()
    try:
        ssh = client.wait_for_ready(pod_id)
        sync.push_to_pod(ssh, ["train_gpt.py", "data/"])
        t0 = time.time()
        exit_code = sync.run_remote_training(ssh, run_id=run_id)
        duration = time.time() - t0
        sync.pull_from_pod(ssh, [f"logs/{run_id}.txt", "run.log"])
    finally:
        client.terminate_pod(pod_id)

    if exit_code != 0:
        print(f"[runpod] Training failed with exit code {exit_code}")
        return

    result = parse_run_log("runpod_results/run.log")
    cost = budget.record_run(run_id, duration)
    append_to_results_tsv(run_id, "runpod", result, cost)
    print(
        f"[runpod] Done. val_bpb={result.get(_KEY_VAL_BPB)} artifact={result.get(_KEY_ARTIFACT_BYTES)} cost=${cost:.2f}"
    )


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


def append_to_results_tsv(
    run_id: str, tier: str, result: dict, cost: float, source_item: str = ""
) -> None:
    path = Path("results.tsv")
    if not path.exists():
        path.write_text(
            "commit\ttier\tval_bpb\tartifact_bytes\tmemory_gb\tstatus\tpromoted\tcost_usd\tdescription\tsource_item\n"
        )
    with open(path, "a") as f:
        val_bpb = result.get(_KEY_VAL_BPB, "")
        artifact_bytes = result.get(_KEY_ARTIFACT_BYTES, "")
        f.write(
            f"{run_id}\t{tier}\t{val_bpb}\t{artifact_bytes}\t\tkeep\tyes\t{cost:.2f}\t\t{source_item}\n"
        )


def print_budget_status() -> None:
    budget = _make_budget_manager()
    status = budget.status()
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value}")


async def _run_breaking_news() -> None:
    from research.sources.tavily_breakingnews import fetch_tavily_breaking
    from research.fetch import _append_to_cache, _load_existing_ids

    try:
        items = await fetch_tavily_breaking()
        existing = _load_existing_ids()
        new_items = [i for i in items if i.id not in existing]
        if new_items:
            _append_to_cache(new_items)
            print(f"[breaking] {len(new_items)} new items cached.")
    except Exception as exc:
        print(f"[breaking] failed: {exc}")


def _breaking_news_loop() -> None:
    while True:
        asyncio.run(_run_breaking_news())
        time.sleep(_BREAKING_NEWS_INTERVAL_HOURS * _SECONDS_PER_HOUR)


_BOOL_TRUE = "true"


def _start_breaking_news_thread() -> None:
    breaking_enabled = (
        os.getenv("TAVILY_HOURLY_BREAKING_NEWS", _BOOL_TRUE).lower() == _BOOL_TRUE
    )
    tavily_key_set = bool(os.environ.get("TAVILY_API_KEY"))
    should_start = breaking_enabled and tavily_key_set
    if not should_start:
        return
    thread = threading.Thread(target=_breaking_news_loop, daemon=True)
    thread.start()
    print("[orchestrate] Breaking news thread started (hourly)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter Golf Autoresearch Orchestrator"
    )
    parser.add_argument("--run-tag", type=str, help="Tag for this run session")
    parser.add_argument("--refresh-hours", type=int, default=_DEFAULT_REFRESH_HOURS)
    parser.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N)
    parser.add_argument("--promote", type=str, metavar="COMMIT_HASH")
    _store_true = "store_true"
    parser.add_argument("--budget-status", action=_store_true)
    parser.add_argument("--dry-run", action=_store_true)
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
    parser.add_argument("--threshold-status", action=_store_true)
    parser.add_argument("--refresh", action=_store_true,
                        help="Run a full research refresh now (all sources)")
    parser.add_argument("--refresh-fast", action=_store_true,
                        help="Run a fast refresh (GitHub + Tavily only)")
    parser.add_argument("--fast-refresh-hours", type=int,
                        default=_DEFAULT_FAST_REFRESH_HOURS)
    parser.add_argument("--slow-refresh-hours", type=int,
                        default=_DEFAULT_SLOW_REFRESH_HOURS)
    parser.add_argument("--check-constraints", action=_store_true)
    parser.add_argument("--params", type=int, default=20_000_000)
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--code-bytes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
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
            append_to_results_tsv(
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
        since_hours = int(os.getenv("SINCE_HOURS", "48"))
        asyncio.run(refresh_research(since_hours, args.top_n))
    elif args.refresh_fast:
        since_hours = int(os.getenv("SINCE_HOURS", "48"))
        asyncio.run(refresh_fast_sources(since_hours, args.top_n))
    else:
        since_hours = int(os.getenv("SINCE_HOURS", "48"))

        _start_breaking_news_thread()

        # Tiered refresh loop:
        # - Fast sources (GitHub, Tavily) every fast_refresh_hours (default 2h)
        # - Full refresh (all sources + verify + reflect) every slow_refresh_hours (default 12h)
        fast_interval = args.fast_refresh_hours * _SECONDS_PER_HOUR
        slow_interval = args.slow_refresh_hours * _SECONDS_PER_HOUR
        last_slow = 0.0  # force full refresh on first iteration

        while True:
            now = time.time()
            time_since_slow = now - last_slow

            if time_since_slow >= slow_interval:
                print("[orchestrate] Running full refresh (all sources)...")
                asyncio.run(refresh_research(since_hours, args.top_n))
                last_slow = time.time()
            else:
                print("[orchestrate] Running fast refresh (GitHub + Tavily)...")
                asyncio.run(refresh_fast_sources(since_hours, args.top_n))

            next_slow_in = slow_interval - (time.time() - last_slow)
            sleep_time = min(fast_interval, next_slow_in)
            print(f"[orchestrate] Next fast refresh in {sleep_time / _SECONDS_PER_HOUR:.1f}h, "
                  f"next full refresh in {next_slow_in / _SECONDS_PER_HOUR:.1f}h")
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
