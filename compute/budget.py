import json
from pathlib import Path
from datetime import datetime, timezone

BUDGET_FILE = Path("budget.json")
_KEY_TOTAL = "total_credits"
_KEY_SPENT = "spent"
_KEY_RESERVE = "min_reserve"
_KEY_RUNS = "runs"
_KEY_STARTED = "started_at"
_KEY_BEST_BPB = "best_h100_bpb"
_DEFAULT_RESERVE = 50.0
_SECONDS_PER_HOUR = 3600
_RESERVE_WARNING_MULTIPLIER = 2
_COST_ROUND_DIGITS = 2
_DEFAULT_GPU_COUNT = 8
_DEFAULT_HOURLY_RATE = 20.0
_BASE_GPU_DIVISOR = 8
_ZERO_COST = 0.0


class BudgetManager:
    def __init__(
        self, total_credits: float, min_reserve: float = _DEFAULT_RESERVE
    ) -> None:
        self.total_credits = total_credits
        self.min_reserve = min_reserve
        self.spent: float = _ZERO_COST
        self.runs: list[dict] = []
        self.best_h100_bpb: float = 0.0
        if BUDGET_FILE.exists():
            self._load()
        else:
            self._save()

    def can_submit(self) -> tuple[bool, str]:
        remaining = self.total_credits - self.spent
        if remaining < self.min_reserve:
            return (
                False,
                f"Blocked: remaining ${remaining:.2f} is below reserve ${self.min_reserve:.2f}",
            )

        if self._is_rate_limited():
            return False, "Rate limited: less than 1 hour since last Tier 2 run"

        warning_threshold = self.min_reserve * _RESERVE_WARNING_MULTIPLIER
        if remaining < warning_threshold:
            return (
                True,
                f"Warning: remaining ${remaining:.2f} is below 2x reserve ${warning_threshold:.2f}",
            )

        return True, "OK"

    def _is_rate_limited(self) -> bool:
        if not self.runs:
            return False
        last_started = self.runs[-1].get(_KEY_STARTED, "")
        if not last_started:
            return False
        try:
            last_time = datetime.fromisoformat(last_started.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
            return elapsed < _SECONDS_PER_HOUR
        except (ValueError, TypeError):
            return False

    def record_run(
        self,
        run_id: str,
        duration_seconds: float,
        gpu_count: int = _DEFAULT_GPU_COUNT,
        hourly_rate: float = _DEFAULT_HOURLY_RATE,
    ) -> float:
        cost = (
            (duration_seconds / _SECONDS_PER_HOUR)
            * gpu_count
            * (hourly_rate / _BASE_GPU_DIVISOR)
        )
        self.spent += cost
        self.runs.append(
            {
                "run_id": run_id,
                _KEY_STARTED: datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration_seconds,
                "cost_usd": round(cost, _COST_ROUND_DIGITS),
                "val_bpb": None,
                "artifact_bytes": None,
                "promoted_from": None,
            }
        )
        self._save()
        print(
            f"Recorded run {run_id}: ${cost:.2f} ({duration_seconds}s, {gpu_count} GPUs)"
        )
        return cost

    def update_best_bpb(self, val_bpb: float) -> None:
        """Update the best known H100 val_bpb if this run is better."""
        if val_bpb > 0 and (self.best_h100_bpb == 0 or val_bpb < self.best_h100_bpb):
            self.best_h100_bpb = val_bpb
            self._save()

    def status(self) -> dict:
        remaining = self.total_credits - self.spent
        avg_cost = self.spent / len(self.runs) if self.runs else _ZERO_COST
        estimated_runs_left = int(remaining / avg_cost) if avg_cost > _ZERO_COST else 0
        return {
            "total": self.total_credits,
            _KEY_SPENT: round(self.spent, _COST_ROUND_DIGITS),
            "remaining": round(remaining, _COST_ROUND_DIGITS),
            "runs_completed": len(self.runs),
            "estimated_runs_left": estimated_runs_left,
        }

    def _save(self) -> None:
        data = {
            _KEY_TOTAL: self.total_credits,
            _KEY_SPENT: round(self.spent, _COST_ROUND_DIGITS),
            _KEY_RESERVE: self.min_reserve,
            _KEY_RUNS: self.runs,
            _KEY_BEST_BPB: self.best_h100_bpb,
        }
        BUDGET_FILE.write_text(json.dumps(data, indent=_COST_ROUND_DIGITS))
        from compute.dashboard import DashboardPusher
        DashboardPusher().push_budget(data)

    def _load(self) -> None:
        data = json.loads(BUDGET_FILE.read_text())
        self.total_credits = data[_KEY_TOTAL]
        self.spent = data[_KEY_SPENT]
        self.min_reserve = data[_KEY_RESERVE]
        self.runs = data[_KEY_RUNS]
        self.best_h100_bpb = data.get(_KEY_BEST_BPB, 0.0)
