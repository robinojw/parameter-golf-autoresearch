# agents/hypotheses.py
"""Structured hypothesis tracking for the experiment agent.

Each hypothesis records: what was predicted, why, what happened,
and what was learned. This creates the feedback loop for calibrating
confidence in future local signals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HYPOTHESES_PATH = Path("hypotheses.jsonl")

_SCALE_RISK_LEVELS = ("low", "medium", "high")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Hypothesis:
    """A structured experiment hypothesis with prediction and outcome."""

    technique: str
    prediction: str
    basis: str
    scale_risk: str = "medium"
    outcome: str = ""
    confidence_update: str = ""
    learned_rule: str = ""
    created_at: str = field(default_factory=_now_iso)
    resolved_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def record_hypothesis(h: Hypothesis, path: Path = HYPOTHESES_PATH) -> None:
    """Append a new hypothesis to the tracking file."""
    with open(path, "a") as f:
        f.write(json.dumps(h.to_dict()) + "\n")


def resolve_hypothesis(
    technique: str,
    outcome: str,
    confidence_update: str,
    learned_rule: str = "",
    path: Path = HYPOTHESES_PATH,
) -> None:
    """Find the most recent unresolved hypothesis for *technique* and resolve it.

    If no matching unresolved hypothesis is found, this is a no-op.
    """
    if not path.exists():
        return

    lines: list[str] = []
    resolved = False
    raw_lines = path.read_text().strip().split("\n")

    # Walk backwards to find the most recent match
    for raw in reversed(raw_lines):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            lines.insert(0, raw)
            continue

        if (
            not resolved
            and d.get("technique", "").lower() == technique.lower()
            and not d.get("resolved_at")
        ):
            d["outcome"] = outcome
            d["confidence_update"] = confidence_update
            d["learned_rule"] = learned_rule
            d["resolved_at"] = _now_iso()
            resolved = True

        lines.insert(0, json.dumps(d))

    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")


def get_recent_hypotheses(
    limit: int = 10,
    path: Path = HYPOTHESES_PATH,
) -> list[dict]:
    """Return the most recent hypotheses, newest first."""
    if not path.exists():
        return []

    entries: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return entries[-limit:][::-1]


def get_learned_rules(path: Path = HYPOTHESES_PATH) -> list[str]:
    """Extract all non-empty learned_rule entries from resolved hypotheses."""
    if not path.exists():
        return []

    rules: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rule = d.get("learned_rule", "")
                if rule and d.get("resolved_at"):
                    rules.append(rule)
            except json.JSONDecodeError:
                continue

    return rules
