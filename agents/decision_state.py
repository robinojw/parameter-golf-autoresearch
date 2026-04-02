# agents/decision_state.py
"""Generate a compact decision-state handoff artifact for each agent cycle.

Instead of loading the full program.md (400+ lines, ~14K tokens) on every
cycle, this generates a focused summary (~2-4K tokens) containing only
the information needed for the next experiment decision:

  - Current best H100 val_bpb
  - Last 5 experiments with outcomes
  - Top 3 unacked research findings to act on
  - Active blockers/failures
  - Dead ends to avoid
  - Budget status
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.shared import (
    RESEARCH_ACK_PATH,
    RESEARCH_RESULTS_PATH,
    get_unacked_results,
)
from research.experiments import (
    get_current_best_bpb,
    get_failed_experiments,
    get_proven_techniques,
)


_DECISION_STATE_PATH = Path("decision_state.md")
_BUDGET_PATH = Path("budget.json")
_TECHNIQUE_MAP_PATH = Path("technique_map.json")
_MAX_RECENT_EXPERIMENTS = 5
_MAX_UNACKED_FINDINGS = 5
_MAX_DEAD_ENDS = 10


def _read_budget_summary() -> str:
    if not _BUDGET_PATH.exists():
        return "Budget file not found."
    try:
        data = json.loads(_BUDGET_PATH.read_text())
        spent = data.get("spent", 0)
        total = data.get("total_credits", 0)
        reserve = data.get("min_reserve", 0)
        best_bpb = data.get("best_h100_bpb")
        remaining = total - spent
        available = remaining - reserve
        lines = [
            f"Spent: ${spent:.2f} / ${total:.2f} total",
            f"Remaining: ${remaining:.2f} (${available:.2f} available after ${reserve:.2f} reserve)",
        ]
        if best_bpb is not None:
            lines.append(f"Best H100 BPB tracked: {best_bpb}")
        return "\n".join(lines)
    except (json.JSONDecodeError, OSError):
        return "Budget file unreadable."


def _read_dead_ends() -> list[str]:
    if not _TECHNIQUE_MAP_PATH.exists():
        return []
    try:
        data = json.loads(_TECHNIQUE_MAP_PATH.read_text())
        nodes = data.get("nodes", {})
        dead = []
        for name, info in nodes.items():
            if info.get("status") == "dead_end":
                bpb = info.get("best_bpb")
                bpb_str = f" (bpb {bpb})" if bpb else ""
                dead.append(f"- {name}{bpb_str}")
        return dead[:_MAX_DEAD_ENDS]
    except (json.JSONDecodeError, OSError):
        return []


def _read_recent_experiments(limit: int = _MAX_RECENT_EXPERIMENTS) -> str:
    from research.experiments import _read_rows

    rows = _read_rows()
    recent = rows[-limit:] if rows else []
    if not recent:
        return "[No experiments recorded yet]"
    lines = []
    for r in recent:
        lines.append(
            f"- [{r.tier}] {r.description} — val_bpb={r.val_bpb:.4f}, "
            f"status={r.status}, cost=${r.cost_usd:.2f}"
        )
    return "\n".join(lines)


def generate_decision_state() -> str:
    """Generate a compact decision-state markdown for the next agent cycle."""
    best_bpb = get_current_best_bpb()
    proven = get_proven_techniques()

    # Get unacked research findings
    unacked = get_unacked_results()
    # Sort by priority (high first) then by timestamp (newest first)
    priority_order = {"high": 0, "normal": 1, "low": 2}
    unacked.sort(key=lambda m: (priority_order.get(m.priority, 1), m.timestamp))
    top_findings = unacked[:_MAX_UNACKED_FINDINGS]

    # Get recent experiments
    recent_experiments = _read_recent_experiments()

    # Get failed experiments (last 5)
    failed = get_failed_experiments()
    recent_failed = failed[-5:] if failed else []

    # Dead ends from technique map
    dead_ends = _read_dead_ends()

    # Budget
    budget = _read_budget_summary()

    sections = [
        f"# Decision State",
        f"",
        f"## Current Best: {best_bpb} bpb",
        f"",
        f"## Budget",
        budget,
        f"",
        f"## Last {_MAX_RECENT_EXPERIMENTS} Experiments",
        recent_experiments,
        f"",
    ]

    if top_findings:
        sections.append(f"## Unacked Research Findings ({len(unacked)} total, showing top {len(top_findings)})")
        for msg in top_findings:
            sections.append(f"- [{msg.priority}] ({msg.timestamp}) {msg.message[:200]}")
        sections.append("")

    if recent_failed:
        sections.append("## Recent Failures (avoid re-attempting)")
        for f_exp in recent_failed:
            sections.append(f"- {f_exp['description']} (val_bpb={f_exp['val_bpb']}, tier={f_exp['tier']})")
        sections.append("")

    if dead_ends:
        sections.append("## Dead Ends (do NOT pursue)")
        sections.extend(dead_ends)
        sections.append("")

    if proven:
        sections.append(f"## Proven Techniques ({len(proven)} total)")
        sections.append(", ".join(proven))
        sections.append("")

    # Learned rules from hypothesis tracking
    from agents.hypotheses import get_learned_rules
    rules = get_learned_rules()
    if rules:
        sections.append(f"## Learned Rules ({len(rules)} from past hypotheses)")
        for rule in rules[-5:]:  # Show last 5
            sections.append(f"- {rule}")
        sections.append("")

    return "\n".join(sections)


def write_decision_state(output_path: Path = _DECISION_STATE_PATH) -> Path:
    """Generate and write the decision state to disk. Returns the path."""
    content = generate_decision_state()
    output_path.write_text(content)
    return output_path
