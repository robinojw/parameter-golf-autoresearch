"""Reflection cycle and strategy.md management for parameter-golf autoresearch."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.experiments import BASELINE_PROVEN_TECHNIQUES, _read_rows, get_current_best_bpb
from research.grade import _detect_harness, _run_claude, _run_opencode

MAX_STRATEGY_ENTRIES = 5
STRATEGY_PATH = Path("strategy.md")
TECHNIQUE_MAP_PATH = Path("technique_map.json")

_FRONTMATTER_PATTERN = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_ENTRY_HEADER_PATTERN = re.compile(r"(^## \d{4}-\d{2}-\d{2})", re.MULTILINE)
_FENCE_PATTERN = re.compile(r"^```[a-z]*\n?", re.MULTILINE)

_REQUIRED_KEYS: dict[str, Any] = {
    "failure_patterns": [],
    "exhausted_dimensions": [],
    "promising_dimensions": [],
    "working_hypothesis": "",
    "recommended_next": [],
    "technique_updates": [],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _should_reflect(
    strategy_path: Path = STRATEGY_PATH,
    results_path: Path = Path("results.tsv"),
) -> bool:
    """Return True if there are data rows in results.tsv and reflection is warranted.

    Warranted means either:
    - no strategy file exists yet, or
    - strategy file exists but results have data rows (new experiments since last run)
    """
    rows = _read_rows_from(results_path)
    if not rows:
        return False

    if not strategy_path.exists():
        return True

    # Strategy file exists — reflect if there are any experiment rows
    return True


def _read_rows_from(results_path: Path) -> list:
    """Read rows from a custom path (used in tests with tmp_path)."""
    import csv
    import warnings

    if not results_path.exists():
        return []
    rows = []
    try:
        with open(results_path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for lineno, raw in enumerate(reader, start=2):
                try:
                    from research.experiments import _parse_single_row
                    rows.append(_parse_single_row(raw))
                except (ValueError, TypeError) as exc:
                    warnings.warn(
                        f"results.tsv line {lineno}: skipping malformed row — {exc}",
                        stacklevel=2,
                    )
    except OSError:
        pass
    return rows


def _build_reflection_prompt(
    recent_experiments: list[dict],
    current_sota: float,
    previous_strategy: str,
    technique_map: dict | None,
) -> str:
    """Build a structured prompt for the reflection LLM call."""
    experiments_lines = []
    for exp in recent_experiments:
        desc = exp.get("description", "")
        bpb = exp.get("val_bpb", 0.0)
        status = exp.get("status", "")
        tier = exp.get("tier", "")
        experiments_lines.append(
            f"  - [{tier}] {desc} — val_bpb={bpb}, status={status}"
        )
    experiments_text = "\n".join(experiments_lines) if experiments_lines else "  (none)"

    technique_section = ""
    if technique_map:
        technique_section = (
            "\n## TECHNIQUE MAP (current knowledge graph):\n"
            + json.dumps(technique_map, indent=2)
            + "\n"
        )

    previous_strategy_section = (
        f"\n## PREVIOUS STRATEGY:\n{previous_strategy}\n" if previous_strategy else ""
    )

    return f"""You are a research strategist for a competitive ML challenge (parameter golf).

## CHALLENGE GOAL
Minimize val_bpb (bits per byte) on FineWeb validation set.
Current SOTA: {current_sota} bpb
Constraints: ≤16MB artifact, ≤600s training on 8×H100 SXM.

## RECENT EXPERIMENTS:
{experiments_text}
{previous_strategy_section}{technique_section}
## YOUR TASK
Analyze the recent experiments and produce a JSON reflection with these keys:
- failure_patterns: list of strings describing what has consistently failed
- exhausted_dimensions: list of strings describing search directions that are played out
- promising_dimensions: list of strings describing underexplored directions worth pursuing
- working_hypothesis: single string summarizing the current best theory about what will work
- recommended_next: list of objects with keys: idea, rationale, estimated_impact
- technique_updates: list of objects with keys: node, status (active/dead_end/promising), parent, relation

Return ONLY valid JSON with these exact keys. No markdown fences, no explanation."""


def _parse_reflection_response(text: str) -> dict:
    """Strip markdown fences, parse JSON, fill in defaults for missing keys."""
    cleaned = text.strip()

    # Strip markdown fences like ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner_lines = []
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("```"):
                continue
            if i == len(lines) - 1 and line.strip() == "```":
                continue
            inner_lines.append(line)
        cleaned = "\n".join(inner_lines).strip()

    parsed = json.loads(cleaned)

    # Ensure all required keys have defaults
    result = dict(_REQUIRED_KEYS)
    result.update(parsed)
    return result


def _format_entry(entry: dict) -> str:
    """Format a reflection entry as a markdown section with date header."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_header = now.strftime("%Y-%m-%d %H:%M UTC")

    failure_patterns = entry.get("failure_patterns", [])
    exhausted = entry.get("exhausted_dimensions", [])
    promising = entry.get("promising_dimensions", [])
    hypothesis = entry.get("working_hypothesis", "")
    recommended = entry.get("recommended_next", [])

    lines = [f"## {date_header}"]
    lines.append("")

    if failure_patterns:
        lines.append("**Failure patterns:**")
        for p in failure_patterns:
            lines.append(f"- {p}")
        lines.append("")

    if exhausted:
        lines.append("**Exhausted dimensions:**")
        for d in exhausted:
            lines.append(f"- {d}")
        lines.append("")

    if promising:
        lines.append("**Promising dimensions:**")
        for d in promising:
            lines.append(f"- {d}")
        lines.append("")

    if hypothesis:
        lines.append(f"**Working hypothesis:** {hypothesis}")
        lines.append("")

    if recommended:
        lines.append("**Recommended next:**")
        for r in recommended:
            idea = r.get("idea", "")
            rationale = r.get("rationale", "")
            impact = r.get("estimated_impact", "")
            lines.append(f"- {idea} — {rationale} (est. impact: {impact})")
        lines.append("")

    return "\n".join(lines)


def _write_strategy_md(
    entry: dict,
    strategy_path: Path = STRATEGY_PATH,
) -> None:
    """Write/prepend entry to strategy.md with YAML frontmatter, capped at MAX_STRATEGY_ENTRIES."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    new_section = _format_entry(entry)

    existing_entries: list[str] = []
    if strategy_path.exists():
        content = strategy_path.read_text()
        # Strip frontmatter
        body = _FRONTMATTER_PATTERN.sub("", content).strip()
        if body:
            # Split into individual entries by "## 20" headers
            parts = re.split(r"(?=^## \d{4}-\d{2}-\d{2})", body, flags=re.MULTILINE)
            existing_entries = [p.strip() for p in parts if p.strip()]

    # Prepend new entry and cap
    all_entries = [new_section] + existing_entries
    all_entries = all_entries[:MAX_STRATEGY_ENTRIES]

    frontmatter = f"---\nlast_reflection: {timestamp}\n---\n\n"
    body_text = "\n\n".join(all_entries)
    final_content = frontmatter + body_text + "\n"

    strategy_path.write_text(final_content)
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_doc("strategy", final_content)


def _read_strategy_md(strategy_path: Path = STRATEGY_PATH) -> str:
    """Return the most recent entry text (strips frontmatter), or empty string."""
    if not strategy_path.exists():
        return ""

    content = strategy_path.read_text()
    body = _FRONTMATTER_PATTERN.sub("", content).strip()
    if not body:
        return ""

    # Return the first (most recent) entry
    parts = re.split(r"(?=^## \d{4}-\d{2}-\d{2})", body, flags=re.MULTILINE)
    for part in parts:
        stripped = part.strip()
        if stripped:
            return stripped

    return body


def _run_reflection_prompt(prompt: str) -> str:
    """Call LLM harness and return the response text."""
    harness = _detect_harness()
    if harness == "opencode":
        return _run_opencode(prompt)
    if harness == "claude":
        return _run_claude(prompt)
    raise RuntimeError(f"Unknown harness: {harness}")


def _normalize_technique_key(name: str) -> str:
    """Normalize technique name to snake_case key."""
    key = name.lower()
    key = key.replace(" ", "_").replace("-", "_")
    return key


def bootstrap_technique_map(technique_map_path: Path = TECHNIQUE_MAP_PATH) -> dict:
    """Create technique map from BASELINE_PROVEN_TECHNIQUES if it doesn't exist.

    If the file already exists and is readable, returns existing data as-is.
    """
    if technique_map_path.exists():
        try:
            with open(technique_map_path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    nodes = {}
    for technique in BASELINE_PROVEN_TECHNIQUES:
        key = _normalize_technique_key(technique)
        nodes[key] = {"status": "proven", "best_bpb": None, "experiments": 0}

    data: dict = {"nodes": nodes, "edges": []}
    technique_map_path.write_text(json.dumps(data, indent=2))
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_doc("technique_map", json.dumps(data, indent=2))
    return data


def merge_technique_updates(
    updates: list[dict],
    technique_map_path: Path = TECHNIQUE_MAP_PATH,
) -> dict:
    """Read existing map (or create empty), apply updates, write back.

    Each update dict has keys: node, status, parent (optional), relation (optional).
    Edges are deduped by (parent, child) pair.
    """
    # Load existing or create empty
    data: dict = {"nodes": {}, "edges": []}
    if technique_map_path.exists():
        try:
            with open(technique_map_path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    nodes = data.setdefault("nodes", {})
    edges = data.setdefault("edges", [])

    # Build existing edge set for dedup
    existing_edge_pairs: set[tuple[str, str]] = {
        (e["parent"], e["child"]) for e in edges if "parent" in e and "child" in e
    }

    for update in updates:
        node_key = update.get("node", "")
        if not node_key:
            continue

        status = update.get("status", "exploring")
        parent = update.get("parent")
        relation = update.get("relation")

        if node_key in nodes:
            nodes[node_key]["status"] = status
        else:
            nodes[node_key] = {"status": status, "best_bpb": None, "experiments": 0}

        if parent:
            pair = (parent, node_key)
            if pair not in existing_edge_pairs:
                edges.append({"parent": parent, "child": node_key, "relation": relation})
                existing_edge_pairs.add(pair)

    technique_map_path.write_text(json.dumps(data, indent=2))
    from compute.dashboard import DashboardPusher
    DashboardPusher().push_doc("technique_map", json.dumps(data, indent=2))
    return data


async def run_reflection_cycle(
    strategy_path: Path = STRATEGY_PATH,
    technique_map_path: Path = TECHNIQUE_MAP_PATH,
    results_path: Path = Path("results.tsv"),
) -> dict | None:
    """Orchestrate the full reflection cycle.

    Returns parsed reflection dict if reflection was run, or None if skipped.
    """
    if not _should_reflect(strategy_path=strategy_path, results_path=results_path):
        return None

    rows = _read_rows_from(results_path)
    recent_experiments = [
        {
            "description": r.description,
            "val_bpb": r.val_bpb,
            "status": r.status,
            "tier": r.tier,
        }
        for r in rows[-20:]  # last 20 experiments
    ]

    current_sota = get_current_best_bpb()
    previous_strategy = _read_strategy_md(strategy_path=strategy_path)

    technique_map: dict | None = None
    if technique_map_path.exists():
        try:
            with open(technique_map_path) as fh:
                technique_map = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    prompt = _build_reflection_prompt(
        recent_experiments=recent_experiments,
        current_sota=current_sota,
        previous_strategy=previous_strategy,
        technique_map=technique_map,
    )

    response_text = _run_reflection_prompt(prompt)
    parsed = _parse_reflection_response(response_text)
    _write_strategy_md(parsed, strategy_path=strategy_path)

    technique_updates = parsed.get("technique_updates", [])
    if technique_updates:
        merge_technique_updates(technique_updates, technique_map_path=technique_map_path)

    return parsed
