import json
import re
from pathlib import Path
from typing import Union

from research.experiments import (
    get_competitor_scores,
    get_current_best_bpb,
    get_experiment_history_bullets,
    get_tier_correlation,
)
from research.reflect import STRATEGY_PATH, _read_strategy_md

TECHNIQUE_MAP_PATH = Path("technique_map.json")


def inject_into_program_md(
    graded_cache_path: str = "graded_cache.jsonl",
    program_md_path: str = "program.md",
    top_n: int = 12,
    max_token_budget: int = 3000,
) -> None:
    graded_items = _load_graded_sorted(graded_cache_path)
    top_items = graded_items[:top_n]

    bullets: list[str] = []
    char_budget = max_token_budget * 4
    chars_used = 0

    for item in top_items:
        flags_str = ", ".join(item.get("flags", []))
        flag_display = f"[{flags_str}] " if flags_str else ""

        bullet = (
            f"- {flag_display}**{item['title']}** — score {item['score']}/15 ({item.get('published_date', 'n/a')})\n"
            f"  {item.get('agent_summary', '')}\n"
            f"  → {item.get('url', '')}"
        )

        if chars_used + len(bullet) > char_budget:
            break

        bullets.append(bullet)
        chars_used += len(bullet)

    replacement_content = "\n".join(bullets)
    replacement = (
        f"<!-- RESEARCH_START -->\n{replacement_content}\n<!-- RESEARCH_END -->"
    )

    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- RESEARCH_START -->.*?<!-- RESEARCH_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )

    program_path.write_text(new_content)

    inject_experiments_section(program_md_path)
    inject_competitors_section(program_md_path)
    inject_verified_section(program_md_path)
    inject_dynamic_baseline(program_md_path)
    inject_strategy_section(program_md_path)
    inject_technique_map_section(program_md_path)


def inject_experiments_section(program_md_path: str = "program.md") -> None:
    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    bullets = get_experiment_history_bullets(limit=8)
    if not bullets:
        section_body = "[No experiments recorded yet]"
    else:
        section_body = bullets
        correlation = get_tier_correlation()
        if correlation["correlation_reliable"]:
            avg_delta = correlation["avg_delta"]
            pairs = correlation["pairs"]
            section_body += (
                f"\n\n_Tier 1↔2 avg delta: {avg_delta:+.4f} bpb ({pairs} paired runs)_"
            )

    replacement = (
        f"<!-- EXPERIMENTS_START -->\n{section_body}\n<!-- EXPERIMENTS_END -->"
    )

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- EXPERIMENTS_START -->.*?<!-- EXPERIMENTS_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    program_path.write_text(new_content)


def inject_competitors_section(program_md_path: str = "program.md") -> None:
    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    competitors = get_competitor_scores()
    if not competitors:
        section_body = "[No competitor data yet — check openai/parameter-golf PRs]"
    else:
        top = competitors[:15]
        lines = ["| PR # | Author | Technique | val_bpb | Δ baseline |"]
        lines.append("|------|--------|-----------|---------|------------|")
        for c in top:
            delta = c["delta_from_baseline"]
            lines.append(
                f"| #{c['pr_number']} | {c['author']} | {c['technique']} "
                f"| {c['val_bpb']:.4f} | {delta:+.4f} |"
            )
        section_body = "\n".join(lines)

    replacement = (
        f"<!-- COMPETITORS_START -->\n{section_body}\n<!-- COMPETITORS_END -->"
    )

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- COMPETITORS_START -->.*?<!-- COMPETITORS_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    program_path.write_text(new_content)


def inject_verified_section(program_md_path: str = "program.md") -> None:
    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    from research.verify import get_verified_items  # lazy import: verify pulls in tavily
    verified = get_verified_items()
    if not verified:
        section_body = (
            "[No verified items yet — Tier A items will be deep-verified automatically]"
        )
    else:
        top = verified[:5]
        lines: list[str] = []
        for v in top:
            score_change = (
                f"{v.get('original_score', 0)}/15 → {v.get('verified_score', 0)}/15"
            )
            brief = v.get("implementation_brief", "")
            sources_count = len(v.get("verification_sources", []))
            lines.append(
                f"- **{v.get('id', '?')}** ({score_change}, {sources_count} sources verified)\n"
                f"  {brief}"
            )
        section_body = "\n".join(lines)

    replacement = f"<!-- VERIFIED_START -->\n{section_body}\n<!-- VERIFIED_END -->"

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- VERIFIED_START -->.*?<!-- VERIFIED_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    program_path.write_text(new_content)


def inject_dynamic_baseline(program_md_path: str = "program.md") -> None:
    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    current_best = get_current_best_bpb()
    content = program_path.read_text()
    new_content = re.sub(
        r"\*\*SOTA: [\d.]+ bpb\. Baseline: 1\.2244 bpb\.\*\*",
        f"**SOTA: {current_best} bpb. Baseline: 1.2244 bpb.**",
        content,
    )
    program_path.write_text(new_content)


def inject_strategy_section(program_md_path: str = "program.md") -> None:
    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    strategy_text = _read_strategy_md(strategy_path=STRATEGY_PATH)
    if not strategy_text:
        section_body = "[No strategy recorded yet — run reflection cycle to populate]"
    else:
        section_body = strategy_text

    replacement = f"<!-- STRATEGY_START -->\n{section_body}\n<!-- STRATEGY_END -->"

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- STRATEGY_START -->.*?<!-- STRATEGY_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    program_path.write_text(new_content)


def render_technique_tree(data: dict) -> str:
    """Render the technique map as an indented tree string.

    Root nodes (not children of anything) are listed first, with children
    indented by 2 spaces. Format: `- [status] name (bpb X.XXXX)`.
    Returns empty string for empty nodes.
    """
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])

    if not nodes:
        return ""

    # Build parent→children lookup
    children_of: dict[str, list[str]] = {}
    all_children: set[str] = set()
    for edge in edges:
        parent = edge.get("parent", "")
        child = edge.get("child", "")
        if parent and child:
            children_of.setdefault(parent, []).append(child)
            all_children.add(child)

    # Root nodes are those not listed as children
    roots = [name for name in nodes if name not in all_children]
    # Also include orphaned children that reference non-existent parents
    orphans = [name for name in nodes if name in all_children and name not in nodes]

    def _format_node(name: str, indent: int) -> list[str]:
        node_data = nodes.get(name, {})
        status = node_data.get("status", "unknown")
        best_bpb = node_data.get("best_bpb")
        bpb_str = f" (bpb {best_bpb})" if best_bpb is not None else ""
        prefix = "  " * indent
        lines = [f"{prefix}- [{status}] {name}{bpb_str}"]
        for child in children_of.get(name, []):
            lines.extend(_format_node(child, indent + 1))
        return lines

    output_lines: list[str] = []
    for root in roots:
        output_lines.extend(_format_node(root, 0))

    return "\n".join(output_lines)


def inject_technique_map_section(program_md_path: str = "program.md") -> None:
    """Read technique_map.json, render as tree, inject into program.md."""
    program_path = Path(program_md_path)
    if not program_path.exists():
        return

    technique_data: dict = {"nodes": {}, "edges": []}
    if TECHNIQUE_MAP_PATH.exists():
        try:
            with open(TECHNIQUE_MAP_PATH) as fh:
                technique_data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    tree_text = render_technique_tree(technique_data)
    if not tree_text:
        section_body = "[No technique map yet — run reflection cycle to populate]"
    else:
        section_body = tree_text

    replacement = (
        f"<!-- TECHNIQUE_MAP_START -->\n{section_body}\n<!-- TECHNIQUE_MAP_END -->"
    )

    content = program_path.read_text()
    new_content = re.sub(
        r"<!-- TECHNIQUE_MAP_START -->.*?<!-- TECHNIQUE_MAP_END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    program_path.write_text(new_content)


def append_to_research_results(
    message: str,
    priority: str = "normal",
    source_experiment: str = "",
    results_path: Union[Path, str] = "research_results.jsonl",
) -> None:
    """Append a research finding to research_results.jsonl.

    Used by the research agent to signal fresh findings to the experiment agent.
    The experiment agent checks this file's timestamps to know if new research
    is available since its last read.
    """
    from agents.shared import Message, append_message

    msg = Message(
        message=message,
        priority=priority,
        source_experiment=source_experiment,
    )
    append_message(Path(results_path), msg)


def _load_graded_sorted(graded_cache_path: str) -> list[dict]:
    path = Path(graded_cache_path)
    if not path.exists():
        return []

    items: list[dict] = []
    raw_cache_path = Path("raw_cache.jsonl")
    raw_map: dict[str, dict] = {}

    if raw_cache_path.exists():
        with open(raw_cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    raw_map[obj["id"]] = obj
                except (json.JSONDecodeError, KeyError):
                    continue

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                graded = json.loads(line)
                raw = raw_map.get(graded.get("id", ""), {})
                graded["title"] = raw.get("title", graded.get("id", "Unknown"))
                graded["url"] = raw.get("url", "")
                graded["published_date"] = raw.get("published_date", "")
                items.append(graded)
            except (json.JSONDecodeError, KeyError):
                continue

    items.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    return items
