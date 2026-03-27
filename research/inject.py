import json
import re
from pathlib import Path

from research.experiments import (
    get_competitor_scores,
    get_current_best_bpb,
    get_experiment_history_bullets,
    get_tier_correlation,
)
from research.verify import get_verified_items


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
