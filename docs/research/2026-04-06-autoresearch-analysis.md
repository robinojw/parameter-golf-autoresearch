# Autoresearch System Analysis: Architecture, Experiments, Research Grading, and the Case for a Specialized AI Research Harness

**Date:** 2026-04-06
**Scope:** Full analysis of the parameter-golf-autoresearch system — architecture, experiment rationale, research grading, blindspots, improvements implemented, and forward-looking research on agentic patterns and specialized harness design.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [The Two-Tier Compute Model](#2-the-two-tier-compute-model)
3. [The Research Pipeline — How Ideas Are Found and Graded](#3-the-research-pipeline)
4. [Why These Specific Experiments Were Run](#4-why-these-specific-experiments-were-run)
5. [The Technique Map — Knowledge Graph of the Search Space](#5-the-technique-map)
6. [Safety and Cost Control](#6-safety-and-cost-control)
7. [Critical Analysis: Blindspots and Underperformance](#7-critical-analysis-blindspots-and-underperformance)
8. [Improvements Implemented](#8-improvements-implemented)
9. [Autoresearch Patterns and Multi-Agent Evidence](#9-autoresearch-patterns-and-multi-agent-evidence)
10. [Harness Engineering: Why It Dominates Model Selection](#10-harness-engineering)
11. [The Case for a Specialized AI Research Harness](#11-the-case-for-a-specialized-ai-research-harness)
12. [Recommended Architecture for a Research Harness Plugin](#12-recommended-architecture)
13. [Open Questions and Future Directions](#13-open-questions-and-future-directions)
14. [Detailed Harness Design: An AI Research Harness Based on SOTA Evidence](#14-detailed-harness-design)

---

## 1. System Architecture

This repository is an autonomous ML experiment loop for OpenAI's Parameter Golf challenge. The challenge asks you to train the best language model you can under three simultaneous hard constraints:

- **Artifact size**: code + compressed weights must fit in 16 MB
- **Training time**: 600 seconds on 8x H100 SXM GPUs
- **No network access** during evaluation

The metric is bits per byte (val_bpb) on FineWeb. Lower is better.

The system is a three-process supervisor pattern with two independent Claude Code (Opus 4.6) LLM agents:

```
orchestrate.py (Supervisor)
  |
  |-- Experiment Agent (Claude Code, always running)
  |     Designs hypotheses, modifies training scripts,
  |     runs local MLX experiments, requests H100 promotions
  |
  |-- Research Agent (Claude Code, on-demand)
  |     Discovers, grades, verifies, and synthesizes research
  |     from 10 external sources. Spawned by queue signal.
  |
  |-- RunPod Lifecycle (direct API management)
        Creates/monitors/terminates 8xH100 pods
```

**Key design principle**: The supervisor (`orchestrate.py`, ~1080 lines) has zero LLM logic. It spawns agents, monitors health (restarting on crash up to 5 times), polls file queues, and manages pod infrastructure. All intelligence lives in the agent prompts.

**Communication is entirely file-based** (shared JSONL files):

| File | Writer | Reader | Purpose |
|------|--------|--------|---------|
| `program.md` | Research agent | Both | Strategic context, SOTA targets, technique map |
| `results.tsv` | Experiment agent / orchestrator | Both | Full experiment audit trail |
| `research_results.jsonl` | Research agent | Experiment agent | Timestamped findings |
| `research_queue.jsonl` | Experiment agent | Orchestrator (triggers research spawn) | On-demand research requests |
| `research_ack.jsonl` | Experiment agent | Research agent | Acknowledgment of consumed findings |
| `hypotheses.jsonl` | Experiment agent | Decision state generator | Structured predictions and outcomes |
| `promotion_queue.jsonl` | Experiment agent | Orchestrator | H100 run requests |
| `decision_state.md` | Orchestrator | Experiment agent | Compact per-cycle handoff (~2K tokens) |

Both agents run in single-turn `-p` mode — one experiment cycle (or one research cycle), then exit. The orchestrator restarts them in a loop. This prevents context window bloat and makes each cycle independently recoverable.

---

## 2. The Two-Tier Compute Model

The system implements a cost-aware two-tier compute strategy:

**Tier 1 — Local MLX (FREE)**
- Script: `train_gpt_mlx.py` (1118 lines, Apple Silicon)
- 500 iterations, 8192 batch tokens, ~2-3 minutes per run
- Used for: all initial experiments, idea validation, directional signal
- val_bpb from local runs is not the competition score — it's a proxy (r=0.86 correlation with final at step 1000)

**Tier 2 — RunPod 8xH100 (~$3.50-$10/run)**
- Script: `train_gpt.py` (1943 lines, PyTorch/torchrun)
- 6000-7000 steps, 786K batch tokens, 600 seconds wall-clock
- Full pipeline: GPTQ quantization, EMA, TTT, SLOT, EGGROLL, N-gram eval

Promotion from Tier 1 to Tier 2 requires passing 7 independent gates:

1. **Constraint check** — mathematical feasibility (artifact size, training steps, quantization MSE, entropy bound, memory footprint)
2. **Contamination check** — AST-based analysis for validation data leakage + score plausibility ratios
3. **Critic gate** — artifact size, diff size, similarity to past failures, LLM review
4. **Scale-transfer risk** — classifies techniques by likelihood of behaving differently at 500 vs 7000 steps
5. **Dynamic promotion threshold** — requires 0.5%-1.5% improvement (scales with distance from SOTA), with adaptive fallback after 10 consecutive rejections
6. **Budget check** — blocks below $50 reserve floor, with rate limiting (1-hour cooldown)
7. **Early-abort oracle** — monitors step-1000 val_bpb during H100 training; kills underperforming runs before they burn the full budget

---

## 3. The Research Pipeline

The research agent operates a 6-stage pipeline that continuously discovers, evaluates, and injects findings:

### Stage 1: Fetch (10 sources, async)

Sources: ArXiv, OpenReview, Semantic Scholar, GitHub PRs, GitHub code search, RSS feeds, Tavily web search, CodeSOTA. Items below a Tavily relevance score of 0.4 are dropped. Deduplicated against `raw_cache.jsonl`.

### Stage 2: Pre-Filter (regex, deterministic)

Extracts parameter counts and bit-widths from titles/abstracts using regex patterns (e.g., "50M params", "int4", "6-bit"). If both are extractable and `feasibility_report()` says infeasible (won't fit in 16MB or exceeds 600s), the item is auto-rejected to Tier C with zero LLM tokens spent. Items where extraction fails pass through (fail-open design).

### Stage 3: Deduplication

Jaccard keyword similarity (threshold 0.6) is computed against the last 50 research results. Near-duplicates are silently dropped, preventing unbounded repetition of the same technique across research cycles.

### Stage 4: LLM Grading (6 dimensions, /17 base)

Items are graded in batches of 10 by the LLM across these dimensions:

| Dimension | Range | What It Measures |
|-----------|-------|------------------|
| `bpb_impact` | 0-3 | Evidence this reduces bits per byte |
| `size_compatibility` | 0-3 | Fits within 16 MB artifact |
| `time_compatibility` | 0-2 | Within 600s training budget |
| `implementability` | 0-4 | Implementable in <100 lines, no new deps |
| `novelty` | 0-3 | Not already on leaderboard |
| `scale_transfer_risk` | 0-2 | Inverted: 0 = high risk (loss functions, optimizer mods), 2 = low risk (compression, data loading) |

With competitor data available, a 7th dimension `competitor_validated` (0-2) is added, bringing the max to /19.

**Tier classification**:
- **Tier A** (high priority): score >= 14/19 (or >=12/17 without competitor data)
- **Tier B** (moderate): score >= 11/19 (or >=9/17)
- **Tier C** (discard): below threshold

The grading prompt is dynamically injected with: current SOTA bpb, proven techniques list, last 10 failed experiments (to penalise re-attempts), and top 10 competitor scores. This means the grading rubric evolves as the competition progresses.

### Stage 5: Deep Verification (Tier A only)

Items scoring >= 10.0 get full verification (max 5 per cycle):
1. Full content extraction via Tavily Extract API
2. Corroborating evidence search (2 queries per item)
3. LLM re-grading with full context (4000 chars) + evidence (2000 chars)
4. Post-grade feasibility gate (catches cases where LLM scored high but numbers don't fit)

### Stage 6: Reflection Cycle

Synthesises experiment history into strategic guidance:
- Identifies failure patterns (what consistently fails)
- Maps exhausted dimensions (dead ends) vs promising dimensions (underexplored)
- Produces a working hypothesis and recommended next experiments
- Maintains a technique adjacency graph (`technique_map.json`) with statuses: `proven`, `active`, `promising`, `dead_end`, `marginal`
- Output capped at 5 strategy entries in `strategy.md`

---

## 4. Why These Specific Experiments Were Run

The experiment history reveals a clear trajectory driven by the research pipeline's findings:

**Phase 1: Optimizer exploration (local)**
- Tested NS5 Muon vs Turbo-Muon — Turbo-Muon better at 500 steps but confirmed worse at full scale (7000+ steps) per PR #1105
- Tested MuonEq RC equilibration (arxiv:2603.28254) — -0.103 bpb improvement locally, kept
- Tested NorMuon variance reduction — -0.022 locally but marginal on H100 (-0.0002)

**Phase 2: Architecture experiments (local)**
- Tested ResidLambdas — worse at 500 steps (scale-dependent, works at 7000+ steps)
- Tested P2 focal loss (from PR #1180, the unmerged SOTA) — -0.020 bpb locally, promoted

**Phase 3: H100 baseline establishment**
- Multiple crash-recovery cycles establishing infrastructure:
  - Cycle 23: zstandard not installed, zlib fallback, artifact 188KB over limit
  - Cycle 25: SSH timeout exceeded (HF download 327s + GPTQ 38s + EGGROLL 23s)
  - Subsequent runs fixed zstandard, HTTP flow, timeout to 2400s
- First clean H100 baseline: val_bpb=1.1705 (Rascal PR #1120 base), artifact 420KB over limit

**Phase 4: Incremental H100 improvements**
- P2 focal loss on H100 — REGRESSION (+0.067 bpb), confirmed harmful at 5800-step budget
- Brotli-11 compression — saves 4.8 MB, kept
- WARMDOWN_ITERS=4000 — -0.0009 improvement, kept
- LAWA averaging — regression, EMA better
- TTT 3-epoch cosine + SLOT + EGGROLL — NEW BEST val_bpb=1.1563

The research pipeline drove these experiment choices by:
1. Identifying convergent techniques across top entries (XSA-all in 4/7, Full Hessian GPTQ in 4/7, Coprime-stride loader in 3/7)
2. Flagging dead ends (JEPA: 14 negative ablations, Mamba: quantization kills recurrence, depth recurrence: 900x error amplification)
3. Tracking competitor PRs and extracting implementation details
4. Correcting false positives (P2 loss: positive locally, regression on H100 at 5800 steps; Turbo-Muon: positive at 500 steps, negative at 7000+)

---

## 5. The Technique Map

The system maintains a structured knowledge graph (`technique_map.json`) that maps the entire search space:

| Status | Techniques | Count |
|--------|-----------|-------|
| **proven** | BigramHash, zstd-22, EMA, sliding_window_eval, partial_rope, SmearGate, OrthoInit, MuonEq RC, LeakyReLU_sq, ternary_quant, value_residual | 11 |
| **active** | XSA-all (1.1091), EngramLite (1.1091), coprime_loader (1.1099), parallel_muon+AdamW (1.1099) | 4 |
| **promising** | Full Hessian GPTQ (1.1116), legal TTT (1.1154), SLOT (1.1154), ResidLambdas (1.114), param_banking, MTP_auxiliary, PPM7_cache, best_agree_ensemble, EGGROLL_v2, Muon legal TTT | 10 |
| **marginal** | NorMuon VR (1.1597, local -0.022 didn't translate), Brotli (1.1138) | 2 |
| **dead_end** | Turbo-Muon, INT6 QAT, P2 focal loss (REGRESSION at H100 scale), depth recurrence, Mamba SSM (1.5633), JEPA, knowledge distillation (1.1553) | 7 |

This graph, along with parent-child relationships, gives the experiment agent a structured view of which branches to explore and which to avoid.

---

## 6. Safety and Cost Control

The system has defence-in-depth cost safety:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Budget manager | Hard block at $50 reserve, 1-hour rate limiting | Runaway spend |
| atexit + SIGTERM | Terminates all `pgolf-*` pods on exit | Orphaned pods ($0.33/min) |
| Preflight checks | AST parse, import test, model init | Wasting $3-10 on crashes |
| Post-flight checks | FA3 status, shard count, step time, artifact size | Invalid results |
| Promotion threshold | 0.5-1.5% improvement required | Marginal candidates wasting budget |
| One-variable rule | Agent prompt: "never stack untested changes" | Undiagnosable failures |
| Contamination detection | AST-based val data scanning + plausibility ratio | Disqualifying leakage |
| Scale-transfer risk | Flags techniques that behave differently at 500 vs 7000 steps | Misleading local signals |
| Early-abort oracle | Kills H100 runs below threshold at step 1000 | Burning full budget on regressions |

---

## 7. Critical Analysis: Blindspots and Underperformance

### The Numbers

| Metric | Value | Verdict |
|---|---|---|
| Total experiments | 18 (6 local, 12 H100) | Low volume |
| Total H100 spend | $79.83 | Modest |
| H100 crashes | 3 of 12 (25%) | High crash rate |
| H100 regressions | 4 of 9 successes (44%) | Nearly half of successful runs worsened things |
| Runs producing a new best score | 3 of 18 (17%) | Very low hit rate |
| Cost of misleading local signals | $20.88 (26% of spend) | MLX validation is not working |
| Gap to merged SOTA | 0.047 bpb (1.1563 vs 1.1147) | Still far away |
| Research findings produced | 149 entries | Prolific |
| NorMuon duplicate entries | 11 near-identical entries | Research pipeline has no dedup |
| Experiment agent research requests | 1 | Agent barely uses research channel |

### Blindspot 1: MLX Is Being Bypassed, Not Used As a Scratchpad

Only 6 local experiments vs 12 H100 experiments — a 1:2 ratio when it should be 10:1. Two of those 6 local experiments produced actively misleading signals that cost $20.88 in wasted H100 runs. The successful H100 improvements (WARMDOWN=4000, EGGROLL, TTT) were adopted from competitor intelligence, not local MLX validation.

The fundamental problem: 500-step MLX runs have weak correlation with 6000-7000 step H100 runs. The MLX script and PyTorch script have diverged significantly (different NS5 coefficients, different MLP slopes, different skip connection architectures, different embedding approaches) — meaning local MLX results have low predictive value for H100 outcomes.

The system would have been cheaper and faster if it had run more H100 experiments with early-abort (kill at step 1000 if below threshold) rather than trying to use MLX as a filter.

### Blindspot 2: The Research Pipeline Over-Produces and Under-Curates

The research agent is a prolific writer (149 entries) and a poor editor:
- 11 near-identical NorMuon entries because there's no mechanism to mark a reactive request as fulfilled
- 6+ "leaderboard stable, no change" entries that add zero information
- 5 copies of Polar Express coefficients spread across different entries
- Only 50-60 of 149 entries contain truly distinct information

The grading rubric was missing two important real-world signals:
1. **Scale-dependent validity** — Does this technique work at 500 steps AND 7000 steps?
2. **MLX-to-H100 transfer risk** — The grading prompt didn't distinguish between techniques that are architecture-independent (compression, data loading) and techniques that interact with training dynamics (loss functions, optimizer modifications).

### Blindspot 3: No Generator-Evaluator Pattern

The experiment agent was simultaneously generator, executor, and (implicitly) evaluator. The critic gate existed but was voluntary — the experiment agent's mandatory H100 promotion flow skipped it. The result: the experiment agent never validated its hypothesis before spending $3-10 on an H100 run.

### Blindspot 4: Context Engineering Was Poor

Each experiment agent cycle consumed ~14-16K tokens of context. Of this, approximately 8K tokens were irrelevant to any given cycle. The agent read the full competitor table, full technique map, full research context, and full infrastructure docs — whether it needed them or not. There was no conditional context loading based on what the agent was actually doing.

### Blindspot 5: Communication Protocol Was Unidirectional and Lossy

The experiment agent made 1 research request in 18 experiments. The research agent produced 149 findings. This is a 149:1 write:read ratio. The experiment agent read `tail -20 research_results.jsonl` on startup — only the last 13% of research findings. There was no acknowledgment protocol.

### How These Blindspots Compound

```
Research over-produces (149 entries, 11 NorMuon dupes)
  -> Experiment agent can only see tail-20 (87% invisible)
    -> Agent relies on program.md which caps at 12 items
      -> Agent picks technique based on incomplete view
        -> Runs 500-step MLX test (weak predictor of H100)
          -> Gets misleading positive signal (P2 loss, NorMuon VR)
            -> Promotes to H100 without mandatory critic check
              -> Wastes $5-12 discovering technique doesn't work at scale
                -> No structured feedback loop to update beliefs
                  -> Next cycle, agent may try similar technique again
```

Total cost of this cascade: $20.88 in wasted H100 runs (26% of total spend), plus opportunity cost.

---

## 8. Improvements Implemented

Based on the critical analysis, the following improvements were implemented and merged to main:

### Phase 1: Dead Code Removal
- Removed 5 dead files, 3 dead functions, 7 dead constants, 3 unused imports
- Fixed 2 unreachable code paths, removed stale config entries
- ~500 lines of dead code eliminated

### Phase 2: Bug Fixes
- Fixed tournament call signature mismatch (runtime crash on `--tournament`)
- Implemented `_is_rate_limited()` with 1-hour cooldown (was a stub returning False)
- Removed `_calibrate_throughput()` stub

### Phase 3: Mandatory Evaluator Gate
- Wired `run_critique()` as mandatory pre-promotion check — blocks if verdict is "block"
- Integrated `check_data_overlap()` contamination detection into promotion flow
- Added `assess_scale_transfer_risk()` — classifies techniques as high/medium/low risk based on whether they interact with training dynamics

### Phase 4: Early-Abort H100 Runs
- Auto-set `EARLY_ABORT_BPB` env var from budget manager's tracked best H100 bpb (threshold = best * 1.05)
- Dynamic threshold calibration: `update_best_bpb()` tracks best H100 result in `budget.json`
- Early-abort detection in orchestrator marks results appropriately

### Phase 5: Research Pipeline Improvements
- Jaccard keyword deduplication (threshold 0.6) against last 50 entries
- Fulfillment tracking for research queue entries (`read_unfulfilled()` / `mark_fulfilled()`)
- `scale_transfer_risk` as 6th grading dimension (inverted scoring, /15 -> /17 base)
- Acknowledgment protocol (`ack_research_result()` / `get_unacked_results()`)

### Phase 6: Context Engineering
- Decision-state handoff generator: compact ~2K token startup context with best bpb, last 5 experiments, unacked findings, dead ends, budget, learned rules
- Conditional context loading: agent reads `decision_state.md` first, falls back to `program.md` only when deep context needed

### Phase 7: Test Coverage
- 30 new tests (188 total, up from 158)
- Budget manager, deduplication, fulfillment protocol, acknowledgment protocol, decision state generation

### Phase 8: Structural Improvements
- Tournament awareness in experiment agent prompt
- Structured hypothesis tracking with learned rules (`agents/hypotheses.py`)
- CORS placeholder fix in dashboard

---

## 9. Autoresearch Patterns and Multi-Agent Evidence

### The Karpathy Loop

Karpathy's autoresearch demonstrated the minimal viable autoresearch structure: an agent with write access to a single file, a single objectively measurable metric, and a fixed time budget per experiment. The agent reads `program.md` for research directions, modifies `train.py`, runs for 5 minutes, checks `val_bpb`, keeps the commit if better or `git reset` if not, and loops. Over two days, the agent processed ~700 autonomous changes and surfaced ~20 real improvements that manual work had missed — things like attention sharpening via QKnorm scaler multipliers, value-embedding regularization, and AdamW beta tuning. These improvements transferred from a depth-12 model to a depth-24 model, meaning the agent discovered general training recipe improvements, not overfit hacks.

The pattern generalizes: Shopify CEO Tobias Lutke ran an equivalent overnight loop on internal model data and reported 37 experiments with a 19% performance gain.

### The AI Scientist (Sakana AI)

The AI Scientist takes a broader approach: idea generation -> code implementation -> experiment execution -> paper writing -> automated peer review -> iteration. It produces full research papers with figures and citations. The cost is ~$15/paper. The key difference from autoresearch: it optimizes for paper quality (automated reviewer score) rather than a single scalar metric. This makes it more general but less focused — it can explore any research direction but doesn't have the tight feedback loop that makes autoresearch effective at metric optimization.

### Where Multi-Agent Autoresearch Fails

Karpathy tested an 8-agent setup (4 Claude + 4 Codex instances) attempting a "research org" with a chief-scientist model directing juniors. It failed. Agents generated weak hypotheses, ran experiments without strong baselines, skipped ablations, and lacked compute controls. The key insight: agents are strong implementers of well-scoped tasks but currently poor hypothesis generators when given broad authority. The multi-agent org needed human PI oversight for hypothesis generation and experimental rigor.

This matches the parameter-golf-autoresearch design, where the human still authors `program.md` and defines research directions, and the experiment agent only decides which of the pre-specified techniques to try next.

### Google's Multi-Agent Scaling Research

Google Research ran 180 controlled agent configurations across 5 architectures, 3 model families, and 4 benchmarks. Key findings:

| Finding | Magnitude |
|---------|-----------|
| Centralized coordination on parallelizable tasks | +80.8% over single-agent |
| Decentralized coordination on sequential tasks | -39% to -70% |
| Independent agents amplify errors | 17.2x vs single-agent; centralized contains to 4.4x |

The practical rule: if the task is parallelizable and independent, use centralized multi-agent; if the task is sequential with dependencies, use a single agent or tightly supervised orchestration. The parameter-golf-autoresearch architecture correctly applies this — the research agent (parallelizable: fetch, grade, verify multiple papers) and experiment agent (sequential: hypothesis -> implement -> train -> evaluate) are kept independent, coordinating only through file-based messaging.

### Self-Improvement Loop Mechanisms

Several reflective mechanisms produce measurable improvements in agentic behavior:

- **Reflexion** (Shinn et al., 2023): agent writes a natural-language critique of its failed attempt, stores it as context for the next try. On HumanEval, pass@1 improved from GPT-4 baseline levels to ~91%.
- **ERL (Experiential Reflective Learning)**: reflects on task trajectories, generates reusable heuristics, retrieves them at test time. +7.8% over a ReAct baseline on the Gaia2 benchmark.
- **Structured Reflection with External Critic**: reduced mean task completion time by 9.4% (GPT-3.5 teams) and 7.1% (GPT-4 teams). Crucially, reflection alone without a separate critic degrades performance — the critic-reflector synergy is necessary.

The parameter-golf-autoresearch research agent implements a `run_reflection_cycle()` and a `run_critique()` critic gate — architecturally sound in light of this literature.

---

## 10. Harness Engineering: Why It Dominates Model Selection

The evidence for harness dominance has become overwhelming in 2025-2026:

| Comparison | Performance Delta |
|------------|-------------------|
| Same model, different scaffold (SWE-bench) | 42% -> 78% (+36pp) |
| Scaffold vs. model swap on SWE-bench Pro | 22pp (scaffold) vs <1pp (model swap) |
| Weaker Sonnet 4.5 with better scaffold vs Opus 4.5 | Sonnet 52.7% vs Opus 52.0% |
| ForgeCode harness vs Google-shipped Gemini on same model | 78.4% vs 68.5% (+10pp) |
| Meta-Harness vs best hand-designed harness (text classification) | +7.7pp accuracy, 4x fewer tokens |

### The ForgeCode Case Study

ForgeCode's path from 25% to 81.8% on TermBench 2.0 is the most documented case study of systematic harness engineering:

**Phase 1 (25% -> 38%):** Non-interactive mode (eliminated clarification-seeking behavior), tool-call naming aligned with model priors (`old_string`/`new_string` dropped error rates measurably).

**Phase 2 (38% -> 66%):** Enforced `todo_write` — making the planning tool non-optional. Optional tools get deprioritized under pressure; enforcement fixed this.

**Phase 3 (66% -> 78.4%):** Subagent parallelization for low-complexity subtasks + progressive thinking policy (high reasoning budget for messages 1-10 = planning phase, low for messages 11+ = execution phase, back to high for verification).

**Phase 4 (78.4% -> 81.8%):** Model-specific failure mode fixing — field ordering in JSON schemas (`required` before `properties` for GPT 5.4), flattening nested schemas, explicit truncation reminders in tool output text, enforced verification skill that switches the model from builder to reviewer mode.

### Why Claude Code Outperforms OpenCode on Specific Tasks

Claude Code and opencode both wrap frontier models, but they differ in four structural ways:

1. **Co-design advantage**: Claude Code is built by the team that trained Claude. Tool schemas, argument naming, system prompt structures, and context management strategies are developed with knowledge of how Claude's attention mechanisms respond to specific patterns.

2. **Automatic model routing**: Claude Code automatically round-robins between Haiku (fast, cheap, search tasks) and Opus (complex reasoning) without exposing this to the user.

3. **Verification and completeness bias**: Claude Code "writes 73 tests, verifies those specific tests pass, and calls it done without running the full suite" while OpenCode validates the whole system before signing off. Neither is unconditionally superior — Claude Code is faster, OpenCode is more thorough.

4. **Schema and tool tuning**: Tool-call reliability depends on argument naming, schema shape, and description clarity being aligned with how the specific model was trained to interpret those structures. Claude Code's schemas are tuned for Claude; opencode's are generic across providers.

**Important caveat from METR research**: On the time-horizon benchmark (general long-task completion rather than coding-specific tasks), Claude Code beat a simple ReAct loop in only 50.7% of bootstrap samples — statistically a coin flip. This suggests Claude Code's harness advantage is concentrated in coding-specific task types where tight, deterministic feedback loops (tests pass/fail) are present. For research tasks that are more open-ended, the advantage is less clear.

### Anthropic's Harness Design Principles

Two key articles from Anthropic's engineering blog provide foundational principles:

**Generator-Evaluator Separation**: "When asked to evaluate work they've produced, agents tend to respond by confidently praising the work — even when, to a human observer, the quality is obviously mediocre." Their three-agent architecture (planner -> generator -> evaluator) with sprint contracts and file-based communication produced dramatically better results than a solo agent, at 20x the cost ($200 vs $9) but with qualitatively different output quality.

**Structured Handoff Artifacts**: Rather than dumping everything into context, each agent cycle should receive precisely the state needed for the next session. The current approach of "dump everything into program.md and let the agent figure it out" violates this principle. The decision-state handoff we implemented addresses this.

**Sprint Contracts**: The generator and evaluator agree on "what done looks like" before any code is written. This prevents the generator from shipping incomplete work and the evaluator from moving goalposts.

**Context Engineering Over Context Dumping**: Models with 1M token windows degrade severely at 100K tokens, and information buried in the middle causes 30%+ accuracy drops. Proactive context management (structured handoffs, conditional loading) outperforms "give the agent everything."

---

## 11. The Case for a Specialized AI Research Harness

### The Structural Difference Between Coding and Research

| Dimension | Coding Harness | AI Research Harness |
|-----------|---------------|------------------------|
| **Success metric** | Tests pass / feature works (binary) | Scalar metric improves (continuous) |
| **Feedback signal** | Deterministic (compile, test) | Probabilistic (val_bpb improvement, reproducibility) |
| **Context medium** | File system, diffs, test output | ArXiv, GitHub PRs, PDFs, experiment logs |
| **Key tool primitives** | read/write files, run shell, git | semantic search, paper parsing, claim extraction |
| **Error propagation** | Compiler/runtime catches errors quickly | Errors propagate silently through reasoning chains |
| **Verification cost** | Near-zero (run tests) | High (re-implementing a paper is 2-20 GPU hours) |
| **Knowledge accumulation** | Codebase grows | Technique map grows (proven/dead_end/promising) |
| **Principal constraint** | Codebase scope | Cost per validation + artifact/time limits |
| **Progress curve** | Linear (features ship) | Logarithmic (each improvement is harder) |
| **Scale-transfer** | N/A | "Will this work at 7000 steps if it works at 500?" |

### What a Specialized Research Harness Would Add

A general coding harness treats each task independently. An AI research harness must maintain state across experiments — what's been tried, what worked, what failed, what the current frontier is, and what the most promising next direction is.

The specific capabilities a research harness would need beyond a coding harness:

1. **Metric oracle integration**: The harness knows the target metric, can parse training logs, track progress over time, and detect regressions automatically.

2. **Experiment lifecycle management**: Hypothesis -> implementation -> cheap validation -> promotion gate -> expensive validation -> outcome recording -> belief update. This is a first-class workflow, not an afterthought.

3. **Compute tier management**: Automatic routing between cheap (local/CPU/single GPU) and expensive (multi-GPU cluster) compute, with calibrated transfer functions between tiers.

4. **Research memory**: A persistent knowledge graph of techniques, their status (proven/dead_end/promising), parent-child relationships, and the evidence for each status. This survives across agent sessions.

5. **Literature integration**: Continuous monitoring of relevant papers, competitor implementations, and community discussions. Graded and filtered before injection into agent context.

6. **Cost-aware decision making**: Every experiment has a cost. The harness should optimize for information gain per dollar, not just metric improvement.

7. **Enforced scientific method**: Mandatory hypothesis recording before experiments, mandatory outcome recording after, mandatory belief updates when predictions are wrong.

### Why Not Just Use a Coding Harness?

The gap appears in three places:

**Accumulation**: Coding harnesses are designed for sessions with defined end points. Research harnesses must accumulate knowledge across sessions — building a technique map that improves over weeks, not hours.

**Information processing**: The core research loop involves reading, grading, and selectively incorporating findings from a corpus that grows over time. This requires semantic indexing, citation traversal, and claim extraction — none of which are in the standard coding harness tool set.

**Evaluation philosophy**: Coding harnesses optimize for correctness (do tests pass?). Research harnesses must optimize for transferability (does this improvement hold at scale, across architectures, with different data?). The two-tier architecture in parameter-golf-autoresearch is essentially a built-from-scratch answer to this.

---

## 12. Recommended Architecture for a Research Harness Plugin

```
Existing Coding Harness (Claude Code / ForgeCode / etc.)
  |
  +-- Research Plugin Layer
  |     |
  |     +-- Metric Oracle
  |     |     Parses training logs, tracks val_bpb/accuracy/loss
  |     |     Detects regressions, computes improvement significance
  |     |     Maintains best-known-result and improvement threshold
  |     |
  |     +-- Experiment Lifecycle Manager
  |     |     Enforces: hypothesis -> implement -> validate -> record
  |     |     Blocks promotion without recorded hypothesis
  |     |     Auto-records outcomes and updates beliefs
  |     |
  |     +-- Compute Router
  |     |     Cheap tier: local/CPU/single GPU (validation)
  |     |     Expensive tier: cloud GPU cluster (confirmation)
  |     |     Transfer function calibration between tiers
  |     |     Early-abort oracle at configurable checkpoint
  |     |
  |     +-- Research Memory (Knowledge Graph)
  |     |     Techniques: status, evidence, parent-child relationships
  |     |     Experiments: hypothesis, prediction, outcome, cost
  |     |     Learned rules: "loss functions don't transfer from 500 to 7000 steps"
  |     |
  |     +-- Literature Monitor (optional)
  |     |     ArXiv/Semantic Scholar/GitHub monitoring
  |     |     Grading rubric with domain-specific dimensions
  |     |     Deduplication and fulfillment tracking
  |     |
  |     +-- Decision State Generator
  |           Compact handoff artifact for each agent cycle
  |           Conditional context loading based on current phase
  |           Budget summary and constraint status
```

This architecture maps directly onto what we've already built. The difference is making it configurable and reusable rather than hardcoded for Parameter Golf.

### Mapping to the Generalized Plugin

| Component | Parameter Golf Instance | Generalized Plugin |
|---|---|---|
| Metric oracle | val_bpb on FineWeb | Any scalar metric (loss, accuracy, F1, throughput) |
| Constraint checker | 16MB artifact, 600s training | Configurable constraint set (memory, time, FLOPS, cost) |
| Tier 1 compute | MLX on Apple Silicon | Any cheap/free compute (CPU, single GPU, cloud spot) |
| Tier 2 compute | 8xH100 RunPod | Any expensive target hardware |
| Promotion gate | Dynamic threshold + budget + critic | Configurable gate chain |
| Research pipeline | 10-source fetch + LLM grade | Paper/code search with domain-specific grading rubric |
| Technique map | technique_map.json | Knowledge graph of the search space |
| Experiment history | results.tsv | Structured experiment log with outcome tracking |

---

## 13. Open Questions and Future Directions

### High-Impact, Low-Cost Improvements

1. **Experiment vector index**: Embed experiment descriptions + hypotheses into a ChromaDB or FAISS store. Query at hypothesis generation time: "top-3 nearest prior experiments" injected into the experiment agent's context. Estimated implementation: ~100 lines. Expected gain: fewer duplicate dead-end experiments.

2. **Failure-aware verbal critique storage**: After each discard or crash, run a short LLM critique ("why did this fail?") and store it. Inject top-3 failure critiques for semantically similar hypotheses. Directly implements the Reflexion pattern. Currently the technique map says "P2 focal loss: dead_end" but doesn't say "because loss function modifications don't transfer from 500 to 7000 steps due to convergence dynamics."

3. **Research-to-experiment outcome linkage**: When a promoted experiment has `source_item: arxiv:XXXXXXX`, record the outcome (gain magnitude, confirmed/rejected) against that source item. Future grading of papers from the same author, venue, or technique cluster is then calibrated on empirical evidence.

### Medium-Impact, Medium-Cost Improvements

4. **Claim-level extraction**: Restructure `grade.py` to extract specific quantitative claims ("+X% on task Y with Z params") alongside overall scores. Store claims in a structured table, not prose. The verify step then targets specific claims for deeper validation.

5. **Tournament-aware progressive thinking**: Apply the ForgeCode progressive thinking policy within the experiment agent — high reasoning budget for hypothesis formulation and post-run analysis, low budget for mechanical implementation steps.

6. **Temporal relevance decay**: Research findings should decay in weight as the technique map fills with results from that branch. A finding about "Muon optimizer improvements" from 2 weeks ago is less valuable once we've run 5 Muon experiments and mapped the space.

### Lower Priority, Higher Cost

7. **Citation graph traversal**: After a paper is verified, use the Semantic Scholar API to pull forward citations (papers that cite it, published after it). This systematically surfaces technique extensions that ArXiv search misses.

8. **Meta-harness loop**: Given the Meta-Harness result (+7.7pp by auto-discovering harnesses), there is a longer-term opportunity to apply the same technique to the research harness itself — running the orchestrator as the fixed evaluator, and letting an outer agent propose modifications to `research_agent.md` and `experiment_agent.md`. This is the recursive self-improvement loop operating at the harness level rather than the model level.

### Validation Strategy

Before building the generalized plugin, validate the architecture by running the improved loop on Parameter Golf for 50+ experiments and measuring whether the new gates (mandatory critic, early-abort, deduplication, hypothesis tracking) actually reduce the wasted-experiment rate from 44% to below 20%. If the gates work, the architecture is validated and worth generalizing. If they don't, we need to understand why before building a plugin around them.

---

## 14. Detailed Harness Design: An AI Research Harness Based on SOTA Evidence

This section presents a concrete, implementation-level design for an AI research harness. Unlike the abstract plugin architecture in Section 12, this design is grounded in specific patterns proven to work in ForgeCode (25% -> 81.8% on TermBench 2.0), Anthropic's three-agent architecture ($9 solo -> $200 harness with qualitatively different output), and the lessons learned from running this autoresearch system.

The design question is: **what would a ForgeCode-class harness look like if it were purpose-built for AI/ML research instead of software engineering?**

### 14.1 Why Not Just Use a Coding Harness?

The Anthropic and ForgeCode evidence establishes that harness design accounts for 10-56 percentage points of performance on the same model. But both systems are optimized for software engineering, where the feedback signal is binary (tests pass/fail) and verification is near-free (run the test suite).

AI research has fundamentally different properties:

| Property | Coding Harness (ForgeCode/Claude Code) | AI Research Harness |
|---|---|---|
| **Feedback signal** | Binary: tests pass or fail | Continuous: val_bpb improved by 0.003 -- is that signal or noise? |
| **Verification cost** | Near-zero (run tests) | $3-10 per H100 experiment |
| **Session scope** | One task, one session | One hypothesis, one experiment, one cycle -- but knowledge accumulates across hundreds of cycles |
| **Context between sessions** | Git history + progress file | Technique map + hypothesis log + experiment history + research findings + budget state |
| **Failure mode** | Code doesn't compile / tests fail | Experiment runs but produces misleading signal (P2 loss: positive locally, regression on H100) |
| **Planning horizon** | Feature list (known upfront) | Search space (discovered incrementally, with dead ends) |
| **Cost of wrong decision** | Wasted tokens (~$0.10) | Wasted GPU hours (~$3-10) |

A coding harness treats each session as independent. A research harness must maintain **state across hundreds of sessions** -- what's been tried, what worked, what failed, what the current frontier is, and what the most promising next direction is. This is a fundamentally different problem.

### 14.2 Lessons from ForgeCode's Architecture

ForgeCode's Rust-based architecture (23 crates, ~50K lines) reveals several patterns that transfer directly to a research harness:

#### Pattern 1: The Orchestrator Loop

ForgeCode's core is a `while !should_yield` loop in `orch.rs` that:
1. Saves context to the conversation store
2. Fires lifecycle events (where doom loop detection runs)
3. Calls the LLM with retry logic
4. Fires response events (where compaction runs)
5. Checks completion conditions
6. Executes tool calls sequentially
7. Tracks errors and enforces `max_requests_per_turn`

**Research harness equivalent:** Our `orchestrate.py` already implements this pattern, but with a critical difference: ForgeCode's loop runs *within* a single agent session (the agent makes multiple tool calls per session), while ours runs *across* sessions (each agent session is a single cycle that exits). The ForgeCode pattern is more efficient -- the agent maintains context within a session and only resets between sessions.

**Design decision:** The research harness should adopt ForgeCode's intra-session loop for the experiment agent. Instead of spawning a new process per cycle, keep the agent alive for multiple experiment cycles within a single session, using compaction to manage context growth. Reset the session only when the agent's context becomes too stale or when a major strategy shift is needed.

#### Pattern 2: The Hook System

ForgeCode uses a composable event handling framework with 6 lifecycle events: Start, End, Request, Response, ToolcallStart, ToolcallEnd. Four concrete handlers are implemented:

| Handler | Purpose |
|---|---|
| `TracingHandler` | Structured logging |
| `TitleGenerationHandler` | Conversation titles |
| `CompactionHandler` | Context window management |
| `DoomLoopDetector` | Detects repetitive tool call patterns |

**Research harness equivalent:** The research harness needs domain-specific hooks:

| Hook | Fires On | Purpose |
|---|---|---|
| `HypothesisEnforcer` | Before tool call to `promote` or `run_experiment` | Blocks execution if no hypothesis has been recorded for this cycle |
| `BudgetGuard` | Before any compute-spending tool call | Checks remaining budget, enforces rate limiting, blocks if below reserve |
| `ScaleTransferWarner` | Before promotion to expensive tier | Checks if the technique category has known scale-transfer risks |
| `ExperimentTracer` | After experiment completion | Records outcome, resolves hypothesis, updates technique map |
| `CompactionHandler` | After LLM response | Same as ForgeCode -- manages context window pressure |
| `DoomLoopDetector` | Before LLM request | Detects if the agent is repeating failed experiments |

The key insight from ForgeCode is that these hooks are **composable** -- you can chain them with `.and()` and merge them with `.zip()`. This means the research harness can start with a minimal set and add domain-specific hooks incrementally.

#### Pattern 3: Agent-as-Tool Parallelization

ForgeCode's `AgentExecutor` enables a parent agent to delegate multiple tasks to a child agent, with all tasks running in parallel via `join_all()`. The parent specifies tasks as a list of strings, and each task creates a new conversation.

**Research harness equivalent:** This maps directly to the research agent's fetch-and-grade pipeline. Instead of the research agent sequentially fetching from 10 sources, grading items, and verifying findings, a parent "research coordinator" could delegate:
- Task 1: "Fetch and grade ArXiv papers on optimizer modifications for small LMs"
- Task 2: "Fetch and grade GitHub PRs in the parameter-golf repo"
- Task 3: "Check competitor leaderboard for SOTA changes"

All three run in parallel, and the coordinator merges results. This would cut research cycle time by ~3x.

#### Pattern 4: Tool Naming and Schema Design

ForgeCode's most surprising finding: **renaming tool arguments to match training data priors** (e.g., `old_string`/`new_string` for file edits) measurably reduced tool-call error rates. Additionally, putting `required` before `properties` in JSON schemas reduced GPT 5.4 malformed calls.

**Research harness equivalent:** The research harness's tools should be named to match what the model has seen in training:
- `run_experiment` not `execute_training_run`
- `check_results` not `parse_training_log`
- `record_hypothesis` not `log_prediction`
- `promote_to_gpu` not `request_h100_allocation`

The tool schemas should be flat (not nested), with `required` fields listed first.

#### Pattern 5: Progressive Thinking

ForgeCode applies a tiered reasoning budget:
1. First 10 messages: **very high thinking** (planning phase)
2. Messages 11+: **low thinking** (execution phase)
3. Verification skill invoked: **back to high thinking** (decision point)

**Research harness equivalent:** The experiment agent's cycle has three distinct phases that map to different thinking budgets:

| Phase | Messages | Thinking Budget | Why |
|---|---|---|---|
| **Orientation** | 1-3 | High | Read decision state, identify what to try, form hypothesis |
| **Implementation** | 4-8 | Low | Edit training script, run experiment -- mechanical execution |
| **Analysis** | 9-12 | High | Parse results, resolve hypothesis, decide whether to promote |

This is not currently implemented -- our agents run with uniform thinking budget throughout. Adding progressive thinking would reduce token cost per cycle by ~30-40% while maintaining quality on the decisions that matter.

#### Pattern 6: Enforced Verification (The Evaluator)

ForgeCode's biggest single improvement was **enforced verification** -- the model must call a verification skill before completing, switching from builder mode to reviewer mode. Without enforcement, the model "would implement a solution, sound confident, and stop."

Anthropic's three-agent architecture takes this further: the evaluator is a separate agent that uses Playwright to interact with the running application, grading against specific criteria with hard thresholds.

**Research harness equivalent:** After every experiment, the agent must:
1. Parse the training log and extract the actual val_bpb
2. Compare against the hypothesis prediction
3. Check for anomalies (training instability, NaN losses, artifact size violations)
4. Resolve the hypothesis with a learned rule
5. Decide: keep, discard, or investigate further

Currently, steps 1-2 happen but steps 3-5 are advisory. The harness should **enforce** all five steps before the agent can exit the cycle. If the agent tries to exit without resolving its hypothesis, the harness should inject a reminder (like ForgeCode's `DoomLoopDetector`).

### 14.3 The Three-Agent Architecture for Research

Drawing from Anthropic's planner-generator-evaluator pattern and adapting it for research:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESEARCH HARNESS                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   PLANNER    │    │  EXPERIMENT  │    │  EVALUATOR   │      │
│  │   (Research  │───>│    AGENT     │───>│   (Critic +  │      │
│  │    Agent)    │    │  (Generator) │    │  Verifier)   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         │    ┌──────────────┴──────────────┐    │               │
│         │    │      ORCHESTRATOR           │    │               │
│         │    │  (Lifecycle hooks, budget,  │    │               │
│         │    │   compute routing, state)   │    │               │
│         │    └──────────────┬──────────────┘    │               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SHARED STATE (File-Based)                   │   │
│  │  decision_state.md | hypotheses.jsonl | results.tsv      │   │
│  │  technique_map.json | research_results.jsonl | budget    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Planner (Research Agent):** Discovers techniques, grades them, maintains the technique map, and produces a prioritized list of what to try next. Equivalent to Anthropic's planner that expands a one-line prompt into a full spec. Runs on-demand or on a schedule.

**Generator (Experiment Agent):** Picks the highest-priority technique from the planner's output, forms a hypothesis, implements it, runs the experiment, and produces results. Equivalent to Anthropic's generator that implements one feature per sprint.

**Evaluator (Critic + Verifier):** Reviews the experiment results independently of the generator. Checks: did the metric actually improve? Is the improvement statistically significant? Does the technique transfer across tiers? Are there anomalies in the training curve? Equivalent to Anthropic's evaluator that uses Playwright to test the running application.

The critical difference from Anthropic's pattern: **the evaluator runs after every experiment, not just at the end of a sprint.** In research, every experiment is a decision point -- the evaluator's judgment determines whether to keep the change, discard it, or investigate further.

### 14.4 Sprint Contracts for Research

Anthropic's sprint contract pattern -- where the generator and evaluator agree on "what done looks like" before any code is written -- maps naturally to hypothesis-driven research:

**Before each experiment cycle:**
1. The experiment agent reads the decision state and selects a technique
2. It records a structured hypothesis: `{technique, prediction, basis, scale_risk}`
3. The evaluator reviews the hypothesis and the proposed code change
4. Both agree on success criteria: "If val_bpb improves by > 0.005 on local, promote to H100"

**After the experiment:**
1. The experiment agent runs the experiment and records raw results
2. The evaluator independently parses the training log
3. The evaluator grades against the agreed criteria
4. If criteria are met: promote. If not: discard with a learned rule.

This is the **sprint contract** adapted for research. The key benefit: the evaluator catches cases where the generator would self-evaluate positively ("the loss went down!") when the actual result is ambiguous or misleading.

### 14.5 Context Engineering for Research Sessions

ForgeCode's compaction strategy (eviction window + retention window, triggered by token/message thresholds) needs adaptation for research:

#### The Decision State as Structured Handoff

Our `decision_state.md` already implements Anthropic's "structured handoff artifact" pattern. But it can be improved based on ForgeCode's evidence:

**Current state (2-4K tokens):**
```
# Decision State
## Current Best: 1.1563 bpb
## Budget: $X spent, $Y remaining
## Last 5 Experiments: [table]
## Unacked Research: [list]
## Dead Ends: [list]
## Learned Rules: [list]
```

**Improved state (structured for progressive thinking):**
```
# Decision State (Orientation Phase — read this first, think deeply)

## CRITICAL CONTEXT
- Best H100 val_bpb: 1.1563
- Gap to SOTA: 0.047 bpb
- Budget remaining: $X ($Y reserve)
- Experiments this session: 0

## WHAT TO TRY NEXT (ranked by expected impact)
1. TTT 10-epoch cosine (expected -0.030 bpb, basis: TTT 3-epoch gave -0.020)
2. Full Hessian GPTQ (expected -0.010 bpb, basis: 4/7 top entries use it)
3. LoRA TTT rank-8 (expected -0.015 bpb, basis: 24x more effective than SGD TTT)

## WHAT NOT TO TRY (dead ends with reasons)
- P2 focal loss: REGRESSION +0.067 on H100 (loss functions don't transfer from 500 to 7000 steps)
- Turbo-Muon: worse at 7000+ steps (optimizer dynamics are scale-dependent)
- NorMuon VR: marginal on H100 (-0.0002, not worth the complexity)

## LEARNED RULES (from resolved hypotheses)
- "Loss function modifications should NOT be validated at 500 local steps"
- "Optimizer schedule changes need 2000+ step validation to be meaningful"
- "Compression improvements (brotli, zstd) always transfer between tiers"

# Implementation Phase — low thinking from here
## Last 5 Experiments: [compact table]
## Unacked Research: [top 3 only]
```

The key change: the decision state is **structured for progressive thinking**. The orientation section (what to try, what not to try, learned rules) gets high thinking budget. The implementation details (experiment history, research findings) get low thinking budget.

#### Compaction Strategy for Research

ForgeCode compacts by replacing a range of messages with a summary, preserving the last reasoning block for chain continuity. For research, compaction should additionally preserve:

1. **The current hypothesis** -- never compact away the active hypothesis
2. **The last experiment result** -- the agent needs to know what just happened
3. **Learned rules** -- these are the accumulated wisdom; losing them means repeating mistakes
4. **Budget state** -- the agent must always know how much it can spend

Everything else (research findings, competitor scores, technique map details) can be compacted to summaries.

### 14.6 The Tool Catalog for Research

Based on ForgeCode's evidence that tool naming and schema design are reliability variables, here is the research harness tool catalog:

```
Research Harness Tool Catalog
├── Experiment Tools
│   ├── record_hypothesis(technique, prediction, basis, scale_risk)
│   ├── run_experiment(script, tier, max_steps, description)
│   ├── resolve_hypothesis(technique, outcome, learned_rule)
│   ├── promote_to_gpu(script_path, description, expected_improvement)
│   └── check_results(run_id) -> {val_bpb, steps, artifact_size, anomalies}
│
├── Research Tools
│   ├── search_papers(query, sources, max_results)
│   ├── grade_finding(title, abstract, url) -> {score, tier, scale_risk}
│   ├── ack_finding(finding_id, action: adopted|rejected|deferred)
│   └── request_research(topic, urgency)
│
├── Knowledge Tools
│   ├── query_technique_map(technique) -> {status, evidence, parent}
│   ├── update_technique_map(technique, status, evidence)
│   ├── query_experiments(filter) -> [{description, bpb, status, cost}]
│   └── get_learned_rules(topic?) -> [rules]
│
├── Compute Tools
│   ├── check_budget() -> {spent, remaining, reserve, rate_limit_status}
│   ├── estimate_cost(tier, steps) -> {estimated_cost, within_budget}
│   └── abort_run(run_id, reason)
│
└── Standard Tools (inherited from coding harness)
    ├── read, write, patch (file operations)
    ├── shell (command execution)
    ├── search (codebase search)
    └── todo_write (planning state)
```

**Schema design principles (from ForgeCode evidence):**
- All schemas are flat (no nesting)
- `required` fields listed before `properties`
- Argument names match training data priors (`technique` not `method_name`, `prediction` not `expected_outcome`)
- Tool descriptions include explicit examples of correct usage
- Truncation warnings are inline in tool output, not just in metadata

### 14.7 The Evaluator Agent Design

The evaluator is the most novel component -- it doesn't exist in coding harnesses because tests serve the same function. In research, there is no equivalent of "run the test suite." The evaluator must make judgment calls.

**Evaluator grading criteria (adapted from Anthropic's frontend design criteria):**

| Criterion | Weight | What It Measures | Hard Threshold |
|---|---|---|---|
| **Metric improvement** | 40% | Did val_bpb actually improve? By how much? Is it statistically significant? | Must improve by > 0.001 to keep |
| **Hypothesis validity** | 20% | Did the outcome match the prediction? If not, why? | Must resolve hypothesis with a learned rule |
| **Constraint compliance** | 20% | Artifact size < 16MB? Training time < 600s? No data leakage? | Hard fail on any violation |
| **Transfer confidence** | 10% | How likely is this to hold at full scale? (Based on technique category and scale-transfer risk) | Warn if high-risk technique |
| **Implementation quality** | 10% | Is the code change clean? One variable changed? Reversible? | Warn if diff > 100 lines or multiple changes stacked |

The evaluator runs after every experiment and produces a structured verdict:

```json
{
  "verdict": "keep|discard|investigate",
  "metric_delta": -0.003,
  "hypothesis_match": true,
  "constraint_violations": [],
  "transfer_confidence": "high",
  "implementation_quality": "clean",
  "critique": "Improvement is real but small. The technique adds 15 lines of complexity for 0.003 bpb. Consider whether the complexity is worth it given the remaining gap to SOTA.",
  "recommendation": "Keep, but prioritize higher-impact techniques next cycle."
}
```

**Calibration:** Like Anthropic's evaluator, the research evaluator needs calibration against human judgment. The initial calibration set is the 18 experiments already run -- we know which ones were keeps, which were discards, and which were misleading. The evaluator should produce the same verdicts on this historical data before being trusted on new experiments.

### 14.8 Doom Loop Detection for Research

ForgeCode's `DoomLoopDetector` detects repetitive tool call patterns. The research equivalent detects:

1. **Technique repetition:** The agent is trying variations of a dead-end technique (e.g., 11 NorMuon entries)
2. **Scale-transfer blindness:** The agent keeps promoting techniques that fail at H100 scale
3. **Budget drain:** The agent is spending faster than the improvement rate justifies
4. **Hypothesis-free experimentation:** The agent is running experiments without recording hypotheses

Detection rules:

```python
DOOM_LOOP_RULES = {
    "technique_repetition": {
        "trigger": "3+ experiments on the same technique with no improvement",
        "action": "Inject: 'You have tried {technique} 3 times with no improvement. Mark it as dead_end and move on.'"
    },
    "scale_transfer_blindness": {
        "trigger": "2+ H100 regressions from techniques that passed local validation",
        "action": "Inject: 'Your last 2 H100 promotions regressed. Review scale-transfer risk before promoting again.'"
    },
    "budget_drain": {
        "trigger": "3+ consecutive H100 runs with no improvement",
        "action": "Inject: 'You have spent ${cost} on 3 consecutive non-improving runs. Switch to local-only experimentation until you find a strong signal.'"
    },
    "hypothesis_free": {
        "trigger": "Experiment started without record_hypothesis call",
        "action": "Block: 'You must record a hypothesis before running an experiment.'"
    }
}
```

### 14.9 What This Design Changes vs. Our Current System

| Component | Current System | Research Harness Design | Why Change |
|---|---|---|---|
| **Agent lifecycle** | Single-turn `-p` mode, exit after each cycle | Multi-turn session with compaction, reset on strategy shift | Reduces process spawn overhead, maintains intra-session context |
| **Evaluator** | Critic gate is a function call in the orchestrator | Separate evaluator agent with structured grading criteria | Separating generator from evaluator is "the single strongest lever" (Anthropic) |
| **Planning enforcement** | Advisory ("one variable at a time") | Mandatory hypothesis recording via hook, blocks execution without it | "Optional tools get deprioritized under pressure" (ForgeCode) |
| **Thinking budget** | Uniform throughout cycle | Progressive: high for orientation/analysis, low for implementation | Reduces token cost ~30-40% while maintaining decision quality |
| **Tool schemas** | Implicit (agent uses shell commands) | Explicit tool catalog with flat schemas, training-aligned naming | Tool naming is "a reliability variable, not an aesthetic choice" (ForgeCode) |
| **Context handoff** | `decision_state.md` (flat structure) | Structured for progressive thinking (orientation section + implementation section) | Matches ForgeCode's evidence that early messages need high thinking |
| **Doom loop detection** | None | Rule-based detection with injected reminders | Prevents the 11-NorMuon-entry problem and budget drain |
| **Research parallelization** | Sequential fetch-grade-verify | Agent-as-tool parallel delegation for independent research tasks | ForgeCode's `join_all()` pattern for parallelizable subtasks |
| **Verification** | Post-flight checks in orchestrator | Enforced verification skill that switches agent to reviewer mode | "Enforced verification was the biggest single improvement" (ForgeCode) |
| **Sprint contracts** | None | Hypothesis + success criteria agreed before experiment | Anthropic's sprint contract pattern adapted for research |

### 14.10 Implementation Priority

Based on the evidence from ForgeCode (each phase was a targeted intervention against a specific failure class), the implementation should follow the same pattern -- one failure mode at a time, measured before and after:

| Priority | Intervention | Expected Impact | Evidence |
|---|---|---|---|
| **1** | Enforced hypothesis recording (hook that blocks experiment without hypothesis) | Prevents hypothesis-free experimentation; enables learned rules | ForgeCode: `todo_write` enforcement took them from 38% -> 66% |
| **2** | Enforced verification (evaluator grades every experiment result) | Catches misleading signals before they waste H100 budget | ForgeCode: "biggest single improvement"; Anthropic: "strong lever" |
| **3** | Progressive thinking (high for orientation/analysis, low for implementation) | ~30-40% token cost reduction per cycle | ForgeCode: part of the 66% -> 78.4% phase |
| **4** | Doom loop detection (technique repetition, scale-transfer blindness) | Prevents the NorMuon-11-entries problem and budget drain | ForgeCode: `DoomLoopDetector` is a core hook |
| **5** | Research parallelization (agent-as-tool for independent fetch tasks) | ~3x faster research cycles | ForgeCode: subagent parallelization was part of the speed architecture |
| **6** | Multi-turn sessions with compaction (replace single-turn `-p` mode) | Maintains intra-session context, reduces process spawn overhead | Anthropic: Opus 4.6 can drop sprint construct; compaction handles context |
| **7** | Explicit tool catalog with flat schemas | Reduces tool-call errors | ForgeCode: "tool naming is a reliability variable" |

**The key principle from ForgeCode:** "Each phase was a targeted intervention against a specific failure class, not a general quality improvement. That specificity is what makes the result reproducible."

### 14.11 Measuring Success

The harness should be evaluated on the same metrics we identified in the critical analysis (Section 7):

| Metric | Current Baseline | Target | How to Measure |
|---|---|---|---|
| Wasted H100 runs (regressions) | 44% (4/9) | < 20% | Count H100 runs where val_bpb worsened |
| Misleading local signals | 26% of spend ($20.88) | < 10% of spend | Track techniques that pass local but fail H100 |
| Research duplication | 60% redundancy (149 entries, ~60 unique) | < 20% redundancy | Jaccard similarity check on new entries |
| Hypothesis coverage | 0% (no hypothesis tracking existed) | 100% | Every experiment must have a recorded hypothesis |
| Learned rules accumulated | 0 | 5+ per 20 experiments | Count resolved hypotheses with learned rules |
| Experiments per dollar | ~0.23 ($79.83 / 18 experiments) | > 0.5 | Total experiments / total spend |
| Time to new best | ~6 experiments between improvements | < 4 experiments | Count experiments between each new best val_bpb |

---

## 15. Research Pipeline Hardening: Source Authority, Claim Extraction, and Reflection Validation

External review of the research pipeline identified five structural weaknesses in how the system ingests, grades, and acts on web-sourced research. All five have been implemented as code changes.

### 15.1 Problem: No Source Authority Scoring

The Tavily relevance cutoff of 0.4 was a single-dimensional quality signal applied uniformly across all sources. A marketing blog post and a peer-reviewed ArXiv paper could both clear the same threshold. This meant the LLM grading stage received a mix of high-authority and low-authority content with no way to distinguish them before spending tokens.

**Fix implemented** (`research/fetch.py`):

A deterministic, zero-token source authority tier system applied before any LLM processing:

| Tier | Sources | Relevance Floor | Rationale |
|---|---|---|---|
| **Tier 1** (peer-reviewed) | `arxiv.org`, `openreview.net`, `semanticscholar.org`, `proceedings.mlr.press`, `proceedings.neurips.cc`, `aclanthology.org` | 0.30 | Peer-reviewed or curated academic sources; lower floor because the content is inherently higher quality |
| **Tier 2** (technical community) | `github.com`, `huggingface.co`, `paperswithcode.com`, `reddit.com/r/MachineLearning` | 0.40 | Technical community sources with some quality signal (stars, upvotes) but no formal review |
| **Tier 3** (general web) | Everything else | 0.55 | Blog posts, news articles, marketing content; higher floor because quality is unpredictable |

The tier is stamped on each `RawItem` as `authority_tier` (1/2/3) and persisted in the cache. Items below their tier's relevance floor are dropped before deduplication, saving both cache space and downstream LLM tokens.

### 15.2 Problem: Fail-Open Pre-Filter Lets Uncertain Items Through at Full Confidence

The Stage 2 pre-filter extracts parameter counts and bit-widths via regex. If extraction fails (no size/parameter data in the abstract), the item passes through -- a "fail-open" design. This is correct (rejecting unknown items would miss novel techniques), but the downstream grading treated these items identically to items that passed deterministic feasibility checks.

**Fix implemented** (`research/grade.py`):

Items that failed deterministic extraction are tagged as `fail_open` and require **2 points higher** to reach each tier:

| Tier | Normal Threshold (with competitors) | Fail-Open Threshold |
|---|---|---|
| **Tier A** | >= 12/19 | >= 14/19 |
| **Tier B** | >= 9/19 | >= 11/19 |

Without competitor data:

| Tier | Normal Threshold | Fail-Open Threshold |
|---|---|---|
| **Tier A** | >= 12/17 | >= 14/17 |
| **Tier B** | >= 9/17 | >= 11/17 |

This means items with higher uncertainty (no extractable constraints) need stronger LLM evidence to reach the expensive verification and H100 promotion stages.

### 15.3 Problem: LLM Grades Whole Papers, Not Specific Claims

The grading prompt received title + abstract as prose. LLMs grading research quality are susceptible to confident-sounding but unverifiable abstracts. A paper claiming "+15% improvement" in flowing prose scores the same as one with a specific, verifiable claim like "+0.023 bpb on FineWeb at 50M params with INT4 quantization."

**Fix implemented** (`research/extract_claims.py`, `research/grade.py`, `research/verify.py`):

A new `extract_claims()` function performs deterministic regex extraction of quantitative claims from abstracts:

```python
# Extracts structured claims like:
{
    "claim_type": "absolute",        # or "delta", "comparison"
    "metric": "bpb",                 # or "perplexity", "accuracy", etc.
    "value": "1.08",
    "context": "achieves 1.08 bpb on FineWeb validation"
}
```

Claims are extracted at two points:
1. **During batch grading**: Extracted claims are appended to the grading payload so the LLM grades against specific numbers, not prose
2. **During Tier A verification**: Claims are injected into the re-grading prompt alongside full content and corroborating evidence

This dramatically shrinks the hallucination surface area -- the LLM is grading "does this claim of +0.023 bpb at 50M params seem plausible?" rather than "is this paper good?"

### 15.4 Problem: Circular Web-Based Corroboration

The Stage 5 deep verification ran 2 corroborating evidence searches per Tier A item. Both queries went through Tavily -- the same web search infrastructure that produced the original item. If a technique was widely discussed on social media or marketing blogs, multiple low-quality corroborating sources would reinforce it, creating a circular validation loop.

**Fix implemented** (`research/sources/semantic_scholar.py`, `research/verify.py`):

One of the two corroboration queries is now replaced with a **Semantic Scholar forward-citation check** for items that have an S2 paper ID:

```
Corroboration sources:
  1. Tavily web search (unchanged) -- catches community discussion, blog posts, implementations
  2. Semantic Scholar forward citations (NEW) -- structured, deterministic, not circular
```

The `get_forward_citations()` function:
- Looks up the paper's S2 ID (from URL or title search)
- Fetches papers that cite it (forward citations)
- Returns citation count + titles of citing papers as structured evidence
- Falls back to Tavily if no S2 ID is found

This provides a non-circular quality signal: a paper cited by 15 subsequent works in reputable venues is more credible than one with zero citations but heavy blog coverage.

### 15.5 Problem: Self-Referential Reflection Cycle

The reflection cycle (Stage 6) synthesises experiment history into strategic guidance using the LLM. The LLM reasons about its own prior outputs -- technique map updates, strategy recommendations, dead-end classifications. Without external validation, the reflection can produce plausible-sounding but incorrect strategic guidance that compounds over cycles.

**Fix implemented** (`research/reflect.py`):

A new `_validate_reflection_against_results()` function performs a deterministic audit of the reflection's conclusions against the actual empirical record in `results.tsv`:

1. **Dead-end validation**: If the reflection marks a technique as `dead_end`, the function checks whether `results.tsv` actually contains a negative result for that technique. If not, a warning is generated: "Reflection marks X as dead_end but no negative result found in results.tsv"

2. **Proven validation**: If the reflection marks a technique as `proven`, the function checks for a positive result. If not, a warning is generated.

3. **Warning injection**: Validation warnings are appended to the strategy output before it's written to `strategy.md`, making them visible to both agents in subsequent cycles.

The matching uses keyword overlap between technique names and experiment descriptions (lowercased, first 20 chars or word-level matching), with a tolerance for naming variations.

### 15.6 Impact Assessment

| Fix | Cost (tokens) | Expected Impact |
|---|---|---|
| Source authority tiers | 0 (deterministic) | Filters ~30-40% of low-quality web content before LLM grading |
| Fail-open threshold raise | 0 (threshold change) | Reduces false-positive Tier A classifications for uncertain items |
| Claim-level extraction | ~50 tokens/item (regex) + improved grading accuracy | Tighter LLM grading target; fewer hallucinated quality assessments |
| Forward-citation corroboration | ~1 API call/item | Non-circular validation; catches papers with no academic impact |
| Reflection validation | 0 (deterministic) | Prevents compounding strategic errors across cycles |

Total additional cost per research cycle: ~1 Semantic Scholar API call per Tier A item (free tier: 100 requests/5 minutes). All other fixes are zero-cost deterministic operations.

---

## 16. Anthropic Multi-Agent Research System: Lessons for the Harness

Anthropic published two engineering articles detailing their production multi-agent research system and long-running harness patterns. Several findings directly validate, extend, or challenge decisions in this architecture.

### 16.1 Token Volume Is the Primary Quality Driver

Anthropic's multi-agent research system found that **token usage alone explains 80% of performance variance** on BrowseComp, with tool call count and model choice accounting for only the remaining 20%.

**Implication for this harness**: The research agent's quality is gated more by how many tokens it gets to spend reasoning per cycle than by any individual architectural choice. The current model (fixed 10-source fetch, grading in batches of 10) doesn't adapt token budget to query complexity.

The Anthropic team embedded **explicit effort-scaling rules** in their prompts:

| Query Complexity | Subagents | Tool Calls | Example |
|---|---|---|---|
| Simple lookup | 1 | 3-10 | "What is the current SOTA on FineWeb?" |
| Direct comparison | 2-4 | 10-15 each | "Compare XSA-all vs standard attention at 50M params" |
| Complex research | 10+ | Divided responsibilities | "Survey all quantization approaches compatible with 16MB artifact" |

**Research harness equivalent**: Leaderboard-stable cycles (no new competitor PRs, no research queue requests) should fetch 2-3 sources and grade shallowly. Cycles triggered by a specific `research_queue.jsonl` request should get the full 10-source pipeline and deep verification. This is a prompt-level change that requires no code modification -- the orchestrator already knows whether the cycle was triggered by a queue request or by the periodic timer.

### 16.2 Start Wide, Then Narrow

Anthropic found that "agents default to overly long, specific queries that return few results." The fix: prompt agents to start with **short, broad queries, evaluate what's available, then progressively narrow focus** -- mirroring expert human research practice.

**Current problem**: The research agent's Tavily queries are likely precise (`"coprime stride loader parameter golf FineWeb"`) when broader queries (`"data loading tricks language model training"`) at step 1 would surface techniques the specific query misses. The narrow query finds exactly what you already know about; the broad query finds what you don't know you're looking for.

**Fix**: A two-phase query strategy in the research agent prompt:
1. **Phase 1 (broad)**: 2-3 short queries (3-5 words) covering the general technique category
2. **Phase 2 (narrow)**: Refine based on Phase 1 results, targeting specific implementations or papers

This is a zero-cost prompt engineering change.

### 16.3 Source Quality Validation

Anthropic's human evaluation found that early agents "consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources like academic PDFs or personal blogs." This is the exact problem Section 15.1's source authority tier system addresses -- and the Anthropic finding validates that this is a real, production-level problem that human evaluation caught and automated evals missed.

The authority tier system (Tier 1: arxiv/openreview at 0.30 floor, Tier 2: github/huggingface at 0.40, Tier 3: general web at 0.55) is the correct architectural response. Anthropic's experience confirms it.

### 16.4 JSON Over Markdown for Agent-Writable State Files

The long-running harness article contains a specific finding: **agents are less likely to inappropriately overwrite JSON files than Markdown files**. Anthropic switched from Markdown to JSON for their feature list specifically to prevent agents from editing fields they weren't supposed to touch.

**Current exposure**: `strategy.md` and `decision_state.md` are both Markdown files that agents read and the orchestrator writes. The `technique_map.json` is already JSON (correctly). But `strategy.md` (5 strategy entries, cap-enforced) is a write target for the reflection cycle -- if the experiment agent ever writes to it directly (or if a future multi-turn session starts writing context back), JSON would be safer.

**Recommended change**: Convert `strategy.md` to `strategy.json` with an explicit schema:

```json
{
    "entries": [
        {
            "technique": "TTT epoch scaling",
            "rationale": "Expected -0.030 to -0.040 bpb based on PR #1180 evidence",
            "priority": 1,
            "status": "pending"
        }
    ],
    "updated_at": "2026-04-06T12:00:00Z",
    "cycle_count": 15
}
```

This makes the structure machine-parseable and prevents agents from accidentally reformatting or expanding the file beyond its intended scope.

### 16.5 Task Boundaries for Parallel Subagents

Section 14's proposal for parallel research delegation (ArXiv subagent, GitHub subagent, competitor subagent in `join_all()`) is architecturally correct, but the Anthropic article documents precisely where this fails: **without detailed task descriptions, subagents duplicate work, leave gaps, or fail to find necessary information**.

They observed two subagents investigating "current 2025 supply chains" while a third explored the 2021 automotive chip crisis -- no effective division of labor.

**Required specification for each delegated subagent**:

| Field | Purpose | Example |
|---|---|---|
| **Source domain** | Prevents overlap | "Search only arxiv.org and openreview.net" |
| **Output format** | Enables structured aggregation | `{findings: [{title, url, score, claimed_improvement}]}` |
| **Explicit exclusions** | Prevents duplication | "Do not search GitHub; that is covered by another subagent" |
| **Tool call budget** | Prevents runaway cost | "Maximum 8 tool calls" |

Without item 3 especially, the ArXiv subagent and Semantic Scholar subagent will likely fetch the same papers through different APIs.

### 16.6 The Self-Improving Prompt Loop

Anthropic's most actionable finding: "Claude 4 models can be excellent prompt engineers." They built a tool-testing agent that was given a flawed MCP tool description, attempted to use it, identified the failure modes, and rewrote the tool description -- producing a **40% decrease in task completion time** for future agents.

**Application to this system**: After accumulating 30+ resolved hypotheses and failure critiques, run a one-off self-improvement pass:

1. Feed the experiment agent's prompt (`experiment_agent.md`)
2. Feed the last 10 failure critiques from `hypotheses.jsonl`
3. Feed the doom loop events (if any)
4. Ask Claude to diagnose failure patterns and rewrite the prompt

This is especially valuable for the grading prompt in `grade.py`, which has the highest surface area for LLM error. The grading prompt's 6 dimensions, tier thresholds, and dynamic context injection are all candidates for prompt-level optimization based on empirical grading outcomes.

**Constraint**: This should be a human-reviewed one-off operation, not an automated loop. Automated prompt rewriting without human review risks drift toward prompts that game the evaluation metrics rather than improving actual research quality.

### 16.7 Production Tracing

Anthropic identifies full production tracing as the diagnostic that let them systematically fix agent failures -- before it, "users would report agents 'not finding obvious information,' but we couldn't see why."

**Current state**: The system has a dashboard and the JSONL audit trail (`research_results.jsonl`, `results.tsv`), but no span-level tracing of individual agent decisions.

**Recommended addition**: Instrument the research agent's grade/verify cycle to emit structured trace events:

```json
{
    "timestamp": "2026-04-06T12:00:00Z",
    "stage": "grade",
    "item_id": "arxiv:2603.28254",
    "score_before": null,
    "score_after": 14,
    "tier": "A",
    "fail_open": false,
    "authority_tier": 1,
    "claims_extracted": 2,
    "tokens_used": 450,
    "decision": "promote_to_verify"
}
```

Appended to `trace.jsonl` alongside `research_results.jsonl`. This gives the same diagnostic capability Anthropic describes -- when an experiment fails because the research agent missed a key technique, you can trace exactly which fetch query missed it, which grading batch under-scored it, and whether the deduplication step dropped it.

---

## 17. The Initializer Pattern: Structured First-Run Setup

The long-running harness article's core pattern -- **a different prompt for the very first context window** to set up structured state -- maps directly to an unaddressed gap in this system.

### 17.1 The Problem

Currently, the research agent starts every cycle identically. There is no distinction between the first run (where the entire search space needs mapping) and the 50th run (where the search space is well-characterized and only incremental updates matter).

Anthropic's finding: a specialized initializer agent that writes a comprehensive feature list in JSON (marking all features initially "failing") gives subsequent agents a clear, authoritative view of what full success looks like. Without this upfront structure, agents "tended to try to do too much at once" and "declare the job done" prematurely.

The research agent's over-production problem (149 entries, 60 unique in the original run) is exactly this failure mode -- the agent had no authoritative definition of "enough research."

### 17.2 The Three-Agent Initialization Model

| Agent Type | Trigger | Responsibility |
|---|---|---|
| **Initializer** (once per competition) | First run, or `technique_map.json` doesn't exist | Write `technique_map.json` skeleton with all known SOTA techniques pre-populated as `promising`; write `research_charter.json` defining the 5 most important research dimensions; validate the full pipeline end-to-end |
| **Research agent** (on-demand) | `research_queue.jsonl` signal or periodic timer | Fetch, grade, verify, reflect -- against the charter's defined dimensions |
| **Charter updater** (on major SOTA change) | New leaderboard entrant with >0.03 bpb improvement over current best | Re-runs initializer logic to update the charter with new research directions |

### 17.3 The Research Charter

The initializer produces a `research_charter.json` that defines the search space boundaries:

```json
{
    "competition": "parameter-golf",
    "target_metric": "val_bpb",
    "current_best": 1.1563,
    "merged_sota": 1.1147,
    "gap": 0.0416,
    "research_dimensions": [
        {
            "name": "quantization",
            "description": "Weight compression techniques that fit within 16MB artifact",
            "status": "partially_explored",
            "known_techniques": ["GPTQ INT6", "ternary_quant"],
            "open_questions": ["Full Hessian GPTQ", "mixed-precision quantization"]
        },
        {
            "name": "eval_time_optimization",
            "description": "Techniques applied during evaluation, not training",
            "status": "promising",
            "known_techniques": ["TTT", "SLOT", "EGGROLL"],
            "open_questions": ["LoRA TTT", "TTT epoch scaling beyond 10"]
        }
    ],
    "max_research_entries_per_dimension": 5,
    "completion_criteria": "All dimensions have status 'explored' or 'dead_end'",
    "created_at": "2026-04-06T12:00:00Z"
}
```

The `max_research_entries_per_dimension` field directly addresses the over-production problem: once a dimension has 5 entries, the research agent should stop fetching for that dimension and focus on under-explored ones.

### 17.4 Charter-Driven Research Cycles

With the charter in place, each research cycle becomes targeted:

1. **Read charter**: Identify dimensions with status `partially_explored` or `promising`
2. **Effort-scale**: If triggered by queue request, full 10-source pipeline; if periodic, 2-3 sources focused on the least-explored dimension
3. **Broad-first queries**: Start with short queries covering the dimension category, then narrow
4. **Entry cap enforcement**: Skip dimensions that have reached `max_research_entries_per_dimension`
5. **Charter update**: After grading, update dimension status based on findings

This transforms the research agent from "find everything about everything" to "fill in the gaps in a structured map of the search space."

---

## 18. Consolidated Harness Design Map

This section consolidates all findings from Sections 9-17 into a single implementation map for building the research harness in a separate repository. Each item is tagged with its evidence source and implementation complexity.

### 18.1 Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (Rust/Python)                   │
│  - Event loop with hook system (ForgeCode pattern)              │
│  - Deterministic routing, zero LLM logic                        │
│  - Doom loop detection (4 rules)                                │
│  - Budget management with rate limiting                         │
│  - Pod lifecycle management                                     │
│  - Trace event emission to trace.jsonl                          │
└──────────┬──────────────┬──────────────┬───────────────────────┘
           │              │              │
    ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
    │ INITIALIZER │ │RESEARCH │ │ EXPERIMENT  │
    │   AGENT     │ │  AGENT  │ │   AGENT     │
    │             │ │         │ │             │
    │ First-run   │ │ Fetch → │ │ Hypothesis →│
    │ charter +   │ │ Grade → │ │ Implement → │
    │ technique   │ │ Verify →│ │ Local test →│
    │ map setup   │ │ Reflect │ │ Promote →   │
    └─────────────┘ └────┬────┘ │ H100 run    │
                         │      └──────┬──────┘
                    ┌────▼────┐  ┌─────▼──────┐
                    │SUBAGENTS│  │ EVALUATOR  │
                    │(parallel│  │   AGENT    │
                    │ fetch)  │  │            │
                    └─────────┘  │ Grades each│
                                 │ experiment │
                                 │ result     │
                                 └────────────┘
```

### 18.2 Implementation Priority (Ordered by Evidence of Impact)

| # | Intervention | Evidence Source | Complexity | Expected Impact |
|---|---|---|---|---|
| **1** | Enforced hypothesis recording (hook blocks experiment without hypothesis) | ForgeCode: `todo_write` enforcement = 38% -> 66% | Low | Prevents hypothesis-free experimentation; enables learned rules |
| **2** | Effort-scaling rules in research agent prompt | Anthropic: token volume = 80% of variance | Zero (prompt) | Adapts research depth to query complexity; reduces token waste on stable cycles |
| **3** | Broad-first query strategy | Anthropic: agents default to overly specific queries | Zero (prompt) | Expands technique discovery surface |
| **4** | Enforced verification (evaluator grades every result) | ForgeCode: "biggest single improvement"; Anthropic: "strong lever" | Medium | Catches misleading signals before H100 spend |
| **5** | Source authority tiers | Anthropic: agents chose SEO farms over academic sources; Section 15.1 | **Done** | Filters low-quality content before LLM grading |
| **6** | Fail-open threshold raise | Section 15.2 | **Done** | Reduces false-positive Tier A for uncertain items |
| **7** | Claim-level extraction | Section 15.3 | **Done** | Tighter LLM grading; fewer hallucinated assessments |
| **8** | Forward-citation corroboration | Section 15.4; Anthropic: circular corroboration problem | **Done** | Non-circular validation via structured citations |
| **9** | Reflection validation against results.tsv | Section 15.5 | **Done** | Prevents compounding strategic errors |
| **10** | `strategy.md` -> `strategy.json` | Anthropic: agents less likely to overwrite JSON | Low | Prevents accidental agent overwrites of strategy state |
| **11** | Progressive thinking (high for orientation, low for implementation) | ForgeCode: part of 66% -> 78.4% phase | Medium | ~30-40% token cost reduction per cycle |
| **12** | Doom loop detection (4 rules) | ForgeCode: `DoomLoopDetector` is a core hook | Medium | Prevents NorMuon-11-entries problem and budget drain |
| **13** | Research charter + initializer agent | Anthropic: initializer pattern prevents over-production | Medium | Transforms research from "find everything" to "fill gaps in structured map" |
| **14** | Task boundary specification for parallel subagents | Anthropic: without boundaries, subagents duplicate work | Medium | Required before implementing parallel research delegation |
| **15** | Span-level trace logging (`trace.jsonl`) | Anthropic: tracing enabled systematic failure diagnosis | Low | Enables root-cause analysis of missed techniques |
| **16** | Research parallelization (agent-as-tool for independent fetch) | ForgeCode: `join_all()` pattern | High | ~3x faster research cycles |
| **17** | Multi-turn sessions with compaction | Anthropic: Opus 4.6 can drop sprint construct | High | Maintains intra-session context; reduces spawn overhead |
| **18** | Self-improving prompt pass | Anthropic: 40% efficiency gain from prompt rewriting | One-off | Optimizes grading prompt based on empirical outcomes |

### 18.3 State Files and Schemas

All agent-writable state files should use JSON with explicit schemas to prevent accidental overwrites:

| File | Format | Writer | Reader | Schema |
|---|---|---|---|---|
| `technique_map.json` | JSON | Initializer, Research agent | All agents | `{techniques: [{name, status, parent, evidence, bpb}]}` |
| `research_charter.json` | JSON | Initializer, Charter updater | Research agent | `{dimensions: [{name, status, known_techniques, open_questions, max_entries}]}` |
| `strategy.json` | JSON | Reflection cycle | Experiment agent | `{entries: [{technique, rationale, priority, status}], updated_at}` |
| `decision_state.json` | JSON | Orchestrator | Experiment agent | `{best_bpb, last_experiments, unacked_findings, dead_ends, budget, learned_rules}` |
| `hypotheses.jsonl` | JSONL | Experiment agent | Evaluator, Reflection | `{id, technique, prediction, basis, scale_risk, outcome, learned_rule}` |
| `trace.jsonl` | JSONL | All agents | Dashboard, Diagnostics | `{timestamp, stage, item_id, scores, tokens_used, decision}` |
| `research_results.jsonl` | JSONL | Research agent | Experiment agent | `{timestamp, message, tier, score, source, claims}` |
| `results.tsv` | TSV | Orchestrator | All agents | Experiment audit trail |

### 18.4 Tool Catalog

Flat schemas with training-aligned naming (ForgeCode evidence: tool naming is a reliability variable):

**Research tools:**

| Tool | Parameters | Returns |
|---|---|---|
| `search_papers` | `query: str, source: str, max_results: int` | `[{title, url, abstract, authority_tier}]` |
| `grade_item` | `title: str, abstract: str, claims: [{type, metric, value}]` | `{score: int, tier: str, dimensions: {}}` |
| `verify_item` | `item_id: str, full_content: str` | `{verified: bool, evidence: str, citations: int}` |
| `get_forward_citations` | `paper_url: str` | `{citation_count: int, citing_papers: [str]}` |
| `extract_claims` | `text: str` | `[{claim_type, metric, value, context}]` |

**Experiment tools:**

| Tool | Parameters | Returns |
|---|---|---|
| `record_hypothesis` | `technique: str, prediction: str, basis: str, scale_risk: str` | `{id: str}` |
| `resolve_hypothesis` | `id: str, outcome: str, learned_rule: str` | `{updated: bool}` |
| `run_local_experiment` | `script: str, changes: str` | `{val_bpb: float, steps: int, time_s: float}` |
| `promote_to_h100` | `hypothesis_id: str, script: str` | `{queued: bool, estimated_cost: float}` |
| `ack_research` | `result_line: int, action: str` | `{acked: bool}` |

**Evaluation tools:**

| Tool | Parameters | Returns |
|---|---|---|
| `grade_experiment` | `hypothesis_id: str, predicted_bpb: float, actual_bpb: float, artifact_kb: int` | `{verdict: str, score: float, critique: str}` |
| `check_scale_transfer` | `technique: str, local_bpb: float, local_steps: int` | `{risk: str, confidence: float, similar_failures: [str]}` |
| `check_contamination` | `script_path: str` | `{clean: bool, issues: [str]}` |

**Infrastructure tools:**

| Tool | Parameters | Returns |
|---|---|---|
| `get_budget` | (none) | `{spent: float, remaining: float, best_bpb: float, rate_limited: bool}` |
| `get_technique_map` | `status_filter: str` | `[{name, status, evidence, bpb}]` |
| `get_decision_state` | (none) | Full decision state JSON |
| `emit_trace` | `stage: str, item_id: str, decision: str, tokens: int` | `{logged: bool}` |

### 18.5 Evaluator Agent Design

The evaluator is a separate agent (not a function call within the experiment agent) that grades each experiment result independently:

**Grading criteria** (5 dimensions, /20):

| Dimension | Range | Weight | What It Measures |
|---|---|---|---|
| `hypothesis_quality` | 0-4 | 1.0 | Was the hypothesis specific, testable, and grounded in evidence? |
| `implementation_fidelity` | 0-4 | 1.0 | Did the code change match the hypothesis? (AST diff analysis) |
| `metric_movement` | 0-4 | 1.5 | Did val_bpb improve, and by how much relative to prediction? |
| `constraint_compliance` | 0-4 | 1.0 | Artifact size, training time, memory usage within limits? |
| `transferability_signal` | 0-4 | 0.5 | Does the local result predict H100 behavior? (based on technique category) |

**Hard thresholds** (any of these blocks promotion regardless of total score):
- `constraint_compliance < 2` -> block (artifact or time violation)
- `hypothesis_quality == 0` -> block (no hypothesis recorded)
- `metric_movement == 0 AND implementation_fidelity < 2` -> block (no improvement and questionable implementation)

**Calibration strategy**: After 20 experiments, compute correlation between evaluator scores and actual H100 outcomes. If correlation < 0.6, trigger the self-improving prompt pass on the evaluator's grading criteria.

### 18.6 Doom Loop Detection Rules

| Rule | Trigger | Action |
|---|---|---|
| `technique_repetition` | Same technique keyword appears in 3+ consecutive hypotheses | Inject: "You have tried {technique} 3 times. The technique map shows it as {status}. Try a different dimension." |
| `scale_transfer_blindness` | 2+ H100 regressions from techniques that passed local validation | Inject: "Your last 2 H100 promotions regressed. Review scale-transfer risk before promoting again." |
| `budget_drain` | 3+ consecutive H100 runs with no improvement | Inject: "You have spent ${cost} on 3 consecutive non-improving runs. Switch to local-only experimentation until you find a strong signal." |
| `hypothesis_free` | Experiment started without `record_hypothesis` call | Block: "You must record a hypothesis before running an experiment." |

### 18.7 Success Metrics

| Metric | Baseline (from Section 7) | Target | How to Measure |
|---|---|---|---|
| Wasted H100 runs (regressions) | 44% (4/9) | < 20% | Count H100 runs where val_bpb worsened |
| Misleading local signals | 26% of spend ($20.88) | < 10% of spend | Track techniques that pass local but fail H100 |
| Research duplication | 60% redundancy (149 entries, ~60 unique) | < 20% redundancy | Jaccard similarity check on new entries |
| Hypothesis coverage | 0% (no tracking existed) | 100% | Every experiment must have a recorded hypothesis |
| Learned rules accumulated | 0 | 5+ per 20 experiments | Count resolved hypotheses with learned rules |
| Experiments per dollar | ~0.23 ($79.83 / 18 experiments) | > 0.5 | Total experiments / total spend |
| Time to new best | ~6 experiments between improvements | < 4 experiments | Count experiments between each new best val_bpb |

---

## Sources and References

- Karpathy's autoresearch: https://github.com/karpathy/autoresearch
- Sakana AI Scientist: https://github.com/SakanaAI/AI-Scientist
- OpenAI Parameter Golf: https://github.com/openai/parameter-golf
- Anthropic harness design (long-running apps): https://www.anthropic.com/engineering/harness-design-long-running-apps
- Anthropic effective harnesses for long-running agents: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- ForgeCode benchmarks analysis: https://forgecode.dev/blog/benchmarks-dont-matter/
- ForgeCode GPT 5.4 agent improvements: https://forgecode.dev/blog/gpt-5-4-agent-improvements/
- Reflexion (Shinn et al., 2023): https://arxiv.org/abs/2303.11366
- Google multi-agent scaling research (2025)
- Meta-Harness (Stanford/MIT, 2026)
- METR time-horizon benchmark
- ForgeCode open-source repository: https://github.com/antinomyhq/forge
- Multi-agent fact-checking with KG-structured verification: https://www.nature.com/articles/s41598-026-41862-z
- Agentic RAG hallucination reduction: https://arxiv.org/html/2603.00267v1
- Agentic RAG enterprise patterns: https://arxiv.org/abs/2603.01486
- Anthropic multi-agent research system: https://www.anthropic.com/engineering/multi-agent-research-system
