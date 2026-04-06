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
