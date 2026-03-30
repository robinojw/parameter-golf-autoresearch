# Parameter Golf Autoresearch

<img width="1280" height="1391" alt="cart" src="https://github.com/user-attachments/assets/3fbba136-aa62-495a-85e7-80133bdcd460" />


---

An autonomous experiment loop for [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) challenge, expanded from the brilliance of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

---

[Parameter Golf](https://github.com/openai/parameter-golf) asks you to train the best language model you can under simultaneous hard constraints: the entire artifact — code plus compressed weights — must fit in 16MB, training gets 10 minutes on 8x H100 SXMs, and the model can't phone home during evaluation. The metric is bits per byte on FineWeb. It's a compression problem dressed as an ML competition, and the leaderboard moves fast — the current SOTA (1.1194 bpb) combines int6 quantization, parameter banking, test-time training, and a custom bigram tokenizer, all inside a 16MB envelope. Staying competitive means tracking what others are shipping and testing new ideas quickly.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) demonstrated that an agent can run this kind of experiment loop autonomously: modify a training script, train, evaluate, keep or revert, repeat. The core loop is simple and it works — about 12 experiments per hour on a single GPU, designed to run overnight. I wanted to apply that same idea to Parameter Golf, but the challenge adds constraints that autoresearch wasn't designed for. The official runs require 8x H100s at ~$20/hour, so you can't just loop freely — you need cost-aware gating between cheap local validation and expensive official runs. The competition is also a moving target; an agent that only sees its own code will miss techniques that other competitors publish mid-challenge. And the artifact size constraint needs continuous enforcement, not just loss optimization.

This project keeps autoresearch's modify-train-evaluate-decide loop at the center, then splits the intelligence into two independent agents: an experiment agent that designs and runs hypotheses, and a research agent that continuously discovers and synthesizes relevant work from 10 sources. They communicate via shared files and are managed by a thin process supervisor. Around them: a two-tier compute model that uses local MLX runs as a free scratchpad before promoting to RunPod, deterministic hard gates for constraints and contamination detection, a budget manager with atexit pod termination as a safety net, and an adaptive intelligence layer that synthesizes experiment history into strategy, gates promotion on a dynamic threshold, and tests hypotheses in tournament brackets. MLX is your scratchpad; RunPod is your printer.

## Architecture

The system runs as three processes: a thin process supervisor and two independent Claude Code (Opus 4.6) agents — one for experiments, one for research. The agents communicate via shared JSONL files and the supervisor manages infrastructure lifecycle.

```mermaid
graph TB
    subgraph "Process Supervisor (orchestrate.py)"
        ORC["orchestrate.py\nno LLM — process management only"]
        BUD["BudgetManager\nreserve floor + 1hr rate limit"]
        THR["Dynamic Threshold\nscales with SOTA distance"]
        ORC --> BUD
        ORC --> THR
    end

    subgraph "Experiment Agent (Opus 4.6)"
        EXP["hypothesis design\nlocal MLX experiments\ntournament + promotion"]
        CON["Constraint Calculator\nfeasibility before code"]
        CTM["Contamination Check\nAST + score plausibility"]
        CRT["Critic Gate\ndeterministic + LLM checks"]
        EXP --> CON
        EXP --> CTM
        EXP --> CRT
    end

    subgraph "Research Agent (Opus 4.6)"
        RES["autonomous + reactive research\n10 sources, grading pipeline\ncompetitive intelligence"]
    end

    subgraph "Shared Files"
        RQ["research_queue.jsonl\n(experiment → research)"]
        RR["research_results.jsonl\n(research → experiment)"]
        PQ["promotion_queue.jsonl\n(experiment → supervisor)"]
        PM["program.md\n(research → experiment)"]
        TSV["results.tsv\n(experiment → research)"]
    end

    subgraph "Tier 1 - Local Mac (free)"
        MLX["train_gpt_mlx.py\nMLX / Apple Silicon"]
    end

    subgraph "Tier 2 - ~$3.50 per run"
        POD["8xH100 SXM Pod\ntorchrun, 600s max"]
    end

    EXP --> RQ
    EXP --> PQ
    EXP --> MLX
    RES --> RR
    RES --> PM
    RR --> EXP
    PM --> EXP
    TSV --> RES
    PQ --> ORC
    ORC -- "MCP create/delete\nSSH execute" --> POD
    POD --> TSV
```

## How It Works

### Dual-Agent Design

The experiment agent and research agent run as independent Claude Code instances, each with full autonomy over their domain. The experiment agent designs hypotheses, implements them, and runs local experiments. The research agent continuously discovers, grades, and synthesizes research from 10 sources. Neither blocks the other.

When the experiment agent needs targeted research — e.g., it hit an entropy floor with ternary quantization and needs alternatives — it writes a natural language request to `research_queue.jsonl`. The research agent picks it up, searches relevant sources, runs the grading pipeline, and writes findings to `research_results.jsonl`. The experiment agent checks for fresh results before designing its next hypothesis and decides autonomously whether to wait or proceed.

The process supervisor (`orchestrate.py`) has no LLM logic. It spawns both agents, monitors their health (restarting on crash, up to 5 attempts), and polls `promotion_queue.jsonl` for Tier 2 promotion requests. When a promotion arrives, the supervisor handles the full RunPod lifecycle: budget check, threshold check, pod creation via API, code sync and training execution via SSH, result retrieval, and pod termination.

### Hard Gates

Every hypothesis must pass deterministic checks before any code is written or any compute is spent. These are code-level gates the agents cannot override:

1. **Constraint check** — artifact size, training steps, quantization MSE, entropy bounds, and memory footprint must all pass (`compute/constraints.py`)
2. **Contamination check** — AST analysis verifies no validation data is referenced in training loops, and score plausibility checks flag suspiciously disproportionate val improvements (`compute/contamination.py`)
3. **Critic gate** — artifact size, diff size, and similarity to past failures (`research/critic.py`)
4. **Promotion threshold** — dynamic threshold that starts high and narrows as local evidence builds; adaptive fallback relaxes after 10 consecutive rejections (`compute/threshold.py`)
5. **Budget check** — reserve floor and rate limiting block Tier 2 submissions (`compute/budget.py`)

### Competitive Intelligence

The research agent monitors the Parameter Golf leaderboard and competitor repos (`openai/parameter-golf`, `KellerJordan/modded-nanogpt`, `karpathy/autoresearch`). When SOTA moves, it updates the target in `program.md`, signals the experiment agent, and investigates the technique. Any technique extracted from a competitor must pass the full constraint validation suite before being suggested — and accepted leaderboard submissions are prioritized over unverified code as known-legal.

### Tournament Mode

For structured hypothesis testing, tournament mode (`python orchestrate.py --tournament`) generates 4 candidate modifications via LLM, runs each for 100 iterations in an elimination round, advances the top 2 to a full 500-iteration run, and reports the winner.

### Budget and Pod Safety

`BudgetManager.can_submit()` blocks Tier 2 submissions below the reserve floor, with a one-hour rate limit on consecutive runs. `RunPodClient._cleanup_all` registers as both `atexit` and `SIGTERM` handler — if the supervisor dies, active pods get terminated. At $20/hour for 8xH100s, a forgotten pod costs ~$0.33/minute.

## Research Pipeline

The research agent drives its own research cadence — no fixed timers. It decides what's stale and where to focus based on experiment results, competitive landscape changes, and source yield. It also responds to targeted requests from the experiment agent via `research_queue.jsonl`.

```mermaid
graph LR
    subgraph "Fast sources"
        D["GitHub PRs\n+ Code Search"]
        F["Tavily\nscheduled + on-demand"]
    end

    subgraph "Slow sources"
        A["ArXiv"]
        B["OpenReview"]
        C["Semantic Scholar"]
        E["RSS feeds\nCodeSOTA"]
    end

    subgraph "Experiment Agent signals"
        REQ["research_queue.jsonl\ntargeted requests"]
        RES["results.tsv\nexperiment outcomes"]
    end

    D & F --> FAST["fetch_fast()"]
    A & B & C & E --> SLOW["fetch_slow()"]
    FAST & SLOW --> CACHE["raw_cache.jsonl"]
    CACHE --> GRADE["grade_items()\nscore /15 across 5 dimensions"]
    GRADE --> GCACHE["graded_cache.jsonl"]
    GCACHE --> VERIFY["verify top Tier A\nfull content + web evidence"]
    VERIFY --> REFLECT["reflect()\nsynthesize strategy\nupdate technique map"]
    REFLECT --> INJECT["inject into program.md\n+ research_results.jsonl"]
    REQ --> |"triggers targeted search"| FAST
    RES --> |"adapts search strategy"| REFLECT
```

Each paper gets scored across five dimensions: `bpb_impact`, `size_compatibility`, `time_compatibility`, `implementability`, and `novelty`. The grader knows the current SOTA, the techniques already on the leaderboard, and the hard artifact and training constraints. A paper that requires a new pip dependency, pushes the 16MB limit, or would exceed 600s of training time scores low regardless of the idea's quality. The top 12 scored items, by default, get injected into `program.md`.

After grading and verification, a **reflection cycle** synthesizes experiment history into strategic guidance — identifying failure patterns, exhausted vs. promising search dimensions, and recommending next experiments. The output is written to `strategy.md` and injected into `program.md` so the agent sees synthesized strategic state alongside raw data.

The reflection also maintains a **technique adjacency map** (`technique_map.json`) — a graph of technique relationships with status labels (proven, exploring, dead_end, untried). This gives the agent a structured view of the search space: which branches are dead ends, which show monotonic improvement, and which remain unexplored.

## Constraint Calculator

Before writing code for a new experiment, the agent validates mathematical feasibility:

```bash
python orchestrate.py --check-constraints --params 23000000 --bits 6 --code-bytes 30000
```

This checks:
- **Artifact size** — will N parameters at B bits fit in 16MB after zstd compression?
- **Training steps** — how many steps fit in 600s at a given batch size?
- **Quantization MSE** — what's the theoretical noise floor at this bit-width?
- **Entropy bound** — can zstd physically compress these weights below 16MB?
- **Memory footprint** — will model weights + optimizer state + gradients + activations fit in H100 VRAM (80GB)?

The calculator auto-calibrates from weight files on disk when available, using observed compression ratios and weight statistics instead of theoretical defaults. If the report says NOT FEASIBLE, the idea is mathematically doomed and the agent redesigns before wasting a training run.

A separate **contamination detection** module (`compute/contamination.py`) provides deterministic TTT leakage detection. It parses training scripts via AST to find validation data references in training loops, and checks whether val_bpb improvements are plausibly explained by training loss changes. These are hard gates — a failed contamination check blocks the experiment.

## Repository Structure

```
parameter-golf-autoresearch/
├── orchestrate.py          # process supervisor: spawn agents, monitor health, RunPod lifecycle
├── agents/
│   ├── experiment_agent.md # system prompt for the experiment agent (Opus 4.6)
│   ├── research_agent.md   # system prompt for the research agent (Opus 4.6)
│   └── shared.py           # shared communication: Message dataclass, JSONL queue read/write
├── program.md              # agent working context: research, strategy, technique map, experiments
├── train_gpt_mlx.py        # Tier 1 training script (MLX, Apple Silicon)
├── train_gpt.py            # Tier 2 training script (PyTorch, torchrun)
├── measure_artifact.py     # artifact size check: code + zstd-compressed weights, <=16MB
├── compute/
│   ├── budget.py           # BudgetManager: spend tracking, reserve floor, rate limiting
│   ├── threshold.py        # dynamic promotion threshold: scales with SOTA distance
│   ├── constraints.py      # mathematical feasibility: artifact, steps, MSE, entropy, memory
│   ├── contamination.py    # TTT leakage detection: AST analysis + score plausibility
│   ├── tournament.py       # tournament mode: generate -> eliminate -> finalize
│   ├── runpod_client.py    # pod lifecycle: launch, poll, terminate, atexit cleanup
│   └── sync.py             # rsync push/pull and remote torchrun over SSH
├── research/
│   ├── fetch.py            # async fetch from 10 sources (agent-driven cadence)
│   ├── grade.py            # LLM grading in batches of 10, five-dimension scoring
│   ├── verify.py           # deep verification of Tier A items with full content + web evidence
│   ├── reflect.py          # reflection cycle: strategy synthesis + technique map maintenance
│   ├── critic.py           # pre-commit gate: artifact size, diff size, similarity, LLM review
│   ├── inject.py           # section injection into program.md + research_results.jsonl
│   ├── experiments.py      # read-only API for results.tsv, competitor data, source yield
│   └── sources/            # one module per source (arxiv, openreview, semantic_scholar, ...)
├── tests/                  # pytest suite (110 tests)
├── data/                   # FineWeb token data for training
├── runpod_results/         # logs pulled from completed RunPod runs
├── results.tsv             # experiment history: commit, tier, val_bpb, cost, source_item
├── research_queue.jsonl    # experiment agent → research agent requests
├── research_results.jsonl  # research agent → experiment agent findings
├── promotion_queue.jsonl   # experiment agent → supervisor promotion requests
├── budget.json             # persisted spend state
├── strategy.md             # synthesized strategic guidance (generated by reflection)
├── technique_map.json      # technique relationship graph (generated by reflection)
├── raw_cache.jsonl         # fetched research items
└── graded_cache.jsonl      # scored research items
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/you/parameter-golf-autoresearch
   cd parameter-golf-autoresearch
   pip install -e ".[dev]"
   ```

2. Copy the env template and fill in the required keys:
   ```bash
   cp .env.example .env
   # minimum required: RUNPOD_API_KEY, RUNPOD_TEMPLATE_ID, GITHUB_TOKEN
   ```

3. Download FineWeb token data into `data/` per the challenge instructions.

4. Run a baseline MLX smoke test:
   ```bash
   RUN_ID=local_baseline ITERATIONS=500 TRAIN_SEQ_LEN=512 python3 train_gpt_mlx.py > run.log 2>&1
   grep "^val_bpb:" run.log
   ```

5. Start the supervisor (spawns both agents):
   ```bash
   python orchestrate.py
   ```
   This launches the experiment agent and research agent as independent Claude Code instances, monitors their health, and polls for RunPod promotion requests.

## Usage

```bash
# Start the dual-agent supervisor (default)
python orchestrate.py

# Manual research refresh (useful outside the agent loop)
python orchestrate.py --refresh            # full refresh: all sources + grade + verify + reflect
python orchestrate.py --refresh-fast       # fast refresh: GitHub + Tavily only

# Check feasibility before writing code
python orchestrate.py --check-constraints --params 23000000 --bits 6
python orchestrate.py --check-constraints --params 50000000 --bits 4 --code-bytes 40000

# Pre-commit critic check
python orchestrate.py --critique

# Tournament hypothesis testing
python orchestrate.py --tournament
python orchestrate.py --tournament --prompt "focus on test-time training" --candidates 6

# Promote to RunPod
python orchestrate.py --promote <commit_hash> --dry-run
python orchestrate.py --promote <commit_hash>

# Status
python orchestrate.py --budget-status
python orchestrate.py --threshold-status
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `RUNPOD_API_KEY` | Yes | | RunPod compute layer |
| `RUNPOD_TEMPLATE_ID` | Yes | `y5cejece4j` | Official Parameter Golf RunPod template |
| `GITHUB_TOKEN` | Yes | | GitHub PR and commit fetching |
| `GRADING_HARNESS` | No | `auto` | Which coding agent grades research: `auto`, `opencode`, or `claude` |
| `TAVILY_API_KEY` | No | | Web search and extract; disables Tavily sources if unset |
| `TOTAL_COMPUTE_CREDITS` | No | `500` | Starting credit balance for spend tracking |
| `RUNPOD_MIN_RESERVE` | No | `50` | Hard floor, Tier 2 submissions blocked below this |
| `TOP_N_INJECT` | No | `12` | Max research items injected into program.md |
| `SINCE_HOURS` | No | `48` | Lookback window for new papers and posts |
| `PROMOTION_FALLBACK_WINDOW` | No | `10` | Consecutive local runs before adaptive threshold relaxation |
| `S2_API_KEY` | No | | Semantic Scholar API key for higher rate limits |
| `TAVILY_MONTHLY_BUDGET_USD` | No | `5.00` | Soft cap on Tavily spend |

## Cost Model

Two cost surfaces matter: RunPod compute and Tavily search. Research grading runs through your existing coding agent (opencode or claude code), so it costs whatever your configured model costs per token — no separate API key needed.

RunPod dominates. At roughly $2.50/hour per GPU, the 10-minute training window costs around $3.33 at minimum. Pod startup and sync overhead bring the real number closer to $3.50. The budget manager calculates cost from actual wall-clock duration: `(seconds / 3600) * gpu_count * (hourly_rate / 8)`. With `TOTAL_COMPUTE_CREDITS=500` and `RUNPOD_MIN_RESERVE=50`, you have roughly 128 Tier 2 runs before hitting the reserve floor. The one-run-per-hour rate limit means a continuous session can't burn through that in less than five days.

Research grading is handled by the research agent as part of its continuous operation. Each batch of 10 items goes through the LLM grading pipeline. The agent decides its own cadence — there are no fixed refresh timers.

Tavily adds up if the research agent fires many ad-hoc queries in response to experiment requests. `TAVILY_MONTHLY_BUDGET_USD` is a soft reminder, not a hard cap.

## The Systems Problem

The challenge presents itself as an ML optimization problem, but most of the interesting engineering lives in the infrastructure around the model. The metric is `val_bpb` and the model changes happen in two Python files, but the harder problems are: how do you prevent a hung pod from draining your account, how do you route research signal into an agent's context without overwhelming it, how do you decide which local improvements are worth paying $3.50 to validate, and how do you maintain a clean audit trail across hundreds of experiments on two different hardware targets?

The answers here are deliberately unclever. Two independent agents keep research and experimentation from blocking each other. `atexit` handles the pod lifecycle. A five-dimension LLM grader handles the research signal, driven by the research agent's own judgment of what's stale and what's productive. Deterministic constraint and contamination checks catch doomed ideas and data leakage before code is written. A dynamic threshold that scales with distance from SOTA handles the promotion gate. A periodic reflection cycle synthesizes strategy from raw experiment history. A critic catches repeated mistakes before they waste a training run. A tournament tests multiple hypotheses instead of betting on the first plausible idea. `results.tsv` handles the audit trail. None of it is surprising, but it's the kind of plumbing that needs to exist before you can run experiments reliably at any volume. The model is rarely the hard part; the system around it is.
