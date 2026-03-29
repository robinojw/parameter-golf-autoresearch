# Parameter Golf Autoresearch

<img width="1280" height="1391" alt="cart" src="https://github.com/user-attachments/assets/3fbba136-aa62-495a-85e7-80133bdcd460" />


---

An autonomous experiment loop for [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) challenge, expanded from the brilliance of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

---

[Parameter Golf](https://github.com/openai/parameter-golf) asks you to train the best language model you can under simultaneous hard constraints: the entire artifact — code plus compressed weights — must fit in 16MB, training gets 10 minutes on 8x H100 SXMs, and the model can't phone home during evaluation. The metric is bits per byte on FineWeb. It's a compression problem dressed as an ML competition, and the leaderboard moves fast — the current SOTA (1.1194 bpb) combines int6 quantization, parameter banking, test-time training, and a custom bigram tokenizer, all inside a 16MB envelope. Staying competitive means tracking what others are shipping and testing new ideas quickly.

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) demonstrated that an agent can run this kind of experiment loop autonomously: modify a training script, train, evaluate, keep or revert, repeat. The core loop is simple and it works — about 12 experiments per hour on a single GPU, designed to run overnight. I wanted to apply that same idea to Parameter Golf, but the challenge adds constraints that autoresearch wasn't designed for. The official runs require 8x H100s at ~$20/hour, so you can't just loop freely — you need cost-aware gating between cheap local validation and expensive official runs. The competition is also a moving target; an agent that only sees its own code will miss techniques that other competitors publish mid-challenge. And the artifact size constraint needs continuous enforcement, not just loss optimization.

This project keeps autoresearch's modify-train-evaluate-decide loop at the center, then adds the infrastructure around it: a two-tier compute model that uses local MLX runs as a free scratchpad before promoting to RunPod, a research pipeline that ingests papers and competitor activity from 10 sources with tiered refresh cadences, a budget manager that enforces hard spend caps with atexit pod termination as a safety net, and an adaptive intelligence layer that synthesizes experiment history into strategy, validates ideas against mathematical constraints before code is written, gates promotion on a dynamic threshold, and tests hypotheses in tournament brackets. MLX is your scratchpad; RunPod is your printer.

## Architecture

```mermaid
graph LR
    subgraph "Tier 1 - Local Mac (free)"
        MLX["train_gpt_mlx.py\nMLX / Apple Silicon"]
        LOG["run.log\nval_bpb signal"]
        MLX --> LOG
    end

    subgraph "Orchestration"
        ORC["orchestrate.py"]
        BUD["BudgetManager\nreserve floor + 1hr rate limit"]
        THR["Dynamic Threshold\nscales with SOTA distance"]
        CRT["Critic Gate\ndeterministic + LLM checks"]
        CON["Constraint Calculator\nfeasibility before code"]
        ORC --> BUD
        ORC --> THR
        ORC --> CRT
        ORC --> CON
    end

    subgraph "Tier 2 - ~$3.50 per run"
        POD["8xH100 SXM Pod"]
        TRAIN["torchrun train_gpt.py\n8 processes, 600s max"]
        TSV["results.tsv"]
        POD --> TRAIN --> TSV
    end

    LOG -- "passes dynamic\nthreshold" --> ORC
    BUD -- "can_submit = OK" --> POD
    TRAIN -. "atexit / SIGTERM" .-> TERM["terminate_pod()"]
```

## How It Works

The experiment loop is the central unit of work. Each cycle starts with a hypothesis and a mathematical feasibility check (`python orchestrate.py --check-constraints --params N --bits B`) that verifies artifact size, training step budget, quantization MSE floors, and entropy bounds before any code is written. If the math checks out, the agent implements the idea and runs a critic check (`python orchestrate.py --critique`) that validates artifact size, diff size, and similarity to past failures, followed by a 500-iteration MLX smoke run.

If local `val_bpb` passes the dynamic promotion threshold — which scales based on distance from SOTA, requiring larger improvements when far away and accepting smaller gains near the frontier — the commit qualifies for promotion. The agent calls `python orchestrate.py --promote <commit_hash>`, which triggers the full Tier 2 flow: budget check, threshold check, pod launch, rsync of `train_gpt.py` and data, `torchrun` across 8 GPUs, log retrieval, pod termination, and an append to `results.tsv`.

If no experiments pass the threshold after 10 consecutive runs, an adaptive fallback relaxes it to accept the best improvement observed in that window. This prevents permanent stalls near SOTA.

The orchestrator never blocks the experiment loop. After queuing a promotion, the agent continues Tier 1 experiments and checks `results.tsv` periodically. If a RunPod result confirms the improvement, the branch advances. If it's worse than the MLX signal predicted, the agent investigates the PyTorch translation.

For structured hypothesis testing, the **tournament mode** (`python orchestrate.py --tournament`) generates 4 candidate modifications via LLM, runs each for 100 iterations in an elimination round, advances the top 2 to a full 500-iteration run, and reports the winner. Runs are sequential by default to avoid MLX resource contention on Apple Silicon, with a configurable cooldown between runs for thermal settling.

Budget enforcement operates on two axes. `BudgetManager.can_submit()` blocks any Tier 2 submission if the remaining balance is below the configured reserve floor. A separate one-hour rate limit prevents back-to-back submissions even when runs finish quickly. Both checks persist across process restarts in `budget.json`.

Pod lifecycle safety works via two mechanisms. `RunPodClient._cleanup_all` registers as both an `atexit` handler and a `SIGTERM` handler at construction time. If the orchestrator dies for any reason, any active pod gets terminated. At $20/hour for 8xH100s, a hung pod costs roughly $0.33/minute to forget about.

## Research Pipeline

The research pipeline runs on a tiered schedule. Fast-moving sources (GitHub PRs, code search, Tavily web results) are checked every 2 hours to catch competitor moves quickly. Slow-moving sources (ArXiv, OpenReview, Semantic Scholar, RSS feeds, CodeSOTA) are checked every 12 hours since they update daily at most. A breaking news daemon checks Tavily hourly for competition-specific announcements. The agent can also pull research on demand with `--refresh` or `--refresh-fast` at any time.

```mermaid
graph LR
    subgraph "Fast sources - every 2h"
        D["GitHub PRs\n+ Code Search"]
        F["Tavily\nscheduled"]
    end

    subgraph "Slow sources - every 12h"
        A["ArXiv"]
        B["OpenReview"]
        C["Semantic Scholar"]
        E["RSS feeds\nCodeSOTA"]
    end

    subgraph "Daemon - every 1h"
        G["Tavily\nbreaking news"]
    end

    D & F --> FAST["fetch_fast()"]
    A & B & C & E --> SLOW["fetch_slow()"]
    FAST & SLOW --> CACHE["raw_cache.jsonl"]
    G --> CACHE
    CACHE --> GRADE["grade_items()\nscore /15 across 5 dimensions"]
    GRADE --> GCACHE["graded_cache.jsonl"]
    GCACHE --> VERIFY["verify top Tier A\nfull content + web evidence"]
    VERIFY --> REFLECT["reflect()\nsynthesize strategy\nupdate technique map"]
    REFLECT --> INJECT["inject all sections\ninto program.md"]
    INJECT --> PROG["program.md"]
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

The calculator auto-calibrates from weight files on disk when available, using observed compression ratios and weight statistics instead of theoretical defaults. If the report says NOT FEASIBLE, the idea is mathematically doomed and the agent redesigns before wasting a training run.

## Repository Structure

```
parameter-golf-autoresearch/
├── orchestrate.py          # entry point: refresh, promotion, critic, tournament, constraints
├── program.md              # agent working context: research, strategy, technique map, experiments
├── train_gpt_mlx.py        # Tier 1 training script (MLX, Apple Silicon)
├── train_gpt.py            # Tier 2 training script (PyTorch, torchrun)
├── measure_artifact.py     # artifact size check: code + zstd-compressed weights, <=16MB
├── compute/
│   ├── budget.py           # BudgetManager: spend tracking, reserve floor, rate limiting
│   ├── threshold.py        # dynamic promotion threshold: scales with SOTA distance
│   ├── constraints.py      # mathematical feasibility: artifact, steps, MSE, entropy bounds
│   ├── tournament.py       # tournament mode: generate -> eliminate -> finalize
│   ├── runpod_client.py    # pod lifecycle: launch, poll, terminate, atexit cleanup
│   └── sync.py             # rsync push/pull and remote torchrun over SSH
├── research/
│   ├── fetch.py            # tiered async fetch: fast (2h) and slow (12h) source groups
│   ├── grade.py            # LLM grading in batches of 10, five-dimension scoring
│   ├── verify.py           # deep verification of Tier A items with full content + web evidence
│   ├── reflect.py          # reflection cycle: strategy synthesis + technique map maintenance
│   ├── critic.py           # pre-commit gate: artifact size, diff size, similarity, LLM review
│   ├── inject.py           # section injection into program.md
│   ├── experiments.py      # read-only API for results.tsv, competitor data, source yield
│   └── sources/            # one module per source (arxiv, openreview, semantic_scholar, ...)
├── tests/                  # pytest suite (81 tests)
├── data/                   # FineWeb token data for training
├── runpod_results/         # logs pulled from completed RunPod runs
├── results.tsv             # experiment history: commit, tier, val_bpb, cost, source_item
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

5. Start the orchestrator (tiered background loop):
   ```bash
   python orchestrate.py
   ```
   This runs fast sources every 2h, full refresh every 12h, and breaking news every 1h.

## Usage

```bash
# Background loop (fast refresh 2h, full refresh 12h, breaking news 1h)
python orchestrate.py

# On-demand research
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
| `FAST_REFRESH_HOURS` | No | `2` | Fast source (GitHub, Tavily) refresh cadence |
| `SLOW_REFRESH_HOURS` | No | `12` | Full source (academic + all) refresh cadence |
| `TOP_N_INJECT` | No | `12` | Max research items injected into program.md |
| `SINCE_HOURS` | No | `48` | Lookback window for new papers and posts |
| `PROMOTION_FALLBACK_WINDOW` | No | `10` | Consecutive local runs before adaptive threshold relaxation |
| `S2_API_KEY` | No | | Semantic Scholar API key for higher rate limits |
| `TAVILY_MONTHLY_BUDGET_USD` | No | `5.00` | Soft cap on Tavily spend |
| `TAVILY_HOURLY_BREAKING_NEWS` | No | `true` | Enable the hourly Tavily breaking news daemon |

## Cost Model

Two cost surfaces matter: RunPod compute and Tavily search. Research grading runs through your existing coding agent (opencode or claude code), so it costs whatever your configured model costs per token — no separate API key needed.

RunPod dominates. At roughly $2.50/hour per GPU, the 10-minute training window costs around $3.33 at minimum. Pod startup and sync overhead bring the real number closer to $3.50. The budget manager calculates cost from actual wall-clock duration: `(seconds / 3600) * gpu_count * (hourly_rate / 8)`. With `TOTAL_COMPUTE_CREDITS=500` and `RUNPOD_MIN_RESERVE=50`, you have roughly 128 Tier 2 runs before hitting the reserve floor. The one-run-per-hour rate limit means a continuous session can't burn through that in less than five days.

Research grading shells out to your coding agent in headless mode, so each batch of 10 items is one agent invocation. A full 12-hour refresh typically grades 20 to 50 items in 2-5 batches. Fast refreshes (every 2h) grade fewer items since they only pull from GitHub and Tavily.

Tavily adds up if you leave the breaking news daemon running continuously or fire many ad-hoc queries. `TAVILY_MONTHLY_BUDGET_USD` is a soft reminder, not a hard cap.

## The Systems Problem

The challenge presents itself as an ML optimization problem, but most of the interesting engineering lives in the infrastructure around the model. The metric is `val_bpb` and the model changes happen in two Python files, but the harder problems are: how do you prevent a hung pod from draining your account, how do you route research signal into an agent's context without overwhelming it, how do you decide which local improvements are worth paying $3.50 to validate, and how do you maintain a clean audit trail across hundreds of experiments on two different hardware targets?

The answers here are deliberately unclever. `atexit` handles the pod lifecycle. A five-dimension LLM grader handles the research signal, with tiered refresh cadences that match how fast each source actually updates. A constraint calculator catches mathematically doomed ideas before code is written. A dynamic threshold that scales with distance from SOTA handles the promotion gate. A periodic reflection cycle synthesizes strategy from raw experiment history. A critic catches repeated mistakes before they waste a training run. A tournament tests multiple hypotheses instead of betting on the first plausible idea. `results.tsv` handles the audit trail. None of it is surprising, but it's the kind of plumbing that needs to exist before you can run experiments reliably at any volume. The model is rarely the hard part; the system around it is.
