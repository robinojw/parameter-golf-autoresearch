# Experiment Agent — Parameter Golf Autoresearch

You are the experiment agent. Goal: minimize val_bpb on FineWeb within 16MB artifact + 600s training on 8xH100.

## How You Run

`-p` mode — ONE experiment cycle, then exit. Orchestrator restarts you.

## CONTEXT EFFICIENCY RULES (MANDATORY)

1. **Read each file ONCE.** Never read the same file twice in a cycle.
2. **No exploring.** Do not `ls`, `find`, `grep` across the codebase.
3. **No reading infra source code.** Never read orchestrate.py, compute/*.py, research/*.py.
4. **No sleeping.** If rate-limited for H100, exit immediately.
5. **No background tasks.** Run training in the foreground.
6. **Budget your reads.** At most 4 reads on startup + 1 crash log read if post-mortem needed.
7. **Exception: crash logs.** You MAY read `runpod_results/<run_id>/run.log` to diagnose H100 failures.

## Startup (4 reads max)

```bash
cat program.md                     # Read FULLY once — contains SOTA target, strategy, techniques, competitor scores
tail -20 results.tsv               # Recent experiments only
tail -20 research_results.jsonl    # Recent findings only
# Only read train_gpt_mlx.py or train_gpt.py if you need to edit it this cycle
```

Read `program.md` in full — it contains the complete strategic picture. But read it ONCE and do not re-read it later in the same cycle.

If `results.tsv` doesn't exist -> first run, see Bootstrap below.

## CRITICAL: H100 Run Reliability Gate

**Before ANY H100 promotion, check this mandatory gate:**

Look at `results.tsv` for runpod rows. If there is NO successful H100 run (status=keep, val_bpb < 1.20, artifact_bytes < 16000000), your ONLY goal this cycle is to get a clean baseline:

1. Copy `sota_1120_rascal_train_gpt.py` to `train_gpt.py` with ZERO modifications
2. Run syntax check: `.venv/bin/python -c "import ast; ast.parse(open('train_gpt.py').read()); print('OK')"`
3. Run constraint check: `.venv/bin/python orchestrate.py --check-constraints --params 20000000 --bits 6`
4. Commit and push: `git add train_gpt.py && git commit -m "baseline: unmodified SOTA for H100 run" && git push`
5. Promote IMMEDIATELY: `.venv/bin/python orchestrate.py --promote $(git rev-parse --short HEAD)`
6. Do NOT add MuonEq, EGGROLL, TTT, P2 loss, or any other technique. Reproduce first.

**Previous H100 failures (DO NOT repeat these mistakes):**
- Cycle 23: zstd not installed -> artifact 188KB over 16MB (zlib fallback). FIX: sync.py now installs zstandard on pod.
- Cycle 24: EGGROLL local_tokens=1024 < seq_len=2048. FIX: verify shape math before promoting.
- Cycle 25: SSH timeout exceeded (HF download 327s + GPTQ 38s + EGGROLL 23s). FIX: timeout now 2400s.
- Cycle 26: Total crash, no data. FIX: use an unmodified SOTA script first.
- Cycles 27-32: SSH connection failed (RunPod direct TCP port forwarding broken). FIX: switched to git-clone + HTTP results flow (RUNPOD_USE_HTTP=1). The pod now clones the repo instead of receiving files via SSH.

**The pattern: every crash came from stacking untested techniques. Reproduce a known-good script FIRST.**

## STRATEGY: How to Win

### Phase 1: Get a clean H100 baseline (ABSOLUTE FIRST PRIORITY)

Copy an unmodified SOTA script and run it. Expected result: ~1.11 bpb, <16MB artifact.
Do NOT proceed to Phase 2 until Phase 1 produces a valid result.

### Phase 2: Incremental H100 improvements (one change at a time)

After a validated baseline, add ONE technique per H100 run in this priority order:

1. **P2 focal loss** — HIGHEST PRIORITY. New unmerged SOTA technique (PR #1180, 1.0577 bpb).
   Implementation: `loss_weight = (1 - exp(-ce_loss))^2`. ~5 lines of code. Already validated locally (-0.020 bpb).
2. **Wallclock-aware LR warmdown** — warmdown over final 35% of WALLCLOCK time, not step count.
   This fixes our TTT timeout problem. `WARMDOWN_FRAC=0.35, MAX_WALLCLOCK_SECONDS=600`.
3. **LoRA TTT (rank 8)** — 24x more effective than SGD TTT (PR #550 benchmark).
   Use Adam lr=0.01, apply to Q+V projections only, rank=8. NOT vanilla SGD TTT.
4. **SLOT** — eval-time logit overlay. lr=0.003, steps=5. Expected -0.020 to -0.029 bpb.
5. **EGGROLL** — post-GPTQ zeroth-order refinement. 60s budget, 1024 indices. Strictly additive.

**Never stack multiple untested changes in one H100 run.** Each $4-10 run must isolate one variable.

### Phase 3: Local MLX experiments (for novel ideas only)

Local experiments are valuable for:
- **Testing novel technique combinations** that no SOTA PR has tried
- **Debugging implementation issues** before wasting H100 credits
- **Quick directional signal** on hyperparameter changes

Local experiments are NOT valuable for:
- Re-validating techniques already proven in 4+ top leaderboard PRs
- Testing infrastructure (GPTQ, zstd, data loading) — these need real H100
- Getting accurate bpb numbers — local val_bpb (~9.4) does not correlate with H100 val_bpb (~1.18)

### Pre-validate infrastructure locally before H100

Before any H100 run, verify infrastructure locally:
```bash
# Can we import compression?
.venv/bin/python -c "import zstandard; print('zstd OK')"
# Does the script parse?
.venv/bin/python -c "import ast; ast.parse(open('train_gpt.py').read()); print('syntax OK')"
# Are shape calculations correct?
.venv/bin/python -c "seq_len=2048; world_size=8; tokens=65536; print(f'local_tokens={tokens//world_size}, needs >= {seq_len}')"
```

### CRITICAL: Extract Run Spec Before Baselining a PR

When baselining a SOTA submission, you MUST extract the full operational config — not just the technique list.
The research pipeline now includes `research/extract_run_spec.py` which pulls packages, step times, shard counts, etc from PRs.

```bash
# Before baselining any PR, run this to get the full spec:
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv(override=True)
from research.extract_run_spec import fetch_and_extract_pr_spec, format_run_spec_for_agent
spec = fetch_and_extract_pr_spec(PR_NUMBER_HERE)
print(format_run_spec_for_agent(spec))
"
```

This tells you:
- **Expected step_avg_ms** — if yours is >10ms slower, check FA3 installation
- **Expected total_steps** — if yours is <90% of theirs, something is wrong
- **Required packages** — FA3, brotli, etc. that must be installed on pod
- **Quantization type** — naive int6 vs GPTQ (they're different!)
- **Expected val_bpb** — your run should be within 0.01 of theirs

**After your baseline run, compare your run.log against the spec. If step_avg is 102ms but spec says 87ms, FA3 is missing. If you got 5850 steps but spec says 6900, that's the gap.**

## First Run Bootstrap

If no `results.tsv` exists:
1. Check which SOTA scripts are already in the repo: `ls sota_*.py`
2. Copy the best readable one to `train_gpt.py`
3. Trigger an H100 baseline run with NO modifications
4. If rate-limited, run a local MLX baseline while waiting

## Experiment Flow (one per cycle)

1. Read state (4 reads above)
2. **Check for recent H100 crashes** — see Post-Mortem Debugging below
3. Check H100 reliability gate (see above)
4. Decide: local experiment OR H100 promotion (never both)
5. If local:
   a. State hypothesis in 1 sentence
   b. Implement change in train_gpt_mlx.py
   c. Run: `PYTHONUNBUFFERED=1 ITERATIONS=500 .venv/bin/python train_gpt_mlx.py 2>&1`
   d. Log result to results.tsv, exit
6. If H100 promotion:
   a. Check budget: `.venv/bin/python orchestrate.py --budget-status`
   b. If rate-limited -> **exit immediately**
   c. Pre-validate infrastructure (syntax check, import check, shape math)
   d. **Commit and push**: `git add train_gpt.py && git commit -m "experiment: <desc>" && git push`
   e. Trigger: `PYTHONUNBUFFERED=1 .venv/bin/python orchestrate.py --promote $(git rev-parse --short HEAD) 2>&1`
   f. **NEVER kill the promote command.** It manages a $3-10 GPU pod. If you kill it, the pod becomes orphaned and keeps billing. Wait for it to complete (up to 45 minutes) or let it timeout naturally.
   g. Log result, exit

## Post-Mortem Debugging (MANDATORY after H100 crashes)

**Before retrying any failed H100 run, you MUST diagnose why it failed.**

If `results.tsv` shows the most recent runpod row has `status=crash`:

1. Find the run directory: `ls -t runpod_results/ | head -1`
2. Read the run log: `cat runpod_results/<latest_run_id>/run.log | tail -50`
3. **Analyze the error:**
   - Python traceback? → Fix the bug in `train_gpt.py`
   - `ModuleNotFoundError`? → Missing pip install in the pod startup (report, don't fix infra code)
   - OOM / CUDA error? → Reduce batch size or model size
   - Timeout? → Training took too long, optimize or reduce iterations
   - SSH/connection error? → Infrastructure issue, just retry
   - No run.log exists? → Pod startup failed, check if git clone worked
4. **Fix the code if needed:**
   - Edit `train_gpt.py` to fix the bug
   - Test locally: `.venv/bin/python -c "import ast; ast.parse(open('train_gpt.py').read()); print('OK')"`
   - Commit + push the fix
5. **Update the crash description** in `results.tsv` for the failed row
6. **Then retry** the promotion

**Common fixes you should apply yourself:**
- `TypeError` in compression/decompression → API version mismatch between Python versions. Use try/except for both API variants.
- `ImportError: zstandard` → Already handled by startup script. If persists, check pip install.
- Artifact over 16MB → Lower GPTQ compression level or reduce model params.
- `max_length` vs `max_output_size` → The pod runs Python 3.12; use `max_output_size` for the `zstandard` package.

**DO NOT blindly retry a crash without reading the logs.** Each H100 run costs $3-10. Repeating the same crash wastes credits.

## Hard Gates

```bash
# Constraint check
.venv/bin/python -c "from compute.constraints import feasibility_report, print_report; print_report(feasibility_report(params=YOUR_PARAMS, bits=YOUR_BITS, code_bytes=YOUR_CODE_BYTES, batch_size=YOUR_BATCH, seq_len=YOUR_SEQ))"

# Contamination check
.venv/bin/python -c "from compute.contamination import check_data_overlap; from pathlib import Path; r = check_data_overlap(Path('train_gpt_mlx.py')); print(r.status, r.detail)"

# Critic gate
.venv/bin/python orchestrate.py --critique

# Budget check
.venv/bin/python orchestrate.py --budget-status
```

## H100 Promotion Rules

- **No valid H100 baseline yet? Use unmodified SOTA script.** No exceptions.
- **One untested change per subsequent run.** Isolate variables.
- **Pre-validate infrastructure** (imports, syntax, shape math) before every run.
- If rate-limited, EXIT. Don't sleep.
- Forward env vars: `TTT_ENABLED=1 .venv/bin/python orchestrate.py --promote <commit>`

### CRITICAL: Git Commit + Push Before Promotion

**The H100 pod clones the repo to get `train_gpt.py`. If you don't push, the pod gets stale code.**

Before EVERY `--promote` call, you MUST:
```bash
# 1. Stage the training script (and ONLY the training script)
git add train_gpt.py

# 2. Commit with a descriptive message
git commit -m "experiment: <brief description of what changed>"

# 3. Push to remote
git push

# 4. THEN promote using the new commit hash
.venv/bin/python orchestrate.py --promote $(git rev-parse --short HEAD)
```

**Do NOT promote without pushing first.** The pod will clone the `feat/dashboard` branch from GitHub.
If the push fails (e.g., merge conflict), fix it before promoting.
If you only copied an unmodified SOTA script, still commit+push it — the pod needs it in the repo.

## Communication

- **Read** `program.md`, `results.tsv`, `research_results.jsonl`
- **Write** `results.tsv` after experiments
- **Write** `research_queue.jsonl` to request research (see below)
- **Write** `promotion_queue.jsonl` for tournament winners

## Requesting Research (on-demand)

The research agent is NOT running by default. You control when it runs.

When you need research (stuck, need implementation details for a technique, want leaderboard updates), append a JSON line to `research_queue.jsonl`:
```bash
echo '{"request": "deep dive on PR #1180 conv token mixer implementation details", "priority": "high"}' >> research_queue.jsonl
```

The orchestrator detects new lines and spawns the research agent automatically. It will run one cycle, write findings to `research_results.jsonl`, then shut down. You'll see the results on your next cycle.

Use this when:
- You need implementation details for a technique before an H100 run
- You're stuck and need new ideas
- You want to check if SOTA has moved

Do NOT request research for things already documented in `program.md` or `research_results.jsonl`.

## Practical Tips

- `PYTHONUNBUFFERED=1` for real-time output
- `ITERATIONS=200` for quick tests, `ITERATIONS=500` for proper local runs
- `MAX_VAL_TOKENS=524288` to cap local validation time
- Always `.venv/bin/python`

## Competition Rules

- Artifact <= 16MB (zstd compressed)
- Training <= 600s on 8xH100 SXM5
- No test-time training on validation data
- Build on accepted leaderboard submissions
