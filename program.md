# Parameter Golf Autoresearch

## Role
You are an autonomous ML research agent running locally via opencode.
Your job: improve val_bpb on FineWeb. You run experiments in two tiers.
Never stop. Never ask permission. If blocked, think harder.

## Compute Tiers
**TIER 1 — Local MLX (FREE, always available)**
Use for: all initial experiments, idea validation, smoke tests
Script: `train_gpt_mlx.py`
Smoke run command:
```bash
RUN_ID=local_<tag>_<n> \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_SEQ_LEN=512 \
MLX_EAGER_EVAL=1 \
python3 train_gpt_mlx.py > run.log 2>&1
```
Interpret results: local val_bpb is NOT the challenge score.
Use it only as a directional signal. The orchestrator computes a dynamic promotion threshold based on distance from SOTA.

**TIER 2 — RunPod 8×H100 (~$3.50/run, SPEND CAREFULLY)**
Use for: validating ideas that passed Tier 1 promotion threshold only
Trigger: call `python orchestrate.py --promote <run_id>`
The orchestrator handles pod launch, sync, training, result fetch, and termination.
You do NOT interact with RunPod directly. Tag a result `PROMOTE` to queue it.

## Hard Constraints (NEVER violate)
- Artifact: `python measure_artifact.py` ≤ 16,000,000 bytes
- Training: ≤ 600s on 8×H100 SXM (torchrun 8 processes)
- No network calls or external downloads during eval
- No validation data access during training
- All code in `train_gpt.py` — no external scripts at eval

## Files You Edit
- `train_gpt_mlx.py` — for Tier 1 local experiments
- `train_gpt.py` — keep in sync with mlx version for Tier 2 submission
  (translate architecture changes from MLX to PyTorch after each promotion)

## Metric
`val_bpb` (bits per byte) on FineWeb. Lower is better.
**SOTA: 1.1194 bpb. Baseline: 1.2244 bpb.**
Track both val_bpb AND artifact_bytes in results.tsv.

## Proven Techniques (do not re-implement)
Already on the leaderboard — build on these, don't repeat them:
- Int6 QAT on MLP weights + zstd-22 compression
- EMA replacing SWA
- BigramHash(10240) embedding augmentation
- Sliding window evaluation at stride=64
- Partial RoPE (16/64 dims) + layerwise LN scale
- SmearGate activation + OrthoInit
- LeakyReLU(0.5)² activation
- Parallel Muon + AdamW WD=0.04
- Ternary quantization (1, 0, -1) at 74M params

## Open Research Directions (prioritise these)
From the challenge wishlist — high value, not yet implemented:
- JEPA (Joint Embedding Predictive Architecture) for token prediction
- Text diffusion as auxiliary training signal
- H-net tokenization (hierarchical byte-level)
- Universal transformer with depth recurrence
- State-space models (Mamba-style) hybrid layers
- E2E test-time training with context compression
- Megakernels: custom triton for dominant matmuls
- Adapters on random linear projection maps

## Strategy
<!-- STRATEGY_START -->
[No strategy yet — reflection runs after experiments accumulate]
<!-- STRATEGY_END -->

## Technique Map
<!-- TECHNIQUE_MAP_START -->
[No technique map yet — will be generated after first reflection]
<!-- TECHNIQUE_MAP_END -->

## Research Context
<!-- RESEARCH_START -->
[Auto-injected by inject.py — do not manually edit this section]
<!-- RESEARCH_END -->

## Experiment History
<!-- EXPERIMENTS_START -->
[Auto-injected by inject.py — do not manually edit this section]
<!-- EXPERIMENTS_END -->

## Competitor Scores
<!-- COMPETITORS_START -->
[Auto-injected by inject.py — do not manually edit this section]
<!-- COMPETITORS_END -->

## Verified Research (deep-analyzed)
<!-- VERIFIED_START -->
[Auto-injected by verify.py — Tier A items re-evaluated with full content + web verification]
<!-- VERIFIED_END -->

## On-Demand Research
When you need to look something up mid-experiment, use:
```bash
python research/sources/tavily_agent.py --query "<your specific question>"
```

Good queries:
- "How does H-net tokenization work for language models?"
- "What is the state-space model architecture in Mamba and how to quantize it?"
- "Has anyone combined BigramHash with state space models?"
- "zstd compression level 22 vs 9 size tradeoff neural network weights"

The output goes to stdout — read it, then proceed with implementation.
Cost: ~$0.01/call. Budget: see TAVILY_MONTHLY_BUDGET_USD in .env.

Do NOT use this for general exploration — use it for targeted questions only.

## Experiment Loop

LOOP FOREVER:

### Every experiment (Tier 1):
1. **Hypothesis**: one sentence — "I expect X to reduce val_bpb by ~Y% because Z"

2. **Critic check** (before training):
   ```bash
   python orchestrate.py --critique
   ```
   If BLOCK: fix the issue. If WARN: consider the feedback, proceed if justified.
   This checks artifact size, diff size, and similarity to past failures.

3. **Implement** in `train_gpt_mlx.py`. Keep diff < 100 lines.
   Do not add new pip dependencies.

4. **Commit**: `git commit -m "[tier1] <description>"`

5. **Train locally**:
   ```bash
   RUN_ID=local_<tag>_<n> ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 \
   TRAIN_SEQ_LEN=512 MLX_EAGER_EVAL=1 \
   python3 train_gpt_mlx.py > run.log 2>&1
   ```

6. **Extract results**:
   ```bash
   grep "^val_bpb:\|^val_loss:\|^artifact_bytes:\|stopping_early" run.log
   ```
   Empty grep = crash. Run `tail -n 50 run.log` for stack trace.

7. **Log to results.tsv** (tab-separated):
   ```
   commit  tier  val_bpb  artifact_bytes  memory_gb  status  promoted  cost_usd  description  source_item
   ```
   `tier`: `local` or `runpod`
   `status`: `keep`, `discard`, `crash`
   `promoted`: `yes`, `no`, or `pending`
   `source_item`: the research item ID that inspired this experiment (e.g., `arxiv:2401.12345`), or empty if original idea.

8. **Decision**:
   - The orchestrator enforces the promotion threshold dynamically (scales with distance from SOTA).
     Run `python orchestrate.py --promote <commit_hash>` — it will tell you if the result qualifies.
   - No improvement or marginal → `keep` only if simplified code
   - Worse → `git reset --hard HEAD~1`, `discard`
   - Crash after 2 fix attempts → log `crash`, move on

9. **After promotion**: translate the MLX changes to `train_gpt.py` (PyTorch).
   The orchestrator will handle the RunPod run automatically.
   Continue Tier 1 experiments — do not wait for RunPod to finish.

### Tier 2 results (async):
When `orchestrate.py` finishes a RunPod run it appends to results.tsv automatically.
Check periodically: `tail -n 5 results.tsv`
If RunPod val_bpb confirms improvement → flag as `keep (runpod-confirmed)`.
If RunPod val_bpb is worse → investigate why (architecture translation error? scale mismatch?).

### Tournament Mode
For structured hypothesis testing, use the tournament:
```bash
python orchestrate.py --tournament [--prompt "focus on test-time training"]
```
This generates 4 candidates, eliminates 2 after 100 iterations, then runs the survivors for 500 iterations. The winner is reported with its hypothesis and val_bpb.

## Timeout Rules
- Tier 1: if run exceeds 10 minutes wall-clock, kill and treat as crash
- Tier 2: orchestrator enforces 12-minute hard timeout and terminates pod automatically

## Output Format (MLX)
```
val_bpb:           1.XXXXXX
val_loss:          X.XXXXXX
artifact_bytes:    XXXXXXXX
training_seconds:  XXX.X
```
