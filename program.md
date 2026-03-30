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
**SOTA: 1.1091 bpb. Baseline: 1.2244 bpb.**
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
**Convergent techniques (proven across multiple top entries — implement first):**
- XSA-All (Exclusive Self-Attention, all 11 layers) — in 4/7 top entries [IMPLEMENTED in MLX]
- Full Hessian GPTQ (Cholesky + actorder) — supersedes Int6 QAT, in 4/7 top entries
- Coprime-stride data loader — in 3/7 top entries, pure systems win
- EngramLite (multi-head prime hash) — extends BigramHash, in SOTA #1089 [IMPLEMENTED in MLX]
- Turbo-Muon / Polar Express — arxiv:2505.16932. PR #1089 uses 4 iters with AOL preconditioning:
  AOL step: s=1/(A.abs().sum(1).sqrt()+eps), X=s*X, A=s*A*s
  4-iter coefficients: [(4.107,-2.948,0.545), (3.949,-2.909,0.552), (3.318,-2.488,0.510), (2.301,-1.669,0.419)]
  Formula: A=X@X.T, B=b*A+c*(A@A), X=a*X+B@X. Config: LR=0.025, momentum=0.99, WD=0.04

**Promising and implementation details available:**
- Legal Score-First TTT — eval-time -0.002 to -0.003 bpb, in 4 entries
- SLOT (Stochastic Logit Overlay at Test-time) — -0.0008 bpb, free eval-time gain
- ResidLambdas — resid_lambda(init=1.0) + x0_lambda(init=0.1) per layer, best non-TTT at 1.1140
- NorMuon — Polar Express + Adafactor-style variance reduction + cautious WD (modded-nanogpt SOTA)
- MLP tuning: LeakyReLU(0.75)^2 with 3x expansion (PR #1135), value embed with input-dependent gating

**Promising and now understood:**
- ParamBanking — NOT weight sharing, but batched NS optimization on stacked weight banks.
  4 banks: qo(22,512,512), kv(22,256,512), mlp_up(11,1792,512), mlp_down(11,512,1792).
  Forward: F.linear(x, bank[i]). Un-bank for GPTQ, re-bank after. Faster than per-layer NS.
- Mixed-precision GPTQ — base int5 + Hessian-sensitivity greedy promotion to int6/int7.
  Self-generated calibration data. 2% prune headroom.

**Experimental (no competitor validation):**
- Multi-token prediction as auxiliary loss (k=2) — ICLR 2026, no competitor has tried
- Megakernels: custom triton for dominant matmuls (validated in PRs #1105, #1135)

**Exhausted (negative results — do not implement):**
- ~~JEPA~~ — 14 ablations proved negative at 27M/600s scale (PR #1124)
- ~~Universal transformer with depth recurrence~~ — quantization error amplifies ~900x (commit 50390d6)
- ~~State-space models (Mamba-style) hybrid layers~~ — 1.5633 bpb, quantization kills recurrence (PR #1107)
- ~~Knowledge distillation~~ — 600s too tight (PR #1029)
- ~~Adapters on random linear projection maps~~ — 1.607 bpb (PR #874)

## Strategy
<!-- STRATEGY_START -->
## 2026-03-30 20:30Z: Refined Strategy — Implementation Gap Analysis

### Competitive Landscape
SOTA: 1.1091 bpb (PR #1089). Leaderboard stable — two new PRs (#1140, #1141) far below SOTA.
Top 7 span 1.1091-1.1194 (0.0103 bpb range). Field converging on shared technique stack.

### Critical Implementation Gaps (our MLX vs SOTA)
| Gap | Priority | Expected Impact | Status |
|-----|----------|----------------|--------|
| Muon optimizer (NS5) | P0 | +0.01-0.02 bpb vs AdamW | **FEASIBLE on MLX** (52ms/step bf16, benchmarked) |
| Coprime data loader | P1 | Free diversity win | Implementation extracted from #1120 |
| GPTQ quantization | P1 (submission) | Required for 16MB artifact | Not needed for MLX directional testing |
| ResidLambdas | P2 | Best non-TTT at 1.1140 | Implementation extracted from #1130 |
| MiLe loss | P2 | Focuses on hard tokens | ~10 lines, constant gamma=0.75 |
| Legal TTT | P2 (eval-time) | -0.002 to -0.003 bpb | Implementation extracted from #1130 |
| Hyperparameter fixes | P1 | Multiple small gains | See below |

### Hyperparameter Corrections Needed
- QK_GAIN: 1.5 -> 4.0 (PR #1125 ablation proves this)
- VE layers: "9,10" -> "5,9,10", dim 128 -> 196
- LEAKY_SLOPE: 0.5 -> 0.3 (matches SOTA #1089)
- LR: 3e-4 (AdamW) -> 0.025 (Muon) when Muon is implemented

### Recommended Experiment Sequence
**Experiment 1: Muon optimizer** — Single largest gap. Port NS5 to MLX (all ops verified).
Apply to all 2D weight matrices, keep AdamW for embeddings/scalars.
Hyperparams: lr=0.025, momentum=0.85->0.99 warmup, wd=0.04, nesterov=True, ns_steps=5.

**Experiment 2: Hyperparameter corrections** — QK_GAIN=4.0, LEAKY_SLOPE=0.3, VE expansion.
Can run alongside or after Muon to measure combined effect.

**Experiment 3: MiLe loss** — ~10 lines of code. Constant gamma=0.75, clamp_min=0.2.
Entropy-weighted loss focuses training on hard tokens.

**Experiment 4: ResidLambdas** — Per-layer learned scaling: x = resid_lambda*block(x) + x0_lambda*x0.
Init: resid_lambda=1.0, x0_lambda=0.1. Best non-TTT result (1.1140).

**Experiment 5: Coprime loader** — Deterministic coprime walks across data shards.
Pure systems optimization, zero architecture risk.

### Exhausted Directions (do not pursue)
- JEPA: 14 ablations proved negative at 27M/600s (PR #1124)
- Depth recurrence: quantization error amplifies ~900x (commit 50390d6)
- Mamba/SSM hybrids: 1.5633 bpb (PR #1107), quantization kills recurrence
- Knowledge distillation: 1.1553 bpb (PR #1029), 600s too tight

### Open Questions
1. Does Muon on MLX close the AdamW gap? (Expected yes — EVERY top-7 entry uses Muon)
2. Does XSA-all + Muon + Coprime Loader alone match Rascal (1.1099)?
3. Can we combine ResidLambdas (no-TTT path) with TTT for additive gains?
4. Does NorMuon (with Adafactor variance reduction) help at 27M scale?
5. Can MTP auxiliary loss (k=2) improve bpb within 600s budget?
6. Is there an MLX-specific optimization for the NS5 inner loop (e.g., fused kernel)?
<!-- STRATEGY_END -->

## Technique Map
<!-- TECHNIQUE_MAP_START -->
- [convergent] xsa_all (bpb 1.1091) — IMPLEMENTED in MLX
- [sota_component] turbo_muon (bpb 1.1091) — Polar Express w/ AOL, 4 iters, coefficients known
- [promising] normuon — Polar Express + Adafactor variance reduction (modded-nanogpt)
- [proven] parallel_muon_+_adamw (bpb 1.1099)
- [sota_component] engramlite (bpb 1.1091) — IMPLEMENTED in MLX
- [proven] bigramhash (bpb 1.1099)
- [convergent] coprime_loader (bpb 1.1099)
- [convergent] full_hessian_gptq (bpb 1.1116) — mixed-precision: base int5, promote int6/int7
- [proven_but_superseded] int6_qat (bpb 1.1099)
- [proven] zstd_22
- [proven] ema
- [proven] sliding_window_eval
- [proven] partial_rope
- [proven] smeargate
- [proven] orthoinit
- [proven] leakyrelu_sq (bpb 1.1116)
- [proven] ternary_quantization
- [convergent] legal_ttt (bpb 1.1154)
- [promising] slot (bpb 1.1154)
- [promising] resid_lambdas (bpb 1.114) — resid_lambda=1.0, x0_lambda=0.1 per layer
- [sota_component] param_banking (bpb 1.1091) — batched NS on 3D weight banks, NOT weight sharing
- [convergent] triton_fused_mlp (bpb 1.1116)
- [proven] value_residual (bpb 1.1187)
- [marginal] brotli (bpb 1.1138)
- [experimental] mtp_auxiliary — multi-token prediction aux loss, untested in competition
- [exhausted] depth_recurrence
- [exhausted] mamba_ssm (bpb 1.5633)
- [exhausted] jepa
- [exhausted] knowledge_distillation (bpb 1.1553)
<!-- TECHNIQUE_MAP_END -->

## Research Context
<!-- RESEARCH_START -->
- [sota, competitor_validated, multiple_novel_techniques] **[openai/parameter-golf] PR #1089: Record Submission: 1.1091 BPB - Turbo-Muon + EngramLite + ParamBanking + XSA (11L 512d)** — score 16.0/15 (2026-03-29T18:10:07Z)
  Current SOTA (1.1091 bpb). Introduces Turbo-Muon optimizer (4 NS iters + Polar Express), EngramLite (multi-head prime hash extending BigramHash), ParamBanking for parameter efficiency, and XSA on all 11 layers. The winning combination that beats all other entries.
  → https://github.com/openai/parameter-golf/pull/1089
- [near_sota, competitor_validated, simple_quantization] **[openai/parameter-golf] PR #1120: val_bpb 1.1099 (3-seed mean) Rascal** — score 15.0/15 (2026-03-30T04:57:49Z)
  Second-best score (1.1099 bpb) 'Rascal' entry. Uses XSA-all, Parallel Muon, Coprime-stride loader, BigramHash(2048), naive int6+zstd. Proves architecture+training quality can near-match SOTA without fancy quantization.
  → https://github.com/openai/parameter-golf/pull/1120
- [merged_record, self_gen_calibration_novel] **[openai/parameter-golf] PR #1019: Record: AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112 — val_bpb 1.11473 (3-seed mean)** — score 15.0/15 (2026-03-28T13:34:01Z)
  Merged record achieving 1.1147 bpb (3-seed mean) within 15.91 MB and 600s on 8xH100. The key contribution is AR self-generated calibration data for GPTQ, which avoids validation data access during quantization — a novel and rules-compliant approach. Companion mechanistic interpretability analysis adds confidence in the method.
  → https://github.com/openai/parameter-golf/pull/1019
- [record, competitor_validated, systems_optimization] **[openai/parameter-golf] PR #1105: Record: Fused MLP (Triton+CUTLASS EVT) + Brotli + Memmap — 1.1138 BPB** — score 14.0/15 (2026-03-30T00:03:19Z)
  1.1138 bpb via Fused MLP (Triton+CUTLASS EVT) + Brotli compression + Memmap loading. Systems-level optimization: fused kernels save ~1.8ms/step enabling hundreds more training steps. Brotli achieves better compression than zstd.
  → https://github.com/openai/parameter-golf/pull/1105
- [record, competitor_validated, no_ttt, high_statistical_rigor] **[openai/parameter-golf] PR #1130: Record: 1.1140 BPB — ResidLambdas + Split-LR + Train-Budget GPTQ + Coprime Loader (12-seed mean)** — score 14.0/15 (2026-03-30T11:03:22Z)
  1.1140 bpb WITHOUT TTT. ResidLambdas (learned per-layer residual scaling), Split-LR, Train-Budget GPTQ (quantization within training budget), Coprime-stride loader. Most statistically rigorous (12-seed mean). Proves strong training alone suffices.
  → https://github.com/openai/parameter-golf/pull/1130
- [ablation_study, xsa_all_layers, highly_actionable] **[openai/parameter-golf] PR #1125: Non-record: XSA-All + QK Gain 4.0 + LN Scale — 45 Experiments on 1×RTX 5090** — score 14.0/15 (2026-03-30T07:24:55Z)
  45 experiments on 1×RTX 5090 testing XSA variants. KEY FINDING: XSA on ALL 11 layers beats XSA on last-4 by -0.0018 bpb. Also tests QK Gain 4.0 + LN Scale. Invaluable ablation study for XSA implementation.
  → https://github.com/openai/parameter-golf/pull/1125
- [record, competitor_validated, eval_time_technique] **[openai/parameter-golf] PR #1128: Record: SLOT + LeakyReLU² + Legal Score-First TTT + Parallel Muon — val_bpb 1.1154 (3-seed mean) val_bpb = 1.1154 (3-seed mean, std 0.0002) | ~15.9 MB | 8×H100 SXM** — score 13.0/15 (2026-03-30T09:43:20Z)
  1.1154 bpb. SLOT (Stochastic Logit Overlay at Test-time) + LeakyReLU² + Legal Score-First TTT + Parallel Muon. Introduces SLOT as a novel eval-time augmentation technique orthogonal to training improvements.
  → https://github.com/openai/parameter-golf/pull/1128
- [record, engramlite, fa3] **[openai/parameter-golf] PR #1122: Record: EngramLite + Gated Skips + Full GPTQ + FA3 — val_bpb 1.1146 (1-seed, 2 pending)** — score 13.0/15 (2026-03-30T05:30:52Z)
  1.1146 bpb (1-seed). EngramLite + Gated Skips + Full Hessian GPTQ + FlashAttention 3. Validates EngramLite works with gated skip connections. FA3 enables faster attention computation.
  → https://github.com/openai/parameter-golf/pull/1122
- [record, competitor_validated, learned_quantization] **[openai/parameter-golf] PR #1129: Record: CROWN-Q + GPTQ + Legal TTT — val_bpb 1.1174 (3-seed mean)** — score 12.0/15 (2026-03-30T09:49:05Z)
  1.1174 bpb. CROWN-Q (learned quantization grid) + GPTQ + Legal TTT. CROWN-Q learns optimal quantization boundaries per-layer, reducing quantization error vs fixed int6 grid.
  → https://github.com/openai/parameter-golf/pull/1129
- [record, competitor_validated, value_residual] **[openai/parameter-golf] PR #1118: Submission: 11L XSA4 + TrigramHash + ValueResidual + Legal TTT (val_bpb=1.1187)** — score 12.0/15 (2026-03-30T04:15:35Z)
  1.1187 bpb. 11L XSA4 + TrigramHash + ValueResidual + Legal TTT. ValueResidual is a technique where value projections get a direct residual path, improving gradient flow through attention layers.
  → https://github.com/openai/parameter-golf/pull/1118
- [record, adaptive_quantization] **[openai/parameter-golf] PR #1042: Record: Adaptive Precision Embedding Quantization (4-seed mean val_bpb=1.1217)** — score 12.0/15 (2026-03-28T23:06:05Z)
  1.1217 bpb. Adaptive Precision Embedding Quantization — variable bit-width per embedding dimension based on importance. Reduces quantization error on critical embedding dimensions. 4-seed mean.
  → https://github.com/openai/parameter-golf/pull/1042
- [eval_time, slot, orthogonal_gain] **[openai/parameter-golf] PR #1084: Non-Record: SLOT Eval-Time Augmentation on PR #549 SOTA Stack val_bpb = 1.1185 (3-seed mean, std 0.0003) | ~15.9 MB | 8×H100 SXM** — score 11.0/15 (2026-03-29T16:31:32Z)
  SLOT eval-time augmentation gives -0.0008 bpb improvement. Orthogonal to training techniques. Free at inference time. Small but consistent gain.
  → https://github.com/openai/parameter-golf/pull/1084
<!-- RESEARCH_END -->

## Experiment History
<!-- EXPERIMENTS_START -->
[No experiments recorded yet]
<!-- EXPERIMENTS_END -->

## Competitor Scores
<!-- COMPETITORS_START -->
| PR # | Author | Technique | val_bpb | Δ baseline |
|------|--------|-----------|---------|------------|
| #1089 | mikeapedia | Submission: Turbo-Muon + EngramLite + ParamBanking + XSA (11L 512d) | 1.1091 | -0.1153 |
| #1120 |  | Rascal: XSA-all + Parallel Muon + Coprime Loader + BigramHash(2048) + naive int6+zstd | 1.1099 | -0.1145 |
| #1135 |  | Fused Triton MLP + Full GPTQ + Coprime Loader + XSA-all + BH2816 | 1.1116 | -0.1128 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + Brotli + Memmap | 1.1138 | -0.1106 |
| #1130 | Gusanidas | ResidLambdas + Split-LR + Train-Budget GPTQ + Coprime Loader (12-seed mean) | 1.1140 | -0.1104 |
| #1122 | mikeapedia | EngramLite + Gated Skips + Full GPTQ + FA3 | 1.1146 | -0.1098 |
| #1128 | AnubhavBharadwaaj | SLOT + LeakyReLU² + Legal Score-First TTT + Parallel Muon — val_bpb 1.1154 (3-seed mean) val_bpb = 1.1154 (3-seed mean, std 0.0002) | ~15.9 MB | 8×H100 SXM | 1.1154 | -0.1090 |
| #1129 |  | CROWN-Q + GPTQ + Legal TTT | 1.1174 | -0.1070 |
| #965 | Adam-Jacuch | via KGIIR Trajectory Mixing | 1.1184 | -0.1060 |
| #1084 | AnubhavBharadwaaj | SLOT Eval-Time Augmentation on PR #549 SOTA Stack val_bpb = 1.1185 (3-seed mean, std 0.0003) | ~15.9 MB | 8×H100 SXM | 1.1185 | -0.1059 |
| #1118 | adityakm24 | 11L XSA4 + TrigramHash + ValueResidual + Legal TTT | 1.1187 | -0.1057 |
| #1124 | NewyorkDev | v9 Batched Muon + Full GPTQ Random Calib + JEPA Research | 1.1194 | -0.1050 |
| #1042 | nothingLiva | Adaptive Precision Embedding Quantization (4-seed mean val_bpb=1.1217) | 1.1217 | -0.1027 |
| #1086 | Omrigotlieb | Track A: 11L U-Net + BigramHash + SmearGate + Partial RoPE + QAT () | 1.1349 | -0.0895 |
| #1029 | fielding | Knowledge Distillation — A Negative | 1.1553 | -0.0691 |
<!-- COMPETITORS_END -->

## Verified Research (deep-analyzed)
<!-- VERIFIED_START -->
[No verified items yet — Tier A items will be deep-verified automatically]
<!-- VERIFIED_END -->

## On-Demand Research
When you need fresh research context, pull it on demand:
```bash
# Full refresh — all 10 sources + grade + verify + reflect
python orchestrate.py --refresh

# Fast refresh — GitHub PRs + Tavily only (faster, catches competitor moves)
python orchestrate.py --refresh-fast
```

Use `--refresh` when:
- Starting a new experiment direction
- After a string of failures (the reflection cycle will synthesize what went wrong)
- When the technique map shows exhausted branches and you need new ideas

Use `--refresh-fast` when:
- You want to check if competitors shipped something new
- You need a quick context update without waiting for ArXiv/OpenReview

For targeted lookups mid-experiment:
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

The background orchestrator also refreshes automatically: fast sources every 2h, full refresh every 12h.

## Constraint Calculator
Before designing an experiment, verify mathematical feasibility:
```bash
python orchestrate.py --check-constraints --params 23000000 --bits 6 --code-bytes 30000
```
This checks:
- **Artifact size**: will N params at B bits fit in 16MB after zstd compression?
- **Training steps**: how many steps fit in 600s at your batch size?
- **Quantization MSE**: what's the theoretical noise floor at this bit-width?
- **Entropy bound**: can zstd physically compress these weights below 16MB?

Use this to validate ideas BEFORE writing code. Examples:
```bash
# "Can I fit 30M params at int5?"
python orchestrate.py --check-constraints --params 30000000 --bits 5

# "What about int4 with a large model?"
python orchestrate.py --check-constraints --params 50000000 --bits 4 --code-bytes 40000

# "How many steps do I get with a bigger batch?"
python orchestrate.py --check-constraints --params 20000000 --bits 6 --batch-size 128 --seq-len 1024
```

If the report says NOT FEASIBLE, do not proceed — redesign the approach.
The calculator auto-calibrates from weight files on disk when available.

## Experiment Loop

LOOP FOREVER:

### Every experiment (Tier 1):
1. **Hypothesis**: one sentence — "I expect X to reduce val_bpb by ~Y% because Z"
   Validate with `python orchestrate.py --check-constraints` before proceeding.

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
