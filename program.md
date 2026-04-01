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

## Infrastructure & Tooling (MUST READ — recent changes)

**Pod execution flow (git-clone + HTTP, no SSH):**
RunPod direct TCP is broken. Pods now clone the repo from GitHub and serve results via HTTP proxy.
This means `train_gpt.py` MUST be committed and pushed before every promote.
The startup script: clone repo → install zstandard → detect/download data → torchrun → GPTQ → serve results via HTTP on port 18080.
Orchestrator polls `https://{pod_id}-18080.proxy.runpod.net/results.json` for completion, then downloads run.log/model files.

**GPU count is configurable:**
`RUNPOD_GPU_COUNT=1` (current default) — $2.69/hr, use for iteration and debugging.
`RUNPOD_GPU_COUNT=8` — $21.52/hr, use only for final competition submissions.
The training script auto-detects via `NPROC` env var: `torchrun --nproc_per_node=${NPROC}`.
Results from 1-GPU runs are directionally valid but bpb won't match 8-GPU exactly.

**Data: full 80 train shards:**
Previous runs used only 32/80 shards (40% of data), causing 0.06 bpb gap vs SOTA.
Now downloads all 80 train shards from HuggingFace (`willdepueoai/parameter-golf`).
Download happens in startup script BEFORE torchrun — doesn't count against 600s training budget.
Takes ~150s for 80 shards. The `_ensure_data()` fallback in train_gpt.py also downloads 80 shards.

**Step-1000 early eval (the "oracle checkpoint"):**
Research shows r=0.86 correlation between step-1000 val_bpb and final val_bpb (abay.tech/posts/pgolf-meta).
`EARLY_EVAL_STEP=1000` — forces a validation pass at step 1000 regardless of VAL_LOSS_EVERY.
`EARLY_ABORT_BPB=0` (disabled by default) — if set, aborts training when step-1000 val_bpb exceeds threshold.
Use this to kill bad runs after ~90s instead of burning 10 minutes. Calibrate threshold after a few runs.

**zstd API compatibility:**
Pod runs Python 3.12 (`zstandard` pip package uses `max_output_size`).
Local dev may run Python 3.13 (built-in `_zstd` uses `max_length`).
The `_decompress()` function handles both via try/except. Don't change this.

**Pod lifecycle safety:**
Before creating any new pod, the orchestrator terminates ALL running `pgolf-*` pods via API query.
This prevents orphaned pods from killed promote commands. Pods cost $2.69-$21.52/hr — orphans are expensive.
NEVER kill a running promote command — it manages the pod lifecycle. Wait for it to complete or timeout.

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
**Merged SOTA: 1.1147 bpb (PR #1019). Best unmerged: 1.0577 (PR #1180, estesryan: P2 loss + conv token mixer + wallclock LR warmdown). Second unmerged: 1.0781 (PR #672, 30-epoch cosine TTT on PR #518 stack). Third unmerged: 1.0806 (PR #1143, Scylla+TTT). Fourth unmerged: 1.0962 (PR #1176, bigbag: QK-Gain 4.0 + XSA-11 + Muon-TTT + SLOT). Also: PR #1172 (dexhunter, 1.1015, SLOT+Split-LR+GPTQ+XSA-all). Baseline: 1.2244 bpb.**
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
**NEW UNMERGED SOTA techniques (PR #1180, 1.0577 bpb — implement urgently):**
- **P2 LOSS ((1-p)^2)** — Difficulty-aware training loss. Replaces standard CE. Weight = (1 - model_confidence)^2, focusing training on uncertain tokens. Drop-in replacement: `loss = ((1 - probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))**2 * (-log_p)).mean()` where log_p = standard CE per-token. Achieved 1.0577 bpb (new unmerged SOTA). High priority to test.
- **WALLCLOCK-AWARE LR WARMDOWN** (convergent: PR #1180 + PR #1171) — Warmdown triggered by elapsed wall time instead of step count. Ensures full utilization of 600s training budget regardless of step speed. Implementation: record start_time, check `time.time() - start_time >= WARMDOWN_START_SECS`.
- **CONV TOKEN MIXER** — Adds convolutional mixing to residual path. Likely 1D depthwise conv applied to token sequence before/after attention. Details to be confirmed in PR #1180 code.

**INT5 GPTQ (PR #1171, convergent with PR #1105):**
- **INT5 GPTQ (clip_range=15)** — 31 unique levels vs 63 for INT6. 0.476 bytes/param (26% smaller than INT6's 0.64 bytes/param). With 22M params, saves ~3.5MB → enables MLP_MULT 3.5+ or larger embedding. Full Hessian GPTQ same as INT6. Simply set QUANT_CLIP_RANGE=15.

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
- **30-epoch cosine TTT (CRITICAL — PR #672, NEW UNMERGED BEST 1.0781 bpb, -0.041 vs merged SOTA):**
  Single change from PR #518 stack: TTT_EPOCHS=30 with cosine schedule.
  3-seed: 1.0743/1.0774/1.0825 (mean=1.0781, std=0.0041), 15.62 MB. All within budget.
  Timing: Training=600s, TTT=494s, Sliding eval=96s → Total eval=590s (valid, under 10 min).
  Base stack: 11L LeakyReLU(0.5)^2, d=512, 4 KV GQA, MLP 3x, BigramHash(2048), SmearGate,
    XSA4, Partial RoPE, LN Scale, EMA, SWA, Late QAT, OrthoInit, VE128, Int6+zstd-22.
  HIGHEST PRIORITY: implement on our stack immediately.
- **LoRA TTT (rank 8) — CRITICAL UPGRADE (PR #550, 24x more effective than SGD TTT):**
  Benchmark on 100 seqs, 3ep, score-first per 32K chunk, RTX 5090:
  Full-param SGD TTT: delta=-0.004 bpb (-0.2%) ← current approach
  LoRA r=1 (Q+V):    delta=-0.102 bpb (-3.6%)
  LoRA r=4 (Q+V):    delta=-0.131 bpb (-4.6%)
  LoRA r=8 (Q+V):    delta=-0.133 bpb (-4.7%) ← BEST
  LoRA r=8 is ~24x more effective than full-param SGD. Our SGD TTT gives ~-0.010 bpb at H100 scale.
  Expected LoRA TTT gain: ~-0.050 bpb (24x × 0.002 base). TOP PRIORITY to implement.
  Config: Adam lr=0.01 (NOT SGD), apply to Q+V projections only, rank=8.
- Legal Score-First TTT — eval-time -0.010 to -0.018 bpb, in 4+ entries
  Config: TTT_LR=0.002, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768, TTT_FREEZE_BLOCKS=2
- **Muon Legal TTT** (NEW - PR #1148, aamodbhatt, 1.1179): NS orthogonalized updates in TTT loop
  TTT_MUON=1, TTT_NS_STEPS=3. Entropy-adaptive epochs: 2/3/4 per chunk (H_HIGH=2.1, H_LOW=1.75)
  Gives ~-0.018 bpb TTT gain from older base. Drop-in improvement over vanilla SGD TTT.
- **best_agree online n-gram ensemble** (NEW - PR #1145, AnirudhRahul, 1.1109): Causal eval-time overlay, ~0.003 bpb gain. Drop-in on any base model (no architecture change). 3 prefix-only experts: token n-gram (order=16), within-word continuation, word-start (order=4). Selection: pick by expected_gain = p*boost - log(1+q*(exp(boost)-1)). Agreement bonus: +0.500 if 2+ experts agree. Boost: p'(a) = exp(beta)*p(a)/Z (renormalized). Files: online_best_agree_eval.py + online_ngram_state.c (C extension). Hyperparams: TOKEN_ORDER=16, TOKEN_THRESHOLD=0.800, TOKEN_BOOST=2.625, WITHIN_TAU=0.450, WITHIN_BOOST=0.750, WORD_ORDER=4, WORD_NORMALIZE=strip_punct_lower, WORD_TAU=0.650, WORD_BOOST=0.750, AGREE_ADD_BOOST=0.500, CHUNK_TOKENS=131072. Eval time: ~468s on 8xH100. CONFIRMED VALID.
- **SLOT (STILL UNDERESTIMATED — PR #1172 shows -0.029 bpb, not -0.016!)**: Per-batch delta [1,1,512] at last hidden layer.
  Protocol: H=forward_hidden(x, no_grad), H.detach(), 5 AdamW steps on compute_logits(H+delta) only, score with detached delta.
  Config: SLOT_LR=0.003, SLOT_STEPS=5 (PR #1176) or LR=0.005, steps=8 (PR #1172, ~90s). Better base model → bigger SLOT gain.
  PR #1176 breakdown: TTT(-0.003) + SLOT(-0.016) = -0.019 total.
  PR #1172 breakdown: Post-EMA 1.1303 → after SLOT 1.1015 = SLOT contribution -0.029 bpb ← LARGER.
  REVISED ESTIMATE: With our GPTQ stack, expect -0.020 to -0.029 bpb. HIGH PRIORITY.
- ResidLambdas — resid_lambda(init=1.0) + x0_lambda(init=0.1) per layer, best non-TTT at 1.1140
- **QK_GAIN_INIT=4.0 (NEW — PR #1125 sweep, validated PR #1176)**: Per-head learnable Q scaling scalar.
  Implementation: `self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))` applied before QK dot product.
  Default=1.5, optimal=4.0. Monotonic gains: 1.5<2.0<3.0<4.0. Impact: -0.006 bpb on H100, -0.0039 on RTX 5090.
  Adds only 8 params (negligible). Drop-in improvement. Check if our MLX code already has q_gain. If not, add immediately.
- **Split-LR (NEW — PR #1172, PR #1179)**: Different Muon LR per layer group.
  Early layers 0-4: LR=0.025. Late layers 5-10: LR=0.030. Scale bank gradients by layer multiplier before reduce-scatter.
  Consistent across dexhunter submissions. Part of multiple PR improvements.
- **Step-1000 early stopping (PR #1162 meta-analysis)**: Step-1000 BPB correlates 0.86 with final BPB.
  Kill bad local experiments at ITERATIONS=1000 (~90s in). If step-1000 BPB is ≥ expected, abort. Saves ~90% experiment time.
- **MuonEq (NEW — arxiv:2603.28254v1)**: Lightweight O(m+n) row/column equilibration BEFORE Newton-Schulz orthogonalization.
  Three variants: RC (two-sided), R (row-only), C (column-only). Rebalances momentum matrix before NS step.
  No new hyperparameters. Novel direction not yet on leaderboard. ~5 lines in Muon optimizer.
  Potential: improved spectral conditioning → better convergence → lower bpb.
- **NorMuon (FULL DETAILS — modded-nanogpt SOTA, no parameter-golf PR uses it yet):**
  Pipeline: (1) Nesterov momentum FP32, (2) Polar Express 5-iter orthogonalization,
  (3) Adafactor-style per-row variance reduction (EMA of row/col squared norms, O(m+n) memory),
  (4) Cautious WD (only decay when grad agrees with param sign), (5) mantissa tracking for bf16.
  Defaults: lr=0.023, momentum=0.95, beta2=0.9, WD=1.2. ~5% more compute than NS5 Muon.
  Our MuonEq RC already showed -0.103 local bpb (different mechanism: equilibration BEFORE NS).
  NorMuon normalizes AFTER NS (per-row RMS). Complementary to MuonEq. Unexploited in competition.
- MLP tuning: LeakyReLU(0.5)^2 with 3x expansion (MLP_MULT=3.0), value embed VE_ENABLED=1, VE_DIM=128, VE_LAYERS='9,10'
- **MLP 3.5× expansion** (NEW - PR #1105): mechanistic analysis of PR #1019 showed MLP at 94.4% SVD rank utilization (fully packed) vs attention Q at 72.6%. Model was parameter-starved in MLP. MLP 3.5× with mixed int5/int6 quantization enables this without exceeding 16MB budget. Contribution: +0.0037 BPB vs 3× MLP. Combined with Brotli-11 and SLOT → 1.1088 BPB (best unmerged).
- **Brotli-11 + byte-shuffle compression** (PR #1105, PR #1179): saves 581KB vs zstd-22 (Brotli alone), or ~400KB more than LZMA-9 when combined with byte-shuffle. Byte-shuffle reorders bytes before compression for better correlation. Drop-in replacement. PR #1179 used brotli-11+byte-shuffle with code minification (101KB→23KB) to save extra 78KB.
- EGGROLL (Antithetic Ternary Bin Search, NEW - PR #1156, 1.1161): post-GPTQ zeroth-order INT6 bin refinement. 1024 random indices, test ±1 shift, keep improvement. 60s eval budget. 6-14 improvements per seed. Strictly additive (cannot degrade). Co-authored by Claude Opus 4.6.

**Promising and now understood:**
- ParamBanking — NOT weight sharing, but batched NS optimization on stacked weight banks.
  4 banks: qo(22,512,512), kv(22,256,512), mlp_up(11,1792,512), mlp_down(11,512,1792).
  Forward: F.linear(x, bank[i]). Un-bank for GPTQ, re-bank after. Faster than per-layer NS.
  Confirmed active in PR #1135 stack.
- Full Hessian GPTQ (PR #1135 confirmed): Cholesky+actorder, 5-way clip sweep [0.9990,0.9995,0.9999,0.99999,1.0]
  64 training batches calibration (~6.4s). QUANT_CLIP_RANGE=31 (int6 ±31). GPTQ_RESERVE_MS=10000.
  Fallback to quantize_int6_per_row if Cholesky fails.
- Coprime-stride loader (PR #1135 confirmed): np.memmap shards, diversity-weighted shard sampling
  alpha annealing 0.90→0.50, coprime stride = random s where gcd(s,n)=1, phase init to avoid repeats.
- BigramHash config (PR #1135): BIGRAM_VOCAB_SIZE=2816, BIGRAM_DIM=112 (XSA_LAST_N=11 for XSA-all)
- EGGROLL v2 (PR #1156): 60s eval budget, 1024 random indices/step, test ±1 bin shifts, keep improvements
  Strictly additive post-GPTQ refinement. Gets 6-14 improvements/seed. Run after GPTQ during eval budget.

**Hyperparameter tuning wins (low-risk, high-value):**
- WARMDOWN_ITERS=4000 (vs default lower value) — PR #1145 base model achieves 1.1137 bpb, confirmed win
- SLOT hyperparameters: lr=0.003, steps=5 (PR #1150 confirmation)
- **GPTQ_RESERVE_MS=9000 (vs 14000)** — reduces calibration from 14s to 9s, recovers ~55 extra training steps free
- **Sigmoid-gated U-Net skip connections** (PR #1179, 1.1105 bpb, no TTT) — learnable sigmoid gates on encoder-decoder residual skips. Drop-in architectural improvement complementary to all optimizer/quant changes.
- **Soft-round QAT** (PR #1179) — temperature-controlled rounding replacing hard STE. Alpha schedule: 1→16 over training. Formula: soft_round(x) = x - (alpha * tanh(alpha*(x-round(x))))/alpha. More faithful quantization gradients.

**Experimental (no competitor validation):**
- Multi-token prediction as auxiliary loss (k=2) — ICLR 2026, no competitor has tried
- Megakernels: custom triton for dominant matmuls (validated in PRs #1105, #1135)
- **Scylla tokenizer (WATCH - PR #1143, ~1.080 bpb if valid)**: TokenMonster tm0054 (pruned english-1024-clean-v1). Full-data FineWeb retokenization (79 train + 1 val shards). Byte accounting via per-token metadata LUTs. ~0.030 bpb gain if organizers accept. Key risk: requires custom data bundle, organizer review pending. Do NOT adopt until accepted.

**PARADIGM SHIFT — Dirichlet PPM-7 (PR #1159, JDAppleseed, 0.369 BPB, UNDER REVIEW):**
- Achieves 0.369 BPB from 1.237 BPB base — entirely from PPM-7 cache during sliding window eval (NOT TTT, NOT architecture)
- `ScoreFirstCausalCache`: builds causal n-gram hash table (orders 2-7) from eval tokens as processed (score-first)
- Dirichlet posterior: `posterior = (full_counts + 4.0 * model_p) / (ctx_counts + 4.0)` where count_smoothing=4.0
- Hash function: `ctx_hash = XOR(token[k-i] * PRIMES[i%5] for i in range(order-1)); ctx_key = ctx_hash & (4194304-1)`
  `PRIMES = [36313, 27191, 51647, 81929, 131071]`; full_key adds `token[k] * PRIMES[(order-1)%5]`
- Config: `CAUSAL_CACHE_MODE=ppm CAUSAL_CACHE_MAX_ORDER=7 CAUSAL_CACHE_MIXING=dirichlet CAUSAL_CACHE_COUNT_SMOOTHING=4.0`
  `CAUSAL_CACHE_BUCKETS=4194304 CAUSAL_CACHE_MIN_COUNT=2 CAUSAL_CACHE_ALPHA=0.30 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64`
- Timing without TTT: ~514s post-train eval (fits in 600s eval budget: sliding_window=423s + other=91s)
- Distributed eval: cache SHARED across 8 GPUs in causal order via `plan_distributed_window_shard()`; prefix/suffix positions committed without scoring via `replay_scored_positions()`
- TTT on top of PPM adds ~0.001 BPB (negligible); PPM alone is the signal
- Implementation files: `frontier_cache.py` (448 lines pure numpy) + `frontier_eval.py` (151 lines)
- Legality: score-first, causal, no future tokens — structurally identical to approved TTT. No organizer comments yet.
- Gain estimate for our GPT2-50257 vocab: likely 0.5-0.8 BPB (vs 0.369 at vocab=1024). Still -0.3 to -0.6 BPB vs SOTA.
- **ACTION: Implement PPM-7 eval ASAP. Hold Tier 2 until organizer accepts PR #1159.**

**Exhausted (negative results — do not implement):**
- ~~JEPA~~ — 14 ablations proved negative at 27M/600s scale (PR #1124)
- ~~Universal transformer with depth recurrence~~ — quantization error amplifies ~900x (commit 50390d6)
- ~~State-space models (Mamba-style) hybrid layers~~ — 1.5633 bpb, quantization kills recurrence (PR #1107)
- ~~Knowledge distillation~~ — 600s too tight (PR #1029)
- ~~Adapters on random linear projection maps~~ — 1.607 bpb (PR #874)

## Strategy
<!-- STRATEGY_START -->
## Technique Map
<!-- TECHNIQUE_MAP_START -->
- [active] xsa_all (bpb 1.1091)
- [dead_end] turbo_muon (bpb 1.1091)
  - [active] parallel_muon_+_adamw (bpb 1.1099)
  - [promising] param_banking (bpb 1.1091)
  - [active] normuon_vr (bpb 9.349 local, -0.022 vs MuonEq baseline)
- [active] engramlite (bpb 1.1091)
- [proven] bigramhash (bpb 1.1099)
- [active] coprime_loader (bpb 1.1099)
- [promising] full_hessian_gptq (bpb 1.1116)
  - [dead_end] int6_qat (bpb 1.1099)
- [proven] zstd_22
- [proven] ema
- [proven] sliding_window_eval
- [proven] partial_rope
- [proven] smeargate
- [proven] orthoinit
- [proven] muoneq_rc (bpb 1.1599 on H100, -0.0036 vs baseline. arxiv:2603.28254 row/col equilibration before NS5)
- [proven] leakyrelu_sq (bpb 1.1116)
- [proven] ternary_quantization
- [promising] legal_ttt (bpb 1.1154)
- [promising] slot (bpb 1.1154)
- [promising] resid_lambdas (bpb 1.114)
- [promising] triton_fused_mlp (bpb 1.1116)
- [proven] value_residual (bpb 1.1187)
- [marginal] brotli (bpb 1.1138)
- [dead_end] p2_focal_loss (bpb 1.2377 on H100, REGRESSION — downweights confident tokens, reduces effective gradient at 5800-step budget)
- [dead_end] depth_recurrence
- [dead_end] mamba_ssm (bpb 1.5633)
- [dead_end] jepa
- [dead_end] knowledge_distillation (bpb 1.1553)
- [promising] mtp_auxiliary
- [promising] ppm7_cache (bpb 0.369)
- [promising] best_agree_ensemble (bpb 1.1109)
- [promising] eggroll_v2 (bpb 1.1161)
- [promising] muon_legal_ttt (bpb 1.1179)
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
- [top_priority, already_in_competitor_scores, p2_loss_novel_direction] **[openai/parameter-golf] PR #1180: SR-CM-P2Loss: 1.0577 bpb (~15.06MB)** — score 15.0/15 (2026-03-31T11:32:19Z)
  Top leaderboard submission at 1.0577 bpb. P2 loss ((1-p)^2 difficulty-aware weighting) is a novel and simple loss modification (~5 lines). Wallclock-aware LR warmdown and conv token mixer are new directions not in our proven techniques. The 0.29 bpb gap over SOTA makes this the highest-priority item to study and adapt.
  → https://github.com/openai/parameter-golf/pull/1180
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
- [parameter_efficiency, builds_on_proven_stack, below_sota] **[openai/parameter-golf] PR #768: Non-record: 1.1201 BPB - Shared ValueEmbedding (tok_emb reuse, layers 5-10) + Legal TTT** — score 13.0/15 (2026-03-25T21:19:18Z)
  Shared ValueEmbedding reuses tied tok_emb instead of training separate VE weights, freeing parameter budget to expand VE from 2 to 6 layers. Achieves 1.1201 bpb (3-seed mean, std 0.0002)—not SOTA but demonstrates a clean parameter-efficiency trade-off. The technique is well-documented with reproducible results, though it builds on already-proven components (LeakyReLU², Parameter Banking, TTT).
  → https://github.com/openai/parameter-golf/pull/768
- [already_tracked_competitor, eval_time_technique] **[openai/parameter-golf] PR #1145: Record: 1.1109 BPB FullGPTQ XSA11 + online ngram augment** — score 13.0/15 (2026-03-30T18:58:12Z)
  Combines Full GPTQ XSA11 with a novel online n-gram eval-time ensemble (best_agree) that boosts model distribution using prefix-only token/word experts. Achieves 1.1109 bpb. The eval-time augmentation technique is modular and implementable, but this is already a tracked competitor submission.
  → https://github.com/openai/parameter-golf/pull/1145
- [already_known_competitor, top_leaderboard, validates_ttt_scaling] **[openai/parameter-golf] PR #672: Record: 30ep Cosine TTT on LeakyReLU² stack (3-seed mean val_bpb=1.0781)** — score 13.0/15 (2026-03-25T03:22:29Z)
  Current leaderboard SOTA at 1.0781 bpb via 30-epoch cosine TTT on the LeakyReLU² stack, validated across 3 seeds with tight std=0.0041. Already listed as top competitor — no novelty for our implementation, but strongly validates TTT epoch scaling as the highest-impact single lever available.
  → https://github.com/openai/parameter-golf/pull/672
<!-- RESEARCH_END -->

## Experiment History
<!-- EXPERIMENTS_START -->
- [local] Standard NS5 Muon (cubic, a=15/8 b=-5/4 c=3/8) + LEAKY_SLOPE=0.75. 500 iters. val_bpb 9.474 vs 9.354 Turbo-Muon baseline — expected regression at 500 steps (Turbo-Muon faster short-term convergence). H100 correct: PR #1105 confirmed Turbo-Muon +0.0018 BPB worse at 7000+ steps. NS5 now canonical for H100 runs. — val_bpb=9.4739, status=keep (cost=$0.00)
- [local] ResidLambdas: x0_lambda init=0.1 (vs resid_mix init=0.0). val_bpb 9.506 vs 9.474 baseline (-0.031 worse). Scale-dependent: PR #1130 uses it at H100 scale (7000+ steps). At 500 steps the x0 injection adds noise before the model can benefit. Reverted. Note: attn_scale/mlp_scale (init=1.0) already implement the resid_lambda part; only x0 injection was new. — val_bpb=9.5056, status=discard (cost=$0.00)
- [local] MuonEq RC equilibration before NS5 (arxiv:2603.28254): per-row then per-col normalization of gradient matrix before NS iterations. val_bpb 9.371 vs NS5 baseline 9.474 (-0.103). Train loss 6.76→6.38. 2.75s/step. Novel technique not yet on leaderboard. Closes gap vs Turbo-Muon (9.354) while keeping standard NS5 5-iter that works at H100 scale. — val_bpb=9.3708, status=keep (cost=$0.00)
- [local] TTT-3 smoke test: TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768, MAX_VAL_TOKENS=131072. val_bpb 9.380 vs 9.371 MuonEq baseline — neutral at 500 steps (expected: undertrained model has too-high loss for TTT signal). TTT takes 108.9s for 4 chunks (validated impl). H100 scale (7000 steps, well-trained) should show full -0.041 bpb gain from 30 epochs. — val_bpb=9.3800, status=keep (cost=$0.00)
- [runpod] H100 Cycle 23 baseline: NS5+MuonEq+LEAKY=0.75+XSA-all+EngramLite+GPTQ-zlib (TTT disabled). Artifact 188KB over 16MB limit — zstandard not installed on pod (fell back to zlib). Fix: install zstandard before torchrun. — val_bpb=1.3500, status=keep (cost=$4.72)
- [runpod] H100 Cycle 25: training OK (val_bpb 1.1830 at step 5830, +EGGROLL 34 improvements), TTT timed out (SSH 1800s limit exceeded by HF download ~327s + GPTQ 38s + EGGROLL 23s). zstd STILL failing (externally-managed env). Artifact 16.79MB > 16MB (zlib). Fixes: --break-system-packages, timeout 1800→2400s. — val_bpb=1.1830, status=crash (cost=$10.02)
- [local] P2 focal loss (PR #1180): (1-p)^2 * CE weighting. 500 iters, MuonEq+NS5+QK_GAIN=4.0+XSA-all+EngramLite+LEAKY=0.75. val_bpb 9.351 vs MuonEq baseline 9.371 (-0.020). Train loss 6.75→6.41. Clear positive signal at local scale. — val_bpb=9.3512, status=keep (cost=$0.00)
- [runpod]  — val_bpb=0.0000, status=crash (cost=$4.81)
- [runpod] H100 BASELINE SUCCESS (Rascal PR #1120 + HF data download fix in train_gpt.py). val_bpb=1.1705 (sliding window stride=64, exact=1.17049). Post-EMA=1.1876. 5825 steps in 601s. GPTQ int6+zstd=16.29MB + code 131KB = 16.42MB total (420KB OVER 16MB limit). Fix needed: minify code or reduce params. First clean H100 run. Pipeline fully validated. — val_bpb=1.1705, status=keep (cost=$5.30)
- [runpod] Brotli-11 + P2 focal loss (buggy normalization). val_bpb=1.2424 (REGRESSION). Artifact 11.63MB (brotli saves 4.8MB). P2 normalization bug: w.sum() division. — val_bpb=1.2424, status=keep (cost=$5.99)
- [runpod] P2 focal loss (fixed mean normalization) + brotli-11 + INT5 GPTQ. val_bpb=1.2377 (still REGRESSION from 1.1705 baseline, +0.067). P2 loss CONFIRMED HARMFUL at H100 5800-step budget — downweights confident tokens, reduces effective gradient during warmdown. Artifact 11.63MB (under 16MB, brotli works). Conclusion: disable P2 loss, keep brotli-11. — val_bpb=1.2377, status=keep (cost=$5.99)
- [local] NorMuon VR (Adafactor-style variance reduction AFTER NS5): MUON_VR=1 MUON_VR_BETA2=0.95. val_bpb 9.349 vs MuonEq baseline 9.371 (-0.022). Train loss 6.78→6.44. Complementary to MuonEq RC (which equilibrates BEFORE NS5). Ported to train_gpt.py for H100. — val_bpb=9.3488, status=keep (cost=$0.00)
<!-- EXPERIMENTS_END -->

## Competitor Scores
<!-- COMPETITORS_START -->
| PR # | Author | Technique | val_bpb | Δ baseline |
|------|--------|-----------|---------|------------|
| #1180 | estesryan | SR-CM-P2Loss: (~15.06MB) | 1.0577 | -0.1667 |
| #672 | andrewbaggio1 | 30ep Cosine TTT on LeakyReLU² stack | 1.0781 | -0.1463 |
| #1143 | simon-marcus | Scylla (novel tokenizer) + Legal Score-First TTT | 1.0806 | -0.1438 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + MLP 3.5× + Mixed int5/int6 + SLOT + Brotli | 1.1088 | -0.1156 |
| #1089 | mikeapedia | Submission: Turbo-Muon + EngramLite + ParamBanking + XSA (11L 512d) | 1.1091 | -0.1153 |
| #1120 |  | Rascal: XSA-all + Parallel Muon + Coprime Loader + BigramHash(2048) + naive int6+zstd | 1.1099 | -0.1145 |
| #1145 | AnirudhRahul | FullGPTQ XSA11 + online ngram augment | 1.1109 | -0.1135 |
| #1135 |  | Fused Triton MLP + Full GPTQ + Coprime Loader + XSA-all + BH2816 | 1.1116 | -0.1128 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + MLP 3.5× + Mixed int5/int6 + Brotli — (seed 314, more seeds running) | 1.1123 | -0.1121 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + MLP 3.5× + Mixed int5/int6 + Brotli | 1.1125 | -0.1119 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + Brotli + Memmap | 1.1138 | -0.1106 |
| #1130 | Gusanidas | ResidLambdas + Split-LR + Train-Budget GPTQ + Coprime Loader (12-seed mean) | 1.1140 | -0.1104 |
| #1171 | EthanYangTW | : Parallel Muon + INT5 GPTQ + Legal TTT | 1.1145 | -0.1099 |
| #1122 | mikeapedia | EngramLite + Gated Skips + Full GPTQ + FA3 | 1.1146 | -0.1098 |
| #1150 | sahiee-dev | Legal TTT (SGD, 3-epoch) + SLOT (lr=0.003, steps=5) on PR #549 base -- val_bpb: 1.11512 | 1.1151 | -0.1093 |
<!-- COMPETITORS_END -->

## Verified Research (deep-analyzed)
<!-- VERIFIED_START -->
[No verified items yet — Tier A items will be deep-verified automatically]
<!-- VERIFIED_END -->

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

# "Can I fit 30M params at int5?"
python orchestrate.py --check-constraints --params 30000000 --bits 5

# "What about int4 with a large model?"
python orchestrate.py --check-constraints --params 50000000 --bits 4 --code-bytes 40000

# "How many steps do I get with a bigger batch?"
python orchestrate.py --check-constraints --params 20000000 --bits 6 --batch-size 128 --seq-len 1024
```

If the report says NOT FEASIBLE, do not proceed — redesign the approach.
The calculator auto-calibrates from weight files on disk when available.
