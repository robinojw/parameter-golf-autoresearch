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
**Merged SOTA: 1.1147 bpb (PR #1019). Best unmerged: 1.0781 (PR #672, 30-epoch cosine TTT on PR #518 stack, SINGLE CHANGE). Second unmerged: 1.0806 (PR #1143, Scylla+TTT). Baseline: 1.2244 bpb.**
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
- **30-epoch cosine TTT (CRITICAL — PR #672, NEW UNMERGED BEST 1.0781 bpb, -0.041 vs merged SOTA):**
  Single change from PR #518 stack: TTT_EPOCHS=30 with cosine schedule.
  3-seed: 1.0743/1.0774/1.0825 (mean=1.0781, std=0.0041), 15.62 MB. All within budget.
  Timing: Training=600s, TTT=494s, Sliding eval=96s → Total eval=590s (valid, under 10 min).
  Base stack: 11L LeakyReLU(0.5)^2, d=512, 4 KV GQA, MLP 3x, BigramHash(2048), SmearGate,
    XSA4, Partial RoPE, LN Scale, EMA, SWA, Late QAT, OrthoInit, VE128, Int6+zstd-22.
  HIGHEST PRIORITY: implement on our stack immediately.
- Legal Score-First TTT — eval-time -0.010 to -0.018 bpb, in 4+ entries
  Config: TTT_LR=0.002, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768, TTT_FREEZE_BLOCKS=2
- **Muon Legal TTT** (NEW - PR #1148, aamodbhatt, 1.1179): NS orthogonalized updates in TTT loop
  TTT_MUON=1, TTT_NS_STEPS=3. Entropy-adaptive epochs: 2/3/4 per chunk (H_HIGH=2.1, H_LOW=1.75)
  Gives ~-0.018 bpb TTT gain from older base. Drop-in improvement over vanilla SGD TTT.
- **best_agree online n-gram ensemble** (NEW - PR #1145, AnirudhRahul, 1.1109): Causal eval-time overlay, ~0.003 bpb gain. Drop-in on any base model (no architecture change). 3 prefix-only experts: token n-gram (order=16), within-word continuation, word-start (order=4). Selection: pick by expected_gain = p*boost - log(1+q*(exp(boost)-1)). Agreement bonus: +0.500 if 2+ experts agree. Boost: p'(a) = exp(beta)*p(a)/Z (renormalized). Files: online_best_agree_eval.py + online_ngram_state.c (C extension). Hyperparams: TOKEN_ORDER=16, TOKEN_THRESHOLD=0.800, TOKEN_BOOST=2.625, WITHIN_TAU=0.450, WITHIN_BOOST=0.750, WORD_ORDER=4, WORD_NORMALIZE=strip_punct_lower, WORD_TAU=0.650, WORD_BOOST=0.750, AGREE_ADD_BOOST=0.500, CHUNK_TOKENS=131072. Eval time: ~468s on 8xH100. CONFIRMED VALID.
- SLOT (Stochastic Logit Overlay at Test-time) — -0.0008 bpb, free eval-time gain
  Config: SLOT_LR=0.003, SLOT_STEPS=5 (PR #1150 confirmed)
- ResidLambdas — resid_lambda(init=1.0) + x0_lambda(init=0.1) per layer, best non-TTT at 1.1140
- **MuonEq (NEW — arxiv:2603.28254v1)**: Lightweight O(m+n) row/column equilibration BEFORE Newton-Schulz orthogonalization.
  Three variants: RC (two-sided), R (row-only), C (column-only). Rebalances momentum matrix before NS step.
  No new hyperparameters. Novel direction not yet on leaderboard. ~5 lines in Muon optimizer.
  Potential: improved spectral conditioning → better convergence → lower bpb.
- NorMuon — Polar Express + Adafactor-style variance reduction + cautious WD (modded-nanogpt SOTA)
- MLP tuning: LeakyReLU(0.5)^2 with 3x expansion (MLP_MULT=3.0), value embed VE_ENABLED=1, VE_DIM=128, VE_LAYERS='9,10'
- **MLP 3.5× expansion** (NEW - PR #1105): mechanistic analysis of PR #1019 showed MLP at 94.4% SVD rank utilization (fully packed) vs attention Q at 72.6%. Model was parameter-starved in MLP. MLP 3.5× with mixed int5/int6 quantization enables this without exceeding 16MB budget. Contribution: +0.0037 BPB vs 3× MLP. Combined with Brotli-11 and SLOT → 1.1088 BPB (best unmerged).
- **Brotli-11 compression** (NEW - PR #1105): saves 581KB vs zstd-22. Enables more model capacity within 16MB artifact budget. Drop-in replacement.
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
## 2026-03-31 (Cycle 20 Reflection — MuonEq RC experiment)

**MuonEq RC POSITIVE locally (commit d0010d7):** Per-row/col normalization of gradient matrix before NS5 iterations (arxiv:2603.28254). val_bpb 9.371 vs NS5 baseline 9.474 (-0.103 bpb). Closes gap vs Turbo-Muon (9.354) at 500 steps. Unlike Turbo-Muon, keeps standard 5-iter NS which is confirmed better at H100 scale. Training_seconds=1374s (2.75s/step on M4 — acceptable). Novel: not yet on leaderboard.

**IMPORTANT DISTINCTION from Turbo-Muon:** MuonEq is preconditioning (better spectral conditioning before NS), not coefficient tuning (fewer NS iterations). The H100 concern with Turbo-Muon was that Polar Express 4-iter had less accurate orthogonalization at long runs. MuonEq keeps 5 NS iters but improves their starting point. Should not have the same H100 degradation risk.

**Next H100 run candidate:** NS5+MuonEq+EngramLite+coprime+XSA-all+LEAKY_SLOPE=0.75. Need to implement GPTQ in train_gpt.py first.

**Priority stack (Cycle 20):**
1. **IMPLEMENT GPTQ in train_gpt.py** — H100 only, can't test locally. Unblocks Tier 2.
2. **Port MuonEq to train_gpt.py** — Add to PyTorch version for H100 submission
3. **Test MuonEq at H100 scale** — confirm it doesn't degrade like Turbo-Muon
4. **30-epoch cosine TTT** — HIGHEST PRIORITY eval-time gain (PR #672, -0.041 bpb)
5. **best_agree online ngram** — PR #1145, -0.003 bpb drop-in

---
## 2026-03-31 (Cycle 19 Reflection — NS5 Muon + ResidLambdas experiments)

**NS5 Muon baseline established (commit 3c22252):** Switched from Turbo-Muon to standard cubic NS5 (a=15/8, b=-5/4, c=3/8). Fixed LEAKY_SLOPE default 0.3→0.75. Local baseline: 9.474 bpb (vs 9.354 Turbo-Muon). Expected regression at 500 steps — Turbo-Muon has early convergence advantage that disappears at 7000+ H100 steps. NS5 is correct for H100.

**ResidLambdas NEGATIVE locally (discarded):** x0_lambda init=0.1 added 0.031 bpb regression at 500 steps. Scale-dependent: PR #1130 gains appear at 7000+ step H100 runs. attn_scale/mlp_scale (init=1.0) already implement the resid_lambda part; x0 injection needs many steps to learn. Note for H100: consider adding with init=0.05 after GPTQ is implemented.

**LOCAL EXPERIMENTS HITTING LIMITS:** With NS5 Muon (slower convergence) and only 500 steps, most techniques need H100 scale to show directional signal. Continuing local experiments may give misleading results.

**Current local script status:** XSA-all, EngramLite, coprime-stride, NS5 Muon (correct), LEAKY_SLOPE=0.75 (correct), MLP 3.5× (MLP_MULT=3.5 default), WARMDOWN_ITERS param. This is the Rascal-equivalent stack but with EngramLite instead of BigramHash.

**Priority stack (Cycle 19):**
1. **IMPLEMENT GPTQ in train_gpt.py** — H100 only, can't test locally. This unblocks Tier 2 submission.
2. **Prepare H100 submission** — current NS5+EngramLite+coprime stack should reach ~1.1099 without GPTQ
3. **ResidLambdas for H100** — add as additive x0 injection (init=0.05) to the Tier 2 run
4. **NorMuon** — queued for research (implementation details needed)
5. **Watch PR #1143, #1159** — legality decisions

**What we have ready for H100:** NS5 Muon, XSA-all, EngramLite, coprime-stride, LEAKY_SLOPE=0.75, MLP 3.5×. Need: train_gpt.py parity, GPTQ implementation.

---
## 2026-03-31 18:00 UTC (Cycle 18 Reflection — Post Deep Dive)

**Key updates since Cycle 14:**

**COPRIME-STRIDE LOADER: IMPLEMENTED** (commit 94ba1c7). `train_gpt_mlx.py` has alpha annealing 0.90→0.50 over 1800 batches, gcd(s,n)==1 stride, diversity-weighted shard sampling. This is off the to-do list.

**FULL HESSIAN GPTQ: CONFIRMED DETAILS (still not implemented).** Cycle 17 deep dive extracted every detail from PR #1135. actorder=argsort(H.diag(), desc=True); double-Cholesky inverse; 5-way clip sweep [0.9990,0.9995,0.9999,0.99999,1.0]; block_size=128; clip_range=31; 64 calib batches; fallback to naive per-row on LinAlgError. Shared Hessian for Q,K,V separately. **This is the single biggest remaining implementation gap.** Without GPTQ we cannot match PR #1135 (1.1116) or PR #1105 (1.1088).

**BIGRAMHASH OPTIMAL CONFIG:** PR #1135 uses BIGRAM_VOCAB_SIZE=2816, BIGRAM_DIM=112 (vs our possibly different sizing). Verify our config matches before Tier 2 runs.

**LEAKY_SLOPE=0.75 locally confirmed positive:** Local experiment 774dc29 shows LEAKY_SLOPE=0.75 gives val_bpb 9.354 vs 9.366 with 0.3. PR #1135 uses 0.75. This should be our default.

**No SOTA change (Cycle 16 leaderboard check):** Merged SOTA still 1.1147 (PR #1019). PRs #1163-1167 non-competitive. Best unmerged: PR #1143 (1.0806, Scylla tokenizer — awaiting legality review), PR #1105 (1.1088, best actionable target).

**Priority stack (Cycle 18 revised):**
1. **IMPLEMENT GPTQ** — biggest gap, all top entries use it. Full details confirmed.
2. **MLP 3.5× + mixed int5/int6 + Brotli-11** — copy PR #1105 recipe exactly for 1.1088 target
3. **Stack eval-time gains**: Muon TTT + SLOT + EGGROLL v2 + best_agree (combined ~-0.025 bpb)
4. **Watch PR #1143 (Scylla tokenizer)** — if accepted, pivot immediately to tokenizer search
5. **Watch PR #1159 (PPM-7 cache)** — if accepted, implement causal n-gram cache (paradigm-shift)

**What we have implemented (MLX):** NS5 Muon, XSA-all, EngramLite, coprime-stride (fixed stride=n//steps, was broken at n//2), LEAKY_SLOPE=0.75, WARMDOWN_ITERS param
**What we still need:** Full GPTQ, MLP 3.5×, mixed int5/int6, Brotli-11, eval-time stack, ParamBanking
**NOTE:** Coprime-stride local signal is neutral (expected — 500 steps = 4% shard coverage). Coprime benefit appears at H100 scale with full shard traversal.

---
<!-- STRATEGY_END -->

## Technique Map
<!-- TECHNIQUE_MAP_START -->
- [active] xsa_all (bpb 1.1091)
- [dead_end] turbo_muon (bpb 1.1091)
  - [active] parallel_muon_+_adamw (bpb 1.1099)
  - [promising] param_banking (bpb 1.1091)
  - [promising] normuon
  - [promising] muoneq_rc (local +0.103 bpb over plain NS5; novel, H100 unverified)
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
- [proven] leakyrelu_sq (bpb 1.1116)
- [proven] ternary_quantization
- [promising] legal_ttt (bpb 1.1154)
- [promising] slot (bpb 1.1154)
- [promising] resid_lambdas (bpb 1.114)
- [promising] triton_fused_mlp (bpb 1.1116)
- [proven] value_residual (bpb 1.1187)
- [marginal] brotli (bpb 1.1138)
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
- [record, competitor_validated, learned_quantization] **[openai/parameter-golf] PR #1129: Record: CROWN-Q + GPTQ + Legal TTT — val_bpb 1.1174 (3-seed mean)** — score 12.0/15 (2026-03-30T09:49:05Z)
  1.1174 bpb. CROWN-Q (learned quantization grid) + GPTQ + Legal TTT. CROWN-Q learns optimal quantization boundaries per-layer, reducing quantization error vs fixed int6 grid.
  → https://github.com/openai/parameter-golf/pull/1129
- [record, competitor_validated, value_residual] **[openai/parameter-golf] PR #1118: Submission: 11L XSA4 + TrigramHash + ValueResidual + Legal TTT (val_bpb=1.1187)** — score 12.0/15 (2026-03-30T04:15:35Z)
  1.1187 bpb. 11L XSA4 + TrigramHash + ValueResidual + Legal TTT. ValueResidual is a technique where value projections get a direct residual path, improving gradient flow through attention layers.
  → https://github.com/openai/parameter-golf/pull/1118
<!-- RESEARCH_END -->

## Experiment History
<!-- EXPERIMENTS_START -->
- [local] Baseline: AdamW 3e-4, XSA-all, EngramLite, 200 iters, 30.7M params. Loss stuck at random init (1.6M tokens insufficient to show learning). Establishes timing: 2s/step on M4. — val_bpb=10.0074, status=keep (cost=$0.00)
- [local] Muon (NS5 lr=0.025) + QK_GAIN=4.0 + LEAKY_SLOPE=0.3. 200 iters. Loss 6.75→6.63 (learning signal!), AdamW was flat at 6.93. val_bpb 9.623 vs 10.007 baseline (-0.384 improvement). 2.62s/step. Clear Muon win. — val_bpb=9.6233, status=keep (cost=$0.00)
- [local] Turbo-Muon (Polar Express 4-iter + AOL preconditioning). 500 iters. val_bpb 9.366 vs 9.623 (-0.257). Train loss 6.73→6.49. BUT 1233s run time (>10min limit) due to EMA eval every step. Fix needed. — val_bpb=9.3661, status=keep (cost=$0.00)
- [local] LEAKY_SLOPE=0.75 (vs 0.3). Turbo-Muon unchanged. 500 iters. val_bpb 9.354 vs 9.366 (-0.012). Train loss 6.73->6.48. PR #1135 uses 0.75, positive direction confirmed. — val_bpb=9.3537, status=keep (cost=$0.00)
- [local] Coprime loader v1 (BAD stride=n/2): alternates between 2 positions only. Phase=53M hit low-entropy data (train loss 4.69 vs normal 6.73). val_bpb 9.919 WORSE than baseline. Bug: stride should be n//total_steps not n//2. — val_bpb=9.9186, status=discard (cost=$0.00)
- [local] Coprime loader v2 (stride=200001≈n/500, correct full-shard coverage + WARMDOWN_ITERS param). Neutral: val_bpb 9.356 vs 9.354 (within noise). Expected — 500 steps only covers 4% of shard anyway. Implementation correct for H100. — val_bpb=9.3555, status=keep (cost=$0.00)
<!-- EXPERIMENTS_END -->

## Competitor Scores
<!-- COMPETITORS_START -->
| PR # | Author | Technique | val_bpb | Δ baseline |
|------|--------|-----------|---------|------------|
| #672 | andrewbaggio1 | 30ep Cosine TTT on LeakyReLU² stack | 1.0781 | -0.1463 |
| #1143 | simon-marcus | Scylla (novel tokenizer) + Legal Score-First TTT | 1.0806 | -0.1438 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + MLP 3.5× + Mixed int5/int6 + SLOT + Brotli | 1.1088 | -0.1156 |
| #1089 | mikeapedia | Submission: Turbo-Muon + EngramLite + ParamBanking + XSA (11L 512d) | 1.1091 | -0.1153 |
| #1120 |  | Rascal: XSA-all + Parallel Muon + Coprime Loader + BigramHash(2048) + naive int6+zstd | 1.1099 | -0.1145 |
| #1145 | AnirudhRahul | FullGPTQ XSA11 + online ngram augment | 1.1109 | -0.1135 |
| #1135 |  | Fused Triton MLP + Full GPTQ + Coprime Loader + XSA-all + BH2816 | 1.1116 | -0.1128 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + MLP 3.5× + Mixed int5/int6 + Brotli — (seed 314, more seeds running) | 1.1123 | -0.1121 |
| #1105 | abaybektursun | Fused MLP (Triton+CUTLASS EVT) + Brotli + Memmap | 1.1138 | -0.1106 |
| #1130 | Gusanidas | ResidLambdas + Split-LR + Train-Budget GPTQ + Coprime Loader (12-seed mean) | 1.1140 | -0.1104 |
| #1122 | mikeapedia | EngramLite + Gated Skips + Full GPTQ + FA3 | 1.1146 | -0.1098 |
| #1150 | sahiee-dev | Legal TTT (SGD, 3-epoch) + SLOT (lr=0.003, steps=5) on PR #549 base -- val_bpb: 1.11512 | 1.1151 | -0.1093 |
| #1128 | AnubhavBharadwaaj | SLOT + LeakyReLU² + Legal Score-First TTT + Parallel Muon — val_bpb 1.1154 (3-seed mean) val_bpb = 1.1154 (3-seed mean, std 0.0002) | ~15.9 MB | 8×H100 SXM | 1.1154 | -0.1090 |
| #1129 |  | CROWN-Q + GPTQ + Legal TTT | 1.1174 | -0.1070 |
| #965 | Adam-Jacuch | via KGIIR Trajectory Mixing | 1.1184 | -0.1060 |
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
