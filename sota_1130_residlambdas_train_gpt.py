from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as flash_attn_3_func

_LEAKY_SLOPE = 0.5  # set from Hyperparameters.leaky_slope at startup

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 589_824))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.022))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.024))
    matrix_lr_late = float(os.environ.get("MATRIX_LR_LATE", os.environ.get("MATRIX_LR", "0.019")))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.020))
    scalar_lr_late = float(os.environ.get("SCALAR_LR_LATE", os.environ.get("SCALAR_LR", "0.038")))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.985))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 6144))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 7))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 196))
    ve_layers = os.environ.get("VE_LAYERS", "5,9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    leaky_slope = float(os.environ.get("LEAKY_SLOPE", 0.5))
    # MiLe loss
    mile_gamma = float(os.environ.get("MILE_GAMMA", 0.75))
    mile_clamp_min = float(os.environ.get("MILE_CLAMP_MIN", 0.2))
    mile_peak_frac = float(os.environ.get("MILE_PEAK_FRAC", 0.4))  # fraction of training where gamma peaks (triangle schedule)
    cache_layer = int(os.environ.get("CACHE_LAYER", 7))
    backout_init = float(os.environ.get("BACKOUT_INIT", 0.1))
    # GPTQ calibration
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    gptq_ar_calib = bool(int(os.environ.get("GPTQ_AR_CALIB", "0")))  # 1=AR self-gen, 0=train data
    gptq_reserve_ms = float(os.environ.get("GPTQ_RESERVE_MS", "14000"))  # ms reserved from training for GPTQ calibration

# --- Batched Newton-Schulz orthogonalization ---

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization. G: (B,M,N) or (M,N)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

# --- Parallel Muon optimizer ---

class Muon(torch.optim.Optimizer):
    """Parallel Muon: post-backward reduce-scatter -> local NS5 -> all-gather.

    No DDP for bank params. After backward, this optimizer:
    1. Launches async reduce-scatter for all banks (biggest first)
    2. Returns control so Adam can step on small params while RS is in-flight
    3. Waits for each RS, runs local NS5 on the shard, launches async all-gather
    4. Each all-gather overlaps with next bank's NS5
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size

        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        # Sort by size descending -- launch biggest reduce-scatters first
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        """Phase 1: launch async reduce-scatter for all banks. Call right after backward."""
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        """Phase 3: wait for RS, local NS5, all-gather. Call AFTER Adam steps."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            prev_ag_handle = None
            prev_m = None

            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return loss

# --- Tokenizer evaluation helpers ---

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# --- Quantization helpers ---

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda,resid_lambdas",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# --- Data loading (coprime-stride multi-shard, PR #726 / PR #1060 style) ---

_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}

def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n

def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class DistributedTokenLoader:
    """Coprime-stride + multi-shard loader (PR #726 / PR #1060 style)."""
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._num_tokens = np.array([_read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cfg: tuple[int, int, int, int] | None = None
        self._eligible_shards: np.ndarray | None = None
        self._base_block_counts: np.ndarray | None = None
        n = len(self.files)
        self._cursor_phase = np.zeros(n, dtype=np.int64)
        self._cursor_block_count = np.zeros(n, dtype=np.int64)
        self._cursor_next = np.zeros(n, dtype=np.int64)
        self._cursor_start = np.zeros(n, dtype=np.int64)
        self._cursor_stride = np.ones(n, dtype=np.int64)
        self._cursor_init = np.zeros(n, dtype=np.bool_)
        self._batches_built = 0

    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self._rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s

    def _reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self._num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = (nt - 1 - phase) // seq_len
        self._cursor_phase[si] = phase
        self._cursor_block_count[si] = bc
        self._cursor_next[si] = 0
        self._cursor_start[si] = int(self._rng.integers(bc)) if bc > 1 else 0
        self._cursor_stride[si] = self._pick_coprime_stride(bc)
        self._cursor_init[si] = True

    def _ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self._cursor_init[si] or self._cursor_next[si] >= self._cursor_block_count[si]:
            self._reset_cursor(si, seq_len)

    def _take_from_shard(self, si: int, seq_len: int, count: int, out: list[tuple[int, int]]) -> None:
        rem = count
        while rem > 0:
            self._ensure_cursor(si, seq_len)
            bc = int(self._cursor_block_count[si])
            ni = int(self._cursor_next[si])
            take = min(rem, bc - ni)
            phase = int(self._cursor_phase[si])
            start = int(self._cursor_start[si])
            stride = int(self._cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self._cursor_next[si] = ni + take
            rem -= take

    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        num_seqs = local_tokens // seq_len
        global_num_seqs = num_seqs * self.world_size
        self._cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)
        bbc = (self._num_tokens - 1) // seq_len
        eligible = bbc > 0
        self._eligible_shards = np.nonzero(eligible)[0].astype(np.int64)
        self._base_block_counts = bbc[self._eligible_shards].astype(np.int64)

    def _sample_global_windows(self) -> list[tuple[int, int]]:
        assert self._cfg is not None and self._eligible_shards is not None
        _, seq_len, _, gns = self._cfg
        ec = int(self._eligible_shards.size)
        progress = min(self._batches_built / 1800.0, 1.0)
        remaining = np.empty(ec, dtype=np.float64)
        for i, si in enumerate(self._eligible_shards.tolist()):
            if self._cursor_init[si]:
                r = int(self._cursor_block_count[si]) - int(self._cursor_next[si])
                remaining[i] = float(max(r, 1))
            else:
                remaining[i] = float(self._base_block_counts[i])
        alpha = 0.90 - 0.40 * progress
        weights = np.power(remaining, alpha)
        ws = float(weights.sum())
        if not np.isfinite(ws) or ws <= 0.0:
            weights = np.ones(ec, dtype=np.float64)
            ws = float(weights.sum())
        probs = weights / ws
        low = min(max(8, self.world_size), ec, gns)
        high = min(max(32, self.world_size * 8), ec, gns)
        mix = max(1, min(int(round(low + progress * (high - low))), ec, gns))
        cp = self._rng.choice(ec, size=mix, replace=False, p=probs)
        cs = self._eligible_shards[cp]
        cpr = probs[cp].copy()
        cpr /= cpr.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = gns - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, cpr).astype(np.int64)
        perm = self._rng.permutation(mix)
        cs, counts = cs[perm], counts[perm]
        buckets: list[list[tuple[int, int]]] = []
        for si, cnt in zip(cs.tolist(), counts.tolist()):
            b: list[tuple[int, int]] = []
            self._take_from_shard(int(si), seq_len, int(cnt), b)
            if b:
                if len(b) > 1:
                    bp = self._rng.permutation(len(b))
                    b = [b[int(k)] for k in bp.tolist()]
                buckets.append(b)
        windows: list[tuple[int, int]] = []
        active = [i for i, bk in enumerate(buckets) if bk]
        while active:
            order = self._rng.permutation(len(active))
            new_active: list[int] = []
            for oi in order.tolist():
                bi = active[oi]
                if buckets[bi]:
                    windows.append(buckets[bi].pop())
                if buckets[bi]:
                    new_active.append(bi)
            active = new_active
        return windows

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self._cfg is None:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        _, _, num_seqs, gns = self._cfg
        gw = self._sample_global_windows()
        local_w = gw[self.rank::self.world_size]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(local_w):
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self._batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# --- Transformer modules ---

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        # No CastedLinear -- weights come from banks
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
        # Gated attention and value residual (non-banked small params)
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] -- broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            # gate shape: (bsz, seqlen, num_heads) -> (bsz, seqlen, num_heads, 1) for B,T,H,D layout
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        # No CastedLinear -- weights come from banks
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=_LEAKY_SLOPE)
        return F.linear(x.square(), down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated_attention=gated_attention, value_residual=value_residual)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None, resid_lambda_attn: Tensor | None = None, resid_lambda_mlp: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
        if resid_lambda_attn is not None:
            x_out = resid_lambda_attn * x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        if resid_lambda_mlp is not None:
            x_out = resid_lambda_mlp * x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        else:
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        gated_attention: bool = False,
        value_residual: bool = False,
        mile_gamma: float = 0.8,
        mile_clamp_min: float = 0.1,
        cache_layer: int = 7,
        backout_init: float = 0.1,
    ):
        super().__init__()
        self.register_buffer('mile_gamma_buf', torch.tensor(mile_gamma, dtype=torch.float32))
        self.mile_clamp_min = mile_clamp_min
        self.cache_layer = cache_layer
        self.backout_lambda = nn.Parameter(torch.tensor(backout_init, dtype=torch.float32))
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.value_residual = value_residual
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5, dtype=torch.float32))
        # Parameter banks: split into early/late halves for separate LR
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self._bank_split = num_layers // 2
        s = self._bank_split
        n_l = num_layers - s
        self.qo_bank_early = nn.Parameter(torch.empty(2 * s, model_dim, model_dim))
        self.qo_bank_late = nn.Parameter(torch.empty(2 * n_l, model_dim, model_dim))
        self.kv_bank_early = nn.Parameter(torch.empty(2 * s, kv_dim, model_dim))
        self.kv_bank_late = nn.Parameter(torch.empty(2 * n_l, kv_dim, model_dim))
        self.mlp_up_bank_early = nn.Parameter(torch.empty(s, mlp_dim, model_dim))
        self.mlp_up_bank_late = nn.Parameter(torch.empty(n_l, mlp_dim, model_dim))
        self.mlp_down_bank_early = nn.Parameter(torch.empty(s, model_dim, mlp_dim))
        self.mlp_down_bank_late = nn.Parameter(torch.empty(n_l, model_dim, mlp_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                    gated_attention=gated_attention,
                    value_residual=value_residual,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _bank_idx(self, layer_i: int):
        """Return (qo_bank, qo_q_idx, qo_o_idx, kv_bank, kv_k_idx, kv_v_idx,
                   mlp_up_bank, mlp_up_idx, mlp_down_bank, mlp_down_idx) for a layer."""
        s = self._bank_split
        if layer_i < s:
            return (self.qo_bank_early, layer_i, s + layer_i,
                    self.kv_bank_early, layer_i, s + layer_i,
                    self.mlp_up_bank_early, layer_i,
                    self.mlp_down_bank_early, layer_i)
        else:
            j = layer_i - s
            n_l = self.num_layers - s
            return (self.qo_bank_late, j, n_l + j,
                    self.kv_bank_late, j, n_l + j,
                    self.mlp_up_bank_late, j,
                    self.mlp_down_bank_late, j)

    def _get_bank_weights(self, i: int):
        """Return (q_w, k_w, v_w, o_w, up_w, down_w) for layer i."""
        qo, qi, oi, kv, ki, vi, up, ui, down, di = self._bank_idx(i)
        return qo[qi], kv[ki], kv[vi], qo[oi], up[ui], down[di]

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        s = self._bank_split
        n_l = n - s
        proj_scale = 1.0 / math.sqrt(2 * n)
        # Init early banks
        for i in range(s):
            nn.init.orthogonal_(self.qo_bank_early.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank_early.data[s + i])
            nn.init.orthogonal_(self.kv_bank_early.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank_early.data[s + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank_early.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank_early.data[i])
            self.qo_bank_early.data[s + i].mul_(proj_scale)
            self.mlp_down_bank_early.data[i].mul_(proj_scale)
        # Init late banks
        for i in range(n_l):
            nn.init.orthogonal_(self.qo_bank_late.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank_late.data[n_l + i])
            nn.init.orthogonal_(self.kv_bank_late.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank_late.data[n_l + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank_late.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank_late.data[i])
            self.qo_bank_late.data[n_l + i].mul_(proj_scale)
            self.mlp_down_bank_late.data[i].mul_(proj_scale)
        # Init remaining nn.Linear modules (bigram proj, mtp heads, lm_head)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        """Get value embedding for a specific layer using shared table + per-layer scale."""
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        x_cache = None
        rl = self.resid_lambdas.to(dtype=x.dtype)
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            q_w, k_w, v_w, o_w, up_w, down_w = self._get_bank_weights(i)
            x, raw_v = self.blocks[i](x, x0,
                q_w, k_w, v_w, o_w, up_w, down_w,
                v_embed=ve, v0=v0,
                resid_lambda_attn=rl[i, 0], resid_lambda_mlp=rl[i, 1])
            if v0 is None and raw_v is not None:
                v0 = raw_v
            if i == self.cache_layer:
                x_cache = x.clone()
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            q_w, k_w, v_w, o_w, up_w, down_w = self._get_bank_weights(bi)
            x, _ = self.blocks[bi](x, x0,
                q_w, k_w, v_w, o_w, up_w, down_w,
                v_embed=ve, v0=v0,
                resid_lambda_attn=rl[bi, 0], resid_lambda_mlp=rl[bi, 1])
            if bi == self.cache_layer:
                x_cache = x.clone()
        if x_cache is not None:
            x = x - self.backout_lambda.to(dtype=x.dtype) * x_cache
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        logits_f = logits.float()
        if self.training:
            token_losses = F.cross_entropy(logits_f, targets, reduction="none")
            with torch.no_grad():
                gamma = self.mile_gamma_buf
                probs = torch.softmax(logits_f, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                weights = ((1.0 - torch.exp(-entropy)) ** gamma).clamp(min=self.mile_clamp_min)
            main_loss = (token_losses * weights).mean()
        else:
            main_loss = F.cross_entropy(logits_f, targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        x_cache = None
        rl = self.resid_lambdas.to(dtype=x.dtype)
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            q_w, k_w, v_w, o_w, up_w, down_w = self._get_bank_weights(i)
            x, raw_v = self.blocks[i](x, x0,
                q_w, k_w, v_w, o_w, up_w, down_w,
                v_embed=ve, v0=v0,
                resid_lambda_attn=rl[i, 0], resid_lambda_mlp=rl[i, 1])
            if v0 is None and raw_v is not None:
                v0 = raw_v
            if i == self.cache_layer:
                x_cache = x.clone()
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            q_w, k_w, v_w, o_w, up_w, down_w = self._get_bank_weights(bi)
            x, _ = self.blocks[bi](x, x0,
                q_w, k_w, v_w, o_w, up_w, down_w,
                v_embed=ve, v0=v0,
                resid_lambda_attn=rl[bi, 0], resid_lambda_mlp=rl[bi, 1])
            if bi == self.cache_layer:
                x_cache = x.clone()
        if x_cache is not None:
            x = x - self.backout_lambda.to(dtype=x.dtype) * x_cache
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

# --- Sliding window evaluation ---

def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    """Legal score-first TTT (PR #461 recipe): score each chunk with sliding windows,
    then train on it. Every token scored BEFORE any update that could use it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk based on the first token it scores
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride} "
         f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
         f"freeze_blocks={args.ttt_freeze_blocks}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze first N blocks
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE this chunk's windows (inference_mode) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


# --- GPTQ-lite int6 quantization ---

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
