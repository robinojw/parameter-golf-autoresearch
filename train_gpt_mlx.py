"""MLX adaptation of the Parameter Golf stack (Rascal/PR #1120 base, 1.1099 bpb).

Architecture: 11L 512d 8H 4KV, GQA, Partial RoPE (16/64), RMSNorm,
LeakyReLU(0.75)^2, XSA-all (11L), EngramLite, SmearGate, U-Net skips (sigmoid-gated),
ValueEmbedding, LN Scale, logit softcap, tied embeddings, coprime-stride loader.

Optimizer: Standard NS5 Muon (Parallel Muon + AdamW). Turbo-Muon removed —
confirmed +0.0018 BPB worse at H100 scale and over 16MB (PR #1105 negative result).
Purpose: directional signal only. val_bpb NOT the challenge score.
"""
import glob
import math
import os
import struct
import time
from math import gcd
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters (match SOTA PR #1089)
# ---------------------------------------------------------------------------
RUN_ID = os.getenv("RUN_ID", "local_mlx_run")
ITERATIONS = int(os.getenv("ITERATIONS", "500"))
TRAIN_BATCH_TOKENS = int(os.getenv("TRAIN_BATCH_TOKENS", "8192"))
VAL_LOSS_EVERY = int(os.getenv("VAL_LOSS_EVERY", "0"))
TRAIN_SEQ_LEN = int(os.getenv("TRAIN_SEQ_LEN", "512"))
MLX_EAGER_EVAL = int(os.getenv("MLX_EAGER_EVAL", "1"))

DATA_PATH = os.getenv("DATA_PATH", "data/datasets/fineweb10B_sp1024")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "1024"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "11"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "8"))
NUM_KV_HEADS = int(os.getenv("NUM_KV_HEADS", "4"))
MODEL_DIM = int(os.getenv("MODEL_DIM", "512"))
MLP_MULT = float(os.getenv("MLP_MULT", "3.5"))
LR = float(os.getenv("LR", "3e-4"))
LOGIT_SOFTCAP = float(os.getenv("LOGIT_SOFTCAP", "30.0"))
ROPE_BASE = float(os.getenv("ROPE_BASE", "10000.0"))
ROPE_DIMS = int(os.getenv("ROPE_DIMS", "16"))
QK_GAIN_INIT = float(os.getenv("QK_GAIN_INIT", "4.0"))
LEAKY_SLOPE = float(os.getenv("LEAKY_SLOPE", "0.75"))
# EngramLite params
NGRAM_BUCKETS = int(os.getenv("NGRAM_BUCKETS", "8192"))
NGRAM_HEADS = int(os.getenv("NGRAM_HEADS", "2"))
NGRAM_ORDERS = int(os.getenv("NGRAM_ORDERS", "2"))
NGRAM_DIM_PER_HEAD = int(os.getenv("NGRAM_DIM_PER_HEAD", "32"))
# ValueEmbedding
VE_ENABLED = bool(int(os.getenv("VE_ENABLED", "1")))
VE_DIM = int(os.getenv("VE_DIM", "128"))
VE_LAYERS = os.getenv("VE_LAYERS", "9,10")
# XSA
XSA_LAST_N = int(os.getenv("XSA_LAST_N", "11"))

WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "20"))
WARMDOWN_FRAC = float(os.getenv("WARMDOWN_FRAC", "0.3"))  # last 30% of training
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", "0.3"))
# Muon optimizer
USE_MUON = bool(int(os.getenv("USE_MUON", "1")))
MUON_LR = float(os.getenv("MUON_LR", "0.025"))
MUON_MOMENTUM = float(os.getenv("MUON_MOMENTUM", "0.85"))
MUON_MOMENTUM_END = float(os.getenv("MUON_MOMENTUM_END", "0.99"))
MUON_WARMUP = int(os.getenv("MUON_WARMUP", "500"))
MUON_WD = float(os.getenv("MUON_WD", "0.04"))
MUON_NS_STEPS = int(os.getenv("MUON_NS_STEPS", "5"))
EMA_DECAY = float(os.getenv("EMA_DECAY", "0.995"))
EMA_START = int(os.getenv("EMA_START", "100"))  # start EMA after step 100
EMA_EVAL_EVERY = int(os.getenv("EMA_EVAL_EVERY", "50"))  # force eval EMA every N steps
# Coprime-stride data loader
COPRIME_LOADER = bool(int(os.getenv("COPRIME_LOADER", "1")))
# WARMDOWN_ITERS: absolute count (0 = use WARMDOWN_FRAC fraction instead)
WARMDOWN_ITERS = int(os.getenv("WARMDOWN_ITERS", "0"))

BATCH_SIZE = TRAIN_BATCH_TOKENS // TRAIN_SEQ_LEN
LOG_2 = math.log(2)
LOG_INTERVAL = 50

# ---------------------------------------------------------------------------
# Data loading (FineWeb .bin shards)
# ---------------------------------------------------------------------------

def _load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    offset = 256 * 4
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)
    return tokens


class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = _load_shard(Path(self.files[0]))
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_shard(Path(self.files[self.file_idx]))
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = len(self.tokens) - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class CoprimeTokenStream:
    """Diversity-weighted shard sampling with coprime stride (PR #1135 technique)."""

    def __init__(self, pattern: str, total_steps: int = 500,
                 alpha_start: float = 0.90, alpha_end: float = 0.50):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files for pattern: {pattern}")
        self.total_steps = max(1, total_steps)
        self.step = 0
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

        # Load shard lengths from headers
        self.shard_lengths = []
        for f in self.files:
            hdr = np.fromfile(f, dtype="<i4", count=256)
            self.shard_lengths.append(int(hdr[2]))

        rng = np.random.default_rng()
        # Random phase init: don't always start at position 0
        self.positions = [int(rng.integers(0, max(1, l))) for l in self.shard_lengths]
        # Coprime stride ≈ shard_len//2: ensures full coverage over training
        self.strides = [self._coprime_stride(l) for l in self.shard_lengths]
        self._cache_idx = -1
        self._cache = None

    def _coprime_stride(self, n: int) -> int:
        # Stride ≈ n/total_steps so 500 batches cover the full shard
        target = max(1, n // max(1, self.total_steps))
        s = target
        while gcd(s, n) != 1:
            s += 1
        return s

    def _alpha(self) -> float:
        progress = min(1.0, self.step / self.total_steps)
        return self.alpha_start - (self.alpha_start - self.alpha_end) * progress

    def _select_shard(self) -> int:
        if len(self.files) == 1:
            return 0
        alpha = self._alpha()
        w = np.array(self.shard_lengths, dtype=np.float64) ** alpha
        w /= w.sum()
        return int(np.random.choice(len(self.files), p=w))

    def _load(self, idx: int) -> np.ndarray:
        if idx != self._cache_idx:
            hdr = np.fromfile(self.files[idx], dtype="<i4", count=256)
            n = int(hdr[2])
            self._cache = np.fromfile(self.files[idx], dtype="<u2", count=n, offset=1024)
            self._cache_idx = idx
        return self._cache

    def take(self, n: int) -> np.ndarray:
        self.step += 1
        si = self._select_shard()
        tok = self._load(si)
        L = len(tok)
        p = self.positions[si]
        if p + n <= L:
            out = tok[p:p + n].copy()
        else:
            out = np.concatenate([tok[p:], tok[:n - (L - p)]])
        self.positions[si] = (p + self.strides[si]) % L
        return out.astype(np.int32)


def _have_data() -> bool:
    train_pattern = os.path.join(DATA_PATH, "fineweb_train_*.bin")
    return len(glob.glob(train_pattern)) > 0


def get_batch_real(stream, seq_len: int, batch_size: int):
    total = batch_size * seq_len + 1
    chunk = stream.take(total).astype(np.int32)
    x = chunk[:-1].reshape(batch_size, seq_len)
    y = chunk[1:].reshape(batch_size, seq_len)
    return mx.array(x), mx.array(y)


def get_batch_random(seq_len: int, batch_size: int, vocab_size: int):
    x = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
    y = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
    return x, y


def load_val_tokens() -> mx.array | None:
    """Load validation tokens for eval_mode evaluation only (no gradients)."""
    val_pattern = os.path.join(DATA_PATH, "fineweb_val_*.bin")
    val_files = sorted(glob.glob(val_pattern))
    if not val_files:
        return None
    chunks = [_load_shard(Path(f)) for f in val_files]
    tokens = np.concatenate(chunks).astype(np.int32)
    return mx.array(tokens)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, eps=1e-5)


def rms_norm_no_weight(x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (mx.arange(0, self.rope_dims, 2).astype(mx.float32) / self.rope_dims))
        self.inv_freq = inv_freq

    def __call__(self, seq_len: int):
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        return cos, sin


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array, rope_dims: int = 0) -> mx.array:
    """Apply RoPE to first rope_dims dimensions only (partial RoPE)."""
    if rope_dims > 0 and rope_dims < x.shape[-1]:
        x_rope = x[..., :rope_dims]
        x_pass = x[..., rope_dims:]
        half = rope_dims // 2
        x1 = x_rope[..., :half]
        x2 = x_rope[..., half:]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        rotated = mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)
        return mx.concatenate([rotated, x_pass], axis=-1)
    else:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


class SmearGate(nn.Module):
    """Causal shift blending with predecessor token."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate)[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        return (1 - g) * x + g * x_prev


class EngramLite(nn.Module):
    """Multi-head prime-based hash n-gram embedding with learned gating."""
    def __init__(self, num_buckets: int, num_heads: int, num_orders: int,
                 dim_per_head: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.num_orders = num_orders
        self.dim_per_head = dim_per_head
        total_slots = num_orders * num_heads * num_buckets
        concat_dim = num_orders * num_heads * dim_per_head
        self.embed = nn.Embedding(total_slots, dim_per_head)
        self.proj = nn.Linear(concat_dim, model_dim, bias=False)
        self.ngram_gate = mx.zeros((model_dim,))

    def __call__(self, input_ids: mx.array) -> mx.array:
        B = self.num_buckets
        prev_ids = mx.concatenate([mx.zeros_like(input_ids[:, :1]), input_ids[:, :-1]], axis=1)
        # Bigram hashes (2 heads)
        bi_h0 = (prev_ids * 1009 + input_ids) % B
        bi_h1 = ((prev_ids * 2719 + 314159) ^ (input_ids * 3137)) % B
        indices = [bi_h0, bi_h1 + B]
        # Trigram hashes (2 heads) if enabled
        if self.num_orders >= 2:
            pp_ids = mx.concatenate([mx.zeros_like(prev_ids[:, :1]), prev_ids[:, :-1]], axis=1)
            tri_h0 = ((pp_ids * 36313) ^ (prev_ids * 27191) ^ (input_ids * 4903)) % B
            tri_h1 = ((pp_ids * 7919) ^ (prev_ids * 4391) ^ (input_ids * 6151)) % B
            offset = 2 * B
            indices.extend([tri_h0 + offset, tri_h1 + offset + B])
        all_idx = mx.stack(indices, axis=-1)
        all_emb = self.embed(all_idx)
        flat = all_emb.reshape(*input_ids.shape, -1)
        out = self.proj(flat)
        gate = mx.sigmoid(self.ngram_gate)[None, None, :]
        return out * gate


class ValueEmbedding(nn.Module):
    """Token identity reinjection at specific layers."""
    def __init__(self, vocab_size: int, ve_dim: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        self.proj = nn.Linear(ve_dim, kv_dim, bias=False) if ve_dim != kv_dim else None
        self.scale = mx.array(0.1)

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, rope_dims: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = mx.full((num_heads,), qk_gain_init)
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=rope_dims)
        self.use_xsa = False

    def _xsa_efficient(self, y: mx.array, v: mx.array) -> mx.array:
        """XSA: subtract self-value projection via GQA-aware reshape."""
        B, T, H, D = y.shape
        Hkv = v.shape[2]
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        # Normalize v
        v_norm = v * mx.rsqrt(mx.sum(v * v, axis=-1, keepdims=True) + 1e-6)
        vn = mx.expand_dims(v_norm, axis=-2)  # [B, T, Hkv, 1, D]
        proj = mx.sum(y_g * vn, axis=-1, keepdims=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def __call__(self, x: mx.array, v_embed: mx.array = None) -> mx.array:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        # QK normalization (RMS norm per head)
        q = rms_norm_no_weight(q)
        k = rms_norm_no_weight(k)

        # Partial RoPE
        cos, sin = self.rotary(T)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)

        # Q gain
        q = q * self.q_gain[None, None, :, None]

        # Transpose to (B, H, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)

        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, reps, axis=1)
            v_t = mx.repeat(v_t, reps, axis=1)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(scores.dtype)
        scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v_t

        # Back to (B, T, H, D)
        out = out.transpose(0, 2, 1, 3)

        # XSA: subtract self-value projection
        if self.use_xsa:
            out = self._xsa_efficient(out, v)

        out = out.reshape(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, leaky_slope: float = 0.3):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj_down = nn.Linear(hidden, dim, bias=False)
        self.leaky_slope = leaky_slope

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        # LeakyReLU(slope)^2
        x = mx.where(x >= 0, x, self.leaky_slope * x)
        return self.proj_down(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, rope_dims: int = 0,
                 layer_idx: int = 0, ln_scale: bool = True,
                 leaky_slope: float = 0.3):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                         qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult, leaky_slope=leaky_slope)
        self.attn_scale = mx.ones((dim,))
        self.mlp_scale = mx.ones((dim,))
        self.resid_mix = mx.stack([mx.ones((dim,)), mx.zeros((dim,))])
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def __call__(self, x: mx.array, x0: mx.array, v_embed: mx.array = None) -> mx.array:
        x_in = self.resid_mix[0][None, None, :] * x + self.resid_mix[1][None, None, :] * x0
        normed = self.attn_norm(x_in)
        if self.ln_scale_factor != 1.0:
            normed = normed * self.ln_scale_factor
        attn_out = self.attn(normed, v_embed=v_embed)
        x_out = x_in + self.attn_scale[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x_out)
        if self.ln_scale_factor != 1.0:
            mlp_normed = mlp_normed * self.ln_scale_factor
        mlp_out = self.mlp(mlp_normed)
        x_out = x_out + self.mlp_scale[None, None, :] * mlp_out
        return x_out


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, logit_softcap: float,
                 rope_dims: int = 0, leaky_slope: float = 0.3,
                 ngram_buckets: int = 0, ngram_heads: int = 2,
                 ngram_orders: int = 2, ngram_dim_per_head: int = 32,
                 xsa_last_n: int = 0, ln_scale: bool = True,
                 ve_enabled: bool = False, ve_dim: int = 128,
                 ve_layers: str = "9,10"):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # EngramLite
        self.engram = EngramLite(ngram_buckets, ngram_heads, ngram_orders,
                                  ngram_dim_per_head, model_dim) if ngram_buckets > 0 else None

        # SmearGate
        self.smear = SmearGate(model_dim)

        # U-Net architecture
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, model_dim))
        self.skip_gates = mx.zeros((self.num_skip_weights, model_dim))

        # Transformer blocks
        self.blocks = [
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, rope_dims=rope_dims, layer_idx=i,
                  ln_scale=ln_scale, leaky_slope=leaky_slope)
            for i in range(num_layers)
        ]

        # XSA on last N layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        # ValueEmbedding
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = num_kv_heads * (model_dim // num_heads)
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = [mx.array(1.0) for _ in self.ve_layer_indices]
        else:
            self.ve_shared = None
            self.ve_layer_scales = []

        self.final_norm = RMSNorm(model_dim)
        self._init_weights(num_layers)

    def _init_weights(self, n_layers: int):
        """Orthogonal init for Q/K/V, zero init for output projections."""
        proj_scale = 1.0 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            attn = block.attn
            # Q, K, V: orthogonal
            for lin in [attn.c_q, attn.c_k, attn.c_v]:
                w = lin.weight
                if w.ndim == 2:
                    lin.weight = orthogonal_init(w.shape)
            # Out proj: zero init scaled
            attn.proj.weight = orthogonal_init(attn.proj.weight.shape, gain=proj_scale)
            # MLP up: orthogonal, down: zero init scaled
            block.mlp.fc.weight = orthogonal_init(block.mlp.fc.weight.shape)
            block.mlp.proj_down.weight = orthogonal_init(
                block.mlp.proj_down.weight.shape, gain=proj_scale
            )
        # Token embedding: small std
        self.tok_emb.weight = mx.random.normal(self.tok_emb.weight.shape) * 0.005

    def _get_ve(self, layer_idx: int, ve_base: mx.array) -> mx.array | None:
        if ve_base is None or layer_idx not in self.ve_layer_indices:
            return None
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx]

    def __call__(self, idx: mx.array) -> mx.array:
        B, T = idx.shape
        x = self.tok_emb(idx)

        # EngramLite
        if self.engram is not None:
            x = x + self.engram(idx)

        # RMS normalize initial embedding
        x = rms_norm_no_weight(x)

        # SmearGate
        x = self.smear(x)
        x0 = x
        skips = []

        # Precompute value embedding base
        ve_base = self.ve_shared(idx) if self.ve_shared is not None else None

        # Encoder half
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, ve_base)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)

        # Decoder half with sigmoid-gated U-Net skip connections
        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                g = mx.sigmoid(self.skip_gates[i])[None, None, :]
                scaled_skip = self.skip_weights[i][None, None, :] * skip
                x = (1 - g) * scaled_skip + g * x
            bi = self.num_encoder_layers + i
            ve = self._get_ve(bi, ve_base)
            x = self.blocks[bi](x, x0, v_embed=ve)

        x = self.final_norm(x)
        # Tied embeddings
        logits = x @ self.tok_emb.weight.T
        # Logit softcapping
        logits = self.logit_softcap * mx.tanh(logits / self.logit_softcap)
        return logits


# ---------------------------------------------------------------------------
# Loss and training
# ---------------------------------------------------------------------------

def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).mean()


def get_artifact_bytes() -> int:
    return Path(__file__).stat().st_size


def get_lr(step: int, total_steps: int, base_lr: float) -> float:
    """Warmup + cosine decay with warmdown. Supports absolute WARMDOWN_ITERS."""
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    if WARMDOWN_ITERS > 0:
        warmdown_start = max(WARMUP_STEPS, total_steps - WARMDOWN_ITERS)
    else:
        warmdown_start = int(total_steps * (1 - WARMDOWN_FRAC))
    if step >= warmdown_start:
        progress = (step - warmdown_start) / max(1, total_steps - warmdown_start)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr


def clip_grad_norm(grads, max_norm: float):
    """Clip gradient norm in-place, returns clipped grads tree."""
    def _sq_sum(tree):
        if isinstance(tree, mx.array):
            return mx.sum(tree * tree)
        if isinstance(tree, dict):
            return sum(_sq_sum(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(_sq_sum(v) for v in tree)
        return mx.array(0.0)

    def _scale(tree, s):
        if isinstance(tree, mx.array):
            return tree * s
        if isinstance(tree, dict):
            return {k: _scale(v, s) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(_scale(v, s) for v in tree)
        return tree

    total_sq = _sq_sum(grads)
    total_norm = mx.sqrt(total_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))
    return _scale(grads, clip_coef)


def orthogonal_init(shape, gain: float = 1.0) -> mx.array:
    """Orthogonal initialization for 2D weight matrices."""
    rows, cols = shape
    n = max(rows, cols)
    flat = np.random.randn(n, n).astype(np.float32)
    q, _ = np.linalg.qr(flat)
    return mx.array(q[:rows, :cols] * gain)


MAX_VAL_TOKENS = int(os.getenv("MAX_VAL_TOKENS", "524288"))  # 512K tokens max for local eval


# ---------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalization)
# ---------------------------------------------------------------------------

def zeropower_via_ns(G: mx.array, steps: int = 5) -> mx.array:
    """Standard Newton-Schulz 5-step orthogonalization (Muon optimizer).

    Uses cubic NS iteration: X_{k+1} = (15/8)X - (5/4)(XX^T)X + (3/8)(XX^T)^2 X
    Converges to the polar factor of G. Used by Rascal (1.1099 bpb, PR #1120).
    Turbo-Muon (AOL + Polar Express) was removed: +0.0018 BPB worse at H100 scale
    and exceeds 16MB artifact budget (PR #1105 negative result).
    """
    norm = mx.sqrt(mx.sum(G * G)) + 1e-7
    X = G / norm
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = 15.0 / 8.0 * X - 5.0 / 4.0 * (A @ X) + 3.0 / 8.0 * (A @ (A @ X))
    if transposed:
        X = X.T
    return X


def _get_nested(d, path: tuple):
    for key in path:
        d = d[key]
    return d


def _set_nested(d, path: tuple, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def build_muon_paths(num_layers: int, has_engram: bool, ve_enabled: bool, ve_has_proj: bool) -> list:
    """List of parameter dict paths for 2D linear weights (Muon targets)."""
    paths = []
    for i in range(num_layers):
        for sub in ["c_q", "c_k", "c_v", "proj"]:
            paths.append(("blocks", i, "attn", sub, "weight"))
        paths.append(("blocks", i, "mlp", "fc", "weight"))
        paths.append(("blocks", i, "mlp", "proj_down", "weight"))
    if has_engram:
        paths.append(("engram", "proj", "weight"))
    if ve_enabled and ve_has_proj:
        paths.append(("ve_shared", "proj", "weight"))
    return paths


class MuonState:
    """Momentum buffers for Muon optimizer."""
    def __init__(self, paths: list, params: dict):
        self.paths = paths
        self.bufs = {path: mx.zeros_like(_get_nested(params, path)) for path in paths}
        self.step = 0


def muon_apply(model, grads: dict, state: MuonState, lr: float, wd: float) -> None:
    """Apply Muon update to all 2D linear weights. Zero their grads in-place."""
    params = model.parameters()
    step = state.step
    state.step += 1
    # Momentum warmup: 0.85 -> 0.99 over MUON_WARMUP steps
    mom = MUON_MOMENTUM + (MUON_MOMENTUM_END - MUON_MOMENTUM) * min(1.0, step / max(1, MUON_WARMUP))

    for path in state.paths:
        grad = _get_nested(grads, path)
        param = _get_nested(params, path)
        buf = state.bufs[path]

        buf = mom * buf + grad
        state.bufs[path] = buf

        update = grad + mom * buf  # Nesterov
        update = zeropower_via_ns(update, MUON_NS_STEPS)

        M, N = param.shape
        scale = math.sqrt(max(1.0, M / N))

        _set_nested(params, path, param * (1.0 - lr * wd) - lr * scale * update)
        _set_nested(grads, path, mx.zeros_like(grad))  # prevent double-update via AdamW

    model.update(params)


def eval_val(model: GPT, val_tokens: mx.array, seq_len: int, batch_size: int) -> float:
    total_tokens = min(val_tokens.shape[0], MAX_VAL_TOKENS)
    usable = ((total_tokens - 1) // seq_len) * seq_len
    total_seqs = usable // seq_len
    total_loss = 0.0
    count = 0
    for start_seq in range(0, total_seqs, batch_size):
        end_seq = min(start_seq + batch_size, total_seqs)
        raw_start = start_seq * seq_len
        raw_end = end_seq * seq_len + 1
        local = val_tokens[raw_start:raw_end]
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        logits = model(x)
        loss = cross_entropy(logits, y)
        mx.eval(loss)
        n_tokens = (end_seq - start_seq) * seq_len
        total_loss += loss.item() * n_tokens
        count += n_tokens
    return total_loss / max(count, 1)


def main() -> None:
    if MLX_EAGER_EVAL:
        mx.disable_compile()

    model = GPT(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        mlp_mult=MLP_MULT,
        rope_base=ROPE_BASE,
        qk_gain_init=QK_GAIN_INIT,
        logit_softcap=LOGIT_SOFTCAP,
        rope_dims=ROPE_DIMS,
        leaky_slope=LEAKY_SLOPE,
        ngram_buckets=NGRAM_BUCKETS,
        ngram_heads=NGRAM_HEADS,
        ngram_orders=NGRAM_ORDERS,
        ngram_dim_per_head=NGRAM_DIM_PER_HEAD,
        xsa_last_n=XSA_LAST_N,
        ln_scale=True,
        ve_enabled=VE_ENABLED,
        ve_dim=VE_DIM,
        ve_layers=VE_LAYERS,
    )
    mx.eval(model.parameters())

    def _count_params(tree):
        if isinstance(tree, mx.array):
            return tree.size
        if isinstance(tree, dict):
            return sum(_count_params(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(_count_params(v) for v in tree)
        return 0
    n_params = _count_params(model.parameters())
    print(f"model_params: {n_params}")

    # Muon for 2D linear weights, AdamW (no WD) for everything else
    muon_state = None
    if USE_MUON:
        ve_has_proj = VE_ENABLED and VE_DIM != (NUM_KV_HEADS * (MODEL_DIM // NUM_HEADS))
        muon_paths = build_muon_paths(NUM_LAYERS, NGRAM_BUCKETS > 0, VE_ENABLED, ve_has_proj)
        muon_state = MuonState(muon_paths, model.parameters())
        print(f"muon: enabled, {len(muon_paths)} param groups")
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.0 if USE_MUON else 0.04)

    loss_and_grad = nn.value_and_grad(
        model, lambda mdl, inp, tgt: cross_entropy(mdl(inp), tgt)
    )

    # Data setup
    has_data = _have_data()
    train_stream = None
    val_tokens = None
    if has_data:
        train_pattern = os.path.join(DATA_PATH, "fineweb_train_*.bin")
        if COPRIME_LOADER:
            train_stream = CoprimeTokenStream(train_pattern, total_steps=ITERATIONS)
            print(f"coprime_loader: enabled, stride={train_stream.strides[0]}, phase={train_stream.positions[0]}")
        else:
            train_stream = TokenStream(train_pattern)
        val_tokens = load_val_tokens()
        print(f"data: loaded from {DATA_PATH}")
        if val_tokens is not None:
            print(f"val_tokens: {val_tokens.shape[0]}")
    else:
        print("[warn] No FineWeb data found, using random data")

    # EMA model state
    ema_params = None

    def _deep_copy_params(params):
        if isinstance(params, mx.array):
            return mx.array(params)
        if isinstance(params, dict):
            return {k: _deep_copy_params(v) for k, v in params.items()}
        if isinstance(params, (list, tuple)):
            return type(params)(_deep_copy_params(v) for v in params)
        return params

    def _ema_update(ema, params, decay):
        if isinstance(params, mx.array):
            return decay * ema + (1 - decay) * params
        if isinstance(params, dict):
            return {k: _ema_update(ema[k], params[k], decay) for k in params}
        if isinstance(params, (list, tuple)):
            return type(params)(_ema_update(e, p, decay) for e, p in zip(ema, params))
        return params

    t0 = time.time()
    val_loss = float("nan")

    for step in range(1, ITERATIONS + 1):
        # LR schedule
        lr = get_lr(step - 1, ITERATIONS, LR)
        muon_lr = get_lr(step - 1, ITERATIONS, MUON_LR) if USE_MUON else 0.0
        optimizer.learning_rate = lr

        if has_data and train_stream is not None:
            inputs, targets = get_batch_real(train_stream, TRAIN_SEQ_LEN, BATCH_SIZE)
        else:
            inputs, targets = get_batch_random(TRAIN_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)

        loss, grads = loss_and_grad(model, inputs, targets)

        # Gradient clipping
        if GRAD_CLIP_NORM > 0:
            grads = clip_grad_norm(grads, GRAD_CLIP_NORM)

        if USE_MUON and muon_state is not None:
            muon_apply(model, grads, muon_state, muon_lr, MUON_WD)
        optimizer.apply_gradients(grads, model)
        mx.eval(model.parameters(), optimizer.state)

        # EMA update (defer eval to every EMA_EVAL_EVERY steps to avoid per-step overhead)
        if step >= EMA_START:
            if ema_params is None:
                ema_params = _deep_copy_params(model.parameters())
                mx.eval(ema_params)
            else:
                ema_params = _ema_update(ema_params, model.parameters(), EMA_DECAY)
                if step % EMA_EVAL_EVERY == 0:
                    mx.eval(ema_params)

        should_validate = VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0
        if should_validate:
            if val_tokens is not None:
                val_loss = eval_val(model, val_tokens, TRAIN_SEQ_LEN, BATCH_SIZE)
            else:
                val_inputs, val_targets = get_batch_random(TRAIN_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)
                val_logits = model(val_inputs)
                val_loss = cross_entropy(val_logits, val_targets).item()
            print(
                f"step {step}/{ITERATIONS}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}"
            )
        elif step % LOG_INTERVAL == 0:
            print(f"step {step}/{ITERATIONS}  train_loss={loss.item():.4f}  lr={lr:.6f}")

    # Swap in EMA params for final eval if available
    train_params = model.parameters()
    if ema_params is not None:
        model.update(ema_params)
        mx.eval(model.parameters())
        print("using EMA params for final eval")

    # Final validation
    if val_tokens is not None:
        val_loss = eval_val(model, val_tokens, TRAIN_SEQ_LEN, BATCH_SIZE)
    elif math.isnan(val_loss):
        val_inputs, val_targets = get_batch_random(TRAIN_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)
        val_logits = model(val_inputs)
        val_loss = cross_entropy(val_logits, val_targets).item()

    training_seconds = time.time() - t0
    val_bpb = val_loss / LOG_2
    artifact_bytes = get_artifact_bytes()

    print(f"val_bpb:           {val_bpb:.6f}")
    print(f"val_loss:          {val_loss:.6f}")
    print(f"artifact_bytes:    {artifact_bytes}")
    print(f"training_seconds:  {training_seconds:.1f}")


if __name__ == "__main__":
    main()
