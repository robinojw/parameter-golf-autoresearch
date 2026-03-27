import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

RUN_ID = os.getenv("RUN_ID", "local_mlx_run")
ITERATIONS = int(os.getenv("ITERATIONS", "500"))
TRAIN_BATCH_TOKENS = int(os.getenv("TRAIN_BATCH_TOKENS", "8192"))
VAL_LOSS_EVERY = int(os.getenv("VAL_LOSS_EVERY", "0"))
TRAIN_SEQ_LEN = int(os.getenv("TRAIN_SEQ_LEN", "512"))
MLX_EAGER_EVAL = int(os.getenv("MLX_EAGER_EVAL", "1"))

VOCAB_SIZE = 8192
N_LAYERS = 6
N_HEADS = 6
D_MODEL = 384
LR = 3e-4
BATCH_SIZE = TRAIN_BATCH_TOKENS // TRAIN_SEQ_LEN
MLP_EXPANSION = 4
LOG_2 = math.log(2)
LOG_INTERVAL = 50
HEADS_SEQ_TRANSPOSE = (0, 2, 1, 3)
KEY_TRANSPOSE = (0, 1, 3, 2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, hidden: mx.array) -> mx.array:
        batch, seq, channels = hidden.shape
        queries = (
            self.q_proj(hidden)
            .reshape(batch, seq, self.n_heads, self.head_dim)
            .transpose(*HEADS_SEQ_TRANSPOSE)
        )
        keys = (
            self.k_proj(hidden)
            .reshape(batch, seq, self.n_heads, self.head_dim)
            .transpose(*HEADS_SEQ_TRANSPOSE)
        )
        values = (
            self.v_proj(hidden)
            .reshape(batch, seq, self.n_heads, self.head_dim)
            .transpose(*HEADS_SEQ_TRANSPOSE)
        )
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq).astype(
            queries.dtype
        )
        scale = math.sqrt(self.head_dim)
        scores = (queries @ keys.transpose(*KEY_TRANSPOSE)) / scale + mask
        weights = mx.softmax(scores, axis=-1)
        attn_out = (
            (weights @ values)
            .transpose(*HEADS_SEQ_TRANSPOSE)
            .reshape(batch, seq, channels)
        )
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * MLP_EXPANSION),
            nn.GELU(),
            nn.Linear(d_model * MLP_EXPANSION, d_model),
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden + self.attn(self.ln1(hidden))
        hidden = hidden + self.mlp(self.ln2(hidden))
        return hidden


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, seq_len: int
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, idx: mx.array) -> mx.array:
        batch, seq = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(mx.arange(seq))
        out = tok + pos
        for block in self.blocks:
            out = block(out)
        out = self.ln_f(out)
        return self.head(out)


def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).mean()


def get_batch(
    seq_len: int, batch_size: int, vocab_size: int
) -> tuple[mx.array, mx.array]:
    inputs = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
    targets = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
    return inputs, targets


def get_artifact_bytes() -> int:
    return Path(__file__).stat().st_size


def _run_eager_step(
    model: GPT,
    optimizer: optim.AdamW,
    loss_and_grad,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    loss, grads = loss_and_grad(model, inputs, targets)
    optimizer.apply_gradients(grads, model)
    mx.eval(model.parameters(), optimizer.state)
    return loss


def main() -> None:
    if MLX_EAGER_EVAL:
        mx.disable_compile()

    model = GPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, TRAIN_SEQ_LEN)
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=LR)

    loss_and_grad = nn.value_and_grad(
        model, lambda mdl, inp, tgt: cross_entropy(mdl(inp), tgt)
    )

    @mx.compile
    def train_step(model_params, optimizer_state, inputs, targets):
        model.update(model_params)
        optimizer.state = optimizer_state
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.apply_gradients(grads, model)
        return loss, model.parameters(), optimizer.state

    model_params = model.parameters()
    optimizer_state = optimizer.state

    t0 = time.time()
    val_loss = float("nan")

    for step in range(1, ITERATIONS + 1):
        inputs, targets = get_batch(TRAIN_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)

        if MLX_EAGER_EVAL:
            loss = _run_eager_step(model, optimizer, loss_and_grad, inputs, targets)
        else:
            loss, model_params, optimizer_state = train_step(
                model_params, optimizer_state, inputs, targets
            )
            mx.eval(loss)

        should_validate = VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0
        if should_validate:
            val_inputs, val_targets = get_batch(TRAIN_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)
            val_logits = model(val_inputs)
            val_loss = cross_entropy(val_logits, val_targets).item()
            print(
                f"step {step}/{ITERATIONS}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}"
            )
        elif step % LOG_INTERVAL == 0:
            print(f"step {step}/{ITERATIONS}  train_loss={loss.item():.4f}")

    if math.isnan(val_loss):
        val_inputs, val_targets = get_batch(TRAIN_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)
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
