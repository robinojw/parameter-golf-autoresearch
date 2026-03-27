import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

RUN_ID = os.getenv("RUN_ID", "local_run")
DATA_PATH = os.getenv("DATA_PATH", "data/train.bin")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "data/tokenizer.json")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "8192"))

N_LAYERS = 6
N_HEADS = 6
D_MODEL = 384
SEQ_LEN = 512
BATCH_SIZE = 16
ITERATIONS = 1000
LR = 3e-4
VAL_EVERY = 100
VAL_STEPS = 10
QKV_FACTOR = 3
MLP_EXPANSION = 4
QKV_PERMUTE = (2, 0, 3, 1, 4)
TRANSPOSE_DIMS = (1, 2)
NAN_INIT = float("nan")
LOG_2 = math.log(2)
INITIAL_LOSS = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * QKV_FACTOR)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = hidden.shape
        qkv = (
            self.qkv(hidden)
            .reshape(batch, seq, QKV_FACTOR, self.n_heads, self.head_dim)
            .permute(*QKV_PERMUTE)
        )
        queries, keys, values = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attn_out = attn_out.transpose(*TRANSPOSE_DIMS).reshape(batch, seq, channels)
        return self.proj(attn_out)


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

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
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
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch, seq = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(seq, device=idx.device))
        out = self.blocks(tok + pos)
        out = self.ln_f(out)
        return self.head(out)


class TokenDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int) -> None:
        import numpy as np

        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(1, len(self.data) // self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype("int64")
        )
        return chunk[:-1], chunk[1:]


def estimate_loss(
    model: nn.Module, val_loader: DataLoader, device: torch.device, steps: int
) -> float:
    model.eval()
    total_loss = INITIAL_LOSS
    count = 0
    with torch.no_grad():
        for step_idx, (inputs, targets) in enumerate(val_loader):
            if step_idx >= steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


def get_artifact_bytes() -> int:
    return Path(__file__).stat().st_size


def _setup_distributed() -> tuple[torch.device, bool]:
    distributed_available = torch.distributed.is_available() and "RANK" in os.environ
    if not distributed_available:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, True
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return device, local_rank == 0


def _load_data() -> tuple[DataLoader | None, bool]:
    if not Path(DATA_PATH).exists():
        return None, False
    dataset = TokenDataset(DATA_PATH, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return loader, True


def _get_batch(
    loader: DataLoader | None, data_iter: iter, has_data: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, iter]:
    if has_data:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)
    else:
        inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    return inputs.to(device), targets.to(device), data_iter


def _print_results(val_loss: float, training_seconds: float) -> None:
    val_bpb = val_loss / LOG_2 if not math.isnan(val_loss) else NAN_INIT
    artifact_bytes = get_artifact_bytes()
    print(f"val_bpb:           {val_bpb:.6f}")
    print(f"val_loss:          {val_loss:.6f}")
    print(f"artifact_bytes:    {artifact_bytes}")
    print(f"training_seconds:  {training_seconds:.1f}")


def main() -> None:
    device, is_master = _setup_distributed()

    model = GPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loader, has_data = _load_data()
    should_warn_no_data = not has_data and is_master
    if should_warn_no_data:
        print(f"[warn] {DATA_PATH} not found, using random data")

    data_iter = iter(loader) if loader else None
    t0 = time.time()
    val_loss = NAN_INIT

    model.train()
    for step in range(1, ITERATIONS + 1):
        inputs, targets, data_iter = _get_batch(loader, data_iter, has_data, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_validate = is_master and step % VAL_EVERY == 0
        if should_validate:
            val_loss = (
                estimate_loss(model, loader, device, VAL_STEPS)
                if has_data
                else loss.item()
            )
            print(
                f"step {step}/{ITERATIONS}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}"
            )

    training_seconds = time.time() - t0
    if is_master:
        _print_results(val_loss, training_seconds)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
