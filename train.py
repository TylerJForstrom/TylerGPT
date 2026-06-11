"""
Train TylerGPT on a prepared dataset (see data/prepare_shakespeare.py).

Defaults are sized for CPU training (~0.8M params, ~15-25 min for 2000 iters).
On a GPU (e.g. free Colab T4), pass bigger settings — see README.

  python train.py                          # CPU-friendly defaults
  python train.py --max-iters 200          # quick smoke test
  python train.py --n-layer 6 --n-head 6 --n-embd 384 --block-size 256 \
                  --batch-size 64 --max-iters 5000     # GPU config (~10M params)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data", help="dir with train.bin/val.bin/meta.json")
    p.add_argument("--out-dir", default="out", help="where checkpoints are saved")
    p.add_argument("--resume", action="store_true", help="resume from out-dir/ckpt.pt")
    # model size
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimization
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    """Sample batch_size random windows of block_size tokens. The target y is
    the same window shifted one to the right: predict the next character."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    """Average loss over a few random batches per split — a smoother signal
    than any single batch's loss."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it, args):
    """Linear warmup, then cosine decay to min_lr."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    splits = {
        "train": np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r"),
        "val": np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r"),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"

    start_iter = 0
    best_val = float("inf")
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt["iter"]
        best_val = ckpt["best_val"]
        print(f"resumed from iter {start_iter} (best val loss {best_val:.4f})")
    else:
        config = GPTConfig(
            block_size=args.block_size, vocab_size=meta["vocab_size"],
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout,
        )
        model = GPT(config).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = lr_at(it, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(splits["train"], args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % 50 == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s")

        last = it == args.max_iters - 1
        if (it > 0 and it % args.eval_interval == 0) or last:
            losses = estimate_loss(model, splits, args, device)
            print(f"  eval @ {it}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val or last:
                best_val = min(best_val, losses["val"])
                torch.save({
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "iter": it + 1,
                    "best_val": best_val,
                    "meta": meta,
                }, ckpt_path)
                print(f"  saved checkpoint -> {ckpt_path}")

    print(f"done in {time.time() - t0:.0f}s. Try:  python sample.py")


if __name__ == "__main__":
    main()
