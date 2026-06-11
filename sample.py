"""
Generate text from a trained TylerGPT checkpoint.

  python sample.py
  python sample.py --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8
"""
import argparse

import torch

from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8, help="<1 safer, >1 wilder")
    p.add_argument("--top-k", type=int, default=40, help="sample only from the k likeliest tokens")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    chars = ckpt["meta"]["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(chars[i] for i in ids)

    x = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(decode(y[0].tolist()))
        print("-" * 60)


if __name__ == "__main__":
    main()
