"""
"Modern English" blend: five clean, contemporary sources mixed at fixed ratios.

  30%  TinyStories        (roneneldan/TinyStories)        grammar + narrative coherence
  20%  Simple Wikipedia   (wikimedia/wikipedia, simple)   modern declarative prose
  20%  FineWeb-Edu        (HuggingFaceFW/fineweb-edu)     general modern web English
  15%  DailyDialog        (agentlans/li2017dailydialog)   everyday conversation
  15%  Hacker News        (Algolia API)                   internet discussion voice

Hugging Face sources are streamed, so only the slice we need is downloaded.
Docs are shuffled before the train/val split. Writes data/modern/.

  python data/prepare_modern.py                    # ~16MB default
  python data/prepare_modern.py --total-bytes 40000000   # bigger, for Colab runs
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

from prepare_blend import chunk_book, fetch_hn_comments, normalize

HERE = Path(__file__).parent

RATIOS = {
    "tinystories": 0.30,
    "simple_wikipedia": 0.20,
    "fineweb_edu": 0.20,
    "dailydialog": 0.15,
    "hackernews": 0.15,
}


def take_streamed(dataset, budget: int, get_text) -> list[str]:
    """Pull rows from a streaming dataset until we've collected budget bytes."""
    docs, total = [], 0
    for row in dataset:
        text = normalize(get_text(row)).strip()
        if len(text) < 200:
            continue
        # chop long docs so the shuffle mixes sources evenly
        for doc in (chunk_book(text) if len(text) > 4000 else [text]):
            docs.append(doc)
            total += len(doc)
        if total >= budget:
            break
    return docs


def dailydialog_text(row) -> str:
    """Join a dialogue's turns into lines, fixing DailyDialog's spaced
    punctuation ('I ’ m fine .' -> 'I'm fine.'). Uses the agentlans parquet
    mirror's chat schema: a `conversations` list of {from, value} messages."""
    turns = []
    for msg in row["conversations"]:
        if msg["from"] == "system":
            continue
        turn = msg["value"].replace(" ’ ", "'").replace("’", "'")
        turn = re.sub(r"\s+([.,!?;:%)])", r"\1", turn)
        turn = re.sub(r"([($])\s+", r"\1", turn)
        turns.append(turn.strip())
    return "\n".join(turns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-bytes", type=int, default=16_000_000)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    budgets = {name: int(args.total_bytes * frac) for name, frac in RATIOS.items()}

    docs = {}

    print(f"TinyStories ({budgets['tinystories'] / 1e6:.0f}MB)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    docs["tinystories"] = take_streamed(ds, budgets["tinystories"], lambda r: r["text"])

    print(f"Simple English Wikipedia ({budgets['simple_wikipedia'] / 1e6:.0f}MB)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
    docs["simple_wikipedia"] = take_streamed(ds, budgets["simple_wikipedia"], lambda r: r["text"])

    print(f"FineWeb-Edu ({budgets['fineweb_edu'] / 1e6:.0f}MB)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    docs["fineweb_edu"] = take_streamed(ds, budgets["fineweb_edu"], lambda r: r["text"])

    print(f"DailyDialog ({budgets['dailydialog'] / 1e6:.0f}MB)...")
    # the official li2017dailydialog repo is script-based, unsupported by datasets>=3;
    # this mirror is parquet
    ds = load_dataset("agentlans/li2017dailydialog", split="train", streaming=True)
    docs["dailydialog"] = take_streamed(ds, budgets["dailydialog"], dailydialog_text)

    print(f"Hacker News ({budgets['hackernews'] / 1e6:.0f}MB)...")
    docs["hackernews"] = fetch_hn_comments(budgets["hackernews"])

    all_docs = [d for source in docs.values() for d in source]
    random.Random(args.seed).shuffle(all_docs)
    corpus = "\n\n".join(all_docs)

    total = len(corpus)
    print(f"\ncorpus: {total / 1e6:.1f}MB total")
    for name, source_docs in docs.items():
        size = sum(len(d) + 2 for d in source_docs)
        print(f"  {100 * size / total:4.1f}%  {size / 1e6:5.1f}MB  {name} ({len(source_docs):,} docs)")

    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"vocab: {len(chars)} characters")

    ids = np.array([stoi[ch] for ch in corpus], dtype=np.uint16)
    out = HERE / "modern"
    out.mkdir(exist_ok=True)
    n = int(0.9 * len(ids))
    ids[:n].tofile(out / "train.bin")
    ids[n:].tofile(out / "val.bin")
    (out / "meta.json").write_text(
        json.dumps({"vocab_size": len(chars), "chars": chars}), encoding="utf-8"
    )
    print(f"train.bin: {n:,} tokens | val.bin: {len(ids) - n:,} tokens -> {out}")


if __name__ == "__main__":
    main()
