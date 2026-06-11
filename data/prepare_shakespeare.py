"""
Phase 1 dataset: tiny Shakespeare (~1MB of text), character-level tokenization.

Downloads the text, builds a character vocabulary, encodes the whole corpus
to integer ids, and writes train.bin / val.bin (uint16) plus meta.json
(the vocabulary, needed to decode generated text back to characters).
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = Path(__file__).parent


def main():
    txt_path = HERE / "shakespeare.txt"
    if not txt_path.exists():
        print(f"downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, txt_path)
    text = txt_path.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters")

    # character-level vocab: every distinct character gets an id
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"vocab: {len(chars)} characters")

    ids = np.array([stoi[ch] for ch in text], dtype=np.uint16)

    # 90/10 train/validation split
    n = int(0.9 * len(ids))
    ids[:n].tofile(HERE / "train.bin")
    ids[n:].tofile(HERE / "val.bin")
    (HERE / "meta.json").write_text(
        json.dumps({"vocab_size": len(chars), "chars": chars}), encoding="utf-8"
    )
    print(f"train.bin: {n:,} tokens | val.bin: {len(ids) - n:,} tokens | meta.json written")


if __name__ == "__main__":
    main()
