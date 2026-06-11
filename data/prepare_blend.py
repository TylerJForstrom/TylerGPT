"""
Blended dataset: public-domain novels (Project Gutenberg) + Hacker News comments.

Books supply volume and rich vocabulary; HN comments supply modern,
conversational internet English. Sources are chopped into small documents and
shuffled together before the train/val split so both splits contain the blend.

Writes data/blend/{train.bin,val.bin,meta.json}. Book downloads are cached
in data/cache/ so re-runs are fast.

  python data/prepare_blend.py
  python data/prepare_blend.py --hn-bytes 5000000   # more forum flavor
"""
import argparse
import html
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# The most modern prose Project Gutenberg offers (public domain = pre-~1930).
BOOKS = {
    64317: "The Great Gatsby (Fitzgerald, 1925)",
    76: "Adventures of Huckleberry Finn (Twain, 1884)",
    74: "The Adventures of Tom Sawyer (Twain, 1876)",
    1661: "The Adventures of Sherlock Holmes (Doyle, 1892)",
    2852: "The Hound of the Baskervilles (Doyle, 1902)",
    174: "The Picture of Dorian Gray (Wilde, 1890)",
    345: "Dracula (Stoker, 1897)",
    215: "The Call of the Wild (London, 1903)",
    910: "White Fang (London, 1906)",
    35: "The Time Machine (Wells, 1895)",
    36: "The War of the Worlds (Wells, 1898)",
    120: "Treasure Island (Stevenson, 1883)",
}

# Normalize fancy unicode to plain ASCII so the character vocabulary stays small.
REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...", " ": " ",
    "\r": "",
}
ALLOWED = set("\n" + "".join(chr(c) for c in range(32, 127)))


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "TylerGPT-data-prep"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def normalize(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    text = "".join(ch for ch in text if ch in ALLOWED)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def fetch_book(book_id: int) -> str:
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"pg{book_id}.txt"
    if not path.exists():
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        path.write_bytes(http_get(url))
    text = path.read_text(encoding="utf-8", errors="replace")
    # strip Gutenberg's license header/footer
    start = re.search(r"\*\*\* START OF .*? \*\*\*", text)
    end = re.search(r"\*\*\* END OF .*? \*\*\*", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return normalize(text)


def clean_comment(raw_html: str) -> str:
    text = raw_html.replace("<p>", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return normalize(text).strip()


def fetch_hn_comments(target_bytes: int) -> list[str]:
    """Page backwards through HN comments via the free Algolia API until we
    have target_bytes of cleaned text."""
    docs, total = [], 0
    ts = int(time.time())
    while total < target_bytes:
        url = (
            "https://hn.algolia.com/api/v1/search_by_date"
            f"?tags=comment&hitsPerPage=1000&numericFilters=created_at_i<{ts}"
        )
        hits = json.loads(http_get(url)).get("hits", [])
        if not hits:
            break
        for h in hits:
            text = clean_comment(h.get("comment_text") or "")
            # skip one-liners and walls of text; both are poor style teachers
            if 100 <= len(text) <= 4000:
                docs.append(text)
                total += len(text)
        ts = min(h["created_at_i"] for h in hits)
        print(f"  HN: {total / 1e6:.1f}MB collected...")
        time.sleep(0.25)
    return docs


def chunk_book(text: str, target_chars: int = 3000) -> list[str]:
    """Chop a book into ~3KB paragraph groups. The model trains on short
    windows anyway, and small docs let the shuffle mix sources evenly."""
    docs, current, size = [], [], 0
    for para in text.split("\n\n"):
        current.append(para)
        size += len(para)
        if size >= target_chars:
            docs.append("\n\n".join(current))
            current, size = [], 0
    if current:
        docs.append("\n\n".join(current))
    return docs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hn-bytes", type=int, default=3_000_000,
                   help="how much Hacker News text to blend in (bytes)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    docs = []
    book_bytes = 0
    for book_id, title in BOOKS.items():
        try:
            text = fetch_book(book_id)
        except Exception as e:
            print(f"  SKIPPING {title}: {e}")
            continue
        docs.extend(chunk_book(text))
        book_bytes += len(text)
        print(f"  {len(text) / 1e3:7.0f}KB  {title}")

    print("fetching Hacker News comments...")
    hn_docs = fetch_hn_comments(args.hn_bytes)
    hn_bytes = sum(len(d) for d in hn_docs)
    docs.extend(hn_docs)

    random.Random(args.seed).shuffle(docs)
    corpus = "\n\n".join(docs)
    total = len(corpus)
    print(f"\ncorpus: {total / 1e6:.1f}MB total | "
          f"books {book_bytes / 1e6:.1f}MB ({100 * book_bytes / total:.0f}%) | "
          f"HN {hn_bytes / 1e6:.1f}MB ({100 * hn_bytes / total:.0f}%)")

    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"vocab: {len(chars)} characters")

    ids = np.array([stoi[ch] for ch in corpus], dtype=np.uint16)
    out = HERE / "blend"
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
