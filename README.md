# TylerGPT

A GPT language model built from scratch in PyTorch — every weight trained by me.

The plan (see roadmap below): start with a character-level model on Shakespeare
that trains on a laptop CPU, then scale the same code up on cloud GPUs.

## Quickstart (Phase 1: Shakespeare, CPU-friendly)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python data/prepare_shakespeare.py   # download + tokenize the corpus
python train.py                      # ~0.8M params, roughly an hour on a laptop CPU
python sample.py --prompt "ROMEO:"   # generate fake Shakespeare
```

`train.py` prints train/val loss as it goes and saves the best checkpoint to
`out/ckpt.pt`. Loss starts around 4.17 (= ln 65, random guessing over the
65-character vocabulary) and should fall below ~1.8 by the end, at which point
samples look recognizably Shakespeare-shaped.

## Bigger model on a free GPU (Google Colab)

Runtime → Change runtime type → T4 GPU, then:

```
!git clone https://github.com/TylerJForstrom/TylerGPT.git
%cd TylerGPT
!pip install -r requirements.txt -q
!python data/prepare_shakespeare.py
!python train.py --n-layer 6 --n-head 6 --n-embd 384 --block-size 256 \
                 --batch-size 64 --max-iters 5000 --dropout 0.2 --lr 6e-4
!python sample.py --prompt "ROMEO:" --max-new-tokens 500
```

That's a ~10M parameter model (nanoGPT's shakespeare-char config); it reaches
val loss ~1.47 and writes much more convincing verse.

## Repo layout

| File | What it is |
|---|---|
| `model.py` | The GPT itself: attention, MLP, transformer blocks, generation |
| `train.py` | Training loop: batching, AdamW, LR schedule, eval, checkpoints |
| `sample.py` | Load a checkpoint and generate text |
| `data/prepare_shakespeare.py` | Download corpus, build char vocab, write train/val bins |

## Roadmap

- [x] **Phase 1** — char-level GPT on tiny Shakespeare (this repo, CPU/Colab, $0)
- [ ] **Phase 2** — ~30-50M param model on TinyStories: coherent English ($5-20, rented GPU)
- [ ] **Phase 3** — [nanochat](https://github.com/karpathy/nanochat): full chat assistant, end to end (~$100, 8xH100 for ~4h)
- [ ] **Side quest** — QLoRA fine-tune of Llama 3.2 3B for something actually useful

## Credits

Architecture and training recipe closely follow Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and his
[Zero to Hero](https://karpathy.ai/zero-to-hero.html) lectures.
