"""
Caption vocabulary (Eq. 3.2-3.3: C_i = {w_1, ..., w_L}, e_t = phi(w_t)).

Builds a fixed-size whitespace/punctuation tokenizer vocabulary from the
`caption` (and `summary`) columns of the dataset manifests, with the
standard special tokens:

    0: <pad>   1: <bos>   2: <eos>   3: <unk>

Usage:
    python data/vocab.py            # builds and saves vocab.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import pandas as pd

import config

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


class Vocabulary:
    def __init__(self, word2idx: dict[str, int]):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, text: str, max_len: int) -> list[int]:
        """C_i -> token ids, truncated/padded to max_len, wrapped in <bos>/<eos> (Eq. 3.2)."""
        tokens = tokenize(text)[: max_len - 2]
        ids = [self.word2idx[BOS]] + [
            self.word2idx.get(t, self.word2idx[UNK]) for t in tokens
        ] + [self.word2idx[EOS]]
        ids += [self.word2idx[PAD]] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK)
            if w == EOS:
                break
            if w in (PAD, BOS):
                continue
            words.append(w)
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            word2idx = json.load(f)
        return cls(word2idx)


def build_vocab(manifest_paths: list[str], vocab_size: int = config.VOCAB_SIZE) -> Vocabulary:
    counter = Counter()
    for path in manifest_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for col in ("caption", "summary"):
            if col not in df.columns:
                continue
            for text in df[col].dropna().astype(str):
                counter.update(tokenize(text))

    most_common = counter.most_common(vocab_size - len(SPECIAL_TOKENS))
    word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return Vocabulary(word2idx)


VOCAB_PATH = os.path.join(config.DATA_ROOT, "vocab.json")


if __name__ == "__main__":
    manifests = [config.ROCO_MANIFEST, config.ROCOV2_MANIFEST, config.MEDVIDEOCAP_MANIFEST]
    vocab = build_vocab(manifests)
    vocab.save(VOCAB_PATH)
    print(f"Vocabulary size: {len(vocab)} -> {VOCAB_PATH}")
