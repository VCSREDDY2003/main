"""
Unified PyTorch Dataset over the manifest abstraction (Eq. 3.6-3.9 preprocessing).

Each manifest row is:
    id, image_path, caption, label, video_id, frame_index, summary

- ROCO / ROCOv2 (static):   image_path set, label=-1, summary="", video_id=""
- MedVideoCap-55K (video):  image_path="" (caption-only), label=class id,
                              summary=brief_caption, video_id set

Image preprocessing (Eq. 3.6-3.8):
    X~ = (X - mu) / sigma   resized to config.DEFAULT_IMAGE_SIZE, values in [0, 1]
    (the IAT module operates on [0,1] images and applies its own normalization
     internally, so no further channel-wise standardization is done here)

Caption/summary tokenization uses the shared Vocabulary (data/vocab.py) and
Eq. 3.2: C_i = {w_1, ..., w_L} -> fixed-length <bos>...<eos><pad> sequences.
"""

from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from data.vocab import Vocabulary

_TRANSFORM = transforms.Compose([
    transforms.Resize(config.DEFAULT_IMAGE_SIZE),
    transforms.ToTensor(),  # -> [0, 1], shape (C, H, W)
])


class ManifestDataset(Dataset):
    """A single dataset source (ROCO / ROCOv2 / MedVideoCap) loaded from its manifest."""

    def __init__(self, manifest_path: str, root_dir: str, vocab: Vocabulary,
                 max_caption_len: int = config.MAX_CAPTION_LEN):
        self.df = pd.read_csv(manifest_path, keep_default_na=False)
        self.root_dir = root_dir
        self.vocab = vocab
        self.max_len = max_caption_len

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> torch.Tensor:
        if not image_path:
            return torch.zeros(3, *config.DEFAULT_IMAGE_SIZE)
        img = Image.open(os.path.join(self.root_dir, image_path)).convert("RGB")
        return _TRANSFORM(img)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        caption_ids = self.vocab.encode(row["caption"], self.max_len)
        summary_text = row["summary"] if str(row.get("summary", "")) else row["caption"]
        summary_ids = self.vocab.encode(summary_text, self.max_len)

        label = int(row["label"]) if str(row["label"]) not in ("", "nan") else -1

        return {
            "id": str(row["id"]),
            "image": self._load_image(str(row["image_path"])),
            "has_image": bool(str(row["image_path"])),
            "caption_ids": torch.tensor(caption_ids, dtype=torch.long),
            "summary_ids": torch.tensor(summary_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "label_binary": torch.tensor(int(row["label_binary"]), dtype=torch.long),
            "label_3class": torch.tensor(int(row["label_3class"]), dtype=torch.long),
            "label_5class": torch.tensor(int(row["label_5class"]), dtype=torch.long),
            "video_id": str(row["video_id"]),
        }


def build_datasets(vocab: Vocabulary, roco_manifest_override: str | None = None,
                    rocov2_manifest_override: str | None = None) -> dict[str, ManifestDataset]:
    """Returns one ManifestDataset per active source (skips sources with no manifest yet).

    roco_manifest_override / rocov2_manifest_override: use a different manifest
    csv for that source (same image root) -- e.g. a balanced/relabeled variant
    built by feature_pipeline/build_pathology_binary_manifest.py or
    feature_pipeline/build_4class_balanced_manifest.py.

    MedVideoCap-55K was dropped from the paper/experiments (2026-09-02): the
    project now trains/evaluates on the two static-image sources only
    (ROCO + ROCOv2). MedVideoCap is intentionally left out of `sources` below
    -- do not re-add it without also reverting the dataset/methodology text
    in the paper. (config.MEDVIDEOCAP_MANIFEST/_ROOT are kept in config.py
    only so old code/paths referencing them don't hard-crash on import.)
    """
    sources = {
        "roco": (roco_manifest_override or config.ROCO_MANIFEST, config.ROCO_ROOT),
        "rocov2": (rocov2_manifest_override or config.ROCOV2_MANIFEST, config.ROCOV2_ROOT),
    }
    datasets = {}
    for name, (manifest_path, root_dir) in sources.items():
        if os.path.exists(manifest_path):
            datasets[name] = ManifestDataset(manifest_path, root_dir, vocab)
    return datasets


def split_indices(n: int, seed: int = config.RANDOM_SEED):
    """70/15/15 train/val/test split (Table 5.1)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def split_indices_grouped(group_ids, seed: int = config.RANDOM_SEED):
    """70/15/15 train/val/test split that keeps all rows sharing the same
    group (e.g. video_id) in the same split.

    A plain row-level `split_indices()` applied to MedVideoCap-55K scatters
    frames from the same video_id across train/val/test -- the model then
    sees near-duplicate frames/captions from a given video during training
    and again at "test" time, which leaks label/caption information and is
    the most likely explanation for the previously observed 1.0 test-set
    classification accuracy. Group-aware splitting removes that leak by
    assigning whole videos (not individual frames) to a single split.

    group_ids: sequence of length n; rows with the same value are kept
    together. Rows with an empty/falsy group id (static images with no
    video_id) are treated as their own singleton group, so this reduces to
    a normal per-row split for ROCO/ROCOv2.
    """
    import numpy as np

    group_ids = list(group_ids)
    n = len(group_ids)
    # assign each row a group key; empty video_id -> unique singleton group
    row_to_group = []
    seen_singletons = 0
    for i, gid in enumerate(group_ids):
        if gid:
            row_to_group.append(gid)
        else:
            row_to_group.append(f"__row_{i}")
            seen_singletons += 1

    unique_groups = sorted(set(row_to_group))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_groups))
    n_train = int(len(unique_groups) * config.TRAIN_SPLIT)
    n_val = int(len(unique_groups) * config.VAL_SPLIT)

    group_split = {}
    for rank, gi in enumerate(perm):
        g = unique_groups[gi]
        if rank < n_train:
            group_split[g] = "train"
        elif rank < n_train + n_val:
            group_split[g] = "val"
        else:
            group_split[g] = "test"

    train_idx, val_idx, test_idx = [], [], []
    for i, g in enumerate(row_to_group):
        split = group_split[g]
        (train_idx if split == "train" else val_idx if split == "val" else test_idx).append(i)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def build_leakage_safe_split(sources: dict, audit_json_path: str | None = None,
                              seed: int = config.RANDOM_SEED):
    """Global, cross-source leakage-safe split (Phase C, 2026-11).

    The existing split_indices_grouped is applied PER SOURCE independently
    in training/train.py's build_dataloaders, then the per-source splits
    are concatenated. That misses the real leak scripts/audit_duplicates.py
    found: ROCO and ROCOv2 share a large pool of exact/near-duplicate
    images (9,495 exact-phash-match groups, 1,050 near-duplicate pairs,
    overwhelmingly crossing the roco<->rocov2 boundary) -- a per-source
    split can put one copy of a duplicate in ROCO's train split and its
    twin in ROCOv2's test split, which is just as much a leak as the
    within-video_id leak split_indices_grouped was built to prevent.

    This builds ONE global union-find over "<source>:<row_id>" keys,
    unioning (a) rows sharing a nonempty video_id (same as before, now
    global) and (b) rows connected by the duplicate audit's exact-MD5
    groups, exact-phash groups, or near-duplicate pairs -- then assigns
    whole connected components to train/val/test, so no duplicate or
    near-duplicate pair, within or across source, can ever cross a split
    boundary.

    sources: dict[name -> ManifestDataset], as returned by build_datasets.
    Returns dict[name -> (train_idx, val_idx, test_idx)] of per-source
    integer index arrays, same shape as what split_indices_grouped
    returns for a single source.
    """
    import json

    import numpy as np

    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # global key per (source, row) + remember which rows map to which key
    row_keys: dict[str, list[str]] = {}
    for name, ds in sources.items():
        keys = [f"{name}:{rid}" for rid in ds.df["id"].astype(str).tolist()]
        row_keys[name] = keys
        for k in keys:
            parent.setdefault(k, k)
        video_ids = ds.df["video_id"].astype(str).tolist()
        by_video: dict[str, str] = {}
        for k, vid in zip(keys, video_ids):
            if vid:
                if vid in by_video:
                    union(k, by_video[vid])
                else:
                    by_video[vid] = k

    if audit_json_path and os.path.exists(audit_json_path):
        with open(audit_json_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        all_known = set(parent.keys())
        for group in audit.get("exact_md5_groups", []):
            group = [g for g in group if g in all_known]
            for g in group[1:]:
                union(g, group[0])
        for group in audit.get("exact_phash_groups", []):
            group = [g for g in group if g in all_known]
            for g in group[1:]:
                union(g, group[0])
        for a, b, _d in audit.get("near_duplicate_pairs", []):
            if a in all_known and b in all_known:
                union(a, b)

    unique_components = sorted({find(k) for k in parent})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_components))
    n_train = int(len(unique_components) * config.TRAIN_SPLIT)
    n_val = int(len(unique_components) * config.VAL_SPLIT)

    comp_split = {}
    for rank, ci in enumerate(perm):
        comp = unique_components[ci]
        comp_split[comp] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    result = {}
    for name, keys in row_keys.items():
        train_idx, val_idx, test_idx = [], [], []
        for i, k in enumerate(keys):
            split = comp_split[find(k)]
            (train_idx if split == "train" else val_idx if split == "val" else test_idx).append(i)
        result[name] = (np.array(train_idx), np.array(val_idx), np.array(test_idx))
    return result
