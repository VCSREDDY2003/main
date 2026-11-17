"""
Builds a manifest matched to the TinyCNN baseline setup: excludes
label_5class==4 ("Other" -- keyword-miss catch-all, not a real modality),
then undersamples the remaining 4 classes to equal counts. Used so
CNN+IATnew (training/train.py) can be compared fairly against the
TinyCNN result (both trained on the identical filtered/balanced rows).

Usage:
    python feature_pipeline/build_4class_balanced_manifest.py --source roco
    python feature_pipeline/build_4class_balanced_manifest.py --source rocov2

Output: feature_pipeline/data/<source>_manifest_4class_balanced.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import config

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42


def build(source: str):
    manifest_path = config.ROCO_MANIFEST if source == "roco" else config.ROCOV2_MANIFEST
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest = manifest[manifest["image_path"].astype(str) != ""].copy()
    manifest = manifest[manifest["label_5class"].astype(int) != 4].copy()

    counts = manifest["label_5class"].value_counts()
    min_count = counts.min()
    print(f"[{source}] 4-class counts before balancing: {counts.to_dict()}")

    balanced = (manifest.groupby("label_5class", group_keys=False)
                .apply(lambda g: g.sample(n=min_count, random_state=SEED)))
    balanced = balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"[{source}] Balanced to {min_count} samples/class -> {len(balanced)} total rows")
    print(balanced["label_5class"].value_counts().sort_index())

    out_path = os.path.join(FEATURE_DIR, f"{source}_manifest_4class_balanced.csv")
    balanced.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(balanced)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["roco", "rocov2", "both"], default="both")
    args = p.parse_args()
    if args.source == "both":
        build("roco")
        build("rocov2")
    else:
        build(args.source)


if __name__ == "__main__":
    main()
