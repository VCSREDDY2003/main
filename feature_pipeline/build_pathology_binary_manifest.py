"""
Builds a ROCO manifest for CNN+IATnew where label_binary is overwritten with
the new pathology_binary label (mass/lesion/cyst/tumor/fracture keyword hit
in the caption vs none), instead of the old modality-based label_binary
(Other-vs-rest). Rows are undersampled so both classes are equal size.

Reuses feature_pipeline/build_feature_csv.py's pathology_binary logic
(already computed in feature_pipeline/data/roco_features.csv) -- just joins
it onto the real manifest (which has image_path/caption/etc. needed by
ManifestDataset) by row order (both built from the same source manifest, so
row order is identical).

Output: feature_pipeline/data/roco_manifest_pathology_binary_balanced.csv
"""
from __future__ import annotations

import os

import pandas as pd

import config

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42


def main():
    manifest = pd.read_csv(config.ROCO_MANIFEST, keep_default_na=False)
    features = pd.read_csv(os.path.join(FEATURE_DIR, "roco_features.csv"), keep_default_na=False)

    assert len(manifest) == len(features), "row count mismatch between manifest and features csv"
    manifest["pathology_binary"] = features["pathology_binary"].astype(int)

    # keep only rows with a real image (pathology label is caption-derived either way,
    # but IAT/CNN need pixels)
    manifest = manifest[manifest["image_path"].astype(str) != ""].copy()

    manifest["label_binary"] = manifest["pathology_binary"]

    counts = manifest["label_binary"].value_counts()
    min_count = counts.min()
    print(f"Pathology-binary class counts before balancing: {counts.to_dict()}")

    balanced = (manifest.groupby("label_binary", group_keys=False)
                .apply(lambda g: g.sample(n=min_count, random_state=SEED)))
    balanced = balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"Balanced to {min_count} samples/class -> {len(balanced)} total rows")
    print(balanced["label_binary"].value_counts())

    out_path = os.path.join(FEATURE_DIR, "roco_manifest_pathology_binary_balanced.csv")
    balanced.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(balanced)} rows)")


if __name__ == "__main__":
    main()
