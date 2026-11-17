"""
XGBoost baseline on the SAME combined ROCO+ROCOv2 4-class-balanced rows,
using the existing keyword/text-derived features from
feature_pipeline/data/{roco,rocov2}_features.csv (caption length, keyword
hits, etc. -- NOT deep pixel features; no image-embedding feature pipeline
exists in this project). XGBoost has no neural "parameter count"; its own
capacity knobs (n_estimators, max_depth) are reported instead, honestly
noted as not comparable 1:1 to a parameter count.

Usage:
    python feature_pipeline/train_baseline_xgboost.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import config
from evaluation.metrics import classification_metrics, confusion_matrix_report

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42

FEATURE_COLS = [
    "caption_char_len", "caption_word_count", "caption_unique_word_count",
    "caption_avg_word_len", "kw_has_mri", "kw_has_ct", "kw_has_us",
    "kw_has_xray", "kw_has_pet_nm", "kw_has_mass", "kw_has_lesion",
    "kw_has_cyst", "kw_has_tumor", "kw_has_fracture", "kw_pathology_match_count",
]


def load_matched(source_csv: str, balanced_manifest_csv: str) -> pd.DataFrame:
    feats = pd.read_csv(os.path.join(FEATURE_DIR, source_csv), keep_default_na=False)
    balanced = pd.read_csv(balanced_manifest_csv, keep_default_na=False)
    # restrict feats to exactly the ids present in the balanced/filtered manifest
    # used by every other baseline this session, for a fair comparison.
    return feats[feats["id"].astype(str).isin(set(balanced["id"].astype(str)))]


def main():
    roco = load_matched("roco_features.csv", "feature_pipeline/data/roco_manifest_4class_balanced.csv")
    rocov2 = load_matched("rocov2_features.csv", "feature_pipeline/data/rocov2_manifest_4class_balanced.csv")
    df = pd.concat([roco, rocov2], ignore_index=True)
    print(f"Combined matched rows: {len(df)}")
    print(df["label_5class"].value_counts().sort_index())

    X = df[FEATURE_COLS].astype(float).to_numpy()
    y = df["label_5class"].astype(int).to_numpy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softmax", num_class=5, random_state=SEED,
        eval_metric="mlogloss", early_stopping_rounds=20)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    cls = classification_metrics(preds, y_test, num_classes=5)
    cm = confusion_matrix_report(preds, y_test, num_classes=5)

    n_trees = model.get_booster().num_boosted_rounds()
    print(f"XGBoost: n_estimators(cap)=200, actual_trees(early-stopped)={n_trees}, max_depth=6")
    print(json.dumps({"test_acc": cls["accuracy"], **cls, **cm}, indent=2, default=str))

    out = {
        "arch": "xgboost_keyword_features", "n_params": None,
        "note": "XGBoost trained on keyword/text-derived features (caption length, "
                "modality/pathology keyword hits), NOT deep pixel features -- no "
                "image-embedding feature pipeline exists in this project. No neural "
                "parameter count applies; n_estimators/max_depth reported instead.",
        "n_estimators_cap": 200, "actual_trees": n_trees, "max_depth": 6,
        "test_acc": cls["accuracy"],
        "classification": {"5class": {**cls, **cm}},
    }
    out_path = os.path.join(config.LOGS_DIR, "test_metrics_baseline_xgboost_combined.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
