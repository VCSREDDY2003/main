"""
Builds a per-image "feature CSV" (25+ columns) for ROCO, ROCOv2, and the
combined set, reusing the existing manifest columns (id, image_path,
caption, label_5class/3class/binary from data/build_classification_labels.py)
and adding new engineered columns: caption text stats, modality-keyword
flags, and pathology-keyword flags/labels (mass/lesion/cyst/tumor/fracture,
the exclusive-priority scheme discussed this session).

Does NOT read image pixels (kept out for speed, per "make it simple /
fast" instruction) -- every column here is derived from the manifest's
existing columns or from caption text. image_path is kept as a column so
downstream code (the tiny classifier) can load pixels itself.

Output: feature_pipeline/data/<name>_features.csv for name in
        {roco, rocov2, combined}

Usage:
    python feature_pipeline/build_feature_csv.py
"""
from __future__ import annotations

import os
import re

import pandas as pd

import config

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

_MRI_KW = ["mri", "magnetic resonance", "t1-weighted", "t2-weighted", "flair"]
_CT_KW = ["ct scan", "computed tomography", " ct ", "cat scan", "tomography"]
_US_KW = ["ultrasound", "sonograph", "echocardiog", "doppler", "sonogram"]
_XRAY_KW = ["x-ray", "xray", "radiograph", "plain film", "roentgen"]
_PET_KW = ["pet scan", "pet/ct", "positron emission", "scintigra", "nuclear medicine"]

_PATH_TERMS = ["mass", "lesion", "cyst", "tumor", "fracture"]
_PATH_PRIORITY = ["mass", "lesion", "cyst", "tumor", "fracture"]


def _flag(text_series: pd.Series, keywords: list[str]) -> pd.Series:
    return text_series.apply(lambda t: int(any(kw in t for kw in keywords)))


def build_features(manifest_path: str, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, keep_default_na=False, encoding="utf-8",
                      encoding_errors="replace")
    df = df.copy()
    df["source_dataset"] = source_name

    text = (df["caption"].astype(str) + " " + df.get("summary", "").astype(str)).str.lower()

    # --- caption text-stat features ---
    df["caption_char_len"] = df["caption"].astype(str).str.len()
    df["caption_word_count"] = df["caption"].astype(str).str.split().apply(len)
    df["caption_unique_word_count"] = df["caption"].astype(str).apply(
        lambda c: len(set(re.findall(r"[a-zA-Z]+", c.lower()))))
    df["caption_avg_word_len"] = df["caption"].astype(str).apply(
        lambda c: (sum(len(w) for w in c.split()) / max(len(c.split()), 1)))

    # --- modality-keyword flags (independent re-derivation, for feature use) ---
    df["kw_has_mri"] = _flag(text, _MRI_KW)
    df["kw_has_ct"] = _flag(text, _CT_KW)
    df["kw_has_us"] = _flag(text, _US_KW)
    df["kw_has_xray"] = _flag(text, _XRAY_KW)
    df["kw_has_pet_nm"] = _flag(text, _PET_KW)

    # --- pathology-keyword flags (this session's finding-term analysis) ---
    for term in _PATH_TERMS:
        df[f"kw_has_{term}"] = text.str.contains(term, regex=False).astype(int)
    df["kw_pathology_match_count"] = df[[f"kw_has_{t}" for t in _PATH_TERMS]].sum(axis=1)

    def _exclusive_pathology_label(row) -> str:
        for t in _PATH_PRIORITY:
            if row[f"kw_has_{t}"]:
                return t
        return "none"

    df["pathology_label_exclusive"] = df.apply(_exclusive_pathology_label, axis=1)
    df["pathology_binary"] = (df["pathology_label_exclusive"] != "none").astype(int)

    # --- carry over existing classification labels (already in manifest) ---
    for col in ("label_5class", "label_3class", "label_binary"):
        if col not in df.columns:
            df[col] = -1

    # --- row identity / bookkeeping ---
    df["has_image"] = (df["image_path"].astype(str) != "").astype(int)
    df["video_id"] = df.get("video_id", "")
    df["frame_index"] = df.get("frame_index", "")

    keep_cols = [
        "id", "source_dataset", "image_path", "has_image",
        "caption", "summary",
        "caption_char_len", "caption_word_count", "caption_unique_word_count",
        "caption_avg_word_len",
        "kw_has_mri", "kw_has_ct", "kw_has_us", "kw_has_xray", "kw_has_pet_nm",
        "kw_has_mass", "kw_has_lesion", "kw_has_cyst", "kw_has_tumor", "kw_has_fracture",
        "kw_pathology_match_count", "pathology_label_exclusive", "pathology_binary",
        "label_5class", "label_3class", "label_binary",
        "video_id", "frame_index",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    out = df[keep_cols]
    print(f"{source_name}: {len(out)} rows, {len(out.columns)} columns")
    return out


def main():
    roco = build_features(config.ROCO_MANIFEST, "roco")
    rocov2 = build_features(config.ROCOV2_MANIFEST, "rocov2")
    combined = pd.concat([roco, rocov2], ignore_index=True)

    roco.to_csv(os.path.join(OUT_DIR, "roco_features.csv"), index=False)
    rocov2.to_csv(os.path.join(OUT_DIR, "rocov2_features.csv"), index=False)
    combined.to_csv(os.path.join(OUT_DIR, "combined_features.csv"), index=False)

    print(f"\nWrote:")
    print(f"  {OUT_DIR}/roco_features.csv       ({len(roco)} rows, {len(roco.columns)} cols)")
    print(f"  {OUT_DIR}/rocov2_features.csv     ({len(rocov2)} rows, {len(rocov2.columns)} cols)")
    print(f"  {OUT_DIR}/combined_features.csv   ({len(combined)} rows, {len(combined.columns)} cols)")


if __name__ == "__main__":
    main()
