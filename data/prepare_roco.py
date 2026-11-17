"""
Extract the gaodrew/roco-65k-256px parquet shards into JPEG files +
a manifest.csv matching the project's unified manifest schema
(see config.py):

    id, image_path, caption, label, video_id, frame_index

ROCO is a static image+caption dataset, so `video_id`/`frame_index` are
left empty and `label` is a placeholder (-1) until a labeling scheme is
defined for the classification head.

Usage:
    python data/prepare_roco.py
"""

from __future__ import annotations

import io
import os

import pandas as pd
from PIL import Image

import config

RAW_DIR = os.path.join(config.ROCO_ROOT, "raw")
IMAGES_DIR = os.path.join(config.ROCO_ROOT, "images")
SHARDS = ["train-00000.parquet", "train-00001.parquet"]


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)

    rows = []
    next_idx = 0
    for shard in SHARDS:
        shard_path = os.path.join(RAW_DIR, shard)
        if not os.path.exists(shard_path):
            print(f"[skip] missing shard: {shard_path}")
            continue

        df = pd.read_parquet(shard_path)
        print(f"[{shard}] {len(df)} rows, columns={list(df.columns)}")

        for _, row in df.iterrows():
            img_field = row["image"]
            caption = str(row["text"]).strip()

            img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                print(f"  [warn] failed to decode image at index {next_idx}: {exc}")
                continue

            sample_id = f"ROCO_{next_idx:06d}"
            filename = f"{sample_id}.jpg"
            out_path = os.path.join(IMAGES_DIR, filename)
            img.save(out_path, format="JPEG", quality=90)

            rows.append({
                "id": sample_id,
                "image_path": os.path.join("images", filename),
                "caption": caption,
                "label": -1,
                "video_id": "",
                "frame_index": "",
                "summary": "",
            })
            next_idx += 1

            if next_idx % 5000 == 0:
                print(f"  ... extracted {next_idx} images")

    manifest = pd.DataFrame(rows, columns=["id", "image_path", "caption", "label", "video_id", "frame_index", "summary"])
    manifest.to_csv(config.ROCO_MANIFEST, index=False)
    print(f"Wrote {len(manifest)} rows to {config.ROCO_MANIFEST}")


if __name__ == "__main__":
    main()
