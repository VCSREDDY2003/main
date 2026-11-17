"""
Extract the GlitchGhost/ROCOv2-radiology-mini parquet shards (train +
validation + test) into JPEG files + a manifest.csv matching the project's
unified manifest schema (see config.py):

    id, image_path, caption, label, video_id, frame_index

ROCOv2 is a static image+caption dataset (radiology images with UMLS CUI
codes), so `video_id`/`frame_index` are left empty and `label` is a
placeholder (-1) until a labeling scheme is defined for the classification
head.

Usage:
    python data/prepare_rocov2.py
"""

from __future__ import annotations

import glob
import io
import os

import pandas as pd
from PIL import Image

import config

RAW_DIR = os.path.join(config.ROCOV2_ROOT, "raw")
IMAGES_DIR = os.path.join(config.ROCOV2_ROOT, "images")


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not shard_paths:
        print(f"[warn] no parquet shards found in {RAW_DIR}")
        return

    rows = []
    next_idx = 0
    for shard_path in shard_paths:
        df = pd.read_parquet(shard_path)
        print(f"[{os.path.basename(shard_path)}] {len(df)} rows")

        for _, row in df.iterrows():
            img_field = row["image"]
            caption = str(row["caption"]).strip()

            img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                print(f"  [warn] failed to decode image at index {next_idx}: {exc}")
                continue

            sample_id = f"ROCOV2_{next_idx:06d}"
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
    manifest.to_csv(config.ROCOV2_MANIFEST, index=False)
    print(f"Wrote {len(manifest)} rows to {config.ROCOV2_MANIFEST}")


if __name__ == "__main__":
    main()
