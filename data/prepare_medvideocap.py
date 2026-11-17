"""
Build a manifest.csv for MedVideoCap-55K from the captions JSON
(MedVideoCap-55k.json), matching the project's unified manifest schema
(see config.py):

    id, image_path, caption, label, video_id, frame_index

The raw video files (videos_1.zip .. videos_6.zip, ~25.7GB total) are
intentionally NOT downloaded (too large for the local disk budget). This
manifest is therefore caption-only:

  - `video_id`   = the dataset's video id (e.g. "00001")
  - `caption`    = detailed_caption (used for the captioning/LMM caption target)
  - `label`      = integer class id derived from `video_categories`
                    (classification target, Eq. 4.9)
  - `summary`    = brief_caption (LMM summary target, Eq. 4.6-4.8 /
                    Algorithm 3) -- extra column beyond the base
                    `id, image_path, caption, label, video_id, frame_index`
                    schema; static datasets (ROCO/ROCOv2) leave it empty.
  - `image_path` = "" (no frames extracted -- caption-only sample)
  - `frame_index`= "" (no temporal frame sequence available)

Capped at MAX_SAMPLES rows (default 70_000, i.e. the whole 55,803-entry
file) to match the project's "~70K per source" dataset plan.

Usage:
    python data/prepare_medvideocap.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

import config

RAW_JSON = os.path.join(config.MEDVIDEOCAP_ROOT, "raw", "MedVideoCap-55k.json")
MAX_SAMPLES = 70_000

# video_categories -> class id (Eq. 4.9 classification target)
CATEGORY_TO_LABEL = {
    "medical teaching videos": 0,
    "clinical practice videos": 1,
    "medical popular science videos": 2,
    "medical animation videos": 3,
    "medical imaging videos": 4,
    "": 5,  # uncategorized
}


def main() -> None:
    with open(RAW_JSON, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for entry in data[:MAX_SAMPLES]:
        category = entry.get("video_categories", "")
        rows.append({
            "id": f"MEDVIDEOCAP_{entry['id']}",
            "image_path": "",
            "caption": str(entry.get("detailed_caption", "")).strip(),
            "label": CATEGORY_TO_LABEL.get(category, 5),
            "video_id": entry["id"],
            "frame_index": "",
            "summary": str(entry.get("brief_caption", "")).strip(),
        })

    manifest = pd.DataFrame(rows, columns=["id", "image_path", "caption", "label", "video_id", "frame_index", "summary"])
    manifest.to_csv(config.MEDVIDEOCAP_MANIFEST, index=False)
    print(f"Wrote {len(manifest)} rows to {config.MEDVIDEOCAP_MANIFEST}")


if __name__ == "__main__":
    main()
