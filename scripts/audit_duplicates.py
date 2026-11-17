"""
Dataset duplicate/near-duplicate audit (Phase 2, 2026-11).

`split_indices_grouped` (data/datasets.py) keys its leakage-protection
grouping on `video_id`, which is empty for every ROCO/ROCOv2 row -- so the
split currently degenerates to a plain row-level split for exactly the
data the baseline comparison uses. This script audits whether that split
is actually leak-free by checking for exact and near-duplicate images
across the full ROCO + ROCOv2 manifest (97,335 rows).

Exact duplicates: MD5 of raw file bytes.
Near duplicates:  perceptual hash (phash), Hamming distance <= 5, the
                   standard "likely the same or a trivial crop/recolor of
                   the same image" threshold.

Usage:
    python scripts/audit_duplicates.py
"""
from __future__ import annotations

import hashlib
import os
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import imagehash
except ImportError:
    imagehash = None


def load_manifest_rows():
    rows = []
    for name, manifest, root in [
        ("roco", config.ROCO_MANIFEST, config.ROCO_ROOT),
        ("rocov2", config.ROCOV2_MANIFEST, config.ROCOV2_ROOT),
    ]:
        df = pd.read_csv(manifest, keep_default_na=False)
        df = df[df["image_path"] != ""]
        for _, row in df.iterrows():
            rows.append((f"{name}:{row['id']}", os.path.join(root, row["image_path"])))
    return rows


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = load_manifest_rows()
    print(f"Auditing {len(rows)} images across ROCO + ROCOv2...")

    md5_map: dict[str, list[str]] = {}
    phash_map: dict[object, list[str]] = {}
    missing = 0

    for i, (sample_id, path) in enumerate(rows):
        if i % 10000 == 0:
            print(f"  ...{i}/{len(rows)}")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            digest = md5_of_file(path)
            md5_map.setdefault(digest, []).append(sample_id)
            if imagehash is not None:
                with Image.open(path) as img:
                    ph = imagehash.phash(img.convert("RGB"))
                phash_map.setdefault(ph, []).append(sample_id)
        except Exception as e:
            print(f"  [skip] {sample_id}: {e}")

    exact_dupe_groups = {k: v for k, v in md5_map.items() if len(v) > 1}
    exact_dupe_count = sum(len(v) - 1 for v in exact_dupe_groups.values())

    print(f"\nMissing/unreadable files: {missing}")
    print(f"Exact-duplicate groups (identical file bytes): {len(exact_dupe_groups)}")
    print(f"Exact-duplicate images (excess over 1 per group): {exact_dupe_count}")

    if imagehash is not None:
        phash_dupe_groups = {k: v for k, v in phash_map.items() if len(v) > 1}
        phash_dupe_count = sum(len(v) - 1 for v in phash_dupe_groups.values())
        print(f"Exact-phash-match groups (distance 0): {len(phash_dupe_groups)}")
        print(f"Exact-phash-match images (excess over 1 per group): {phash_dupe_count}")

        # Near-duplicate (Hamming distance 1-5, not just exact phash match):
        # a full O(n^2) pairwise scan over ~90-95k unique hashes is ~4-8
        # billion comparisons and not tractable on this hardware. Instead,
        # bucket by the hash's top 16 bits (a cheap locality-sensitive
        # prefilter: two phashes within Hamming distance 5 of a 64-bit hash
        # very likely still share most of their high-order bits) and only
        # do exact pairwise comparison within each bucket. This is an
        # approximate near-duplicate search, not exhaustive -- it will miss
        # near-duplicate pairs whose high-order bits happen to differ, and
        # that limitation is reported honestly below rather than silently.
        unique_hashes = list(phash_map.keys())
        print(f"Checking near-duplicates (phash distance <= 5, bucketed) over {len(unique_hashes)} unique hashes...")
        buckets: dict[int, list] = {}
        for h in unique_hashes:
            key = int(str(h), 16) >> 48  # top 16 bits of the 64-bit hash
            buckets.setdefault(key, []).append(h)
        near_dupe_pairs = []
        for bucket_hashes in buckets.values():
            if len(bucket_hashes) < 2:
                continue
            for i in range(len(bucket_hashes)):
                hi = bucket_hashes[i]
                for j in range(i + 1, len(bucket_hashes)):
                    hj = bucket_hashes[j]
                    d = hi - hj
                    if 0 < d <= 5:
                        near_dupe_pairs.append((phash_map[hi][0], phash_map[hj][0], int(d)))
        print(f"Near-duplicate pairs found (bucketed, approximate): {len(near_dupe_pairs)}")

        out_path = os.path.join(config.PROJECT_ROOT, "outputs", "logs", "duplicate_audit_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Total images audited: {len(rows)}\n")
            f.write(f"Missing/unreadable: {missing}\n")
            f.write(f"Exact-duplicate (MD5) groups: {len(exact_dupe_groups)}, excess images: {exact_dupe_count}\n")
            f.write(f"Exact-phash-match groups: {len(phash_dupe_groups)}, excess images: {phash_dupe_count}\n")
            f.write(f"Near-duplicate pairs (phash distance 1-5, bucketed/approximate -- "
                    f"NOT an exhaustive pairwise scan, see script docstring): {len(near_dupe_pairs)}\n\n")
            f.write("Sample exact-MD5-duplicate groups (first 20):\n")
            for k, v in list(exact_dupe_groups.items())[:20]:
                f.write(f"  {v}\n")
            f.write("\nSample near-duplicate pairs (first 50):\n")
            for a, b, d in near_dupe_pairs[:50]:
                f.write(f"  {a} <-> {b}  (hamming={d})\n")
        print(f"\nFull report written to {out_path}")

        # Full machine-readable dump (Phase C, 2026-11): the text report
        # above only samples the first 20/50 entries for human reading;
        # split-rebuilding needs the COMPLETE exact-duplicate groups and
        # near-duplicate pairs, not a sample, so they're written in full
        # here as JSON.
        import json
        json_path = os.path.join(config.PROJECT_ROOT, "outputs", "logs", "duplicate_audit_full.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "exact_md5_groups": list(exact_dupe_groups.values()),
                "exact_phash_groups": list(phash_dupe_groups.values()),
                "near_duplicate_pairs": [[a, b, d] for a, b, d in near_dupe_pairs],
            }, f)
        print(f"Full machine-readable duplicate data written to {json_path}")
    else:
        print("\n[imagehash not installed -- near-duplicate check skipped, exact-MD5 check only]")


if __name__ == "__main__":
    main()
