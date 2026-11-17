"""
Derives multi-granularity classification labels (binary / 3-class / 5-class)
for every sample across all manifests, generalizing L_cls (Eq. 3.29) to three
classification heads.

ROCO / ROCOv2 captions describe the imaging modality of a static radiology
image, so a keyword heuristic over (caption + summary) text assigns a
modality class. MedVideoCap-55K rows have no image (image_path == "") and are
always treated as "Other / video content".

5-class (imaging modality):
    0 = X-ray / radiograph
    1 = CT
    2 = MRI
    3 = Ultrasound
    4 = Other (video / clinical / endoscopy / pathology / unspecified)

3-class (modality group), derived from the 5-class label:
    0 = Radiograph (X-ray)            <- 5-class 0
    1 = Cross-sectional (CT/MRI/US)   <- 5-class 1, 2, 3
    2 = Other / video                 <- 5-class 4

Binary, derived from the 5-class label:
    0 = Static radiology image        <- 5-class 0, 1, 2, 3
    1 = Video / other content         <- 5-class 4

Adds columns label_5class, label_3class, label_binary to each manifest CSV
in place.
"""

from __future__ import annotations

import pandas as pd

import config

_MRI_KEYWORDS = ["mri", "magnetic resonance", "t1-weighted", "t2-weighted", "t1 weighted",
                 "t2 weighted", "flair", "diffusion-weighted", "diffusion weighted"]
_CT_KEYWORDS = ["ct scan", "computed tomography", " ct ", "cat scan", "axial ct", "tomography", "ct of"]
_US_KEYWORDS = ["ultrasound", "sonograph", "echocardiog", "doppler", "sonogram"]
_XRAY_KEYWORDS = ["x-ray", "xray", "radiograph", "plain film", "chest film", "roentgen",
                  # Bug-fix additions (2026-09): these projection/fluoroscopic modalities are
                  # plain-film/contrast X-ray family and were previously falling into "Other".
                  "angiogra", "angiogram", "angiography", "arteriogra", "venogra",
                  "ercp", "cholangiogra", "cholangiopancreatograph",
                  "fluoroscop", "mammogra", "myelogra", "urogra", "pyelogra",
                  "barium", "contrast study", "contrast enema"]
# PET / nuclear medicine gets its own explicit keyword check instead of silently
# falling through to "Other"; histology/pathology is intentionally left in
# "Other" (see module docstring) since it is not a radiology modality.
_PET_NM_KEYWORDS = ["pet scan", "pet/ct", "pet-ct", "positron emission", "scintigra",
                     "nuclear medicine", "bone scan", "spect", "radionuclide", "technetium",
                     "gallium scan", "thyroid scan", "v/q scan", "ventilation-perfusion"]

_THREE_CLASS_FROM_FIVE = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
_BINARY_FROM_FIVE = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}


def classify_modality(caption: str, summary: str, has_image: bool) -> int:
    """Returns the 5-class modality label for a single sample."""
    if not has_image:
        return 4  # MedVideoCap (caption-only) -> Other / video content

    text = f" {caption} {summary} ".lower()
    if any(kw in text for kw in _MRI_KEYWORDS):
        return 2
    if any(kw in text for kw in _CT_KEYWORDS):
        return 1
    if any(kw in text for kw in _US_KEYWORDS):
        return 3
    if any(kw in text for kw in _XRAY_KEYWORDS):
        return 0
    if any(kw in text for kw in _PET_NM_KEYWORDS):
        return 1  # PET/SPECT/nuclear medicine is cross-sectional/tomographic like CT
    return 4


def add_labels(manifest_path: str) -> int:
    df = pd.read_csv(manifest_path, keep_default_na=False)

    label_5class = []
    for _, row in df.iterrows():
        has_image = bool(str(row["image_path"]))
        label_5class.append(classify_modality(str(row["caption"]), str(row.get("summary", "")), has_image))

    df["label_5class"] = label_5class
    df["label_3class"] = df["label_5class"].map(_THREE_CLASS_FROM_FIVE)
    df["label_binary"] = df["label_5class"].map(_BINARY_FROM_FIVE)

    df.to_csv(manifest_path, index=False)
    return len(df)


def main() -> None:
    for manifest_path in (config.ROCO_MANIFEST, config.ROCOV2_MANIFEST, config.MEDVIDEOCAP_MANIFEST):
        import os
        if not os.path.exists(manifest_path):
            continue
        n = add_labels(manifest_path)
        print(f"{manifest_path}: labeled {n} rows")


if __name__ == "__main__":
    main()
