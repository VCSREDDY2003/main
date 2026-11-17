# IAT-GAM: CNN + Invasive Augmented Transform + GAN-Memory Fusion for Medical Image Classification & Captioning

IAT-GAM-LSTM is a lightweight framework for joint medical image classification and
caption generation on radiology image-caption corpora (ROCO / ROCOv2). It combines a
small CNN backbone, an "Invasive Augmented Transform" (IAT) feature-augmentation
stage, an optional GAN-driven memory-coupling module (GAM), and an LSTM/attention
caption decoder (LMMDecoder), with the explicit design goal of **parameter efficiency**
(matching or beating larger baselines at a fraction of the parameter budget) rather than
chasing raw SOTA accuracy with a bigger model.

This repository contains the **code and result artifacts** (metrics JSON, training
curves, result plots) for the project. It does **not** contain the manuscript/paper,
model checkpoints, or the raw ROCO/ROCOv2 dataset — see "What's excluded" below.

---

## Repository structure

```
config.py                  Central configuration: dataset paths, split ratios, seed,
                            architecture dims, baseline vs. tuned hyperparameters.
data/                       Dataset manifests, loaders, vocabulary building.
  prepare_roco.py           Builds the ROCO (v1) manifest.
  prepare_rocov2.py         Builds the ROCOv2 manifest.
  prepare_medvideocap.py    Builds the MedVideoCap-55K (video) manifest.
  build_classification_labels.py  Derives classification labels from captions.
  datasets.py, vocab.py     PyTorch Dataset classes and caption vocabulary.
feature_pipeline/           Baseline architectures, manifest builders for the
                            balanced 4-class / binary-pathology splits, and
                            standalone baseline training scripts (incl. XGBoost
                            on keyword features and a frozen-backbone DenseNet
                            caption-only trainer).
models/                     Model definitions: cnn/, iat/, gan/, fusion/, lmm/
                            (caption decoder), lstm/, text/, baseline/, and the
                            top-level iatgam_model.py that wires them together.
training/                   train.py (main training entry point), baseline and
                            literature-baseline trainers, augmentation methods,
                            loss functions.
evaluation/                 generate_results.py (main metrics generator),
                            generate_results_baseline*.py / _lit_baselines.py /
                            _lstm_baselines.py (baseline comparison runners),
                            metrics.py (classification + captioning metric
                            implementations), evaluate_by_group.py,
                            evaluate_medical_keywords.py, generate_report.py,
                            profile_complexity.py (param/latency profiling),
                            generate_statistical_complexity_report.py.
scripts/                    audit_duplicates.py (MD5/phash duplicate audit across
                            ROCO/ROCOv2), significance_test.py, sanity-check
                            scripts for the IAT and feature-GAM modules,
                            summarize_binary_matrix.py.
run_*.sh                    Experiment "queue" scripts that launch sequences of
                            training/evaluation runs (see table below).
outputs/logs/               Metrics JSON files, training-curve CSVs, and run logs
                            for completed experiments (results only).
outputs/plots/               Result figures used for analysis/the paper
  plots/paper_figures/      (classification bars, captioning bars, confusion
                            matrices, ablation plots, training curves, the
                            architecture diagram, and the scripts that render
                            them from the logged metrics/CSVs).
```

### What's excluded from this repository

- `outputs/model/` — trained model checkpoints (~19 GB). Not pushed.
- `datasets` — a local symlink to the raw ROCO/ROCOv2/MedVideoCap image data outside
  this project folder. Not pushed; not a dataset redistribution.
- `__pycache__/`, `*.pyc` — build artifacts.
- `*.docx`, `*.pdf` — the paper/manuscript is intentionally **not** included here.

---

## How to reproduce

1. Point `config.py` (`ROCO_ROOT`, `ROCOV2_ROOT`, `MEDVIDEOCAP_ROOT`) at local copies of
   the datasets and build manifests with `data/prepare_roco.py`,
   `data/prepare_rocov2.py`, `data/prepare_medvideocap.py`.
2. Build the balanced classification manifests used by most runs:
   `feature_pipeline/build_4class_balanced_manifest.py`,
   `feature_pipeline/build_pathology_binary_manifest.py`.
3. Train the main IAT-GAM model with `training/train.py` (joint
   classification + captioning, with switches for disabling LSTM/GAN/IAT
   components for ablations — see `run_iat_variants_queue.sh` and
   `run_phase2_featgan_ablation.sh` / `run_phase3_iatfix_ablation.sh`).
4. Train comparison baselines with `training/train_baseline.py` /
   `train_baseline2.py`, `feature_pipeline/train_baseline_classifier.py`,
   `feature_pipeline/train_baseline_with_caption.py`,
   `feature_pipeline/train_baseline_xgboost.py`, and
   `training/train_lit_baselines.py` / `train_lstm_baselines.py` — orchestrated
   by `run_baselines_with_caption_queue.sh` and `run_param_matched_baselines_queue.sh`
   (baselines are parameter-matched to the proposed model, ~2.1-2.7M backbone
   params, for a fair efficiency comparison).
5. Generate final metrics with `evaluation/generate_results.py` (and the
   `generate_results_baseline*.py` variants for the comparison models). This is
   the script that produces the `test_metrics_*.json` files under `outputs/logs/`.
6. Regenerate the paper figures from the logged metrics/CSVs with the scripts in
   `outputs/plots/paper_figures/` (`make_figures.py`, `make_training_curves.py`,
   `make_diagram.py`).

Other `run_*.sh` queue scripts: `run_binary_case_matrix.sh` /
`run_binary_improvements_queue.sh` / `run_binary_isolated_focal.sh` (binary
pathology classification sweeps with class-weighting/focal-loss variants),
`run_cross_dataset_queue.sh` (ROCO-only vs. ROCOv2-only vs. combined runs),
`run_full_metrics_eval.sh` and `run_case3_eval_retry.sh` (extended-metrics
evaluation and a retry pass), `run_remaining_scenarios_queue.sh`.

---

## Results

All numbers below are read directly out of the JSON files in `outputs/logs/`
and are **not re-derived or invented**. Dataset for all rows is the **combined
ROCO + ROCOv2** test split unless noted; accuracy/precision/recall/F1 are for the
**5-class** pathology task (one of the 5 classes is structurally empty in this
test split — see Known limitations). Augmentation is "aug_strong" with
`seed = 42` for all main-table single-seed runs (see Known limitations re:
multi-seed).

### Proposed model and ablations (combined ROCO+ROCOv2 test split)

| Config | 5-class Acc | Precision | Recall | F1 | Bal. Acc | Params | ms/sample | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr-D | SPICE | Source JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IAT-GAM (full: GAN+memory, IAT, LSTM) | 0.9490 | 0.9487 | 0.9484 | 0.9484 | 0.9484 | — | 23.12 | 0.217 | 0.106 | 0.213 | 0.237 | 0.424 | 0.136 | `test_metrics_gam_combined.json` |
| CNN+IATnew (no GAN) | 0.9501 | 0.9497 | 0.9495 | 0.9495 | 0.9495 | — | 10.07 | 0.239 | 0.119 | 0.202 | 0.227 | 0.369 | 0.118 | `test_metrics_iatnew_combined.json` |
| LightCNN (ROCO only) | 0.8726 | 0.8713 | 0.8719 | 0.8710 | 0.8719 | — | 8.03 | 0.187 | 0.086 | 0.167 | 0.181 | 0.228 | 0.081 | `test_metrics_lightcnn_roco.json` |
| MacroCNN (ROCO only) | 0.8962 | 0.8969 | 0.8953 | 0.8948 | 0.8953 | — | 8.69 | 0.137 (sum.) | 0.077 (sum.) | 0.142 (sum.) | 0.169 (sum.) | 0.242 (sum.) | 0.085 (sum.) | `test_metrics_macrocnn_roco.json` |

Note: removing GAN/memory-coupling (CNN+IATnew) is *not* worse than the full
GAM model on this combined split — accuracy is essentially flat (0.9501 vs.
0.9490) while inference is roughly **2.3x faster** (10.07 ms vs. 23.12 ms per
sample at batch size 16). The GAM variant does improve CIDEr-D (0.42 vs. 0.37)
and binary/3-class accuracy slightly. Binary-class accuracy is 1.0 for every
config above because the binary test split in these particular logs is
single-class (4,767 or 2,976 positive samples, 0 negative) — see Known
limitations.

### `case3` / `test_metrics.json` runs (ROCOv2-style full split, greedy vs. beam decoding)

| Config | decoding | 5-class Acc | Bal. Acc | AUROC (macro OvR, 5-class) | BLEU-1 | CIDEr-D | ms/sample |
|---|---|---|---|---|---|---|---|
| `case3_full_test_metrics.json` | beam=1 | 0.6612 | 0.7474 | 0.8849 | 0.262 | 0.263 | 5.43 |
| `test_metrics.json` | beam=5 | 0.6612 | 0.7474 | 0.8849 | 0.256 | 0.286 | 27.33 |

These two runs are on a different (larger, 5-way with a real negative/"none"
class) split than the combined-dataset table above, which is why AUROC is a
real number here instead of null and 5-class accuracy is lower — this split
is a genuinely harder multi-pathology discrimination task, not a regression.

### Parameter-matched baselines (combined split, backbones ~2.1-2.7M params)

| Backbone | Params (backbone) | 5-class Acc (cls-only) | 5-class Acc (+caption head) | Params (+caption head) | Note |
|---|---|---|---|---|---|
| ResNet-small | 2,229,429 | 0.9285 | 0.9338 | 15,697,473 | best param-matched baseline |
| DenseNet-style | 2,258,009 | 0.9252 | 0.9252 (frozen) | 19,515,765 | caption head trained on a **frozen** backbone (see limitations) |
| EfficientNet-small | 2,360,149 | 0.9112 | 0.9123 | 18,302,433 | |
| SqueezeNet-style | 2,693,573 | 0.9054 | 0.9093 | 29,769,937 | |
| MobileNet-style | 2,112,053 | 0.8987 | 0.9037 | 35,374,017 | |
| ShuffleNet-style | 2,270,525 | 0.8886 | 0.8975 | 31,821,129 | |
| MLP (non-conv) | 2,475,781 | 0.6831 | — | — | weakest baseline, no spatial inductive bias |
| XGBoost (keyword features, not deep) | n/a (179 trees, depth 6) | 0.9593 | — | — | trained on caption-keyword features, **not** image pixels — see note below |

The proposed IAT-GAM backbone (CNN+IATnew core) is in the same ~2-3M-parameter
class as these baselines, consistent with the project's "parameter-efficiency"
framing: it matches or exceeds every CNN baseline above except the keyword-feature
XGBoost model, which is not a fair image-classification comparison (see below).

**XGBoost caveat (from the JSON's own `note` field):** *"XGBoost trained on
keyword/text-derived features (caption length, modality/pathology keyword hits),
NOT deep pixel features -- no image-embedding feature pipeline exists in this
project. No neural parameter count applies; n_estimators/max_depth reported
instead."* Its 0.9593 accuracy is therefore not directly comparable to the
image-based CNN rows and is included for completeness, not as a SOTA claim.

**DenseNet caption-head caveat (from the JSON's own `note` field):** *"backbone
frozen (loaded from classification-only checkpoint); only a fresh BatchNorm2d +
LMMDecoder trained on captioning loss, to avoid the loss-explosion bug when
densenet's unnormalized forward_features() output was fed into LMMDecoder's
spatial cross-attention under full joint backprop."* Its captioning numbers
therefore come from a frozen-backbone workaround, not full joint training like
the other rows.

### Qualitative captioning (greedy vs. beam-5), `qualitative_greedy_vs_beam.json`

5 sample cases are logged comparing greedy decoding to beam-5 decoding against
the human reference caption. Beam search generally produces less-repetitive,
more anatomically specific captions (e.g. "sagittal radiograph of the right eye
shows a well defined lesion in the left maxillary sinus" vs. greedy's "mri of
the right hip showing a large mass in the left mandibular sinus" for the same
image) but neither fully recovers the ground-truth finding, consistent with the
modest BLEU/METEOR/CIDEr scores above.

### Dataset duplicate audit (`duplicate_audit_report.txt` / `duplicate_audit_full.json`)

Across 97,333 audited images: 0 missing/unreadable, 29 exact-MD5-duplicate
groups (29 excess images), 9,495 exact-perceptual-hash-match groups (9,507
excess images), and 1,050 near-duplicate pairs (hash distance 1-5, bucketed
approximate scan, not exhaustive) — including several ROCO<->ROCOv2 cross-dataset
near-duplicates. This is disclosed as-is; it has not been re-verified by this
push, only transcribed from the existing audit report.

### Methodology notes actually present in the logs/config

- Seed: `RANDOM_SEED = 42` in `config.py`; multi-seed training-curve CSVs exist
  for seeds 42-46 (`train_log_aug_strong_seed4{2..6}_micro2m_...csv`) for the
  training-curve/variance check, but the final test-metrics JSON files above
  are single-seed (seed 42) runs.
- Split ratios: 70/15/15 train/val/test (`config.py`, `TRAIN_SPLIT`/`VAL_SPLIT`/`TEST_SPLIT`).
  Baseline backbones were deliberately parameter-matched to the proposed
  model's ~2.1-2.7M-parameter backbone (see table above) for a fair efficiency
  comparison rather than matching on FLOPs or depth.
- Decoding: beam width 5 used for most captioning evaluation runs; beam width 1
  (greedy) used for the `case3_full_test_metrics.json` comparison run.
- Efficiency figures (`ms_per_sample`, `samples_per_sec`) are measured at
  batch size 16, as recorded in each JSON's `efficiency` block.

---

## Known limitations

These are transcribed from fields actually present in this project's own logs
and code (the `note` fields in the JSON files above, and commentary in
`outputs/plots/paper_figures/build_docx.py`, which documents the limitations
section of the associated manuscript). Nothing below is invented; where a
commonly-expected limitation could not be verified from the available JSON, it
is labeled as such.

1. **Checkpoint-filename collision (verified, from `build_docx.py`):** *"a
   checkpoint-filename collision during this study's experimentation
   overwrote the ROCO-v1-only and ROCOv2-only model weights for the
   CNN+IATnew (no-GAN) and IAT+GAM (GAN+memory-coupling) configurations with
   their combined-dataset counterparts, because the training script's
   checkpoint-naming convention encodes hyperparameters but not dataset
   source; as a result, only training-time validation accuracy ..., not full
   extended test metrics (confusion matrix, captioning, AUROC), survives for
   the per-dataset (non-combined) versions of these two configurations."*
   This is why the results table above only has combined-split (not
   ROCO-only / ROCOv2-only) extended metrics for the full IAT-GAM and
   CNN+IATnew configurations, while LightCNN/MacroCNN do have ROCO-only rows.

2. **AUROC structural null (verified, from JSON + `build_docx.py`):**
   `auroc_macro_ovr` is `null` in every `test_metrics_*_combined.json` file
   above because the 5th class is empty (0 samples) in that test split, making
   one-vs-rest AUROC for that class mathematically undefined; scikit-learn
   correctly returns null rather than a fabricated value, and this project
   reports it as null rather than substituting a number. On the different,
   harder split used by `case3_full_test_metrics.json` / `test_metrics.json`
   (which has a real 5th class), AUROC is a real value (0.6440 binary, 0.8102
   3-class, 0.8849 5-class, macro OvR).

3. **DenseNet caption-head frozen-backbone workaround (verified, from JSON
   `note` field):** see the DenseNet caveat in the Results table above —
   trained to avoid a loss-explosion bug from feeding DenseNet's unnormalized
   features into the caption decoder's cross-attention under full joint
   backprop.

4. **Single-seed point estimates (verified, from `build_docx.py`):** *"all
   results in this revision are single-seed (seed = 42); we have not yet run a
   multi-seed significance study, and point estimates should be read with that
   caveat."* Multi-seed training-curve CSVs exist for seeds 42-46 for one
   configuration (`micro2m_ae1.0_softcw_binpow0.75`) but a full multi-seed
   test-metrics re-evaluation was not found in `outputs/logs/`.
   `scripts/significance_test.py` exists in this repo for running such a test.
- **GAN/memory-coupling (GAM) benefit** is referenced as a limitation topic in
  `build_docx.py` ("limited accuracy benefit over the plain CNN+IAT
  configuration... across ROCO v1, ROCOv2, and their combination") — consistent
  with the Results table above, where CNN+IATnew (no GAN) performs
  essentially on par with the full GAM model on 5-class accuracy.
- **Binary-class "accuracy = 1.0" rows** in the combined-split JSON files are
  not a genuine binary-classification result — the logged confusion matrices
  show the negative class has 0 samples in that split, so this is a degenerate
  single-class result, not a claim of perfect binary classification.
- A specific claim of "2 configs affected" by the checkpoint collision is
  consistent with what's documented above (CNN+IATnew and IAT+GAM), but this
  push did not independently re-verify which exact checkpoint files on disk
  were overwritten — that statement is taken as-is from the project's own
  disclosure in `build_docx.py`.

---

## Paper / manuscript

The full manuscript associated with this project is **intentionally not
included** in this repository (by request), along with model checkpoints and
the raw dataset. Only code and result artifacts (metrics, logs, training
curves, and figures) are published here.
