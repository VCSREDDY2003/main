"""
Generates a standalone "Statistical Analysis and Computational Complexity"
DOCX report (Sections 5.6 and 5.7) for IAT-GAM-LSTM.

Reads:
  - outputs/logs/train_log_<hparams>.csv               (main 100-epoch run)
  - outputs/logs/train_log_<hparams>_seed<S>.csv        (multi-seed runs, S in --seeds)
  - outputs/logs/test_metrics.json                      (held-out test metrics)
  - outputs/logs/group_metrics.json                     (per-modality-group metrics,
                                                          from evaluation/evaluate_by_group.py)
  - outputs/logs/complexity_metrics.json                (from evaluation/profile_complexity.py)
  - outputs/plots/fig_binary_curves.png
  - outputs/plots/fig_3class_curves.png
  - outputs/plots/fig_5class_curves.png
  - outputs/plots/fig_group_metrics.png

Writes:
  outputs/IATGAMLSTM_Statistical_and_Complexity_Analysis.docx

Usage:
    python evaluation/generate_statistical_complexity_report.py --hparams aug_strong --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import config


# ----------------------------------------------------------------------
# DOCX helpers (mirrors evaluation/generate_report.py)
# ----------------------------------------------------------------------
def heading1(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(12)
    run.font.name = "Times New Roman"
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)


def heading2(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.italic = True
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)


def body(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)


def fig_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(10)


def tbl_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)


def add_figure(doc, img_path, caption_text, width_in=6.0):
    if not os.path.exists(img_path):
        body(doc, f"[Figure not available: {os.path.basename(img_path)}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_in))
    fig_caption(doc, caption_text)


def add_table(doc, headers, rows, title):
    tbl_caption(doc, title)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "D9D9D9")
        tcPr.append(shd)

    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            cell = t.rows[ri + 1].cells[ci]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            run.font.name = "Times New Roman"
            run.font.size = Pt(9)

    doc.add_paragraph()


def num(x, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def pct(x) -> str:
    return f"{100.0 * float(x):.2f}"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Full checklist of metrics tracked across the pipeline (Section 5.6 reference table)
METRICS_CHECKLIST = [
    ("Captioning", "BLEU-1", "Unigram precision vs. reference caption (Eq. 3.5/3.28a).", "evaluation/metrics.py:bleu_score"),
    ("Captioning", "BLEU-4", "Up-to-4-gram precision with brevity penalty.", "evaluation/metrics.py:bleu_score"),
    ("Captioning", "ROUGE-L", "Longest-common-subsequence F-measure.", "evaluation/metrics.py:rouge_l"),
    ("Captioning", "CIDEr-D", "TF-IDF-weighted n-gram cosine consensus score.", "evaluation/metrics.py:cider_d"),
    ("Classification", "Accuracy", "Overall accuracy, binary / 3-class / 5-class (Eq. 3.29).", "evaluation/metrics.py:classification_metrics"),
    ("Classification", "Precision (macro)", "Macro-averaged precision across classes.", "evaluation/metrics.py:classification_metrics"),
    ("Classification", "Recall (macro)", "Macro-averaged recall across classes.", "evaluation/metrics.py:classification_metrics"),
    ("Classification", "F1 (macro)", "Macro-averaged F1 across classes.", "evaluation/metrics.py:classification_metrics"),
    ("Classification", "Per-class Precision/Recall/F1", "Per-class breakdown for binary/3-class/5-class heads, logged every epoch.", "evaluation/metrics.py:classification_metrics (num_classes=)"),
    ("Generative quality", "SSIM (IAT vs. original)", "Structural similarity of IAT-augmented image vs. source (Eq. 3.10).", "evaluation/metrics.py:ssim_batch"),
    ("Generative quality", "SSIM (GAN vs. original)", "Structural similarity of cGAN-reconstructed image vs. source (Eq. 3.11-3.14).", "evaluation/metrics.py:ssim_batch"),
    ("Training dynamics", "Composite loss (train/val)", "L_total = L_cap + L_summary + L_cls + lambda_1 L_GAN + lambda_2 L_temporal (Eq. 3.31).", "training/losses.py:CompositeLoss"),
    ("Training dynamics", "Per-class accuracy/F1 curves", "Binary / 3-class / 5-class per-class F1 vs. epoch over the full run.", "evaluation/plot_label_curves.py"),
    ("Per-group breakdown", "Captioning + classification by modality group", "X-ray / CT / MRI / Ultrasound / Other-video, grouped by keyword-derived 5-class label.", "evaluation/evaluate_by_group.py"),
    ("Statistical analysis", "Mean / Std / 95% CI (multi-seed)", "Across independent training runs with different init/shuffle seeds.", "this script, Section 5.6"),
    ("Statistical analysis", "Paired t-test", "Train vs. validation 5-class accuracy across seeds (same seeds, paired).", "this script, Section 5.6 (scipy.stats.ttest_rel)"),
    ("Statistical analysis", "One-way ANOVA", "5-class accuracy across the >=3 seed runs.", "this script, Section 5.6 (scipy.stats.f_oneway)"),
    ("Computational complexity", "Parameter count per module", "Total / trainable parameters for IAT, GAN G/D, CNN, Transformer encoder, fusion, LSTM, LMM decoder.", "evaluation/profile_complexity.py"),
    ("Computational complexity", "FLOPs per sample", "Forward-pass FLOPs at batch size 1 (torch.profiler).", "evaluation/profile_complexity.py"),
    ("Computational complexity", "Peak training memory (MB)", "Peak GPU memory for one forward+backward step at the training batch size.", "evaluation/profile_complexity.py"),
    ("Computational complexity", "Inference latency / throughput", "ms/sample and samples/sec across a batch-size sweep.", "evaluation/profile_complexity.py"),
    ("Computational complexity", "Training time per epoch", "Wall-clock seconds per epoch, read from the training log.", "evaluation/profile_complexity.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="aug_strong")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    args = parser.parse_args()

    main_log = load_log(os.path.join(config.LOGS_DIR, f"train_log_{args.hparams}.csv"))
    seed_logs = {
        s: load_log(os.path.join(config.LOGS_DIR, f"train_log_{args.hparams}_seed{s}.csv"))
        for s in args.seeds
    }
    seed_logs = {s: rows for s, rows in seed_logs.items() if rows}

    test_metrics = load_json(os.path.join(config.LOGS_DIR, "test_metrics.json"))
    group_metrics = load_json(os.path.join(config.LOGS_DIR, "group_metrics.json"))
    complexity = load_json(os.path.join(config.LOGS_DIR, "complexity_metrics.json"))

    out_path = os.path.join(config.OUTPUT_DIR, "IATGAMLSTM_Statistical_and_Complexity_Analysis.docx")

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.left_margin = Inches(1.18)
    section.right_margin = Inches(1.18)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(10)

    # ==================================================================
    # 5.6 Statistical Analysis
    # ==================================================================
    heading1(doc, "5.6. Statistical Analysis")

    heading2(doc, "5.6.1. Per-class Training Curves over the Full Run")
    n_epochs = len(main_log)
    body(doc,
         f"To verify that the IAT-GAM-LSTM model converges consistently across all label "
         f"granularities, the model was trained for {n_epochs} epochs with the '{args.hparams}' "
         f"hyperparameter set, logging per-class precision/recall/F1 for the binary "
         f"(static-image vs. video/other), 3-class (radiograph / cross-sectional / other), "
         f"and 5-class (X-ray / CT / MRI / Ultrasound / Other) classification heads at every "
         f"epoch. Figures 1-3 show the classification loss and per-class F1 trajectories.")
    add_figure(doc, os.path.join(config.PLOTS_DIR, "fig_binary_curves.png"),
               "Figure 1. Binary classification: loss and per-class F1 vs. epoch.")
    add_figure(doc, os.path.join(config.PLOTS_DIR, "fig_3class_curves.png"),
               "Figure 2. 3-class classification: loss and per-class F1 vs. epoch.")
    add_figure(doc, os.path.join(config.PLOTS_DIR, "fig_5class_curves.png"),
               "Figure 3. 5-class classification: loss and per-class F1 vs. epoch.")

    heading2(doc, "5.6.2. Per-modality-group Evaluation")
    if group_metrics:
        body(doc,
             "The held-out test split was further partitioned by the keyword-derived imaging "
             "modality label (5-class) and re-evaluated within each group, to check whether "
             "captioning and classification quality is uniform across modalities or "
             "concentrated in over-represented groups.")
        rows = []
        for name, m in group_metrics.items():
            if m.get("n", 0) == 0:
                continue
            cap = m["captioning"]["caption"]
            cls5 = m["classification"]["5class"]
            rows.append([
                name, m["n"], num(cap["bleu1"]), num(cap["bleu4"]), num(cap["rouge_l"]),
                num(cap["cider_d"]), pct(cls5["accuracy"]),
            ])
        add_table(
            doc,
            ["Modality group", "N", "BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr-D", "5-class Acc (%)"],
            rows,
            "Table 1. Captioning and classification quality by imaging-modality group.",
        )
        add_figure(doc, os.path.join(config.PLOTS_DIR, "fig_group_metrics.png"),
                   "Figure 4. Per-modality-group metrics (bar chart).")
    else:
        body(doc, "Per-group metrics not available (run evaluation/evaluate_by_group.py).")

    heading2(doc, "5.6.3. Multi-seed Reproducibility")
    if seed_logs:
        body(doc,
             f"To assess sensitivity to weight initialization and data-loader shuffling, the "
             f"model was retrained from scratch with {len(seed_logs)} independent random seeds "
             f"({', '.join(str(s) for s in seed_logs)}), keeping the '{args.hparams}' "
             f"hyperparameter set and dataset split fixed (random seed {config.RANDOM_SEED} for "
             f"the split). For each seed, the final-epoch training and validation 5-class "
             f"accuracy and total composite loss (Eq. 3.31) are reported below.")

        final_train_acc = np.array([float(rows[-1]["train_acc_5class_accuracy"]) for rows in seed_logs.values()])
        final_val_acc = np.array([float(rows[-1]["val_acc_5class_accuracy"]) for rows in seed_logs.values()])
        final_train_loss = np.array([float(rows[-1]["train_total"]) for rows in seed_logs.values()])
        final_val_loss = np.array([float(rows[-1]["val_total"]) for rows in seed_logs.values()])

        def mean_std_ci(arr: np.ndarray) -> tuple:
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            if len(arr) > 1:
                ci = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=std / np.sqrt(len(arr)))
            else:
                ci = (mean, mean)
            return mean, std, ci

        rows_out = []
        for label, arr in (
            ("Train 5-class accuracy", final_train_acc),
            ("Val 5-class accuracy", final_val_acc),
            ("Train total loss", final_train_loss),
            ("Val total loss", final_val_loss),
        ):
            mean, std, ci = mean_std_ci(arr)
            rows_out.append([label, num(mean, 4), num(std, 4), f"[{num(ci[0], 4)}, {num(ci[1], 4)}]"])

        add_table(
            doc,
            ["Metric (final epoch)", "Mean", "Std. dev.", "95% CI"],
            rows_out,
            f"Table 2. Multi-seed (n={len(seed_logs)}) mean / std. dev. / 95% confidence interval, "
            f"final epoch.",
        )

        # Paired t-test: train vs. val 5-class accuracy, same seeds
        if len(final_train_acc) > 1:
            t_stat, p_val = stats.ttest_rel(final_train_acc, final_val_acc)
            body(doc,
                 f"A paired t-test comparing final-epoch train vs. validation 5-class accuracy "
                 f"across the {len(final_train_acc)} seeds gives t={num(t_stat, 3)}, "
                 f"p={num(p_val, 4)}, indicating "
                 + ("a statistically significant difference (p < 0.05) between train and "
                    "validation accuracy at the final epoch."
                    if p_val < 0.05 else
                    "no statistically significant difference (p >= 0.05) between train and "
                    "validation accuracy at the final epoch, i.e. the model generalizes "
                    "consistently to held-out validation data across seeds."))

            f_stat, p_anova = stats.f_oneway(*[
                [float(r["val_acc_5class_accuracy"]) for r in rows] for rows in seed_logs.values()
            ])
            body(doc,
                 f"A one-way ANOVA across the per-epoch validation 5-class accuracy series of "
                 f"the {len(seed_logs)} seed runs gives F={num(f_stat, 3)}, p={num(p_anova, 4)}, "
                 + ("indicating that the choice of random seed has a statistically significant "
                    "effect on the validation accuracy trajectory (p < 0.05)."
                    if p_anova < 0.05 else
                    "indicating no statistically significant effect of random seed on the "
                    "validation accuracy trajectory (p >= 0.05), supporting the reproducibility "
                    "of the reported results."))
    else:
        body(doc,
             "Multi-seed logs not found. Run, e.g.:\n"
             "  python training/train.py --hparams aug_strong --seed 42 ...\n"
             "  python training/train.py --hparams aug_strong --seed 43 ...\n"
             "  python training/train.py --hparams aug_strong --seed 44 ...\n"
             "and re-run this script.")

    heading2(doc, "5.6.4. Performance Metrics Checklist")
    body(doc,
         "Table 3 summarizes every performance metric tracked across the IAT-GAM-LSTM "
         "pipeline, the category it belongs to, and the script that produces it.")
    add_table(
        doc,
        ["Category", "Metric", "Definition", "Source"],
        [[c, m, d, s] for c, m, d, s in METRICS_CHECKLIST],
        "Table 3. Full performance-metrics checklist.",
    )

    # ==================================================================
    # 5.7 Computational Complexity
    # ==================================================================
    heading1(doc, "5.7. Computational Complexity")

    if complexity:
        heading2(doc, "5.7.1. Model Size by Module")
        body(doc,
             f"Table 4 reports the parameter count of each IAT-GAM-LSTM component "
             f"(hyperparameter set: '{complexity.get('hparams', args.hparams)}', "
             f"device: {complexity.get('device', 'unknown')}).")
        rows = []
        for name, p in complexity.get("module_parameters", {}).items():
            rows.append([name, f"{p['total']:,}", f"{p['trainable']:,}"])
        add_table(doc, ["Module", "Total parameters", "Trainable parameters"], rows,
                  "Table 4. Parameter count per module.")

        heading2(doc, "5.7.2. FLOPs and Peak Training Memory")
        flops = complexity.get("flops_per_sample")
        peak_mem = complexity.get("peak_train_memory_mb")
        bs = complexity.get("train_batch_size")
        body(doc,
             f"A single forward pass (batch size 1) requires "
             f"{(flops / 1e9) if flops else 0:.2f} GFLOPs. A forward+backward step at the "
             f"training batch size ({bs}) reaches a peak GPU memory usage of "
             f"{num(peak_mem, 1)} MB.")

        heading2(doc, "5.7.3. Inference Latency and Throughput")
        latency = complexity.get("inference_latency", {})
        rows = []
        for bs_str, m in latency.items():
            rows.append([bs_str, num(m["ms_per_sample"], 3), num(m["samples_per_sec"], 2)])
        add_table(doc, ["Batch size", "Latency (ms/sample)", "Throughput (samples/sec)"], rows,
                  "Table 5. Inference latency / throughput vs. batch size.")
        body(doc,
             "Latency decreases and throughput increases with batch size due to better GPU "
             "utilization, but for real-time single-frame inference the batch-size-1 latency "
             "is the relevant figure.")

        n_epochs_logged = complexity.get("n_epochs_logged")
        if n_epochs_logged:
            body(doc,
                 f"The training log for hyperparameter set '{complexity.get('hparams', args.hparams)}' "
                 f"contains {n_epochs_logged} logged epoch(s) at the time this report was generated.")
    else:
        body(doc,
             "Complexity metrics not found. Run:\n"
             "  python evaluation/profile_complexity.py --hparams aug_strong\n"
             "and re-run this script.")

    doc.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
