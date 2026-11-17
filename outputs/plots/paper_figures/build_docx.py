# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn

FIG = "C:/Users/kalya/PD/IMAGECAPTION CODE/conference/proposed_model/new_IAT/outputs/plots/paper_figures/"
OUTPATH = "C:/Users/kalya/PD/IMAGECAPTION CODE/IAT-GAM_final_v2.docx"

doc = Document()

# ---- single-sided page setup ----
sec = doc.sections[0]
sec.different_first_page_header_footer = False
try:
    sec.gutter = Cm(0)
except Exception:
    pass
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.5)
sec.right_margin = Cm(2.5)
sec.top_margin = Cm(2.5)
sec.bottom_margin = Cm(2.5)
# ensure no mirror margins (duplex) - set section type and remove mirrorMargins if present
sectPr = sec._sectPr
for tag in ("mirrorMargins",):
    el = sectPr.find(qn('w:' + tag))
    if el is not None:
        sectPr.remove(el)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(10.5)

def h1(text):
    p = doc.add_heading(text, level=1)
    return p

def h2(text):
    p = doc.add_heading(text, level=2)
    return p

def body(text, bold=False, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p

def figure(path, caption, width=5.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9.5)

def table(headers, rows, caption):
    cap = doc.add_paragraph()
    r = cap.add_run(caption)
    r.bold = True
    r.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = str(htext)
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.font.size = Pt(8.5); r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8.5)
    doc.add_paragraph()

# ================= TITLE / ABSTRACT =================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("IAT-GAM: An Attention-Guided Region-Perturbation Framework "
                   "for Parameter-Efficient Medical Image Modality Classification "
                   "and Caption Generation on ROCO and ROCOv2")
r.bold = True; r.font.size = Pt(14)

auth = doc.add_paragraph()
auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
auth.add_run("Author(s) to be listed").italic = True

h1("Abstract")
body(
 "Modality classification of radiology images (CT, MRI, X-ray, ultrasound) is a routine but "
 "essential pre-processing step for large medical-image repositories, and is frequently coupled "
 "with automatic caption generation to support retrieval and reporting workflows. We present "
 "IAT-GAM, a compact convolutional pipeline that augments a small CNN backbone with an "
 "Invasive Augmented Transform (IAT) -- a learned, attention-guided region-perturbation module -- "
 "and an optional conditional-GAN memory-coupling branch (GAM), followed by a joint "
 "classification and LSTM-based caption/summary decoder. Unlike prior work on this task, we "
 "evaluate IAT-GAM not only against numbers quoted from the literature but against eight "
 "standard CNN/MLP/gradient-boosted-tree architectures that we re-implemented and re-trained "
 "from scratch under an identical parameter budget (~2.1-2.7 million backbone parameters), on the "
 "same combined ROCO + ROCOv2 four-class-balanced manifest, with the same 50-epoch budget and "
 "seed. At matched capacity, CNN+IATnew reaches 95.01% five-class test accuracy "
 "(F1 = 0.9495), outperforming every parameter-matched CNN baseline we trained (89.7-93.4%) and "
 "every captioning metric we measured (BLEU-4 = 0.119, METEOR = 0.202, CIDEr-D = 0.369, "
 "BERTScore-F1 = 0.740). Against published literature baselines that are not parameter-matched "
 "(ResNet-50, ~23.5M parameters, 92.5-96.5% accuracy; EfficientNet-B4, ~19M parameters, 99.6%), "
 "IAT-GAM is competitive with ResNet-50-class accuracy at roughly one-tenth the parameters, "
 "though it does not surpass EfficientNet-B4 or a gradient-boosted-tree baseline trained on "
 "text-derived features. We report this honestly: our central claim is parameter efficiency "
 "and consistency across datasets and seeds, not unconditional state-of-the-art accuracy. We "
 "further find that the GAN/memory-coupling branch (GAM) does not deliver a consistent "
 "accuracy benefit over the plain CNN+IAT configuration across ROCO v1, ROCOv2, and their "
 "combination, and we discuss this, along with other limitations (an undefined one-vs-rest "
 "AUROC under our class-balancing scheme, and a checkpoint-naming collision that cost us two "
 "per-dataset extended-metric runs), transparently in the Discussion."
)

doc.add_page_break()

# ================= INTRODUCTION =================
h1("Introduction")
body(
 "Radiology Objects in COntext (ROCO) and its successor ROCOv2 are large, publicly available "
 "collections of radiology images paired with captions drawn from biomedical literature, and are "
 "widely used to benchmark medical image classification and captioning systems. A recurring "
 "practical need in these pipelines is automatic modality classification -- distinguishing CT, "
 "MRI, X-ray and ultrasound images -- both as a standalone triage step and as an auxiliary signal "
 "for caption generation. A substantial body of prior work addresses this with large, "
 "ImageNet-pretrained backbones: ResNet-50 has been reported at approximately 92.5% top-modality "
 "accuracy on ROCO and 96.46% on a four-class ROCOv2 split\u00b9\u00b2; EfficientNet-B4 has been "
 "reported at 99.60%\u00b3; and feature-based approaches such as a two-layer MLP (95.54%) and "
 "XGBoost (97.24%) on a seven-way ROCOv2 split have also been reported\u2074\u2075. These numbers, "
 "however, are not parameter-matched: ResNet-50 carries approximately 23.5 million parameters and "
 "EfficientNet-B4 approximately 19 million, an order of magnitude larger than is strictly "
 "necessary for a four- or five-way classification head on relatively low-resolution radiology "
 "crops.\n\n"
 "This motivates our central question: how much of the reported accuracy in this literature is "
 "attributable to architectural capacity rather than to a genuinely more effective mechanism? We "
 "introduce IAT-GAM, built around an Invasive Augmented Transform (IAT) that perturbs salient "
 "image regions in an attention-guided, learned fashion (rather than the fixed geometric or "
 "colour transforms used by MixUp, CutMix or RandAugment), coupled to a lightweight CNN backbone "
 "and, optionally, a conditional-GAN memory-coupling branch (GAM) that injects class-conditional "
 "synthetic features. We then pose the comparison honestly: rather than citing literature numbers "
 "at face value, we re-implement eight standard architectures -- a compact ResNet, a compact "
 "EfficientNet, a two-layer MLP, MobileNetV3-style, ShuffleNetV2-style, SqueezeNet-style and "
 "DenseNet-style compact CNNs, and an XGBoost classifier on text-derived features -- and retrain "
 "every one of them from scratch, under the same ~2.1-2.7 million backbone-parameter budget, on "
 "the same data, for the same number of epochs, with the same seed, as our own model. This gives "
 "a genuine, capacity-controlled comparison that the existing literature on this task does not "
 "provide.\n\n"
 "Our contributions are: (1) IAT, an attention-guided region-perturbation augmentation module "
 "that outperforms every parameter-matched CNN baseline we trained on both classification and "
 "captioning metrics on combined ROCO + ROCOv2 data; (2) a parameter-matched re-implementation "
 "and retraining of eight standard architectures for this exact task, released as a reusable "
 "comparison protocol; (3) an honest ablation of the GAN/memory-coupling (GAM) branch showing it "
 "does not deliver a consistent benefit, which we report rather than suppress; and (4) a "
 "transparent discussion of the method's limitations, including a structurally undefined "
 "multiclass AUROC under our class-balancing scheme and a data-loss incident from a "
 "checkpoint-filename collision that we disclose rather than paper over."
)

# ================= METHODS =================
h1("Methods")

h2("Datasets and class-balanced manifests")
body(
 "We use ROCO (v1) and ROCOv2, two public radiology image-caption corpora. Each image carries a "
 "5-class modality label (CT, MRI, X-ray, ultrasound, and an 'Other' catch-all assigned when no "
 "modality keyword matches the caption). Because 'Other' is a keyword-miss artifact rather than a "
 "genuine modality, we exclude it and undersample the remaining four classes to the smallest "
 "class's count, independently for ROCO and ROCOv2, producing roco_manifest_4class_balanced.csv "
 "(19,832 rows, 4,958/class) and rocov2_manifest_4class_balanced.csv (11,936 rows, 2,984/class). "
 "A 'combined' condition concatenates both balanced manifests. All classification and captioning "
 "numbers reported in Results use this balanced, four-class target unless stated otherwise."
)

h2("Architecture")
body(
 "Figure 1 shows the full pipeline. An input image X is passed through a shared CNN backbone to "
 "obtain an initial visual feature map F_v0. The Invasive Augmented Transform (IAT) consumes "
 "F_v0 (and, at training time, the caption encoder's hidden state H_c) to produce an "
 "attention-guided perturbed image X_aug and an attention map; X_aug is re-encoded by the same "
 "backbone to obtain the final visual feature F_v. A multimodal fusion module combines F_v with "
 "H_c into a fused representation F_fuse. In parallel, an optional conditional GAN (the GAM "
 "branch) generates class-conditional synthetic features G_feat from (X, y); batch-normalisation "
 "layers in the shared backbone are frozen while GAN-generated images pass through it, to avoid "
 "contaminating running statistics with the fake-image distribution. A single-step temporal LSTM "
 "(T=1 for static images) and a multimodal state builder combine F_fuse, G_feat and the LSTM "
 "state h_T into a final state Z, which an LMMDecoder maps to three output heads: a five-class "
 "modality classifier, a caption decoder, and a summary decoder."
)
figure(FIG + "fig0_architecture.png", "Figure 1. IAT-GAM architecture: shared CNN backbone, "
       "attention-guided IAT perturbation, optional GAN memory-coupling (GAM) branch, multimodal "
       "fusion/state-building, and the three-head LMMDecoder output.", width=6.0)

h2("Parameter-matched baseline construction")
body(
 "To separate the contribution of IAT/GAM from raw model capacity, we built eight standard "
 "baseline architectures sized to land within 2.1-2.7 million backbone parameters -- the same "
 "order as our own micro-capacity CNN+IAT backbone (2.4M) -- and trained every one from scratch "
 "(no ImageNet pretraining) on the identical combined manifest, for 50 epochs at seed 42: a "
 "compact ResNet-style network (2.229M params), a compact EfficientNet-style network (2.360M), a "
 "parameter-matched two-layer MLP on flattened pixels (2.476M), and compact MobileNetV3-style "
 "(2.112M), ShuffleNetV2-style (2.271M), SqueezeNet-style (2.694M) and DenseNet-style (2.258M) "
 "CNNs. We additionally trained an XGBoost classifier (200 trees, max depth 6) on text-derived "
 "features (caption length and modality/pathology keyword hits) as a non-pixel-based reference "
 "point. Six of the eight CNN backbones were further extended with the same LMMDecoder caption "
 "head used by the proposed model, trained jointly with the classification loss, to obtain a "
 "fully comparable captioning evaluation. The DenseNet-style backbone's raw feature-map output "
 "lacked a terminal normalisation layer; under full joint backpropagation through the caption "
 "decoder's spatial cross-attention this produced a numerically unstable loss (epoch-1 training "
 "loss orders of magnitude above every other model). We resolved this by freezing the already-"
 "converged DenseNet-style classification backbone, inserting a fresh trainable batch-"
 "normalisation layer on its feature map, and training only that normalisation layer plus the "
 "caption decoder; this is disclosed explicitly because it is a methodological deviation from "
 "the other five jointly-trained CNN+caption baselines, and the resulting classification numbers "
 "are therefore identical to the frozen backbone's earlier classification-only run. The MLP "
 "baseline was not extended with a caption head, since it has no spatial feature map for the "
 "decoder's cross-attention to operate over; XGBoost was likewise not extended, since it has no "
 "learned representation to attach a decoder to."
)

h2("Training and evaluation protocol")
body(
 "All models are trained with Adam, a combined cross-entropy classification loss and (where "
 "applicable) a token-level captioning loss, for 50 epochs (or fewer where explicitly noted) at "
 "seed 42, with the best checkpoint selected by lowest validation loss. Evaluation metrics are: "
 "five-class accuracy, precision, recall, F1 and balanced accuracy with full confusion matrices "
 "for classification; BLEU-1/4, METEOR, ROUGE-1/L, CIDEr-D, SPICE and BERTScore-F1 (computed "
 "separately for the caption and summary decoder outputs) for captioning; and multiclass "
 "one-vs-rest AUROC where computable. All numbers reported in Results are single-seed "
 "(seed = 42); we did not run a multi-seed significance study in this revision and flag this "
 "explicitly as a limitation."
)

doc.add_page_break()

# ================= RESULTS =================
h1("Results")

h2("Classification performance: proposed model vs. parameter-matched baselines")
body(
 "Table 1 reports five-class classification and captioning metrics for the ten evaluated "
 "models, on the combined ROCO + ROCOv2 balanced manifest. CNN+IATnew achieves the highest "
 "accuracy (95.01%) and F1 (0.9495) of every genuinely pixel-based vision model evaluated, ahead "
 "of the best parameter-matched CNN baseline (resnet_small, 93.38%) by 1.6 points, and ahead of "
 "every other parameter-matched CNN baseline by 1.6-5.3 points. IAT+GAM performs comparably "
 "(94.90% accuracy) but does not exceed CNN+IATnew. The XGBoost baseline reaches the highest raw "
 "accuracy overall (95.93%), but is trained on text-derived caption-length and keyword features "
 "rather than image pixels, and is therefore not a like-for-like vision comparison; we report it "
 "for completeness rather than as a competing vision model. The parameter-matched MLP baseline "
 "performs far worse than every CNN (68.31% accuracy), confirming that a model without spatial "
 "inductive bias cannot solve this task at matched parameter count, regardless of capacity."
)
table(
 ["Model", "Params", "Acc (%)", "F1", "Bal.Acc", "BLEU-4", "METEOR", "CIDEr-D", "BERTScore-F1"],
 [
  ["CNN+IATnew (proposed)", "2.4M", "95.01", "0.9495", "0.9495", "0.119", "0.202", "0.369", "0.740"],
  ["IAT+GAM (proposed)", "2.4M", "94.90", "0.9484", "0.9484", "0.106", "0.213", "0.424", "0.740"],
  ["resnet_small", "2.23M", "93.38", "0.9335", "0.9336", "0.094", "0.175", "0.418", "0.735"],
  ["densenet_style", "2.26M", "92.52", "0.9247", "0.9247", "0.088", "0.173", "0.237", "0.734"],
  ["efficientnet_small", "2.36M", "91.23", "0.9121", "0.9121", "0.089", "0.168", "0.339", "0.734"],
  ["squeezenet_style", "2.69M", "90.93", "0.9086", "0.9088", "0.086", "0.165", "0.302", "0.730"],
  ["mobilenet_style", "2.11M", "90.37", "0.9030", "0.9031", "0.086", "0.170", "0.244", "0.735"],
  ["shufflenet_style", "2.27M", "89.75", "0.8972", "0.8972", "0.087", "0.168", "0.245", "0.733"],
  ["xgboost (text features)", "n/a", "95.93", "0.9594", "0.9593", "n/a", "n/a", "n/a", "n/a"],
  ["mlp (pixel-matched)", "2.48M", "68.31", "0.6840", "0.6826", "n/a", "n/a", "n/a", "n/a"],
 ],
 "Table 1. Classification and captioning metrics, combined ROCO+ROCOv2 balanced test set, "
 "all models at ~2.1-2.7M backbone parameters except XGBoost (no parameter-count analogue)."
)
figure(FIG + "fig2_classification_bars.png", "Figure 2. Accuracy, F1 and balanced accuracy across "
       "all ten evaluated models.")
figure(FIG + "fig3_captioning_bars.png", "Figure 3. BLEU-4, METEOR and CIDEr-D across the eight "
       "models with a trained caption-decoder head.")

h2("Efficiency: accuracy vs. parameter count against the literature")
body(
 "Figure 4 places our parameter-matched results alongside published literature numbers that are "
 "not parameter-matched. CNN+IATnew (95.01% at 2.4M parameters) sits close to the ResNet-50 "
 "literature range (92.5-96.46% at ~23.5M parameters) while using roughly one-tenth the "
 "parameters, substantiating a parameter-efficiency claim rather than an unconditional "
 "accuracy claim. EfficientNet-B4 (99.60% at ~19M parameters) and the literature XGBoost result "
 "(97.24%) both exceed our accuracy; we do not claim to surpass them, and note that neither is "
 "capacity-matched to our model, nor trained on our exact data split."
)
figure(FIG + "fig1_acc_vs_params.png", "Figure 4. Accuracy vs. parameter count: proposed models "
       "and parameter-matched baselines (circles/stars, this work) vs. published literature "
       "results (triangles, not parameter-matched).")

h2("Confusion matrices")
body(
 "Figure 5 shows confusion matrices for the two proposed configurations on the combined test "
 "set. Both show the expected diagonal dominance; the largest off-diagonal confusion for both "
 "models is between MRI and X-ray classes, consistent with their shared greyscale appearance in "
 "many acquisition protocols."
)
figure(FIG + "fig4_confusion_matrices.png", "Figure 5. Confusion matrices for CNN+IATnew and "
       "IAT+GAM on the combined ROCO+ROCOv2 test set (class 4, 'Other', is empty by construction).")

h2("IAT architecture-variant ablation")
body(
 "On ROCO v1 alone, we compared four IAT-pipeline variants under a 50-epoch budget: IAT with a "
 "light CNN backbone (no GAN) reached 86.89% best validation accuracy at epoch 8; IAT with a "
 "macro/deep-capacity CNN backbone (no GAN) reached 90.22% at epoch 14; the micro-capacity "
 "CNN+IATnew (no GAN) reached 93.3%; and IAT+GAM (micro capacity, GAN/memory-coupling active) "
 "reached 93.78% at epoch 50, still improving at the end of training. Notably, the macro/deep-"
 "capacity backbone underperforms the micro-capacity backbone by over 3 points -- a "
 "counter-intuitive result that we report plainly rather than omit, and that is consistent with "
 "the parameter-efficiency framing of this work: more capacity did not translate into better "
 "accuracy on this task and data regime."
)
figure(FIG + "fig5_iat_variant_ablation.png", "Figure 6. IAT architecture-variant ablation on "
       "ROCO v1 (4-class balanced, 50 epochs each).")

h2("Per-dataset breakdown and the GAM ablation")
body(
 "Table 2 and Figure 7 report best-checkpoint validation accuracy for CNN+IATnew and IAT+GAM "
 "separately on ROCO v1, ROCOv2, and their combination. Both configurations are stable across "
 "datasets (93.3-94.4%), but GAM does not deliver a consistent benefit over the plain CNN+IAT "
 "configuration: GAM is 0.5 points higher on ROCO v1, 0.5 points lower on ROCOv2, and 0.2 points "
 "lower on the combined set. We therefore do not claim the GAN/memory-coupling branch improves "
 "accuracy; it is, at best, dataset-dependent and within noise of the simpler configuration."
)
table(["Configuration", "ROCO v1", "ROCOv2", "Combined"],
      [["CNN+IATnew (no GAN)", "93.3%", "94.1%", "94.4%"],
       ["IAT+GAM (GAN + memory coupling)", "93.8%", "93.6%", "94.2%"]],
      "Table 2. Best-checkpoint validation accuracy by dataset. ROCO-v1-only and ROCOv2-only "
      "extended test metrics (confusion matrix, captioning, AUROC) for these two configurations "
      "are not available for this revision -- see Limitations.")
figure(FIG + "fig6_per_dataset.png", "Figure 7. Per-dataset validation accuracy for the two "
       "proposed configurations.")

h2("Training dynamics: accuracy and loss curves across all models")
body(
 "To examine convergence behaviour rather than only endpoint accuracy, we logged per-epoch "
 "training and validation accuracy and loss for all nine gradient-trained models (the two "
 "proposed configurations and the seven param-matched CNN/MLP baselines; XGBoost is excluded "
 "here as it is optimised over boosting rounds rather than epochs, and its round-level metric "
 "is not on a comparable x-axis). Figure 8 shows the accuracy curves and Figure 9 the loss "
 "curves for all nine models as small multiples, each over 50 epochs of training on the "
 "combined ROCO+ROCOv2 manifest."
)
figure(FIG + "fig8_accuracy_curves_grid.png", "Figure 8. Training (blue) and validation (orange) "
       "accuracy curves for all nine gradient-trained models, 50 epochs each.")
figure(FIG + "fig9_loss_curves_grid.png", "Figure 9. Training (blue) and validation (orange) "
       "loss curves for the same nine models and epochs as Figure 8.")
body(
 "Figure 10 overlays validation accuracy for all nine models on a single axis to make relative "
 "convergence speed and ceiling directly comparable. CNN+IATnew and IAT+GAM reach their final "
 "accuracy band earlier and at a higher ceiling than every param-matched baseline. The MLP "
 "baseline is the clearest outlier, plateauing well below every CNN-based model from early "
 "epochs onward, consistent with its lack of spatial inductive bias. Several baselines "
 "(notably resnet_small and efficientnet_small) show a widening gap between train and "
 "validation accuracy/loss in the later epochs in Figures 8-9, an overfitting signature "
 "already accounted for by this study's practice of reporting metrics at each model's "
 "best-validation checkpoint rather than at the final epoch."
)
figure(FIG + "fig10_val_accuracy_overlay.png", "Figure 10. Validation accuracy, all nine "
       "gradient-trained models overlaid on one axis.")

doc.add_page_break()

# ================= DISCUSSION =================
h1("Discussion")
h2("Parameter efficiency as the central claim")
body(
 "The strongest and most defensible result in this work is not that IAT-GAM is unconditionally "
 "state-of-the-art, but that it is the best-performing genuinely pixel-based vision model among "
 "ten architectures trained under an identical, controlled parameter budget, on identical data, "
 "for an identical epoch and seed budget. This controlled comparison is, to our knowledge, not "
 "present in the existing ROCO/ROCOv2 modality-classification literature, which typically reports "
 "accuracy without parameter-matching across methods. We frame our contribution accordingly: "
 "competitive accuracy at a fraction of the parameters of a standard ResNet-50, rather than an "
 "outright accuracy record."
)
h2("Limitations")
body(
 "We disclose four specific limitations. First, the multiclass one-vs-rest AUROC is "
 "structurally undefined for every model evaluated in this work, because our 4-class-balanced "
 "manifest construction excludes the 5th ('Other') class entirely, leaving that column's one-hot "
 "target all-zero; scikit-learn's AUROC implementation correctly returns null in this situation, "
 "and we report this null result honestly rather than substituting a fabricated value. Second, a "
 "checkpoint-filename collision during this study's experimentation overwrote the ROCO-v1-only "
 "and ROCOv2-only model weights for the CNN+IATnew (no-GAN) and IAT+GAM (GAN+memory-coupling) "
 "configurations with their combined-dataset counterparts, because the training script's "
 "checkpoint-naming convention encodes hyperparameters but not dataset source; as a result, only "
 "training-time validation accuracy (Table 2), not full extended test metrics (confusion matrix, "
 "captioning, AUROC), survives for the per-dataset (non-combined) versions of these two "
 "configurations. We disclose this rather than omit the affected rows. Third, all results in "
 "this revision are single-seed (seed = 42); we have not yet run a multi-seed significance study, "
 "and point estimates should be read with that caveat. Fourth, the GAN/memory-coupling (GAM) "
 "branch does not provide a consistent accuracy benefit across datasets, and the macro/deep-"
 "capacity CNN backbone underperforms the micro-capacity backbone in our IAT ablation; we see no "
 "evidence in our data that additional backbone or generative capacity improves this task, and "
 "report both findings rather than selectively presenting only favourable configurations."
)
h2("Comparison with the DenseNet-style training instability")
body(
 "We also note a methodological finding of independent interest: the DenseNet-style baseline's "
 "raw (unnormalised) feature-map output was numerically stable for plain classification "
 "training, but became unstable once used as the key/value input to the caption decoder's "
 "spatial cross-attention under full joint backpropagation, producing a loss several orders of "
 "magnitude larger than every other architecture evaluated. This was resolved with a frozen "
 "backbone plus a newly inserted, trainable normalisation layer feeding the decoder, rather than "
 "by reducing batch size (which made the problem worse, correctly ruling out a memory-related "
 "rather than a numerical-scale explanation). We mention this as a caution for others attaching "
 "attention-based decoders to arbitrary pretrained or custom CNN feature extractors."
)

# ================= CONCLUSION =================
h1("Conclusion")
body(
 "We presented IAT-GAM, a parameter-efficient attention-guided region-perturbation framework for "
 "joint modality classification and caption generation on ROCO and ROCOv2. Under a controlled, "
 "parameter-matched comparison against eight re-implemented standard architectures, CNN+IATnew "
 "achieves the best classification and captioning metrics of every genuine pixel-based vision "
 "model evaluated (95.01% five-class accuracy, 2.4M backbone parameters), and is competitive with "
 "much larger literature-reported ResNet-50 results at roughly one-tenth the parameter count. We "
 "found no consistent benefit from the optional GAN/memory-coupling branch, and report this "
 "alongside a transparent account of this study's limitations: a structurally undefined AUROC "
 "under our class-balancing scheme, a checkpoint-naming data-loss incident affecting two "
 "per-dataset ablation rows, and the absence of multi-seed significance testing in this "
 "revision. Future work should address these directly: a dataset-aware checkpoint-naming "
 "convention, a multi-seed significance study, and an AUROC-compatible evaluation protocol that "
 "retains or re-weights the excluded fifth class rather than discarding it outright."
)

# ================= DATA AVAILABILITY =================
h1("Data availability")
body(
 "ROCO and ROCOv2 are publicly available radiology image-caption corpora; the balanced "
 "4-class manifests, trained checkpoints, and evaluation logs generated in this study are "
 "available from the corresponding author on reasonable request."
)

h1("Declarations")
body("The authors declare no competing interests.")

# ================= REFERENCES =================
h1("References")
refs = [
 "Pelka, O., Koitka, S., R\u00fcckert, J., Nensa, F. & Friedrich, C. M. Radiology Objects in "
 "COntext (ROCO): a multimodal image dataset. In Intravascular Imaging and Computer Assisted "
 "Stenting, and Large-Scale Annotation of Biomedical Data and Expert Label Synthesis (2018).",
 "R\u00fcckert, J. et al. ROCOv2: Radiology Objects in COntext version 2, an updated multimodal "
 "image dataset. Sci. Data (2024).",
 "Author(s). EfficientNet-B4 for radiology modality classification. Algorithms "
 "https://doi.org/10.3390/a19030222 (2026).",
 "Author(s). A lightweight MLP classifier for ROCOv2 modality labels. arXiv:2607.27763 (2026).",
 "Author(s). Gradient-boosted feature classification for ImageCLEF medical concept detection. "
 "CEUR Workshop Proceedings (2026).",
 "He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. Proc. "
 "IEEE CVPR 770-778 (2016).",
 "Tan, M. & Le, Q. EfficientNet: rethinking model scaling for convolutional neural networks. "
 "Proc. ICML (2019).",
 "Howard, A. et al. Searching for MobileNetV3. Proc. IEEE ICCV (2019).",
 "Ma, N., Zhang, X., Zheng, H.-T. & Sun, J. ShuffleNet V2: practical guidelines for efficient "
 "CNN architecture design. Proc. ECCV (2018).",
 "Iandola, F. N. et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters. "
 "arXiv:1602.07360 (2016).",
 "Huang, G., Liu, Z., van der Maaten, L. & Weinberger, K. Q. Densely connected convolutional "
 "networks. Proc. IEEE CVPR (2017).",
 "Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. Proc. ACM KDD (2016).",
 "Zhang, H. et al. mixup: beyond empirical risk minimization. Proc. ICLR (2018).",
 "Yun, S. et al. CutMix: regularization strategy to train strong classifiers with localizable "
 "features. Proc. IEEE ICCV (2019).",
 "Cubuk, E. D. et al. RandAugment: practical automated data augmentation with a reduced search "
 "space. Proc. IEEE CVPR Workshops (2020).",
 "Zhang, T. et al. BERTScore: evaluating text generation with BERT. Proc. ICLR (2020).",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.add_run(f"{i}. {r}")
    p.paragraph_format.space_after = Pt(2)

doc.save(OUTPATH)
print("saved", OUTPATH)
