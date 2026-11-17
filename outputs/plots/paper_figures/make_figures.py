import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "figure.dpi": 300,
})
OUT = "C:/Users/kalya/PD/IMAGECAPTION CODE/conference/proposed_model/new_IAT/outputs/plots/paper_figures/"

COLOR_PROPOSED = "#1b5e8a"
COLOR_BASELINE = "#6fa8c7"
COLOR_LIT = "#c0392b"

models = ["CNN+IATnew", "IAT+GAM", "resnet_small", "densenet_style", "efficientnet_small",
          "squeezenet_style", "mobilenet_style", "shufflenet_style", "xgboost", "mlp"]
params = [2.4, 2.4, 2.23, 2.26, 2.36, 2.69, 2.11, 2.27, None, 2.48]
acc = [95.01, 94.90, 93.38, 92.52, 91.23, 90.93, 90.37, 89.75, 95.93, 68.31]
f1 = [94.95, 94.84, 93.35, 92.47, 91.21, 90.86, 90.30, 89.72, 95.94, 68.40]
balacc = [94.95, 94.84, 93.36, 92.47, 91.21, 90.88, 90.31, 88.72, 95.93, 68.26]
is_proposed = [True, True, False, False, False, False, False, False, False, False]

# Fig 1: accuracy vs params scatter (vision models only, exclude xgboost which has no params)
fig, ax = plt.subplots(figsize=(7, 5))
lit_names = ["EfficientNet-B4"]
lit_params = [19.0]
lit_acc = [99.60]
for m, p, a, prop in zip(models, params, acc, is_proposed):
    if p is None:
        continue
    ax.scatter(p, a, s=140 if prop else 90, color=COLOR_PROPOSED if prop else COLOR_BASELINE,
               edgecolor="black", linewidth=0.8, zorder=3, marker="*" if prop else "o")
    ax.annotate(m, (p, a), textcoords="offset points", xytext=(6, 4), fontsize=8)
for n, p, a in zip(lit_names, lit_params, lit_acc):
    if p is None:
        continue
    ax.scatter(p, a, s=90, color=COLOR_LIT, edgecolor="black", linewidth=0.8, marker="^", zorder=3)
    ax.annotate(n, (p, a), textcoords="offset points", xytext=(6, -10), fontsize=8, color=COLOR_LIT)
ax.set_xlabel("Parameters (millions)")
ax.set_ylabel("5-class test accuracy (%)")
ax.set_title("Accuracy vs. parameter count: proposed, param-matched baselines, literature")
ax.set_xscale("log")
ax.grid(alpha=0.25)
handles = [plt.Line2D([0],[0], marker="*", color="w", markerfacecolor=COLOR_PROPOSED, markersize=14, markeredgecolor="black", label="Proposed (IAT-GAM family)"),
           plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=COLOR_BASELINE, markersize=10, markeredgecolor="black", label="Param-matched baseline (this work)"),
           plt.Line2D([0],[0], marker="^", color="w", markerfacecolor=COLOR_LIT, markersize=10, markeredgecolor="black", label="Published literature (not param-matched)")]
ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT + "fig1_acc_vs_params.png", dpi=300)
plt.close(fig)

# Fig 2: grouped bar acc/F1/balacc across 10 models
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
w = 0.26
colors = [COLOR_PROPOSED if p else COLOR_BASELINE for p in is_proposed]
ax.bar(x - w, acc, w, label="Accuracy", color=colors, alpha=0.95)
ax.bar(x, f1, w, label="F1 (macro)", color=colors, alpha=0.65)
ax.bar(x + w, balacc, w, label="Balanced accuracy", color=colors, alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Score (%)")
ax.set_ylim(60, 100)
ax.set_title("5-class classification metrics across all evaluated models")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT + "fig2_classification_bars.png", dpi=300)
plt.close(fig)

# Fig 3: captioning metrics bar (8 models with captions)
cap_models = ["CNN+IATnew", "IAT+GAM", "resnet_small", "densenet_style", "efficientnet_small",
              "squeezenet_style", "mobilenet_style", "shufflenet_style"]
bleu4 = [0.119, 0.106, 0.094, 0.088, 0.089, 0.086, 0.086, 0.087]
meteor = [0.202, 0.213, 0.175, 0.173, 0.168, 0.165, 0.170, 0.168]
cider = [0.369, 0.424, 0.418, 0.237, 0.339, 0.302, 0.244, 0.245]
cap_proposed = [True, True, False, False, False, False, False, False]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cap_models))
w = 0.26
colors = [COLOR_PROPOSED if p else COLOR_BASELINE for p in cap_proposed]
ax.bar(x - w, bleu4, w, label="BLEU-4", color=colors, alpha=0.95)
ax.bar(x, meteor, w, label="METEOR", color=colors, alpha=0.65)
ax.bar(x + w, cider, w, label="CIDEr-D", color=colors, alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels(cap_models, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Captioning metrics across models with a caption-decoder head")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT + "fig3_captioning_bars.png", dpi=300)
plt.close(fig)

# Fig 4: confusion matrix heatmaps for the 2 proposed models
def plot_cm(ax, cm, title):
    cm = np.array(cm)[:4, :4]
    im = ax.imshow(cm, cmap="Blues")
    classes = ["CT", "MRI", "X-ray", "US"]
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)

cm_iatnew = [[855,18,15,5,0],[48,778,42,10,0],[14,43,822,17,0],[15,13,16,871,0],[0,0,0,0,0]]
cm_gam    = [[856,17,18,2,0],[50,750,67,11,0],[21,26,843,6,0],[19,13,18,865,0],[0,0,0,0,0]]
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
plot_cm(axs[0], cm_iatnew, "CNN+IATnew (proposed)")
plot_cm(axs[1], cm_gam, "IAT+GAM (proposed)")
fig.suptitle("Confusion matrices, 5-class modality classification (combined ROCO+ROCOv2 test set)")
fig.tight_layout()
fig.savefig(OUT + "fig4_confusion_matrices.png", dpi=300)
plt.close(fig)

# Fig 5: IAT-variant ablation on ROCO v1
variants = ["IAT + light CNN\n(no GAN)", "IAT + macro/deep CNN\n(no GAN)", "IAT+GAM\n(micro, 50ep)", "CNN+IATnew\n(micro, no GAN)"]
vals = [86.89, 90.22, 93.78, 93.3]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(variants, vals, color=[COLOR_BASELINE, COLOR_BASELINE, COLOR_PROPOSED, COLOR_PROPOSED])
ax.set_ylabel("Best validation accuracy (%)")
ax.set_ylim(80, 98)
ax.set_title("IAT architecture-variant ablation (ROCO v1, 4-class balanced)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT + "fig5_iat_variant_ablation.png", dpi=300)
plt.close(fig)

# Fig 6: per-dataset breakdown
datasets = ["ROCO v1", "ROCOv2", "Combined"]
iatnew_vals = [93.3, 94.1, 94.4]
gam_vals = [93.8, 93.6, 94.2]
x = np.arange(3); w = 0.32
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(x - w/2, iatnew_vals, w, label="CNN+IATnew", color=COLOR_PROPOSED)
ax.bar(x + w/2, gam_vals, w, label="IAT+GAM", color="#4fa3d1")
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("Best validation accuracy (%)")
ax.set_ylim(90, 96)
ax.set_title("Per-dataset breakdown, proposed models")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT + "fig6_per_dataset.png", dpi=300)
plt.close(fig)

print("all figures written")
