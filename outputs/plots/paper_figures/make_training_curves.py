import re, ast, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = r"C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT\outputs\logs"
OUT_DIR = r"C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT\outputs\plots\paper_figures"

def parse_cross_dataset(path, section_header):
    """Parse cross_dataset_queue.log style: '[epoch N/50] lr=.. train={...} val={...}' within one === section."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    sections = text.split("=== ")
    target = None
    for s in sections:
        if s.startswith(section_header):
            target = s
            break
    if target is None:
        return None
    rows = []
    for line in target.splitlines():
        m = re.match(r"\[epoch (\d+)/\d+\] lr=([\d.]+) train=(\{.*\}) val=(\{.*\})", line)
        if not m:
            continue
        ep = int(m.group(1))
        train_d = ast.literal_eval(m.group(3))
        val_d = ast.literal_eval(m.group(4))
        rows.append({
            "epoch": ep,
            "train_acc": train_d.get("acc_5class_accuracy"),
            "val_acc": val_d.get("acc_5class_accuracy"),
            "train_loss": train_d.get("total"),
            "val_loss": val_d.get("total"),
        })
    return rows

def parse_baseline(path, arch_prefix):
    """Parse param_matched_baselines_queue.log style: '[arch] epoch N/50  train_loss=.. train_acc=..  val_loss=.. val_acc=..'"""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(f"[{arch_prefix}] epoch"):
                continue
            m = re.search(r"epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)", line)
            if not m:
                continue
            rows.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_acc": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
            })
    return rows

models = {}
models["CNN+IATnew (proposed)"] = parse_cross_dataset(f"{LOG_DIR}/cross_dataset_queue.log", "CNN+IATnew micro (no GAN) - combined")
models["IAT+GAM (proposed)"] = parse_cross_dataset(f"{LOG_DIR}/cross_dataset_queue.log", "IAT+GAM (GAN + memory coupling) - combined")
for arch in ["resnet_small", "efficientnet_small", "mlp", "mobilenet_style", "shufflenet_style", "squeezenet_style", "densenet_style"]:
    models[arch] = parse_baseline(f"{LOG_DIR}/param_matched_baselines_queue.log", arch)

for k, v in list(models.items()):
    if not v:
        print(f"WARNING: no data parsed for {k}")
    else:
        print(f"{k}: {len(v)} epochs parsed")

json.dump({k: v for k, v in models.items() if v}, open(f"{OUT_DIR}/_training_curves_data.json", "w"))

# ---- Figure A: 3x3 grid of accuracy curves ----
names = [k for k in models if models[k]]
ncols, nrows = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 11), dpi=300)
axes = axes.flatten()
for i, name in enumerate(names):
    ax = axes[i]
    rows = models[name]
    ep = [r["epoch"] for r in rows]
    ax.plot(ep, [r["train_acc"] for r in rows], label="train acc", color="#2b6cb0", lw=1.3)
    ax.plot(ep, [r["val_acc"] for r in rows], label="val acc", color="#dd6b20", lw=1.3)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    if i == 0:
        ax.legend(fontsize=7, loc="lower right")
for j in range(len(names), nrows * ncols):
    axes[j].axis("off")
fig.suptitle("Training and validation accuracy curves across all models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT_DIR}/fig8_accuracy_curves_grid.png", dpi=300)
plt.close(fig)

# ---- Figure B: 3x3 grid of loss curves ----
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 11), dpi=300)
axes = axes.flatten()
for i, name in enumerate(names):
    ax = axes[i]
    rows = models[name]
    ep = [r["epoch"] for r in rows]
    ax.plot(ep, [r["train_loss"] for r in rows], label="train loss", color="#2b6cb0", lw=1.3)
    ax.plot(ep, [r["val_loss"] for r in rows], label="val loss", color="#dd6b20", lw=1.3)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_ylabel("loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")
for j in range(len(names), nrows * ncols):
    axes[j].axis("off")
fig.suptitle("Training and validation loss curves across all models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT_DIR}/fig9_loss_curves_grid.png", dpi=300)
plt.close(fig)

# ---- Figure C: single overlay of validation accuracy, all models ----
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
palette = ["#2b6cb0", "#dd6b20", "#38a169", "#d53f8c", "#805ad5", "#d69e2e", "#319795", "#e53e3e", "#4a5568"]
for i, name in enumerate(names):
    rows = models[name]
    ep = [r["epoch"] for r in rows]
    ax.plot(ep, [r["val_acc"] for r in rows], label=name, color=palette[i % len(palette)], lw=1.6)
ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy (5-class)")
ax.set_title("Validation accuracy comparison across all models")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig10_val_accuracy_overlay.png", dpi=300)
plt.close(fig)

print("DONE")
print("names:", names)
