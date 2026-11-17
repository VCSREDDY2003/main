"""
Task (1) only: Image-labelling classification with a genuinely tiny
(~20k-parameter) CNN, reading images + label_5class straight from the
feature CSVs built by build_feature_csv.py.

Reuses the existing image-loading convention (image_path relative to the
dataset's own folder) instead of rewriting a new loader.

Usage:
    python feature_pipeline/tiny_image_classifier.py --source roco --epochs 5
    python feature_pipeline/tiny_image_classifier.py --source combined --epochs 5
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGE_SIZE = 64


class TinyCNN(nn.Module):
    """~22.6k params (target ~20k): 3 conv layers + 1 linear head."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1)   # 64->32
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1)  # 32->16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 16->8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ImageLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str):
        self.df = df.reset_index(drop=True)
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, row["image_path"])
        try:
            img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(
                IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1) / 255.0
        except Exception:
            x = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        y = int(row["label_5class"])
        return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["roco", "rocov2", "combined"], default="roco")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance", action="store_true", default=False,
                    help="undersample every class to match the smallest class count")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    df = pd.read_csv(os.path.join(FEATURE_DIR, f"{args.source}_features.csv"),
                      keep_default_na=False)
    df = df[df["has_image"] == 1]
    # Exclude class 4 ("Other"): it's a keyword-miss catch-all, not a real
    # modality, and it's the single largest class (31.1% of ROCO) -- leaving
    # it in lets the model "win" by just learning to detect absent keywords
    # rather than real modality signal.
    df = df[df["label_5class"].astype(int) != 4]

    if args.balance:
        counts = df["label_5class"].astype(int).value_counts()
        min_count = counts.min()
        df = (df.groupby("label_5class", group_keys=False)
                .apply(lambda g: g.sample(n=min_count, random_state=args.seed)))
        print(f"Balanced to {min_count} samples/class -> {len(df)} total rows")
        print(df["label_5class"].value_counts().sort_index())

    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)

    if args.source == "roco":
        root = config.ROCO_ROOT
    elif args.source == "rocov2":
        root = config.ROCOV2_ROOT
    else:
        # combined: image_path is relative to each row's own source root;
        # resolve per-row instead of a single root.
        root = None

    n = len(df)
    n_train = int(n * 0.8)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_df, val_df = df.iloc[:n_train], df.iloc[n_train:]

    def make_ds(sub_df):
        if root is not None:
            return ImageLabelDataset(sub_df, root)
        # combined case: build per-row absolute paths up front
        sub_df = sub_df.copy()
        roots = {"roco": config.ROCO_ROOT, "rocov2": config.ROCOV2_ROOT}
        sub_df["image_path"] = sub_df.apply(
            lambda r: os.path.join(roots[r["source_dataset"]], r["image_path"]), axis=1)
        return ImageLabelDataset(sub_df, "")

    train_ds, val_ds = make_ds(train_df), make_ds(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyCNN(num_classes=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_n += x.size(0)

        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_n += x.size(0)

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss/train_n:.4f} "
              f"train_acc={train_correct/train_n:.4f}  val_acc={val_correct/val_n:.4f}")


if __name__ == "__main__":
    main()
