# Fine-tune ResNet-50 on rental price prediction.
# Only layer4 + regression head are unfrozen. Each image gets its listing's log(price).

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.config import IMAGES_DIR, PROCESSED_DIR, PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
MAX_IMAGES_PER_LISTING = 10  # cap so listings w/ tons of photos don't dominate


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ListingImageDataset(Dataset):
    """Individual images labeled with their listing's log(price)."""

    def __init__(self, image_paths, targets, transform):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, target


class PriceResNet(nn.Module):
    """ResNet-50 + regression head for price."""

    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def extract_embedding(self, x):
        """Forward through backbone up to avgpool -> 2048-dim vector."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def build_dataset():
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    df["log_price"] = np.log1p(df["price"])

    # Load shared split manifest so ResNet fine-tuning NEVER sees a listing
    # that ends up in the downstream GB val/test set. Previously finetune.py
    # used its own train_test_split on a differently-ordered DataFrame and
    # caused cross-split leakage.
    with open(PROCESSED_DIR / "splits.json") as f:
        manifest = json.load(f)
    assignment = manifest["listings"]
    df["split"] = df["listing_id"].map(assignment)

    # Only listings tagged "train" in the shared manifest feed the fine-tune.
    # Val here is used for best-checkpoint selection — treated as held-out
    # model-selection data inside the fine-tune, NOT as the GB pipeline's
    # test set. Listings with split=="test" are untouched by fine-tuning.
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_n = (df["split"] == "test").sum()

    def collect_images(subset_df):
        # Reset the RNG for each subset so photo selection is deterministic
        # and independent of call order. Previous version consumed the
        # seeded RNG stream on the first call, making the second
        # non-deterministic across re-runs.
        rng = random.Random(42)
        paths, targets = [], []
        for _, row in subset_df.iterrows():
            listing_dir = IMAGES_DIR / str(row["listing_id"])
            if not listing_dir.exists():
                continue
            imgs = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))
            if len(imgs) > MAX_IMAGES_PER_LISTING:
                imgs = rng.sample(imgs, MAX_IMAGES_PER_LISTING)
            for img_path in imgs:
                paths.append(img_path)
                targets.append(row["log_price"])
        return paths, targets

    train_paths, train_targets = collect_images(train_df)
    val_paths, val_targets = collect_images(val_df)

    print(
        f"Train: {len(train_paths)} images from {len(train_df)} listings"
    )
    print(f"Val:   {len(val_paths)} images from {len(val_df)} listings")
    print(f"Test:  {test_n} listings held out (never seen by ResNet)")

    return train_paths, train_targets, val_paths, val_targets


def train():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    train_paths, train_targets, val_paths, val_targets = build_dataset()

    train_ds = ListingImageDataset(train_paths, train_targets, TRAIN_TRANSFORM)
    val_ds = ListingImageDataset(val_paths, val_targets, VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = PriceResNet().to(device)

    # trainable params count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total "
          f"({trainable/total:.1%} unfrozen)\n")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.train(False)
        val_losses = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                val_losses.append(criterion(preds, targets).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d}/{EPOCHS}: train={train_loss:.4f}  "
              f"val={val_loss:.4f}  lr={lr_now:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, MODELS_DIR / "resnet_finetuned.pt")
    with open(MODELS_DIR / "finetune_history.json", "w") as f:
        json.dump(history, f)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODELS_DIR / 'resnet_finetuned.pt'}")

    return model


if __name__ == "__main__":
    train()
