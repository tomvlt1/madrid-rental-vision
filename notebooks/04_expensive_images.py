# Score images by predicted price using fine-tuned ResNet,
# visualize the most "expensive" and "cheap" looking ones

import sys
sys.path.insert(0, ".")

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import IMAGES_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.vision.finetune import PriceResNet

FIGURES_DIR = Path("notebooks/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    model = PriceResNet()
    state = torch.load(MODELS_DIR / "resnet_finetuned.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.train(False)

    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    all_images = []
    for _, row in df.iterrows():
        listing_id = row["url"].rstrip("/").split("/")[-1]
        listing_dir = IMAGES_DIR / str(listing_id)
        if listing_dir.exists():
            for img_path in sorted(listing_dir.glob("*.jpg"))[:5]:  # Max 5 per listing
                all_images.append((img_path, row["price"], row["zone"], row["sqft_m2"]))

    random.seed(42)
    sample = random.sample(all_images, min(5000, len(all_images)))
    print(f"Scoring {len(sample)} images...")

    scores = []
    valid_sample = []

    batch_tensors = []
    batch_indices = []

    for i, (img_path, price, zone, sqft) in enumerate(tqdm(sample, desc="Scoring")):
        try:
            img = Image.open(img_path).convert("RGB")
            t = TRANSFORM(img)
            batch_tensors.append(t)
            batch_indices.append(i)
        except Exception:
            continue

        if len(batch_tensors) == 64 or i == len(sample) - 1:
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    preds = model(batch).cpu().numpy()
                for j, idx in enumerate(batch_indices):
                    scores.append(preds[j])
                    valid_sample.append(sample[idx])
                batch_tensors = []
                batch_indices = []

    scores = np.array(scores)
    predicted_prices = np.expm1(scores)

    print(f"Scored {len(scores)} images")
    print(f"Predicted price range: EUR {predicted_prices.min():.0f} - EUR {predicted_prices.max():.0f}")

    # most "expensive looking"
    top_idx = np.argsort(scores)[-24:][::-1]
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()
    fig.suptitle("Images the Model Considers Most \"Expensive\"",
                 fontsize=16, fontweight="bold")

    for j, ax in enumerate(axes):
        idx = top_idx[j]
        img_path, actual_price, zone, sqft = valid_sample[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            pass
        ax.set_title(f"Pred: EUR {predicted_prices[idx]:.0f}\n"
                     f"Actual: EUR {actual_price:.0f} | {zone}",
                     fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_most_expensive_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Most expensive images saved")

    # most "cheap looking"
    bottom_idx = np.argsort(scores)[:24]
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()
    fig.suptitle("Images the Model Considers Most \"Budget\"",
                 fontsize=16, fontweight="bold")

    for j, ax in enumerate(axes):
        idx = bottom_idx[j]
        img_path, actual_price, zone, sqft = valid_sample[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            pass
        ax.set_title(f"Pred: EUR {predicted_prices[idx]:.0f}\n"
                     f"Actual: EUR {actual_price:.0f} | {zone}",
                     fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "19_most_budget_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Most budget images saved")

    # "hidden gems" — cheap listings with expensive-looking photos
    actual_prices = np.array([s[1] for s in valid_sample])
    overperform = predicted_prices - actual_prices  # positive = looks more expensive than it is

    top_over = np.argsort(overperform)[-12:][::-1]
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    axes = axes.flatten()
    fig.suptitle("\"Hidden Gems\" - Look Expensive but Rent is Low",
                 fontsize=16, fontweight="bold")

    for j, ax in enumerate(axes):
        idx = top_over[j]
        img_path, actual_price, zone, sqft = valid_sample[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            pass
        ax.set_title(f"Pred: EUR {predicted_prices[idx]:.0f} | "
                     f"Actual: EUR {actual_price:.0f}\n{zone} | {sqft}m2",
                     fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_hidden_gems.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Hidden gems saved")

    # "overpriced" — expensive rent but cheap-looking photos
    top_under = np.argsort(overperform)[:12]
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    axes = axes.flatten()
    fig.suptitle("\"Overpriced?\" - Look Budget but Rent is High",
                 fontsize=16, fontweight="bold")

    for j, ax in enumerate(axes):
        idx = top_under[j]
        img_path, actual_price, zone, sqft = valid_sample[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            pass
        ax.set_title(f"Pred: EUR {predicted_prices[idx]:.0f} | "
                     f"Actual: EUR {actual_price:.0f}\n{zone} | {sqft}m2",
                     fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_overpriced.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Overpriced saved")

    # score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(predicted_prices, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(np.median(predicted_prices), color="red", linestyle="--",
               label=f"Median: EUR {np.median(predicted_prices):.0f}")
    ax.set_xlabel("Model Predicted Price (EUR)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Image-Based Price Predictions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "22_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
