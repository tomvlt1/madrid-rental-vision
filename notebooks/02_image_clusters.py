# Cluster individual image embeddings and visualize with UMAP

import sys
sys.path.insert(0, ".")

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import models, transforms
from tqdm import tqdm
import umap

from src.config import IMAGES_DIR, PROCESSED_DIR

FIGURES_DIR = Path("notebooks/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE_IMAGES = 5000
N_CLUSTERS = 10
BATCH_SIZE = 64

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

    print("Collecting image paths...")
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    all_images = []
    for _, row in df.iterrows():
        listing_id = row["url"].rstrip("/").split("/")[-1]
        listing_dir = IMAGES_DIR / listing_id
        if listing_dir.exists():
            for img_path in sorted(listing_dir.glob("*.jpg")):
                all_images.append((img_path, listing_id, row["price"], row["zone"]))
            for img_path in sorted(listing_dir.glob("*.webp")):
                all_images.append((img_path, listing_id, row["price"], row["zone"]))

    print(f"Total images found: {len(all_images)}")

    random.seed(42)
    if len(all_images) > N_SAMPLE_IMAGES:
        sample = random.sample(all_images, N_SAMPLE_IMAGES)
    else:
        sample = all_images
    print(f"Sampled: {len(sample)} images")

    print("Loading model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    print("Extracting individual image embeddings...")
    embeddings = []
    valid_sample = []
    tensors_batch = []
    indices_batch = []

    for i, (img_path, lid, price, zone) in enumerate(tqdm(sample, desc="Loading")):
        try:
            img = Image.open(img_path).convert("RGB")
            t = TRANSFORM(img)
            tensors_batch.append(t)
            indices_batch.append(i)
        except Exception:
            continue

        if len(tensors_batch) == BATCH_SIZE or i == len(sample) - 1:
            if tensors_batch:
                batch = torch.stack(tensors_batch).to(device)
                with torch.no_grad():
                    embs = model(batch).cpu().numpy()
                embeddings.append(embs)
                for idx in indices_batch:
                    valid_sample.append(sample[idx])
                tensors_batch = []
                indices_batch = []

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)
    print("UMAP done.")

    print(f"KMeans with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
               c=labels, cmap="tab10", s=3, alpha=0.5)
    ax.set_title("Image Embeddings - UMAP Projection with K-Means Clusters", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    for k in range(N_CLUSTERS):
        mask = labels == k
        cx, cy = coords_2d[mask, 0].mean(), coords_2d[mask, 1].mean()
        ax.annotate(f"Cluster {k}", (cx, cy), fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_umap_image_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("UMAP plot saved.")

    prices = np.array([s[2] for s in valid_sample])
    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                    c=np.log1p(prices), cmap="RdYlGn_r", s=3, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Log(Price)")
    ax.set_title("Image Embeddings - Colored by Listing Price", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_umap_by_price.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("UMAP by price saved.")

    print("\nCluster example grids...")
    for k in range(N_CLUSTERS):
        mask = np.where(labels == k)[0]
        if len(mask) > 12:
            chosen = random.sample(list(mask), 12)
        else:
            chosen = list(mask)

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        axes = axes.flatten()

        cluster_prices = [valid_sample[i][2] for i in mask]
        avg_price = np.mean(cluster_prices)

        fig.suptitle(
            f"Cluster {k} - {len(mask)} images, Avg listing price: EUR {avg_price:.0f}",
            fontsize=13, fontweight="bold")

        for j, ax in enumerate(axes):
            if j < len(chosen):
                idx = chosen[j]
                img_path = valid_sample[idx][0]
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    pass
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"11_cluster_{k:02d}_examples.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"Cluster grids saved.")

    print(f"\n{'='*60}")
    print("CLUSTER SUMMARY")
    print(f"{'='*60}")
    zones = [s[3] for s in valid_sample]
    for k in range(N_CLUSTERS):
        mask = labels == k
        cluster_prices = [valid_sample[i][2] for i in np.where(mask)[0]]
        cluster_zones = [zones[i] for i in np.where(mask)[0]]
        top_zone = pd.Series(cluster_zones).value_counts().index[0]
        print(f"Cluster {k:2d}: {mask.sum():5d} imgs | "
              f"Avg price EUR {np.mean(cluster_prices):7.0f} | "
              f"Top zone: {top_zone}")


if __name__ == "__main__":
    main()
