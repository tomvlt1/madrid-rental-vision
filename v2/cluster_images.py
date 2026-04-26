"""
Cluster the SigLIP per-photo embeddings to visually demonstrate the
representation captures semantic structure: room types, price tiers,
listing quality.

Mirrors notebooks/02_image_clusters.py from v1 (which used ResNet on
5,000 sampled images), but reads from the per-photo SigLIP embeddings
we already extracted in v2/data/siglip_per_photo.npy. No new model
forward passes needed.

Output:
  v2/figures/siglip_umap_clusters.png    UMAP 2D, colored by k-means cluster
  v2/figures/siglip_umap_by_price.png    UMAP 2D, colored by listing log-price
  v2/figures/siglip_cluster_<k>.png      12-image grid for each cluster
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import umap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import V2_DATA_DIR, V2_DIR  # noqa: E402

AI2_IMAGES = Path("/Users/tom/School/AI/ai2/data/raw/images")
FIG_DIR = V2_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PER_PHOTO_NPY = V2_DATA_DIR / "siglip_per_photo.npy"
PER_PHOTO_IDX = V2_DATA_DIR / "siglip_per_photo_index.csv"
CLEAN_CSV = V2_DATA_DIR / "listings_clean_v2.csv"

N_SAMPLE = 5000
N_CLUSTERS = 10
SEED = 42


def main():
    print(f"Loading per-photo embeddings from {PER_PHOTO_NPY}")
    embs_all = np.load(PER_PHOTO_NPY)
    idx_all = pd.read_csv(PER_PHOTO_IDX)
    print(f"  total photos: {len(idx_all)}, dim={embs_all.shape[1]}")

    print(f"Loading listings_clean_v2.csv for price + zone")
    listings = pd.read_csv(CLEAN_CSV)
    listings["listing_id"] = (
        listings["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    )
    listings = listings.set_index("listing_id")[["price", "zone"]]

    # Join price + zone onto each photo row
    idx_all["listing_id"] = idx_all["listing_id"].astype(str)
    df_photos = idx_all.join(listings, on="listing_id", how="left")
    df_photos = df_photos.dropna(subset=["price"]).reset_index(drop=True)
    print(f"  photos with price: {len(df_photos)}")

    # Sample uniformly (deterministic with SEED)
    rng = np.random.RandomState(SEED)
    if len(df_photos) > N_SAMPLE:
        sample_idx = rng.choice(len(df_photos), N_SAMPLE, replace=False)
        sample_idx.sort()
        df_sample = df_photos.iloc[sample_idx].reset_index(drop=True)
        embs_sample = embs_all[sample_idx]
    else:
        df_sample = df_photos
        embs_sample = embs_all[df_photos.index.values]
    print(f"  sampled: {len(df_sample)} photos")

    print(f"\nRunning UMAP (n_neighbors=30, min_dist=0.3)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=SEED)
    coords_2d = reducer.fit_transform(embs_sample)
    print(f"  UMAP done. coords: {coords_2d.shape}")

    print(f"\nKMeans with k={N_CLUSTERS}...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(embs_sample)

    # ----- Figure 1: UMAP colored by cluster -----
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab10", s=4, alpha=0.6)
    ax.set_title(
        f"SigLIP per-photo embeddings (n={len(df_sample)})\n"
        f"UMAP 2D projection, colored by K-means cluster (k={N_CLUSTERS})",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    for k in range(N_CLUSTERS):
        m = labels == k
        cx, cy = coords_2d[m, 0].mean(), coords_2d[m, 1].mean()
        ax.annotate(
            f"C{k}", (cx, cy), fontsize=11, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
    plt.tight_layout()
    out = FIG_DIR / "siglip_umap_clusters.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    # ----- Figure 2: UMAP colored by log price -----
    prices = df_sample["price"].values
    fig, ax = plt.subplots(figsize=(13, 9))
    sc = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=np.log1p(prices), cmap="RdYlGn_r", s=4, alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, label="log(rent)")
    ax.set_title(
        "SigLIP per-photo embeddings, colored by listing log-rent\n"
        "(green = cheap, red = expensive)",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    out = FIG_DIR / "siglip_umap_by_price.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    # ----- Figure 3+: 12-image grid per cluster -----
    print("\nBuilding cluster example grids...")
    for k in range(N_CLUSTERS):
        m = np.where(labels == k)[0]
        chosen = m if len(m) <= 12 else rng.choice(m, 12, replace=False)
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        axes = axes.flatten()
        cl_prices = [df_sample.iloc[i]["price"] for i in m]
        avg_price = float(np.mean(cl_prices))
        median_price = float(np.median(cl_prices))
        fig.suptitle(
            f"SigLIP cluster {k} — n={len(m)} photos | "
            f"mean rent €{avg_price:.0f} | median €{median_price:.0f}",
            fontsize=12, fontweight="bold",
        )
        for j, ax in enumerate(axes):
            if j < len(chosen):
                idx = chosen[j]
                row = df_sample.iloc[idx]
                src = AI2_IMAGES / row["source_path"].replace("data/raw/images/", "")
                # source_path is stored as the relative path under IMAGES_DIR;
                # might already be relative if we set IMAGES_DIR from ai2.
                if not src.exists():
                    # try resolving as absolute or as relative to AI2_IMAGES
                    candidates = [
                        Path(row["source_path"]),
                        AI2_IMAGES / row["source_path"],
                        AI2_IMAGES / row["listing_id"] / Path(row["source_path"]).name,
                    ]
                    for c in candidates:
                        if c.exists():
                            src = c
                            break
                if src.exists():
                    try:
                        ax.imshow(Image.open(src).convert("RGB"))
                    except Exception:
                        pass
            ax.axis("off")
        plt.tight_layout()
        out = FIG_DIR / f"siglip_cluster_{k:02d}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)

    # ----- Cluster summary -----
    print(f"\n{'=' * 70}")
    print("CLUSTER SUMMARY")
    print('=' * 70)
    print(f"{'Cluster':<8} {'N photos':>10} {'Mean rent':>12} {'Median rent':>14} {'Top zone':>22}")
    print("-" * 70)
    for k in range(N_CLUSTERS):
        m = labels == k
        cl_prices = df_sample.loc[m, "price"]
        cl_zones = df_sample.loc[m, "zone"]
        top_zone = cl_zones.value_counts().idxmax() if len(cl_zones) > 0 else "?"
        print(
            f"{k:<8} {int(m.sum()):>10d} €{cl_prices.mean():>10.0f} "
            f"€{cl_prices.median():>12.0f} {top_zone:>22}"
        )

    print(f"\nFigures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
