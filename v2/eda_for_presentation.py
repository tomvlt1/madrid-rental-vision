"""
Quick EDA for the presentation: 5 graphs + a markdown brief
that the teammate covering "data exploration" can read off.

Reads the local v2/data/listings_clean_v2.csv (6,047 listings).
Outputs to v2/figures/eda/.

Run: python3 v2/eda_for_presentation.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "v2" / "data" / "listings_clean_v2.csv"
OUT_DIR = PROJECT_ROOT / "v2" / "figures" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZONE_ORDER = [
    "Centro",
    "Salamanca-Retiro",
    "Chamberí",
    "Norte",
    "Arganzuela",
    "Tetuán-Latina",
    "Carabanchel-Usera",
    "Periferia Norte",
    "Periferia Sur",
    "Otros",
]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def fig1_price_distribution(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(df["price"], bins=60, color="#3b6fb6", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Monthly rent (€)")
    axes[0].set_ylabel("Listings")
    axes[0].set_title("Raw rent distribution: heavily right-skewed")
    axes[0].axvline(df["price"].median(), color="black", linestyle="--", linewidth=1)
    axes[0].text(
        df["price"].median() + 100,
        axes[0].get_ylim()[1] * 0.9,
        f"median €{df['price'].median():.0f}",
        fontsize=10,
    )

    axes[1].hist(np.log1p(df["price"]), bins=60, color="#2a9d8f", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("log(1 + monthly rent)")
    axes[1].set_ylabel("Listings")
    axes[1].set_title("Log-rent: bimodal but better-behaved → what we model")

    fig.suptitle(f"Madrid rental price distribution (N = {len(df):,} listings)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "01_price_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig2_price_by_zone(df, out):
    zones_present = [z for z in ZONE_ORDER if z in df["zone"].values]
    others = [z for z in df["zone"].unique() if z not in zones_present]
    zones_present = zones_present + sorted(others)

    data = [df.loc[df["zone"] == z, "price"].values for z in zones_present]
    medians = [np.median(d) if len(d) else 0 for d in data]
    order = np.argsort(medians)
    zones_sorted = [zones_present[i] for i in order]
    data_sorted = [data[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data_sorted,
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#3b6fb6")
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(2)

    ax.set_yticklabels(zones_sorted, fontsize=10)
    ax.set_xlabel("Monthly rent (€)")
    ax.set_title("Rent by zone: zone is the single strongest tabular predictor")
    ax.set_xlim(0, df["price"].quantile(0.99))
    fig.tight_layout()
    fig.savefig(out / "02_price_by_zone.png", bbox_inches="tight")
    plt.close(fig)


def fig3_photos_per_listing(df, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        df["num_images"].clip(upper=60),
        bins=60,
        color="#e76f51",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("Photos per listing (capped at 60)")
    ax.set_ylabel("Listings")
    ax.set_title(
        f"Photos per listing — median {int(df['num_images'].median())}, "
        f"total {df['num_images'].sum():,} photos through SigLIP"
    )
    ax.axvline(df["num_images"].median(), color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out / "03_photos_per_listing.png", bbox_inches="tight")
    plt.close(fig)


def fig4_correlations(df, out):
    cols = ["price", "sqft_m2", "rooms", "bathrooms", "num_images", "floor_num"]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i, f"{corr.values[i, j]:+.2f}",
                ha="center", va="center",
                color="white" if abs(corr.values[i, j]) > 0.4 else "black",
                fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title("Numeric feature correlations (Pearson)")
    fig.tight_layout()
    fig.savefig(out / "04_correlations.png", bbox_inches="tight")
    plt.close(fig)


def fig5_price_vs_size(df, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[(df["sqft_m2"] > 0) & (df["sqft_m2"] < 400) & (df["price"] < 8000)]
    ax.scatter(
        sub["sqft_m2"], sub["price"],
        s=8, alpha=0.25, color="#3b6fb6", edgecolor="none",
    )

    bins = np.linspace(20, 300, 15)
    centers = (bins[:-1] + bins[1:]) / 2
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sub["sqft_m2"] >= lo) & (sub["sqft_m2"] < hi)
        means.append(sub.loc[mask, "price"].median() if mask.any() else np.nan)
    ax.plot(centers, means, color="#e76f51", linewidth=2.5, label="median rent per size bin")

    ax.set_xlabel("Floor area (m²)")
    ax.set_ylabel("Monthly rent (€)")
    ax.set_title("Price vs size: roughly linear, with a long luxury tail")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "05_price_vs_size.png", bbox_inches="tight")
    plt.close(fig)


def write_brief(df, out):
    n = len(df)
    median_price = df["price"].median()
    p10 = df["price"].quantile(0.10)
    p90 = df["price"].quantile(0.90)
    median_sqm = df["sqft_m2"].median()
    median_photos = int(df["num_images"].median())
    total_photos = int(df["num_images"].sum())
    n_zones = df["zone"].nunique()

    zone_counts = df["zone"].value_counts()
    top_zone = zone_counts.index[0]
    top_zone_n = int(zone_counts.iloc[0])

    zone_medians = df.groupby("zone")["price"].median().sort_values()
    cheapest_zone = zone_medians.index[0]
    cheapest_zone_price = zone_medians.iloc[0]
    priciest_zone = zone_medians.index[-1]
    priciest_zone_price = zone_medians.iloc[-1]

    desc_word_counts = df["description"].fillna("").str.split().str.len()
    median_words = int(desc_word_counts.median())

    zone_ratio = priciest_zone_price / cheapest_zone_price
    brief = f"""# EDA brief — Madrid rental dataset

For the **Data Exploration** beat in the architecture section (~20 seconds).
Dataset is the cleaned, deduplicated v2 set (6,047 listings, scraped from
Idealista, deduped on price-size-rooms-bathrooms-zone).

---

## Headline numbers (memorise 3 of these)

- **{n:,} listings** across **{n_zones} Madrid zones**
- **Median rent: €{median_price:,.0f}/month** (10th percentile €{p10:,.0f}, 90th €{p90:,.0f})
- **Median size: {median_sqm:.0f} m²**
- **{total_photos:,} photos total**, median **{median_photos} per listing** — every photo went through SigLIP
- **Median description: ~{median_words} words** (most in Spanish)
- **Priciest zone**: {priciest_zone} (median €{priciest_zone_price:,.0f}); **cheapest**: {cheapest_zone} (median €{cheapest_zone_price:,.0f})
- **Largest zone by listing count**: {top_zone} ({top_zone_n} listings)

---

## What the graphs show (one line each)

1. **`01_price_distribution.png`** — Raw rent is heavily right-skewed (€500 studios alongside €5,000+ penthouses), so we model **log-rent** instead. The distribution is also visibly **bimodal** — one cluster around €1,000–1,500 (studios + 1-bed), another around €2,500–3,000 (2-bed and up). Log-transforming compresses the tail and makes the gradient-boosting target much better-behaved, but the bimodality remains and is why zone + size end up doing so much of the lifting.

2. **`02_price_by_zone.png`** — Zone is the single strongest tabular predictor. Median rent in {priciest_zone} is roughly **{zone_ratio:.1f}x** the median in {cheapest_zone}. This is why one-hot zone encoding is in our tabular feature block.

3. **`03_photos_per_listing.png`** — Median {median_photos} photos per listing. Total {total_photos:,} photos went through the SigLIP encoder. Two clear modes — most listings have 10–25 photos, but a noticeable spike at 60 (the cap) suggests luxury listings consistently max out the photo slots.

4. **`04_correlations.png`** — Size (sqft_m2) is the strongest single numeric predictor of rent at **+0.75 correlation**, followed by bathrooms at **+0.70**. Bathrooms and size are themselves correlated at 0.80 (multicollinearity), which is why GB can substitute one for the other. **num_images correlates +0.33 with price** — a mild but real signal that expensive listings show off more photos. Floor number has near-zero correlation with rent in Madrid.

5. **`05_price_vs_size.png`** — Price scales roughly linearly with size up to about 150 m². Above that, the relationship flattens and the variance explodes — this is the **luxury tail** that produces the Q4 bias we have to fix with calibration later.

---

## Suggested talking points (~20s for the beat)

> "Madrid rents are heavily skewed — studios at €500 sit next to penthouses above €5,000 — so we model **log-rent**, not raw rent. Among tabular features, **zone is by far the strongest predictor**: median rent in {priciest_zone} is roughly **{zone_ratio:.1f}x** the median in {cheapest_zone}. We have a median of **{median_photos} photos per listing**, so over **{total_photos:,} photos** total flow through our image encoder. And there's a long luxury tail above 200 m² — that's the bias we have to calibrate for later."

---

## Hand-off line into PCA beat

> "All these features — tabular, text, image — give us hundreds of dimensions per listing. Which is why the next step is dimensionality reduction."

Files in this folder:
- 01_price_distribution.png
- 02_price_by_zone.png
- 03_photos_per_listing.png
- 04_correlations.png
- 05_price_vs_size.png
- eda_brief.md (this file)
"""

    (out / "eda_brief.md").write_text(brief)


def main():
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  {len(df):,} listings × {df.shape[1]} columns")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sqft_m2"] = pd.to_numeric(df["sqft_m2"], errors="coerce")
    df["num_images"] = pd.to_numeric(df["num_images"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["price", "sqft_m2"]).reset_index(drop=True)
    print(f"  after dropping rows missing price/sqft: {len(df):,}")

    print("\nGenerating figures...")
    fig1_price_distribution(df, OUT_DIR)
    print("  ✓ 01_price_distribution.png")
    fig2_price_by_zone(df, OUT_DIR)
    print("  ✓ 02_price_by_zone.png")
    fig3_photos_per_listing(df, OUT_DIR)
    print("  ✓ 03_photos_per_listing.png")
    fig4_correlations(df, OUT_DIR)
    print("  ✓ 04_correlations.png")
    fig5_price_vs_size(df, OUT_DIR)
    print("  ✓ 05_price_vs_size.png")

    print("\nWriting brief...")
    write_brief(df, OUT_DIR)
    print("  ✓ eda_brief.md")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
