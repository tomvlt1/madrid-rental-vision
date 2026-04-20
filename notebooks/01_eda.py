# EDA — Madrid rental listings
# Generates figures in notebooks/figures/

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("notebooks/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/listings_clean.csv")
print(f"Dataset: {len(df)} listings\n")

sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("husl", 8)


# 1. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["price"], bins=50, color=palette[0], edgecolor="white")
axes[0].set_xlabel("Monthly Rent (€)")
axes[0].set_ylabel("Count")
axes[0].set_title("Price Distribution")
axes[0].axvline(df["price"].median(), color="red", linestyle="--", label=f'Median: €{df["price"].median():.0f}')
axes[0].legend()

axes[1].hist(np.log1p(df["price"]), bins=50, color=palette[1], edgecolor="white")
axes[1].set_xlabel("Log(Price)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed Price Distribution")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("1. Price distribution saved")


# 2. Price by zone
zone_order = df.groupby("zone")["price"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="zone", y="price", order=zone_order, palette="husl", ax=ax,
            flierprops=dict(marker="o", markersize=3, alpha=0.3))
ax.set_xlabel("Zone")
ax.set_ylabel("Monthly Rent (€)")
ax.set_title("Rental Price by Zone")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_price_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()
print("2. Price by zone saved")


# 3. Price vs size
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df["sqft_m2"], df["price"], c=df["zone"].astype("category").cat.codes,
                     cmap="tab10", alpha=0.5, s=20)
ax.set_xlabel("Size (m²)")
ax.set_ylabel("Monthly Rent (€)")
ax.set_title("Price vs Size by Zone")

zones = df["zone"].unique()
handles = [plt.scatter([], [], c=plt.cm.tab10(i / len(zones)), s=40, label=z)
           for i, z in enumerate(sorted(zones))]
ax.legend(handles=handles, title="Zone", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_price_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("3. Price vs size saved")


# 4. Correlation heatmap
numeric_cols = ["price", "sqft_m2", "rooms", "bathrooms", "floor_num",
                "num_images", "price_per_m2"]
bool_cols = ["elevator", "ac", "terrace", "furnished", "heating",
             "exterior", "parking", "storage"]

corr_df = df[numeric_cols + bool_cols].copy()
for col in bool_cols:
    corr_df[col] = corr_df[col].astype(float)

corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("4. Correlation heatmap saved")


# 5. Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Rooms
axes[0, 0].hist(df["rooms"].dropna(), bins=range(0, 12), color=palette[2], edgecolor="white")
axes[0, 0].set_title("Rooms")
axes[0, 0].set_xlabel("Number of rooms")

# Bathrooms
axes[0, 1].hist(df["bathrooms"].dropna(), bins=range(0, 8), color=palette[3], edgecolor="white")
axes[0, 1].set_title("Bathrooms")
axes[0, 1].set_xlabel("Number of bathrooms")

# Floor
axes[0, 2].hist(df["floor_num"].dropna(), bins=range(-1, 15), color=palette[4], edgecolor="white")
axes[0, 2].set_title("Floor Number")
axes[0, 2].set_xlabel("Floor")

# Size
axes[1, 0].hist(df["sqft_m2"], bins=40, color=palette[5], edgecolor="white")
axes[1, 0].set_title("Size (m²)")
axes[1, 0].set_xlabel("m²")

# Number of images
axes[1, 1].hist(df["num_images"], bins=40, color=palette[6], edgecolor="white")
axes[1, 1].set_title("Images per Listing")
axes[1, 1].set_xlabel("Number of images")

# Price per m2
axes[1, 2].hist(df["price_per_m2"], bins=40, color=palette[7], edgecolor="white")
axes[1, 2].set_title("Price per m²")
axes[1, 2].set_xlabel("€/m²")

plt.suptitle("Feature Distributions", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("5. Feature distributions saved")


# 6. Boolean features vs price
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(bool_cols):
    data = df.groupby(col)["price"].median()
    axes[i].bar(["No", "Yes"], [data.get(False, 0), data.get(True, 0)],
                color=[palette[0], palette[2]])
    axes[i].set_title(col.capitalize())
    axes[i].set_ylabel("Median Price (€)")

plt.suptitle("Median Price by Feature", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_boolean_features_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("6. Boolean features vs price saved")


# 7. Zone stats
zone_stats = df.groupby("zone").agg(
    count=("price", "size"),
    median_price=("price", "median"),
    mean_price=("price", "mean"),
    median_m2=("sqft_m2", "median"),
    mean_rooms=("rooms", "mean"),
    pct_elevator=("elevator", "mean"),
    pct_ac=("ac", "mean"),
    pct_furnished=("furnished", "mean"),
    median_images=("num_images", "median"),
).round(2)
zone_stats = zone_stats.sort_values("median_price", ascending=False)

print("\n7. Zone Summary:")
print(zone_stats.to_string())
zone_stats.to_csv(FIGURES_DIR / "07_zone_summary.csv")
print("   Saved to CSV")


# 8. Price per m2 by zone
fig, ax = plt.subplots(figsize=(12, 6))
zone_ppm2 = df.groupby("zone")["price_per_m2"].median().sort_values(ascending=False)
zone_ppm2.plot(kind="bar", color=palette[:len(zone_ppm2)], ax=ax, edgecolor="white")
ax.set_ylabel("Median €/m²")
ax.set_title("Median Price per m² by Zone")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_price_per_m2_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()
print("8. Price per m² by zone saved")


# 9. Sample listings by price range
print("\n9. Sample listings by price range:")
for label, low, high in [("Budget (<€1200)", 0, 1200),
                          ("Mid (€1200-€2500)", 1200, 2500),
                          ("Premium (>€2500)", 2500, 99999)]:
    subset = df[(df["price"] >= low) & (df["price"] < high)]
    print(f"  {label}: {len(subset)} listings, avg {subset['sqft_m2'].mean():.0f}m², "
          f"avg {subset['num_images'].mean():.0f} images")


# 10. Summary
print(f"\n{'='*60}")
print(f"DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total listings:     {len(df)}")
print(f"Total images:       {df['num_images'].sum():.0f}")
print(f"Zones:              {df['zone'].nunique()}")
print(f"Price range:        €{df['price'].min():.0f} - €{df['price'].max():.0f}")
print(f"Median price:       €{df['price'].median():.0f}")
print(f"Size range:         {df['sqft_m2'].min():.0f} - {df['sqft_m2'].max():.0f} m²")
print(f"Median size:        {df['sqft_m2'].median():.0f} m²")
print(f"\nCorrelation with price:")
for col in numeric_cols[1:] + bool_cols:
    r = corr_df["price"].corr(corr_df[col])
    if not np.isnan(r):
        bar = "+" * int(abs(r) * 20)
        sign = "+" if r > 0 else "-"
        print(f"  {col:20s} {sign}{bar:20s} {r:+.3f}")

print(f"\nFigures saved to {FIGURES_DIR}/")
