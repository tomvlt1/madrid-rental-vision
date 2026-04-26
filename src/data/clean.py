# Clean raw listings: dedup, remove outliers, map neighborhoods to zones

import json
import re

import pandas as pd

from src.data.neighborhoods import get_zone
from src.config import PROCESSED_DIR

RAW_FILE = PROCESSED_DIR / "listings.csv"
CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"


# resolution quality ranking (best first)
RESOLUTION_RANK = [
    "WEB_DETAIL_TOP-L-P",
    "WEB_DETAIL_TOP-L-L",
    "WEB_DETAIL_TOP",
    "WEB_DETAIL-M-L",
    "WEB_DETAIL",
]


def deduplicate_images(image_urls_json: str) -> str:
    """Keep best resolution per unique image."""
    try:
        urls = json.loads(image_urls_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps([])

    image_groups = {}
    for url in urls:
        id_match = re.search(r'image\.master/([a-f0-9/]+\d+)', url)
        if not id_match:
            continue
        img_id = id_match.group(1)

        res_match = re.search(r'/blur/([^/]+)/', url)
        resolution = res_match.group(1) if res_match else "UNKNOWN"

        if img_id not in image_groups:
            image_groups[img_id] = []
        image_groups[img_id].append((url, resolution))

    best_urls = []
    for img_id, variants in image_groups.items():
        def sort_key(item):  # lower rank = better res, prefer jpg
            url, res = item
            rank = RESOLUTION_RANK.index(res) if res in RESOLUTION_RANK else 99
            is_webp = 1 if ".webp" in url else 0
            return (rank, is_webp)

        variants.sort(key=sort_key)
        best_urls.append(variants[0][0])

    return json.dumps(best_urls)


def clean_data():
    print(f"Loading {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE)
    n_raw = len(df)
    print(f"Raw listings: {n_raw}")

    df = df.drop_duplicates(subset="url", keep="last")
    print(f"After dedup by URL: {len(df)} (removed {n_raw - len(df)})")

    # Dedup by feature tuple: same property re-listed under new IDs leaks
    # across train/val/test splits otherwise. Groups must match on
    # (price, sqft, rooms, bathrooms, location, num_images) to collapse.
    before = len(df)
    feature_cols = ["price", "sqft_m2", "rooms", "bathrooms", "location"]
    # Fill NaN with sentinel to allow drop_duplicates to group identical rows
    dedup_key = df[feature_cols].fillna(-1).astype(str).agg("|".join, axis=1)
    df = df.loc[~dedup_key.duplicated(keep="first")]
    collisions = before - len(df)
    if collisions > 0:
        print(f"After dedup by feature tuple: {len(df)} (collapsed {collisions} re-listings)")
    else:
        print(f"Dedup by feature tuple: no collisions detected")

    critical = ["price", "sqft_m2", "location"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"After dropping missing critical fields: {len(df)} (removed {before - len(df)})")

    before = len(df)
    # price < 400 or > 15000 are likely errors; sqft < 15 is a parking spot
    df = df[(df["price"] >= 400) & (df["price"] <= 15000)]
    df = df[(df["sqft_m2"] >= 15) & (df["sqft_m2"] <= 500)]

    print(f"After removing outliers: {len(df)} (removed {before - len(df)})")

    df["zone"] = df["location"].apply(get_zone)
    unmatched = (df["zone"] == "Otro").sum()
    print(f"Zone mapping: {unmatched} unmatched neighborhoods")
    if unmatched > 0:
        print("  Unmatched:")
        for loc in df[df["zone"] == "Otro"]["location"].unique():
            print(f"    {loc}")

    print("Deduplicating images...")
    df["image_urls"] = df["image_urls"].apply(deduplicate_images)
    df["num_images"] = df["image_urls"].apply(lambda x: len(json.loads(x)))

    # need at least 3 images for meaningful embeddings
    before = len(df)
    df = df[df["num_images"] >= 3]
    print(f"After filtering <3 images: {len(df)} (removed {before - len(df)})")

    df["price_per_m2"] = (df["price"] / df["sqft_m2"]).round(2)

    bool_cols = ["elevator", "ac", "terrace", "furnished", "heating",
                 "exterior", "parking", "storage"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    def parse_floor(floor_str):
        if pd.isna(floor_str):
            return None
        floor_str = str(floor_str).lower()
        if "bajo" in floor_str or "baja" in floor_str:
            return 0
        if "sótano" in floor_str or "sotano" in floor_str:
            return -1
        digits = "".join(c for c in floor_str if c in "0123456789")
        if digits:
            return int(digits)
        return None

    df["floor_num"] = df["floor"].apply(parse_floor)

    df = df.reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"CLEAN DATASET: {len(df)} listings")
    print(f"{'='*50}")
    print(f"\nPrice:  €{df['price'].mean():.0f} avg | €{df['price'].median():.0f} median | €{df['price'].min():.0f}-€{df['price'].max():.0f}")
    print(f"Size:   {df['sqft_m2'].mean():.0f}m² avg | {df['sqft_m2'].median():.0f}m² median")
    print(f"Images: {df['num_images'].mean():.1f} avg | {df['num_images'].median():.0f} median")
    print(f"\nZone distribution:")
    print(df["zone"].value_counts().to_string())
    print(f"\nFeature completeness:")
    for col in ["rooms", "bathrooms", "floor_num"]:
        pct = df[col].notna().mean() * 100
        print(f"  {col}: {pct:.0f}%")

    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved to {CLEAN_FILE}")


if __name__ == "__main__":
    clean_data()
