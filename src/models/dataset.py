# Merge tabular features with image embeddings for model training

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR

CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
EMBEDDINGS_INDEX_FILE = PROCESSED_DIR / "embeddings_index.csv"
SPLITS_FILE = PROCESSED_DIR / "splits.json"

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = ["elevator", "ac", "terrace", "furnished", "heating",
                 "exterior", "parking", "storage"]
CATEGORICAL_FEATURES = ["zone"]


def load_dataset(random_state=42):
    """Load clean data + embeddings, split from shared manifest, scale."""
    df = pd.read_csv(CLEAN_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    emb_index = pd.read_csv(EMBEDDINGS_INDEX_FILE)

    emb_index["emb_row"] = range(len(emb_index))
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    emb_index["listing_id"] = emb_index["listing_id"].astype(str)
    df = df.merge(emb_index, on="listing_id", how="inner")

    print(f"Listings with embeddings: {len(df)}")

    # Load the shared split manifest (same file finetune.py uses) so the
    # fine-tuned ResNet never sees a listing that lands in our test set.
    with open(SPLITS_FILE) as f:
        manifest = json.load(f)
    assignment = manifest["listings"]
    df["split"] = df["listing_id"].map(assignment)
    # Drop any listings not in the manifest (e.g. if clean.py was re-run
    # after make_splits.py without regenerating). Should be 0 in normal flow.
    n_unmapped = df["split"].isna().sum()
    if n_unmapped > 0:
        print(f"Warning: {n_unmapped} listings missing from splits.json — dropping")
        df = df[df["split"].notna()].reset_index(drop=True)

    df["log_price"] = np.log1p(df["price"])

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)

    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)

    tab_features = pd.concat([
        df[NUMERIC_FEATURES],
        df[BOOL_FEATURES],
        zone_dummies,
    ], axis=1)
    feature_names = list(tab_features.columns)

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values
    test_mask = (df["split"] == "test").values
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(
        f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"(from shared manifest)"
    )

    X_tab = tab_features.values.astype(np.float32)
    X_img = embeddings[df["emb_row"].values].astype(np.float32)
    y = df["log_price"].values.astype(np.float32)

    X_tab_train, X_tab_val, X_tab_test = X_tab[train_idx], X_tab[val_idx], X_tab[test_idx]
    X_img_train, X_img_val, X_img_test = X_img[train_idx], X_img[val_idx], X_img[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    tab_scaler = StandardScaler()
    X_tab_train = tab_scaler.fit_transform(X_tab_train)
    X_tab_val = tab_scaler.transform(X_tab_val)
    X_tab_test = tab_scaler.transform(X_tab_test)

    img_scaler = StandardScaler()
    X_img_train = img_scaler.fit_transform(X_img_train)
    X_img_val = img_scaler.transform(X_img_val)
    X_img_test = img_scaler.transform(X_img_test)

    return {
        "X_tab_train": X_tab_train, "X_tab_val": X_tab_val, "X_tab_test": X_tab_test,
        "X_img_train": X_img_train, "X_img_val": X_img_val, "X_img_test": X_img_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "tab_scaler": tab_scaler, "img_scaler": img_scaler,
        "feature_names": feature_names,
        "df_train": df.iloc[train_idx], "df_val": df.iloc[val_idx], "df_test": df.iloc[test_idx],
    }
