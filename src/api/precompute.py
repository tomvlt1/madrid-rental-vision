# Precompute per-listing scores for the scout/diagnose views.
# Runs once, saves to data/processed/listings_enriched.parquet.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import IMAGES_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.vision.finetune import PriceResNet

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_FILE = PROCESSED_DIR / "listings_enriched.parquet"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 32
NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = ["elevator", "ac", "terrace", "furnished", "heating",
                 "exterior", "parking", "storage"]


def build_tab_matrix(df, feature_names):
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    tab = pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1)
    for f in feature_names:
        if f not in tab.columns:
            tab[f] = 0.0
    return tab[feature_names].values.astype(np.float32)


def score_photos(model, listing_dir, device):
    """Run each image through the fine-tuned ResNet -> predicted log-price."""
    image_files = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))
    if not image_files:
        return [], []

    tensors, valid_files = [], []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            tensors.append(TRANSFORM(img))
            valid_files.append(img_path.name)
        except Exception:
            continue

    if not tensors:
        return [], []

    log_preds = []
    for i in range(0, len(tensors), BATCH_SIZE):
        batch = torch.stack(tensors[i:i + BATCH_SIZE]).to(device)
        with torch.no_grad():
            out = model(batch).cpu().numpy()
        log_preds.extend(out.tolist())

    prices = [float(np.expm1(lp)) for lp in log_preds]
    return valid_files, prices


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    print(f"Listings: {len(df)}")

    # --- load sklearn pipeline -----------------------------------------
    gb_tab = joblib.load(MODELS_DIR / "gb_tabular.joblib")
    gb_all = joblib.load(MODELS_DIR / "gb_tabular_text_finetuned_image.joblib")
    tab_scaler = joblib.load(MODELS_DIR / "tab_scaler.joblib")
    img_scaler_ft = joblib.load(MODELS_DIR / "img_scaler_finetuned.joblib")
    pca_ft = joblib.load(MODELS_DIR / "pca_finetuned.joblib")
    text_scaler = joblib.load(MODELS_DIR / "text_scaler.joblib")
    pca_text = joblib.load(MODELS_DIR / "pca_text.joblib")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    # --- load precomputed embeddings -----------------------------------
    ft_emb = np.load(PROCESSED_DIR / "embeddings_finetuned.npy")
    ft_idx = pd.read_csv(PROCESSED_DIR / "embeddings_finetuned_index.csv")
    ft_idx["listing_id"] = ft_idx["listing_id"].astype(str)
    ft_lookup = dict(zip(ft_idx["listing_id"], range(len(ft_idx))))

    text_emb = np.load(PROCESSED_DIR / "text_embeddings.npy")
    text_idx = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    text_idx["listing_id"] = text_idx["listing_id"].astype(str)
    text_lookup = dict(zip(text_idx["listing_id"], range(len(text_idx))))

    # --- premium text centroid (top quartile by rent) ------------------
    q75 = df["price"].quantile(0.75)
    premium_ids = df[df["price"] >= q75]["listing_id"].tolist()
    premium_rows = [text_lookup[lid] for lid in premium_ids if lid in text_lookup]
    premium_centroid = text_emb[premium_rows].mean(axis=0)
    cent_norm = premium_centroid / (np.linalg.norm(premium_centroid) + 1e-9)
    print(f"Premium centroid built from {len(premium_rows)} listings (rent >= EUR {q75:.0f})")

    # --- fine-tuned ResNet for per-photo scoring -----------------------
    model = PriceResNet()
    state = torch.load(MODELS_DIR / "resnet_finetuned.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.train(False)

    # --- scale + transform tabular once for all listings ---------------
    tab_raw = build_tab_matrix(df, feature_names)
    tab_scaled = tab_scaler.transform(tab_raw)

    pred_tab_log = gb_tab.predict(tab_scaled)
    pred_tab_eur = np.expm1(pred_tab_log)

    # --- iterate listings, score photos + compute full prediction ------
    records = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        lid = row["listing_id"]
        listing_dir = IMAGES_DIR / lid

        photo_files, photo_prices = [], []
        image_score_eur = np.nan
        if listing_dir.exists():
            photo_files, photo_prices = score_photos(model, listing_dir, device)
            if photo_prices:
                image_score_eur = float(np.mean(photo_prices))

        # full prediction needs ft + text embeddings
        pred_full_log = np.nan
        has_ft = lid in ft_lookup
        has_text = lid in text_lookup
        if has_ft and has_text:
            ft_vec = img_scaler_ft.transform(ft_emb[ft_lookup[lid]].reshape(1, -1))
            ft_pca_vec = pca_ft.transform(ft_vec)
            txt_vec = text_scaler.transform(text_emb[text_lookup[lid]].reshape(1, -1))
            txt_pca_vec = pca_text.transform(txt_vec)
            combined = np.hstack([tab_scaled[i:i + 1], txt_pca_vec, ft_pca_vec])
            pred_full_log = float(gb_all.predict(combined)[0])

        pred_full_eur = float(np.expm1(pred_full_log)) if not np.isnan(pred_full_log) else np.nan

        # text distance from premium centroid (cosine)
        text_dist = np.nan
        if has_text:
            v = text_emb[text_lookup[lid]]
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            text_dist = float(1.0 - np.dot(v_norm, cent_norm))

        records.append({
            "listing_id": lid,
            "url": row["url"],
            "title": row.get("title"),
            "location": row.get("location"),
            "zone": row["zone"],
            "rent_eur": float(row["price"]),
            "sqft_m2": float(row["sqft_m2"]),
            "rooms": float(row["rooms"]) if pd.notna(row["rooms"]) else None,
            "bathrooms": float(row["bathrooms"]) if pd.notna(row["bathrooms"]) else None,
            "floor": row.get("floor"),
            "num_images": int(row["num_images"]) if pd.notna(row["num_images"]) else 0,
            "description": row.get("description"),
            "predicted_rent_tabular_eur": float(pred_tab_eur[i]),
            "predicted_rent_full_eur": pred_full_eur,
            "image_score_eur": image_score_eur,
            "photo_filenames": photo_files,
            "photo_scores_eur": photo_prices,
            "text_distance_premium": text_dist,
        })

    out = pd.DataFrame.from_records(records)

    # zone medians for display
    zone_medians = out.groupby("zone")["rent_eur"].median().to_dict()
    out["zone_median_rent_eur"] = out["zone"].map(zone_medians)

    # image score percentile (within the whole dataset, higher = more expensive-looking)
    has_score = out["image_score_eur"].notna()
    ranks = out.loc[has_score, "image_score_eur"].rank(pct=True) * 100
    out["image_score_percentile"] = np.nan
    out.loc[has_score, "image_score_percentile"] = ranks.values

    # under-marketing score: positive when rent is below tabular prediction AND photos score low
    # normalized rent gap (actual vs tabular-only prediction)
    out["rent_gap_pct"] = (out["predicted_rent_tabular_eur"] - out["rent_eur"]) / out["predicted_rent_tabular_eur"]
    # combine: big positive gap (rent below model) AND low image score percentile
    # use a simple additive score; both normalized to ~0-1 range
    img_low = 1.0 - (out["image_score_percentile"].fillna(50) / 100.0)
    rent_low = out["rent_gap_pct"].clip(-0.5, 0.5)
    out["under_marketing_score"] = (0.6 * rent_low + 0.4 * img_low * 0.5).round(4)

    out.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  Rows: {len(out)}")
    print(f"  With photo scores: {has_score.sum()}")
    print(f"  With full prediction: {out['predicted_rent_full_eur'].notna().sum()}")


if __name__ == "__main__":
    main()
