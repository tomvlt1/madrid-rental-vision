# Predict rental price for a listing given features + images

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from src.config import IMAGES_DIR, PROCESSED_DIR, PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_models():
    gb_tab = joblib.load(MODELS_DIR / "gb_tabular.joblib")
    gb_full = joblib.load(MODELS_DIR / "gb_tabular_image.joblib")
    tab_scaler = joblib.load(MODELS_DIR / "tab_scaler.joblib")
    img_scaler = joblib.load(MODELS_DIR / "img_scaler.joblib")
    pca = joblib.load(MODELS_DIR / "pca.joblib")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return gb_tab, gb_full, tab_scaler, img_scaler, pca, feature_names


def extract_embedding(image_dir):
    """Mean-pooled ResNet-50 embedding from a directory of images."""
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.train(False)

    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.webp"))

    if not image_files:
        return None

    tensors = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            tensors.append(TRANSFORM(img))
        except Exception:
            continue

    if not tensors:
        return None

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        embeddings = model(batch).cpu().numpy()

    return embeddings.mean(axis=0)


def build_tabular_features(sqft, rooms, bathrooms, zone, floor_num=None,
                           elevator=False, ac=False, terrace=False,
                           furnished=False, heating=False, exterior=False,
                           parking=False, storage=False, num_images=0,
                           feature_names=None):
    """Build feature vector matching the training format."""
    row = {
        "sqft_m2": sqft,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "floor_num": floor_num or 0,
        "num_images": num_images,
        "elevator": float(elevator),
        "ac": float(ac),
        "terrace": float(terrace),
        "furnished": float(furnished),
        "heating": float(heating),
        "exterior": float(exterior),
        "parking": float(parking),
        "storage": float(storage),
    }

    all_zones = ["Arganzuela", "Centro", "Chamberí", "Norte",
                 "Oeste", "Periferia Norte", "Salamanca-Retiro", "Sur-Sureste"]
    for z in all_zones:
        row[f"zone_{z}"] = 1.0 if z == zone else 0.0

    return np.array([row.get(f, 0.0) for f in feature_names], dtype=np.float32)


def predict_listing(listing_id):
    gb_tab, gb_full, tab_scaler, img_scaler, pca, feature_names = load_models()

    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    row = df[df["listing_id"] == str(listing_id)]

    if row.empty:
        print(f"Listing {listing_id} not found in dataset.")
        return

    row = row.iloc[0]
    actual_price = row["price"]

    from src.data.neighborhoods import get_zone
    def safe_num(val, default=0):
        if pd.isna(val):
            return default
        return val

    tab = build_tabular_features(
        sqft=row["sqft_m2"], rooms=safe_num(row.get("rooms"), 0),
        bathrooms=safe_num(row.get("bathrooms"), 0), zone=row["zone"],
        floor_num=safe_num(row.get("floor_num"), 0),
        elevator=row.get("elevator", False), ac=row.get("ac", False),
        terrace=row.get("terrace", False), furnished=row.get("furnished", False),
        heating=row.get("heating", False), exterior=row.get("exterior", False),
        parking=row.get("parking", False), storage=row.get("storage", False),
        num_images=row.get("num_images", 0),
        feature_names=feature_names,
    )

    tab_scaled = tab_scaler.transform(tab.reshape(1, -1))

    pred_tab_log = gb_tab.predict(tab_scaled)[0]
    pred_tab = np.expm1(pred_tab_log)

    image_dir = IMAGES_DIR / str(listing_id)
    embedding = extract_embedding(image_dir)

    if embedding is not None:
        emb_scaled = img_scaler.transform(embedding.reshape(1, -1))
        emb_pca = pca.transform(emb_scaled)
        combined = np.hstack([tab_scaled, emb_pca])
        pred_full_log = gb_full.predict(combined)[0]
        pred_full = np.expm1(pred_full_log)
    else:
        pred_full = None

    print(f"\n{'='*50}")
    print(f"LISTING: {row['title']}")
    print(f"{'='*50}")
    print(f"Location:   {row['location']} ({row['zone']})")
    print(f"Size:       {row['sqft_m2']}m² | {int(safe_num(row.get('rooms'), 0))} rooms | "
          f"{int(safe_num(row.get('bathrooms'), 0))} bathrooms")
    print(f"Floor:      {row.get('floor', 'N/A')}")
    print(f"Images:     {row['num_images']}")
    print(f"\nActual rent:           EUR {actual_price:.0f}/month")
    print(f"Predicted (tabular):   EUR {pred_tab:.0f}/month  "
          f"(error: EUR {abs(actual_price - pred_tab):.0f})")
    if pred_full is not None:
        print(f"Predicted (+ images):  EUR {pred_full:.0f}/month  "
              f"(error: EUR {abs(actual_price - pred_full):.0f})")
        delta = abs(actual_price - pred_tab) - abs(actual_price - pred_full)
        if delta > 0:
            print(f"\nImages improved prediction by EUR {delta:.0f}")
        else:
            print(f"\nImages worsened prediction by EUR {-delta:.0f}")
    print()


def predict_custom(args):
    gb_tab, gb_full, tab_scaler, img_scaler, pca, feature_names = load_models()

    tab = build_tabular_features(
        sqft=args.sqft, rooms=args.rooms, bathrooms=args.bathrooms,
        zone=args.zone, floor_num=args.floor or 0,
        elevator=args.elevator, ac=args.ac, terrace=args.terrace,
        furnished=args.furnished, heating=args.heating,
        exterior=args.exterior, parking=args.parking,
        storage=args.storage, num_images=0,
        feature_names=feature_names,
    )

    tab_scaled = tab_scaler.transform(tab.reshape(1, -1))
    pred_tab_log = gb_tab.predict(tab_scaled)[0]
    pred_tab = np.expm1(pred_tab_log)

    print(f"\n{'='*50}")
    print(f"CUSTOM PREDICTION")
    print(f"{'='*50}")
    print(f"Size: {args.sqft}m² | {args.rooms} rooms | {args.bathrooms} bath | {args.zone}")
    print(f"\nPredicted rent (tabular): EUR {pred_tab:.0f}/month")

    if args.images:
        embedding = extract_embedding(args.images)
        if embedding is not None:
            emb_scaled = img_scaler.transform(embedding.reshape(1, -1))
            emb_pca = pca.transform(emb_scaled)
            combined = np.hstack([tab_scaled, emb_pca])
            pred_full_log = gb_full.predict(combined)[0]
            pred_full = np.expm1(pred_full_log)
            print(f"Predicted rent (+ images): EUR {pred_full:.0f}/month")
            print(f"Image delta: EUR {pred_full - pred_tab:+.0f}/month")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rental price")
    parser.add_argument("--listing", type=str, help="Listing ID from dataset")
    parser.add_argument("--sqft", type=float, help="Size in m2")
    parser.add_argument("--rooms", type=int, default=2)
    parser.add_argument("--bathrooms", type=int, default=1)
    parser.add_argument("--zone", type=str, default="Centro")
    parser.add_argument("--floor", type=int, default=None)
    parser.add_argument("--elevator", action="store_true")
    parser.add_argument("--ac", action="store_true")
    parser.add_argument("--terrace", action="store_true")
    parser.add_argument("--furnished", action="store_true")
    parser.add_argument("--heating", action="store_true")
    parser.add_argument("--exterior", action="store_true")
    parser.add_argument("--parking", action="store_true")
    parser.add_argument("--storage", action="store_true")
    parser.add_argument("--images", type=str, help="Path to image directory")

    args = parser.parse_args()

    if args.listing:
        predict_listing(args.listing)
    elif args.sqft:
        predict_custom(args)
    else:
        # Demo: predict a few listings from the dataset
        df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
        df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1])

        # Pick 3 diverse examples
        cheap = df.nsmallest(1, "price").iloc[0]["listing_id"]
        mid = df.iloc[(df["price"] - df["price"].median()).abs().argsort().iloc[0]]["listing_id"]
        expensive = df.nlargest(1, "price").iloc[0]["listing_id"]

        for lid in [cheap, mid, expensive]:
            predict_listing(lid)
