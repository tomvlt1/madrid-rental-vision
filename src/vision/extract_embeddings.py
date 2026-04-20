# Extract ResNet-50 embeddings for all listings (mean-pooled per listing)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from src.config import IMAGES_DIR, PROCESSED_DIR

CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
EMBEDDINGS_INDEX_FILE = PROCESSED_DIR / "embeddings_index.csv"

# same preprocessing ResNet was trained with
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 32


def load_model(device):
    # ResNet-50 without the classification head
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    # drop FC layer -> 2048-dim output from avgpool
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.set_grad_enabled = False
    return model


def load_image(path: Path):
    """Returns preprocessed tensor or None if image can't be loaded."""
    try:
        img = Image.open(path).convert("RGB")
        return TRANSFORM(img)
    except Exception:
        return None


def extract_batch(model, image_tensors, device):
    batch = torch.stack(image_tensors).to(device)
    with torch.no_grad():
        embeddings = model(batch)
    return embeddings.cpu().numpy()


def extract_all():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    model.eval()
    print("Model loaded: ResNet-50 (2048-dim embeddings)")

    df = pd.read_csv(CLEAN_FILE)
    print(f"Listings: {len(df)}")

    all_embs = []
    listing_ids = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        lid = row["url"].rstrip("/").split("/")[-1]
        listing_dir = IMAGES_DIR / lid

        if not listing_dir.exists():
            skipped += 1
            continue

        image_files = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))
        if not image_files:
            skipped += 1
            continue

        tensors = []
        for img_path in image_files:
            t = load_image(img_path)
            if t is not None:
                tensors.append(t)

        if not tensors:
            skipped += 1
            continue

        image_embeddings = []
        for i in range(0, len(tensors), BATCH_SIZE):
            batch = tensors[i:i + BATCH_SIZE]
            embs = extract_batch(model, batch, device)
            image_embeddings.append(embs)

        image_embeddings = np.concatenate(image_embeddings, axis=0)

        # mean pool across all images for this listing
        listing_emb = image_embeddings.mean(axis=0)
        all_embs.append(listing_emb)
        listing_ids.append(lid)

    embeddings_array = np.stack(all_embs)
    np.save(EMBEDDINGS_FILE, embeddings_array)

    index_df = pd.DataFrame({"listing_id": listing_ids})
    index_df.to_csv(EMBEDDINGS_INDEX_FILE, index=False)

    print(f"\nDone!")
    print(f"  Embeddings shape: {embeddings_array.shape}")
    print(f"  Listings processed: {len(listing_ids)}")
    print(f"  Skipped (no images): {skipped}")
    print(f"  Saved to: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    extract_all()
