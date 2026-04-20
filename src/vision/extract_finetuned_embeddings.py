# Same as extract_embeddings.py but using the fine-tuned ResNet instead of frozen ImageNet

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import IMAGES_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.vision.finetune import PriceResNet

MODELS_DIR = PROJECT_ROOT / "models"
CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings_finetuned.npy"
EMBEDDINGS_INDEX_FILE = PROCESSED_DIR / "embeddings_finetuned_index.csv"

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 32


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
    print("Loaded fine-tuned ResNet-50")

    df = pd.read_csv(CLEAN_FILE)
    print(f"Listings: {len(df)}")

    all_embs = []
    listing_ids = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        lid = row["url"].rstrip("/").split("/")[-1]
        listing_dir = IMAGES_DIR / str(lid)

        if not listing_dir.exists():
            skipped += 1
            continue

        image_files = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))
        if not image_files:
            skipped += 1
            continue

        tensors = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                tensors.append(TRANSFORM(img))
            except Exception:
                continue

        if not tensors:
            skipped += 1
            continue

        image_embeddings = []
        for i in range(0, len(tensors), BATCH_SIZE):
            batch = torch.stack(tensors[i:i + BATCH_SIZE]).to(device)
            with torch.no_grad():
                embs = model.extract_embedding(batch).cpu().numpy()
            image_embeddings.append(embs)

        image_embeddings = np.concatenate(image_embeddings, axis=0)
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
    print(f"  Skipped: {skipped}")
    print(f"  Saved to: {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
