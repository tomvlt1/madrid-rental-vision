# v2: Extract SigLIP image embeddings for each listing.
#
# Produces two artifacts (both written under v2/data/, not the shared
# data/processed/ tree, so the original ResNet embeddings are untouched):
#
#   1) siglip_embeddings.npy          — (n_listings, 768), mean-pooled across
#                                        a listing's photos. Drop-in replacement
#                                        for embeddings.npy in the CV script.
#   2) siglip_embeddings_index.csv    — aligned to (1), columns: listing_id.
#
#   3) siglip_per_photo.npy           — (n_total_photos, 768), one row per
#                                        individual photo. Needed by the
#                                        attention-pooling module (attention_pool.py).
#   4) siglip_per_photo_index.csv     — aligned to (3), columns:
#                                        listing_id, photo_idx, source_path.
#
# Model: google/siglip-base-patch16-224 (vision output dim = 768).
# Why SigLIP over ResNet-50: trained on 10B image-text pairs with a sigmoid
# contrastive loss; semantic concepts (modern, luxury, bright, cramped) are
# already in the representation. ResNet-50's ImageNet training is furniture/
# animal classification, a worse fit for rental aesthetics.
#
# Dependencies: transformers>=4.38. If not installed:
#   pip install "transformers>=4.38" "Pillow>=10" "torch>=2.0" "tqdm>=4.65"

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import IMAGES_DIR, PROCESSED_DIR, V2_DATA_DIR  # noqa: E402

LISTING_EMB_FILE = V2_DATA_DIR / "siglip_embeddings.npy"
LISTING_INDEX_FILE = V2_DATA_DIR / "siglip_embeddings_index.csv"
PER_PHOTO_EMB_FILE = V2_DATA_DIR / "siglip_per_photo.npy"
PER_PHOTO_INDEX_FILE = V2_DATA_DIR / "siglip_per_photo_index.csv"

MODEL_NAME = "google/siglip-base-patch16-224"
EMBED_DIM = 768
BATCH_SIZE = 32
CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    # AutoImageProcessor avoids pulling in the SigLIP tokenizer (which
    # needs sentencepiece). We only do image embedding, so no tokenizer
    # is needed.
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.train(False)  # inference mode (equivalent to .eval() on nn.Module)
    return processor, model


def load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_batch(processor, model, pil_images, device) -> np.ndarray:
    with torch.no_grad():
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        # In transformers >= 5.x, SiglipModel.get_image_features() can
        # return a structured output. Fall through to the vision tower's
        # pooler_output, which is what we want anyway (pooled image rep).
        features = model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            # structured output — pull pooler_output
            features = getattr(features, "pooler_output", None)
            if features is None:
                features = model.vision_model(**inputs).pooler_output
    return features.detach().cpu().numpy()


SAVE_EVERY = 100  # checkpoint after this many newly-processed listings


def _load_existing_state():
    """Resume support: read any prior listing/per-photo arrays + indices and
    return them so we can append to them. Returns (listing_embs, listing_ids,
    per_photo_embs, per_photo_rows, done_set)."""
    listing_embs: list[np.ndarray] = []
    listing_ids: list[str] = []
    per_photo_embs: list[np.ndarray] = []
    per_photo_rows: list[dict] = []
    done: set[str] = set()
    if LISTING_EMB_FILE.exists() and LISTING_INDEX_FILE.exists():
        try:
            arr = np.load(LISTING_EMB_FILE)
            idx = pd.read_csv(LISTING_INDEX_FILE)
            if len(arr) == len(idx) and len(idx) > 0:
                for i in range(len(arr)):
                    listing_embs.append(arr[i])
                    listing_ids.append(str(idx.iloc[i]["listing_id"]))
                done.update(listing_ids)
                print(f"  resume: {len(done)} listings already in {LISTING_EMB_FILE.name}")
        except Exception as e:
            print(f"  resume failed (will start fresh): {e}")
            listing_embs, listing_ids, done = [], [], set()
    if (
        PER_PHOTO_EMB_FILE.exists()
        and PER_PHOTO_INDEX_FILE.exists()
        and listing_embs  # only restore per-photo state if listing state was loaded
    ):
        try:
            arr = np.load(PER_PHOTO_EMB_FILE)
            idx = pd.read_csv(PER_PHOTO_INDEX_FILE)
            if len(arr) == len(idx):
                for i in range(len(arr)):
                    per_photo_embs.append(arr[i])
                for row in idx.to_dict(orient="records"):
                    per_photo_rows.append(row)
        except Exception as e:
            print(f"  resume per-photo failed (will start fresh): {e}")
            per_photo_embs, per_photo_rows = [], []
    return listing_embs, listing_ids, per_photo_embs, per_photo_rows, done


def _atomic_save_npy(arr, target: Path):
    """Write to target.tmp then rename — but np.save auto-appends '.npy'
    so we have to use a temp path that ALREADY ends in .npy."""
    tmp = target.with_name(target.name + ".tmp.npy")  # literal '.tmp.npy' suffix
    np.save(tmp, arr)
    # numpy actually wrote to tmp (since the path already ended in .npy)
    tmp.replace(target)


def _atomic_save_csv(df, target: Path):
    tmp = Path(str(target) + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(target)


def _checkpoint(listing_embs, listing_ids, per_photo_embs, per_photo_rows):
    """Atomic save: write to .tmp then os.replace so a kill mid-write doesn't
    corrupt the artifacts."""
    if not listing_embs:
        return
    _atomic_save_npy(np.stack(listing_embs).astype(np.float32), LISTING_EMB_FILE)
    _atomic_save_csv(pd.DataFrame({"listing_id": listing_ids}), LISTING_INDEX_FILE)

    if per_photo_embs:
        _atomic_save_npy(np.stack(per_photo_embs).astype(np.float32), PER_PHOTO_EMB_FILE)
        _atomic_save_csv(pd.DataFrame(per_photo_rows), PER_PHOTO_INDEX_FILE)


def extract_all():
    device = pick_device()
    print(f"Device: {device}")
    processor, model = load_model(device)
    print(f"Model loaded: {MODEL_NAME} (dim={EMBED_DIM})")

    df = pd.read_csv(CLEAN_FILE)
    print(f"Listings in clean dataset: {len(df)}")

    V2_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from any existing artifacts so a crashed previous run doesn't
    # mean redoing all the listings already embedded.
    listing_embs, listing_ids, per_photo_embs, per_photo_rows, done = _load_existing_state()

    skipped = 0
    new_since_last_checkpoint = 0

    pending = [r for _, r in df.iterrows()
               if str(r["url"]).rstrip("/").split("/")[-1] not in done]
    print(f"Listings to process this run: {len(pending)}  (skipping {len(done)} already done)")

    for row in tqdm(pending, desc="SigLIP"):
        lid = str(row["url"]).rstrip("/").split("/")[-1]
        listing_dir = IMAGES_DIR / lid
        if not listing_dir.exists():
            skipped += 1
            continue

        image_files = sorted(listing_dir.glob("*.jpg")) + sorted(listing_dir.glob("*.webp"))
        if not image_files:
            skipped += 1
            continue

        pil_images = []
        kept_paths = []
        for p in image_files:
            img = load_image(p)
            if img is not None:
                pil_images.append(img)
                kept_paths.append(p)
        if not pil_images:
            skipped += 1
            continue

        try:
            photo_embs = []
            for i in range(0, len(pil_images), BATCH_SIZE):
                batch = pil_images[i : i + BATCH_SIZE]
                embs = embed_batch(processor, model, batch, device)
                photo_embs.append(embs)
            photo_embs = np.concatenate(photo_embs, axis=0)
        except Exception as e:
            print(f"  embed failed for {lid}: {e}")
            skipped += 1
            continue

        listing_embs.append(photo_embs.mean(axis=0))
        listing_ids.append(lid)

        for idx, (emb, p) in enumerate(zip(photo_embs, kept_paths)):
            per_photo_embs.append(emb)
            try:
                src_rel = str(p.relative_to(IMAGES_DIR)) if IMAGES_DIR else str(p)
            except ValueError:
                src_rel = str(p)
            per_photo_rows.append({"listing_id": lid, "photo_idx": idx, "source_path": src_rel})

        new_since_last_checkpoint += 1
        if new_since_last_checkpoint >= SAVE_EVERY:
            _checkpoint(listing_embs, listing_ids, per_photo_embs, per_photo_rows)
            new_since_last_checkpoint = 0

    # final save
    _checkpoint(listing_embs, listing_ids, per_photo_embs, per_photo_rows)

    print("\nDone.")
    print(f"  Listings in output: {len(listing_ids)}  (skipped this run: {skipped})")
    if listing_embs:
        print(f"  Listing embeddings: ({len(listing_embs)}, {EMBED_DIM}) -> {LISTING_EMB_FILE}")
    if per_photo_embs:
        print(f"  Per-photo embeddings: ({len(per_photo_embs)}, {EMBED_DIM}) -> {PER_PHOTO_EMB_FILE}")


if __name__ == "__main__":
    extract_all()
