"""
Zero-shot room-type classification of every per-photo SigLIP embedding.

Uses SigLIP's text encoder to embed a small set of room-type captions, then
computes cosine similarity between each photo's image embedding and every
caption embedding. Top-1 caption becomes the photo's room label.

This is "free" classification: no retraining, no per-photo annotation. SigLIP
was pre-trained to align images with their captions, so similarity between a
photo and "a kitchen" is meaningful out of the box.

Inputs (gitignored, kept locally):
  v2/data/siglip_per_photo.npy           per-photo SigLIP image embeddings
  v2/data/siglip_per_photo_index.csv     listing_id, photo_idx, source_path

Output (committed):
  v2/data/photo_room_labels.csv          listing_id, photo_idx, room_label, room_confidence

Run: python3 v2/precompute_room_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "v2" / "data"
PHOTO_EMB_FILE = DATA_DIR / "siglip_per_photo.npy"
PHOTO_IDX_FILE = DATA_DIR / "siglip_per_photo_index.csv"
OUT_FILE = DATA_DIR / "photo_room_labels.csv"

# Caption set: short, distinct, covers rooms a Spanish rental listing
# normally photographs. The "label" column is what shows up in the UI; the
# "caption" is what we feed to SigLIP's text encoder.
ROOM_CAPTIONS = [
    ("Kitchen",          "a photo of a kitchen"),
    ("Bedroom",          "a photo of a bedroom"),
    ("Bathroom",         "a photo of a bathroom"),
    ("Living room",      "a photo of a living room"),
    ("Dining room",      "a photo of a dining room"),
    ("Terrace",          "a photo of a balcony or terrace"),
    ("Exterior",         "a photo of the exterior of a building"),
    ("Hallway",          "a photo of a hallway or corridor"),
    ("Storage",          "a photo of a closet or storage"),
    ("Floor plan",       "a floor plan diagram"),
    ("Garage",           "a photo of a garage or parking space"),
    ("Pool",             "a photo of a swimming pool"),
]

MODEL_NAME = "google/siglip-base-patch16-224"


def main():
    if not PHOTO_EMB_FILE.exists():
        print(f"ERROR: per-photo SigLIP embeddings missing at {PHOTO_EMB_FILE}")
        print("Run v2/extract_siglip_embeddings.py first to generate them.")
        sys.exit(1)

    print(f"Loading per-photo embeddings from {PHOTO_EMB_FILE}...")
    photo_emb = np.load(PHOTO_EMB_FILE)
    photo_idx = pd.read_csv(PHOTO_IDX_FILE)
    print(f"  {photo_emb.shape[0]:,} photos x {photo_emb.shape[1]} dims")
    assert photo_emb.shape[0] == len(photo_idx), \
        "embedding rows must match index rows"

    print(f"\nLoading SigLIP from {MODEL_NAME}...")
    from transformers import SiglipModel, SiglipProcessor
    model = SiglipModel.from_pretrained(MODEL_NAME)
    processor = SiglipProcessor.from_pretrained(MODEL_NAME)
    model.train(False)  # inference mode

    captions = [c for _, c in ROOM_CAPTIONS]
    labels = [lab for lab, _ in ROOM_CAPTIONS]

    print(f"Embedding {len(captions)} room captions through SigLIP text encoder...")
    inputs = processor(text=captions, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        if hasattr(out, "pooler_output"):
            text_emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            text_emb = out.last_hidden_state[:, 0, :]
        else:
            text_emb = out  # already a tensor
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb_np = text_emb.cpu().numpy()
    print(f"  text emb shape: {text_emb_np.shape}")

    print("\nNormalizing photo embeddings + computing cosine similarity...")
    photo_norm = np.linalg.norm(photo_emb, axis=1, keepdims=True)
    photo_norm[photo_norm == 0] = 1.0
    photo_unit = photo_emb / photo_norm
    sim = photo_unit @ text_emb_np.T  # (n_photos, n_captions)

    top1_idx = sim.argmax(axis=1)
    top1_score = sim[np.arange(len(top1_idx)), top1_idx]

    out = photo_idx.copy()
    out["room_label"] = [labels[i] for i in top1_idx]
    out["room_confidence"] = top1_score.astype(np.float32)

    print("\nLabel distribution:")
    counts = out["room_label"].value_counts()
    for lab, n in counts.items():
        print(f"  {lab:18s} {n:6,}  ({n / len(out) * 100:.1f}%)")

    print(f"\nMean confidence: {out['room_confidence'].mean():.3f}")
    print(f"Median confidence: {out['room_confidence'].median():.3f}")

    keep_cols = ["listing_id", "photo_idx", "room_label", "room_confidence"]
    if "source_path" in out.columns:
        keep_cols.append("source_path")
    out[keep_cols].to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(out):,} room labels to {OUT_FILE}")
    print("\nNext step: load this CSV in the FastAPI backend and attach")
    print('`room_label` to each entry of `per_photo_impact` in the response.')


if __name__ == "__main__":
    main()
