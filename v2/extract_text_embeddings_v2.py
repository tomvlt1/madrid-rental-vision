"""
Extract text embeddings on the unified listings_clean_v2.csv.

Reads:
  - v2/data/listings_clean_v2.csv  (4,236 listings)

Writes (REAL files, not symlinks: replaces any existing symlink to ai2):
  - v2/data/text_embeddings.npy
  - v2/data/text_embeddings_index.csv

The morning_pipeline created symlinks here pointing at ai2's 1,425-row
embeddings. We replace them with real files containing all 4,236 listings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from v2.paths import V2_DATA_DIR  # noqa: E402

CLEAN_FILE = V2_DATA_DIR / "listings_clean_v2.csv"
OUT_NPY = V2_DATA_DIR / "text_embeddings.npy"
OUT_IDX = V2_DATA_DIR / "text_embeddings_index.csv"


def main():
    if not CLEAN_FILE.exists():
        print(f"ERROR: {CLEAN_FILE} not found: run clean_and_merge first")
        sys.exit(1)

    # If outputs are symlinks (from morning_pipeline staging), remove them
    # before writing: otherwise np.save would overwrite the symlink target
    # (i.e. ai2's original embeddings file).
    for p in (OUT_NPY, OUT_IDX):
        if p.is_symlink():
            print(f"  removing symlink {p.name} (was pointing to ai2)")
            p.unlink()

    df = pd.read_csv(CLEAN_FILE)
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1])
    print(f"Listings: {len(df)}")
    df["description"] = df["description"].fillna("")

    print("Loading sentence transformer...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding...")
    embeddings = model.encode(
        df["description"].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(OUT_NPY, embeddings)
    pd.DataFrame({"listing_id": df["listing_id"].values}).to_csv(OUT_IDX, index=False)
    print(f"Saved {OUT_NPY}")
    print(f"Saved {OUT_IDX}")


if __name__ == "__main__":
    main()
