# Get text embeddings from listing descriptions using multilingual sentence-transformers

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DIR

CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"
TEXT_EMBEDDINGS_FILE = PROCESSED_DIR / "text_embeddings.npy"
TEXT_INDEX_FILE = PROCESSED_DIR / "text_embeddings_index.csv"


def main():
    df = pd.read_csv(CLEAN_FILE)
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1])
    print(f"Listings: {len(df)}")

    df["description"] = df["description"].fillna("")

    # multilingual model since descriptions are in Spanish
    print("Loading sentence transformer...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding...")
    descriptions = df["description"].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)

    print(f"Embeddings shape: {embeddings.shape}")

    np.save(TEXT_EMBEDDINGS_FILE, embeddings)
    pd.DataFrame({"listing_id": df["listing_id"].values}).to_csv(TEXT_INDEX_FILE, index=False)

    print(f"Saved to {TEXT_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
