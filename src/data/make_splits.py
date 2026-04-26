# Single source of truth for train/val/test splits.
# Both finetune.py and dataset.py read this file so the ResNet is never
# fine-tuned on a listing that ends up in a downstream GB test set.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR

CLEAN_FILE = PROCESSED_DIR / "listings_clean.csv"
SPLITS_FILE = PROCESSED_DIR / "splits.json"

SEED = 42
# 70/15/15: matches the original downstream convention
VAL_FRAC_OF_REMAINING = 0.5  # 30% temp → 15% val / 15% test


def main():
    df = pd.read_csv(CLEAN_FILE)
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    ids = df["listing_id"].tolist()

    train_ids, temp_ids = train_test_split(ids, test_size=0.3, random_state=SEED)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=VAL_FRAC_OF_REMAINING, random_state=SEED,
    )

    assignment = {}
    for lid in train_ids:
        assignment[lid] = "train"
    for lid in val_ids:
        assignment[lid] = "val"
    for lid in test_ids:
        assignment[lid] = "test"

    # sanity: every listing assigned, no collisions
    assert len(assignment) == len(ids)
    assert set(assignment.values()) == {"train", "val", "test"}

    manifest = {
        "seed": SEED,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(ids),
        },
        "listings": assignment,
    }
    SPLITS_FILE.write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {SPLITS_FILE}: "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )


if __name__ == "__main__":
    main()
