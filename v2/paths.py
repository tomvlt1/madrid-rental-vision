# Path helper for v2 scripts.
#
# The public-release repo at madrid-rental-vision/ does NOT ship the raw
# dataset. The active dev dataset lives at /Users/tom/School/AI/ai2/.
# We resolve PROCESSED_DIR and IMAGES_DIR against a few candidates so v2
# scripts can run regardless of which repo they were invoked from.
#
# Override via environment variables if needed:
#   MRV_DATA_DIR=/path/to/data/processed
#   MRV_IMAGES_DIR=/path/to/data/raw/images

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "v2"
V2_DATA_DIR = V2_DIR / "data"
V2_MODELS_DIR = V2_DIR / "models"


def _first_existing(env_key: str, filename: str, candidates: list[Path]) -> Path:
    override = os.environ.get(env_key)
    if override:
        p = Path(override)
        if (p / filename).exists():
            return p
        raise FileNotFoundError(
            f"{env_key}={p} does not contain {filename}"
        )
    for c in candidates:
        if (c / filename).exists():
            return c
    raise FileNotFoundError(
        f"Could not locate {filename} in any candidate. Set {env_key} to override. "
        f"Searched: {[str(c) for c in candidates]}"
    )


PROCESSED_DIR = _first_existing(
    "MRV_DATA_DIR",
    "listings_clean.csv",
    [
        PROJECT_ROOT / "data" / "processed",
        Path("/Users/tom/School/AI/ai2/data/processed"),
    ],
)

try:
    IMAGES_DIR = _first_existing(
        "MRV_IMAGES_DIR",
        # use the directory itself as the "marker" — any subfolder of images/
        # would do; we just check the path exists.
        ".",
        [
            PROJECT_ROOT / "data" / "raw" / "images",
            Path("/Users/tom/School/AI/ai2/data/raw/images"),
        ],
    )
except FileNotFoundError:
    # images may legitimately not be present (e.g. when only doing CV over
    # precomputed embeddings). Don't raise at import time.
    IMAGES_DIR = None
