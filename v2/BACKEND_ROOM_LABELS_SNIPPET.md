# Wire room labels into the local FastAPI backend

The `v2/precompute_room_labels.py` script produces
`v2/data/photo_room_labels.csv` — 143,458 (listing_id, photo_idx, room_label,
room_confidence) rows. To make the room labels show up in the extension's
gallery overlay, the local backend needs to attach `room_label` to each entry
of `per_photo_impact` in the `/predict-live` response. Below is the minimal
edit, to be applied to the **local-only** `src/api/app.py`.

## 1. Load the CSV at backend startup

In the `lifespan` async function (or wherever you load other artifacts), add:

```python
import pandas as pd
from pathlib import Path

# Pull from the v2 data dir (v2 unified CSV ships with the same listings)
ROOM_LABELS_FILE = Path(__file__).resolve().parents[2] / "v2" / "data" / "photo_room_labels.csv"

if ROOM_LABELS_FILE.exists():
    room_df = pd.read_csv(ROOM_LABELS_FILE, dtype={"listing_id": str, "photo_idx": int})
    # Build a lookup: (listing_id, photo_idx) -> room_label
    state["room_labels"] = {
        (row.listing_id, int(row.photo_idx)): row.room_label
        for row in room_df.itertuples()
    }
    print(f"Loaded {len(state['room_labels']):,} room labels")
else:
    state["room_labels"] = {}
    print("Room labels CSV missing -- skipping zero-shot room classification")
```

## 2. Attach to per-photo response

Find where `per_photo_impact` is built (in the cached or live path) and inside
the loop that creates each entry, look up the label. The structure depends on
whether you have `listing_id + photo_idx` in scope at that point:

```python
# Inside the per-photo loop, after creating the PhotoImpact instance:
key = (str(req.listing_id), int(photo_idx))
if key in state["room_labels"]:
    photo_impact.room_label = state["room_labels"][key]
```

## 3. Add the field to the Pydantic model

In the `PhotoImpact` model class, add:

```python
class PhotoImpact(BaseModel):
    image_url: str
    rank_in_listing: int
    score_eur: float
    tone: str
    room_label: Optional[str] = None  # <-- add this line
```

## 4. Test

Reload the FastAPI backend. Hit any cached listing's detail page in the
extension. The per-photo gallery overlay should now read e.g.
`Weak  kitchen  #4 / 22` instead of just `Weak  #4 / 22`.

## How the extension renders this

The extension is already wired: `injectPhotoOverlay` looks for
`impact.room_label` and renders a small lowercase pill next to the tone
label. If the backend doesn't serve the field, the pill simply doesn't
render — graceful degradation.

## What about uncached / live listings?

The precomputed CSV only covers the 6,047 listings in our dataset. For
listings the user pulls up live (mode="full" with image URLs the model has
never seen), there are two options:

1. **Skip room labels for live mode** (current behaviour). The pill just
   doesn't appear. Tone + rank still show.

2. **Compute SigLIP image embeddings on-the-fly + cosine against cached
   text-caption embeddings.** Adds ~2-3 seconds per photo to live inference.
   If you want this, also embed the captions once at startup (see
   `precompute_room_labels.py` for the text-encoding code) and add a forward
   pass through the SigLIP vision encoder for each downloaded photo.

For the demo, option 1 is fine: the cached path is what the audience sees
when you pull up a Madrid listing the model has trained on.
