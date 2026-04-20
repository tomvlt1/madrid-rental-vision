# CasaIntel browser extension

Injects peer-expected rent badges and detail panels onto Madrid rental listing pages as you browse. Reads the DOM of whatever page you're on; no scraping, no background requests.

## Install (Chrome / Edge / Brave)

1. Make sure the backend is running:
   ```
   uvicorn src.api.app:app --port 8000
   ```
2. Open `chrome://extensions/` and enable **Developer mode** (top right).
3. Click **Load unpacked** and select this `extension/` folder.
4. Navigate to a supported Madrid rental listing URL. Badges appear automatically.

## What it does

- **Search results.** Reads each card (price, size, rooms, location) and calls `/predict-live` in *tabular* mode. Instant soft badges showing peer estimate.
- **Detail pages.** Reads full listing features + image URLs + description, calls `/predict-live` in *full* mode. Backend runs the fine-tuned ResNet + sentence-transformer + gradient-boosting pipeline and returns a feature-by-feature breakdown.
- **Gallery overlays.** Each listing photo gets a HELPS / HURTS / NEUTRAL tag based on its per-image model activation (rank within the listing).

## Diagnosis logic

The detail panel labels the listing based on the gap between asking price and model prediction:

- **Priced on peer.** Gap smaller than the model's typical ±€457 error. Treated as noise.
- **Above peer.** Asking higher than predicted by more than MAE.
- **Below peer.** Asking lower than predicted by more than MAE.

## Security / privacy

- All processing is local (your machine's backend + your browser). No data leaves your machine.
- The backend downloads listing images server-side only when you request a full-mode prediction for a listing you're viewing.
- The extension only activates on the configured URL patterns.

## Limitations

- Madrid only. The model has never seen Barcelona, Valencia, or other cities.
- Model MAE is €457. Predicted deltas narrower than that are labeled inside-noise.
- Out-of-distribution listings (studios < 20m², mansions > 500m²) get less reliable predictions.
- DOM selectors can drift when listing sites change their markup. If badges stop appearing, check the browser devtools console for errors and update the selectors in `src/content.js`.
