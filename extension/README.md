# CasaIntel browser extension

Injects peer-expected rent badges and detail panels onto Madrid rental listing pages as you browse. Reads the DOM of whatever page you're on; no scraping, no background requests.

## Install (Chrome / Edge / Brave)

1. Open `chrome://extensions/` and enable **Developer mode** (top right).
2. Click **Load unpacked** and select this `extension/` folder.
3. Navigate to a supported Madrid rental listing URL. The frontend loads; badges populate once the backend is reachable at `http://127.0.0.1:8000`.

> **Note on the backend.** The FastAPI server that powers `/predict-live` ships separately from this public repo (it depends on the v1 fine-tuned ResNet artifacts kept on the author's machine). Without it running, the extension will load but its API calls will fail. Demo screenshots and a video walkthrough are in the project report.

## What it does

- **Search results.** Reads each card (price, size, rooms, location) and calls `/predict-live` in *tabular* mode. Instant soft badges showing peer estimate.
- **Detail pages.** Reads full listing features + image URLs + description, calls `/predict-live` in *full* mode. Backend runs the SigLIP + sentence-transformer + gradient-boosting pipeline and returns a feature-by-feature breakdown.
- **Gallery overlays.** Each listing photo gets a STRONG / WEAK / NEUTRAL tag based on its per-image model activation (rank within the listing).
- **What-if simulator.** Toggle amenity buttons (AC, terrace, parking, etc.) and a size slider to see how peer-expected rent changes. Each toggle triggers a debounced `/predict-live` call in tabular mode. Instant feedback, no backend image work.
- **Negotiation message generator.** Renders only when the asking price is 5–25% above the peer estimate. Picks one of two Spanish templates ("slight" or "notable" tier), pre-fills the listing's specifics, and offers a copy-to-clipboard textarea you can edit before sending. Above 25%, the panel shows an outlier warning instead of a message — likely a typo or luxury tier the model can't see.
- **Saved listings.** A floating 📌 launcher (always visible) opens a drawer of your pinned listings. Each entry stores a snapshot history (asking + predicted at each visit), an open-listing link, an on-demand re-evaluate button, and an unpin button. Persisted via `chrome.storage.local`. Visiting a pinned listing's page automatically appends a new history snapshot, so the timeline grows organically.

## Diagnosis logic

The detail panel labels the listing based on the gap between asking price and model prediction:

- **Priced on peer.** Gap smaller than the model's typical ±€274 error. Treated as noise.
- **Above peer.** Asking higher than predicted by more than MAE.
- **Below peer.** Asking lower than predicted by more than MAE.

## Security / privacy

- All processing is local (your machine's backend + your browser). No data leaves your machine.
- The backend downloads listing images server-side only when you request a full-mode prediction for a listing you're viewing.
- The extension only activates on the configured URL patterns.

## Limitations

- Madrid only. The model has never seen Barcelona, Valencia, or other cities.
- Model MAE is €274. Predicted deltas narrower than that are labeled inside-noise.
- Out-of-distribution listings (studios < 20m², mansions > 500m²) get less reliable predictions.
- Saved listings re-evaluate against the *backend's cache*, not against a fresh Idealista scrape — actual asking-price changes only update when the user re-opens the listing page (which auto-appends a new history snapshot).
- DOM selectors can drift when listing sites change their markup. If badges stop appearing, check the browser devtools console for errors and update the selectors in `src/content.js`.
