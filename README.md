# madrid-rental-vision

I wanted to see whether a photo of a flat tells you how much it rents for, on top of the obvious stuff (size, zone, bedrooms). It does. Photos add about +0.05 R² over the tabular baseline, mostly in expensive neighborhoods where listings actually look different from each other.

Dataset isn't in this repo (coursework + licensing). Code, trained models, evaluation results, and the full product layer (Chrome extension + Next.js dashboard) are. Bring your own listings CSV and everything re-runs.

## Results

5-fold cross-validation (mean ± std across folds, N = 1,425 listings):

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Gradient Boosting (tabular only) | 0.787 ± 0.019 | €471 ± 48 | 18.4% ± 1.2% |
| + text embeddings | 0.812 ± 0.032 | €436 ± 57 | 17.3% ± 2.0% |
| + frozen ResNet-50 embeddings | 0.833 ± 0.029 | €412 ± 52 | 16.1% ± 1.6% |
| + fine-tuned ResNet-50 | 0.835 ± 0.025 | €409 ± 44 | 15.8% ± 1.3% |
| **+ text + fine-tuned image** | **0.838 ± 0.025** | **€411 ± 41** | **16.0% ± 1.4%** |

**Images add +0.05 R²** on top of tabular features. The fine-tuned vs frozen ResNet gap is essentially zero under CV (+0.002 R²), fine-tuning cost 15 minutes of compute and bought nothing generalizable at this sample size. The photos win comes from ImageNet features alone. Most of the improvement concentrates in expensive neighborhoods (Salamanca, Chamberí) where there's more visual variation between listings.

> **Why these numbers are lower than our first report:** we originally reported R² = 0.85 on a single 70/15/15 split. A technical review surfaced two methodology bugs that were inflating the number: (1) listing dedup ran only on the URL field, so the same property re-listed under a new ID could straddle train/val/test. We now also dedup on `(price, sqft, rooms, bathrooms, location)` and collapsed **46 re-listings**. (2) The fine-tune script and the downstream dataset each called `train_test_split` independently on differently-ordered DataFrames, so some listings the ResNet was fine-tuned on ended up in the gradient-boosting test set. We now write a single `data/processed/splits.json` manifest and both scripts read from it. Moving from a single-split headline to 5-fold CV also shifted the number slightly: single-split reports R² = 0.823 / MAE = €457 (see `models/results.json`), CV reports R² = 0.838 ± 0.025 / MAE = €411 ± €41 (see `models/cv_results.json`). The CV numbers are the honest headline.

## Setup

```bash
git clone https://github.com/<your-username>/madrid-rental-vision.git
cd madrid-rental-vision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. You will need your own listings dataset to re-run the full pipeline (schema below). Pretrained model weights are not shipped; the `models/` dir has config and history JSONs only.

### Expected dataset schema

Place a CSV at `data/processed/listings_clean.csv` with at least:

| column | type | notes |
|---|---|---|
| `listing_id` | int | unique per listing |
| `price` | float | monthly rent in EUR |
| `sqft_m2` | float | floor area |
| `rooms`, `bathrooms` | int | |
| `zone` | str | one of the Madrid zones in `src/data/neighborhoods.py` |
| `description` | str | listing copy |
| `image_urls` | JSON list | per-listing image URLs |
| `num_images`, `floor_num`, `elevator`, `ac`, `terrace`, `furnished`, `heating`, `exterior`, `parking`, `storage` | mixed | optional extras used by the GB model |

Images should be downloaded to `data/raw/images/<listing_id>/<idx>.jpg`.

## Pipeline

Run in order. Each step depends on the previous.

### 1. Build the shared split manifest

Generates `data/processed/splits.json` tagging every `listing_id` as train / val / test (70/15/15, seed 42). Both the fine-tune and the gradient-boosting scripts read this manifest so the ResNet never sees a listing that lands in the downstream GB test set.

```bash
python -m src.data.make_splits
```

### 2. Extract image embeddings (frozen ResNet-50)

Passes every image through a pretrained ResNet-50 (ImageNet) and saves a 2048-dim vector per listing. ~3 min on Apple Silicon, longer on CPU.

```bash
python -m src.vision.extract_embeddings
```

### 3. Fine-tune ResNet-50

Trains `layer4` + a regression head to predict log-rent from individual photos, using the `splits.json` manifest. ~20 min on Apple Silicon.

```bash
python -m src.vision.finetune
```

### 4. Extract fine-tuned embeddings

Same as step 2 but using the fine-tuned model. ~3 min.

```bash
python -m src.vision.extract_finetuned_embeddings
```

### 5. Extract text embeddings

Runs listing descriptions through a multilingual sentence transformer (384-dim). Almost instant.

```bash
python -m src.vision.extract_text_embeddings
```

### 6. Train all models

Trains Ridge, Gradient Boosting (several variants), and Neural Net models and prints comparative results. ~5 min.

```bash
python -m src.models.train
```

### 6b. Cross-validate (optional, recommended)

Refits every GB model under 5-fold CV so the headline isn't a single-seed draw. ~3 min. Saves `models/cv_results.json`.

```bash
python -m src.models.train_cv
```

### 7. Predictions (CLI)

```bash
# predict from an existing listing in the dataset
python -m src.models.inference --listing 101580197

# predict from custom inputs
python -m src.models.inference --sqft 80 --rooms 2 --bathrooms 1 --zone Centro

# demo mode, runs 3 examples (cheap, mid, expensive)
python -m src.models.inference
```

### 8. Analysis figures (optional)

Pre-computed figures live in `notebooks/figures/`. To regenerate:

```bash
python notebooks/01_eda.py
python notebooks/02_image_clusters.py
python notebooks/03_evaluation.py
python notebooks/04_expensive_images.py
```

## Project structure

```
src/
  data/            cleaning, zone mapping, split manifest
  vision/          ResNet embeddings, fine-tuning, text embeddings
  models/          dataset prep, model architectures, training, inference
  api/             FastAPI backend for the dashboard + browser extension
notebooks/         EDA, clustering, evaluation plots
extension/         Chrome (Manifest V3) browser extension
web/               Next.js frontend dashboard
data/processed/    embedding indexes, splits.json, feature-aggregate CSVs
models/            trained weights (gitignored) + results.json + cv_results.json
```

## How it works

Listing photos pass through ResNet-50 (pretrained on ImageNet, then fine-tuned on our data) producing a 2048-dim embedding per image. Mean-pool across all photos in a listing. PCA to 50 dims, concatenate with tabular features (sqm, rooms, zone, etc) and feed into Gradient Boosting. Same pipeline with descriptions through a multilingual sentence transformer (384-dim, PCA to 30).

We tried neural nets for the regression and they overfit with ~1,000 training samples. Gradient Boosting handles the high-dim embeddings much better at this scale.

## Known limitations

- **Listing-level dedup is conservative, not exhaustive.** We catch exact feature-tuple matches (46 re-listings collapsed). Near-duplicates with slightly different text still slip through.
- **"Combined effect" in the feature breakdown is a remainder, not a proper interaction.** The feature decomposition computes `full − tabular − text_delta − photos_delta` and labels it "combined effect." For a gradient-boosting ensemble this mixes true interaction with different-subsample noise across separately-fit ablation models.
- **Per-photo score is a model activation, not a rent figure.** The fine-tuned ResNet's regression head was trained on per-image log-rent where all photos in a listing share one target. Per-image output lands in the rent distribution but doesn't represent the rent contribution of any single photo. The UI shows rank within listing, not absolute €.
- **Neural network ablation does not converge** at this dataset size with our hyperparameters (reported R² = −1.4 / −5.0 in `results.json`). Kept for ablation honesty; not a serious baseline.
- **Fine-tune gain vanishes under CV.** Single-split showed fine-tuned beating frozen ResNet by +0.010 R²; 5-fold CV shows +0.002. Fine-tuning taught us how fine-tuning works, but didn't actually move the headline. Frozen ImageNet ResNet-50 is enough at N=1,425.

## Potential improvements

- **Attention pooling instead of mean pooling.** Let the model learn which photos matter most for predicting price.
- **CLIP instead of ResNet.** Trained on image-text pairs so it already has some understanding of concepts like "luxury" or "modern." Could give better embeddings out of the box and enable zero-shot room-type classification.
- **Per-room-type embeddings.** Right now we just average all images together. A kitchen photo and a bathroom photo get mixed into one vector.
- **Hyperparameter tuning.** We never tuned the gradient boosting (500 trees, max_depth=4, lr=0.05). Grid search or Bayesian optimization would probably squeeze out a couple of points.
- **Prediction intervals.** MC Dropout or deep ensembles would give per-listing confidence instead of a constant ±MAE band.
- **More cities.** Barcelona, Valencia to test generalization; temporal re-scrape to get days-on-market signal.

---

# CasaIntel: browser extension and web app

After training the base model we built a small product layer on top: a FastAPI backend that serves predictions, a Next.js dashboard, and a Chrome extension that overlays peer-expected rent on real Madrid rental listings as you browse.

## What it does

**Chrome extension** (`extension/`): runs on Madrid rental listing pages. It reads the listing features from the page DOM (no scraping, piggybacking on whatever page the user is already viewing), calls the local backend, and injects:

- A small neutral badge on every search-result card (tabular-only peer estimate).
- A full panel on each listing detail page (full-model prediction ± MAE, feature-by-feature breakdown, and a bottom-line "to lift the prediction, improve X" action line).
- Colored overlays on each gallery photo (`HELPS`, `WEAK`, `NEUTRAL` + rank within the listing) so you can see which photos are pulling the listing up or dragging it down.

**Web app** (`web/`, Next.js): an agency-framed dashboard with three workflows:

1. **Portfolio.** A mixed demo portfolio (6 flagged + 6 healthy listings seeded on first visit). Aggregate stats: managed rent, commission @ 6%, peer-rent gap, potential commission upside × realization-rate slider.
2. **Listing detail / Action Plan.** Photo scorecard with weakest/strongest photos ringed, feature-by-feature breakdown ("Why this number"), action plan bullets (re-shoot / re-price / rewrite), commission upside panel.
3. **Intake.** Paste a listing ID or URL, see its baseline, drop new candidate photos. Uses **replace-worst-N** semantics: your uploads swap out the N weakest existing photos (ranked by precomputed per-photo scores), then the model re-predicts. Honest confidence band: any predicted change smaller than MAE is labeled "inside noise."

## Dual utility

Same tool, both sides of the transaction:

- **Renter hunting a deal.** Blue "below peer" badge flags underpriced listings; red "above peer" flags overpriced ones to skip or negotiate down.
- **Agent / landlord diagnosing weakness.** The feature breakdown shows exactly which block (tabular / text / photos) is dragging predicted rent down, and the per-photo overlays show which specific images need re-shooting.

## How to run locally

Backend (port 8000):

```bash
source venv/bin/activate
uvicorn src.api.app:app --port 8000
```

Frontend (port 3000):

```bash
cd web
npm install       # first time only
npm run dev
# → http://localhost:3000
```

Chrome extension (load unpacked):

1. `chrome://extensions/` → toggle **Developer mode** on.
2. Click **Load unpacked** → select the `extension/` folder.
3. Browse to any supported Madrid rental listing URL. Badges appear automatically.

The extension calls `http://127.0.0.1:8000/predict-live`, so the backend must be running locally.

## Extension API in two sentences

`POST /predict-live` takes tabular + optional text/image description of a listing and returns a peer-expected rent with feature-by-feature decomposition. If you pass a `listing_id` that's in the precomputed dataset, you get an instant cache hit; otherwise the backend downloads the images, runs the ResNet + sentence-transformer + gradient-boosting pipeline live, and returns the same structure in a few seconds.

## Caveats for the product layer

- Search-card badges are tabular-only and deliberately don't show strong over/under-priced labels: tabular predictions are too noisy on outlier listings (penthouses, large terraces) to be confidently directional. The colored verdict only appears on the detail page where the full model (with photos + text) has enough information.
- The "commission upside" number on the portfolio assumes a realization-rate slider (default 25%). It's a pitch framing, not a production forecast.
- Per-photo `HELPS / HURTS / NEUTRAL` overlays are rank-based within the listing. They reflect the model's individual-photo activation, not a counterfactual €-contribution to the listing's rent.
