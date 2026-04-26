# v2: what changed and why

This folder is the rewrite that addresses the review feedback we got on v1.
It lives separately from `src/` so the original v1 pipeline keeps working
unmodified, which lets you compare the two side-by-side.

## TL;DR

| | v1 (original) | v2 (this folder) |
|---|---|---|
| Listings | 1,425 | **6,047** |
| Image encoder | ResNet-50 (ImageNet) | **SigLIP-base** (image-text contrastive) |
| Headline model R² | 0.838 | **0.884** |
| Headline MAE | €411 | **€274** (−33%) |
| Headline RMSE | reported only as MAE | **€507** |
| Q4 luxury bias (calibrated) | not addressed | **−€164** (was −€433 baseline) |

## What the review asked for

1. **Report RMSE alongside MAE.** Done in every CV table.
2. **Report all combinations of {tabular, text, image}.** Full 7-subset ablation grid in `train_cv_full_ablation.py`.
3. **Better way of combining images than mean pooling.** Attention pooling implemented in `attention_pool.py`, evaluated in `train_cv_siglip.py`. Honest result: tied with mean pool on 1,425 listings (not enough data to learn per-photo attention weights well).
4. **Try CLIP / DINO / SigLIP instead of ResNet.** SigLIP-base swapped in via `extract_siglip_embeddings.py`. This was the biggest single win.

## What's in this folder

```
v2/
├── README.md                          this file
├── paths.py                           shared path resolution
├── extract_siglip_embeddings.py       SigLIP-base feature extractor (per-listing + per-photo)
├── extract_text_embeddings_v2.py      multilingual sentence-transformer on the unified dataset
├── attention_pool.py                  additive attention pooling module
├── train_cv_v2.py                     v2 CV grid (12 models, joins on ResNet → 1,425 rows)
├── train_cv_siglip.py                 ResNet vs SigLIP vs SigLIP+attention comparison
├── train_cv_robust.py                 Huber / quantile loss experiments (negative result, kept for honesty)
├── train_cv_full_ablation.py          full {tabular,text,siglip} grid on all 6,047 listings ← headline
├── calibration.py                     post-hoc linear calibration to fix Q4 luxury bias
├── residual_analysis.py               error distribution + per-zone + per-quartile breakdown
├── make_figures.py                    generates v2/figures/*.png from OOF predictions
├── tests/test_attention_pool.py       smoke test for the attention pooling module
├── models/                            CV result JSONs (committed, all <1 MB)
└── figures/                           PNG figures (committed)
```

## Headline numbers (full ablation grid, 6,047 listings, 5-fold CV)

| Model | R² | MAE (€) | RMSE (€) | MAPE (%) |
|---|---|---|---|---|
| ridge_tabular | 0.749 ± 0.009 | 417 ± 13 | 746 ± 88 | 19.2 |
| gb_tabular | 0.802 ± 0.009 | 351 ± 9 | 618 ± 62 | 16.4 |
| gb_text | 0.381 ± 0.022 | 627 ± 25 | 1042 ± 87 | 30.7 |
| gb_siglip | 0.662 ± 0.006 | 467 ± 21 | 803 ± 83 | 22.3 |
| gb_tabular_text | 0.832 ± 0.009 | 326 ± 10 | 593 ± 63 | 15.0 |
| **gb_tabular_siglip** | **0.882 ± 0.007** | **274 ± 8** | **506 ± 71** | **12.6** |
| gb_text_siglip | 0.690 ± 0.004 | 448 ± 21 | 785 ± 84 | 21.1 |
| **gb_tabular_text_siglip** | **0.884 ± 0.007** | **274 ± 8** | **507 ± 72** | **12.6** |

Saved to `v2/models/cv_results_full_ablation.json`.

### What this tells us
- Tabular alone gives R² 0.80. Size, zone, and bedrooms do most of the work.
- **SigLIP adds +0.08 R²** on top of tabular (vs +0.05 for ResNet in v1).
- **Text becomes essentially redundant** once SigLIP is in the mix (0.882 vs 0.884 is within noise). SigLIP was trained on image-text pairs, so it already encodes the semantic content the description was contributing.
- Text alone is the weakest signal (R² 0.38). Descriptions help only if they're combined with structural data.

## Calibration (Q4 luxury bias fix)

The model regresses toward the mean on luxury listings: predicted rent is too low for €4k+ listings. We fit a simple linear correction (`pred_calibrated = a · pred + b`) using leakage-free 5-fold CV.

| Variant | MAE | RMSE | Q1 bias | Q4 bias |
|---|---|---|---|---|
| baseline (no calibration) | €274 | €511 | +€72 | **−€280** |
| linear_euro_cv | €279 | €507 | +€53 | **−€164** (−41%) |
| isotonic_cv | €274 | €512 | +€85 | −€207 |

Linear calibration cuts the luxury under-prediction in half for a trivial cost in global MAE. Saved to `v2/models/calibration_results.json`.

## SigLIP cluster analysis: visual proof the embeddings carry meaning

Run `python v2/cluster_images.py` to reproduce. We sample 5,000 individual
photos from the 143,458 SigLIP per-photo embeddings, run K-Means with
k=10, and project to 2D with UMAP. No supervision: K-Means just looks at
the 768-dim SigLIP vectors.

The clusters separate cleanly by visual concept:

| Cluster | N | Mean rent | Concept |
|---|---|---|---|
| 0 | 535 | €2,114 | Hallways, closets, dressing rooms |
| 1 | 476 | €2,307 | Empty rooms with hardwood floors (post-reform) |
| 2 | 630 | €1,990 | Kitchens |
| 3 | 650 | €2,267 | Bathrooms |
| 4 | 482 | €2,252 | Building exteriors, plazas, street views |
| 5 | 693 | €2,407 | Bedrooms |
| 6 | 677 | €2,317 | Modern living rooms |
| 7 | 413 | €2,064 | Older / eclectic living rooms |
| **8** | **133** | **€3,351** | **Luxury (E&V branded photos, marble, pro lighting)** |
| 9 | 311 | €2,033 | Floor plans, architectural diagrams |

Two takeaways:

1. **Unsupervised room-type discovery.** Kitchens, bathrooms, bedrooms,
   living rooms each form distinct clusters. Even style is split:
   "modern" vs "older" living rooms (C6 vs C7) end up apart. This
   wasn't trained for; it falls out of SigLIP's image-text pretraining.

2. **Cluster 8 is direct visual evidence SigLIP captures price.** Only
   133 of 5,000 photos (2.7%), but mean rent €3,351 vs €2,200 average,
   a 50% premium. The cluster picked up the *visual signature* of
   luxury photography: minimalist white spaces, professional lighting,
   watermarks from premium agencies (Engel & Völkers visible in many).
   Plain ResNet on ImageNet wouldn't group these. SigLIP did because
   its training corpus included captions like "luxury apartment" /
   "professional photo" alongside such images.

Saved figures:
- `figures/siglip_umap_clusters.png` — UMAP 2D, colored by cluster
- `figures/siglip_umap_by_price.png` — same UMAP, colored by log rent
- `figures/siglip_cluster_<k>.png` — 12-photo grid per cluster

## Methodology bugs from v1 that are now fixed

1. **Listing-level dedup ran only on URL** in v1, so the same property re-listed under a new ID could straddle train/test. v2's `clean_and_merge.py` dedups on `(price, sqft, rooms, bathrooms, location)` tuples too. ~46 re-listings collapsed in the original 1,425; ~567 collapsed in the new 4,708 net-new listings.
2. **Independent `train_test_split`s** in v1's fine-tune and CV scripts created a leak. v2 uses a single `splits.json` manifest (already fixed in v1's later cleanup, preserved here).

## How to reproduce

The dataset isn't shipped (coursework + licensing, same stance as v1). Bring your own listings CSV at `data/processed/listings_clean.csv` (schema in the root README), do your own scrape for any expansion, then:

```bash
# 1. SigLIP image embeddings (~30 min on Apple Silicon MPS)
python -m v2.extract_siglip_embeddings

# 2. Text embeddings on the unified set (~5 min)
python v2/extract_text_embeddings_v2.py

# 3. Full ablation grid (~5 min)
python v2/train_cv_full_ablation.py

# 4. Calibration (~30 sec)
python v2/calibration.py

# 5. Diagnostic figures
python v2/make_figures.py
```

## Limitations we're explicit about

- **Q4 luxury MAE is still €541 baseline / €528 calibrated.** Calibration fixed the systematic bias (model now predicts the right average for €4k+ listings) but not the variance: individual luxury predictions can still be off by €500+. Need more luxury training data (or a separate luxury head) to fix this.
- **Attention pooling didn't beat mean pooling at this scale.** Negative result from `train_cv_siglip.py`, preserved for honesty, not used in the headline.
- **`train_cv_v2.py` and `train_cv_siglip.py` are still capped at 1,425 listings** because they inner-join on the ResNet embeddings (which we never re-extracted on the new listings). The ablation we trust for the 6,047 headline is `train_cv_full_ablation.py`. The other two CV scripts are for v1-comparison purposes.
- **Robust loss experiments** (`train_cv_robust.py`). Huber and quantile losses didn't beat squared error. Kept the script + results for the writeup but no production use.
