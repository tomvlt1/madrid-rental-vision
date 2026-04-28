# EDA brief — Madrid rental dataset

For the **Data Exploration** beat in the architecture section (~20 seconds).
Dataset is the cleaned, deduplicated v2 set (6,047 listings, scraped from
Idealista, deduped on price-size-rooms-bathrooms-zone).

---

## Headline numbers (memorise 3 of these)

- **6,047 listings** across **9 Madrid zones**
- **Median rent: €1,500/month** (10th percentile €1,000, 90th €3,150)
- **Median size: 66 m²**
- **143,462 photos total**, median **22 per listing** — every photo went through SigLIP
- **Median description: ~186 words** (most in Spanish)
- **Priciest zone**: Salamanca-Retiro (median €2,500); **cheapest**: Sur-Sureste (median €1,200)
- **Largest zone by listing count**: Centro (1406 listings)

---

## What the graphs show (one line each)

1. **`01_price_distribution.png`** — Raw rent is heavily right-skewed (€500 studios alongside €5,000+ penthouses), so we model **log-rent** instead. The distribution is also visibly **bimodal** — one cluster around €1,000–1,500 (studios + 1-bed), another around €2,500–3,000 (2-bed and up). Log-transforming compresses the tail and makes the gradient-boosting target much better-behaved, but the bimodality remains and is why zone + size end up doing so much of the lifting.

2. **`02_price_by_zone.png`** — Zone is the single strongest tabular predictor. Median rent in Salamanca-Retiro is roughly **2.1x** the median in Sur-Sureste. This is why one-hot zone encoding is in our tabular feature block.

3. **`03_photos_per_listing.png`** — Median 22 photos per listing. Total 143,462 photos went through the SigLIP encoder. Two clear modes — most listings have 10–25 photos, but a noticeable spike at 60 (the cap) suggests luxury listings consistently max out the photo slots.

4. **`04_correlations.png`** — Size (sqft_m2) is the strongest single numeric predictor of rent at **+0.75 correlation**, followed by bathrooms at **+0.70**. Bathrooms and size are themselves correlated at 0.80 (multicollinearity), which is why GB can substitute one for the other. **num_images correlates +0.33 with price** — a mild but real signal that expensive listings show off more photos. Floor number has near-zero correlation with rent in Madrid.

5. **`05_price_vs_size.png`** — Price scales roughly linearly with size up to about 150 m². Above that, the relationship flattens and the variance explodes — this is the **luxury tail** that produces the Q4 bias we have to fix with calibration later.

---

## Suggested talking points (~20s for the beat)

> "Madrid rents are heavily skewed — studios at €500 sit next to penthouses above €5,000 — so we model **log-rent**, not raw rent. Among tabular features, **zone is by far the strongest predictor**: median rent in Salamanca-Retiro is roughly **2.1x** the median in Sur-Sureste. We have a median of **22 photos per listing**, so over **143,462 photos** total flow through our image encoder. And there's a long luxury tail above 200 m² — that's the bias we have to calibrate for later."

---

## Hand-off line into PCA beat

> "All these features — tabular, text, image — give us hundreds of dimensions per listing. Which is why the next step is dimensionality reduction."

Files in this folder:
- 01_price_distribution.png
- 02_price_by_zone.png
- 03_photos_per_listing.png
- 04_correlations.png
- 05_price_vs_size.png
- eda_brief.md (this file)
