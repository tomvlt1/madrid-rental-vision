# Madrid Rental Vision — Presentation Script
**Total: 10 minutes** · 6 presenters · 5 sections

| # | Section | Speaker(s) | Time |
|---|---|---|---|
| 1 | Product & Value | Carlo | 2:00 |
| 2 | Market | Lev | 2:00 |
| 3 | Viability | Meropi (+ Ana) | 2:30 |
| 4 | Architecture | Javi / Meropi (1:30) + Tom (1:00) | 2:30 |
| 5 | Demo | Tom | 1:00 |

---

## 1. Product & Value — Carlo (2:00)

> **Hook (0:00–0:15)**
>
> "Imagine two people opening the same Idealista listing. A tenant who doesn't know whether €1,800 a month is fair for what they're seeing. And a landlord whose listing has been sitting for three weeks without a single visit, and they have no idea why. We built one tool that answers both."

> **The problem (0:15–0:50)**
>
> "The real problem isn't price estimation — Idealista already shows the price. It's information asymmetry. Existing valuation tools, including Idealista's own, only use structured data: square metres, rooms, zone. They ignore the two things that actually shape a tenant's decision: **the photos and the description**. A €2,000 flat with bad photos and a €2,000 flat with great photos rent at completely different speeds — and current tools cannot tell you which is which."

> **What we built (0:50–1:30)**
>
> "We built a multimodal model that reads a listing the way a human does. Square metres, rooms, zone, the description, and every photo — all going into one prediction. From that, two outputs:
>
> One: a **peer-expected rent**. Whether the listing is priced above or below what comparable properties are renting for.
>
> Two: a **per-photo diagnostic** — for each image in the gallery, the model labels it as helping the prediction, hurting it, or neutral. So a landlord doesn't just learn that their listing is overpriced; they learn that **photo number four is dragging the price down by €120 a month**."

> **Distribution (1:30–1:50)**
>
> "And we deliver this as a Chrome extension, not a website. The reason is simple: rental decisions get made *while* you're browsing listings. If we ask the user to copy a URL into a separate tool, they'll never use it. The extension injects directly into the page they're already on."

> **Hand-off (1:50–2:00)**
>
> "So that's the product. The question is whether the market is big enough to matter. Lev?"

---

## 2. Market — Lev (2:00)

> **Hook (0:00–0:15)**
>
> "Madrid has more than **11,000 active rental listings** on Idealista right now. That's the opportunity, and that's the noise."

> **Two sides of the same trade (0:15–0:55)**
>
> "Two customer segments, both feeling the pain.
>
> **Tenants**: comparing 30, 50, sometimes 100 listings in a few days, mostly online. Even a €100-a-month difference compounds to €1,200 over a one-year contract. The cost of getting it wrong is real.
>
> **Landlords and agencies**: pricing too high means weeks of vacancy. Pricing too low leaves money on the table. Most private landlords price on intuition, comparing to a handful of nearby listings. Agencies have better tools but still rely on tabular comps."

> **Competitive landscape (0:55–1:35)**
>
> "There are competitors, and we're honest about that.
>
> **Idealista's own AVM** estimates rent based on tabular data. **Engel & Völkers** does agency-internal valuations. **Zillow's Zestimate** in the US is the most advanced — it does use listing photos in its neural network — but it's a US product tied to a single platform.
>
> What no one offers in Spain is a **multimodal valuation that lives in the user's browsing flow**. We're not competing on accuracy alone — we're competing on **where** the prediction shows up."

> **Why now / why Madrid (1:35–1:50)**
>
> "Spain's real estate sector is digitalising fast. PwC reported in 2025 that valuation tooling is one of the fastest-growing technology categories in the industry. Madrid is the obvious starting point because the market is dense, transparent, and overwhelmingly online."

> **Hand-off (1:50–2:00)**
>
> "But can we actually build this — and is the math viable? Over to Meropi."

---

## 3. Viability — Meropi (2:30)

> **Hook (0:00–0:15)**
>
> "Three lenses on viability: can we build it, can we afford it, and is anything legally stopping us. Let me take them one at a time."

> **Technical viability (0:15–0:50)**
>
> "We've built and tested the MVP, so technical feasibility isn't theoretical anymore. But the validation is also that **comparable systems already work at scale**: Zillow's Zestimate is a multimodal model on US listings, Redfin's home value estimator does something similar. The architecture pattern is proven. What we're doing differently is the surface — Chrome extension instead of website — and the geography — focused on Madrid where we can be more accurate than a generalist tool."

> **Economic viability (0:50–1:35)**
>
> "We costed the MVP at roughly **€83,000 for a 4-to-6-month build**.
>
> Most of that is people: two ML engineers, one data engineer, one frontend developer. About **€74,000 in salaries** for the build period.
>
> Compute and infrastructure are surprisingly small — roughly **€3,500 per year** for cloud hosting, GPU training, and storage combined. Modern ML doesn't need a server farm.
>
> The remaining €5,000-ish is miscellaneous: legal review, the Chrome Web Store fee, domain, basic marketing.
>
> The headline is that this is a **software-economics business**: low fixed cost to build, near-zero marginal cost per user."

> **Legal & risk (1:35–2:15)**
>
> "Three real risks.
>
> **One: data licensing.** Idealista's terms of service don't allow scraping for commercial purposes. To go to production, we'd need a data partnership — either with Idealista directly, or with one of their competitors like Fotocasa.
>
> **Two: GDPR.** Listing descriptions sometimes include landlord phone numbers and names — personal data. Production pipeline needs to strip PII before training.
>
> **Three: generalisation.** The model is trained on Madrid only. Extending to Barcelona or Valencia is technically straightforward but commercially requires retraining and a fresh validation dataset for each city."

> **Revenue model (2:15–2:25)**
>
> "B2B subscription is the primary driver — agencies pay per portfolio. The Chrome extension stays freemium for tenants and acts as the acquisition channel. Same playbook Compass and Reonomy used in the US."

> **Hand-off (2:25–2:30)**
>
> "And now — the model itself. Javi, take it away."

---

## 4. Architecture — Javi/Meropi (1:30) + Tom (1:00) = 2:30 total

> **One-line frame for the section** (anyone, optional):
> *"We took a journey: scraping, exploration, dimensionality reduction, a first attempt that didn't fully work, a swap that did, and a rigorous ablation to prove it."*

### 4a. Javi/Meropi — 1:30

> **Beat 1 · Scraping (0:00–0:15)**
>
> "Before any modelling, we needed real data. We scraped **6,047 Madrid rental listings** from Idealista — full structured features, every description, every photo. That alone took weeks: rotating proxies, anti-bot evasion, deduplication on price-size-rooms-bathrooms-zone, because the same flat re-listed under a new ID would otherwise leak across our train and test splits."

> **Beat 2 · Data exploration (0:15–0:35)**
>
> "Then we looked at what we had. Madrid rents are heavily skewed and **bimodal** — one cluster of studios and one-beds around €1,200, another of larger flats around €2,800 — so we model **log-rent**, not raw rent. Among tabular features, **zone is by far the strongest predictor**: median rent in Salamanca-Retiro is **2.1 times** the median in the southern peripheries. We have a median of **22 photos per listing**, which means more than **143,000 photos** flow through our image encoder."

> **Beat 3 · Features and PCA (0:35–0:55)**
>
> "Three feature blocks. **Tabular**: sqm, rooms, bathrooms, eight zones, eight amenity flags. **Text**: each Spanish description through MiniLM, a multilingual sentence transformer, gives us 384 dimensions. **Image**: each photo through a deep encoder, 768 dimensions per photo, mean-pooled per listing.
>
> That's a lot of dimensions for 6,000 listings. So we apply **PCA per block** — image down to 50, text down to 30 — keeping the signal, dropping the noise. Then concatenate everything and train."

> **Beat 4 · The first attempt: ResNet + NN (0:55–1:25)**
>
> "Our first version used **ResNet-50**, the standard ImageNet encoder, with a **neural network** as the regressor. It overfit. With around 1,000 effective training samples after splits, the network couldn't generalise on high-dimensional embeddings, and we plateaued at R² around 0.80.
>
> Two changes followed. We swapped the neural net for **Gradient Boosting**, which handles high-dim features much better at small N. That helped. But the bigger move was the encoder."

> **Hand-off (1:25–1:30)**
>
> "And that's where Tom takes over."

### 4b. Tom — 1:00

> **Beat 1 · SigLIP (0:00–0:20)**
>
> "We replaced ResNet with **SigLIP** — Google's image-text contrastive encoder. ResNet was trained to classify ImageNet objects, dogs and birds and cars. SigLIP was trained to match images with their captions. For a rental listing, where the photo is essentially captioned by the description it lives next to, that pre-training is much closer to our task.
>
> The swap moved us from **R² 0.80 to 0.88, MAE from €351 down to €274**. **Seventy-seven euros per listing of accuracy, just from the encoder change.**"

> **Beat 2 · Ablation (0:20–0:40)**
>
> "We then ran every non-empty combination of tabular, text, and image — seven configurations total — to make sure SigLIP was actually doing the work. The headline finding: **once SigLIP is in the model, the text description becomes almost redundant**. The image-text contrastive pre-training already encodes the semantic content the description was contributing. That surprised us, and it's why we kept the full ablation table in the report."

> **Beat 3 · Honesty (0:40–0:55)**
>
> "Two things to flag. We tried **attention pooling** over per-photo embeddings — it tied with mean-pool, a negative result that we kept in. And our first version had **two methodology bugs**: a deduplication leak and a train-test split mismatch. Fixing both moved us from a single-split R² of 0.85 down to a clean five-fold CV R² of 0.88. **The lower number is the honest one.**"

> **Transition (0:55–1:00)**
>
> "Now let me show you what this looks like to someone actually using it."

---

## 5. Demo — Tom (1:00)

> **Setup (0:00–0:05)**
>
> *(switch to browser, real Idealista page already loaded)*
>
> "This is a real listing on Idealista. The extension is loaded. Watch the page."

> **Search-results badges (0:05–0:25)**
>
> *(navigate to search results)*
>
> "On the search page, every card gets a soft peer-estimate badge — blue if the listing is below what we'd expect for a comparable property, red if above. A tenant scans the page once and sees what's worth clicking on. Two seconds, no spreadsheet."

> **Detail panel (0:25–0:50)**
>
> *(click into a listing)*
>
> "Click into a listing and the full panel injects on the right. Predicted rent with our error band, asking-vs-peer verdict, and a feature-by-feature breakdown showing what the tabular features contributed, what the description contributed, and what the photos contributed.
>
> Below that — and this is the part landlords care about — every photo in the gallery gets a label. **Helps, weak, or neutral**, ranked within the listing. So a landlord can see at a glance: replace photo three, reshoot the kitchen, and you'd add roughly this much to your peer-expected rent."

> **Close (0:50–1:00)**
>
> "Same model, two surfaces. A tenant saves money. A landlord moves their listing faster. Thanks for listening — happy to take questions."

---

## Rehearsal checklist

- [ ] Run end-to-end with a stopwatch. If you're at 10:30, cut from viability (legal beat) and architecture (PCA beat), each by ~10s. **Don't talk faster — cut.**
- [ ] Rehearse every hand-off line three times. The seam between sections is where group presentations fall apart.
- [ ] Have a **30-second screen recording** of the extension working, ready to play if live demo fails.
- [ ] Pre-load the Idealista listing in the browser **before** the talk starts. Don't navigate live.
- [ ] Memorise the three numbers that have to land: **6,047 listings**, **R² 0.88 / MAE €274**, **€77 per listing accuracy from the SigLIP swap**.

## Numbers cheat-sheet (in case anyone gets asked a question)

| | Value |
|---|---|
| Total listings | 6,047 |
| Madrid zones | 9 |
| Median rent | €1,500 / month |
| Median size | 66 m² |
| Median photos per listing | 22 |
| Total photos through SigLIP | 143,462 |
| Image encoder | SigLIP-base-patch16-224 (768-dim) |
| Text encoder | paraphrase-multilingual-MiniLM-L12-v2 (384-dim) |
| PCA dims | image 50, text 30 |
| Regressor | Gradient Boosting (500 trees, depth 4, lr 0.05) |
| **Best R² (5-fold CV)** | **0.884 ± 0.007** |
| **Best MAE** | **€274 ± €8** |
| **Best RMSE** | **€507 ± €72** |
| Tabular-only baseline R² | 0.80 |
| MVP cost | ~€83,000 |
