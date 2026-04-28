# Architecture section — full script
**Total: 2:30** · split between two presenters · order: scraping → data exploration → PCA → ResNet/NN → SigLIP → ablation → extension

| Speaker | Time | Beats |
|---|---|---|
| Javi / Meropi | 1:30 | scraping → EDA → PCA → ResNet/NN |
| Tom | 1:00 | SigLIP → ablation → extension transition |

> **The single line you both want them to remember:**
> *"Switching from ResNet to SigLIP added more accuracy than text, calibration, and tuning combined — €77 per listing, just from the encoder."*

---

## Slide

A **horizontal timeline** stays on screen the whole 2:30:

```
Scraping → EDA → PCA → ResNet/NN → SigLIP → Ablation → Extension
```

Each stage **highlights as the speaker reaches it**. No bullets, no equations. The audience tracks where you are in the story by which stage is lit.

---

## Javi / Meropi — 1:30

### Beat 1 · Scraping (0:00–0:15)

> "Before any modelling we needed real data. We scraped **6,047 Madrid rental listings** from Idealista — full structured features, every description, every photo. That alone took weeks: rotating proxies, anti-bot evasion, and deduplication on price-size-rooms-bathrooms-zone, because the same flat re-listed under a new ID would otherwise leak across our train and test splits."

*[Highlight "Scraping". Pause briefly after "6,047."]*

### Beat 2 · Data exploration (0:15–0:35)

> "Then we looked at what we had. Madrid rents are heavily skewed and **bimodal** — one cluster of studios and one-beds around €1,200, another of larger flats around €2,800 — so we model **log-rent**, not raw rent. Among tabular features, **zone is by far the strongest predictor**: median rent in Salamanca-Retiro is **2.1 times** the median in the southern peripheries. We have a median of **22 photos per listing**, which means more than **143,000 photos** flow through our image encoder."

*[Highlight "EDA". Optional: gesture at the price-by-zone graph or the neighbourhood heatmap if you have it on the slide.]*

### Beat 3 · Features and PCA (0:35–0:55)

> "Three feature blocks. **Tabular**: sqm, rooms, bathrooms, eight zones, eight amenity flags. **Text**: each Spanish description through MiniLM, a multilingual sentence transformer, gives us 384 dimensions. **Image**: each photo through a deep encoder, 768 dimensions per photo, mean-pooled per listing.
>
> That's a lot of dimensions for 6,000 listings. So we apply **PCA per block** — image down to 50, text down to 30 — keeping the signal, dropping the noise. Then concatenate everything and train."

*[Highlight "PCA".]*

### Beat 4 · The first attempt: ResNet + neural net (0:55–1:25)

> "Our first version used **ResNet-50**, the standard ImageNet encoder, with a **neural network** as the regressor. It overfit. With around 1,000 effective training samples after splits, the network couldn't generalise on high-dimensional embeddings, and we plateaued at R² around 0.80.
>
> Two changes followed. We swapped the neural net for **Gradient Boosting**, which handles high-dim features much better at small N. That helped. But the bigger move was the encoder."

*[Highlight "ResNet/NN". Use "It overfit." as a deliberate beat.]*

### Hand-off (1:25–1:30)

> *(turn slightly toward Tom)*
>
> "And that's where Tom takes over."

---

## Tom — 1:00

### Beat 1 · SigLIP (0:00–0:20)

> "We replaced ResNet with **SigLIP** — Google's image-text contrastive encoder. ResNet was trained to classify ImageNet objects: dogs, birds, cars. SigLIP was trained to match images with their captions. For a rental listing, where the photo is essentially captioned by the description it lives next to, that pre-training is much closer to our task.
>
> The swap moved us from **R² 0.80 to 0.88, MAE from €351 down to €274**. **Seventy-seven euros per listing of accuracy, just from the encoder change.**"

*[Highlight "SigLIP". Slow down on the three numbers.]*

### Beat 2 · Ablation (0:20–0:40)

> "We then ran every non-empty combination of tabular, text, and image — seven configurations total — to make sure SigLIP was actually doing the work. The headline finding: **once SigLIP is in the model, the text description becomes almost redundant**. The image-text contrastive pre-training already encodes the semantic content the description was contributing. That surprised us, and it's why we kept the full ablation table in the report."

*[Highlight "Ablation".]*

### Beat 3 · Honesty (0:40–0:55)

> "Two things to flag. We tried **attention pooling** over per-photo embeddings — it tied with mean-pool, a negative result we kept in. And our first version had **two methodology bugs**: a deduplication leak and a train-test split mismatch. Fixing both moved us from a single-split R² of 0.85 down to a clean five-fold CV R² of 0.88. **The lower number is the honest one.**"

*[Last line gets eye contact, not the slide.]*

### Beat 4 · Extension transition (0:55–1:00)

> "Then we wrapped the model in a Chrome extension, because that's where rental decisions actually get made. Let me show you."

*[Highlight "Extension". Switch to browser. Demo begins.]*

---

## Rehearsal rules

1. **Run the hand-off three times.** "And that's where Tom takes over." → "Thanks." Smooth seam = rehearsed group.
2. **Time it once with a stopwatch.** If you're at 2:45, cut the EDA beat by 5s and the bug-fix beat by 5s. **Don't try to talk faster — cut, don't rush.**
3. **Numbers need silence around them.** When you say "R² 0.88" or "€77," pause for a beat after. The audience hears the number once and the silence is what tells them it matters.

## The three numbers that must land

- **6,047 listings** — the dataset
- **R² 0.88, MAE €274** — the headline result
- **€77 per listing** — what SigLIP added
