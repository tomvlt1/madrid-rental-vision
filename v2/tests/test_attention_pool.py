# Smoke test for v2/attention_pool.py — runs on synthetic per-photo
# embeddings so we can validate the training loop, masking, and
# variable-length batching without waiting for SigLIP extraction.
#
# Test design:
#   - Target is a linear function of the MEAN per-photo embedding.
#     This makes mean-pooling optimal; attention pool should learn
#     near-uniform weights and reach comparable MSE. We don't require
#     attention to beat mean-pool (can't, by construction) — we just
#     require it to not blow up and to produce valid attention weights.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.attention_pool import (  # noqa: E402
    AttnRegressor,
    BagDataset,
    collate_bags,
    group_per_photo_by_listing,
    pool_all,
    train_attention_pool,
)


def make_synthetic_bags(n_listings=120, d=16, min_k=3, max_k=10, seed=0):
    """Target = w · mean(photos). Mean pooling is optimal; attention pool
    should converge to approximately uniform weights."""
    rng = np.random.RandomState(seed)
    true_w = rng.randn(d).astype(np.float32) * 0.3
    per_photo = []
    rows = []
    targets = {}
    for i in range(n_listings):
        k = rng.randint(min_k, max_k + 1)
        lid = f"L{i:04d}"
        photos = rng.randn(k, d).astype(np.float32)
        target = float(photos.mean(axis=0) @ true_w)
        targets[lid] = target
        for idx, emb in enumerate(photos):
            per_photo.append(emb)
            rows.append({"listing_id": lid, "photo_idx": idx})
    return np.stack(per_photo), pd.DataFrame(rows), targets, d


def mse(pred, true):
    return float(np.mean((np.asarray(pred) - np.asarray(true)) ** 2))


def predict(model, bags, batch_size=32):
    model.train(False)
    device = next(model.parameters()).device
    loader = DataLoader(
        BagDataset(bags), batch_size=batch_size, shuffle=False, collate_fn=collate_bags,
    )
    preds, trues = [], []
    with torch.no_grad():
        for photos, mask, y, _ in loader:
            photos, mask = photos.to(device), mask.to(device)
            y_hat, _, _ = model(photos, mask)
            preds.append(y_hat.cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def test_end_to_end():
    per_photo, idx_df, targets, d = make_synthetic_bags()

    # group_per_photo_by_listing should produce one bag per listing
    bags = group_per_photo_by_listing(per_photo, idx_df, targets)
    assert len(bags) == len(targets)
    for b in bags:
        assert b.photos.shape[1] == d
        assert b.photos.shape[0] >= 3

    n = len(bags)
    rng = np.random.RandomState(0)
    rng.shuffle(bags)
    train_bags = bags[: int(0.8 * n)]
    val_bags = bags[int(0.8 * n):]

    y_var = float(np.var([b.target for b in val_bags]))
    print(f"val target variance: {y_var:.4f} (this is the MSE of a zero-predictor)")

    torch.manual_seed(0)
    model = train_attention_pool(
        train_bags, val_bags, embed_dim=d,
        epochs=50, lr=3e-3, patience=10, seed=0, verbose=False,
    )
    assert isinstance(model, AttnRegressor)

    # Sanity: attention weights are valid probability distributions
    pooled, ids, alphas = pool_all(model, val_bags)
    assert pooled.shape == (len(val_bags), d)
    assert len(alphas) == len(val_bags)
    for a, b in zip(alphas, val_bags):
        assert a.shape[0] == b.photos.shape[0]
        assert abs(a.sum() - 1.0) < 1e-4, (
            f"attention weights should sum to 1, got {a.sum()}"
        )
        assert float(a.min()) >= 0.0, "attention weights must be non-negative"

    # End-to-end MSE should beat the zero-predictor (var of target).
    preds, trues = predict(model, val_bags)
    attn_mse = mse(preds, trues)
    assert np.isfinite(attn_mse), "val MSE must be finite"
    print(f"attention-pool val MSE: {attn_mse:.4f}  (vs zero-predictor {y_var:.4f})")
    assert attn_mse < y_var * 0.9, (
        f"attention-pool should beat zero-predictor by 10%; "
        f"got {attn_mse:.3f} vs {y_var:.3f}"
    )

    print("test_attention_pool: PASS")


if __name__ == "__main__":
    test_end_to_end()
