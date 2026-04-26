# v2: 5-fold CV comparing ResNet mean, SigLIP mean, and SigLIP attention-pooled
# image representations as the image block in the downstream GB pipeline.
#
# Depends on v2/extract_siglip_embeddings.py having been run (it produces
# v2/data/siglip_embeddings.npy + siglip_per_photo.npy).
#
# For the attention-pooling variant we refit the AttnRegressor per fold,
# train on the CV train-split ONLY with a small internal val slice for
# early stopping. No leakage between folds. This is stricter than v1's
# handling of fine-tuned ResNet (which was fit once on splits.json and
# reused across folds) -- worth mentioning in the writeup.
#
# Output: v2/models/cv_results_siglip.json, plus per-fold attention-pool
# training artifacts in v2/logs/.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import PROCESSED_DIR, V2_DATA_DIR, V2_MODELS_DIR  # noqa: E402
from v2.attention_pool import (  # noqa: E402
    group_per_photo_by_listing,
    pool_all,
    train_attention_pool,
)

CV_RESULTS_FILE = V2_MODELS_DIR / "cv_results_siglip.json"

N_FOLDS = 5
SEED = 42
IMAGE_PCA_COMPONENTS = 50
TEXT_PCA_COMPONENTS = 30
INTERNAL_VAL_FRAC = 0.1  # slice of train fold used for early stopping

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = [
    "elevator", "ac", "terrace", "furnished", "heating",
    "exterior", "parking", "storage",
]


def compute_metrics(preds_log, targets_log):
    rmse_log = float(np.sqrt(np.mean((preds_log - targets_log) ** 2)))
    preds_price = np.expm1(preds_log)
    targets_price = np.expm1(targets_log)
    rmse_euros = float(np.sqrt(np.mean((preds_price - targets_price) ** 2)))
    mae = float(np.mean(np.abs(preds_price - targets_price)))
    mape = float(np.mean(np.abs(preds_price - targets_price) / targets_price) * 100)
    ss_res = np.sum((targets_log - preds_log) ** 2)
    ss_tot = np.sum((targets_log - targets_log.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return {
        "r2": r2,
        "mae_euros": mae,
        "rmse_euros": rmse_euros,
        "mape": mape,
        "rmse_log": rmse_log,
    }


def gb():
    return GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=SEED,
    )


def build_tabular(df):
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    tab = pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1)
    return tab.values.astype(np.float32), np.log1p(df["price"].values.astype(np.float32))


def load_all():
    """Join listings, ResNet (frozen), text, SigLIP mean, and per-photo SigLIP."""
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)

    # Existing ResNet frozen embeddings (baseline comparison)
    img = np.load(PROCESSED_DIR / "embeddings.npy")
    img_idx = pd.read_csv(PROCESSED_DIR / "embeddings_index.csv")
    img_idx["listing_id"] = img_idx["listing_id"].astype(str)
    img_idx["img_row"] = range(len(img_idx))

    # Text embeddings
    txt = np.load(PROCESSED_DIR / "text_embeddings.npy")
    txt_idx = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    txt_idx["listing_id"] = txt_idx["listing_id"].astype(str)
    txt_idx["txt_row"] = range(len(txt_idx))

    # SigLIP mean-pooled
    siglip_mean = np.load(V2_DATA_DIR / "siglip_embeddings.npy")
    siglip_idx = pd.read_csv(V2_DATA_DIR / "siglip_embeddings_index.csv")
    siglip_idx["listing_id"] = siglip_idx["listing_id"].astype(str)
    siglip_idx["siglip_row"] = range(len(siglip_idx))

    # SigLIP per-photo (for attention pooling)
    siglip_photos = np.load(V2_DATA_DIR / "siglip_per_photo.npy")
    siglip_photos_idx = pd.read_csv(V2_DATA_DIR / "siglip_per_photo_index.csv")
    siglip_photos_idx["listing_id"] = siglip_photos_idx["listing_id"].astype(str)

    df = (
        df.merge(img_idx[["listing_id", "img_row"]], on="listing_id", how="inner")
          .merge(txt_idx[["listing_id", "txt_row"]], on="listing_id", how="inner")
          .merge(siglip_idx[["listing_id", "siglip_row"]], on="listing_id", how="inner")
          .reset_index(drop=True)
    )
    # Require at least one siglip per-photo row
    valid_ids = set(siglip_photos_idx["listing_id"].unique())
    df = df[df["listing_id"].isin(valid_ids)].reset_index(drop=True)
    print(f"Listings joined across ResNet + text + SigLIP: {len(df)}")

    tab, y = build_tabular(df)
    X_img = img[df["img_row"].values].astype(np.float32)
    X_txt = txt[df["txt_row"].values].astype(np.float32)
    X_siglip_mean = siglip_mean[df["siglip_row"].values].astype(np.float32)

    # Filter per-photo index to listings present in df, preserving row order
    # aligned to siglip_photos:
    keep_mask = siglip_photos_idx["listing_id"].isin(df["listing_id"]).values
    per_photo_embs = siglip_photos[keep_mask]
    per_photo_idx = siglip_photos_idx[keep_mask].reset_index(drop=True)

    return {
        "tab": tab, "X_img": X_img, "X_txt": X_txt,
        "X_siglip_mean": X_siglip_mean,
        "per_photo_embs": per_photo_embs,
        "per_photo_idx": per_photo_idx,
        "y": y,
        "listing_ids": df["listing_id"].values,
    }


def scale_pca(X_tr, X_te, n_components):
    s = StandardScaler()
    Xtr = s.fit_transform(X_tr)
    Xte = s.transform(X_te)
    p = PCA(n_components=n_components, random_state=SEED)
    return p.fit_transform(Xtr), p.transform(Xte)


def fit_attention_for_fold(data, train_idx, fold_k):
    """Train attention pool on the train-fold-internal subset, return pooled
    embeddings for ALL listings (train + test). We split train_idx again
    into attn_train/attn_val for early stopping (not used for reporting)."""
    train_ids = set(data["listing_ids"][train_idx])
    # carve an internal val slice out of train_idx
    rng = np.random.RandomState(SEED + fold_k)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(perm) * INTERNAL_VAL_FRAC))
    val_idx_in_train = perm[:n_val]
    fit_idx_in_train = perm[n_val:]
    fit_ids = set(data["listing_ids"][fit_idx_in_train])
    val_ids = set(data["listing_ids"][val_idx_in_train])

    targets_by_id = {
        lid: float(y) for lid, y in zip(data["listing_ids"], data["y"])
    }
    train_bags_for_fit = [
        b for b in group_per_photo_by_listing(
            data["per_photo_embs"], data["per_photo_idx"],
            {lid: targets_by_id[lid] for lid in fit_ids},
        )
    ]
    val_bags = [
        b for b in group_per_photo_by_listing(
            data["per_photo_embs"], data["per_photo_idx"],
            {lid: targets_by_id[lid] for lid in val_ids},
        )
    ]
    print(
        f"  attn-fit: train={len(train_bags_for_fit)} val={len(val_bags)} "
        f"(from train fold of {len(train_idx)})"
    )
    model = train_attention_pool(
        train_bags_for_fit, val_bags, embed_dim=data["per_photo_embs"].shape[1],
        seed=SEED + fold_k, verbose=False,
    )
    all_bags = group_per_photo_by_listing(
        data["per_photo_embs"], data["per_photo_idx"], targets_by_id,
    )
    pooled, pooled_ids, alphas = pool_all(model, all_bags)
    id_to_row = {lid: i for i, lid in enumerate(pooled_ids)}
    ordered = np.stack([pooled[id_to_row[lid]] for lid in data["listing_ids"]])
    return ordered.astype(np.float32)


def fold_metrics(data, train_idx, test_idx, fold_k):
    y_tr, y_te = data["y"][train_idx], data["y"][test_idx]

    tab_scaler = StandardScaler()
    Xtab_tr = tab_scaler.fit_transform(data["tab"][train_idx])
    Xtab_te = tab_scaler.transform(data["tab"][test_idx])

    # --- image blocks, each scaled+PCA per fold ---
    Xresnet_tr, Xresnet_te = scale_pca(
        data["X_img"][train_idx], data["X_img"][test_idx], IMAGE_PCA_COMPONENTS
    )
    Xsigmean_tr, Xsigmean_te = scale_pca(
        data["X_siglip_mean"][train_idx], data["X_siglip_mean"][test_idx],
        IMAGE_PCA_COMPONENTS,
    )
    Xtxt_tr, Xtxt_te = scale_pca(
        data["X_txt"][train_idx], data["X_txt"][test_idx], TEXT_PCA_COMPONENTS
    )

    # --- Attention-pooled SigLIP: refit per fold ---
    attn_embs = fit_attention_for_fold(data, train_idx, fold_k)
    Xsigattn_tr, Xsigattn_te = scale_pca(
        attn_embs[train_idx], attn_embs[test_idx], IMAGE_PCA_COMPONENTS
    )

    results = {}

    def run(name, Xtr, Xte):
        m = gb().fit(Xtr, y_tr)
        results[name] = compute_metrics(m.predict(Xte), y_te)

    # Direct image-block comparisons on top of tabular:
    run("gb_tab_resnet_frozen", np.hstack([Xtab_tr, Xresnet_tr]), np.hstack([Xtab_te, Xresnet_te]))
    run("gb_tab_siglip_mean", np.hstack([Xtab_tr, Xsigmean_tr]), np.hstack([Xtab_te, Xsigmean_te]))
    run("gb_tab_siglip_attn", np.hstack([Xtab_tr, Xsigattn_tr]), np.hstack([Xtab_te, Xsigattn_te]))

    # Three-modality, SigLIP variants:
    run(
        "gb_tab_text_siglip_mean",
        np.hstack([Xtab_tr, Xtxt_tr, Xsigmean_tr]),
        np.hstack([Xtab_te, Xtxt_te, Xsigmean_te]),
    )
    run(
        "gb_tab_text_siglip_attn",
        np.hstack([Xtab_tr, Xtxt_tr, Xsigattn_tr]),
        np.hstack([Xtab_te, Xtxt_te, Xsigattn_te]),
    )

    # Image-only ablations (no tabular): SigLIP mean vs attention
    run("gb_siglip_mean_only", Xsigmean_tr, Xsigmean_te)
    run("gb_siglip_attn_only", Xsigattn_tr, Xsigattn_te)

    return results


def aggregate(per_fold):
    out = {}
    if not per_fold:
        return out
    for model in per_fold[0].keys():
        out[model] = {}
        for metric in per_fold[0][model].keys():
            values = np.array([fold[model][metric] for fold in per_fold])
            out[model][metric + "_mean"] = float(values.mean())
            out[model][metric + "_std"] = float(values.std(ddof=1))
            out[model][metric + "_per_fold"] = [float(v) for v in values]
    return out


def main():
    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = load_all()
    n = len(data["y"])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold = []
    for k, (train_idx, test_idx) in enumerate(kf.split(np.arange(n)), 1):
        print(f"\n=== Fold {k}/{N_FOLDS} — train={len(train_idx)} test={len(test_idx)} ===")
        fold_res = fold_metrics(data, train_idx, test_idx, k)
        for model, m in fold_res.items():
            print(
                f"  {model:30s} R²={m['r2']:.4f}  "
                f"MAE=€{m['mae_euros']:6.0f}  RMSE=€{m['rmse_euros']:6.0f}  "
                f"MAPE={m['mape']:5.2f}%"
            )
        per_fold.append(fold_res)

    agg = aggregate(per_fold)

    print("\n" + "=" * 100)
    print(f"V2 SIGLIP CV — ResNet vs SigLIP vs SigLIP+attention ({N_FOLDS}-fold)")
    print("=" * 100)
    print(
        f"{'Model':<32} {'R²':>16} {'MAE (€)':>14} "
        f"{'RMSE (€)':>14} {'MAPE (%)':>11}"
    )
    print("-" * 100)
    for model, m in agg.items():
        r2 = f"{m['r2_mean']:.4f}±{m['r2_std']:.4f}"
        mae = f"{m['mae_euros_mean']:.0f}±{m['mae_euros_std']:.0f}"
        rmse = f"{m['rmse_euros_mean']:.0f}±{m['rmse_euros_std']:.0f}"
        mape = f"{m['mape_mean']:.2f}±{m['mape_std']:.2f}"
        print(f"{model:<32} {r2:>16} {mae:>14} {rmse:>14} {mape:>11}")

    out = {
        "n_folds": N_FOLDS,
        "n_listings": int(n),
        "seed": SEED,
        "image_pca_components": IMAGE_PCA_COMPONENTS,
        "text_pca_components": TEXT_PCA_COMPONENTS,
        "internal_val_frac_for_attention": INTERNAL_VAL_FRAC,
        "models": agg,
    }
    CV_RESULTS_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {CV_RESULTS_FILE}")


if __name__ == "__main__":
    main()
