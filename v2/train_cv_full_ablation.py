"""
Full 2^3 - 1 = 7-combination ablation grid on the FULL 6,047-listing
unified dataset, using SigLIP as the image encoder (no ResNet dependency).

This addresses the teacher's "we need all the combinations of the model"
note while using all the listings we collected (vs train_cv_v2.py which
inner-joins on ResNet and caps at 1,425).

Combinations of {tabular, text, siglip}:
  1. tabular
  2. text
  3. siglip (mean-pooled)
  4. tabular + text
  5. tabular + siglip
  6. text + siglip
  7. tabular + text + siglip          (the headline model)

Plus a Ridge tabular baseline for sanity.

Output: v2/models/cv_results_full_ablation.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import V2_DATA_DIR, V2_MODELS_DIR  # noqa: E402

# Always read from the v2 unified file; ignore MRV_DATA_DIR / PROCESSED_DIR
# fallback so we can never accidentally read ai2's 1,425-row listings_clean.csv.
CLEAN_CSV = V2_DATA_DIR / "listings_clean_v2.csv"
TEXT_NPY = V2_DATA_DIR / "text_embeddings.npy"
TEXT_IDX = V2_DATA_DIR / "text_embeddings_index.csv"
SIGLIP_NPY = V2_DATA_DIR / "siglip_embeddings.npy"
SIGLIP_IDX = V2_DATA_DIR / "siglip_embeddings_index.csv"

CV_RESULTS_FILE = V2_MODELS_DIR / "cv_results_full_ablation.json"

N_FOLDS = 5
SEED = 42
IMAGE_PCA_COMPONENTS = 50
TEXT_PCA_COMPONENTS = 30

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = [
    "elevator", "ac", "terrace", "furnished",
    "heating", "exterior", "parking", "storage",
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
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=SEED,
    )


def build_tab(df):
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    tab = pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1)
    return tab.values.astype(np.float32), np.log1p(df["price"].values.astype(np.float32))


def scale_pca(Xtr, Xte, n):
    s = StandardScaler().fit(Xtr)
    Xtr_s = s.transform(Xtr)
    Xte_s = s.transform(Xte)
    p = PCA(n_components=n, random_state=SEED).fit(Xtr_s)
    return p.transform(Xtr_s), p.transform(Xte_s)


def load_all():
    """Join unified clean listings with text + SigLIP embeddings (no ResNet)."""
    print(f"Reading: {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    print(f"  raw listings: {len(df)}")

    txt = np.load(TEXT_NPY)
    txt_idx = pd.read_csv(TEXT_IDX)
    txt_idx["listing_id"] = txt_idx["listing_id"].astype(str)
    txt_idx["txt_row"] = range(len(txt_idx))
    print(f"  text embeddings: {len(txt_idx)} rows")

    siglip = np.load(SIGLIP_NPY)
    siglip_idx = pd.read_csv(SIGLIP_IDX)
    siglip_idx["listing_id"] = siglip_idx["listing_id"].astype(str)
    siglip_idx["siglip_row"] = range(len(siglip_idx))
    print(f"  SigLIP embeddings: {len(siglip_idx)} rows")

    df = (
        df.merge(txt_idx[["listing_id", "txt_row"]], on="listing_id", how="inner")
          .merge(siglip_idx[["listing_id", "siglip_row"]], on="listing_id", how="inner")
          .reset_index(drop=True)
    )
    print(f"Listings joined (text + SigLIP, no ResNet dependency): {len(df)}")

    tab, y = build_tab(df)
    X_txt = txt[df["txt_row"].values].astype(np.float32)
    X_sig = siglip[df["siglip_row"].values].astype(np.float32)
    return {"tab": tab, "X_txt": X_txt, "X_sig": X_sig, "y": y}


def fold_metrics(data, train_idx, test_idx):
    y_tr, y_te = data["y"][train_idx], data["y"][test_idx]
    tab_scaler = StandardScaler().fit(data["tab"][train_idx])
    Xtab_tr = tab_scaler.transform(data["tab"][train_idx])
    Xtab_te = tab_scaler.transform(data["tab"][test_idx])
    Xtxt_tr, Xtxt_te = scale_pca(data["X_txt"][train_idx], data["X_txt"][test_idx], TEXT_PCA_COMPONENTS)
    Xsig_tr, Xsig_te = scale_pca(data["X_sig"][train_idx], data["X_sig"][test_idx], IMAGE_PCA_COMPONENTS)

    out = {}

    def run(name, Xtr, Xte):
        m = gb().fit(Xtr, y_tr)
        out[name] = compute_metrics(m.predict(Xte), y_te)

    # Ridge baseline on tabular
    m = Ridge(alpha=1.0).fit(Xtab_tr, y_tr)
    out["ridge_tabular"] = compute_metrics(m.predict(Xtab_te), y_te)

    # Full 2^3-1 = 7 subsets of {tabular, text, siglip}
    run("gb_tabular",            Xtab_tr,                                       Xtab_te)
    run("gb_text",               Xtxt_tr,                                       Xtxt_te)
    run("gb_siglip",             Xsig_tr,                                       Xsig_te)
    run("gb_tabular_text",       np.hstack([Xtab_tr, Xtxt_tr]),                 np.hstack([Xtab_te, Xtxt_te]))
    run("gb_tabular_siglip",     np.hstack([Xtab_tr, Xsig_tr]),                 np.hstack([Xtab_te, Xsig_te]))
    run("gb_text_siglip",        np.hstack([Xtxt_tr, Xsig_tr]),                 np.hstack([Xtxt_te, Xsig_te]))
    run("gb_tabular_text_siglip", np.hstack([Xtab_tr, Xtxt_tr, Xsig_tr]),       np.hstack([Xtab_te, Xtxt_te, Xsig_te]))
    return out


def aggregate(per_fold):
    out = {}
    if not per_fold:
        return out
    for model in per_fold[0].keys():
        out[model] = {}
        for metric in per_fold[0][model].keys():
            v = np.array([f[model][metric] for f in per_fold])
            out[model][metric + "_mean"] = float(v.mean())
            out[model][metric + "_std"] = float(v.std(ddof=1))
            out[model][metric + "_per_fold"] = [float(x) for x in v]
    return out


def main():
    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = load_all()
    n = len(data["y"])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold = []
    for k, (tr, te) in enumerate(kf.split(np.arange(n)), 1):
        print(f"\n=== Fold {k}/{N_FOLDS} (train={len(tr)} test={len(te)}) ===")
        res = fold_metrics(data, tr, te)
        for name, m in res.items():
            print(
                f"  {name:30s} R²={m['r2']:.4f}  "
                f"MAE=€{m['mae_euros']:5.0f}  RMSE=€{m['rmse_euros']:5.0f}  "
                f"MAPE={m['mape']:5.2f}%"
            )
        per_fold.append(res)

    agg = aggregate(per_fold)

    print("\n" + "=" * 100)
    print(f"FULL ABLATION GRID — {N_FOLDS}-fold CV on {n} listings (tabular + text + SigLIP)")
    print("=" * 100)
    print(f"{'Model':<32} {'R²':>16} {'MAE (€)':>14} {'RMSE (€)':>14} {'MAPE (%)':>11}")
    print("-" * 100)
    for name, m in agg.items():
        r2 = f"{m['r2_mean']:.4f}±{m['r2_std']:.4f}"
        mae = f"{m['mae_euros_mean']:.0f}±{m['mae_euros_std']:.0f}"
        rmse = f"{m['rmse_euros_mean']:.0f}±{m['rmse_euros_std']:.0f}"
        mape = f"{m['mape_mean']:.2f}±{m['mape_std']:.2f}"
        print(f"{name:<32} {r2:>16} {mae:>14} {rmse:>14} {mape:>11}")

    out = {
        "n_folds": N_FOLDS,
        "n_listings": int(n),
        "seed": SEED,
        "image_pca_components": IMAGE_PCA_COMPONENTS,
        "text_pca_components": TEXT_PCA_COMPONENTS,
        "models": agg,
    }
    CV_RESULTS_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {CV_RESULTS_FILE}")


if __name__ == "__main__":
    main()
