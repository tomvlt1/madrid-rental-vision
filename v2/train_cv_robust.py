# v2: robust-loss CV targeting the luxury-tier collapse.
#
# Motivation. The v2 headline model has overall MAE=€388 and RMSE=€777,
# but the residual analysis exposed a savage per-quartile breakdown:
#
#     Quartile   price~€   MAE      RMSE      bias
#     Q1         1,126     183      234      +138
#     Q2         1,596     207      273       +40
#     Q3         2,226     345      471       -47
#     Q4         4,407     945    1,572      -542
#
# Q4 (luxury) is 5-6x worse than Q1 and systematically under-predicts by
# €542. This is a classic regress-to-mean signature from training on
# squared-error loss over a heavy-tailed target. Try three robust
# alternatives and compare, holding the feature stack constant:
#
#   baseline_squared_error    sklearn default, MSE on log-rent (v2 headline)
#   huber                     Huber loss, MSE near zero, linear in tail
#   quantile_0.5              explicit median regression (LAD)
#   asymmetric                quantile_0.5 below target, quantile_0.55 above —
#                             penalises UNDER-prediction (the direction we're
#                             biased in). Closest available in sklearn is
#                             loss='quantile', alpha>0.5 which tilts toward
#                             over-predicting. alpha=0.55 modest tilt.
#
# Feature stack: tabular + text(PCA30) + SigLIP-mean(PCA50). Same as
# v2/train_cv_siglip.py's `gb_tab_text_siglip_mean` model.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import PROCESSED_DIR, V2_DATA_DIR, V2_MODELS_DIR  # noqa: E402

CV_RESULTS_FILE = V2_MODELS_DIR / "cv_results_robust.json"
OOF_CSV = V2_MODELS_DIR / "oof_predictions_robust.csv"

N_FOLDS = 5
SEED = 42
IMAGE_PCA_COMPONENTS = 50
TEXT_PCA_COMPONENTS = 30

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = [
    "elevator", "ac", "terrace", "furnished", "heating",
    "exterior", "parking", "storage",
]

# --- loss variants to compare ---
# Each entry: name -> kwargs for GradientBoostingRegressor
LOSSES: dict[str, dict] = {
    "baseline_squared_error": {"loss": "squared_error"},
    "huber":                  {"loss": "huber", "alpha": 0.9},   # default alpha
    "quantile_0.5":           {"loss": "quantile", "alpha": 0.5},
    # alpha > 0.5 in quantile loss pushes predictions UP (penalises
    # under-prediction harder than over-prediction). Modest tilt 0.55.
    "quantile_0.55_asym":     {"loss": "quantile", "alpha": 0.55},
}


def gb(loss_kwargs):
    return GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=SEED,
        **loss_kwargs,
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
    s = StandardScaler()
    Xtr_s = s.fit_transform(Xtr)
    Xte_s = s.transform(Xte)
    p = PCA(n_components=n, random_state=SEED)
    return p.fit_transform(Xtr_s), p.transform(Xte_s)


def metrics_from_log(pred_log, true_log):
    rmse_log = float(np.sqrt(np.mean((pred_log - true_log) ** 2)))
    pred_eur = np.expm1(pred_log)
    true_eur = np.expm1(true_log)
    rmse_eur = float(np.sqrt(np.mean((pred_eur - true_eur) ** 2)))
    mae_eur = float(np.mean(np.abs(pred_eur - true_eur)))
    mape = float(np.mean(np.abs(pred_eur - true_eur) / true_eur) * 100)
    bias = float(np.mean(pred_eur - true_eur))
    ss_res = np.sum((true_log - pred_log) ** 2)
    ss_tot = np.sum((true_log - true_log.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return {
        "r2": r2,
        "mae_euros": mae_eur,
        "rmse_euros": rmse_eur,
        "mape": mape,
        "bias_euros": bias,
        "rmse_log": rmse_log,
    }


def main():
    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load everything
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)

    txt = np.load(PROCESSED_DIR / "text_embeddings.npy")
    txt_idx = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    txt_idx["listing_id"] = txt_idx["listing_id"].astype(str)
    txt_idx["txt_row"] = range(len(txt_idx))

    siglip = np.load(V2_DATA_DIR / "siglip_embeddings.npy")
    siglip_idx = pd.read_csv(V2_DATA_DIR / "siglip_embeddings_index.csv")
    siglip_idx["listing_id"] = siglip_idx["listing_id"].astype(str)
    siglip_idx["siglip_row"] = range(len(siglip_idx))

    df = (
        df.merge(txt_idx[["listing_id", "txt_row"]], on="listing_id", how="inner")
          .merge(siglip_idx[["listing_id", "siglip_row"]], on="listing_id", how="inner")
          .reset_index(drop=True)
    )
    print(f"Listings joined: {len(df)}")

    tab, y_log = build_tab(df)
    X_txt = txt[df["txt_row"].values].astype(np.float32)
    X_sig = siglip[df["siglip_row"].values].astype(np.float32)
    y_price = df["price"].values.astype(np.float32)

    n = len(df)
    # price quartiles based on the WHOLE dataset (fixed boundaries so per-fold
    # counts are comparable)
    q_labels = ["Q1_cheap", "Q2", "Q3", "Q4_expensive"]
    price_q = pd.qcut(pd.Series(y_price), 4, labels=q_labels)
    q_to_idx = {label: np.where((price_q == label).to_numpy())[0] for label in q_labels}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Store OOF predictions for each loss variant so we can aggregate per-quartile
    oof_preds_log: dict[str, np.ndarray] = {
        name: np.full(n, np.nan) for name in LOSSES
    }
    fold_assignment = np.full(n, -1, dtype=int)

    # Per-fold global metrics
    per_fold_metrics: dict[str, list[dict]] = {name: [] for name in LOSSES}

    for k, (tr, te) in enumerate(kf.split(np.arange(n)), 1):
        print(f"\n=== Fold {k}/{N_FOLDS}  train={len(tr)} test={len(te)} ===")
        tab_scaler = StandardScaler().fit(tab[tr])
        Xtab_tr, Xtab_te = tab_scaler.transform(tab[tr]), tab_scaler.transform(tab[te])
        Xtxt_tr, Xtxt_te = scale_pca(X_txt[tr], X_txt[te], TEXT_PCA_COMPONENTS)
        Xsig_tr, Xsig_te = scale_pca(X_sig[tr], X_sig[te], IMAGE_PCA_COMPONENTS)

        Xtr = np.hstack([Xtab_tr, Xtxt_tr, Xsig_tr])
        Xte = np.hstack([Xtab_te, Xtxt_te, Xsig_te])
        fold_assignment[te] = k

        for name, kwargs in LOSSES.items():
            m = gb(kwargs).fit(Xtr, y_log[tr])
            pred = m.predict(Xte)
            oof_preds_log[name][te] = pred
            met = metrics_from_log(pred, y_log[te])
            per_fold_metrics[name].append(met)
            print(
                f"  {name:28s}  R²={met['r2']:.4f}  "
                f"MAE=€{met['mae_euros']:4.0f}  RMSE=€{met['rmse_euros']:4.0f}  "
                f"bias=€{met['bias_euros']:+4.0f}"
            )

    # --- Aggregate global ---
    def aggregate(per_fold):
        out = {}
        for name, folds in per_fold.items():
            out[name] = {}
            for metric in folds[0].keys():
                vals = np.array([f[metric] for f in folds])
                out[name][metric + "_mean"] = float(vals.mean())
                out[name][metric + "_std"] = float(vals.std(ddof=1))
        return out

    agg_global = aggregate(per_fold_metrics)

    # --- Aggregate per-quartile using OOF predictions ---
    per_quartile: dict[str, dict[str, dict]] = {name: {} for name in LOSSES}
    for name, pred_log in oof_preds_log.items():
        for q_label, idx in q_to_idx.items():
            pred_eur = np.expm1(pred_log[idx])
            true_eur = y_price[idx]
            per_quartile[name][q_label] = {
                "n": int(len(idx)),
                "price_mean": float(true_eur.mean()),
                "mae_euros": float(np.mean(np.abs(pred_eur - true_eur))),
                "rmse_euros": float(np.sqrt(np.mean((pred_eur - true_eur) ** 2))),
                "mape": float(100 * np.mean(np.abs(pred_eur - true_eur) / true_eur)),
                "bias_euros": float(np.mean(pred_eur - true_eur)),
            }

    # Print summary tables
    print("\n" + "=" * 88)
    print(f"ROBUST-LOSS CV SUMMARY ({N_FOLDS}-fold, N={n})")
    print("=" * 88)
    print(
        f"{'Loss':<26} {'R²':>16} {'MAE (€)':>12} {'RMSE (€)':>12} {'Bias (€)':>12}"
    )
    print("-" * 88)
    for name, m in agg_global.items():
        r2 = f"{m['r2_mean']:.4f}±{m['r2_std']:.4f}"
        mae = f"{m['mae_euros_mean']:.0f}±{m['mae_euros_std']:.0f}"
        rmse = f"{m['rmse_euros_mean']:.0f}±{m['rmse_euros_std']:.0f}"
        bias = f"{m['bias_euros_mean']:+.0f}"
        print(f"{name:<26} {r2:>16} {mae:>12} {rmse:>12} {bias:>12}")

    # Per-quartile table, focused on Q4
    print("\n" + "=" * 92)
    print("PER-QUARTILE MAE (€)   — the column that matters is Q4_expensive")
    print("=" * 92)
    header = f"{'Loss':<26}"
    q_order = ["Q1_cheap", "Q2", "Q3", "Q4_expensive"]
    for q in q_order:
        header += f"{q:>14}"
    print(header)
    print("-" * 92)
    for name, q_dict in per_quartile.items():
        row = f"{name:<26}"
        for q in q_order:
            row += f"{q_dict[q]['mae_euros']:>14.0f}"
        print(row)

    print("\n" + "=" * 92)
    print("PER-QUARTILE BIAS (€) — negative = under-prediction; v2 baseline was -€542 in Q4")
    print("=" * 92)
    header = f"{'Loss':<26}"
    for q in q_order:
        header += f"{q:>14}"
    print(header)
    print("-" * 92)
    for name, q_dict in per_quartile.items():
        row = f"{name:<26}"
        for q in q_order:
            row += f"{q_dict[q]['bias_euros']:>+14.0f}"
        print(row)

    # Save results + OOF predictions
    out = {
        "n_folds": N_FOLDS,
        "n_listings": int(n),
        "seed": SEED,
        "losses": agg_global,
        "per_quartile": per_quartile,
    }
    CV_RESULTS_FILE.write_text(json.dumps(out, indent=2))

    rows = []
    for name, preds_log in oof_preds_log.items():
        preds_eur = np.expm1(preds_log)
        for i in range(n):
            rows.append({
                "listing_id": df["listing_id"].iloc[i],
                "loss": name,
                "price": float(y_price[i]),
                "pred_price": float(preds_eur[i]),
                "error_eur": float(preds_eur[i] - y_price[i]),
                "fold": int(fold_assignment[i]),
                "zone": df["zone"].iloc[i],
                "price_q": str(price_q.iloc[i]),
            })
    pd.DataFrame(rows).to_csv(OOF_CSV, index=False)

    print(f"\nSaved: {CV_RESULTS_FILE}")
    print(f"Saved: {OOF_CSV}")


if __name__ == "__main__":
    main()
