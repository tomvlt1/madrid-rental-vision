# v2 residual analysis: answers the question the RMSE/MAE gap raises:
#
#   The headline model has MAE=€411 but RMSE=€810. That's a 2x gap and
#   implies a heavy-tailed error distribution: a handful of listings
#   have errors >€1,500. Who are they?
#
# This script refits the headline model under 5-fold CV, collects raw
# out-of-fold predictions for every listing, and breaks the residuals
# down by:
#   - price tier (quartiles of asking rent)
#   - zone
#   - residual percentile (where does the tail concentrate?)
#
# Output: v2/models/residual_analysis.json  and  v2/models/oof_predictions.csv

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

from v2.paths import PROCESSED_DIR, V2_MODELS_DIR  # noqa: E402

OUT_JSON = V2_MODELS_DIR / "residual_analysis.json"
OUT_CSV = V2_MODELS_DIR / "oof_predictions.csv"

N_FOLDS = 5
SEED = 42
IMAGE_PCA_COMPONENTS = 50
TEXT_PCA_COMPONENTS = 30

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = [
    "elevator", "ac", "terrace", "furnished", "heating",
    "exterior", "parking", "storage",
]


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
    s = StandardScaler()
    Xtr_s = s.fit_transform(Xtr)
    Xte_s = s.transform(Xte)
    p = PCA(n_components=n, random_state=SEED)
    return p.fit_transform(Xtr_s), p.transform(Xte_s)


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)

    ft = np.load(PROCESSED_DIR / "embeddings_finetuned.npy")
    ft_idx = pd.read_csv(PROCESSED_DIR / "embeddings_finetuned_index.csv")
    ft_idx["listing_id"] = ft_idx["listing_id"].astype(str)
    ft_idx["ft_row"] = range(len(ft_idx))

    txt = np.load(PROCESSED_DIR / "text_embeddings.npy")
    txt_idx = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    txt_idx["listing_id"] = txt_idx["listing_id"].astype(str)
    txt_idx["txt_row"] = range(len(txt_idx))

    df = (
        df.merge(ft_idx[["listing_id", "ft_row"]], on="listing_id", how="inner")
          .merge(txt_idx[["listing_id", "txt_row"]], on="listing_id", how="inner")
          .reset_index(drop=True)
    )
    print(f"Listings: {len(df)}")

    tab, y_log = build_tab(df)
    X_ft = ft[df["ft_row"].values].astype(np.float32)
    X_txt = txt[df["txt_row"].values].astype(np.float32)
    y_price = df["price"].values.astype(np.float32)

    n = len(df)
    oof_pred_log = np.full(n, np.nan, dtype=np.float64)
    fold_assignment = np.full(n, -1, dtype=int)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(kf.split(np.arange(n)), 1):
        print(f"  fold {k}/{N_FOLDS}")
        tab_s = StandardScaler().fit(tab[tr])
        Xtab_tr, Xtab_te = tab_s.transform(tab[tr]), tab_s.transform(tab[te])
        Xft_tr, Xft_te = scale_pca(X_ft[tr], X_ft[te], IMAGE_PCA_COMPONENTS)
        Xtxt_tr, Xtxt_te = scale_pca(X_txt[tr], X_txt[te], TEXT_PCA_COMPONENTS)
        Xtr = np.hstack([Xtab_tr, Xtxt_tr, Xft_tr])
        Xte = np.hstack([Xtab_te, Xtxt_te, Xft_te])
        m = gb().fit(Xtr, y_log[tr])
        oof_pred_log[te] = m.predict(Xte)
        fold_assignment[te] = k

    oof_pred_price = np.expm1(oof_pred_log)
    err_eur = oof_pred_price - y_price     # signed (pos = over-predict)
    abs_err = np.abs(err_eur)

    out_df = pd.DataFrame({
        "listing_id": df["listing_id"].values,
        "zone": df["zone"].values,
        "price": y_price,
        "pred_price": oof_pred_price,
        "error_eur": err_eur,
        "abs_error_eur": abs_err,
        "fold": fold_assignment,
    })
    out_df.to_csv(OUT_CSV, index=False)

    summary = {}

    # Global metrics
    summary["global"] = {
        "n": int(n),
        "mae_euros": float(abs_err.mean()),
        "rmse_euros": float(np.sqrt(np.mean(err_eur ** 2))),
        "median_abs_error_euros": float(np.median(abs_err)),
        "p90_abs_error_euros": float(np.percentile(abs_err, 90)),
        "p95_abs_error_euros": float(np.percentile(abs_err, 95)),
        "p99_abs_error_euros": float(np.percentile(abs_err, 99)),
        "max_abs_error_euros": float(abs_err.max()),
        "bias_euros": float(err_eur.mean()),   # mean signed error
    }

    # Percent of total squared error concentrated in the top decile of |errors|
    order = np.argsort(abs_err)[::-1]
    top10pct = order[: max(1, n // 10)]
    total_sse = float(np.sum(err_eur ** 2))
    top10pct_sse = float(np.sum(err_eur[top10pct] ** 2))
    summary["tail_concentration"] = {
        "top_10pct_listings": int(len(top10pct)),
        "pct_of_total_squared_error_in_top_10pct": 100.0 * top10pct_sse / total_sse,
    }

    # By price quartile
    q_labels = ["Q1_cheap", "Q2", "Q3", "Q4_expensive"]
    q = pd.qcut(pd.Series(y_price), 4, labels=q_labels)
    by_q = {}
    for label in q_labels:
        mask_q = (q == label).to_numpy()
        if mask_q.sum() == 0:
            continue
        e = err_eur[mask_q]
        ae = abs_err[mask_q]
        by_q[label] = {
            "n": int(mask_q.sum()),
            "price_mean": float(y_price[mask_q].mean()),
            "mae_euros": float(ae.mean()),
            "rmse_euros": float(np.sqrt(np.mean(e ** 2))),
            "mape": float(100 * (ae / y_price[mask_q]).mean()),
            "bias_euros": float(e.mean()),
        }
    summary["by_price_quartile"] = by_q

    # By zone (top-5 best and worst by MAE)
    by_zone = []
    for z, sub in out_df.groupby("zone"):
        if len(sub) < 10:
            continue  # skip zones with too few listings for a stable estimate
        by_zone.append({
            "zone": z,
            "n": int(len(sub)),
            "price_mean": float(sub["price"].mean()),
            "mae_euros": float(sub["abs_error_eur"].mean()),
            "rmse_euros": float(np.sqrt((sub["error_eur"] ** 2).mean())),
            "mape": float(100 * (sub["abs_error_eur"] / sub["price"]).mean()),
        })
    by_zone.sort(key=lambda r: r["mae_euros"])
    summary["zone_best_5_mae"] = by_zone[:5]
    summary["zone_worst_5_mae"] = by_zone[-5:]

    # Worst 10 individual listings (by |error|)
    worst10 = out_df.nlargest(10, "abs_error_eur")
    summary["worst_10_listings"] = worst10[
        ["listing_id", "zone", "price", "pred_price", "error_eur"]
    ].to_dict(orient="records")

    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OUT_CSV}  and  {OUT_JSON}")

    # Pretty print to stdout
    g = summary["global"]
    print(
        f"\nGlobal:  MAE=€{g['mae_euros']:.0f}  RMSE=€{g['rmse_euros']:.0f}  "
        f"median=€{g['median_abs_error_euros']:.0f}  "
        f"p90=€{g['p90_abs_error_euros']:.0f}  p95=€{g['p95_abs_error_euros']:.0f}  "
        f"p99=€{g['p99_abs_error_euros']:.0f}  max=€{g['max_abs_error_euros']:.0f}"
    )
    tc = summary["tail_concentration"]
    print(
        f"Tail concentration: top 10% of listings account for "
        f"{tc['pct_of_total_squared_error_in_top_10pct']:.1f}% of squared error."
    )
    print("\nBy price quartile:")
    for k, v in summary["by_price_quartile"].items():
        print(
            f"  {k:<16} n={v['n']:4d}  price~€{v['price_mean']:5.0f}  "
            f"MAE=€{v['mae_euros']:5.0f}  RMSE=€{v['rmse_euros']:5.0f}  "
            f"MAPE={v['mape']:5.1f}%  bias=€{v['bias_euros']:+5.0f}"
        )
    print("\nWorst 5 zones (highest MAE):")
    for z in summary["zone_worst_5_mae"]:
        print(
            f"  {z['zone']:<20} n={z['n']:3d}  price~€{z['price_mean']:5.0f}  "
            f"MAE=€{z['mae_euros']:5.0f}  MAPE={z['mape']:4.1f}%"
        )
    print("\nBest 5 zones (lowest MAE):")
    for z in summary["zone_best_5_mae"]:
        print(
            f"  {z['zone']:<20} n={z['n']:3d}  price~€{z['price_mean']:5.0f}  "
            f"MAE=€{z['mae_euros']:5.0f}  MAPE={z['mape']:4.1f}%"
        )


if __name__ == "__main__":
    main()
