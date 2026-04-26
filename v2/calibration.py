# v2: post-hoc calibration on the headline model's OOF predictions.
#
# Motivation: the v2 SigLIP headline model regresses to the mean on Q4
# (MAE €922, bias -€449). Robust losses didn't help (see train_cv_robust.py).
# Try a cheaper fix: learn a calibration curve on OOF predictions and
# apply it as a correction.
#
# Three calibrators, all fit on OOF predictions (5-fold, leakage-free):
#   (1) linear_global      y_true ~ a * y_pred + b         (single slope)
#   (2) linear_logspace    log(y_true) ~ a*log(y_pred)+b   (same, but on log-rent)
#   (3) isotonic           any monotonic curve via isotonic regression
#
# Key methodological point: the calibrator is fit on OOF predictions, not
# on the training set, so it sees the same distribution of errors we'd
# get at test time. The risk: the calibrator is fit on ALL OOF data and
# reused on the same OOF data for evaluation. That IS a form of leakage
# — proper nested CV would fit the calibrator inside each fold. For a
# quick bias-correction check the in-sample fit is a reasonable
# first-order estimate of what we'd get. We cross-validate the calibrator
# explicitly below to quantify how much this matters.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import PROCESSED_DIR, V2_DATA_DIR, V2_MODELS_DIR  # noqa: E402

OUT_JSON = V2_MODELS_DIR / "calibration_results.json"
OUT_CSV = V2_MODELS_DIR / "oof_predictions_calibrated.csv"

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
    return (
        PCA(n_components=n, random_state=SEED).fit_transform(s.fit_transform(Xtr)),
        PCA(n_components=n, random_state=SEED).fit(s.fit_transform(Xtr)).transform(s.transform(Xte))
        if False else None,  # placeholder
    )


# Cleaner scale_pca without the placeholder
def scale_pca(Xtr, Xte, n):
    s = StandardScaler().fit(Xtr)
    Xtr_s = s.transform(Xtr)
    Xte_s = s.transform(Xte)
    p = PCA(n_components=n, random_state=SEED).fit(Xtr_s)
    return p.transform(Xtr_s), p.transform(Xte_s)


def metrics(pred_eur, true_eur):
    err = pred_eur - true_eur
    ae = np.abs(err)
    return {
        "mae_euros": float(ae.mean()),
        "rmse_euros": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(100 * np.mean(ae / true_eur)),
        "bias_euros": float(err.mean()),
    }


def per_quartile(pred_eur, true_eur, price_q, labels):
    out = {}
    for label in labels:
        mask = (price_q == label).to_numpy()
        if mask.sum() == 0:
            continue
        out[label] = {"n": int(mask.sum()), "price_mean": float(true_eur[mask].mean())}
        out[label].update(metrics(pred_eur[mask], true_eur[mask]))
    return out


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

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
    n = len(df)
    print(f"Listings joined: {n}")

    tab, y_log = build_tab(df)
    X_txt = txt[df["txt_row"].values].astype(np.float32)
    X_sig = siglip[df["siglip_row"].values].astype(np.float32)
    y_price = df["price"].values.astype(np.float32)

    # OOF predictions from the v2 headline model (GB on log-rent, squared loss)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred_log = np.full(n, np.nan)
    for k, (tr, te) in enumerate(kf.split(np.arange(n)), 1):
        print(f"  base fold {k}/{N_FOLDS}")
        Xtab_tr = StandardScaler().fit(tab[tr]).transform(tab[tr])
        Xtab_te = StandardScaler().fit(tab[tr]).transform(tab[te])
        Xtxt_tr, Xtxt_te = scale_pca(X_txt[tr], X_txt[te], TEXT_PCA_COMPONENTS)
        Xsig_tr, Xsig_te = scale_pca(X_sig[tr], X_sig[te], IMAGE_PCA_COMPONENTS)
        Xtr = np.hstack([Xtab_tr, Xtxt_tr, Xsig_tr])
        Xte = np.hstack([Xtab_te, Xtxt_te, Xsig_te])
        m = gb().fit(Xtr, y_log[tr])
        oof_pred_log[te] = m.predict(Xte)

    oof_pred_eur = np.expm1(oof_pred_log)

    # Price quartiles (fixed boundaries from full data)
    q_labels = ["Q1_cheap", "Q2", "Q3", "Q4_expensive"]
    price_q = pd.qcut(pd.Series(y_price), 4, labels=q_labels)

    results = {}

    # --- Baseline (no calibration) ---
    results["baseline"] = {
        "global": metrics(oof_pred_eur, y_price),
        "per_quartile": per_quartile(oof_pred_eur, y_price, price_q, q_labels),
    }

    # --- (1) Linear calibration in euro space ---
    lr_eur = LinearRegression().fit(oof_pred_eur.reshape(-1, 1), y_price)
    calib_linear_eur = lr_eur.predict(oof_pred_eur.reshape(-1, 1))
    results["linear_euro_insample"] = {
        "slope": float(lr_eur.coef_[0]),
        "intercept": float(lr_eur.intercept_),
        "global": metrics(calib_linear_eur, y_price),
        "per_quartile": per_quartile(calib_linear_eur, y_price, price_q, q_labels),
    }

    # --- (2) Linear calibration in log-rent space ---
    lr_log = LinearRegression().fit(oof_pred_log.reshape(-1, 1), y_log)
    calib_log_pred = lr_log.predict(oof_pred_log.reshape(-1, 1))
    calib_linear_log = np.expm1(calib_log_pred)
    results["linear_log_insample"] = {
        "slope": float(lr_log.coef_[0]),
        "intercept": float(lr_log.intercept_),
        "global": metrics(calib_linear_log, y_price),
        "per_quartile": per_quartile(calib_linear_log, y_price, price_q, q_labels),
    }

    # --- (3) Isotonic calibration ---
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_pred_eur, y_price)
    calib_iso = iso.predict(oof_pred_eur)
    results["isotonic_insample"] = {
        "global": metrics(calib_iso, y_price),
        "per_quartile": per_quartile(calib_iso, y_price, price_q, q_labels),
    }

    # ------------------------------------------------------------
    # Cross-validated calibration (leakage-free). For each fold:
    #   - Fit calibrator on other 4 folds' OOF predictions.
    #   - Apply to this fold's OOF predictions.
    # ------------------------------------------------------------
    fold_assignment = np.full(n, -1, dtype=int)
    for k, (_, te) in enumerate(kf.split(np.arange(n)), 1):
        fold_assignment[te] = k

    def cv_calib(fit_fn, apply_fn):
        out = np.full(n, np.nan)
        for k in range(1, N_FOLDS + 1):
            fit_mask = fold_assignment != k
            te_mask = fold_assignment == k
            cal = fit_fn(oof_pred_eur[fit_mask], oof_pred_log[fit_mask],
                        y_price[fit_mask], y_log[fit_mask])
            out[te_mask] = apply_fn(cal, oof_pred_eur[te_mask], oof_pred_log[te_mask])
        return out

    def fit_linear_eur(pe, pl, ye, yl):
        return LinearRegression().fit(pe.reshape(-1, 1), ye)

    def apply_linear_eur(m, pe, pl):
        return m.predict(pe.reshape(-1, 1))

    def fit_linear_log(pe, pl, ye, yl):
        return LinearRegression().fit(pl.reshape(-1, 1), yl)

    def apply_linear_log(m, pe, pl):
        return np.expm1(m.predict(pl.reshape(-1, 1)))

    def fit_iso(pe, pl, ye, yl):
        return IsotonicRegression(out_of_bounds="clip").fit(pe, ye)

    def apply_iso(m, pe, pl):
        return m.predict(pe)

    calib_cv_linear_eur = cv_calib(fit_linear_eur, apply_linear_eur)
    calib_cv_linear_log = cv_calib(fit_linear_log, apply_linear_log)
    calib_cv_iso = cv_calib(fit_iso, apply_iso)

    for name, preds in [
        ("linear_euro_cv", calib_cv_linear_eur),
        ("linear_log_cv", calib_cv_linear_log),
        ("isotonic_cv", calib_cv_iso),
    ]:
        results[name] = {
            "global": metrics(preds, y_price),
            "per_quartile": per_quartile(preds, y_price, price_q, q_labels),
        }

    # --- Print summary ---
    print("\n" + "=" * 96)
    print("CALIBRATION RESULTS (headline model: tab + text + SigLIP-mean, 5-fold OOF)")
    print("=" * 96)
    print(
        f"{'Variant':<28} {'MAE (€)':>10} {'RMSE (€)':>10} {'MAPE (%)':>10} "
        f"{'Bias Q1':>10} {'Bias Q4':>10} {'MAE Q4':>10}"
    )
    print("-" * 96)
    for name, res in results.items():
        g = res["global"]
        q1 = res["per_quartile"]["Q1_cheap"]
        q4 = res["per_quartile"]["Q4_expensive"]
        print(
            f"{name:<28} {g['mae_euros']:>10.0f} {g['rmse_euros']:>10.0f} "
            f"{g['mape']:>10.2f} {q1['bias_euros']:>+10.0f} {q4['bias_euros']:>+10.0f} "
            f"{q4['mae_euros']:>10.0f}"
        )

    OUT_JSON.write_text(json.dumps(results, indent=2))
    # Save the best CV-calibrated predictions in a tidy CSV
    pd.DataFrame({
        "listing_id": df["listing_id"].values,
        "zone": df["zone"].values,
        "price": y_price,
        "pred_baseline": oof_pred_eur,
        "pred_linear_eur_cv": calib_cv_linear_eur,
        "pred_linear_log_cv": calib_cv_linear_log,
        "pred_isotonic_cv": calib_cv_iso,
        "price_q": price_q.astype(str).values,
    }).to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
