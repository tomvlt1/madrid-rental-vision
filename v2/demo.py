"""
End-to-end demo on the bundled sample dataset.

Runs the full v2 pipeline on the 50-row synthetic listings_clean_sample.csv
that ships with the repo. Lets the instructor (or a teammate) verify the
code executes correctly without needing the real Idealista dataset.

Because the sample doesn't ship with image embeddings (no real photos),
this demo:
  1. Reads data/processed/listings_clean_sample.csv
  2. Synthesizes Gaussian "image" + "text" embeddings of the right shape,
     so the pipeline executes against realistic feature dimensions
  3. Runs the same 5-fold CV on tabular + text + SigLIP
  4. Prints the synthetic-data CV table
  5. Prints, for reference, the actual headline numbers from the real-data
     run that ships in v2/models/cv_results_full_ablation.json

Usage:
    python v2/demo.py
"""

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

SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean_sample.csv"
REAL_RESULTS = PROJECT_ROOT / "v2" / "models" / "cv_results_full_ablation.json"

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = [
    "elevator", "ac", "terrace", "furnished",
    "heating", "exterior", "parking", "storage",
]
SEED = 42


def gb():
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=3, random_state=SEED,
    )


def metrics(pred_log, true_log):
    rmse_log = float(np.sqrt(np.mean((pred_log - true_log) ** 2)))
    pred_e = np.expm1(pred_log)
    true_e = np.expm1(true_log)
    rmse_e = float(np.sqrt(np.mean((pred_e - true_e) ** 2)))
    mae = float(np.mean(np.abs(pred_e - true_e)))
    mape = float(np.mean(np.abs(pred_e - true_e) / true_e) * 100)
    ss_res = np.sum((true_log - pred_log) ** 2)
    ss_tot = np.sum((true_log - true_log.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return {"r2": r2, "mae_euros": mae, "rmse_euros": rmse_e, "mape": mape}


def build_features(df, rng):
    df = df.copy()
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    tab = pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1)
    n = len(df)
    # Synthetic SigLIP embeddings (correct dim 768) and text embeddings (384)
    # Anchored on log-price so the demo actually shows the model learning something.
    log_price = np.log1p(df["price"].values).reshape(-1, 1)
    sig = rng.normal(size=(n, 768)).astype(np.float32) * 0.5
    sig[:, :8] += log_price * 0.3   # plant a bit of signal in first dims
    txt = rng.normal(size=(n, 384)).astype(np.float32) * 0.5
    txt[:, :4] += log_price * 0.2
    return (
        tab.values.astype(np.float32),
        sig,
        txt,
        np.log1p(df["price"].values.astype(np.float32)),
    )


def scale_pca(Xtr, Xte, n_components):
    s = StandardScaler().fit(Xtr)
    Xtr_s = s.transform(Xtr); Xte_s = s.transform(Xte)
    n_components = min(n_components, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    p = PCA(n_components=n_components, random_state=SEED).fit(Xtr_s)
    return p.transform(Xtr_s), p.transform(Xte_s)


def run_fold(tab, sig, txt, y, tr, te):
    tab_scaler = StandardScaler().fit(tab[tr])
    Xtab_tr, Xtab_te = tab_scaler.transform(tab[tr]), tab_scaler.transform(tab[te])
    Xsig_tr, Xsig_te = scale_pca(sig[tr], sig[te], 20)
    Xtxt_tr, Xtxt_te = scale_pca(txt[tr], txt[te], 10)
    out = {}

    def run(name, Xtr, Xte):
        m = gb().fit(Xtr, y[tr])
        out[name] = metrics(m.predict(Xte), y[te])

    run("gb_tabular",            Xtab_tr,                                  Xtab_te)
    run("gb_text",               Xtxt_tr,                                  Xtxt_te)
    run("gb_siglip",             Xsig_tr,                                  Xsig_te)
    run("gb_tabular_text",       np.hstack([Xtab_tr, Xtxt_tr]),            np.hstack([Xtab_te, Xtxt_te]))
    run("gb_tabular_siglip",     np.hstack([Xtab_tr, Xsig_tr]),            np.hstack([Xtab_te, Xsig_te]))
    run("gb_text_siglip",        np.hstack([Xtxt_tr, Xsig_tr]),            np.hstack([Xtxt_te, Xsig_te]))
    run("gb_tabular_text_siglip", np.hstack([Xtab_tr, Xtxt_tr, Xsig_tr]),  np.hstack([Xtab_te, Xtxt_te, Xsig_te]))
    return out


def aggregate(per_fold):
    out = {}
    for name in per_fold[0].keys():
        out[name] = {}
        for metric in per_fold[0][name].keys():
            v = np.array([f[name][metric] for f in per_fold])
            out[name][metric] = float(v.mean())
            out[name][f"{metric}_std"] = float(v.std(ddof=1))
    return out


def main():
    if not SAMPLE_CSV.exists():
        print(f"ERROR: {SAMPLE_CSV} not found.")
        print("Run: python scripts/make_sample_dataset.py")
        sys.exit(1)

    print("=" * 78)
    print("v2 PIPELINE DEMO — running on bundled 50-row sample dataset")
    print("=" * 78)
    df = pd.read_csv(SAMPLE_CSV)
    print(f"loaded {len(df)} sample listings from {SAMPLE_CSV.name}")
    print(f"price range: €{int(df['price'].min())}-€{int(df['price'].max())}, "
          f"mean €{int(df['price'].mean())}\n")

    rng = np.random.default_rng(SEED)
    tab, sig, txt, y = build_features(df, rng)
    print(f"features built: tab={tab.shape}, siglip-mock={sig.shape}, text-mock={txt.shape}")

    n = len(y)
    n_folds = min(5, n // 6)  # ensure fold-train is large enough
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    per_fold = []
    for k, (tr, te) in enumerate(kf.split(np.arange(n)), 1):
        print(f"  fold {k}/{n_folds}: train={len(tr)} test={len(te)}")
        per_fold.append(run_fold(tab, sig, txt, y, tr, te))
    agg = aggregate(per_fold)

    print()
    print("-" * 78)
    print(f"SAMPLE-DATA CV RESULTS ({n_folds}-fold, n={n} synthetic listings)")
    print("-" * 78)
    print(f"{'Model':<32} {'R²':>10} {'MAE (€)':>12} {'RMSE (€)':>12} {'MAPE (%)':>10}")
    print("-" * 78)
    for name, m in agg.items():
        print(f"{name:<32} {m['r2']:>10.3f} {m['mae_euros']:>12.0f} "
              f"{m['rmse_euros']:>12.0f} {m['mape']:>10.2f}")

    print()
    print("=" * 78)
    print("FOR REFERENCE — actual numbers from the real-data run (6,047 listings)")
    print("=" * 78)
    print("Source: v2/models/cv_results_full_ablation.json (committed, no recompute needed)")
    if REAL_RESULTS.exists():
        real = json.loads(REAL_RESULTS.read_text())
        print(f"\n{'Model':<32} {'R²':>16} {'MAE (€)':>12} {'RMSE (€)':>12} {'MAPE (%)':>10}")
        print("-" * 84)
        for name, m in real["models"].items():
            r2 = f"{m['r2_mean']:.4f}±{m['r2_std']:.4f}"
            mae = f"{m['mae_euros_mean']:.0f}±{m['mae_euros_std']:.0f}"
            rmse = f"{m['rmse_euros_mean']:.0f}±{m['rmse_euros_std']:.0f}"
            mape = f"{m['mape_mean']:.2f}"
            print(f"{name:<32} {r2:>16} {mae:>12} {rmse:>12} {mape:>10}")

    print()
    print("Demo complete. Synthetic numbers above prove the pipeline executes;")
    print("real numbers prove the system actually works on real Madrid data.")


if __name__ == "__main__":
    main()
