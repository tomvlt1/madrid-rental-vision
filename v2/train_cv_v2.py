# v2: 5-fold CV with the full ablation grid + euro-RMSE alongside MAE.
#
# Changes vs src/models/train_cv.py:
#   (1) compute_metrics() now returns rmse_euros (in addition to rmse_log).
#   (2) The ablation grid is the full 2^3 - 1 = 7 non-empty subsets of
#       {tabular, text, image}, for both frozen-ResNet and fine-tuned-ResNet
#       image embeddings where applicable.
#   (3) Writes to v2/models/cv_results_v2.json so the original
#       models/cv_results.json is untouched.
#
# Still reads data/processed/ (shared, read-only). Same KFold + seed as v1.

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

from v2.paths import PROCESSED_DIR, V2_MODELS_DIR  # noqa: E402

CV_RESULTS_FILE = V2_MODELS_DIR / "cv_results_v2.json"

# Inlined so v2 is self-contained (identical to src/models/dataset.py in
# both the public-release repo and the ai2 dev tree).
NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = [
    "elevator",
    "ac",
    "terrace",
    "furnished",
    "heating",
    "exterior",
    "parking",
    "storage",
]

N_FOLDS = 5
SEED = 42
IMAGE_PCA_COMPONENTS = 50
TEXT_PCA_COMPONENTS = 30


def compute_metrics(preds_log: np.ndarray, targets_log: np.ndarray) -> dict:
    """Metrics in both log-space and euro-space.

    rmse_log:    RMSE of log-rent residuals (what the GB optimizes).
    rmse_euros:  RMSE after expm1 back to euros — comparable units with MAE.
    mae_euros:   MAE in euros.
    mape:        Mean absolute percent error (%).
    r2:          R² computed in log space (same as what the GB optimizes).
    """
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


def gb(random_state: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=random_state,
    )


def build_features(df: pd.DataFrame):
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    tab = pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1)
    return (
        tab.values.astype(np.float32),
        list(tab.columns),
        np.log1p(df["price"].values.astype(np.float32)),
    )


def load_all():
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)

    img = np.load(PROCESSED_DIR / "embeddings.npy")
    img_idx = pd.read_csv(PROCESSED_DIR / "embeddings_index.csv")
    img_idx["listing_id"] = img_idx["listing_id"].astype(str)
    img_idx["img_row"] = range(len(img_idx))

    ft = np.load(PROCESSED_DIR / "embeddings_finetuned.npy")
    ft_idx = pd.read_csv(PROCESSED_DIR / "embeddings_finetuned_index.csv")
    ft_idx["listing_id"] = ft_idx["listing_id"].astype(str)
    ft_idx["ft_row"] = range(len(ft_idx))

    txt = np.load(PROCESSED_DIR / "text_embeddings.npy")
    txt_idx = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    txt_idx["listing_id"] = txt_idx["listing_id"].astype(str)
    txt_idx["txt_row"] = range(len(txt_idx))

    df = (
        df.merge(img_idx[["listing_id", "img_row"]], on="listing_id", how="inner")
          .merge(ft_idx[["listing_id", "ft_row"]], on="listing_id", how="inner")
          .merge(txt_idx[["listing_id", "txt_row"]], on="listing_id", how="inner")
          .reset_index(drop=True)
    )
    print(f"Listings with all three embeddings: {len(df)}")

    tab, feature_names, y = build_features(df)
    X_img = img[df["img_row"].values].astype(np.float32)
    X_ft = ft[df["ft_row"].values].astype(np.float32)
    X_txt = txt[df["txt_row"].values].astype(np.float32)

    return {
        "tab": tab,
        "X_img": X_img,
        "X_ft": X_ft,
        "X_txt": X_txt,
        "y": y,
        "feature_names": feature_names,
    }


def fit_scaler_pca(X_tr, X_te, n_components):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    pca = PCA(n_components=n_components, random_state=SEED)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    return X_tr_p, X_te_p


def fold_metrics(data: dict, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    y_train = data["y"][train_idx]
    y_test = data["y"][test_idx]

    tab_scaler = StandardScaler()
    X_tab_tr = tab_scaler.fit_transform(data["tab"][train_idx])
    X_tab_te = tab_scaler.transform(data["tab"][test_idx])

    X_img_tr, X_img_te = fit_scaler_pca(
        data["X_img"][train_idx], data["X_img"][test_idx], IMAGE_PCA_COMPONENTS
    )
    X_ft_tr, X_ft_te = fit_scaler_pca(
        data["X_ft"][train_idx], data["X_ft"][test_idx], IMAGE_PCA_COMPONENTS
    )
    X_txt_tr, X_txt_te = fit_scaler_pca(
        data["X_txt"][train_idx], data["X_txt"][test_idx], TEXT_PCA_COMPONENTS
    )

    fold_results = {}

    def run(name, Xtr, Xte):
        m = gb().fit(Xtr, y_train)
        fold_results[name] = compute_metrics(m.predict(Xte), y_test)

    m = Ridge(alpha=1.0).fit(X_tab_tr, y_train)
    fold_results["ridge_tabular"] = compute_metrics(m.predict(X_tab_te), y_test)

    # Full 2^3 - 1 subsets of {tab, text, image}, with image split into
    # frozen and fine-tuned variants everywhere image appears.
    run("gb_tabular", X_tab_tr, X_tab_te)
    run("gb_text", X_txt_tr, X_txt_te)
    run("gb_image_frozen", X_img_tr, X_img_te)
    run("gb_image_finetuned", X_ft_tr, X_ft_te)
    run(
        "gb_tabular_text",
        np.hstack([X_tab_tr, X_txt_tr]),
        np.hstack([X_tab_te, X_txt_te]),
    )
    run(
        "gb_tabular_image_frozen",
        np.hstack([X_tab_tr, X_img_tr]),
        np.hstack([X_tab_te, X_img_te]),
    )
    run(
        "gb_tabular_image_finetuned",
        np.hstack([X_tab_tr, X_ft_tr]),
        np.hstack([X_tab_te, X_ft_te]),
    )
    run(
        "gb_text_image_frozen",
        np.hstack([X_txt_tr, X_img_tr]),
        np.hstack([X_txt_te, X_img_te]),
    )
    run(
        "gb_text_image_finetuned",
        np.hstack([X_txt_tr, X_ft_tr]),
        np.hstack([X_txt_te, X_ft_te]),
    )
    run(
        "gb_tabular_text_image_frozen",
        np.hstack([X_tab_tr, X_txt_tr, X_img_tr]),
        np.hstack([X_tab_te, X_txt_te, X_img_te]),
    )
    run(
        "gb_tabular_text_image_finetuned",
        np.hstack([X_tab_tr, X_txt_tr, X_ft_tr]),
        np.hstack([X_tab_te, X_txt_te, X_ft_te]),
    )

    return fold_results


def aggregate(per_fold: list[dict]) -> dict:
    out = {}
    if not per_fold:
        return out
    model_names = list(per_fold[0].keys())
    metric_names = list(per_fold[0][model_names[0]].keys())
    for model in model_names:
        out[model] = {}
        for metric in metric_names:
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
        fold_res = fold_metrics(data, train_idx, test_idx)
        for model, m in fold_res.items():
            print(
                f"  {model:38s} R²={m['r2']:.4f}  "
                f"MAE=€{m['mae_euros']:6.0f}  RMSE=€{m['rmse_euros']:6.0f}  "
                f"MAPE={m['mape']:5.2f}%"
            )
        per_fold.append(fold_res)

    agg = aggregate(per_fold)

    print("\n" + "=" * 100)
    print(f"V2 5-FOLD CV — full ablation grid (mean ± std across {N_FOLDS} folds)")
    print("=" * 100)
    print(
        f"{'Model':<38} {'R²':>16} {'MAE (€)':>14} "
        f"{'RMSE (€)':>14} {'MAPE (%)':>11}"
    )
    print("-" * 100)
    for model, m in agg.items():
        r2 = f"{m['r2_mean']:.4f}±{m['r2_std']:.4f}"
        mae = f"{m['mae_euros_mean']:.0f}±{m['mae_euros_std']:.0f}"
        rmse = f"{m['rmse_euros_mean']:.0f}±{m['rmse_euros_std']:.0f}"
        mape = f"{m['mape_mean']:.2f}±{m['mape_std']:.2f}"
        print(f"{model:<38} {r2:>16} {mae:>14} {rmse:>14} {mape:>11}")

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
