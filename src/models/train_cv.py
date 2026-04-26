# 5-fold cross-validation over the same ablation models train.py fits.
# Reports each metric as mean ± std across folds so the headline R² isn't
# a single-seed lucky/unlucky draw. Skips the NN models (they don't
# converge reliably at this sample size).
#
# Data-leakage note: we fix scalers and PCA per-fold on the TRAIN split
# only. The ResNet fine-tune itself runs once against splits.json and
# those embeddings are reused across folds: so this CV measures the
# GB/Ridge tail of the pipeline's stability, not the ResNet's. That's
# the right scope for "is our R² a lucky seed?"

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.dataset import BOOL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = PROJECT_ROOT / "models"
CV_RESULTS_FILE = MODELS_DIR / "cv_results.json"

N_FOLDS = 5
SEED = 42
IMAGE_PCA_COMPONENTS = 50
TEXT_PCA_COMPONENTS = 30


def compute_metrics(preds_log: np.ndarray, targets_log: np.ndarray) -> dict:
    rmse_log = float(np.sqrt(np.mean((preds_log - targets_log) ** 2)))
    preds_price = np.expm1(preds_log)
    targets_price = np.expm1(targets_log)
    mae = float(np.mean(np.abs(preds_price - targets_price)))
    mape = float(np.mean(np.abs(preds_price - targets_price) / targets_price) * 100)
    ss_res = np.sum((targets_log - preds_log) ** 2)
    ss_tot = np.sum((targets_log - targets_log.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return {"r2": r2, "mae_euros": mae, "mape": mape, "rmse_log": rmse_log}


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
    """Build the full tabular feature matrix + log-price target."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    tab = pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1)
    return tab.values.astype(np.float32), list(tab.columns), np.log1p(
        df["price"].values.astype(np.float32)
    )


def load_all():
    """Load clean listings + all three embedding tables aligned by listing_id.
    Returns dict with np arrays already merged on listing_id."""
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)

    # frozen image embeddings + index
    img = np.load(PROCESSED_DIR / "embeddings.npy")
    img_idx = pd.read_csv(PROCESSED_DIR / "embeddings_index.csv")
    img_idx["listing_id"] = img_idx["listing_id"].astype(str)
    img_idx["img_row"] = range(len(img_idx))

    # fine-tuned image embeddings + index
    ft = np.load(PROCESSED_DIR / "embeddings_finetuned.npy")
    ft_idx = pd.read_csv(PROCESSED_DIR / "embeddings_finetuned_index.csv")
    ft_idx["listing_id"] = ft_idx["listing_id"].astype(str)
    ft_idx["ft_row"] = range(len(ft_idx))

    # text embeddings + index
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


def fold_metrics(data: dict, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Fit every ablation model on train_idx, evaluate on test_idx."""
    y_train = data["y"][train_idx]
    y_test = data["y"][test_idx]

    # Tabular: scale per fold
    tab_scaler = StandardScaler()
    X_tab_tr = tab_scaler.fit_transform(data["tab"][train_idx])
    X_tab_te = tab_scaler.transform(data["tab"][test_idx])

    # Frozen image: scale + PCA per fold
    img_scaler = StandardScaler()
    X_img_tr = img_scaler.fit_transform(data["X_img"][train_idx])
    X_img_te = img_scaler.transform(data["X_img"][test_idx])
    pca_img = PCA(n_components=IMAGE_PCA_COMPONENTS, random_state=SEED)
    X_img_tr = pca_img.fit_transform(X_img_tr)
    X_img_te = pca_img.transform(X_img_te)

    # Fine-tuned image: separate scaler + PCA
    ft_scaler = StandardScaler()
    X_ft_tr = ft_scaler.fit_transform(data["X_ft"][train_idx])
    X_ft_te = ft_scaler.transform(data["X_ft"][test_idx])
    pca_ft = PCA(n_components=IMAGE_PCA_COMPONENTS, random_state=SEED)
    X_ft_tr = pca_ft.fit_transform(X_ft_tr)
    X_ft_te = pca_ft.transform(X_ft_te)

    # Text: scale + PCA
    txt_scaler = StandardScaler()
    X_txt_tr = txt_scaler.fit_transform(data["X_txt"][train_idx])
    X_txt_te = txt_scaler.transform(data["X_txt"][test_idx])
    pca_txt = PCA(n_components=TEXT_PCA_COMPONENTS, random_state=SEED)
    X_txt_tr = pca_txt.fit_transform(X_txt_tr)
    X_txt_te = pca_txt.transform(X_txt_te)

    fold_results = {}

    # Ridge (tabular)
    m = Ridge(alpha=1.0).fit(X_tab_tr, y_train)
    fold_results["ridge_tabular"] = compute_metrics(m.predict(X_tab_te), y_test)

    # GB tabular
    m = gb().fit(X_tab_tr, y_train)
    fold_results["gb_tabular"] = compute_metrics(m.predict(X_tab_te), y_test)

    # GB + frozen image
    m = gb().fit(np.hstack([X_tab_tr, X_img_tr]), y_train)
    fold_results["gb_tabular_image"] = compute_metrics(
        m.predict(np.hstack([X_tab_te, X_img_te])), y_test
    )

    # GB + fine-tuned image
    m = gb().fit(np.hstack([X_tab_tr, X_ft_tr]), y_train)
    fold_results["gb_tabular_finetuned_image"] = compute_metrics(
        m.predict(np.hstack([X_tab_te, X_ft_te])), y_test
    )

    # GB + text
    m = gb().fit(np.hstack([X_tab_tr, X_txt_tr]), y_train)
    fold_results["gb_tabular_text"] = compute_metrics(
        m.predict(np.hstack([X_tab_te, X_txt_te])), y_test
    )

    # GB + text + fine-tuned image (the headline model)
    m = gb().fit(np.hstack([X_tab_tr, X_txt_tr, X_ft_tr]), y_train)
    fold_results["gb_tabular_text_finetuned_image"] = compute_metrics(
        m.predict(np.hstack([X_tab_te, X_txt_te, X_ft_te])), y_test
    )

    return fold_results


def aggregate(per_fold: list[dict]) -> dict:
    """Aggregate per-fold dicts into mean ± std per (model, metric)."""
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
    data = load_all()
    n = len(data["y"])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold = []
    for k, (train_idx, test_idx) in enumerate(kf.split(np.arange(n)), 1):
        print(f"\n=== Fold {k}/{N_FOLDS}: train={len(train_idx)} test={len(test_idx)} ===")
        fold_res = fold_metrics(data, train_idx, test_idx)
        for model, m in fold_res.items():
            print(
                f"  {model:38s} R²={m['r2']:.4f}  "
                f"MAE=€{m['mae_euros']:6.0f}  MAPE={m['mape']:5.2f}%"
            )
        per_fold.append(fold_res)

    agg = aggregate(per_fold)

    # Summary table
    print("\n" + "=" * 78)
    print(f"5-FOLD CROSS-VALIDATION SUMMARY (mean ± std across {N_FOLDS} folds)")
    print("=" * 78)
    print(f"{'Model':<40} {'R²':>16} {'MAE (€)':>14} {'MAPE (%)':>10}")
    print("-" * 78)
    for model, m in agg.items():
        r2 = f"{m['r2_mean']:.4f}±{m['r2_std']:.4f}"
        mae = f"{m['mae_euros_mean']:.0f}±{m['mae_euros_std']:.0f}"
        mape = f"{m['mape_mean']:.2f}±{m['mape_std']:.2f}"
        print(f"{model:<40} {r2:>16} {mae:>14} {mape:>10}")

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
