# Evaluation plots and error analysis

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.models.dataset import load_dataset
from src.config import PROJECT_ROOT

FIGURES_DIR = Path("notebooks/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    data = load_dataset()
    gb_tab = joblib.load(MODELS_DIR / "gb_tabular.joblib")
    gb_full = joblib.load(MODELS_DIR / "gb_tabular_image.joblib")
    pca = joblib.load(MODELS_DIR / "pca.joblib")
    ridge = joblib.load(MODELS_DIR / "ridge_tabular.joblib")

    pred_ridge = ridge.predict(data["X_tab_test"])
    pred_gb_tab = gb_tab.predict(data["X_tab_test"])

    img_pca_test = pca.transform(data["X_img_test"])
    X_combined_test = np.hstack([data["X_tab_test"], img_pca_test])
    pred_gb_full = gb_full.predict(X_combined_test)

    y_test = data["y_test"]
    y_test_price = np.expm1(y_test)

    # 1. Predicted vs actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, preds, title in [
        (axes[0], pred_ridge, "Ridge (Tabular)"),
        (axes[1], pred_gb_tab, "Gradient Boosting (Tabular)"),
        (axes[2], pred_gb_full, "Gradient Boosting (Tabular + Image)"),
    ]:
        p_price = np.expm1(preds)
        ax.scatter(y_test_price, p_price, alpha=0.4, s=15)
        lims = [0, max(y_test_price.max(), p_price.max()) * 1.05]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual Price (EUR)")
        ax.set_ylabel("Predicted Price (EUR)")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("1. Predicted vs actual saved")

    # 2. Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    residuals_tab = np.expm1(pred_gb_tab) - y_test_price
    residuals_full = np.expm1(pred_gb_full) - y_test_price

    axes[0].hist(residuals_tab, bins=40, alpha=0.6, color="steelblue", label="Tabular only", edgecolor="white")
    axes[0].hist(residuals_full, bins=40, alpha=0.6, color="coral", label="Tabular + Image", edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Prediction Error (EUR)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual Distribution")
    axes[0].legend()

    abs_err_tab = np.abs(residuals_tab)
    abs_err_full = np.abs(residuals_full)
    axes[1].hist(abs_err_tab, bins=40, alpha=0.6, color="steelblue", label="Tabular only", edgecolor="white")
    axes[1].hist(abs_err_full, bins=40, alpha=0.6, color="coral", label="Tabular + Image", edgecolor="white")
    axes[1].set_xlabel("Absolute Error (EUR)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Absolute Error Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("2. Residuals saved")

    # 3. Error by zone
    df_test = data["df_test"].copy()
    df_test = df_test.reset_index(drop=True)
    df_test["pred_tab"] = np.expm1(pred_gb_tab)
    df_test["pred_full"] = np.expm1(pred_gb_full)
    df_test["err_tab"] = np.abs(df_test["price"] - df_test["pred_tab"])
    df_test["err_full"] = np.abs(df_test["price"] - df_test["pred_full"])

    zone_errs = df_test.groupby("zone").agg(
        mae_tab=("err_tab", "mean"),
        mae_full=("err_full", "mean"),
        count=("price", "size"),
        median_price=("price", "median"),
    ).sort_values("median_price", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(zone_errs))
    width = 0.35
    ax.bar(x - width/2, zone_errs["mae_tab"], width, label="Tabular only", color="steelblue")
    ax.bar(x + width/2, zone_errs["mae_full"], width, label="Tabular + Image", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(zone_errs.index, rotation=30, ha="right")
    ax.set_ylabel("Mean Absolute Error (EUR)")
    ax.set_title("Prediction Error by Zone")
    ax.legend()

    for i, (_, row) in enumerate(zone_errs.iterrows()):
        ax.text(i, max(row["mae_tab"], row["mae_full"]) + 20,
                f"n={int(row['count'])}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_error_by_zone.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("3. Error by zone saved")

    # 4. Error by price range
    bins = [0, 1200, 2000, 3000, 15001]
    labels = ["<1200", "1200-2000", "2000-3000", ">3000"]
    df_test["price_bin"] = pd.cut(df_test["price"], bins=bins, labels=labels)

    bin_errs = df_test.groupby("price_bin", observed=True).agg(
        mape_tab=("err_tab", lambda x: (x / df_test.loc[x.index, "price"]).mean() * 100),
        mape_full=("err_full", lambda x: (x / df_test.loc[x.index, "price"]).mean() * 100),
        count=("price", "size"),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bin_errs))
    ax.bar(x - width/2, bin_errs["mape_tab"], width, label="Tabular only", color="steelblue")
    ax.bar(x + width/2, bin_errs["mape_full"], width, label="Tabular + Image", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_errs.index)
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Price Range (EUR/month)")
    ax.set_title("Prediction Error by Price Range")
    ax.legend()

    for i, (_, row) in enumerate(bin_errs.iterrows()):
        ax.text(i, max(row["mape_tab"], row["mape_full"]) + 0.5,
                f"n={int(row['count'])}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_error_by_price_range.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("4. Error by price range saved")

    # 5. Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))

    imp = sorted(zip(data["feature_names"], gb_tab.feature_importances_),
                 key=lambda x: x[1])
    names, values = zip(*imp[-15:])
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Gradient Boosting Feature Importance (Tabular Model)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("5. Feature importance saved")

    # 6. Where do images help?
    df_test["improvement"] = df_test["err_tab"] - df_test["err_full"]
    df_test["img_helped"] = df_test["improvement"] > 0

    print("\n" + "=" * 60)
    print("WHERE IMAGES HELP MOST")
    print("=" * 60)

    zone_help = df_test.groupby("zone").agg(
        pct_helped=("img_helped", "mean"),
        avg_improvement=("improvement", "mean"),
        count=("price", "size"),
    ).sort_values("avg_improvement", ascending=False)

    print("\nBy zone:")
    for zone, row in zone_help.iterrows():
        print(f"  {zone:25s}: {row['pct_helped']:5.1%} improved, "
              f"avg EUR {row['avg_improvement']:+.0f}, n={int(row['count'])}")

    print("\nBy price range:")
    bin_help = df_test.groupby("price_bin", observed=True).agg(
        pct_helped=("img_helped", "mean"),
        avg_improvement=("improvement", "mean"),
    )
    for price_bin, row in bin_help.iterrows():
        print(f"  {price_bin:15s}: {row['pct_helped']:5.1%} improved, "
              f"avg EUR {row['avg_improvement']:+.0f}")

    # 7. Model comparison
    with open(MODELS_DIR / "results.json") as f:
        all_results = json.load(f)

    models_to_plot = ["ridge_tabular", "gb_tabular", "gb_tabular_image"]
    model_labels = ["Ridge\n(Tabular)", "Grad. Boost\n(Tabular)", "Grad. Boost\n(Tab + Image)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    r2s = [all_results[m]["r2"] for m in models_to_plot]
    colors = ["steelblue", "steelblue", "coral"]
    axes[0].bar(model_labels, r2s, color=colors)
    axes[0].set_ylabel("R-squared")
    axes[0].set_title("R-squared (higher is better)")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    maes = [all_results[m]["mae_euros"] for m in models_to_plot]
    axes[1].bar(model_labels, maes, color=colors)
    axes[1].set_ylabel("MAE (EUR)")
    axes[1].set_title("Mean Absolute Error (lower is better)")
    for i, v in enumerate(maes):
        axes[1].text(i, v + 10, f"EUR {v:.0f}", ha="center", fontweight="bold")

    mapes = [all_results[m]["mape"] for m in models_to_plot]
    axes[2].bar(model_labels, mapes, color=colors)
    axes[2].set_ylabel("MAPE (%)")
    axes[2].set_title("Mean Abs. Pct. Error (lower is better)")
    for i, v in enumerate(mapes):
        axes[2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n6. Model comparison chart saved")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
