# v2/make_figures.py — produce figures from the OOF predictions for the
# writeup. Three plots:
#   1) Residual histogram with ±MAE band (shows the heavy tail concretely)
#   2) Predicted-vs-actual scatter, colored by price quartile
#   3) |Error| vs price scatter + rolling median (shows error grows with price)
#
# Reads from v2/models/oof_predictions_calibrated.csv (the headline
# tab + text + SigLIP-mean model on all 6,047 listings).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import V2_DIR, V2_MODELS_DIR  # noqa: E402

FIG_DIR = V2_DIR / "figures"
OOF_CSV = V2_MODELS_DIR / "oof_predictions_calibrated.csv"


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(OOF_CSV)

    # Schema: listing_id, zone, price, pred_baseline, pred_linear_eur_cv,
    # pred_linear_log_cv, pred_isotonic_cv, price_q. Use baseline (uncalibrated)
    # so the figures show the model's raw error distribution.
    df["pred_price"] = df["pred_baseline"]
    df["error_eur"] = df["pred_price"] - df["price"]
    df["abs_error_eur"] = df["error_eur"].abs()

    mae = df["abs_error_eur"].mean()
    rmse = float(np.sqrt((df["error_eur"] ** 2).mean()))

    # ---- Figure 1: residual histogram -------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(df["error_eur"], bins=80, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.6)
    ax.axvspan(-mae, mae, color="tab:green", alpha=0.15, label=f"±MAE (€{mae:.0f})")
    ax.axvspan(-rmse, -mae, color="tab:red", alpha=0.05)
    ax.axvspan(mae, rmse, color="tab:red", alpha=0.05, label=f"±RMSE (€{rmse:.0f})")
    ax.set_xlabel("prediction error, € (positive = over-prediction)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution — headline model, 5-fold OOF, N=1,425")
    ax.legend(loc="upper right")
    ax.set_xlim(-4000, 4000)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_histogram.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: predicted vs actual ------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # color by price quartile
    df["price_q"] = pd.qcut(df["price"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    colors = {"Q1": "#1f77b4", "Q2": "#2ca02c", "Q3": "#ff7f0e", "Q4": "#d62728"}
    for q, sub in df.groupby("price_q", observed=True):
        ax.scatter(
            sub["price"], sub["pred_price"],
            s=10, alpha=0.5, label=f"{q} (n={len(sub)})", color=colors[q],
        )
    lim = [0, max(df["price"].max(), df["pred_price"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual rent (€)")
    ax.set_ylabel("predicted rent (€)")
    ax.set_title("Predicted vs actual (5-fold OOF)\nregress-to-mean visible in Q4")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # ---- Figure 3: |error| vs price with rolling median -------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = df.sort_values("price")
    ax.scatter(order["price"], order["abs_error_eur"], s=6, alpha=0.35)
    # rolling median in a sliding window of 100 listings
    window = 100
    med = order["abs_error_eur"].rolling(window, center=True, min_periods=20).median()
    ax.plot(order["price"], med, color="tab:red", lw=2, label=f"rolling median (w={window})")
    ax.axhline(mae, color="tab:green", lw=1, ls="--", label=f"overall MAE €{mae:.0f}")
    ax.set_xlabel("actual rent (€)")
    ax.set_ylabel("|prediction error| (€)")
    ax.set_title("Error grows with price — tail concentrates above ~€3k")
    ax.legend(loc="upper left")
    ax.set_ylim(0, df["abs_error_eur"].quantile(0.99))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "abs_error_vs_price.png", dpi=150)
    plt.close(fig)

    print(f"Wrote 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
