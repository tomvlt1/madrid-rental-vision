"""
Plot training/validation curves required by the project rubric:

1. v1 ResNet fine-tune + NN-on-image history (from `models/*_history.json`)
   These are real per-epoch curves from the v1 PyTorch training runs.

2. v2 GB staged curves on the headline `tabular + text + SigLIP` model.
   sklearn's GradientBoostingRegressor exposes `staged_predict()` which
   yields predictions after each tree, so we can plot test-set MAE vs.
   number of trees -- the ensemble equivalent of an epoch curve.

Output:
    notebooks/figures/18_v1_training_curves.png
    v2/figures/v2_gb_staged_curves.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.paths import V2_DATA_DIR, V2_DIR  # noqa: E402

V1_FIG = PROJECT_ROOT / "notebooks" / "figures" / "18_v1_training_curves.png"
V2_FIG = V2_DIR / "figures" / "v2_gb_staged_curves.png"


# -------------------------- v1 NN curves --------------------------

def plot_v1_curves():
    histories = {
        "ResNet-50 fine-tune (layer4 + head)": "models/finetune_history.json",
        "NN — image-only baseline": "models/nn_image_only_history.json",
        "NN — tabular + image": "models/nn_tabular_image_history.json",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (label, path) in zip(axes, histories.items()):
        p = PROJECT_ROOT / path
        if not p.exists():
            ax.set_title(f"{label}\n(history file missing)")
            ax.axis("off")
            continue
        h = json.loads(p.read_text())
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="train", linewidth=2)
        ax.plot(epochs, h["val_loss"], label="val", linewidth=2)
        final = h["val_loss"][-1]
        best = min(h["val_loss"])
        best_ep = int(np.argmin(h["val_loss"])) + 1
        ax.axvline(best_ep, ls="--", lw=0.8, color="grey", alpha=0.6)
        ax.set_title(
            f"{label}\nbest val_loss={best:.3f} @ ep {best_ep}, final={final:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE on log-rent")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("v1 PyTorch training/validation loss curves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    V1_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(V1_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {V1_FIG}")


# ---------------------- v2 GB staged curves -----------------------

NUMERIC_FEATURES = ["sqft_m2", "rooms", "bathrooms", "floor_num", "num_images"]
BOOL_FEATURES = ["elevator", "ac", "terrace", "furnished", "heating", "exterior", "parking", "storage"]
SEED = 42
N_TREES = 500


def build_tab(df):
    df = df.copy()
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(float)
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", dtype=float)
    return (
        pd.concat([df[NUMERIC_FEATURES], df[BOOL_FEATURES], zone_dummies], axis=1).values.astype(np.float32),
        np.log1p(df["price"].values.astype(np.float32)),
    )


def scale_pca(Xtr, Xte, n):
    s = StandardScaler().fit(Xtr)
    p = PCA(n_components=n, random_state=SEED).fit(s.transform(Xtr))
    return p.transform(s.transform(Xtr)), p.transform(s.transform(Xte))


def plot_v2_staged_curves():
    """Train the headline GB model once, dump per-tree validation MAE.

    Uses the v2 unified clean dataset + SigLIP/text embeddings (all
    committed-or-locally-built). On a single 80/20 holdout (not full CV)
    so the figure is cheap and the curve interpretation is simple.
    """
    clean = V2_DATA_DIR / "listings_clean_v2.csv"
    sig_npy = V2_DATA_DIR / "siglip_embeddings.npy"
    sig_idx = V2_DATA_DIR / "siglip_embeddings_index.csv"
    txt_npy = V2_DATA_DIR / "text_embeddings.npy"
    txt_idx = V2_DATA_DIR / "text_embeddings_index.csv"
    if not all(p.exists() for p in (clean, sig_npy, sig_idx, txt_npy, txt_idx)):
        print("v2 data files missing -- skipping v2 GB staged curves figure")
        return

    df = pd.read_csv(clean)
    df["listing_id"] = df["url"].apply(lambda u: u.rstrip("/").split("/")[-1]).astype(str)
    sig = np.load(sig_npy); sig_idx_df = pd.read_csv(sig_idx)
    sig_idx_df["listing_id"] = sig_idx_df["listing_id"].astype(str)
    sig_idx_df["sig_row"] = range(len(sig_idx_df))
    txt = np.load(txt_npy); txt_idx_df = pd.read_csv(txt_idx)
    txt_idx_df["listing_id"] = txt_idx_df["listing_id"].astype(str)
    txt_idx_df["txt_row"] = range(len(txt_idx_df))
    df = (
        df.merge(sig_idx_df[["listing_id", "sig_row"]], on="listing_id", how="inner")
          .merge(txt_idx_df[["listing_id", "txt_row"]], on="listing_id", how="inner")
          .reset_index(drop=True)
    )
    print(f"v2 GB staged curves: joined {len(df)} listings")

    tab, y = build_tab(df)
    X_sig = sig[df["sig_row"].values].astype(np.float32)
    X_txt = txt[df["txt_row"].values].astype(np.float32)

    # 80/20 hold-out
    n = len(df)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]

    tab_s = StandardScaler().fit(tab[tr])
    Xtab_tr, Xtab_te = tab_s.transform(tab[tr]), tab_s.transform(tab[te])
    Xsig_tr, Xsig_te = scale_pca(X_sig[tr], X_sig[te], 50)
    Xtxt_tr, Xtxt_te = scale_pca(X_txt[tr], X_txt[te], 30)

    # Headline (tabular + text + siglip) -- fit and stage
    Xtr_full = np.hstack([Xtab_tr, Xtxt_tr, Xsig_tr])
    Xte_full = np.hstack([Xtab_te, Xtxt_te, Xsig_te])
    Xtr_tab = Xtab_tr; Xte_tab = Xtab_te
    Xtr_tab_sig = np.hstack([Xtab_tr, Xsig_tr]); Xte_tab_sig = np.hstack([Xtab_te, Xsig_te])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pane 1: train vs val MAE across staged trees, headline model
    print("  fitting headline GB and computing staged predictions...")
    gb = GradientBoostingRegressor(
        n_estimators=N_TREES, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=SEED,
    ).fit(Xtr_full, y[tr])

    train_mae, val_mae = [], []
    for tr_pred, te_pred in zip(gb.staged_predict(Xtr_full), gb.staged_predict(Xte_full)):
        train_mae.append(np.mean(np.abs(np.expm1(tr_pred) - np.expm1(y[tr]))))
        val_mae.append(np.mean(np.abs(np.expm1(te_pred) - np.expm1(y[te]))))
    n_trees = np.arange(1, len(train_mae) + 1)
    best_n = int(np.argmin(val_mae)) + 1
    axes[0].plot(n_trees, train_mae, label="train", linewidth=2)
    axes[0].plot(n_trees, val_mae, label="val", linewidth=2)
    axes[0].axvline(best_n, ls="--", color="grey", alpha=0.6)
    axes[0].set_title(
        f"v2 headline GB (tab + text + SigLIP)\n"
        f"best val MAE €{val_mae[best_n - 1]:.0f} @ {best_n} trees",
        fontsize=11,
    )
    axes[0].set_xlabel("number of trees (boosting iterations)")
    axes[0].set_ylabel("MAE (€)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Pane 2: val MAE for three nested models on same axes
    print("  fitting two ablation models for comparison...")
    val_curves = {"tab + text + SigLIP": val_mae}
    for label, Xtr_, Xte_ in [
        ("tab + SigLIP", Xtr_tab_sig, Xte_tab_sig),
        ("tab only",     Xtr_tab,     Xte_tab),
    ]:
        m = GradientBoostingRegressor(
            n_estimators=N_TREES, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=SEED,
        ).fit(Xtr_, y[tr])
        v = [np.mean(np.abs(np.expm1(p) - np.expm1(y[te]))) for p in m.staged_predict(Xte_)]
        val_curves[label] = v

    for label, vc in val_curves.items():
        axes[1].plot(np.arange(1, len(vc) + 1), vc, label=label, linewidth=2)
    axes[1].set_title("v2 ablation: validation MAE vs trees", fontsize=11)
    axes[1].set_xlabel("number of trees")
    axes[1].set_ylabel("val MAE (€)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("v2 Gradient Boosting staged training curves (80/20 hold-out)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    V2_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(V2_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {V2_FIG}")


def main():
    plot_v1_curves()
    plot_v2_staged_curves()


if __name__ == "__main__":
    main()
