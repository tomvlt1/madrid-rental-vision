# Train all price prediction models (Ridge, GB, NN) with various feature combos

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.dataset import load_dataset
from src.models.networks import TabularImageModel, ImageOnlyModel
from src.config import PROJECT_ROOT, PROCESSED_DIR

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def compute_metrics(preds_log, targets_log):
    rmse_log = np.sqrt(np.mean((preds_log - targets_log) ** 2))

    preds_price = np.expm1(preds_log)
    targets_price = np.expm1(targets_log)
    mae = np.mean(np.abs(preds_price - targets_price))
    mape = np.mean(np.abs(preds_price - targets_price) / targets_price) * 100

    ss_res = np.sum((targets_log - preds_log) ** 2)
    ss_tot = np.sum((targets_log - targets_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"rmse_log": float(rmse_log), "r2": float(r2),
            "mae_euros": float(mae), "mape": float(mape)}


def train_nn(model, train_loader, val_loader, model_name, device,
             lr=5e-4, epochs=300, patience=30):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 3:
                x_tab, x_img, y = [b.to(device) for b in batch]
                pred = model(x_tab, x_img)
            else:
                x, y = [b.to(device) for b in batch]
                pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.train(False)
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_tab, x_img, y = [b.to(device) for b in batch]
                    pred = model(x_tab, x_img)
                else:
                    x, y = [b.to(device) for b in batch]
                    pred = model(x)
                val_losses.append(criterion(pred, y).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(device)

    torch.save(best_state, MODELS_DIR / f"{model_name}.pt")
    with open(MODELS_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f)

    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history


def nn_predict(model, loader, device):
    model.train(False)
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x_tab, x_img, y = [b.to(device) for b in batch]
                pred = model(x_tab, x_img)
            else:
                x, y = [b.to(device) for b in batch]
                pred = model(x)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    data = load_dataset()
    tab_dim = data["X_tab_train"].shape[1]
    print(f"Tabular features: {tab_dim}")
    print(f"Image embedding dim: {data['X_img_train'].shape[1]}")
    print(f"Train: {len(data['y_train'])} | Val: {len(data['y_val'])} | Test: {len(data['y_test'])}\n")

    results = {}

    # ================================================================
    # MODEL 1: Ridge Regression: tabular only
    # ================================================================
    print("=" * 60)
    print("MODEL 1: Ridge Regression (Tabular Only)")
    print("=" * 60)
    ridge = Ridge(alpha=1.0)
    ridge.fit(data["X_tab_train"], data["y_train"])
    preds = ridge.predict(data["X_tab_test"])
    results["ridge_tabular"] = compute_metrics(preds, data["y_test"])
    print(f"  Test: R2={results['ridge_tabular']['r2']:.4f}  "
          f"MAE=EUR {results['ridge_tabular']['mae_euros']:.0f}  "
          f"MAPE={results['ridge_tabular']['mape']:.1f}%")
    joblib.dump(ridge, MODELS_DIR / "ridge_tabular.joblib")

    # ================================================================
    # MODEL 2: Gradient Boosting: tabular only
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 2: Gradient Boosting (Tabular Only)")
    print("=" * 60)
    gb_tab = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    gb_tab.fit(data["X_tab_train"], data["y_train"])
    preds = gb_tab.predict(data["X_tab_test"])
    results["gb_tabular"] = compute_metrics(preds, data["y_test"])
    print(f"  Test: R2={results['gb_tabular']['r2']:.4f}  "
          f"MAE=EUR {results['gb_tabular']['mae_euros']:.0f}  "
          f"MAPE={results['gb_tabular']['mape']:.1f}%")
    joblib.dump(gb_tab, MODELS_DIR / "gb_tabular.joblib")

    print("\n  Top 10 features:")
    imp = sorted(zip(data["feature_names"], gb_tab.feature_importances_),
                 key=lambda x: -x[1])
    for name, score in imp[:10]:
        bar = "#" * int(score * 100)
        print(f"    {name:20s} {score:.3f} {bar}")

    # ================================================================
    # MODEL 3: Gradient Boosting: tabular + PCA image features
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 3: Gradient Boosting (Tabular + PCA Image)")
    print("=" * 60)

    N_COMPONENTS = 50
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    img_pca_train = pca.fit_transform(data["X_img_train"])
    img_pca_val = pca.transform(data["X_img_val"])
    img_pca_test = pca.transform(data["X_img_test"])
    print(f"  PCA: 2048 -> {N_COMPONENTS} dims ({pca.explained_variance_ratio_.sum():.1%} variance)")

    X_combined_train = np.hstack([data["X_tab_train"], img_pca_train])
    X_combined_test = np.hstack([data["X_tab_test"], img_pca_test])

    gb_full = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    gb_full.fit(X_combined_train, data["y_train"])
    preds = gb_full.predict(X_combined_test)
    results["gb_tabular_image"] = compute_metrics(preds, data["y_test"])
    print(f"  Test: R2={results['gb_tabular_image']['r2']:.4f}  "
          f"MAE=EUR {results['gb_tabular_image']['mae_euros']:.0f}  "
          f"MAPE={results['gb_tabular_image']['mape']:.1f}%")
    joblib.dump(gb_full, MODELS_DIR / "gb_tabular_image.joblib")
    joblib.dump(pca, MODELS_DIR / "pca.joblib")

    # ================================================================
    # MODEL 4: Gradient Boosting: tabular + PCA fine-tuned image
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 4: Gradient Boosting (Tabular + Fine-tuned Image)")
    print("=" * 60)

    ft_embeddings = np.load(PROCESSED_DIR / "embeddings_finetuned.npy")
    ft_index = pd.read_csv(PROCESSED_DIR / "embeddings_finetuned_index.csv")
    ft_index["ft_row"] = range(len(ft_index))
    ft_index["listing_id"] = ft_index["listing_id"].astype(str)

    ft_lookup = dict(zip(ft_index["listing_id"], ft_index["ft_row"]))

    def align_ft(df_split, X_tab_split, y_split):
        ids = df_split["listing_id"].astype(str).values
        mask = np.array([lid in ft_lookup for lid in ids])
        ft_rows = np.array([ft_lookup[lid] for lid, m in zip(ids, mask) if m])
        X_ft = ft_embeddings[ft_rows].astype(np.float32)
        X_tab_matched = X_tab_split[mask]
        y_matched = y_split[mask]
        return X_ft, X_tab_matched, y_matched

    X_ft_train, X_tab_train_ft, y_train_ft = align_ft(data["df_train"], data["X_tab_train"], data["y_train"])
    X_ft_val, X_tab_val_ft, y_val_ft = align_ft(data["df_val"], data["X_tab_val"], data["y_val"])
    X_ft_test, X_tab_test_ft, y_test_ft = align_ft(data["df_test"], data["X_tab_test"], data["y_test"])

    print(f"  Fine-tuned embeddings dim: {X_ft_train.shape[1]}")
    print(f"  Matched: train={len(X_ft_train)} val={len(X_ft_val)} test={len(X_ft_test)}")

    ft_scaler = StandardScaler()
    X_ft_train = ft_scaler.fit_transform(X_ft_train)
    X_ft_val = ft_scaler.transform(X_ft_val)
    X_ft_test = ft_scaler.transform(X_ft_test)

    pca_ft = PCA(n_components=N_COMPONENTS, random_state=42)
    ft_pca_train = pca_ft.fit_transform(X_ft_train)
    ft_pca_val = pca_ft.transform(X_ft_val)
    ft_pca_test = pca_ft.transform(X_ft_test)
    print(f"  PCA: {X_ft_train.shape[1]} -> {N_COMPONENTS} dims ({pca_ft.explained_variance_ratio_.sum():.1%} variance)")

    X_combined_ft_train = np.hstack([X_tab_train_ft, ft_pca_train])
    X_combined_ft_test = np.hstack([X_tab_test_ft, ft_pca_test])

    gb_ft = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    gb_ft.fit(X_combined_ft_train, y_train_ft)
    preds = gb_ft.predict(X_combined_ft_test)
    results["gb_tabular_finetuned_image"] = compute_metrics(preds, y_test_ft)
    print(f"  Test: R2={results['gb_tabular_finetuned_image']['r2']:.4f}  "
          f"MAE=EUR {results['gb_tabular_finetuned_image']['mae_euros']:.0f}  "
          f"MAPE={results['gb_tabular_finetuned_image']['mape']:.1f}%")
    joblib.dump(gb_ft, MODELS_DIR / "gb_tabular_finetuned_image.joblib")
    joblib.dump(ft_scaler, MODELS_DIR / "img_scaler_finetuned.joblib")
    joblib.dump(pca_ft, MODELS_DIR / "pca_finetuned.joblib")

    # ================================================================
    # MODEL 5: Gradient Boosting: tabular + PCA text embeddings
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 5: Gradient Boosting (Tabular + Text)")
    print("=" * 60)

    TEXT_PCA_COMPONENTS = 30
    text_embeddings = np.load(PROCESSED_DIR / "text_embeddings.npy")
    text_index = pd.read_csv(PROCESSED_DIR / "text_embeddings_index.csv")
    text_index["text_row"] = range(len(text_index))
    text_index["listing_id"] = text_index["listing_id"].astype(str)

    text_lookup = dict(zip(text_index["listing_id"], text_index["text_row"]))

    def align_text(df_split, X_tab_split, y_split):
        ids = df_split["listing_id"].astype(str).values
        mask = np.array([lid in text_lookup for lid in ids])
        text_rows = np.array([text_lookup[lid] for lid, m in zip(ids, mask) if m])
        X_text = text_embeddings[text_rows].astype(np.float32)
        X_tab_matched = X_tab_split[mask]
        y_matched = y_split[mask]
        return X_text, X_tab_matched, y_matched

    X_text_train, X_tab_train_txt, y_train_txt = align_text(data["df_train"], data["X_tab_train"], data["y_train"])
    X_text_val, X_tab_val_txt, y_val_txt = align_text(data["df_val"], data["X_tab_val"], data["y_val"])
    X_text_test, X_tab_test_txt, y_test_txt = align_text(data["df_test"], data["X_tab_test"], data["y_test"])

    print(f"  Text embeddings dim: {X_text_train.shape[1]}")
    print(f"  Matched: train={len(X_text_train)} val={len(X_text_val)} test={len(X_text_test)}")

    text_scaler = StandardScaler()
    X_text_train = text_scaler.fit_transform(X_text_train)
    X_text_val = text_scaler.transform(X_text_val)
    X_text_test = text_scaler.transform(X_text_test)

    pca_text = PCA(n_components=TEXT_PCA_COMPONENTS, random_state=42)
    text_pca_train = pca_text.fit_transform(X_text_train)
    text_pca_val = pca_text.transform(X_text_val)
    text_pca_test = pca_text.transform(X_text_test)
    print(f"  PCA: {X_text_train.shape[1]} -> {TEXT_PCA_COMPONENTS} dims ({pca_text.explained_variance_ratio_.sum():.1%} variance)")

    X_combined_txt_train = np.hstack([X_tab_train_txt, text_pca_train])
    X_combined_txt_test = np.hstack([X_tab_test_txt, text_pca_test])

    gb_txt = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    gb_txt.fit(X_combined_txt_train, y_train_txt)
    preds = gb_txt.predict(X_combined_txt_test)
    results["gb_tabular_text"] = compute_metrics(preds, y_test_txt)
    print(f"  Test: R2={results['gb_tabular_text']['r2']:.4f}  "
          f"MAE=EUR {results['gb_tabular_text']['mae_euros']:.0f}  "
          f"MAPE={results['gb_tabular_text']['mape']:.1f}%")
    joblib.dump(gb_txt, MODELS_DIR / "gb_tabular_text.joblib")
    joblib.dump(text_scaler, MODELS_DIR / "text_scaler.joblib")
    joblib.dump(pca_text, MODELS_DIR / "pca_text.joblib")

    # ================================================================
    # MODEL 6: Gradient Boosting: tabular + text + fine-tuned image
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 6: Gradient Boosting (Tabular + Text + Fine-tuned Image)")
    print("=" * 60)

    # only keep listings that have both text and fine-tuned image embeddings
    def align_all(df_split, X_tab_split, y_split):
        ids = df_split["listing_id"].astype(str).values
        mask = np.array([lid in text_lookup and lid in ft_lookup for lid in ids])
        text_rows = np.array([text_lookup[lid] for lid, m in zip(ids, mask) if m])
        ft_rows_arr = np.array([ft_lookup[lid] for lid, m in zip(ids, mask) if m])
        X_text = text_embeddings[text_rows].astype(np.float32)
        X_ft = ft_embeddings[ft_rows_arr].astype(np.float32)
        X_tab_matched = X_tab_split[mask]
        y_matched = y_split[mask]
        return X_text, X_ft, X_tab_matched, y_matched

    X_text_all_train, X_ft_all_train, X_tab_all_train, y_all_train = align_all(data["df_train"], data["X_tab_train"], data["y_train"])
    X_text_all_val, X_ft_all_val, X_tab_all_val, y_all_val = align_all(data["df_val"], data["X_tab_val"], data["y_val"])
    X_text_all_test, X_ft_all_test, X_tab_all_test, y_all_test = align_all(data["df_test"], data["X_tab_test"], data["y_test"])

    print(f"  Matched (all modalities): train={len(y_all_train)} val={len(y_all_val)} test={len(y_all_test)}")

    # reuse scalers from models 4 & 5
    text_all_train_scaled = text_scaler.transform(X_text_all_train)
    text_all_val_scaled = text_scaler.transform(X_text_all_val)
    text_all_test_scaled = text_scaler.transform(X_text_all_test)
    text_all_pca_train = pca_text.transform(text_all_train_scaled)
    text_all_pca_val = pca_text.transform(text_all_val_scaled)
    text_all_pca_test = pca_text.transform(text_all_test_scaled)

    ft_all_train_scaled = ft_scaler.transform(X_ft_all_train)
    ft_all_val_scaled = ft_scaler.transform(X_ft_all_val)
    ft_all_test_scaled = ft_scaler.transform(X_ft_all_test)
    ft_all_pca_train = pca_ft.transform(ft_all_train_scaled)
    ft_all_pca_val = pca_ft.transform(ft_all_val_scaled)
    ft_all_pca_test = pca_ft.transform(ft_all_test_scaled)

    X_all_train = np.hstack([X_tab_all_train, text_all_pca_train, ft_all_pca_train])
    X_all_test = np.hstack([X_tab_all_test, text_all_pca_test, ft_all_pca_test])

    gb_all = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    gb_all.fit(X_all_train, y_all_train)
    preds = gb_all.predict(X_all_test)
    results["gb_tabular_text_finetuned_image"] = compute_metrics(preds, y_all_test)
    print(f"  Test: R2={results['gb_tabular_text_finetuned_image']['r2']:.4f}  "
          f"MAE=EUR {results['gb_tabular_text_finetuned_image']['mae_euros']:.0f}  "
          f"MAPE={results['gb_tabular_text_finetuned_image']['mape']:.1f}%")
    joblib.dump(gb_all, MODELS_DIR / "gb_tabular_text_finetuned_image.joblib")

    # ================================================================
    # MODEL 7: Neural Network: tabular + image (full embeddings)
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 7: Neural Network (Tabular + Image)")
    print("=" * 60)

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    full_train = DataLoader(TensorDataset(
        to_t(data["X_tab_train"]), to_t(data["X_img_train"]), to_t(data["y_train"])),
        batch_size=64, shuffle=True)
    full_val = DataLoader(TensorDataset(
        to_t(data["X_tab_val"]), to_t(data["X_img_val"]), to_t(data["y_val"])),
        batch_size=64)
    full_test = DataLoader(TensorDataset(
        to_t(data["X_tab_test"]), to_t(data["X_img_test"]), to_t(data["y_test"])),
        batch_size=64)

    model_full = TabularImageModel(tab_dim=tab_dim)
    model_full, hist_full = train_nn(model_full, full_train, full_val, "nn_tabular_image", device)
    preds, targets = nn_predict(model_full, full_test, device)
    results["nn_tabular_image"] = compute_metrics(preds, targets)
    print(f"  Test: R2={results['nn_tabular_image']['r2']:.4f}  "
          f"MAE=EUR {results['nn_tabular_image']['mae_euros']:.0f}  "
          f"MAPE={results['nn_tabular_image']['mape']:.1f}%")

    # ================================================================
    # MODEL 8: Neural Network: image only (ablation)
    # ================================================================
    print("\n" + "=" * 60)
    print("MODEL 8: Neural Network (Image Only: Ablation)")
    print("=" * 60)

    img_train = DataLoader(TensorDataset(to_t(data["X_img_train"]), to_t(data["y_train"])),
                           batch_size=64, shuffle=True)
    img_val = DataLoader(TensorDataset(to_t(data["X_img_val"]), to_t(data["y_val"])),
                         batch_size=64)
    img_test = DataLoader(TensorDataset(to_t(data["X_img_test"]), to_t(data["y_test"])),
                          batch_size=64)

    model_img = ImageOnlyModel()
    model_img, hist_img = train_nn(model_img, img_train, img_val, "nn_image_only", device)
    preds, targets = nn_predict(model_img, img_test, device)
    results["nn_image_only"] = compute_metrics(preds, targets)
    print(f"  Test: R2={results['nn_image_only']['r2']:.4f}  "
          f"MAE=EUR {results['nn_image_only']['mae_euros']:.0f}  "
          f"MAPE={results['nn_image_only']['mape']:.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<40} {'R2':>8} {'MAE (EUR)':>10} {'MAPE':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<40} {m['r2']:>8.4f} {m['mae_euros']:>10.0f} {m['mape']:>7.1f}%")

    delta_gb = results["gb_tabular_image"]["r2"] - results["gb_tabular"]["r2"]
    delta_gb_mae = results["gb_tabular"]["mae_euros"] - results["gb_tabular_image"]["mae_euros"]
    print(f"\nGB image delta:  R2 {delta_gb:+.4f}, MAE {-delta_gb_mae:+.0f} EUR")

    delta_ft = results["gb_tabular_finetuned_image"]["r2"] - results["gb_tabular"]["r2"]
    delta_ft_mae = results["gb_tabular"]["mae_euros"] - results["gb_tabular_finetuned_image"]["mae_euros"]
    print(f"GB fine-tuned image delta:  R2 {delta_ft:+.4f}, MAE {-delta_ft_mae:+.0f} EUR")

    delta_finetune_vs_pretrained = results["gb_tabular_finetuned_image"]["r2"] - results["gb_tabular_image"]["r2"]
    delta_finetune_vs_pretrained_mae = results["gb_tabular_image"]["mae_euros"] - results["gb_tabular_finetuned_image"]["mae_euros"]
    print(f"Fine-tuned vs pretrained:  R2 {delta_finetune_vs_pretrained:+.4f}, MAE {-delta_finetune_vs_pretrained_mae:+.0f} EUR")

    delta_text = results["gb_tabular_text"]["r2"] - results["gb_tabular"]["r2"]
    delta_text_mae = results["gb_tabular"]["mae_euros"] - results["gb_tabular_text"]["mae_euros"]
    print(f"GB text delta:  R2 {delta_text:+.4f}, MAE {-delta_text_mae:+.0f} EUR")

    delta_all = results["gb_tabular_text_finetuned_image"]["r2"] - results["gb_tabular"]["r2"]
    delta_all_mae = results["gb_tabular"]["mae_euros"] - results["gb_tabular_text_finetuned_image"]["mae_euros"]
    print(f"GB text + fine-tuned image delta:  R2 {delta_all:+.4f}, MAE {-delta_all_mae:+.0f} EUR")

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(data["tab_scaler"], MODELS_DIR / "tab_scaler.joblib")
    joblib.dump(data["img_scaler"], MODELS_DIR / "img_scaler.joblib")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(data["feature_names"], f)
    with open(MODELS_DIR / "config.json", "w") as f:
        json.dump({"tab_dim": tab_dim, "img_dim": 2048, "pca_components": N_COMPONENTS,
                    "pca_finetuned_components": N_COMPONENTS,
                    "text_dim": 384, "pca_text_components": TEXT_PCA_COMPONENTS}, f)

    print(f"\nAll models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
