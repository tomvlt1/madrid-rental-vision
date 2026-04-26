# v2: Learned attention pooling over per-photo image embeddings.
#
# The v1 pipeline mean-pools all photos in a listing. This treats every
# photo as equally informative, which is obviously wrong: a hero kitchen
# shot carries more signal than a blurry closet photo. Attention pooling
# lets the model learn per-photo weights.
#
# Architecture (additive/Bahdanau attention):
#   - Photo embeddings:  (k, D) where D = 768 for SigLIP-base
#   - Attention score:   score_i = v^T tanh(W * emb_i)   (W: D->h, v: h->1)
#   - Weights:           alpha   = softmax(scores)
#   - Pooled listing:    pooled  = sum_i alpha_i * emb_i
#   - Regression head:   rent_hat = MLP(pooled)
#
# Training target: log1p(price). Loss: MSE on log-rent.
# The attention module and regression head are trained jointly.
#
# Once trained, for every listing we emit one 768-dim attention-pooled
# embedding + the per-photo attention weights (useful for the extension's
# per-photo overlay too).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- Module

class PhotoAttentionPool(nn.Module):
    """Additive attention over a variable-length set of photo embeddings."""

    def __init__(self, embed_dim: int = 768, attn_hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(embed_dim, attn_hidden)
        self.score = nn.Linear(attn_hidden, 1, bias=False)

    def forward(
        self, photos: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        photos: (B, K_max, D)   per-photo embeddings, zero-padded to K_max
        mask:   (B, K_max) bool, True where a real photo is present
        returns:
            pooled: (B, D)         attention-weighted listing embedding
            alpha:  (B, K_max)     attention weights (0 at padded positions)
        """
        h = torch.tanh(self.proj(photos))               # (B, K_max, h)
        scores = self.score(h).squeeze(-1)              # (B, K_max)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)           # (B, K_max)
        alpha = torch.nan_to_num(alpha, nan=0.0)        # all-False row -> 0
        pooled = (alpha.unsqueeze(-1) * photos).sum(dim=1)  # (B, D)
        return pooled, alpha


class AttnRegressor(nn.Module):
    """Attention pool + small MLP head -> log-rent."""

    def __init__(self, embed_dim: int = 768, attn_hidden: int = 64, head_hidden: int = 128):
        super().__init__()
        self.pool = PhotoAttentionPool(embed_dim, attn_hidden)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, photos: torch.Tensor, mask: torch.Tensor):
        pooled, alpha = self.pool(photos, mask)
        y_hat = self.head(pooled).squeeze(-1)   # (B,)
        return y_hat, pooled, alpha


# ---------------------------------------------------------------- Dataset

@dataclass
class ListingPhotoBag:
    listing_id: str
    photos: np.ndarray   # (k, D)
    target: float        # log1p(price)


class BagDataset(Dataset):
    def __init__(self, bags: list[ListingPhotoBag]):
        self.bags = bags

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        b = self.bags[i]
        return b.photos, b.target, b.listing_id


def collate_bags(batch):
    """Pad variable-length photo sequences to the batch max."""
    photos_list, targets, ids = zip(*batch)
    D = photos_list[0].shape[1]
    K_max = max(p.shape[0] for p in photos_list)
    B = len(photos_list)
    padded = np.zeros((B, K_max, D), dtype=np.float32)
    mask = np.zeros((B, K_max), dtype=bool)
    for i, p in enumerate(photos_list):
        k = p.shape[0]
        padded[i, :k] = p
        mask[i, :k] = True
    return (
        torch.from_numpy(padded),
        torch.from_numpy(mask),
        torch.tensor(targets, dtype=torch.float32),
        list(ids),
    )


# ---------------------------------------------------------------- Training

def group_per_photo_by_listing(
    per_photo_embs: np.ndarray,
    per_photo_index: "pd.DataFrame",
    listing_targets: dict[str, float],
) -> list[ListingPhotoBag]:
    """Turn (N_photos, D) + index into one ListingPhotoBag per listing."""
    bags: list[ListingPhotoBag] = []
    # per_photo_index is expected to have columns listing_id, photo_idx, and
    # rows aligned to per_photo_embs. Group by listing_id preserving row order.
    for lid, group in per_photo_index.groupby("listing_id", sort=False):
        if lid not in listing_targets:
            continue
        rows = group.index.values  # positions in per_photo_embs
        photos = per_photo_embs[rows]
        bags.append(
            ListingPhotoBag(
                listing_id=str(lid),
                photos=photos.astype(np.float32),
                target=float(listing_targets[lid]),
            )
        )
    return bags


def train_attention_pool(
    train_bags: list[ListingPhotoBag],
    val_bags: list[ListingPhotoBag],
    embed_dim: int = 768,
    device: torch.device | None = None,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    patience: int = 6,
    seed: int = 42,
    verbose: bool = True,
) -> AttnRegressor:
    """Fit AttnRegressor on train_bags with early stopping on val MSE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = AttnRegressor(embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        BagDataset(train_bags), batch_size=batch_size, shuffle=True,
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        BagDataset(val_bags), batch_size=batch_size, shuffle=False,
        collate_fn=collate_bags,
    )

    best_val = float("inf")
    best_state = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train(True)
        tot_loss, n = 0.0, 0
        for photos, mask, y, _ in train_loader:
            photos, mask, y = photos.to(device), mask.to(device), y.to(device)
            y_hat, _, _ = model(photos, mask)
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.size(0)
            n += y.size(0)
        train_mse = tot_loss / n

        model.train(False)
        v_tot, v_n = 0.0, 0
        with torch.no_grad():
            for photos, mask, y, _ in val_loader:
                photos, mask, y = photos.to(device), mask.to(device), y.to(device)
                y_hat, _, _ = model(photos, mask)
                v_tot += loss_fn(y_hat, y).item() * y.size(0)
                v_n += y.size(0)
        val_mse = v_tot / max(v_n, 1)

        if verbose:
            print(f"  ep{epoch:02d}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")

        if val_mse < best_val - 1e-5:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val_mse={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def pool_all(
    model: AttnRegressor,
    bags: list[ListingPhotoBag],
    device: torch.device | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    """Run the trained model to emit pooled listing embeddings + per-photo alphas."""
    device = device or next(model.parameters()).device
    model.train(False)
    loader = DataLoader(
        BagDataset(bags), batch_size=batch_size, shuffle=False,
        collate_fn=collate_bags,
    )
    all_pooled = []
    all_ids: list[str] = []
    all_alphas: list[np.ndarray] = []
    for photos, mask, _, ids in loader:
        photos, mask = photos.to(device), mask.to(device)
        _, pooled, alpha = model(photos, mask)
        all_pooled.append(pooled.cpu().numpy())
        # strip padded positions from alpha
        alpha_np = alpha.cpu().numpy()
        mask_np = mask.cpu().numpy()
        for i in range(alpha_np.shape[0]):
            k = int(mask_np[i].sum())
            all_alphas.append(alpha_np[i, :k].astype(np.float32))
        all_ids.extend(ids)
    pooled = np.concatenate(all_pooled, axis=0).astype(np.float32)
    return pooled, all_ids, all_alphas
