# NN architectures for price prediction

import torch
import torch.nn as nn


class TabularModel(nn.Module):
    """Tabular-only baseline."""

    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_tab):
        return self.net(x_tab).squeeze(-1)


class TabularImageModel(nn.Module):
    """Tabular + image embeddings, fused with concat."""

    def __init__(self, tab_dim, img_dim=2048, tab_hidden=64, img_hidden=64,
                 fusion_hidden=64, dropout=0.5):
        super().__init__()

        self.tab_branch = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, img_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(tab_hidden + img_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_tab, x_img):
        tab_out = self.tab_branch(x_tab)
        img_out = self.img_branch(x_img)
        combined = torch.cat([tab_out, img_out], dim=1)
        return self.fusion(combined).squeeze(-1)


class ImageOnlyModel(nn.Module):
    """Image-only ablation model."""

    def __init__(self, img_dim=2048, hidden_dims=(128, 64), dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = img_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_img):
        return self.net(x_img).squeeze(-1)
