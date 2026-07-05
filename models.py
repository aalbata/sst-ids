"""
models.py
=========
The SST-IDS feature-token Transformer (masked-feature self-supervised
pretraining + supervised fine-tuning) and the five baselines used for
comparison: Random Forest, XGBoost, a 1-D CNN, an LSTM, and a TabTransformer.

Design overview:
  * each selected feature is ONE token (Eq. 2: e_i = W_e x_i + b_e),
  * a learned feature-INDEX embedding marks the fixed position of each feature
    (explicitly NOT temporal positional encoding -- the inputs are tabular),
  * pretraining reconstructs masked feature values with MSE (Eq. 6),
  * fine-tuning pools the encoder output and classifies with a sigmoid head
    trained by binary cross-entropy (Eq. 7-8).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SST-IDS backbone
# ---------------------------------------------------------------------------
class FeatureTokenTransformer(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = cfg.embed_dim

        # per-feature linear projection of a scalar -> embedding (Eq. 2)
        self.feat_w = nn.Parameter(torch.randn(n_features, cfg.embed_dim) * 0.02)
        self.feat_b = nn.Parameter(torch.zeros(n_features, cfg.embed_dim))
        # fixed feature-index embedding (NOT temporal positional encoding)
        self.index_embed = nn.Embedding(n_features, cfg.embed_dim)
        # learned [MASK] token used during self-supervised pretraining
        self.mask_token = nn.Parameter(torch.randn(cfg.embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_hidden_dim, dropout=cfg.dropout,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_encoder_blocks)

        self.recon_head = nn.Linear(cfg.embed_dim, 1)          # masked reconstruction
        self.cls_head = nn.Sequential(                          # binary classifier
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.embed_dim // 2, 1))

    def _tokens(self, x):
        # x: (B, n_features) -> (B, n_features, embed_dim)
        tok = x.unsqueeze(-1) * self.feat_w.unsqueeze(0) + self.feat_b.unsqueeze(0)
        idx = torch.arange(self.n_features, device=x.device)
        return tok + self.index_embed(idx).unsqueeze(0)

    def forward_pretrain(self, x, mask):
        """mask: (B, n_features) bool, True = masked. Returns recon (B, n_features)."""
        tok = self._tokens(x)
        m = mask.unsqueeze(-1)
        tok = torch.where(m, self.mask_token.view(1, 1, -1).expand_as(tok), tok)
        h = self.encoder(tok)
        return self.recon_head(h).squeeze(-1)

    def forward_logits(self, x):
        h = self.encoder(self._tokens(x))
        return self.cls_head(h.mean(dim=1)).squeeze(-1)         # mean-pool tokens

    def forward(self, x):
        return self.forward_logits(x)


# ---------------------------------------------------------------------------
# Baseline neural nets (torch)
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    """Treat the feature vector as a length-d 1-channel signal."""
    def __init__(self, n_features, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, hidden), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1))).squeeze(-1)


class LSTMClassifier(nn.Module):
    """Treat the feature vector as a sequence of length d (input size 1)."""
    def __init__(self, n_features, hidden=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :]).squeeze(-1)


class TabTransformer(nn.Module):
    """Feature-token Transformer trained supervised from scratch (NO masked-
    feature pretraining). Acts as the 'transformer-without-our-SSL' baseline."""
    def __init__(self, n_features, cfg):
        super().__init__()
        self.backbone = FeatureTokenTransformer(n_features, cfg)

    def forward(self, x):
        return self.backbone.forward_logits(x)


def build_torch_model(name, n_features, cfg):
    name = name.lower()
    if name in ("sst-ids", "sstids"):
        return FeatureTokenTransformer(n_features, cfg)
    if name == "cnn":
        return CNN1D(n_features, dropout=cfg.dropout)
    if name == "lstm":
        return LSTMClassifier(n_features, dropout=cfg.dropout)
    if name == "tabtransformer":
        return TabTransformer(n_features, cfg)
    raise ValueError(f"unknown torch model: {name}")


# ---------------------------------------------------------------------------
# Sklearn / XGBoost baselines (return fitted estimator + predict_proba fn)
# ---------------------------------------------------------------------------
def build_sklearn_baseline(name, random_state=0):
    name = name.lower()
    if name in ("random forest", "rf"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=random_state)
    if name in ("xgboost", "xgb"):
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("xgboost not installed: pip install xgboost") from e
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", random_state=random_state, n_jobs=-1)
    raise ValueError(f"unknown sklearn baseline: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
