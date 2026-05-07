"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that detects hallucinations.
Uses a Hybrid Ensemble of MLPs and Random Forest classifiers to maximise accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class HallucinationProbe(nn.Module):
    """Hybrid Ensemble Probe (MLP + Random Forest).

    Combines the non-linear feature extraction of neural networks with the
    robust decision boundaries of Random Forests to maximise accuracy on
    small, high-dimensional datasets.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mlps = nn.ModuleList()
        self._forests = []
        self._scaler = StandardScaler()
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=192, random_state=42)
        self._threshold: float = 0.5

    def _build_mlp(self, input_dim: int) -> nn.Sequential:
        """Build a single MLP member."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train a Hybrid Ensemble of 5 models (3 MLPs, 2 Random Forests)."""
        import random
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        X_scaled = self._scaler.fit_transform(X)
        X_pca = self._pca.fit_transform(X_scaled)

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        input_dim = X_pca.shape[1]
        self._mlps = nn.ModuleList()
        self._forests = []

        for i, (tr_idx, val_idx) in enumerate(skf.split(X_pca, y)):
            X_tr, X_val = X_pca[tr_idx], X_pca[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # 1. Train MLP Member
            mlp = self._build_mlp(input_dim)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            best_val_loss = float("inf")
            best_state = None
            for epoch in range(300):
                mlp.train()
                optimizer.zero_grad()
                logits = mlp(torch.from_numpy(X_tr).float()).squeeze(-1)
                loss = criterion(logits, torch.from_numpy(y_tr.astype(np.float32)))
                loss.backward()
                optimizer.step()

                mlp.eval()
                with torch.no_grad():
                    val_logits = mlp(torch.from_numpy(X_val).float()).squeeze(-1)
                    val_loss = criterion(val_logits, torch.from_numpy(y_val.astype(np.float32))).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

            if best_state:
                mlp.load_state_dict(best_state)
            self._mlps.append(mlp)

            # 2. Train Random Forest Member (on alternate folds)
            if i % 2 == 0:
                rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                rf.fit(X_tr, y_tr)
                self._forests.append(rf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble average of MLP and Random Forest predictions."""
        X_pca = self._pca.transform(self._scaler.transform(X))
        
        mlp_probs = []
        for mlp in self._mlps:
            mlp.eval()
            with torch.no_grad():
                probs = torch.sigmoid(mlp(torch.from_numpy(X_pca).float())).numpy().flatten()
                mlp_probs.append(probs)
        
        rf_probs = [rf.predict_proba(X_pca)[:, 1] for rf in self._forests]
        
        # Mean across all models (3 MLPs + 2 RFs)
        avg_prob = np.mean(mlp_probs + rf_probs, axis=0)
        return np.stack([1.0 - avg_prob, avg_prob], axis=1)

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        """Tune threshold for maximum accuracy."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.linspace(0, 1, 101)
        best_acc = -1.0
        for t in candidates:
            y_pred = (probs >= t).astype(int)
            acc = accuracy_score(y_val, y_pred)
            if acc > best_acc:
                best_acc = acc
                self._threshold = t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
