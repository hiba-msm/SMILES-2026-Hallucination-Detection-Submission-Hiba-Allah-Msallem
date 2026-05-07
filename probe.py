"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- THERMAL MANAGEMENT ---
if torch.get_num_threads() > 4:
    torch.set_num_threads(4)


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Uses PCA for dimensionality reduction followed by an ensemble of MLPs
    trained with early stopping. Threshold is tuned for accuracy (the primary
    competition metric).
    """

    def __init__(self) -> None:
        super().__init__()
        self._ensemble = nn.ModuleList()
        self._scaler = StandardScaler()
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=128, random_state=42)
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_new_member(self, input_dim: int) -> nn.Sequential:
        """Instantiate a new MLP member for the ensemble."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns averaged logits of the ensemble."""
        if len(self._ensemble) == 0:
            raise RuntimeError("Ensemble is empty. Call fit() first.")
        all_logits = torch.stack([member(x).squeeze(-1) for member in self._ensemble])
        return torch.mean(all_logits, dim=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train an ensemble of probes on labelled feature vectors."""
        import random
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        X_scaled = self._scaler.fit_transform(X)
        X_pca = self._pca.fit_transform(X_scaled)

        # 5-fold internal ensembling
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self._ensemble = nn.ModuleList()
        input_dim = X_pca.shape[1]

        for train_idx, val_idx in skf.split(X_pca, y):
            X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            X_t_tr = torch.from_numpy(X_tr).float()
            y_t_tr = torch.from_numpy(y_tr.astype(np.float32))
            X_t_val = torch.from_numpy(X_val).float()
            y_t_val = torch.from_numpy(y_val.astype(np.float32))

            member = self._build_new_member(input_dim)
            optimizer = torch.optim.Adam(member.parameters(), lr=1e-3, weight_decay=1e-4)

            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val_loss = float("inf")
            best_state = None
            patience = 15
            patience_counter = 0

            for epoch in range(400):
                member.train()
                optimizer.zero_grad()
                logits = member(X_t_tr).squeeze(-1)
                loss = criterion(logits, y_t_tr)
                loss.backward()
                optimizer.step()

                member.eval()
                with torch.no_grad():
                    val_logits = member(X_t_val).squeeze(-1)
                    val_loss = criterion(val_logits, y_t_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in member.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if best_state:
                member.load_state_dict(best_state)
            self._ensemble.append(member)

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy.

        Since the primary competition metric is accuracy, we optimise
        the threshold for accuracy rather than F1.

        Args:
            X_val: Validation feature matrix.
            y_val: Integer label vector.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))

        best_threshold = 0.5
        best_acc = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = accuracy_score(y_val, y_pred_t)
            if score > best_acc:
                best_acc = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        X_t = torch.from_numpy(X_pca).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
