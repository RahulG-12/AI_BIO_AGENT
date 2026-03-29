"""
models/fusion_model.py

Late-fusion model: combines biological age estimates from:
  1. Hannum methylation clock
  2. Blood biomarker MLP

Produces a single Composite Biological Age (CBA) score and
an Aging Acceleration Index (AAI = CBA - chronological_age).

Architecture: learned weighted average (attention-style gating)
The gate learns which modality is more reliable per-sample.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


class FusionGate(nn.Module):
    """
    Learns per-sample weights for each modality.
    Input:  [bio_age_methylation, bio_age_blood]  (2 features)
    Output: scalar composite biological age
    """
    def __init__(self):
        super().__init__()
        # Gate: learns which modality to trust
        self.gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=1),
        )
        # Refine: small residual correction on top of weighted avg
        self.refine = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        weights = self.gate(x)            # (B, 2)
        weighted_avg = (weights * x).sum(dim=1, keepdim=True)  # (B, 1)
        residual = self.refine(x)         # (B, 1)
        return (weighted_avg + 0.1 * residual).squeeze(-1)


class FusionModel:
    """
    Trains the fusion gate on top of individual model predictions.
    Requires both methylation and blood predictions to be available.
    """

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        self.model: FusionGate | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = ["biological_age_methylation", "biological_age_blood"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Run both sub-models first.")
        return df[cols].values.astype(np.float32)

    def train(self, df: pd.DataFrame, epochs: int = 200, lr: float = 5e-4):
        X = self._extract_features(df)
        y = df["chronological_age"].values.astype(np.float32)

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model = FusionGate().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print("Training fusion gate...")
        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(xb), yb).backward()
                optimizer.step()

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X).to(self.device)).cpu().numpy()
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        print(f"  Fusion MAE: {mae:.2f} years  R²: {r2:.3f}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Fusion model not trained.")

        X = self._extract_features(df)
        self.model.eval()
        with torch.no_grad():
            composite = self.model(
                torch.tensor(X).to(self.device)
            ).cpu().numpy()

        result = df.copy()
        result["composite_biological_age"] = composite.round(2)
        result["aging_acceleration_index"] = (
            composite - df["chronological_age"].values
        ).round(2)
        result["aging_category"] = self._categorize(
            composite - df["chronological_age"].values
        )
        return result

    @staticmethod
    def _categorize(aai: np.ndarray) -> list[str]:
        cats = []
        for a in aai:
            if a <= -7:   cats.append("exceptional_longevity")
            elif a <= -3: cats.append("slower_aging")
            elif a <  3:  cats.append("typical_aging")
            elif a <  7:  cats.append("accelerated_aging")
            else:         cats.append("significantly_accelerated")
        return cats

    def get_modality_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-sample modality weights — shows which data source dominates."""
        X = self._extract_features(df)
        self.model.eval()
        with torch.no_grad():
            weights = self.model.gate(
                torch.tensor(X).to(self.device)
            ).cpu().numpy()
        return pd.DataFrame({
            "sample_id": df.get("sample_id", range(len(df))),
            "methylation_weight": weights[:, 0].round(3),
            "blood_weight": weights[:, 1].round(3),
        })

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_dir}/fusion_gate.pt")
        print(f"Fusion model saved.")

    def load(self):
        self.model = FusionGate().to(self.device)
        self.model.load_state_dict(
            torch.load(f"{self.model_dir}/fusion_gate.pt", map_location=self.device)
        )
        return self
