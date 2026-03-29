"""
models/blood_biomarker_model.py

MLP-based biological age estimator from blood biomarkers (NHANES-style).
Uses PyTorch with batch normalization and dropout for regularization.

Reference features derived from:
  - Levine ME et al. An epigenetic biomarker of aging... Aging 2018
  - Liu Z et al. Underlying features of epigenetic aging clocks... 2020
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


BLOOD_FEATURES = [
    "crp_mg_l",
    "glucose_mg_dl",
    "hdl_mg_dl",
    "ldl_mg_dl",
    "hba1c_pct",
    "albumin_g_dl",
    "creatinine_mg_dl",
    "lymphocyte_pct",
    "rdw_pct",
    "telomere_score",
]

# Clinical reference ranges for normalcy checks
CLINICAL_RANGES = {
    "crp_mg_l":         (0.0, 10.0,  "mg/L",   "Inflammation marker"),
    "glucose_mg_dl":    (70,  126,    "mg/dL",  "Fasting glucose"),
    "hdl_mg_dl":        (40,  100,    "mg/dL",  "HDL cholesterol"),
    "ldl_mg_dl":        (0,   160,    "mg/dL",  "LDL cholesterol"),
    "hba1c_pct":        (4.0, 5.7,   "%",      "Glycated haemoglobin"),
    "albumin_g_dl":     (3.5, 5.0,   "g/dL",  "Serum albumin"),
    "creatinine_mg_dl": (0.6, 1.2,   "mg/dL",  "Creatinine (kidney)"),
    "lymphocyte_pct":   (20,  40,    "%",      "Lymphocyte percentage"),
    "rdw_pct":          (11.5,14.5,  "%",      "Red cell distribution width"),
    "telomere_score":   (0.5, 1.5,   "a.u.",   "Telomere length score"),
}


class BloodAgeMLP(nn.Module):
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BloodBiomarkerModel:
    """
    Trains a PyTorch MLP on blood biomarkers to predict biological age.
    """

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        self.model: BloodAgeMLP | None = None
        self.scaler: StandardScaler | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = [f for f in BLOOD_FEATURES if f in df.columns]
        X = df[features].fillna(df[features].median()).values.astype(np.float32)
        y = df["chronological_age"].values.astype(np.float32)
        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 150, lr: float = 1e-3, batch_size: int = 64):
        X, y = self._prepare(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        train_ds = TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train)
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.model = BloodAgeMLP(input_dim=X_train.shape[1]).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        print(f"Training blood biomarker model ({self.device})...")
        best_val_mae = float("inf")
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 30 == 0 or epoch == epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(
                        torch.tensor(X_val).to(self.device)
                    ).cpu().numpy()
                val_mae = mean_absolute_error(y_val, val_pred)
                print(f"  Epoch {epoch+1:3d}/{epochs}  val_MAE={val_mae:.2f}y")
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        print(f"Best val MAE: {best_val_mae:.2f} years")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load().")

        features = [f for f in BLOOD_FEATURES if f in df.columns]
        X = df[features].fillna(0.0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            bio_age = self.model(
                torch.tensor(X_scaled).to(self.device)
            ).cpu().numpy()

        result = pd.DataFrame()
        if "sample_id" in df.columns:
            result["sample_id"] = df["sample_id"].values
        result["chronological_age"] = df["chronological_age"].values
        result["biological_age_blood"] = bio_age.round(2)
        result["blood_age_gap"] = (bio_age - df["chronological_age"].values).round(2)
        return result

    def flag_abnormal(self, row: dict) -> list[dict]:
        """Return list of out-of-range biomarkers for a single sample."""
        flags = []
        for feat, (lo, hi, unit, desc) in CLINICAL_RANGES.items():
            val = row.get(feat)
            if val is None:
                continue
            if val < lo:
                flags.append({"biomarker": feat, "value": val, "unit": unit,
                               "status": "LOW", "description": desc,
                               "ref_range": f"{lo}–{hi}"})
            elif val > hi:
                flags.append({"biomarker": feat, "value": val, "unit": unit,
                               "status": "HIGH", "description": desc,
                               "ref_range": f"{lo}–{hi}"})
        return flags

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_dir}/blood_mlp.pt")
        joblib.dump(self.scaler, f"{self.model_dir}/blood_scaler.pkl")
        print(f"Blood model saved to {self.model_dir}/")

    def load(self):
        self.scaler = joblib.load(f"{self.model_dir}/blood_scaler.pkl")
        self.model = BloodAgeMLP().to(self.device)
        self.model.load_state_dict(
            torch.load(f"{self.model_dir}/blood_mlp.pt", map_location=self.device)
        )
        return self
