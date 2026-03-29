"""
models/methylation_model.py

Implements the Hannum Epigenetic Clock.
Reference: Hannum G et al. Mol Cell 2013;49(2):359-367.

The clock is an elastic-net regression on 71 CpG beta values → predicted age.
Biological age gap = predicted_age - chronological_age
  > 0 → aging faster than expected
  < 0 → aging slower than expected
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


# The 71 CpG sites from the Hannum 2013 paper
HANNUM_CPGS = [
    "cg16867657","cg22454769","cg06493994","cg02228185","cg25809905",
    "cg17861230","cg24724428","cg08097417","cg18933331","cg11299964",
    "cg04474832","cg12841266","cg23606718","cg00481951","cg19283806",
    "cg20822990","cg05442902","cg03607117","cg25410668","cg21296230",
    "cg14361627","cg01718688","cg26382148","cg11254979","cg26470501",
    "cg19999072","cg27069726","cg08234504","cg09809672","cg07553761",
    "cg17470237","cg18815943","cg20776363","cg22736354","cg07396958",
    "cg16054275","cg01820374","cg24139302","cg08698782","cg18768621",
    "cg21041194","cg25256723","cg25428494","cg07076056","cg04528819",
    "cg02367849","cg15239557","cg03068993","cg08246323","cg03890877",
    "cg14011319","cg26403843","cg09209420","cg22580512","cg23995914",
    "cg23124451","cg11553655","cg08165561","cg12885166","cg25430028",
    "cg01446836","cg24335620","cg06393904","cg11780044","cg18781680",
    "cg00121626","cg14065342","cg26594919","cg07147118","cg07454552",
    "cg11695696",
]


class HannumClock:
    """
    Elastic-net implementation of the Hannum epigenetic clock.
    Trains on CpG methylation beta values to predict chronological age,
    then uses the residual as a proxy for biological aging rate.
    """

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        self.model: ElasticNet | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.train_mae: float | None = None
        self.train_r2: float | None = None

    def _available_cpgs(self, df: pd.DataFrame) -> list[str]:
        return [c for c in HANNUM_CPGS if c in df.columns]

    def train(self, df: pd.DataFrame, age_col: str = "chronological_age"):
        cpgs = self._available_cpgs(df)
        if len(cpgs) < 10:
            raise ValueError(f"Need at least 10 Hannum CpGs. Found {len(cpgs)}.")

        self.feature_names = cpgs
        X = df[cpgs].values.astype(np.float32)
        y = df[age_col].values.astype(np.float32)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        print(f"Training Hannum clock on {len(cpgs)} CpGs, {len(y)} samples...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        self.model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-3, 1, 30),
            cv=cv,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        preds = self.model.predict(X_scaled)
        self.train_mae = mean_absolute_error(y, preds)
        self.train_r2 = r2_score(y, preds)

        print(f"  α={self.model.alpha_:.4f}  l1_ratio={self.model.l1_ratio_:.2f}")
        print(f"  Train MAE: {self.train_mae:.2f} years  R²: {self.train_r2:.3f}")
        nonzero = np.sum(self.model.coef_ != 0)
        print(f"  Non-zero CpGs: {nonzero}/{len(cpgs)}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load().")
        
        # Fill missing CpGs with column mean (graceful degradation)
        X = np.zeros((len(df), len(self.feature_names)), dtype=np.float32)
        for i, cpg in enumerate(self.feature_names):
            if cpg in df.columns:
                X[:, i] = df[cpg].values
            else:
                X[:, i] = 0.5  # Default beta midpoint

        X_scaled = self.scaler.transform(X)
        bio_age = self.model.predict(X_scaled)

        result = df[["sample_id", "chronological_age"]].copy() if "sample_id" in df.columns \
            else pd.DataFrame({"chronological_age": df["chronological_age"].values})
        result["biological_age_methylation"] = bio_age.round(2)
        result["methylation_age_gap"] = (bio_age - df["chronological_age"].values).round(2)
        result["methylation_aging_rate"] = self._aging_rate_label(
            bio_age - df["chronological_age"].values
        )
        return result

    @staticmethod
    def _aging_rate_label(gap: np.ndarray) -> list[str]:
        labels = []
        for g in gap:
            if g <= -5:
                labels.append("significantly_slower")
            elif g <= -2:
                labels.append("slower")
            elif g < 2:
                labels.append("normal")
            elif g < 5:
                labels.append("faster")
            else:
                labels.append("significantly_faster")
        return labels

    def top_cpgs(self, n: int = 10) -> pd.DataFrame:
        """Return top n most influential CpGs with their coefficient direction."""
        if self.model is None:
            return pd.DataFrame()
        coefs = pd.DataFrame({
            "cpg": self.feature_names,
            "coefficient": self.model.coef_,
        })
        coefs["abs_coef"] = coefs["coefficient"].abs()
        coefs["direction"] = coefs["coefficient"].apply(
            lambda x: "↑ increases with age" if x > 0 else "↓ decreases with age"
        )
        return coefs.sort_values("abs_coef", ascending=False).head(n).reset_index(drop=True)

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, f"{self.model_dir}/hannum_model.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/hannum_scaler.pkl")
        joblib.dump(self.feature_names, f"{self.model_dir}/hannum_features.pkl")
        print(f"Model saved to {self.model_dir}/")

    def load(self):
        self.model = joblib.load(f"{self.model_dir}/hannum_model.pkl")
        self.scaler = joblib.load(f"{self.model_dir}/hannum_scaler.pkl")
        self.feature_names = joblib.load(f"{self.model_dir}/hannum_features.pkl")
        return self
