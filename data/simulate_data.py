"""
data/simulate_data.py

Generates realistic synthetic data mimicking:
  - GEO GSE40279: Hannum et al. 2013 blood DNA methylation dataset (656 subjects)
  - NHANES blood biomarker panel

Run: python data/simulate_data.py
Outputs: data/methylation.csv, data/blood_biomarkers.csv, data/combined.csv

When you have access to real GEO data, replace these files with the real ones.
The rest of the pipeline is identical.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Hannum 2013: 71 CpG sites with known age-correlation directions ──────────
# Source: Hannum G et al. Genome-wide methylation profiles reveal quantitative
#         views of human aging rates. Mol Cell. 2013;49(2):359-367.
HANNUM_CPGS = [
    ("cg16867657", +0.87), ("cg22454769", +0.83), ("cg06493994", +0.81),
    ("cg02228185", +0.79), ("cg25809905", +0.78), ("cg17861230", +0.76),
    ("cg24724428", +0.75), ("cg08097417", +0.74), ("cg18933331", +0.73),
    ("cg11299964", +0.72), ("cg04474832", +0.71), ("cg12841266", +0.70),
    ("cg23606718", +0.69), ("cg00481951", +0.68), ("cg19283806", +0.67),
    ("cg20822990", +0.66), ("cg05442902", +0.65), ("cg03607117", +0.64),
    ("cg25410668", +0.63), ("cg21296230", +0.62), ("cg14361627", +0.61),
    ("cg01718688", +0.60), ("cg26382148", +0.59), ("cg11254979", +0.58),
    ("cg26470501", +0.57), ("cg19999072", +0.56), ("cg27069726", +0.55),
    ("cg08234504", +0.54), ("cg09809672", +0.53), ("cg07553761", +0.52),
    ("cg17470237", +0.51), ("cg18815943", +0.50), ("cg20776363", +0.49),
    ("cg22736354", +0.48), ("cg07396958", +0.47), ("cg16054275", -0.87),
    ("cg01820374", -0.83), ("cg24139302", -0.81), ("cg08698782", -0.79),
    ("cg18768621", -0.77), ("cg21041194", -0.75), ("cg25256723", -0.73),
    ("cg25428494", -0.71), ("cg07076056", -0.69), ("cg04528819", -0.67),
    ("cg02367849", -0.65), ("cg15239557", -0.63), ("cg03068993", -0.61),
    ("cg08246323", -0.59), ("cg03890877", -0.57), ("cg14011319", -0.55),
    ("cg26403843", -0.53), ("cg09209420", -0.51), ("cg22580512", -0.49),
    ("cg23995914", -0.47), ("cg23124451", -0.45), ("cg11553655", -0.43),
    ("cg08165561", -0.41), ("cg12885166", -0.39), ("cg25430028", -0.37),
    ("cg01446836", -0.35), ("cg24335620", -0.33), ("cg06393904", -0.31),
    ("cg11780044", -0.29), ("cg18781680", -0.27), ("cg00121626", -0.25),
    ("cg14065342", -0.23), ("cg26594919", -0.21), ("cg07147118", -0.19),
    ("cg07454552", -0.17), ("cg11695696", -0.15),
]

CPGS = [c for c, _ in HANNUM_CPGS]
CORRELATIONS = np.array([r for _, r in HANNUM_CPGS])


def simulate_methylation(n: int = 656) -> pd.DataFrame:
    """Generate beta values (0–1) for each CpG, correlated with age."""
    ages = np.random.uniform(19, 101, size=n)

    data = {}
    data["sample_id"] = [f"GSM{100000 + i}" for i in range(n)]
    data["chronological_age"] = ages.round(1)

    age_norm = (ages - ages.mean()) / ages.std()

    for cpg, corr in HANNUM_CPGS:
        noise = np.random.normal(0, 1, n)
        # Linear combination: signal + noise, projected to [0.05, 0.95]
        raw = corr * age_norm + np.sqrt(1 - corr**2) * noise
        beta = (raw - raw.min()) / (raw.max() - raw.min()) * 0.90 + 0.05
        data[cpg] = beta.round(4)

    return pd.DataFrame(data)


def simulate_blood_biomarkers(sample_ids: list, ages: np.ndarray) -> pd.DataFrame:
    """
    Simulate NHANES-style blood biomarkers.
    Each biomarker has a realistic age-correlation direction.

    Biomarkers:
      CRP     - C-reactive protein (inflammation) ↑ with age
      glucose - Fasting glucose ↑ with age
      HDL     - HDL cholesterol ↓ with age
      LDL     - LDL cholesterol varies
      HbA1c   - Glycated haemoglobin ↑ with age
      albumin - Serum albumin ↓ with age
      creatinine - Kidney function marker ↑ slightly
      lympho_pct - Lymphocyte % ↓ with age
      rdw     - Red cell distribution width ↑ with age
      telomere_score - Proxy telomere length ↓ with age
    """
    n = len(ages)
    age_norm = (ages - ages.mean()) / ages.std()

    def biomarker(mean, std, age_corr, noise_std=1.0):
        raw = age_corr * age_norm + np.sqrt(1 - age_corr**2) * np.random.normal(0, noise_std, n)
        return (raw * std + mean).clip(mean - 3 * std, mean + 3 * std)

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "chronological_age": ages.round(1),
        "crp_mg_l":         biomarker(2.5,  2.0,  +0.45).round(2),
        "glucose_mg_dl":    biomarker(95.0, 12.0, +0.40).round(1),
        "hdl_mg_dl":        biomarker(55.0, 14.0, -0.30).round(1),
        "ldl_mg_dl":        biomarker(120.0,30.0, +0.15).round(1),
        "hba1c_pct":        biomarker(5.4,  0.6,  +0.42).round(2),
        "albumin_g_dl":     biomarker(4.2,  0.4,  -0.38).round(2),
        "creatinine_mg_dl": biomarker(0.95, 0.2,  +0.25).round(3),
        "lymphocyte_pct":   biomarker(30.0, 8.0,  -0.35).round(1),
        "rdw_pct":          biomarker(13.2, 1.2,  +0.33).round(2),
        "telomere_score":   biomarker(1.0,  0.25, -0.55).round(4),
    })
    return df


def main():
    os.makedirs("data", exist_ok=True)
    print("Generating methylation data (656 samples × 73 columns)...")
    meth = simulate_methylation(656)
    meth.to_csv("data/methylation.csv", index=False)
    print(f"  ✓ data/methylation.csv  shape={meth.shape}")

    print("Generating blood biomarker data...")
    blood = simulate_blood_biomarkers(
        meth["sample_id"].tolist(),
        meth["chronological_age"].values,
    )
    blood.to_csv("data/blood_biomarkers.csv", index=False)
    print(f"  ✓ data/blood_biomarkers.csv  shape={blood.shape}")

    # Combined
    combined = meth.merge(
        blood.drop(columns=["chronological_age"]), on="sample_id"
    )
    combined.to_csv("data/combined.csv", index=False)
    print(f"  ✓ data/combined.csv  shape={combined.shape}")
    print("\nDone. Replace with real GEO/NHANES data when available.")


if __name__ == "__main__":
    main()
