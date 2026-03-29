"""
scripts/train_models.py

End-to-end training pipeline. Run this once before starting the API or dashboard.

Steps:
  1. Generate synthetic data (or use real GEO/NHANES data)
  2. Train Hannum epigenetic clock (elastic-net)
  3. Train blood biomarker MLP (PyTorch)
  4. Train fusion gate
  5. Build FAISS RAG index
  6. Save all models + print evaluation summary

Usage:
  cd bioage-agent
  python scripts/train_models.py

  # To use real GEO data (after downloading):
  python scripts/train_models.py --real-data
"""

import argparse
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def step1_generate_data(use_real: bool = False):
    print_section("STEP 1: Data Preparation")

    if use_real:
        print("Looking for real data files...")
        if not os.path.exists("data/methylation.csv"):
            print("  ✗ data/methylation.csv not found.")
            print("  → Download GEO GSE40279 from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279")
            print("  → Place the processed file at data/methylation.csv")
            print("  → Falling back to synthetic data.")
            use_real = False
        else:
            print("  ✓ Found real methylation data")

    if not use_real:
        print("Generating synthetic data (mimics GEO GSE40279 + NHANES)...")
        from data.simulate_data import main as simulate
        simulate()

    meth  = pd.read_csv("data/methylation.csv")
    blood = pd.read_csv("data/blood_biomarkers.csv")
    combined = pd.read_csv("data/combined.csv")

    print(f"\nDataset summary:")
    print(f"  Methylation: {meth.shape[0]} samples, {meth.shape[1]-2} CpG features")
    print(f"  Blood biomarkers: {blood.shape[0]} samples, {blood.shape[1]-2} features")
    print(f"  Age range: {meth['chronological_age'].min():.0f}–{meth['chronological_age'].max():.0f} years")
    print(f"  Mean age: {meth['chronological_age'].mean():.1f} ± {meth['chronological_age'].std():.1f} years")

    return meth, blood, combined


def step2_train_hannum(meth: pd.DataFrame):
    print_section("STEP 2: Hannum Epigenetic Clock (Elastic-Net)")
    from models.methylation_model import HannumClock

    clock = HannumClock()
    clock.train(meth)
    clock.save()

    # Cross-val evaluation
    preds = clock.predict(meth)
    mae = mean_absolute_error(meth["chronological_age"], preds["biological_age_methylation"])
    r2  = r2_score(meth["chronological_age"], preds["biological_age_methylation"])

    print(f"\n  Hannum Clock Evaluation:")
    print(f"    MAE:  {mae:.2f} years  (Hannum 2013 reported: ~3.9 years on held-out)")
    print(f"    R²:   {r2:.3f}")
    print(f"\n  Top 5 CpGs:")
    top = clock.top_cpgs(5)
    for _, row in top.iterrows():
        print(f"    {row['cpg']:20s}  coef={row['coefficient']:+.4f}  {row['direction']}")

    return clock, preds


def step3_train_blood(blood: pd.DataFrame):
    print_section("STEP 3: Blood Biomarker MLP (PyTorch)")
    from models.blood_biomarker_model import BloodBiomarkerModel

    model = BloodBiomarkerModel()
    model.train(blood, epochs=150)
    model.save()

    preds = model.predict(blood)
    mae = mean_absolute_error(blood["chronological_age"], preds["biological_age_blood"])
    r2  = r2_score(blood["chronological_age"], preds["biological_age_blood"])

    print(f"\n  Blood MLP Evaluation:")
    print(f"    MAE: {mae:.2f} years")
    print(f"    R²:  {r2:.3f}")

    # Check example flags
    sample = blood.iloc[0].to_dict()
    model_obj = BloodBiomarkerModel()
    flags = model_obj.flag_abnormal(sample)
    print(f"\n  Sample abnormal biomarkers: {len(flags)} flagged")

    return model, preds


def step4_train_fusion(blood_preds: pd.DataFrame, meth_preds: pd.DataFrame, combined: pd.DataFrame):
    print_section("STEP 4: Multi-Modal Fusion Gate (PyTorch)")
    from models.fusion_model import FusionModel

    # Merge predictions with ground truth
    df = combined[["sample_id", "chronological_age"]].copy()
    df = df.merge(meth_preds[["sample_id", "biological_age_methylation"]], on="sample_id")
    df = df.merge(blood_preds[["sample_id", "biological_age_blood"]], on="sample_id")

    model = FusionModel()
    model.train(df, epochs=200)
    model.save()

    result = model.predict(df)
    mae = mean_absolute_error(df["chronological_age"], result["composite_biological_age"])
    r2  = r2_score(df["chronological_age"], result["composite_biological_age"])

    print(f"\n  Fusion Model Evaluation:")
    print(f"    MAE: {mae:.2f} years")
    print(f"    R²:  {r2:.3f}")

    # Category distribution
    cats = result["aging_category"].value_counts()
    print(f"\n  Aging category distribution:")
    for cat, count in cats.items():
        pct = 100 * count / len(result)
        print(f"    {cat:35s} {count:4d} ({pct:.1f}%)")

    # AAI stats
    aai = result["aging_acceleration_index"]
    print(f"\n  Aging Acceleration Index (AAI):")
    print(f"    Mean:    {aai.mean():+.2f} years")
    print(f"    Std:     {aai.std():.2f} years")
    print(f"    Range:   {aai.min():+.1f} to {aai.max():+.1f} years")

    return model, result


def step5_build_rag():
    print_section("STEP 5: RAG Index (FAISS + Sentence Transformers)")
    from rag.retriever import LongevityRetriever
    from rag.corpus import get_all_papers

    papers = get_all_papers()
    print(f"  Corpus size: {len(papers)} papers")

    retriever = LongevityRetriever()
    retriever.build(force=True)

    # Test a few queries
    test_queries = [
        "rapamycin mTOR lifespan extension",
        "DNA methylation epigenetic clock aging",
        "senolytic dasatinib quercetin senescence",
    ]
    print("\n  Test searches:")
    for q in test_queries:
        results = retriever.search(q, k=1)
        if results:
            print(f"    '{q}'")
            print(f"     → {results[0]['title'][:60]}... (score={results[0]['relevance_score']:.3f})")

    return retriever


def step6_summary(start_time: float):
    print_section("TRAINING COMPLETE")
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f} seconds")
    print(f"\n  Saved artifacts:")
    for f in os.listdir("models/saved"):
        path = f"models/saved/{f}"
        size_kb = os.path.getsize(path) / 1024
        print(f"    {f:40s} {size_kb:.1f} KB")

    print(f"""
  Next steps:
    1. Start the API:         uvicorn api.main:app --reload --port 8000
    2. Open the dashboard:    streamlit run dashboard/app.py
    3. Run the notebook:      jupyter notebook notebooks/analysis.ipynb

  API docs:  http://localhost:8000/docs
  Dashboard: http://localhost:8501
""")


def main():
    parser = argparse.ArgumentParser(description="Train all Biological Age Agent models")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real GEO/NHANES data instead of synthetic")
    args = parser.parse_args()

    start = time.time()

    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    meth, blood, combined = step1_generate_data(use_real=args.real_data)
    clock, meth_preds      = step2_train_hannum(meth)
    blood_model, blood_preds = step3_train_blood(blood)
    fusion_model, fusion_result = step4_train_fusion(blood_preds, meth_preds, combined)
    retriever = step5_build_rag()
    step6_summary(start)


if __name__ == "__main__":
    main()
