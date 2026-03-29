"""
scripts/demo.py

Offline demo — runs the full pipeline without any API key.
Shows: data generation → model training → prediction → interventions → RAG search.

Usage:
  cd bioage-agent
  python scripts/demo.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def separator(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    print("=" * 60)
    print("  BIOLOGICAL AGE AGENT — OFFLINE DEMO")
    print("  Immortigen ML Research Engineer Candidate Project")
    print("=" * 60)

    # ── 1. Generate data ───────────────────────────────────────────
    separator("1. Generating Synthetic Data (GEO + NHANES)")
    from data.simulate_data import simulate_methylation, simulate_blood_biomarkers

    meth  = simulate_methylation(200)
    blood = simulate_blood_biomarkers(meth["sample_id"].tolist(), meth["chronological_age"].values)
    print(f"  ✓ {len(meth)} samples | {meth.shape[1]-2} CpG sites | {blood.shape[1]-2} blood features")

    # ── 2. Train Hannum clock ──────────────────────────────────────
    separator("2. Training Hannum Epigenetic Clock")
    from models.methylation_model import HannumClock

    clock = HannumClock()
    clock.train(meth)
    meth_preds = clock.predict(meth)
    print(f"\n  Top 3 age-correlated CpGs:")
    for _, row in clock.top_cpgs(3).iterrows():
        print(f"    {row['cpg']}  {row['direction']}  (coef={row['coefficient']:+.3f})")

    # ── 3. Train blood MLP ─────────────────────────────────────────
    separator("3. Training Blood Biomarker MLP (PyTorch)")
    from models.blood_biomarker_model import BloodBiomarkerModel

    blood_model = BloodBiomarkerModel()
    blood_model.train(blood, epochs=100)
    blood_preds = blood_model.predict(blood)

    # ── 4. Train fusion ────────────────────────────────────────────
    separator("4. Training Fusion Gate (Late Fusion)")
    from models.fusion_model import FusionModel

    df_fused = meth[["sample_id", "chronological_age"]].merge(
        meth_preds[["sample_id", "biological_age_methylation"]], on="sample_id"
    ).merge(
        blood_preds[["sample_id", "biological_age_blood"]], on="sample_id"
    )

    fusion = FusionModel()
    fusion.train(df_fused, epochs=150)
    result = fusion.predict(df_fused)

    print(f"\n  Population aging profile (n={len(result)}):")
    for cat, cnt in result["aging_category"].value_counts().items():
        bar = "█" * (cnt // 5)
        print(f"    {cat:35s} {bar} {cnt}")

    # ── 5. Single subject demo ────────────────────────────────────
    separator("5. Single Subject Analysis")

    subject = {
        "sample_id": "DEMO_SUBJECT",
        "chronological_age": 45.0,
        "crp_mg_l": 4.2,
        "glucose_mg_dl": 105,
        "hdl_mg_dl": 38,
        "ldl_mg_dl": 118,       # ← was missing, caused the 9 vs 10 feature error
        "hba1c_pct": 5.8,
        "albumin_g_dl": 4.1,
        "creatinine_mg_dl": 0.98,
        "lymphocyte_pct": 22,
        "rdw_pct": 14.8,
        "telomere_score": 0.71,
    }

    print(f"\n  Subject: {subject['sample_id']} | Age: {subject['chronological_age']}")

    # Methylation prediction (use median CpG values)
    s_meth = pd.DataFrame([{"sample_id": "DEMO_SUBJECT", "chronological_age": 45.0}])
    for cpg in clock.feature_names:
        s_meth[cpg] = 0.52  # Median beta
    m_pred = clock.predict(s_meth).iloc[0]

    # Blood prediction
    s_blood = pd.DataFrame([subject])
    b_pred = blood_model.predict(s_blood).iloc[0]

    # Fusion
    s_fused = pd.DataFrame([{
        "sample_id": "DEMO_SUBJECT",
        "chronological_age": 45.0,
        "biological_age_methylation": m_pred["biological_age_methylation"],
        "biological_age_blood": b_pred["biological_age_blood"],
    }])
    f_pred = fusion.predict(s_fused).iloc[0]

    print(f"\n  Methylation Bio Age: {m_pred['biological_age_methylation']:.1f} years")
    print(f"  Blood Bio Age:       {b_pred['biological_age_blood']:.1f} years")
    print(f"  Composite Bio Age:   {f_pred['composite_biological_age']:.1f} years")
    print(f"  Aging Category:      {f_pred['aging_category'].upper()}")
    print(f"  AAI:                 {f_pred['aging_acceleration_index']:+.1f} years")

    # Flag abnormal
    from agent.tools import flag_abnormal_biomarkers
    flags = flag_abnormal_biomarkers(subject)
    if flags["flags"]:
        print(f"\n  ⚠ Abnormal biomarkers ({flags['abnormal_count']}):")
        for f in flags["flags"]:
            print(f"    {f['biomarker']:25s} {f['value']:6.2f}  {f['status']}  (ref: {f['normal_range']})")

    # ── 6. Interventions ───────────────────────────────────────────
    separator("6. Evidence-Based Interventions")
    from agent.tools import suggest_interventions

    interventions = suggest_interventions(
        aging_category=f_pred["aging_category"],
        biomarkers=subject,
        aging_acceleration_index=float(f_pred["aging_acceleration_index"]),
    )

    for rec in interventions["recommendations"][:2]:
        print(f"\n  🎯 {rec['target']}")
        for action in rec["actions"][:3]:
            print(f"     • {action}")
        print(f"     📚 {rec['evidence_base']}")

    # ── 7. RAG search ──────────────────────────────────────────────
    separator("7. RAG: Retrieving Relevant Research")
    from rag.retriever import LongevityRetriever

    retriever = LongevityRetriever()
    retriever.build()

    queries = ["inflammation CRP aging intervention", "glucose HbA1c longevity"]
    for q in queries:
        results = retriever.search(q, k=2)
        print(f"\n  Query: '{q}'")
        for r in results:
            print(f"    [{r['relevance_score']:.3f}] {r['title'][:55]}... ({r['year']})")

    # ── 8. Hallucination check ─────────────────────────────────────
    separator("8. Grounding / Hallucination Detection")
    claim = "Rapamycin extends lifespan in mice by inhibiting mTOR signalling"
    result_check = retriever.verify_claim(claim)
    print(f"\n  Claim:    '{claim}'")
    print(f"  Grounded: {result_check['grounded']} (score={result_check['max_score']:.3f})")
    if result_check["supporting_papers"]:
        print(f"  Source:   {result_check['supporting_papers'][0][:60]}")

    fake_claim = "Drinking coffee cures aging by activating quantum DNA repair"
    result_fake = retriever.verify_claim(fake_claim)
    print(f"\n  Claim:    '{fake_claim}'")
    print(f"  Grounded: {result_fake['grounded']} (score={result_fake['max_score']:.3f})")

    # ── Done ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DEMO COMPLETE — all systems working ✓")
    print(f"{'='*60}")
    print("""
  Next:
    python scripts/train_models.py    # Full training on 656 samples
    streamlit run dashboard/app.py    # Open interactive dashboard
    uvicorn api.main:app --port 8000  # Start REST API
""")


if __name__ == "__main__":
    main()
