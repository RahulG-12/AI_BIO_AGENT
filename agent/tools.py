"""
agent/tools.py

The 5 tools available to the Biological Age Agent.
Each tool is a standalone function — the agent decides which to call.

Tools:
  1. predict_biological_age   — run fusion model on sample
  2. search_longevity_papers  — RAG query over PubMed corpus
  3. suggest_interventions    — evidence-based recommendations based on age gap
  4. explain_biomarker        — explain what a specific biomarker means
  5. flag_abnormal_biomarkers — identify out-of-range blood values
"""

import json
from typing import Any

# Lazy imports to avoid loading models unless needed
_fusion_model = None
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import LongevityRetriever
        _retriever = LongevityRetriever()
        _retriever.build()
    return _retriever


def _get_fusion_model():
    global _fusion_model
    if _fusion_model is None:
        from models.fusion_model import FusionModel
        _fusion_model = FusionModel()
        _fusion_model.load()
    return _fusion_model


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Predict biological age
# ─────────────────────────────────────────────────────────────────────────────

def predict_biological_age(
    chronological_age: float,
    methylation_age: float | None = None,
    blood_age: float | None = None,
) -> dict:
    """
    Compute composite biological age from sub-model predictions.
    If only one modality is available, use that directly.
    Returns aging_acceleration_index and aging_category.
    """
    import pandas as pd

    if methylation_age is None and blood_age is None:
        return {"error": "At least one of methylation_age or blood_age must be provided."}

    if methylation_age is not None and blood_age is not None:
        try:
            model = _get_fusion_model()
            df = pd.DataFrame([{
                "chronological_age": chronological_age,
                "biological_age_methylation": methylation_age,
                "biological_age_blood": blood_age,
            }])
            result = model.predict(df)
            row = result.iloc[0]
            return {
                "composite_biological_age": float(row["composite_biological_age"]),
                "aging_acceleration_index": float(row["aging_acceleration_index"]),
                "aging_category": row["aging_category"],
                "chronological_age": chronological_age,
                "methylation_age": methylation_age,
                "blood_age": blood_age,
            }
        except Exception as e:
            # Fallback: simple average if model not loaded
            composite = (methylation_age + blood_age) / 2
    else:
        composite = methylation_age if methylation_age is not None else blood_age

    gap = composite - chronological_age
    if gap <= -7:   cat = "exceptional_longevity"
    elif gap <= -3: cat = "slower_aging"
    elif gap < 3:   cat = "typical_aging"
    elif gap < 7:   cat = "accelerated_aging"
    else:           cat = "significantly_accelerated"

    return {
        "composite_biological_age": round(composite, 2),
        "aging_acceleration_index": round(gap, 2),
        "aging_category": cat,
        "chronological_age": chronological_age,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Search longevity papers (RAG)
# ─────────────────────────────────────────────────────────────────────────────

def search_longevity_papers(query: str, k: int = 3) -> dict:
    """
    Retrieve the most relevant longevity research papers for a given query.
    Uses FAISS semantic search over 30+ curated PubMed abstracts.
    """
    retriever = _get_retriever()
    results = retriever.search(query, k=k)
    return {
        "query": query,
        "results": [
            {
                "title": r["title"],
                "authors": r["authors"],
                "journal": r["journal"],
                "year": r["year"],
                "abstract_snippet": r["abstract"][:300] + "...",
                "tags": r["tags"],
                "relevance_score": round(r["relevance_score"], 3),
            }
            for r in results
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Suggest interventions
# ─────────────────────────────────────────────────────────────────────────────

INTERVENTIONS = {
    "high_crp": {
        "biomarker": "crp_mg_l",
        "condition": lambda v: v > 3.0,
        "intervention": "Reduce systemic inflammation",
        "actions": [
            "Anti-inflammatory diet (Mediterranean, reduce ultra-processed foods)",
            "Aerobic exercise ≥150 min/week reduces CRP by ~30%",
            "Omega-3 supplementation (2-4g EPA/DHA daily)",
            "Treat underlying infections or autoimmune conditions",
        ],
        "evidence": "Levine 2018 PhenoAge; Mather 2011 blood biomarkers",
        "papers_query": "CRP inflammation aging intervention",
    },
    "high_glucose": {
        "biomarker": "glucose_mg_dl",
        "condition": lambda v: v > 100,
        "intervention": "Improve glucose metabolism",
        "actions": [
            "Time-restricted eating (16:8 intermittent fasting)",
            "Reduce refined carbohydrates and added sugars",
            "Resistance training improves insulin sensitivity acutely",
            "Consider metformin discussion with physician if pre-diabetic",
            "Post-meal walks (10-15 min) reduce glucose spikes by ~30%",
        ],
        "evidence": "Fontana 2010; Barzilai 2016 TAME trial",
        "papers_query": "glucose insulin metformin aging intervention",
    },
    "low_hdl": {
        "biomarker": "hdl_mg_dl",
        "condition": lambda v: v < 40,
        "intervention": "Raise HDL cholesterol",
        "actions": [
            "Regular aerobic exercise is the most effective HDL raiser",
            "Olive oil and avocado increase HDL",
            "Reduce trans fats completely",
            "Moderate alcohol has epidemiological HDL association but not recommended therapeutically",
            "Niacin supplementation (physician-supervised)",
        ],
        "evidence": "Ferrucci 2020 multi-omic aging review",
        "papers_query": "HDL cholesterol cardiovascular aging",
    },
    "low_albumin": {
        "biomarker": "albumin_g_dl",
        "condition": lambda v: v < 3.5,
        "intervention": "Improve nutritional status and liver function",
        "actions": [
            "Increase dietary protein to ≥1.2g/kg body weight",
            "Rule out malabsorption, liver disease, or chronic inflammation",
            "Resistance training preserves muscle and albumin with age",
            "Ensure adequate caloric intake — albumin is a nutritional marker",
        ],
        "evidence": "Levine 2018 PhenoAge (albumin component)",
        "papers_query": "albumin frailty aging nutrition intervention",
    },
    "high_hba1c": {
        "biomarker": "hba1c_pct",
        "condition": lambda v: v >= 5.7,
        "intervention": "Reduce glycated haemoglobin",
        "actions": [
            "Continuous glucose monitoring to identify spike triggers",
            "Low glycaemic index diet",
            "Metformin for pre-diabetes (discuss with physician)",
            "High-intensity interval training (HIIT) improves HbA1c effectively",
        ],
        "evidence": "Barzilai 2016 TAME; Fontana 2010 caloric restriction",
        "papers_query": "HbA1c glycation aging metabolic intervention",
    },
    "low_telomere": {
        "biomarker": "telomere_score",
        "condition": lambda v: v < 0.7,
        "intervention": "Protect telomere integrity",
        "actions": [
            "Chronic stress reduction — telomeres are acutely stress-sensitive",
            "Aerobic exercise — endurance athletes have telomeres 10-15 years younger",
            "Omega-3 supplementation associated with telomere preservation",
            "Avoid smoking — single largest modifiable telomere shortener",
            "Vitamin D optimisation (40-60 ng/mL serum level)",
        ],
        "evidence": "Blackburn 2015 telomeres; Denham 2018 exercise telomere",
        "papers_query": "telomere length intervention exercise aging",
    },
    "high_rdw": {
        "biomarker": "rdw_pct",
        "condition": lambda v: v > 14.5,
        "intervention": "Address nutritional deficiencies and oxidative stress",
        "actions": [
            "Rule out iron, B12, or folate deficiency — most common RDW causes",
            "Antioxidant-rich diet (berries, leafy greens)",
            "Reduce alcohol consumption",
            "Address thyroid dysfunction if present",
        ],
        "evidence": "Mather 2011; Crimmins 2022 epidemiology biomarkers",
        "papers_query": "RDW red cell mortality aging biomarker",
    },
}

# General longevity interventions by aging acceleration level
GENERAL_INTERVENTIONS = {
    "exceptional_longevity": [
        "Maintain current lifestyle — your biological age is exceptional",
        "Document lifestyle factors: sleep, diet, exercise, stress for research value",
        "Consider joining a longevity cohort study",
    ],
    "slower_aging": [
        "Continue current health practices — you are aging more slowly than average",
        "Monitor annually with the same biomarker panel",
        "Consider NAD+ precursor supplementation (NMN/NR) to maintain trajectory",
    ],
    "typical_aging": [
        "Target Mediterranean-style diet and 150+ min/week aerobic exercise",
        "Optimise sleep: 7-9 hours with consistent schedule",
        "Stress management: meditation or breathing practices reduce epigenetic clock acceleration",
        "Annual biomarker monitoring",
    ],
    "accelerated_aging": [
        "Priority: identify and address the highest-impact modifiable factors first",
        "Consult a longevity-focused physician for personalised assessment",
        "Focus on sleep quality, exercise, and anti-inflammatory diet simultaneously",
        "Consider serum NAD+ testing and NMN/NR supplementation",
        "Evaluate metformin candidacy with physician",
    ],
    "significantly_accelerated": [
        "Seek comprehensive medical evaluation — rule out underlying disease",
        "Aggressive lifestyle intervention across all domains simultaneously",
        "Consider rapamycin (discuss with longevity physician) — strongest evidence base for mTOR inhibition",
        "Senolytic protocol (dasatinib+quercetin) may be appropriate — physician supervised",
        "6-month reassessment with full biomarker panel",
    ],
}


def suggest_interventions(
    aging_category: str,
    biomarkers: dict | None = None,
    aging_acceleration_index: float | None = None,
) -> dict:
    """
    Suggest evidence-based longevity interventions based on:
    - Aging category (from fusion model)
    - Specific out-of-range biomarkers
    """
    recommendations = []

    # Biomarker-specific recommendations
    if biomarkers:
        for key, spec in INTERVENTIONS.items():
            bm_val = biomarkers.get(spec["biomarker"])
            if bm_val is not None and spec["condition"](bm_val):
                recommendations.append({
                    "priority": "biomarker_specific",
                    "target": spec["intervention"],
                    "actions": spec["actions"],
                    "evidence_base": spec["evidence"],
                })

    # General recommendations by aging category
    general = GENERAL_INTERVENTIONS.get(aging_category, GENERAL_INTERVENTIONS["typical_aging"])
    recommendations.append({
        "priority": "general",
        "target": f"Overall aging rate ({aging_category.replace('_', ' ')})",
        "actions": general,
        "evidence_base": "Lopez-Otin 2013 Hallmarks; Fontana 2010; Campisi 2019",
    })

    return {
        "aging_category": aging_category,
        "aging_acceleration_index": aging_acceleration_index,
        "recommendations": recommendations,
        "disclaimer": (
            "These suggestions are based on published research. "
            "Consult a physician before starting any supplement or medication protocol."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: Explain biomarker
# ─────────────────────────────────────────────────────────────────────────────

BIOMARKER_EXPLAINERS = {
    "crp_mg_l": {
        "name": "C-Reactive Protein (CRP)",
        "what_it_is": "An acute-phase protein produced by the liver in response to inflammation.",
        "why_it_matters_for_aging": (
            "Chronic low-grade inflammation ('inflammaging') is a hallmark of aging. "
            "Elevated CRP is associated with accelerated PhenoAge and increased risk of "
            "cardiovascular disease, cancer, and all-cause mortality."
        ),
        "optimal_range": "< 1.0 mg/L for longevity; < 3.0 mg/L is normal",
        "reference": "Levine 2018 PhenoAge; Ridker 2003",
    },
    "telomere_score": {
        "name": "Telomere Length Score",
        "what_it_is": "Relative telomere length in leukocytes, normalised to a reference population.",
        "why_it_matters_for_aging": (
            "Telomeres cap chromosome ends and shorten with each cell division. "
            "Short telomeres trigger senescence and are associated with earlier onset "
            "of age-related diseases and shorter lifespan."
        ),
        "optimal_range": "> 1.0 (above population median for age)",
        "reference": "Blackburn 2015; Epel 2009",
    },
    "glucose_mg_dl": {
        "name": "Fasting Glucose",
        "what_it_is": "Blood glucose measured after ≥8 hours of fasting.",
        "why_it_matters_for_aging": (
            "Chronic hyperglycaemia accelerates glycation of proteins (including haemoglobin), "
            "drives oxidative stress, and is a component of the PhenoAge biological clock. "
            "Fasting glucose < 90 mg/dL is associated with optimal longevity outcomes."
        ),
        "optimal_range": "70–90 mg/dL for longevity; up to 99 mg/dL is clinically normal",
        "reference": "Levine 2018; Barzilai 2016",
    },
    "hba1c_pct": {
        "name": "Glycated Haemoglobin (HbA1c)",
        "what_it_is": "Percentage of haemoglobin with attached glucose, reflecting 3-month average blood sugar.",
        "why_it_matters_for_aging": (
            "HbA1c is a PhenoAge component and a marker of cumulative glycaemic exposure. "
            "Elevated HbA1c predicts cardiovascular disease, kidney disease, and accelerated "
            "epigenetic aging. Each 1% increase in HbA1c above 5.7% is associated with "
            "measurable epigenetic clock acceleration."
        ),
        "optimal_range": "< 5.4% for longevity; < 5.7% is non-diabetic",
        "reference": "Levine 2018; Barzilai 2016 TAME",
    },
    "albumin_g_dl": {
        "name": "Serum Albumin",
        "what_it_is": "The most abundant protein in blood plasma, produced by the liver.",
        "why_it_matters_for_aging": (
            "Albumin reflects nutritional status, liver function, and chronic inflammation. "
            "It is a component of PhenoAge. Declining albumin tracks biological aging and "
            "predicts frailty and mortality risk in older adults."
        ),
        "optimal_range": "4.0–5.0 g/dL; < 3.5 g/dL is clinically low",
        "reference": "Levine 2018 PhenoAge; Crimmins 2022",
    },
    "rdw_pct": {
        "name": "Red Cell Distribution Width (RDW)",
        "what_it_is": "A measure of variation in red blood cell size (anisocytosis).",
        "why_it_matters_for_aging": (
            "Elevated RDW reflects oxidative stress, nutritional deficiency, and systemic "
            "inflammation. It is a surprisingly strong predictor of all-cause mortality "
            "and is a component of the PhenoAge biological clock. RDW increases "
            "progressively with age independent of anaemia."
        ),
        "optimal_range": "11.5–13.0% for longevity",
        "reference": "Levine 2018; Patel 2009",
    },
    "hdl_mg_dl": {
        "name": "HDL Cholesterol",
        "what_it_is": "'Good' cholesterol that transports lipids from tissues to the liver.",
        "why_it_matters_for_aging": (
            "HDL has anti-inflammatory and antioxidant functions beyond lipid transport. "
            "Low HDL is associated with cardiovascular disease, inflammation, and accelerated "
            "biological aging. HDL declines with sedentary behaviour and poor diet."
        ),
        "optimal_range": "> 60 mg/dL for longevity; > 40 (M) / > 50 (F) is clinically normal",
        "reference": "Ferrucci 2020; Crimmins 2022",
    },
    "lymphocyte_pct": {
        "name": "Lymphocyte Percentage",
        "what_it_is": "The proportion of lymphocytes (T, B, NK cells) among white blood cells.",
        "why_it_matters_for_aging": (
            "Lymphocyte percentage declines with immunosenescence — the gradual deterioration "
            "of immune function with age. It is a PhenoAge component. Low lymphocyte % "
            "reflects impaired adaptive immunity, increased infection susceptibility, and "
            "reduced cancer surveillance."
        ),
        "optimal_range": "25–40%; < 20% indicates immunosenescence",
        "reference": "Levine 2018 PhenoAge; Crimmins 2022",
    },
    "creatinine_mg_dl": {
        "name": "Serum Creatinine",
        "what_it_is": "A waste product of muscle metabolism cleared by the kidneys.",
        "why_it_matters_for_aging": (
            "Creatinine is a kidney function marker. Declining GFR with age leads to creatinine "
            "accumulation. Chronic kidney disease accelerates cardiovascular aging and is itself "
            "accelerated by inflammation and hyperglycaemia. A PhenoAge component."
        ),
        "optimal_range": "0.6–1.0 mg/dL (women); 0.7–1.1 mg/dL (men)",
        "reference": "Levine 2018; Ferrucci 2020",
    },
}


def explain_biomarker(biomarker_name: str) -> dict:
    """
    Return a detailed explanation of a biomarker's role in aging.
    biomarker_name: one of the BLOOD_FEATURES keys
    """
    # Normalise input
    key = biomarker_name.lower().replace(" ", "_").replace("-", "_")
    if key not in BIOMARKER_EXPLAINERS:
        # Try partial match
        matches = [k for k in BIOMARKER_EXPLAINERS if key in k or k in key]
        if matches:
            key = matches[0]
        else:
            return {
                "error": f"Biomarker '{biomarker_name}' not found.",
                "available": list(BIOMARKER_EXPLAINERS.keys()),
            }
    return BIOMARKER_EXPLAINERS[key]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: Flag abnormal biomarkers
# ─────────────────────────────────────────────────────────────────────────────

def flag_abnormal_biomarkers(biomarkers: dict) -> dict:
    """
    Identify which blood biomarkers are outside healthy ranges.
    Input: dict of {biomarker_name: value}
    """
    from models.blood_biomarker_model import BloodBiomarkerModel, CLINICAL_RANGES

    flags = []
    optimal = []

    for feat, (lo, hi, unit, desc) in CLINICAL_RANGES.items():
        val = biomarkers.get(feat)
        if val is None:
            continue
        if val < lo:
            flags.append({
                "biomarker": feat, "value": val, "unit": unit,
                "status": "LOW", "description": desc,
                "normal_range": f"{lo}–{hi} {unit}",
                "severity": "mild" if val > lo * 0.85 else "moderate",
            })
        elif val > hi:
            flags.append({
                "biomarker": feat, "value": val, "unit": unit,
                "status": "HIGH", "description": desc,
                "normal_range": f"{lo}–{hi} {unit}",
                "severity": "mild" if val < hi * 1.15 else "moderate",
            })
        else:
            optimal.append(feat)

    return {
        "abnormal_count": len(flags),
        "optimal_count": len(optimal),
        "flags": flags,
        "optimal_biomarkers": optimal,
    }


# ── Tool registry for agent ───────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "predict_biological_age",
            "description": (
                "Compute composite biological age and aging acceleration index "
                "from sub-model predictions (methylation clock and/or blood biomarker MLP)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chronological_age": {"type": "number", "description": "Subject's actual age in years"},
                    "methylation_age": {"type": "number", "description": "Biological age from epigenetic clock (optional)"},
                    "blood_age": {"type": "number", "description": "Biological age from blood biomarker MLP (optional)"},
                },
                "required": ["chronological_age"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_longevity_papers",
            "description": "Search the longevity research literature for relevant papers using semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research question or topic to search for"},
                    "k": {"type": "integer", "description": "Number of papers to return (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_interventions",
            "description": "Suggest evidence-based longevity interventions based on aging category and biomarkers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aging_category": {
                        "type": "string",
                        "enum": ["exceptional_longevity", "slower_aging", "typical_aging",
                                 "accelerated_aging", "significantly_accelerated"],
                    },
                    "biomarkers": {
                        "type": "object",
                        "description": "Dict of biomarker name → value for personalised recommendations",
                    },
                    "aging_acceleration_index": {
                        "type": "number",
                        "description": "Composite biological age minus chronological age",
                    },
                },
                "required": ["aging_category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_biomarker",
            "description": "Get a detailed explanation of a biomarker's role in biological aging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "biomarker_name": {
                        "type": "string",
                        "description": "e.g. 'crp_mg_l', 'telomere_score', 'hba1c_pct'",
                    },
                },
                "required": ["biomarker_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_abnormal_biomarkers",
            "description": "Identify which blood biomarkers are outside healthy ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "biomarkers": {
                        "type": "object",
                        "description": "Dict of biomarker names to values",
                    },
                },
                "required": ["biomarkers"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "predict_biological_age": predict_biological_age,
    "search_longevity_papers": search_longevity_papers,
    "suggest_interventions": suggest_interventions,
    "explain_biomarker": explain_biomarker,
    "flag_abnormal_biomarkers": flag_abnormal_biomarkers,
}
