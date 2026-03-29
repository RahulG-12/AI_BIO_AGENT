"""
api/main.py

FastAPI backend for the Biological Age Agent.

Endpoints:
  POST /analyse          — full biological age assessment
  POST /chat             — follow-up question
  POST /predict/methylation  — run only Hannum clock
  POST /predict/blood        — run only blood biomarker MLP
  GET  /papers/search    — RAG search over longevity literature
  POST /biomarkers/flag  — check which markers are out of range
  GET  /health           — liveness check

Run: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import time
import os

app = FastAPI(
    title="Biological Age Agent API",
    description="Multi-modal biological age prediction powered by Hannum clock, blood MLP, and longevity RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_hannum = None
_blood_model = None
_fusion_model = None
_agent = None
_retriever = None


def get_hannum():
    global _hannum
    if _hannum is None:
        from models.methylation_model import HannumClock
        _hannum = HannumClock()
        _hannum.load()
    return _hannum


def get_blood_model():
    global _blood_model
    if _blood_model is None:
        from models.blood_biomarker_model import BloodBiomarkerModel
        _blood_model = BloodBiomarkerModel()
        _blood_model.load()
    return _blood_model


def get_fusion():
    global _fusion_model
    if _fusion_model is None:
        from models.fusion_model import FusionModel
        _fusion_model = FusionModel()
        _fusion_model.load()
    return _fusion_model


def get_agent():
    global _agent
    if _agent is None:
        from agent.agent import BiologicalAgeAgent
        _agent = BiologicalAgeAgent()
    return _agent


def get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import LongevityRetriever
        _retriever = LongevityRetriever()
        _retriever.build()
    return _retriever


# ── Request / Response models ─────────────────────────────────────────────────

class BloodBiomarkers(BaseModel):
    crp_mg_l: Optional[float] = None
    glucose_mg_dl: Optional[float] = None
    hdl_mg_dl: Optional[float] = None
    ldl_mg_dl: Optional[float] = None
    hba1c_pct: Optional[float] = None
    albumin_g_dl: Optional[float] = None
    creatinine_mg_dl: Optional[float] = None
    lymphocyte_pct: Optional[float] = None
    rdw_pct: Optional[float] = None
    telomere_score: Optional[float] = None


class MethylationInput(BaseModel):
    sample_id: str = "sample_001"
    chronological_age: float = Field(..., ge=0, le=120)
    cpg_values: dict[str, float] = Field(
        default={},
        description="CpG ID → beta value (0-1). If empty, uses median defaults."
    )


class AnalyseRequest(BaseModel):
    sample_id: str = "sample_001"
    chronological_age: float = Field(..., ge=0, le=120)
    methylation_age: Optional[float] = None
    blood_age: Optional[float] = None
    biomarkers: Optional[BloodBiomarkers] = None
    question: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class PaperSearchRequest(BaseModel):
    query: str
    k: int = Field(default=4, ge=1, le=10)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/predict/methylation")
def predict_methylation(input_data: MethylationInput):
    """Run only the Hannum epigenetic clock on provided CpG values."""
    try:
        model = get_hannum()
        row = {"sample_id": input_data.sample_id, "chronological_age": input_data.chronological_age}
        row.update(input_data.cpg_values)
        df = pd.DataFrame([row])
        result = model.predict(df)
        top_cpgs = model.top_cpgs(5).to_dict("records")
        return {
            "prediction": result.to_dict("records")[0],
            "top_influential_cpgs": top_cpgs,
        }
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run scripts/train_models.py first.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict/blood")
def predict_blood(biomarkers: BloodBiomarkers, chronological_age: float):
    """Run only the blood biomarker MLP."""
    try:
        model = get_blood_model()
        data = biomarkers.model_dump()
        data["chronological_age"] = chronological_age
        df = pd.DataFrame([data])
        result = model.predict(df)
        flags = model.flag_abnormal(data)
        return {
            "prediction": result.to_dict("records")[0],
            "abnormal_biomarkers": flags,
        }
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run scripts/train_models.py first.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/analyse")
def analyse(request: AnalyseRequest):
    """
    Full agentic biological age assessment.
    Requires OPENAI_API_KEY to be set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")

    subject_data = {
        "sample_id": request.sample_id,
        "chronological_age": request.chronological_age,
    }
    if request.methylation_age is not None:
        subject_data["methylation_age"] = request.methylation_age
    if request.blood_age is not None:
        subject_data["blood_age"] = request.blood_age
    if request.biomarkers:
        subject_data["biomarkers"] = request.biomarkers.model_dump(exclude_none=True)

    try:
        agent = get_agent()
        report = agent.analyse(subject_data, request.question)
        return {"report": report, "sample_id": request.sample_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    """Follow-up questions after an analysis."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not set.")
    try:
        agent = get_agent()
        reply = agent.chat(request.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/papers/search")
def search_papers(request: PaperSearchRequest):
    """Semantic search over the longevity literature corpus."""
    try:
        retriever = get_retriever()
        results = retriever.search(request.query, k=request.k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/biomarkers/flag")
def flag_biomarkers(biomarkers: BloodBiomarkers):
    """Identify which blood biomarkers are outside healthy reference ranges."""
    from agent.tools import flag_abnormal_biomarkers
    data = biomarkers.model_dump(exclude_none=True)
    return flag_abnormal_biomarkers(data)


@app.get("/papers/all")
def get_all_papers():
    """Return the full paper corpus metadata."""
    from rag.corpus import get_all_papers
    papers = get_all_papers()
    # Strip abstract for lightweight listing
    return [
        {"id": p["id"], "title": p["title"], "authors": p["authors"],
         "journal": p["journal"], "year": p["year"], "tags": p["tags"]}
        for p in papers
    ]
