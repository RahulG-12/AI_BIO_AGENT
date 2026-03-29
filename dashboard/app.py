"""
dashboard/app.py

Premium Luxury Biotech UI — Immortigen Biological Age Agent
Top-level 0.01% design · Light theme only
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
import torch
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.blood_biomarker_model import BloodBiomarkerModel


st.set_page_config(
    page_title="Biological Age Agent — Immortigen",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PREMIUM LUXURY CSS ─────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── ROOT VARIABLES ── */
:root {
    --pearl:        #fafaf8;
    --cream:        #f5f3ee;
    --warm-white:   #ffffff;
    --ink:          #0d0d0d;
    --ink-muted:    #3d3935;
    --ink-faint:    #8a837a;
    --gold:         #c9973a;
    --gold-light:   #e8c97a;
    --teal:         #0e7b6b;
    --teal-light:   #12a08a;
    --teal-pale:    #e6f5f2;
    --rose:         #c0473e;
    --rose-pale:    #fdf0ef;
    --amber:        #d97706;
    --amber-pale:   #fef3c7;
    --border:       #e8e2d9;
    --border-soft:  #f0ece5;
    --shadow-sm:    0 2px 8px rgba(13,13,13,0.06);
    --shadow-md:    0 8px 32px rgba(13,13,13,0.08);
    --shadow-lg:    0 24px 64px rgba(13,13,13,0.10);
    --radius:       16px;
    --radius-sm:    10px;
}

/* ── GLOBAL RESET ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--pearl) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

[data-testid="stMain"] {
    background: var(--pearl) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    bac0kground: var(--warm-white) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 24px rgba(13,13,13,0.04) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.4rem;
}

/* Sidebar logo area */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 1.5rem;
}
.sidebar-brand-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--teal), #0a5a4e);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 12px rgba(14,123,107,0.25);
}
.sidebar-brand-name {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    font-size: 1.3rem;
    color: var(--ink);
    letter-spacing: 0.02em;
}
.sidebar-brand-tag {
    font-size: 0.62rem;
    font-weight: 500;
    color: var(--teal);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: -2px;
}

/* Sidebar section labels */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 1.4rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-soft);
}

/* Sidebar inputs */
[data-testid="stSidebar"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--ink-muted) !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] {
    background: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--ink) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stSidebar"] input[type="number"]:focus,
[data-testid="stSidebar"] input[type="text"]:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(14,123,107,0.12) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--teal), var(--teal-light)) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: var(--teal) !important;
    color: white !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── RUN BUTTON ── */
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, var(--teal) 0%, #0a5a4e 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 16px rgba(14,123,107,0.30), 0 1px 4px rgba(14,123,107,0.20) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(14,123,107,0.35), 0 2px 8px rgba(14,123,107,0.25) !important;
}
[data-testid="stSidebar"] .stButton button:active {
    transform: translateY(0) !important;
}

/* ── HEADER AREA ── */
.page-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.5rem;
}
.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.1;
    margin: 0;
    letter-spacing: -0.02em;
}
.page-title span {
    background: linear-gradient(135deg, var(--teal), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-subtitle {
    font-size: 0.9rem;
    color: var(--ink-faint);
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
    margin: 1.8rem 0;
}

/* ── METRIC CARDS ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: var(--warm-white);
    border-radius: var(--radius);
    padding: 1.4rem 1.5rem 1.3rem;
    border: 1px solid var(--border-soft);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.3s, transform 0.3s;
    animation: cardIn 0.5s ease both;
}
.metric-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.metric-card.teal::before  { background: linear-gradient(90deg, var(--teal), var(--teal-light)); }
.metric-card.gold::before  { background: linear-gradient(90deg, var(--gold), var(--gold-light)); }
.metric-card.rose::before  { background: linear-gradient(90deg, var(--rose), #e06060); }
.metric-card.ink::before   { background: linear-gradient(90deg, #3d3935, #8a837a); }

.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.metric-delta {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 20px;
}
.metric-delta.positive {
    background: var(--rose-pale);
    color: var(--rose);
}
.metric-delta.negative {
    background: var(--teal-pale);
    color: var(--teal);
}
.metric-delta.neutral {
    background: var(--cream);
    color: var(--ink-faint);
}

/* ── STATUS CARD ── */
.status-card {
    background: var(--warm-white);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    border: 1px solid var(--border-soft);
    box-shadow: var(--shadow-sm);
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    animation: cardIn 0.6s ease both;
    animation-delay: 0.1s;
}
.status-icon {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    flex-shrink: 0;
}
.status-content {}
.status-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.2rem;
}
.status-meta {
    font-size: 0.82rem;
    color: var(--ink-faint);
    font-weight: 400;
}
.status-badge {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}
.aai-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    letter-spacing: -0.02em;
}
.aai-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--ink-faint);
}

/* ── BIOMARKER TABLE ── */
.bm-section {
    background: var(--warm-white);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    border: 1px solid var(--border-soft);
    box-shadow: var(--shadow-sm);
    animation: cardIn 0.6s ease both;
    animation-delay: 0.2s;
}
.bm-section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.bm-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border-soft);
}
.bm-row:last-child { border-bottom: none; }
.bm-name {
    width: 110px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--ink-muted);
    letter-spacing: 0.01em;
}
.bm-bar-wrap {
    flex: 1;
    height: 6px;
    background: var(--cream);
    border-radius: 99px;
    overflow: hidden;
}
.bm-bar {
    height: 100%;
    border-radius: 99px;
    transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.bm-val {
    width: 70px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--ink);
}
.bm-status {
    width: 20px;
    text-align: center;
    font-size: 0.8rem;
}

/* ── CHART SECTION ── */
.chart-section {
    background: var(--warm-white);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    border: 1px solid var(--border-soft);
    box-shadow: var(--shadow-sm);
    animation: cardIn 0.6s ease both;
    animation-delay: 0.15s;
}
.chart-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.2rem;
}
.chart-sub {
    font-size: 0.78rem;
    color: var(--ink-faint);
    margin-bottom: 1rem;
}

/* ── INFO BANNER ── */
.info-banner {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.85rem;
    color: var(--ink-muted);
}

/* ── ANIMATIONS ── */
@keyframes cardIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── HIDE STREAMLIT METRICS (replaced by custom) ── */
[data-testid="metric-container"] { display: none !important; }

/* ── PLOTLY TWEAKS ── */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── STINFO ── */
.stAlert {
    background: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-muted) !important;
}

/* ── RESPONSIVE ── */
@media (max-width: 1100px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧬</div>
        <div>
            <div class="sidebar-brand-name">Immortigen</div>
            <div class="sidebar-brand-tag">BioAge Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sample_id = st.text_input("Sample ID", "SAMPLE_001")
    chron_age = st.slider("Chronological Age", 20, 100, 45)

    st.markdown('<div class="section-label">Epigenetic</div>', unsafe_allow_html=True)
    methylation_age = st.number_input("Methylation Age (Horvath Clock)", value=float(chron_age), step=0.1)

    st.markdown('<div class="section-label">Blood Biomarkers</div>', unsafe_allow_html=True)
    crp      = st.number_input("CRP  (mg/L)", 0.0, 30.0, 2.0, step=0.1)
    glucose  = st.number_input("Glucose  (mg/dL)", 50, 300, 90)
    hdl      = st.number_input("HDL Cholesterol  (mg/dL)", 10, 120, 50)
    hba1c    = st.number_input("HbA1c  (%)", 3.0, 15.0, 5.2, step=0.1)
    albumin  = st.number_input("Albumin  (g/dL)", 1.0, 6.0, 4.0, step=0.1)
    rdw      = st.number_input("RDW  (%)", 9.0, 22.0, 13.0, step=0.1)
    telomere = st.number_input("Telomere Length  (T/S ratio)", 0.2, 2.0, 0.9, step=0.01)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔬  Run Biological Analysis", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem; padding:1rem; background:#f5f3ee; border-radius:10px; border:1px solid #e8e2d9;">
        <div style="font-size:0.65rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#8a837a; margin-bottom:0.5rem;">Disclaimer</div>
        <div style="font-size:0.72rem; color:#8a837a; line-height:1.6;">
            Research tool only. Not for clinical diagnosis. Consult a qualified physician for medical decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── PAGE HEADER ────────────────────────────────────────────
st.markdown("""
<div class="page-eyebrow">Immortigen · Longevity Science Platform</div>
<div class="page-title">Biological Age <span>Intelligence</span></div>
<div class="page-subtitle">Multi-modal aging biomarker analysis · Epigenetic + blood-based composite scoring</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div class="info-banner">
        <span style="font-size:1.3rem">🔬</span>
        <span>Enter subject biomarkers in the left panel and click <strong>Run Biological Analysis</strong> to generate a comprehensive aging profile.</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── COMPUTATION ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = BloodBiomarkerModel()

    # Load weights manually
    model.model.load_state_dict(
        torch.load("models/saved/blood_mlp.pt", map_location="cpu")
    )

    # Load scaler
    model.scaler = joblib.load("models/saved/blood_scaler.pkl")

    model.model.eval()

    return model

blood_model = load_model()

# Correct format for ML model
bm_model = {
    "crp_mg_l": crp,
    "glucose_mg_dl": glucose,
    "hdl_mg_dl": hdl,
    "hba1c_pct": hba1c,
    "albumin_g_dl": albumin,
    "rdw_pct": rdw,
    "telomere_score": telomere
}

# REAL prediction
blood_age = float(blood_model.predict(bm_model))
composite_age  = round((methylation_age * 0.55 + blood_age * 0.45), 1)
aai            = composite_age - chron_age

# Category
if aai <= -5:
    cat_label = "Exceptional Aging"
    cat_icon  = "🔥"
    cat_color = "#c9973a"
    cat_bg    = "#fef8ec"
    cat_border= "#e8c97a"
    aai_color = "#c9973a"
elif aai <= -2:
    cat_label = "Slower Aging"
    cat_icon  = "✦"
    cat_color = "#0e7b6b"
    cat_bg    = "#e6f5f2"
    cat_border= "#a7ddd5"
    aai_color = "#0e7b6b"
elif aai < 3:
    cat_label = "Normal Aging"
    cat_icon  = "◎"
    cat_color = "#3d3935"
    cat_bg    = "#f5f3ee"
    cat_border= "#e8e2d9"
    aai_color = "#3d3935"
else:
    cat_label = "Accelerated Aging"
    cat_icon  = "▲"
    cat_color = "#c0473e"
    cat_bg    = "#fdf0ef"
    cat_border= "#f0a09c"
    aai_color = "#c0473e"

# ── METRIC CARDS ────────────────────────────────────────────
epi_delta  = methylation_age - chron_age
blood_delta = blood_age - chron_age

def delta_class(d): return "negative" if d < 0 else ("positive" if d > 0 else "neutral")
def delta_arrow(d): return "↓" if d < 0 else ("↑" if d > 0 else "→")

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card ink">
        <div class="metric-label">Chronological Age</div>
        <div class="metric-value">{chron_age}</div>
        <span class="metric-delta neutral">Reference</span>
    </div>
    <div class="metric-card teal">
        <div class="metric-label">Epigenetic Age</div>
        <div class="metric-value">{methylation_age:.1f}</div>
        <span class="metric-delta {delta_class(epi_delta)}">{delta_arrow(epi_delta)} {abs(epi_delta):.1f} yrs</span>
    </div>
    <div class="metric-card gold">
        <div class="metric-label">Blood Biomarker Age</div>
        <div class="metric-value">{blood_age:.1f}</div>
        <span class="metric-delta {delta_class(blood_delta)}">{delta_arrow(blood_delta)} {abs(blood_delta):.1f} yrs</span>
    </div>
    <div class="metric-card rose">
        <div class="metric-label">Composite Bio-Age</div>
        <div class="metric-value">{composite_age:.1f}</div>
        <span class="metric-delta {delta_class(aai)}">{delta_arrow(aai)} {abs(aai):.1f} yrs AAI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS CARD ─────────────────────────────────────────────
st.markdown(f"""
<div class="status-card" style="border-color:{cat_border}; background:linear-gradient(135deg, {cat_bg} 0%, white 60%);">
    <div class="status-icon" style="background:{cat_bg}; border:1px solid {cat_border}; font-size:1.4rem;">
        {cat_icon}
    </div>
    <div class="status-content">
        <div class="status-title" style="color:{cat_color};">{cat_label}</div>
        <div class="status-meta">Sample: <strong>{sample_id}</strong> &nbsp;·&nbsp; Chronological: <strong>{chron_age} yrs</strong> &nbsp;·&nbsp; Composite: <strong>{composite_age} yrs</strong></div>
    </div>
    <div class="status-badge">
        <div class="aai-value" style="color:{aai_color};">{aai:+.1f}</div>
        <div class="aai-label">Age Acceleration Index</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ────────────────────────────────────────
col_a, col_b = st.columns([1.35, 1], gap="large")

with col_a:
    # ── RADAR / COMPOSITE CHART ──
    ages_labels = ["Chronological", "Epigenetic", "Blood Biomarker", "Composite"]
    ages_values = [chron_age, methylation_age, blood_age, composite_age]

    bar_colors = ["#8a837a", "#0e7b6b", "#c9973a", "#c0473e"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=ages_labels,
        y=ages_values,
        marker=dict(
            color=bar_colors,
            cornerradius=8,
        ),
        text=[f"{v:.1f}" for v in ages_values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#0d0d0d"),
        width=0.45,
    ))

    # Reference line
    fig_bar.add_hline(
        y=chron_age,
        line=dict(color="#8a837a", width=1.5, dash="dot"),
        annotation_text="Chronological",
        annotation_font=dict(size=10, color="#8a837a", family="DM Sans"),
        annotation_position="top right",
    )

    fig_bar.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(family="DM Sans", size=12, color="#3d3935"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f0ece5", zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=11, color="#8a837a"),
            ticksuffix=" yr",
        ),
        showlegend=False,
    )

    st.markdown("""
    <div class="chart-section">
        <div class="chart-title">Age Comparison</div>
        <div class="chart-sub">Chronological vs biological age across all modalities</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    # ── BIOMARKER PANEL ── (fully inline-styled for Streamlit column compatibility)
    biomarkers = [
        {"name": "CRP",      "val": crp,      "min": 0,   "max": 10,  "ok_max": 3,    "unit": "mg/L",  "lower_better": True},
        {"name": "Glucose",  "val": glucose,  "min": 50,  "max": 200, "ok_max": 95,   "unit": "mg/dL", "lower_better": True},
        {"name": "HDL",      "val": hdl,      "min": 10,  "max": 100, "ok_min": 50,   "unit": "mg/dL", "lower_better": False},
        {"name": "HbA1c",    "val": hba1c,    "min": 3,   "max": 10,  "ok_max": 5.4,  "unit": "%",     "lower_better": True},
        {"name": "Albumin",  "val": albumin,  "min": 1,   "max": 6,   "ok_min": 4,    "unit": "g/dL",  "lower_better": False},
        {"name": "RDW",      "val": rdw,      "min": 9,   "max": 22,  "ok_max": 14,   "unit": "%",     "lower_better": True},
        {"name": "Telomere", "val": telomere, "min": 0.2, "max": 2.0, "ok_min": 1,    "unit": "T/S",   "lower_better": False},
    ]

    rows_html = ""
    for bm_item in biomarkers:
        v    = bm_item["val"]
        vmin = bm_item["min"]
        vmax = bm_item["max"]
        pct  = max(0, min(100, (v - vmin) / (vmax - vmin) * 100))
        good = (v <= bm_item["ok_max"]) if bm_item["lower_better"] else (v >= bm_item["ok_min"])

        bar_col   = "#0e7b6b" if good else "#c0473e"
        s_col     = "#0e7b6b" if good else "#c0473e"
        s_bg      = "#e6f5f2" if good else "#fdf0ef"
        status    = "✓" if good else "✗"
        val_disp  = f"{v:.2f}" if isinstance(v, float) else str(v)

        rows_html += f"""
        <div style="display:flex; align-items:center; gap:10px; padding:10px 0;
                    border-bottom:1px solid #f0ece5;">
            <div style="width:72px; font-size:0.76rem; font-weight:600;
                        color:#3d3935; font-family:'DM Sans',sans-serif;
                        letter-spacing:0.01em; flex-shrink:0;">
                {bm_item['name']}
            </div>
            <div style="flex:1; height:6px; background:#f5f3ee;
                        border-radius:99px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:{bar_col};
                            border-radius:99px;"></div>
            </div>
            <div style="width:80px; text-align:right; font-family:'JetBrains Mono',monospace;
                        font-size:0.76rem; font-weight:500; color:#0d0d0d; flex-shrink:0;">
                {val_disp}
                <span style="color:#8a837a; font-size:0.62rem; margin-left:2px;">{bm_item['unit']}</span>
            </div>
            <div style="width:26px; height:26px; border-radius:50%; display:flex;
                        align-items:center; justify-content:center; flex-shrink:0;
                        background:{s_bg}; color:{s_col}; font-size:0.75rem; font-weight:700;">
                {status}
            </div>
        </div>
        """

    import streamlit.components.v1 as components
    bm_panel = f"""<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}body{{background:transparent;font-family:'DM Sans',sans-serif;}}</style>
</head><body>
<div style="background:white;border-radius:16px;padding:20px 22px;border:1px solid #f0ece5;box-shadow:0 2px 8px rgba(13,13,13,0.06);">
  <div style="display:flex;align-items:center;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #f0ece5;">
    <span style="font-family:'Cormorant Garamond',serif;font-size:1.25rem;font-weight:600;color:#0d0d0d;">Biomarker Panel</span>
    <span style="margin-left:auto;font-size:0.65rem;font-weight:600;letter-spacing:0.1em;color:#8a837a;">✓ OPTIMAL &nbsp;·&nbsp; ✗ OUT OF RANGE</span>
  </div>
  {rows_html}
</div>
</body></html>"""
    components.html(bm_panel, height=430, scrolling=False)

# ── RADAR CHART ─────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

radar_categories = ["CRP", "Glucose", "HDL", "HbA1c", "Albumin", "RDW", "Telomere"]

# Normalize to 0–1 scale (inverted for "lower better" markers)
def norm(v, vmin, vmax, invert=False):
    n = (v - vmin) / (vmax - vmin)
    return (1 - n) if invert else n

radar_vals = [
    norm(crp,      0,   10,  invert=True),
    norm(glucose,  50,  200, invert=True),
    norm(hdl,      10,  100, invert=False),
    norm(hba1c,    3,   10,  invert=True),
    norm(albumin,  1,   6,   invert=False),
    norm(rdw,      9,   22,  invert=True),
    norm(telomere, 0.2, 2.0, invert=False),
]
radar_vals_pct = [v * 100 for v in radar_vals]
radar_vals_pct.append(radar_vals_pct[0])  # close polygon
cats_closed = radar_categories + [radar_categories[0]]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_vals_pct,
    theta=cats_closed,
    fill="toself",
    fillcolor="rgba(14, 123, 107, 0.10)",
    line=dict(color="#0e7b6b", width=2.5),
    marker=dict(color="#0e7b6b", size=7),
    name="Subject",
))
# Optimal reference
fig_radar.add_trace(go.Scatterpolar(
    r=[90] * 8,
    theta=cats_closed,
    fill="toself",
    fillcolor="rgba(201, 151, 58, 0.05)",
    line=dict(color="#c9973a", width=1.5, dash="dot"),
    marker=dict(size=0),
    name="Optimal",
))

fig_radar.update_layout(
    polar=dict(
        bgcolor="white",
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickvals=[25, 50, 75, 100],
            tickfont=dict(size=9, color="#8a837a", family="JetBrains Mono"),
            gridcolor="#f0ece5",
            linecolor="#e8e2d9",
        ),
        angularaxis=dict(
            tickfont=dict(size=12, color="#3d3935", family="DM Sans"),
            linecolor="#e8e2d9",
            gridcolor="#f0ece5",
        ),
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    height=380,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.04,
        xanchor="center", x=0.5,
        font=dict(family="DM Sans", size=11, color="#3d3935"),
    ),
    showlegend=True,
)

col_r1, col_r2, col_r3 = st.columns([0.5, 2, 0.5])
with col_r2:
    st.markdown("""
    <div class="chart-section">
        <div class="chart-title">Biomarker Radar</div>
        <div class="chart-sub">Normalized health score — higher is better for each axis</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────
st.markdown(f"""
<div style="
    margin-top:3rem;
    padding:1.5rem 2rem;
    background:white;
    border-radius:var(--radius);
    border:1px solid var(--border-soft);
    display:flex;
    align-items:center;
    gap:1.5rem;
    box-shadow:var(--shadow-sm);
">
    <div style="font-family:'Cormorant Garamond',serif; font-size:1.4rem; font-weight:600; color:var(--ink);">
        Immortigen
        <span style="font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:500; color:var(--ink-faint); margin-left:8px; letter-spacing:0.08em;">
            LONGEVITY SCIENCE PLATFORM
        </span>
    </div>
    <div style="margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--ink-faint);">
        {sample_id} &nbsp;·&nbsp; Age Acceleration Index: <span style="color:{aai_color}; font-weight:600;">{aai:+.1f} yr</span>
    </div>
</div>
""", unsafe_allow_html=True)