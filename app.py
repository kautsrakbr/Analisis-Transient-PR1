from __future__ import annotations

import secrets

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import expi

from pressure_model import (
    ReservoirInputs,
    laplace_wellbore_pressure,
    simulate_drawdown,
    stehfest_weights,
)


st.set_page_config(
    page_title="RadialPTA | Pressure Transient Analysis",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#111827; --muted:#5f6b7a; --blue:#176bce; --border:#d8dee8; --white:#ffffff; }
      html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family:"Segoe UI", Arial, sans-serif !important;
      }
      .stApp { background:#ffffff; color:var(--ink); }
      [data-testid="stHeader"] { height:3.35rem; background:rgba(255,255,255,.94); backdrop-filter:blur(10px); }
      [data-testid="stToolbar"] { visibility:visible !important; background:transparent !important; }
      [data-testid="stAppDeployButton"] { display:none !important; }
      [data-testid="stSidebarCollapsedControl"] { display:block !important; visibility:visible !important; }
      [data-testid="stSidebarCollapsedControl"] button {
        width:2.35rem; height:2.35rem; border-radius:10px !important; color:#176bce !important;
        background:#ffffff !important; border:1px solid #d8e1eb !important; box-shadow:0 5px 14px rgba(22,45,75,.1);
      }
      [data-testid="stMainBlockContainer"], .block-container {
        width:100% !important; max-width:none !important;
        padding:2.15rem clamp(1.25rem,3vw,3.5rem) 3rem !important;
      }
      [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#071a2d 0%,#0a2540 58%,#0b2d4d 100%) !important;
        border-right:1px solid #183b59; box-shadow:8px 0 28px rgba(6,24,41,.12);
        transition:min-width .22s ease,max-width .22s ease,transform .22s ease !important;
      }
      [data-testid="stSidebar"][aria-expanded="true"] { min-width:22rem !important; max-width:22rem !important; }
      [data-testid="stSidebar"][aria-expanded="false"] {
        min-width:0 !important; max-width:0 !important; border-right:0 !important; box-shadow:none !important;
      }
      [data-testid="stSidebar"] > div:first-child { background:transparent !important; }
      [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:.9rem; }
      [data-testid="stSidebar"] hr { margin:.9rem 0; border-color:rgba(180,210,235,.16); }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] h4,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color:#eaf3fb !important; line-height:1.35 !important; white-space:normal !important;
        overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere;
      }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color:#bdccda !important; font-size:.79rem !important; font-weight:600 !important;
        letter-spacing:.01em; margin-bottom:.28rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(8,29,48,.56); border:1px solid rgba(151,186,216,.17) !important;
        border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,.10);
      }
      [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div { padding:.9rem .9rem .65rem; }
      [data-testid="stSidebar"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] [data-baseweb="base-input"] {
        min-height:2.65rem; background:#061522 !important; border:1px solid #294762 !important;
        border-radius:10px !important; transition:border-color .18s ease,box-shadow .18s ease;
      }
      [data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
      [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
      [data-testid="stSidebar"] [data-baseweb="base-input"]:focus-within {
        border-color:#55a5f4 !important; box-shadow:0 0 0 3px rgba(62,145,231,.16) !important;
      }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] [data-baseweb="select"] *,
      [data-testid="stSidebar"] [role="button"] {
        color:#f6f9fc !important; -webkit-text-fill-color:#f6f9fc !important;
        caret-color:#ffffff !important; opacity:1 !important;
      }
      [data-testid="stSidebar"] input {
        font-size:.9rem !important; font-weight:650 !important; line-height:1.25 !important; min-width:0 !important;
      }
      [data-testid="stSidebar"] .stButton > button {
        min-height:2.8rem; background:linear-gradient(135deg,#267ed8,#176bce) !important;
        color:#ffffff !important; border:1px solid #4d9bea !important; border-radius:11px !important;
        font-weight:700; box-shadow:0 7px 18px rgba(10,92,176,.25); transition:all .18s ease;
      }
      [data-testid="stSidebar"] .stButton > button:hover {
        background:linear-gradient(135deg,#3390eb,#1c76dc) !important;
        border-color:#7bbcff !important; transform:translateY(-1px); box-shadow:0 9px 22px rgba(10,92,176,.34);
      }
      [data-testid="stSidebar"] .stNumberInput button {
        background:#0b2236 !important; color:#cfe3f5 !important; border-color:#294762 !important;
      }
      [data-testid="stSidebar"] svg { fill:currentColor !important; color:#dceaf6 !important; }
      [data-testid="stSidebar"] .stNumberInput,
      [data-testid="stSidebar"] .stSelectbox,
      [data-testid="stSidebar"] .stSlider { margin-bottom:.15rem; }
      [data-testid="stSidebar"] [data-testid="stAlert"] {
        background:rgba(27,74,108,.5) !important; border:1px solid rgba(131,184,222,.3) !important;
        border-radius:12px; color:#e8f4fd !important;
      }
      [data-testid="stSidebarCollapseButton"] button { background:#ffffff !important; border:1px solid #d8dee8 !important; }
      .sidebar-brand { display:flex; gap:.8rem; align-items:center; padding:.35rem .15rem .55rem; }
      .sidebar-logo { width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
        color:#fff; font-size:1.25rem; background:linear-gradient(145deg,#3f9df2,#176bce);
        box-shadow:0 8px 20px rgba(14,108,202,.35),inset 0 1px 0 rgba(255,255,255,.22); }
      .sidebar-brand-name { color:#fff; font-weight:760; font-size:1.08rem; letter-spacing:-.02em; }
      .sidebar-brand-sub { color:#8eabc2; font-size:.72rem; margin-top:.08rem; }
      .sidebar-badges { display:flex; gap:.4rem; margin:.15rem 0 .9rem; }
      .sidebar-badge { padding:.27rem .52rem; border-radius:99px; color:#b9d9f4; background:rgba(41,126,202,.14);
        border:1px solid rgba(78,154,219,.28); font-size:.65rem; font-weight:700; letter-spacing:.04em; }
      .sidebar-section { display:flex; align-items:center; gap:.55rem; margin:0 0 .7rem; }
      .sidebar-section-icon { width:28px; height:28px; display:grid; place-items:center; border-radius:8px;
        background:rgba(48,137,219,.16); color:#79baf5; font-size:.88rem; }
      .sidebar-section-title { color:#f5f9fd; font-size:.82rem; font-weight:750; letter-spacing:.045em; text-transform:uppercase; }
      .sidebar-section-sub { color:#7895ad; font-size:.66rem; margin-top:.04rem; }
      .sidebar-assumption { display:flex; align-items:center; gap:.7rem; margin-top:.15rem; padding:.72rem .8rem;
        border-radius:12px; color:#dbeaf6; background:rgba(31,86,126,.48); border:1px solid rgba(111,174,220,.28); }
      .sidebar-assumption-icon { flex:0 0 29px; width:29px; height:29px; display:grid; place-items:center;
        border-radius:8px; color:#90caf8; background:rgba(40,139,220,.18); font-size:.85rem; }
      .sidebar-assumption strong { display:block; color:#f2f8fc; font-size:.73rem; line-height:1.2; }
      .sidebar-assumption span { display:block; color:#9fbbd1; font-size:.68rem; line-height:1.25; margin-top:.14rem; white-space:nowrap; }
      .sidebar-footnote { color:#7895ad; font-size:.64rem; letter-spacing:.045em; text-align:center; padding:.35rem 0 .05rem; }
      [data-testid="stAlert"] { background:#ffffff; color:#111827; border:1px solid var(--border); }
      .hero { padding:.9rem 1.05rem; border-radius:14px; color:#111827; margin:.1rem 0 .6rem;
        background:#ffffff; border:1px solid #dce4ed; box-shadow:0 4px 14px rgba(15,44,75,.035); }
      .hero-grid { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:center; gap:1.25rem; }
      .eyebrow { color:#176bce; font:750 .62rem "Segoe UI",Arial,sans-serif; letter-spacing:.13em; text-transform:uppercase; }
      .hero h1 { color:#111827; margin:.22rem 0 .18rem; font-size:clamp(1.25rem,1.6vw,1.55rem); line-height:1.12; letter-spacing:-.03em; }
      .hero p { color:#5f6f82; max-width:960px; margin:0; font-size:.74rem; line-height:1.45; }
      .hero-status { min-width:155px; padding:.25rem 0 .25rem .9rem; border-left:1px solid #dce4ed; }
      .hero-status-label { display:none; }
      .hero-status-value { display:flex; align-items:center; gap:.4rem; color:#17324d; font-size:.74rem; font-weight:750; margin:0 0 .12rem; }
      .hero-status-dot { width:7px; height:7px; border-radius:50%; background:#21a179; box-shadow:0 0 0 3px rgba(33,161,121,.11); }
      .hero-status-sub { color:#8291a2; font-size:.58rem; }
      .hero-assumptions { display:flex; flex-wrap:wrap; gap:.28rem; margin-top:.5rem; }
      .assumption { display:inline-block; font:700 .57rem "Segoe UI",Arial,sans-serif; padding:.22rem .42rem;
        border:1px solid #d2dce7; border-radius:99px; color:#405166; background:#fafcff; }
      .summary-heading { display:flex; align-items:center; justify-content:space-between; margin:.1rem 0 .3rem; padding:0 .1rem; }
      .summary-heading strong { color:#314258; font-size:.68rem; letter-spacing:.11em; text-transform:uppercase; }
      .summary-heading span { color:#718096; font-size:.65rem; }
      .metric-card { position:relative; overflow:hidden; height:94px; box-sizing:border-box; display:flex; flex-direction:column; padding:.68rem .85rem;
        border-radius:12px; background:#ffffff; border:1px solid var(--border); box-shadow:0 3px 11px rgba(22,45,75,.03); }
      .metric-card::before { content:""; position:absolute; inset:0 auto 0 0; width:3px; background:linear-gradient(#2f8be5,#176bce); }
      .metric-label { color:#667085; font-size:.59rem; font-weight:750; text-transform:uppercase; letter-spacing:.085em; }
      .metric-value { color:#111827; font:750 clamp(.88rem,1.05vw,1.12rem) "Segoe UI",Arial,sans-serif;
        line-height:1.14; margin:.28rem 0 .1rem; overflow-wrap:anywhere; }
      .metric-value--range { font-size:clamp(.78rem,.9vw,.96rem); white-space:nowrap; letter-spacing:-.02em; }
      .metric-note { color:#718096; font-size:.58rem; margin-top:auto; }
      .flow-card { border-radius:16px; background:#ffffff; border:1px solid var(--border); padding:1rem; text-align:center; min-height:112px; }
      .flow-index { color:#176bce; font:600 .7rem "Segoe UI",Arial,sans-serif; }
      .flow-title { color:#111827; font-weight:700; margin:.3rem 0; }
      .flow-desc { color:#5f6b7a; font-size:.75rem; }
      .guide-hero { padding:1.35rem 1.5rem; margin:.25rem 0 1.15rem; border-radius:18px; color:#ffffff;
        background:linear-gradient(135deg,#0a2945 0%,#0e4f86 64%,#176bce 100%); box-shadow:0 12px 30px rgba(13,68,113,.16); }
      .guide-eyebrow { color:#9fd1ff; font-size:.68rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
      .guide-hero h2 { color:#ffffff; margin:.35rem 0 .3rem; font-size:clamp(1.35rem,2.3vw,1.85rem); letter-spacing:-.025em; }
      .guide-hero p { max-width:900px; margin:0; color:#d8eaf8; font-size:.84rem; }
      .guide-card { height:116px; box-sizing:border-box; padding:.9rem 1rem; border-radius:14px; background:#ffffff;
        border:1px solid #dce4ed; box-shadow:0 5px 16px rgba(20,48,77,.04); }
      .guide-card-index { color:#176bce; font-size:.66rem; font-weight:800; letter-spacing:.09em; }
      .guide-card-title { color:#172033; font-size:.84rem; font-weight:750; margin:.27rem 0 .18rem; }
      .guide-card-desc { color:#667085; font-size:.7rem; line-height:1.35; }
      .guide-note { padding:.8rem 1rem; border-left:4px solid #176bce; border-radius:0 12px 12px 0;
        color:#344054; background:#f4f8fd; font-size:.8rem; line-height:1.5; }
      .guide-note strong { color:#111827; }
      .backend-flow { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:.55rem 0 1rem; }
      .backend-card { position:relative; min-height:104px; padding:.8rem .85rem; border-radius:13px; color:#dcecf8;
        background:linear-gradient(145deg,#09233b,#0e4776); border:1px solid #1d5d8f; box-shadow:0 7px 18px rgba(8,47,80,.1); }
      .backend-card::after { content:""; position:absolute; right:-20px; top:-28px; width:72px; height:72px; border-radius:50%; background:rgba(66,158,236,.12); }
      .backend-index { color:#78bdf6; font-size:.6rem; font-weight:800; letter-spacing:.1em; }
      .backend-title { color:#ffffff; font-size:.78rem; font-weight:750; margin:.3rem 0 .2rem; }
      .backend-desc { color:#adc5d8; font-size:.66rem; line-height:1.4; }
      .source-note { padding:.75rem .9rem; border-radius:12px; background:linear-gradient(135deg,#eef7ff,#f7fbff);
        border:1px solid #cfe3f5; color:#3b5066; font-size:.72rem; line-height:1.5; }
      .source-note strong { color:#0d5da5; }
      .diagram-shell { margin:.45rem 0 .8rem; padding:1rem; border-radius:16px;
        background:linear-gradient(145deg,#061a2d 0%,#0a3457 58%,#0d5d99 100%); border:1px solid #1f6597;
        box-shadow:0 10px 26px rgba(6,35,59,.15); overflow:hidden; }
      .diagram-header { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin-bottom:.65rem; }
      .diagram-title { color:#ffffff; font-size:.86rem; font-weight:800; }
      .diagram-subtitle { color:#9fc4df; font-size:.66rem; line-height:1.4; margin-top:.12rem; }
      .diagram-badge { flex:0 0 auto; color:#cde9ff; background:rgba(70,158,226,.16); border:1px solid rgba(120,190,241,.28);
        border-radius:99px; padding:.25rem .5rem; font-size:.57rem; font-weight:800; letter-spacing:.08em; }
      .diagram-shell svg { display:block; width:100%; height:auto; min-height:220px; border-radius:12px; background:#f8fbff; }
      .diagram-shell svg text { font-family:"Segoe UI",Arial,sans-serif; }
      .section-kicker { color:#176bce; font:600 .73rem "Segoe UI",Arial,sans-serif; letter-spacing:.12em; text-transform:uppercase; margin-top:.8rem; }
      div[data-testid="stPlotlyChart"] { background:#ffffff; border:1px solid var(--border); border-radius:18px;
        overflow:hidden; box-shadow:0 9px 26px rgba(22,45,75,.055); transition:box-shadow .2s ease,border-color .2s ease; }
      div[data-testid="stPlotlyChart"]:hover { border-color:#c2d2e3; box-shadow:0 13px 32px rgba(22,45,75,.085); }
      div[data-testid="stPlotlyChart"] .modebar { top:10px !important; right:10px !important; padding:4px 6px !important;
        border:1px solid #dbe4ed; border-radius:9px; background:rgba(255,255,255,.94) !important; box-shadow:0 4px 12px rgba(20,45,72,.08); }
      div[data-testid="stPlotlyChart"] .modebar-btn path { fill:#536273 !important; }
      div[data-testid="stPlotlyChart"] .modebar-btn:hover path { fill:#176bce !important; }
      .stTabs [data-baseweb="tab-list"] { width:100%; min-height:4.35rem; box-sizing:border-box; display:flex; align-items:stretch;
        gap:.42rem; padding:.55rem; margin:.5rem 0 1.2rem; overflow:visible;
        background:linear-gradient(110deg,#071d33 0%,#0c416e 50%,#176bce 100%); border:1px solid #174d78;
        border-radius:16px; box-shadow:0 10px 28px rgba(7,39,68,.18); }
      .stTabs [data-baseweb="tab"] { flex:1 1 0; min-width:0; min-height:3.15rem; box-sizing:border-box;
        display:flex; align-items:center; justify-content:center; color:#d9e9f7; background:transparent; border-radius:11px;
        padding:.78rem 1rem; border:1px solid transparent; font-size:.8rem; font-weight:700; line-height:1.25;
        white-space:nowrap; overflow:visible; transition:all .18s ease; }
      .stTabs [data-baseweb="tab"]:hover { color:#ffffff; background:rgba(255,255,255,.09); }
      .stTabs [data-baseweb="tab"][aria-selected="true"] { color:#0d5da5; background:linear-gradient(135deg,#ffffff,#eef6ff);
        border-color:rgba(255,255,255,.72); box-shadow:0 5px 14px rgba(2,24,43,.2); }
      .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
      .chart-control { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin:.2rem 0 .35rem; }
      .chart-control-title { color:#111827; font-size:.94rem; font-weight:750; }
      .chart-control-sub { color:#667085; font-size:.72rem; margin-top:.12rem; }
      .stRadio [role="radiogroup"] { width:max-content; display:flex; gap:.3rem; padding:.28rem;
        background:#f2f5f9; border:1px solid #dce4ed; border-radius:12px; }
      .stRadio [role="radiogroup"] label { min-height:2.15rem; display:flex; align-items:center; margin:0 !important;
        padding:.38rem .78rem !important; background:transparent; border:1px solid transparent; border-radius:8px;
        cursor:pointer; transition:all .16s ease; }
      .stRadio [role="radiogroup"] label > div:first-child { display:none !important; }
      .stRadio [role="radiogroup"] label p { color:#536273 !important; font-size:.76rem !important; font-weight:700 !important; }
      .stRadio [role="radiogroup"] label:hover { background:#ffffff; border-color:#d8e1eb; }
      .stRadio [role="radiogroup"] label:has(input:checked) { background:#176bce; border-color:#176bce;
        box-shadow:0 4px 11px rgba(23,107,206,.2); }
      .stRadio [role="radiogroup"] label:has(input:checked) p { color:#ffffff !important; }
      .stDownloadButton button, .stButton button { background:#ffffff; color:#111827; border:1px solid #aeb8c6; border-radius:10px; font-weight:700; }
      .stMarkdown, .stCaption, [data-testid="stMetricValue"] { overflow:visible; }
      @media (max-width:1100px) {
        .hero-grid { grid-template-columns:1fr; }
        .hero-status { display:none; }
        .backend-flow { grid-template-columns:repeat(2,minmax(0,1fr)); }
      }
      @media (max-width:900px) {
        [data-testid="stMainBlockContainer"], .block-container { padding-top:1.25rem !important; }
        .metric-card { height:108px; padding:.75rem .8rem; }
        .metric-value--range { white-space:normal; }
        .stTabs [data-baseweb="tab-list"] { overflow-x:auto; }
        .stTabs [data-baseweb="tab"] { flex:0 0 auto; min-width:9rem; }
        .flow-card { min-height:100px; }
        .backend-flow { grid-template-columns:1fr; }
        .hero { padding:1rem; }
      }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "k": 75.0,
    "phi": 0.18,
    "ct": 1.5e-5,
    "mu": 1.20,
    "bo": 1.15,
    "h": 60.0,
    "rw": 0.328,
    "q": 450.0,
    "pi": 3500.0,
    "t_min": 0.001,
    "t_max": 240.0,
    "points": 160,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def generate_scenario() -> None:
    rng = np.random.default_rng(secrets.randbits(32))
    st.session_state.k = round(float(np.exp(rng.uniform(np.log(50), np.log(220)))), 1)
    st.session_state.phi = round(float(rng.uniform(0.13, 0.27)), 3)
    st.session_state.ct = float(f"{rng.uniform(0.8e-5, 2.8e-5):.2e}")
    st.session_state.mu = round(float(rng.uniform(0.7, 2.2)), 2)
    st.session_state.bo = round(float(rng.uniform(1.05, 1.32)), 2)
    st.session_state.h = round(float(rng.uniform(40, 110)), 1)
    st.session_state.rw = round(float(rng.uniform(0.25, 0.50)), 3)
    st.session_state.q = round(float(rng.uniform(250, 700)), 0)
    st.session_state.pi = round(float(rng.uniform(3000, 4800)), 0)


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo">◉</div>
          <div><div class="sidebar-brand-name">RadialPTA</div><div class="sidebar-brand-sub">Pressure transient workspace</div></div>
        </div>
        <div class="sidebar-badges"><span class="sidebar-badge">IARF</span><span class="sidebar-badge">FIELD UNITS</span><span class="sidebar-badge">LIVE MODEL</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.button("↻  Generate skenario realistis", use_container_width=True, on_click=generate_scenario)
    st.caption("Atur parameter di bawah — hasil diperbarui otomatis.")

    with st.container(border=True):
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-icon">◇</div><div><div class="sidebar-section-title">Reservoir</div><div class="sidebar-section-sub">Rock & formation properties</div></div></div>',
            unsafe_allow_html=True,
        )
        k = st.number_input("Permeabilitas, k (mD)", min_value=0.1, max_value=5000.0, step=5.0, key="k")
        phi = st.number_input("Porositas, ϕ (fraksi)", min_value=0.01, max_value=0.45, step=0.01, format="%.3f", key="phi")
        ct = st.number_input("Kompresibilitas total, cₜ (psi⁻¹)", min_value=1e-7, max_value=1e-3, step=1e-6, format="%.2e", key="ct")
        h = st.number_input("Ketebalan net, h (ft)", min_value=1.0, max_value=1000.0, step=5.0, key="h")

    with st.container(border=True):
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-icon">◌</div><div><div class="sidebar-section-title">Fluida & Sumur</div><div class="sidebar-section-sub">Fluid and well controls</div></div></div>',
            unsafe_allow_html=True,
        )
        mu = st.number_input("Viskositas, μ (cP)", min_value=0.05, max_value=100.0, step=0.1, key="mu")
        bo = st.number_input("Formation volume factor, B (rb/STB)", min_value=0.5, max_value=3.0, step=0.05, key="bo")
        rw = st.number_input("Radius sumur, rᵥ (ft)", min_value=0.05, max_value=2.0, step=0.01, format="%.3f", key="rw")
        q = st.number_input("Laju konstan, q (STB/hari)", min_value=1.0, max_value=20000.0, step=25.0, key="q")
        pi = st.number_input("Tekanan awal, pᵢ (psi)", min_value=100.0, max_value=20000.0, step=100.0, key="pi")

    with st.container(border=True):
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-icon">⌁</div><div><div class="sidebar-section-title">Waktu & Numerik</div><div class="sidebar-section-sub">Sampling and solver setup</div></div></div>',
            unsafe_allow_html=True,
        )
        t_min = st.number_input("Waktu minimum (jam)", min_value=1e-6, max_value=100.0, step=0.001, format="%.4f", key="t_min")
        t_max = st.number_input("Waktu maksimum (jam)", min_value=0.01, max_value=100000.0, step=24.0, key="t_max")
        points = st.slider("Jumlah titik", 60, 400, key="points", step=20)
        n_terms = st.selectbox("Suku Gaver–Stehfest", [8, 10, 12, 14], index=2)

    st.markdown(
        '<div class="sidebar-assumption"><div class="sidebar-assumption-icon">◆</div><div><strong>Asumsi model dikunci</strong><span>C = 0 &nbsp;·&nbsp; Cᴅ = 0 &nbsp;·&nbsp; S = 0</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-footnote">RADIALPTA · INFINITE-ACTING MODEL</div>', unsafe_allow_html=True)

if t_max <= t_min:
    st.error("Waktu maksimum harus lebih besar daripada waktu minimum.")
    st.stop()

inputs = ReservoirInputs(
    permeability_md=k,
    porosity=phi,
    total_compressibility_psi_inv=ct,
    viscosity_cp=mu,
    formation_volume_factor_rb_stb=bo,
    thickness_ft=h,
    wellbore_radius_ft=rw,
    rate_stb_day=q,
    initial_pressure_psi=pi,
)
time_hours = np.geomspace(t_min, t_max, points)
try:
    result = simulate_drawdown(time_hours, inputs, n_terms=n_terms)
except (ValueError, FloatingPointError) as exc:
    st.error(f"Perhitungan tidak dapat dijalankan: {exc}")
    st.stop()

df = pd.DataFrame(
    {
        "t (jam)": result["time_hours"],
        "tD": result["dimensionless_time"],
        "CD": result["dimensionless_storage"],
        "pD": result["dimensionless_pressure"],
        "Δp (psi)": result["pressure_drop_psi"],
        "pwf (psi)": result["flowing_pressure_psi"],
        "dΔp/dln(t) (psi)": result["pressure_derivative_psi"],
    }
)

max_drop = float(np.max(result["pressure_drop_psi"]))
final_pwf = float(result["flowing_pressure_psi"][-1])
metric_values = [
    ("tᴅ range", f"{result['dimensionless_time'][0]:.2e} – {result['dimensionless_time'][-1]:.2e}", "Dimensionless time"),
    ("pᴅ akhir", f"{result['dimensionless_pressure'][-1]:.4f}", "Hasil inversi Laplace"),
    ("Δp maksimum", f"{max_drop:,.1f} psi", "Pada akhir periode"),
    ("pwf akhir", f"{final_pwf:,.1f} psi", f"pᵢ = {pi:,.0f} psi"),
]


def render_dashboard_overview() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Pressure transient analysis · vertical well</div>
              <h1>Infinite-acting radial flow</h1>
              <p>Finite-radius well · homogeneous reservoir · single-phase · no storage · no skin · infinite boundary.</p>
            </div>
            <div class="hero-status">
              <div class="hero-status-value"><span class="hero-status-dot"></span>Solver aktif</div>
              <div class="hero-status-sub">Gaver–Stehfest · N = {n_terms}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="summary-heading"><strong>Simulation summary</strong><span>Live output · diperbarui otomatis</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, metric_values):
        value_class = "metric-value metric-value--range" if label == "tᴅ range" else "metric-value"
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div><div class="{value_class}">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )
    if final_pwf <= 0:
        st.warning("Kombinasi input menghasilkan pwf ≤ 0 psi. Model matematis tetap dihitung, tetapi skenario fisiknya perlu disesuaikan (turunkan q atau naikkan kh).")


PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "filename": "radial_pta_chart", "scale": 2},
}


def chart_layout(title: str, x_title: str, y_title: str, *, log_x: bool = False, log_y: bool = False) -> dict:
    axis_style = dict(
        showgrid=True,
        gridcolor="#e4eaf1",
        griddash="dot",
        gridwidth=1,
        linecolor="#93a4b8",
        linewidth=1,
        tickcolor="#93a4b8",
        ticklen=6,
        ticks="outside",
        tickfont=dict(color="#415166", size=11),
        title_font=dict(color="#27364a", size=13),
        zeroline=False,
        showline=True,
        automargin=True,
        showspikes=True,
        spikecolor="#7f93aa",
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
    )
    return dict(
        title=dict(text=f"<b>{title}</b>", x=0.035, y=0.965, font=dict(size=17, color="#111827")),
        xaxis=dict(title=x_title, type="log" if log_x else "linear", **axis_style),
        yaxis=dict(title=y_title, type="log" if log_y else "linear", **axis_style),
        template="plotly_white",
        height=430,
        margin=dict(l=72, r=34, t=80, b=66),
        hovermode="x unified",
        dragmode="pan",
        uirevision="radial-pta-view",
        legend=dict(
            orientation="h",
            y=1.14,
            x=0.02,
            font=dict(color="#27364a", size=11),
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#d8e2ec",
            borderwidth=1,
        ),
        font=dict(family="Segoe UI, Arial, sans-serif", color="#27364a", size=12),
        hoverlabel=dict(bgcolor="#102235", bordercolor="#102235", font_color="#ffffff", font_family="Segoe UI", font_size=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfdff",
    )


def pwf_chart(log_x: bool) -> go.Figure:
    scale_label = "Semi-log time" if log_x else "Cartesian time"
    pwf = np.asarray(result["flowing_pressure_psi"], dtype=float)
    initial_line = np.full_like(time_hours, pi, dtype=float)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=initial_line,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=pwf,
            mode="lines",
            name="Flowing pressure",
            line=dict(color="#176bce", width=3.5, shape="spline", smoothing=0.35),
            fill="tonexty",
            fillcolor="rgba(23,107,206,0.10)",
            hovertemplate="<b>t</b> %{x:.4g} jam<br><b>p<sub>wf</sub></b> %{y:,.2f} psi<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[time_hours[-1]],
            y=[pwf[-1]],
            mode="markers",
            name="Final pressure",
            marker=dict(size=9, color="#ffffff", line=dict(color="#176bce", width=3)),
            hovertemplate="<b>Final</b><br>t = %{x:.4g} jam<br>p<sub>wf</sub> = %{y:,.2f} psi<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=pi,
        line_dash="dot",
        line_color="#8ea1b7",
        line_width=1.8,
        annotation_text=f"pᵢ {pi:,.0f} psi",
        annotation_position="top right",
        annotation_font=dict(color="#596b80", size=10),
    )
    annotation_x = float(np.log10(time_hours[-1])) if log_x else float(time_hours[-1])
    fig.add_annotation(
        x=annotation_x,
        y=pwf[-1],
        text=f"p<sub>wf</sub> akhir<br><b>{pwf[-1]:,.1f} psi</b>",
        showarrow=True,
        arrowhead=2,
        arrowwidth=1,
        arrowcolor="#176bce",
        ax=-58,
        ay=36,
        bgcolor="rgba(255,255,255,.94)",
        bordercolor="#cbd9e7",
        borderpad=5,
        font=dict(color="#27364a", size=10),
    )

    pressure_span = max(float(pi - np.min(pwf)), 1.0)
    fig.update_layout(**chart_layout(scale_label, "Waktu, t (jam)", "p<sub>wf</sub> (psi)", log_x=log_x))
    fig.update_layout(height=405, showlegend=False, margin=dict(l=68, r=30, t=70, b=58))
    fig.update_yaxes(range=[float(np.min(pwf) - 0.12 * pressure_span), float(pi + 0.10 * pressure_span)], tickformat=",.0f")
    fig.update_xaxes(tickformat=".3~g" if log_x else ",.0f")
    return fig


def diagnostic_chart(view_mode: str) -> go.Figure:
    pressure_drop = np.asarray(result["pressure_drop_psi"], dtype=float)
    derivative = np.asarray(result["pressure_derivative_psi"], dtype=float)
    derivative_plot = np.where(derivative > 0, derivative, np.nan)
    show_pressure = view_mode in {"Pressure", "Keduanya"}
    show_derivative = view_mode in {"Derivative", "Keduanya"}
    fig = go.Figure()

    if show_pressure:
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=pressure_drop,
                mode="lines",
                name="Pressure drop, Δp",
                line=dict(color="#176bce", width=3.25, shape="spline", smoothing=0.25),
                hovertemplate="<b>Pressure drop</b><br>t = %{x:.4g} jam<br>Δp = %{y:,.3f} psi<extra></extra>",
            )
        )
    if show_derivative:
        plateau = float(result["radial_derivative_plateau_psi"])
        plateau_low = plateau * 0.925
        plateau_high = plateau * 1.075
        fig.add_trace(
            go.Scatter(
                x=[time_hours[0], time_hours[-1], time_hours[-1], time_hours[0], time_hours[0]],
                y=[plateau_low, plateau_low, plateau_high, plateau_high, plateau_low],
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor="rgba(22,139,130,0.10)",
                hoverinfo="skip",
                showlegend=False,
                name="IARF tolerance zone",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=derivative_plot,
                mode="lines",
                name="Pressure derivative",
                line=dict(color="#e0712f", width=3.25, shape="spline", smoothing=0.25),
                hovertemplate="<b>Pressure derivative</b><br>t = %{x:.4g} jam<br>dΔp/dln(t) = %{y:,.3f} psi<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[time_hours[0], time_hours[-1]],
                y=[plateau, plateau],
                mode="lines",
                name="IARF plateau",
                line=dict(color="#168b82", width=2, dash="dot"),
                hovertemplate="IARF plateau: %{y:.3f} psi<extra></extra>",
            )
        )

    titles = {
        "Pressure": "Pressure drop · log–log",
        "Derivative": "Pressure derivative · log–log",
        "Keduanya": "Pressure & derivative · log–log",
    }
    fig.update_layout(**chart_layout(titles[view_mode], "Waktu, t (jam)", "Pressure response (psi)", log_x=True, log_y=True))
    fig.update_layout(height=455, showlegend=view_mode == "Keduanya", margin=dict(l=72, r=34, t=82, b=62), hovermode="closest")
    fig.update_xaxes(exponentformat="power", showexponent="all")
    fig.update_yaxes(exponentformat="power", showexponent="all")
    return fig


def render_diagram(title: str, subtitle: str, badge: str, view_box: str, svg_content: str) -> None:
    """Render a responsive teaching diagram that is easy to redraw by hand."""
    # Every line must start at column 0 — Markdown treats indented lines as a
    # preformatted code block, which turns the SVG markup into visible text
    # instead of rendering it.
    clean_svg = "\n".join(line.strip() for line in svg_content.strip().splitlines())
    html = (
        '<div class="diagram-shell">'
        '<div class="diagram-header">'
        f'<div><div class="diagram-title">{title}</div><div class="diagram-subtitle">{subtitle}</div></div>'
        f'<div class="diagram-badge">{badge}</div>'
        "</div>"
        f'<svg viewBox="{view_box}" role="img" aria-label="{title}" xmlns="http://www.w3.org/2000/svg">'
        f"{clean_svg}"
        "</svg>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


tab_dashboard, tab_pr1, tab_pr2, tab_guide, tab_method, tab_data = st.tabs(
    ["Dashboard", "PR 1 · Workflow DST", "PR 2 · Diffusivity", "PR 3 · Solver PTA", "Metode numerik", "Data & export"]
)
with tab_dashboard:
    render_dashboard_overview()
    st.markdown('<div class="section-kicker">Pressure response</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    left.plotly_chart(pwf_chart(False), use_container_width=True, theme=None, config=PLOTLY_CONFIG)
    right.plotly_chart(pwf_chart(True), use_container_width=True, theme=None, config=PLOTLY_CONFIG)
    left.caption("Cartesian · detail perubahan tekanan terhadap waktu aktual.")
    right.caption("Semi-log · memperjelas respons early hingga late time.")

    st.markdown('<div class="section-kicker">Diagnostic plot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-control"><div><div class="chart-control-title">Kurva diagnostik</div><div class="chart-control-sub">Pilih kurva yang ingin ditampilkan pada plot log–log.</div></div></div>',
        unsafe_allow_html=True,
    )
    diagnostic_view = st.radio(
        "Tampilan kurva diagnostik",
        ["Pressure", "Derivative", "Keduanya"],
        index=2,
        horizontal=True,
        label_visibility="collapsed",
        key="diagnostic_view",
    )
    st.plotly_chart(
        diagnostic_chart(diagnostic_view),
        use_container_width=True,
        theme=None,
        config=PLOTLY_CONFIG,
    )
    st.caption("Derivative dihitung terhadap ln(t). Pada infinite-acting radial flow, kurva derivative mendekati plateau ½ × pressure scale.")

with tab_method:
    st.markdown("### Alur transformasi model")
    flow_cols = st.columns(4)
    flow_items = [
        ("01", "Real space", "Persamaan difusivitas radial pada reservoir homogen."),
        ("02", "Laplace space", "Transformasi waktu mengubah PDE menjadi ODE radial."),
        ("03", "Bessel solution", "K₀ dan K₁ memenuhi batas laju konstan di rᴅ = 1."),
        ("04", "Real time", "Gaver–Stehfest menginversi p̄ᴅ(u) untuk setiap tᴅ."),
    ]
    for col, (idx, title, desc) in zip(flow_cols, flow_items):
        col.markdown(f'<div class="flow-card"><div class="flow-index">STEP {idx}</div><div class="flow-title">{title}</div><div class="flow-desc">{desc}</div></div>', unsafe_allow_html=True)

    eq1, eq2 = st.columns(2)
    with eq1:
        st.markdown("#### Ruang nyata → Laplace")
        st.latex(r"\frac{1}{r_D}\frac{\partial}{\partial r_D}\left(r_D\frac{\partial p_D}{\partial r_D}\right)=\frac{\partial p_D}{\partial t_D}")
        st.latex(r"t_D=\frac{0.0002637\,k\,t}{\phi\,\mu\,c_t\,r_w^2}")
        st.markdown("Setelah transformasi Laplace, solusi radial umum tersusun dari modified Bessel functions.")
        st.latex(r"\bar p_D(u)=\frac{K_0(\sqrt{u})}{u^{3/2}K_1(\sqrt{u})}")
    with eq2:
        st.markdown("#### Laplace → ruang nyata")
        st.latex(r"p_D(t_D)\approx\frac{\ln 2}{t_D}\sum_{j=1}^{N}V_j\,\bar p_D\left(\frac{j\ln2}{t_D}\right)")
        st.latex(r"\Delta p=\frac{141.2\,qB\mu}{kh}\,p_D\qquad p_{wf}=p_i-\Delta p")
        st.markdown(f"Perhitungan aktif memakai **N = {n_terms}** suku. Rasio Bessel dievaluasi dengan fungsi terskala untuk menghindari numerical underflow.")
        st.latex(r"C=0\;\mathrm{bbl/psi},\qquad C_D=\frac{0.8936C}{\phi c_t h r_w^2}=0,\qquad S=0")

    st.markdown("### Cara kerja Gaver–Stehfest di aplikasi")
    st.markdown(
        """
        Untuk **setiap nilai waktu dimensionless** $t_D$, aplikasi tidak memakai solusi pendekatan langsung.
        Solver membuat sejumlah titik di ruang Laplace, mengevaluasi solusi Bessel pada setiap titik,
        lalu menjumlahkannya dengan bobot Gaver–Stehfest:

        1. Pilih jumlah suku genap **N** (default 12).
        2. Hitung titik Laplace $u_j=j\\ln(2)/t_D$ untuk $j=1,2,\\ldots,N$.
        3. Hitung $\\bar p_D(u_j)$ memakai rasio modified Bessel function $K_0/K_1$.
        4. Hitung bobot $V_j$ yang bertanda selang-seling menggunakan faktorial.
        5. Jumlahkan seluruh $V_j\\bar p_D(u_j)$ dan kalikan dengan $\\ln(2)/t_D$ untuk memperoleh $p_D(t_D)$.
        6. Konversikan $p_D$ menjadi $\\Delta p$ dan $p_{wf}$ dalam satuan lapangan.

        Bobot positif dan negatif dapat bernilai besar. Hal ini **normal** pada Gaver–Stehfest;
        hasil inversi diperoleh dari cancellation antar-suku, sehingga evaluasi numeriknya harus konsisten.
        """
    )
    st.latex(
        r"V_j=(-1)^{j+N/2}\sum_{m=\lceil j/2\rceil}^{\min(j,N/2)}"
        r"\frac{m^{N/2}(2m)!}{(N/2-m)!\,m!\,(m-1)!\,(j-m)!\,(2m-j)!}"
    )

    sample_index = len(time_hours) // 2
    sample_td = float(result["dimensionless_time"][sample_index])
    gs_index = np.arange(1, n_terms + 1, dtype=float)
    gs_weights = stehfest_weights(n_terms)
    gs_u = np.log(2.0) * gs_index / sample_td
    gs_pbar = laplace_wellbore_pressure(gs_u)
    gs_terms = gs_weights * gs_pbar
    sample_pd = np.log(2.0) / sample_td * np.sum(gs_terms)
    gs_table = pd.DataFrame(
        {
            "j": gs_index.astype(int),
            "Vj": gs_weights,
            "uj = j ln(2) / tD": gs_u,
            "p̄D(uj), Bessel": gs_pbar,
            "Vj × p̄D(uj)": gs_terms,
        }
    )
    with st.expander("Lihat bukti perhitungan numerik Gaver–Stehfest", expanded=True):
        check_cols = st.columns(3)
        check_cols[0].metric("Jumlah suku, N", n_terms)
        check_cols[1].metric("Contoh tD", f"{sample_td:.4e}")
        check_cols[2].metric("Hasil inversi pD", f"{sample_pd:.6f}")
        st.dataframe(
            gs_table.style.format(
                {"Vj": "{:.6e}", "uj = j ln(2) / tD": "{:.6e}", "p̄D(uj), Bessel": "{:.6e}", "Vj × p̄D(uj)": "{:.6e}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Tabel ini dihitung langsung oleh solver untuk satu tD di tengah rentang waktu, bukan angka contoh statis.")

    st.markdown("### Validasi terhadap solusi analitik line-source (fungsi Ei)")
    st.markdown(
        """
        Tabel di atas menunjukkan solver *bekerja seperti dirancang*, tetapi belum membuktikan hasilnya
        *benar secara fisik*. Untuk itu diperlukan pembanding independen yang tidak melalui inversi numerik
        sama sekali. Untuk $C_D=0$, $S=0$, reservoir infinite-acting, solusi tersebut tersedia dalam bentuk
        tertutup memakai exponential integral Ei — dikenal sebagai solusi *line-source*:
        """
    )
    st.latex(r"p_D(t_D)=-\tfrac12\,\mathrm{Ei}\!\left(-\frac{1}{4t_D}\right)")
    st.markdown(
        "Untuk $t_D$ besar, $\\mathrm{Ei}(-x)\\to \\ln x+\\gamma$ (γ = konstanta Euler–Mascheroni) sehingga "
        "bentuk di atas tereduksi menjadi pendekatan logaritmik yang sama persis dengan "
        "`late_time_pressure_approximation` pada kode:"
    )
    st.latex(r"p_D(t_D)\approx\tfrac12\big(\ln t_D+0.80907\big)")

    td_check = np.asarray(result["dimensionless_time"], dtype=float)
    pd_solver = np.asarray(result["dimensionless_pressure"], dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pd_exact = -0.5 * expi(-1.0 / (4.0 * td_check))
        rel_error_pct = 100.0 * np.abs(pd_solver - pd_exact) / pd_exact

    late_mask = td_check >= 25.0
    val_cols = st.columns(3)
    val_cols[0].metric("Titik dengan tD ≥ 25", f"{int(np.sum(late_mask))} / {len(td_check)}")
    if np.any(late_mask):
        val_cols[1].metric("Error relatif rata-rata (tD ≥ 25)", f"{np.mean(rel_error_pct[late_mask]):.2e} %")
    else:
        val_cols[1].metric("Error relatif rata-rata (tD ≥ 25)", "n/a — perbesar t_max")
    val_cols[2].metric("Error relatif @ tD terbesar", f"{rel_error_pct[-1]:.2e} %")

    validation_fig = go.Figure()
    validation_fig.add_trace(
        go.Scatter(x=td_check, y=pd_solver, mode="lines", name="Gaver–Stehfest (solver)", line=dict(color="#176bce", width=3))
    )
    validation_fig.add_trace(
        go.Scatter(
            x=td_check,
            y=pd_exact,
            mode="markers",
            name="Ei eksak (line-source)",
            marker=dict(color="#d24646", size=6, symbol="circle-open", line=dict(width=1.5)),
        )
    )
    validation_fig.add_vline(
        x=25.0, line_dash="dot", line_color="#8ea1b7",
        annotation_text="tD = 25", annotation_font=dict(size=10, color="#596b80"),
    )
    validation_fig.update_layout(**chart_layout("Solver vs solusi Ei eksak", "tD", "pD", log_x=True))
    validation_fig.update_layout(height=380)
    st.plotly_chart(validation_fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG)

    st.markdown(
        """
        **Cara membaca hasil validasi:**
        - Pada $t_D$ kecil, solusi line-source (Ei) dan solusi finite-radius wellbore yang dipakai solver
          **memang berbeda** — ini bukan tanda solver salah. Line-source mengasumsikan seluruh laju masuk
          dari satu titik ber-radius nol, sedangkan solver menyelesaikan kondisi batas laju konstan pada
          $r_D=1$ yang sebenarnya. Kedua solusi konvergen setelah $t_D\\gtrsim25$, ambang klasik yang
          dipakai well testing untuk menyatakan "infinite-acting radial flow line-source valid".
        - Jika error relatif pada rentang $t_D\\ge25$ jauh melebihi orde $10^{-6}$–$10^{-4}$ %, barulah itu
          mengindikasikan masalah numerik murni (N terlalu kecil, cancellation berlebihan, presisi
          floating point) — bukan masalah model fisik.
        - Uji ini independen dari cara solver bekerja secara internal sehingga cocok dipakai sebagai
          *regression test*: setiap kali logika Gaver–Stehfest diubah, bandingkan lagi terhadap Ei eksak ini.
        """
    )

    st.info("Model ini khusus drawdown laju konstan, fluida satu fasa, reservoir homogen-isotropik, sumur vertikal fully penetrating, dan batas luar infinite. Efek storage, skin, multiphase, boundary, serta heterogenitas tidak dimodelkan.")

with tab_pr1:
    st.markdown(
        """
        <div class="guide-hero">
          <div class="guide-eyebrow">PR 1 · Workflow Pengambilan Data DST</div>
          <h2>Dari pengeboran sampai Drill Stem Test</h2>
          <p>Tujuan workflow ini adalah mengurangi ketidakpastian secara bertahap: posisi reservoir, kualitas batuan, jenis fluida, tekanan formasi, kemampuan alir, sampai batas reservoir. Setiap tahap menghasilkan data yang menjadi dasar keputusan tahap berikutnya.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Urutan kelas—drilling → logging → RFT & coring → casing/cementing → perforation → well test/DST—paling cocok menggambarkan **cased-hole test**. Di lapangan juga ada open-hole DST yang dilakukan sebelum casing dan perforasi. Tuliskan urutan dosen sebagai alur utama, lalu beri catatan variasi ini agar pemahamannya lengkap."
    )
    st.markdown(
        '<div class="source-note"><strong>Sumber pengayaan:</strong> Amanat U. Chaudhry, <em>Oil Well Testing Handbook</em>—khususnya Chapter 1 tentang data acquisition/management dan reservoir characterization, serta Chapter 12 tentang DST equipment, operational procedure, pressure chart, quality control, dan analysis limitations. Materi handbook dipadukan dengan urutan perkuliahan agar alurnya mudah dipahami pemula.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Mulai dari konsep paling dasar: apa itu well testing?")
    st.markdown(
        """
        **Well testing adalah eksperimen terkontrol pada sumur.** Operator mengubah kondisi alir—misalnya membuka sumur agar berproduksi atau menutup sumur agar pressure pulih—kemudian merekam bagaimana **rate, pressure, temperature, dan fluida** berubah terhadap waktu. Reservoir tidak dapat dilihat langsung, sehingga respons pressure dipakai seperti “sinyal” untuk menyimpulkan sifat batuan dan batas reservoir.

        Empat pertanyaan dasarnya adalah:
        1. **Apa yang diberikan ke reservoir?** Perubahan rate atau status valve sebagai input.
        2. **Apa yang diukur?** Bottom-hole pressure, temperature, rate, waktu event, dan sample fluida.
        3. **Apa yang dicari?** Initial pressure, permeability/kh, skin, productivity, flow regime, fracture, heterogeneity, serta boundary.
        4. **Mengapa perlu data tahap sebelumnya?** Karena pressure transient tidak dapat ditafsirkan dengan benar tanpa thickness, viscosity, compressibility, porosity, completion interval, datum, dan riwayat rate.

        Handbook menekankan bahwa data well test dikumpulkan sepanjang umur sumur—dari exploration sampai abandonment—dan harus dikelola oleh tim terintegrasi. Artinya, DST bukan kegiatan yang berdiri sendiri; ia adalah bagian dari rantai data geologi, petrofisika, PVT, completion, production, dan reservoir engineering.
        """
    )

    beginner_cols = st.columns(4)
    beginner_items = [
        ("INPUT", "Rate & valve event", "Kapan sumur dibuka, ditutup, dan berapa laju aktualnya."),
        ("RESPONSE", "Pressure vs time", "Gauge merekam drawdown dan recovery pressure."),
        ("MODEL", "Rock–fluid system", "k, ϕ, μ, cₜ, h, geometry, dan boundary."),
        ("OUTPUT", "Reservoir diagnosis", "kh, skin, pressure, regime aliran, dan batas."),
    ]
    for col, (idx, title, desc) in zip(beginner_cols, beginner_items):
        col.markdown(
            f'<div class="guide-card"><div class="guide-card-index">{idx}</div><div class="guide-card-title">{title}</div><div class="guide-card-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker">Peta workflow dan keputusan</div>', unsafe_allow_html=True)
    workflow_items = [
        ("01", "Pengeboran", "Membuka lubang, mengenali formation tops, dan menjaga well control."),
        ("02", "Logging", "Menentukan lithology, porosity, saturation, dan kandidat interval."),
        ("03", "RFT & Coring", "Mengukur pressure/gradient/fluid contact dan mengambil batuan nyata."),
        ("04", "Casing & Cement", "Mengisolasi formasi dan memastikan integritas sumur."),
        ("05", "Perforation", "Membuat koneksi terkontrol antara reservoir dan wellbore."),
        ("06", "DST / Well Test", "Mengukur deliverability dan respons tekanan dinamis reservoir."),
    ]
    for row_start in (0, 3):
        stage_cols = st.columns(3)
        for col, (idx, title, desc) in zip(stage_cols, workflow_items[row_start : row_start + 3]):
            col.markdown(
                f'<div class="guide-card"><div class="guide-card-index">STAGE {idx}</div><div class="guide-card-title">{title}</div><div class="guide-card-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("1 · Pengeboran: membuat akses sekaligus mengumpulkan data pertama", expanded=True):
        st.markdown(
            """
            **Tujuan utama** pengeboran bukan hanya mencapai target depth, tetapi membuat lubang yang aman, stabil, dapat dievaluasi, dan nantinya dapat diselesaikan sebagai sumur produksi/injeksi.

            **Data dan observasi selama drilling:**
            - **Rate of penetration (ROP):** perubahan ROP dapat menandai perubahan kekerasan batuan atau formation top, tetapi harus dikoreksi terhadap WOB, RPM, bit type, dan hydraulic condition.
            - **Mud logging:** gas total, komposisi gas, cutting description, fluorescence/cut, calcimetry, dan indikasi hydrocarbon show.
            - **Cuttings:** memberi informasi lithology secara kontinu, tetapi kedalamannya memiliki lag time dan dapat tercampur/cavings.
            - **MWD/LWD:** inclination, azimuth, gamma ray, resistivity, density/neutron, sonic, atau image saat drilling. Keunggulannya adalah data diperoleh sebelum kondisi lubang memburuk.
            - **Mud losses, kick, connection gas, pit gain/loss:** penting untuk pore-pressure prediction dan well control.
            - **Drilling parameters dan mud properties:** diperlukan untuk memahami kualitas log, invasion, formation damage, dan risiko differential sticking.

            **Keputusan yang dihasilkan:** konfirmasi formation top, penentuan casing point, kebutuhan coring, interval logging, mud weight window, dan apakah target layak diteruskan untuk evaluasi lebih lanjut.

            **Quality control:** semua depth harus direkonsiliasi terhadap driller depth, logger depth, dan TVD/TVDSS. Kesalahan depth matching akan terbawa sampai perforasi dan interpretasi pressure.
            """
        )

    with st.expander("2 · Logging: memetakan batuan dan fluida secara kontinu"):
        st.markdown(
            """
            Logging mengubah respons fisika formasi menjadi estimasi sifat reservoir. Tidak ada satu log yang menjawab semuanya; interpretasi harus menggunakan kombinasi log dan dikalibrasi dengan core/RFT.

            **Log umum dan fungsinya:**
            - **Gamma ray:** membedakan clean formation dan shale secara kualitatif; digunakan untuk korelasi stratigrafi dan estimasi shale volume.
            - **Resistivity:** sensitif terhadap fluida pori; bersama porosity dan model Archie/shaly-sand digunakan untuk estimasi water saturation.
            - **Density–neutron:** estimasi porosity dan indikasi lithology/gas melalui overlay/crossover.
            - **Sonic:** slowness, porosity tertentu, geomekanika, synthetic seismogram, dan evaluasi cement melalui tool khusus.
            - **Caliper:** diameter lubang; washout memengaruhi kualitas density, neutron, pad tools, dan keputusan cement volume.
            - **Image log:** fracture, bedding dip, breakout, dan orientasi stress lokal.
            - **CBL/VDL setelah cementing:** mengevaluasi kualitas ikatan casing–cement–formation, bukan log reservoir primer.

            **Workflow interpretasi minimum:** depth match → environmental correction → lithology/shale volume → porosity → fluid saturation → net reservoir/net pay → pilih pressure stations, core points, dan perforation interval.

            **Keterbatasan:** log mengukur volume batuan di sekitar wellbore dan dipengaruhi invasion, borehole condition, salinity, mineralogy, temperature, serta tool resolution. Karena itu hasil log tidak boleh diperlakukan sebagai “nilai benar tunggal”.
            """
        )

    with st.expander("3A · RFT/Formation Tester: pressure, mobility, gradient, FWL, dan fluid sample"):
        st.markdown(
            """
            RFT adalah nama historis; tool modern sering disebut MDT/formation tester. Probe ditempelkan ke dinding lubang, volume kecil fluida ditarik (**pretest**), lalu pressure dibiarkan build up menuju formation pressure.

            **Output utama tiap station:**
            - formation pressure yang telah dikoreksi ke kondisi stabil;
            - pretest drawdown dan buildup untuk indikasi mobility/kh lokal;
            - kualitas seal probe dan kemungkinan supercharge;
            - fluid sample bila tool memiliki pumpout dan sample chamber.

            **Supercharge** terjadi ketika mud filtrate/invasion menaikkan pressure dekat wellbore di atas pressure formasi asli. Station yang seal-nya buruk, buildup belum stabil, atau terpengaruh supercharge harus diberi flag dan tidak langsung dipakai untuk gradient regression.
            """
        )
        render_diagram(
            "Mekanisme probe RFT/MDT dan jejak pressure per station",
            "Panel kiri menunjukkan alat menembus mudcake; panel kanan menunjukkan drawdown-buildup dan efek supercharge.",
            "RFT / MDT",
            "0 0 1200 620",
            """
            <rect x="30" y="35" width="390" height="540" rx="18" fill="#eef6fd" stroke="#bdd7eb" stroke-width="2"/>
            <text x="55" y="72" font-size="19" font-weight="800" fill="#164f7b">MEKANISME PROBE</text>
            <rect x="55" y="100" width="110" height="430" fill="#dbe6f0" stroke="#93a8bd" stroke-width="2"/>
            <text x="110" y="180" text-anchor="middle" font-size="13" font-weight="700" fill="#3d5568">MUD</text>
            <text x="110" y="198" text-anchor="middle" font-size="12" fill="#3d5568">(hidrostatik)</text>
            <rect x="165" y="100" width="14" height="430" fill="#9c7b4f" stroke="#6b4f2c" stroke-width="2"/>
            <rect x="179" y="100" width="206" height="430" fill="#e7c98f" stroke="#b8935a" stroke-width="2"/>
            <text x="290" y="140" text-anchor="middle" font-size="15" font-weight="800" fill="#7a5a25">FORMASI</text>
            <path d="M140 285 H182 V335 H140 Z" fill="#8b98a8" stroke="#4a5866" stroke-width="3"/>
            <circle cx="105" cy="310" r="20" fill="#c9d3dc" stroke="#4a5866" stroke-width="3"/>
            <line x1="85" y1="310" x2="60" y2="310" stroke="#4a5866" stroke-width="4" marker-end="url(#pretestArrow)"/>
            <defs><marker id="pretestArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M9,0 L9,6 L0,3 z" fill="#4a5866"/></marker></defs>
            <text x="55" y="405" font-size="12" font-weight="700" fill="#31485e">PRETEST PISTON</text>
            <text x="55" y="421" font-size="12" fill="#526b80">tarik volume kecil</text>
            <line x1="184" y1="310" x2="230" y2="230" stroke="#31485e" stroke-width="1.5"/>
            <text x="200" y="220" font-size="12" font-weight="700" fill="#31485e">PROBE SEAL</text>
            <text x="55" y="549" font-size="12" fill="#8a3f14">Seal buruk / invasi filtrate dalam</text>
            <text x="55" y="565" font-size="12" fill="#8a3f14">→ pressure terekam bisa supercharged.</text>

            <rect x="455" y="35" width="710" height="540" rx="18" fill="#ffffff" stroke="#c8dae8" stroke-width="2"/>
            <text x="480" y="72" font-size="19" font-weight="800" fill="#164f7b">PRESSURE TRACE SATU STATION</text>
            <line x1="500" y1="480" x2="1130" y2="480" stroke="#405b72" stroke-width="3"/>
            <line x1="500" y1="480" x2="500" y2="100" stroke="#405b72" stroke-width="3"/>
            <text x="1080" y="512" font-size="14" font-weight="700" fill="#405b72">waktu →</text>
            <text x="468" y="105" font-size="14" font-weight="700" fill="#405b72">P ↑</text>
            <line x1="500" y1="280" x2="1130" y2="280" stroke="#19795f" stroke-width="2" stroke-dasharray="7 6"/>
            <text x="940" y="272" font-size="13" font-weight="700" fill="#19795f">pressure formasi (true), stabil</text>
            <line x1="700" y1="240" x2="1130" y2="240" stroke="#c9722e" stroke-width="2" stroke-dasharray="7 6"/>
            <text x="940" y="232" font-size="13" font-weight="700" fill="#c9722e">stabilisasi jika supercharge</text>
            <path d="M560 150 C580 220 605 340 620 380 C650 430 700 320 760 300 C850 290 950 282 1100 280" fill="none" stroke="#176bce" stroke-width="4"/>
            <path d="M560 150 C580 220 605 340 620 380 C650 430 700 300 760 270 C850 250 950 242 1100 240" fill="none" stroke="#c9722e" stroke-width="3" stroke-dasharray="9 6"/>
            <circle cx="560" cy="150" r="6" fill="#31485e"/>
            <text x="540" y="130" text-anchor="end" font-size="12" font-weight="700" fill="#31485e">seal terbentuk</text>
            <circle cx="620" cy="380" r="6" fill="#b3541e"/>
            <text x="630" y="415" font-size="12" font-weight="700" fill="#b3541e">drawdown minimum</text>
            <text x="760" y="345" font-size="12" font-weight="700" fill="#176bce">buildup menuju stabil</text>
            """,
        )
        st.markdown(
            """
            Untuk setiap kelompok fluida, pressure diplot terhadap **TVDSS**, bukan sekadar measured depth. Regresi linear ditulis sebagai:
            """
        )
        st.latex(r"p=a+Gz")
        st.markdown(
            r"""
            dengan $G=dp/dz$ sebagai pressure gradient. Secara pendekatan field unit, $G\approx0.433\,SG$ psi/ft sehingga gradient memberi indikasi density/jenis fluida. Gas memiliki gradient paling kecil, oil lebih besar, dan water/brine biasanya paling besar.

            Jika garis oil dan water masing-masing $p_o=a_o+G_oz$ dan $p_w=a_w+G_wz$, kedalaman perpotongannya adalah estimasi **free-water level (FWL)**:
            """
        )
        st.latex(r"z_{FWL}=\frac{a_o-a_w}{G_w-G_o}")
        st.markdown(
            """
            **FWL tidak selalu sama dengan OWC log.** FWL adalah level kesetimbangan tekanan kapiler nol. OWC yang terlihat dari log dapat berada di atas FWL karena transition zone dan capillary pressure.

            **Pencocokan dengan PVT:** pressure sample dan gradient harus dibandingkan pada datum, temperature, dan kondisi fluida yang konsisten. PVT digunakan untuk mengecek density/gradient, bubble point/dew point, compressibility, viscosity, formation volume factor, GOR, serta apakah sample terkontaminasi mud filtrate. RFT pressure tidak “dicocokkan ke satu angka PVT”; keduanya digabung untuk memastikan fluid column dan sample mewakili reservoir.

            **QC gradient:** gunakan beberapa station per fluid column, tampilkan residual regresi, cek depth uncertainty, hindari station tight/supercharged, dan jangan menyimpulkan contact dari hanya dua titik yang buruk.
            """
        )

    with st.expander("3B · Coring: whole core, sidewall core, RCA, dan SCAL"):
        core_comparison = pd.DataFrame(
            [
                ("Whole core", "Core barrel saat drilling; interval relatif kontinu", "Volume besar, orientasi/struktur lebih baik, cocok RCA dan SCAL", "Mahal, menambah rig time, recovery dapat kurang dari 100%"),
                ("Percussion sidewall", "Peluru mengambil plug kecil setelah logging", "Cepat dan banyak titik", "Sample kecil, dapat remuk/terkontaminasi, kurang ideal untuk permeability/SCAL"),
                ("Rotary sidewall", "Mini rotary bit mengambil plug silindris", "Kualitas lebih baik daripada percussion dan depth selective", "Tetap lebih kecil dan kurang kontinu daripada whole core"),
            ],
            columns=["Metode", "Cara pengambilan", "Kelebihan", "Keterbatasan"],
        )
        st.dataframe(core_comparison, use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Routine Core Analysis (RCA):** porosity, grain density, horizontal/vertical permeability, fluid saturation, dan deskripsi lithology. Nilai plug harus dikoreksi/diinterpretasi terhadap confining stress dan representativeness.

            **Special Core Analysis (SCAL):** capillary pressure, relative permeability, wettability, electrical properties, formation factor, resistivity index, dan compressibility batuan. SCAL dipakai untuk saturation-height function, dynamic model, dan recovery prediction.

            **Integrasi log–core:** core memberi pengukuran langsung skala kecil, sedangkan log memberi profil kontinu tetapi indirect. Core digunakan untuk kalibrasi porosity/permeability transform dan facies; log digunakan untuk memperluas informasi core ke seluruh interval.

            **Handling penting:** tandai top-bottom dan orientasi, jaga preservasi wettability/fluid, minimalkan evaporation, dokumentasikan recovery, lalu lakukan depth shift antara core depth dan log depth.
            """
        )

    with st.expander("4 · Casing, cementing, dan Leak-Off Test (LOT)"):
        st.markdown(
            """
            **Casing** memberi mechanical support, mengisolasi pressure regime, melindungi freshwater zone, dan menyediakan pressure containment. Desain mempertimbangkan burst, collapse, tension, corrosion, connection rating, kick tolerance, dan future completion loads.

            **Cementing workflow:** run casing dan centralizer → circulate/condition mud → pump spacer → pump lead/tail cement slurry → displace cement → bump plug → wait on cement (WOC) → pressure test dan evaluasi cement bila diperlukan. Cement yang baik mencegah channeling dan crossflow antar-zona.

            **LOT dilakukan setelah casing shoe disemen, cement cukup kuat, dan shoe track/drill-out telah disiapkan.** Lubang di bawah shoe diberi tekanan bertahap sambil mencatat pressure versus pumped volume. Perubahan slope menandai awal leak-off/fracture initiation.

            Hasil LOT dipakai untuk:
            - memperkirakan fracture gradient/equivalent mud weight di shoe;
            - menentukan batas maksimum mud weight dan allowable annular pressure;
            - memperbarui kick tolerance dan casing design berikutnya.

            **FIT berbeda dari LOT:** Formation Integrity Test berhenti pada target pressure untuk membuktikan integritas tanpa sengaja memecah formasi. LOT diteruskan sampai indikasi leak-off. Nilai harus dikoreksi terhadap hydrostatic pressure dan dilaporkan bersama datum/depth.

            **QC:** pastikan trapped air diminimalkan, pump rate stabil, gauge terkalibrasi, cement telah mencapai strength, dan bedakan leak di surface equipment dari leak-off formasi.
            """
        )

    with st.expander("5 · Perforation: membuat koneksi reservoir–wellbore"):
        st.markdown(
            """
            Setelah reservoir terisolasi oleh casing dan cement, perforating gun membuat tunnel menembus casing, cement sheath, dan masuk ke formasi. Desain perforasi memengaruhi pressure drop near-wellbore dan skin.

            **Parameter desain:** interval depth, gun OD, charge type, penetration, entrance-hole diameter, shot density (shots/ft), phasing, gun centralization, underbalance/overbalance, dan conveyance (wireline, tubing-conveyed, coiled tubing).

            **Underbalanced perforating** membantu membersihkan crushed zone/debris karena fluida mengalir dari formasi ke sumur setelah charge ditembakkan. Overbalanced perforating dapat dipilih untuk kebutuhan operasional tertentu tetapi berisiko mendorong debris/filtrate ke formasi.

            **Sebelum firing:** korelasi depth dengan gamma ray/CCL, verifikasi barrier dan pressure control, cek explosive safety, konfirmasi interval terhadap petrophysics dan cement quality. **Sesudah firing:** lakukan cleanup, pantau pressure/returns, dan pastikan konektivitas interval sebelum well test.
            """
        )

    with st.expander("6 · Well Testing dan DST: mengukur respons dinamis reservoir"):
        st.markdown(
            """
            **Tujuan DST/well test:** memperoleh representative fluid sample, initial reservoir pressure, deliverability, permeability-thickness (kh), skin, wellbore storage, flow regime, heterogeneity, dan indikasi boundary.

            **Peralatan downhole umum:** packer, tester valve, circulating/reversing valve, safety joint, pressure-temperature gauges, sampler, jars, dan drill collars/tubing. **Surface equipment:** flowhead, choke manifold, heater bila diperlukan, separator, flare/burner, tank, metering, dan emergency shutdown.

            **Urutan cased-hole DST konseptual:**
            1. Run string dan gauge, depth correlate, lalu set packer untuk mengisolasi interval.
            2. Lakukan **initial shut-in** agar pressure mendekati static formation pressure.
            3. Buka tester valve: fluida mengalir pada choke terkontrol; catat rate, pressure, temperature, dan sample.
            4. Tutup valve: lakukan **pressure buildup**. Respons buildup adalah sumber utama PTA karena rate menjadi nol dan pressure recovery direkam.
            5. Ulangi flow/shut-in bila perlu untuk memperoleh stabilitas dan verifikasi deliverability.
            6. Ambil representative sample, reverse/circulate, kill well sesuai program, release packer, dan pull out safely.

            **Data minimum yang harus sinkron waktunya:** downhole pressure-temperature, surface pressure-temperature, oil/gas/water rate, choke size, fluid properties/PVT sample, event log valve, depth/datum gauge, dan operational notes. Kesalahan time synchronization atau rate history langsung memengaruhi pressure-transient interpretation.

            **Interpretasi:** Cartesian memberi overview; semilog membantu radial-flow straight line; log–log pressure dan derivative mengidentifikasi storage, skin transition, radial flow, fracture/dual-porosity behavior, serta boundary.

            **Safety:** DST membawa hydrocarbon ke surface dengan barrier sementara. Program harus mencakup well control, H₂S/CO₂, ignition/flare radiation, erosion, hydrate, pressure rating, exclusion zone, communication, dan contingency shut-in/kill.
            """
        )

    with st.expander("6A · Membaca urutan pressure chart DST menurut Chaudhry", expanded=True):
        st.markdown(
            """
            Chaudhry menjelaskan DST tool sebagai susunan **packer dan valve di ujung drill pipe**. Packer mengisolasi interval target dari drilling mud sehingga formasi dapat mengalir ke test chamber/drill collar/drill pipe. Isolasi ini juga membantu mengurangi volume yang ikut terkompresi sehingga wellbore-storage effect dapat diperkecil.

            Satu pressure chart DST sebaiknya dibaca seperti timeline operasi, bukan langsung dianggap sebagai kurva reservoir:

            1. **Surface baseline.** Gauge berada di surface; baseline harus jelas dan stabil.
            2. **Going into hole.** Pressure naik mengikuti hydrostatic mud column ketika tool diturunkan. Bagian ini menjadi pemeriksaan depth dan mud weight.
            3. **Setting packer.** Packer mengembang/menekan annular mud dan dapat menghasilkan pressure bump. Bump ini adalah respons mekanis, bukan respons reservoir.
            4. **Initial flow.** Tester valve dibuka. Pressure turun dan fluida mulai masuk dari formasi. Initial flow biasanya singkat untuk membuang excess pressure akibat packer setting dan membersihkan volume dekat tool.
            5. **Initial shut-in.** Valve ditutup, flow berhenti, dan pressure build up. Tujuannya mendekati initial formation pressure sebelum reservoir terlalu banyak terdeplesi.
            6. **Final flow.** Valve dibuka kembali lebih lama agar pressure disturbance menyelidiki radius yang lebih jauh dan memberi informasi deliverability.
            7. **Final shut-in.** Valve ditutup lagi. Final buildup biasanya merupakan data interpretasi paling berharga karena durasi investigasinya lebih panjang.
            8. **Release packer.** Ketika packer dilepas, gauge kembali melihat hydrostatic mud pressure.
            9. **Coming out of hole.** Pressure hydrostatic turun saat string ditarik; fluid recovery di drill pipe/surface didokumentasikan.

            Handbook menggambarkan two-cycle test dengan urutan **initial flow → initial shut-in → final flow → final shut-in**. Dua pressure recorder—sering disebut “bombs” pada terminologi lama—memberi redundansi jika satu gauge bermasalah.
            """
        )

    with st.expander("6B · Merencanakan durasi flow/shut-in dan memeriksa kualitas DST"):
        st.markdown(
            """
            **Tidak ada durasi universal.** Waktu harus mengikuti permeability, viscosity, expected pressure, radius investigasi, storage, keselamatan, dan objective test. Sebagai contoh praktik yang dibahas Chaudhry:
            - initial flow sering singkat, sekitar **5–15 menit**, untuk mengurangi excess pressure setelah packer setting;
            - initial buildup dapat sekitar **30–60 menit** ketika targetnya memperoleh initial reservoir pressure yang reliabel;
            - second/final flow dapat dirancang sekitar **60 menit atau lebih** agar menyelidiki formasi lebih jauh;
            - final shut-in harus cukup lama untuk melewati storage/transition dan menangkap interval interpretasi yang dibutuhkan.

            Angka tersebut adalah contoh perencanaan handbook, **bukan aturan tetap**. Test engineer harus melakukan pre-test design menggunakan expected kh, storage, gauge resolution, time limit, dan safety envelope.

            Chaudhry memberikan tiga tanda utama **good DST pressure chart**:
            1. pressure baseline lurus dan jelas;
            2. initial dan final hydrostatic mud pressure konsisten satu sama lain serta cocok dengan depth dan mud weight;
            3. flow dan buildup pressure terekam sebagai kurva yang halus.

            Jika syarat itu tidak terpenuhi, jangan buru-buru menghitung k atau skin. Periksa kemungkinan leaking tool/packer, plugging, bad-hole condition, gauge drift, valve event yang salah, perubahan rate, mud movement, atau time-depth mismatch. Quality control operasional dilakukan **sebelum** reservoir interpretation.

            Pemeriksaan minimum lainnya: hitung ulang expected hydrostatic mud pressure; cek apakah pressure saat going-in/coming-out logis; rekonsiliasi fluid recovery dengan drill-pipe capacity; cocokkan event log dengan infleksi chart; dan pastikan dua gauge menunjukkan tren yang kompatibel.
            """
        )

    with st.expander("6C · Dari pressure chart menjadi parameter reservoir"):
        st.markdown(
            """
            Setelah chart dinyatakan valid, bagian buildup dianalisis seperti pressure-buildup test lainnya. Handbook membahas beberapa pendekatan dan batas penggunaannya:

            - **Horner/MDH straight-line:** berguna bila formation thickness dan viscosity diketahui, shut-in cukup panjang, dan wellbore storage tidak mendominasi. Slope dipakai untuk estimasi transmissibility/permeability dan intercept untuk pressure extrapolation.
            - **Type-curve matching:** membantu ketika storage dan skin memengaruhi early-time sehingga straight-line belum jelas.
            - **Computer matching/numerical interpretation:** bermanfaat ketika rate history atau respons terlalu kompleks untuk metode manual, tetapi hasil tetap harus dikontrol oleh geologi, completion, dan QC data.
            - **Pressure derivative:** menonjolkan perubahan slope dan membantu membedakan wellbore storage, radial flow, dual porosity, fracture, serta boundary.

            **Urutan interpretasi untuk pemula:**
            1. pastikan event dan pressure chart valid;
            2. susun rate history dan time reference;
            3. koreksi gauge depth/datum dan gunakan PVT yang representatif;
            4. plot pressure serta derivative pada log–log;
            5. identifikasi early-, middle-, dan late-time regime;
            6. pilih model yang konsisten dengan completion dan geologi;
            7. estimasi parameter;
            8. lakukan history match dan sensitivity;
            9. laporkan uncertainty serta alternatif model, bukan hanya satu angka.

            **Prinsip penting:** kurva yang tampak cocok belum tentu modelnya unik. Interpretasi harus konsisten dengan log, core, RFT/PVT, perforation interval, dan operasi test.
            """
        )

    st.markdown("### Istilah minimum yang perlu dipahami")
    pr1_glossary = pd.DataFrame(
        [
            ("Drawdown", "Pressure turun ketika sumur dialirkan; input utamanya rate history."),
            ("Buildup", "Pressure pulih setelah sumur ditutup; dipakai untuk melihat respons reservoir tanpa surface-rate noise saat shut-in."),
            ("kh", "Transmissibility capacity: permeability dikalikan net thickness."),
            ("Skin", "Tambahan pressure drop dekat sumur akibat damage, stimulation, completion, atau geometry."),
            ("Wellbore storage", "Early-time flow berasal dari ekspansi/kompresi fluida di wellbore, belum murni dari reservoir."),
            ("Radius of investigation", "Jarak karakteristik yang telah dicapai pressure disturbance selama test."),
            ("Flow regime", "Pola aliran—storage, linear, bilinear, radial, spherical, atau boundary-dominated."),
            ("Hydrostatic pressure", "Pressure akibat kolom fluida; pada DST dipakai untuk QC mud weight dan depth."),
            ("Packer", "Seal downhole yang mengisolasi interval target dari annulus."),
            ("Tester valve", "Valve downhole yang mengatur kapan formasi flow atau shut-in."),
        ],
        columns=["Istilah", "Penjelasan sederhana"],
    )
    st.dataframe(pr1_glossary, use_container_width=True, hide_index=True)

    st.markdown("### Rantai data: apa yang dikonfirmasi pada tahap berikutnya?")
    data_chain = pd.DataFrame(
        [
            ("Drilling/Mud log", "Formation top, show, losses/kick, hole condition", "Logging depth dan interval evaluasi"),
            ("Open-hole logging", "Lithology, ϕ, Sw, net pay, fracture", "RFT station, core point, completion interval"),
            ("RFT/MDT", "Pressure, gradient, mobility, sample, contact", "Fluid model/PVT dan initial pressure"),
            ("Core RCA/SCAL", "k, ϕ, Pc, kr, wettability", "Kalibrasi log dan dynamic model"),
            ("Cement/LOT", "Zonal isolation dan pressure integrity", "Safe perforation dan pressure envelope"),
            ("Perforation", "Reservoir–well connection", "Interval yang benar-benar diuji"),
            ("DST/Well test", "Rate, pressure transient, PVT sample", "kh, skin, flow regime, boundary, deliverability"),
        ],
        columns=["Sumber", "Data utama", "Dipakai untuk"],
    )
    st.dataframe(data_chain, use_container_width=True, hide_index=True)

    st.markdown("### Template jawaban tulisan tangan lengkap + gambar sederhana")
    st.caption("Diagram berikut sengaja dibuat sederhana agar mudah disalin dan digambar ulang di kertas/tablet. Tambahkan warna, panah, dan label dengan gaya sendiri.")

    with st.expander("A · Flowchart enam tahap dan tujuan setiap tahap", expanded=True):
        render_diagram(
            "Workflow data dari rig sampai interpretasi reservoir",
            "Baca dari kiri ke kanan: setiap tahap mengurangi ketidakpastian dan menentukan tahap selanjutnya.",
            "6 STAGES",
            "0 0 1200 300",
            """
            <defs>
              <linearGradient id="wf" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0c4f86"/><stop offset="1" stop-color="#1976d2"/></linearGradient>
              <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#4c89b8"/></marker>
            </defs>
            <text x="32" y="35" font-size="15" font-weight="700" fill="#516579">STATIC DATA ACQUISITION</text>
            <text x="818" y="35" font-size="15" font-weight="700" fill="#516579">DYNAMIC VALIDATION</text>
            <line x1="195" y1="150" x2="218" y2="150" stroke="#4c89b8" stroke-width="4" marker-end="url(#arrow)"/>
            <line x1="390" y1="150" x2="413" y2="150" stroke="#4c89b8" stroke-width="4" marker-end="url(#arrow)"/>
            <line x1="585" y1="150" x2="608" y2="150" stroke="#4c89b8" stroke-width="4" marker-end="url(#arrow)"/>
            <line x1="780" y1="150" x2="803" y2="150" stroke="#4c89b8" stroke-width="4" marker-end="url(#arrow)"/>
            <line x1="975" y1="150" x2="998" y2="150" stroke="#4c89b8" stroke-width="4" marker-end="url(#arrow)"/>
            <g transform="translate(25 70)"><rect width="170" height="160" rx="16" fill="url(#wf)"/><circle cx="28" cy="30" r="16" fill="#74b9f0"/><text x="28" y="35" text-anchor="middle" font-size="14" font-weight="800" fill="#07345a">1</text><text x="18" y="72" font-size="17" font-weight="800" fill="white">DRILLING</text><text x="18" y="100" font-size="12" fill="#d8ecfb">Akses target</text><text x="18" y="120" font-size="12" fill="#d8ecfb">+ well control</text><text x="18" y="145" font-size="10" fill="#9ec7e6">Output: depth, show</text></g>
            <g transform="translate(220 70)"><rect width="170" height="160" rx="16" fill="url(#wf)"/><circle cx="28" cy="30" r="16" fill="#74b9f0"/><text x="28" y="35" text-anchor="middle" font-size="14" font-weight="800" fill="#07345a">2</text><text x="18" y="72" font-size="17" font-weight="800" fill="white">LOGGING</text><text x="18" y="100" font-size="12" fill="#d8ecfb">Batuan &amp; fluida</text><text x="18" y="120" font-size="12" fill="#d8ecfb">secara kontinu</text><text x="18" y="145" font-size="10" fill="#9ec7e6">Output: ϕ, Sw, pay</text></g>
            <g transform="translate(415 70)"><rect width="170" height="160" rx="16" fill="url(#wf)"/><circle cx="28" cy="30" r="16" fill="#74b9f0"/><text x="28" y="35" text-anchor="middle" font-size="14" font-weight="800" fill="#07345a">3</text><text x="18" y="72" font-size="17" font-weight="800" fill="white">RFT + CORE</text><text x="18" y="100" font-size="12" fill="#d8ecfb">Pressure, contact</text><text x="18" y="120" font-size="12" fill="#d8ecfb">+ batuan nyata</text><text x="18" y="145" font-size="10" fill="#9ec7e6">Output: FWL, k, PVT</text></g>
            <g transform="translate(610 70)"><rect width="170" height="160" rx="16" fill="url(#wf)"/><circle cx="28" cy="30" r="16" fill="#74b9f0"/><text x="28" y="35" text-anchor="middle" font-size="14" font-weight="800" fill="#07345a">4</text><text x="18" y="72" font-size="16" font-weight="800" fill="white">CASE + CEMENT</text><text x="18" y="100" font-size="12" fill="#d8ecfb">Isolasi zona</text><text x="18" y="120" font-size="12" fill="#d8ecfb">+ integrity LOT</text><text x="18" y="145" font-size="10" fill="#9ec7e6">Output: pressure limit</text></g>
            <g transform="translate(805 70)"><rect width="170" height="160" rx="16" fill="url(#wf)"/><circle cx="28" cy="30" r="16" fill="#74b9f0"/><text x="28" y="35" text-anchor="middle" font-size="14" font-weight="800" fill="#07345a">5</text><text x="18" y="72" font-size="16" font-weight="800" fill="white">PERFORATION</text><text x="18" y="100" font-size="12" fill="#d8ecfb">Buat tunnel</text><text x="18" y="120" font-size="12" fill="#d8ecfb">reservoir–well</text><text x="18" y="145" font-size="10" fill="#9ec7e6">Output: connection</text></g>
            <g transform="translate(1000 70)"><rect width="170" height="160" rx="16" fill="url(#wf)"/><circle cx="28" cy="30" r="16" fill="#74b9f0"/><text x="28" y="35" text-anchor="middle" font-size="14" font-weight="800" fill="#07345a">6</text><text x="18" y="72" font-size="17" font-weight="800" fill="white">DST / PTA</text><text x="18" y="100" font-size="12" fill="#d8ecfb">Flow + shut-in</text><text x="18" y="120" font-size="12" fill="#d8ecfb">pressure response</text><text x="18" y="145" font-size="10" fill="#9ec7e6">Output: kh, skin</text></g>
            <text x="600" y="270" text-anchor="middle" font-size="13" font-weight="700" fill="#5b7186">INPUT → MEASUREMENT → QUALITY CONTROL → DECISION</text>
            """,
        )
        st.code(
            """[1. PENGEBORAN] ──→ [2. LOGGING] ──→ [3. RFT & CORING]
       │                  │                    │
       ▼                  ▼                    ▼
 akses aman       petakan batuan       pressure, fluida,
 ke target        dan kandidat pay     contact, core

[4. CASING & CEMENT] ──→ [5. PERFORATION] ──→ [6. DST / WELL TEST]
          │                       │                       │
          ▼                       ▼                       ▼
 isolasi & integritas     koneksi reservoir       ukur respons dinamis,
 pressure barrier         ke wellbore              kh, skin, boundary""",
            language="text",
        )
        st.markdown(
            """
            **Kalimat pengantar yang dapat ditulis:**
            “Workflow pengambilan data well testing dimulai saat pengeboran karena data pertama tentang kedalaman, lithology, gas show, dan kondisi lubang sudah diperoleh sejak bit menembus formasi. Setelah target dicapai, logging, RFT, dan coring memperjelas model statik reservoir. Casing, cementing, serta perforation kemudian membuat sumur aman dan menentukan interval yang benar-benar terhubung. Terakhir, DST memberikan respons dinamis pressure dan rate untuk menguji apakah model statik tersebut benar.”
            """
        )

    with st.expander("B · Jawaban empat subjudul untuk setiap tahap"):
        stage_answer = pd.DataFrame(
            [
                ("1. Pengeboran", "Mencapai target secara aman dan membuat lubang evaluasi", "Drill, circulate mud, mud logging, MWD/LWD, monitor well control", "ROP, cutting, gas show, mud loss/kick, depth, hole condition", "Casing point, core/log program; QC depth, mud, dan well control"),
                ("2. Logging", "Mengidentifikasi reservoir dan kandidat pay secara kontinu", "Run GR, resistivity, density-neutron, sonic, caliper/image; depth match", "Lithology, ϕ, Sw, shale volume, fractures, net pay", "Pilih RFT/core/perforation interval; QC borehole dan environmental correction"),
                ("3. RFT & Coring", "Mengukur pressure/fluid column dan batuan secara langsung", "RFT pretest–buildup–sample; whole/sidewall core; RCA/SCAL", "Pressure, mobility, gradient, FWL, sample PVT, k, ϕ, Pc, kr", "Fluid contact dan rock type; QC seal, supercharge, recovery, preservation"),
                ("4. Casing & Cement", "Menopang sumur, isolasi zona, dan menjaga pressure integrity", "Run casing, centralize, cement, WOC, pressure test, FIT/LOT", "Cement placement, CBL/VDL, leak-off/FIT pressure, shoe integrity", "Safe mud/pressure envelope; QC cement bond dan hydrostatic correction"),
                ("5. Perforation", "Membuat tunnel terkontrol dari wellbore ke reservoir", "Depth correlate, set gun, fire charges, cleanup", "Interval, shot density, phasing, penetration, initial inflow", "Koneksi dan skin completion; QC depth, barrier, gun, cement"),
                ("6. DST/Well Test", "Mengukur kemampuan alir dan respons pressure terhadap perubahan rate", "Set packer, flow, shut-in, repeat cycle, sample, recover tool", "Downhole P/T, rate, valve event, fluid recovery/sample", "kh, skin, pᵢ, flow regime, boundary; QC chart, gauge, time/rate history"),
            ],
            columns=["Tahap", "Tujuan", "Proses", "Data", "Keputusan / QC"],
        )
        st.dataframe(stage_answer, use_container_width=True, hide_index=True)
        st.markdown("Saat menulis tangan, jadikan setiap baris tabel sebagai satu subbab. Jangan hanya menyalin istilah; jelaskan hubungan **data → interpretasi → keputusan**.")

    with st.expander("C · Gambar crossplot RFT pressure–TVDSS dan FWL"):
        render_diagram(
            "RFT pressure gradient dan fluid contact",
            "Titik adalah pressure station; garis adalah regresi. Perpotongan oil–water menunjukkan FWL.",
            "RFT / MDT",
            "0 0 1000 460",
            """
            <defs><marker id="a2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#405b73"/></marker></defs>
            <line x1="105" y1="385" x2="900" y2="385" stroke="#405b73" stroke-width="3" marker-end="url(#a2)"/>
            <line x1="105" y1="385" x2="105" y2="55" stroke="#405b73" stroke-width="3" marker-end="url(#a2)"/>
            <text x="890" y="425" text-anchor="end" font-size="18" font-weight="700" fill="#31485e">Pressure (psi) →</text>
            <text x="35" y="45" font-size="18" font-weight="700" fill="#31485e">TVDSS</text><text x="120" y="72" font-size="13" fill="#61778b">lebih dangkal</text>
            <text x="24" y="375" font-size="13" fill="#61778b">lebih dalam</text>
            <line x1="195" y1="90" x2="390" y2="190" stroke="#ef9b35" stroke-width="5"/>
            <line x1="390" y1="190" x2="600" y2="315" stroke="#1976d2" stroke-width="5"/>
            <line x1="390" y1="190" x2="730" y2="350" stroke="#19a58b" stroke-width="5"/>
            <g fill="#ef9b35" stroke="white" stroke-width="3"><circle cx="225" cy="105" r="8"/><circle cx="275" cy="130" r="8"/><circle cx="330" cy="160" r="8"/></g>
            <g fill="#1976d2" stroke="white" stroke-width="3"><circle cx="430" cy="215" r="8"/><circle cx="485" cy="248" r="8"/><circle cx="545" cy="282" r="8"/></g>
            <g fill="#19a58b" stroke="white" stroke-width="3"><rect x="443" y="208" width="14" height="14"/><rect x="522" y="247" width="14" height="14"/><rect x="610" y="287" width="14" height="14"/><rect x="690" y="327" width="14" height="14"/></g>
            <circle cx="390" cy="190" r="12" fill="#ffffff" stroke="#b8295b" stroke-width="5"/>
            <line x1="390" y1="190" x2="830" y2="190" stroke="#b8295b" stroke-width="2" stroke-dasharray="8 7"/>
            <text x="840" y="196" font-size="17" font-weight="800" fill="#b8295b">FWL</text>
            <rect x="715" y="65" width="220" height="105" rx="12" fill="#edf5fc" stroke="#c8dceb"/>
            <circle cx="742" cy="92" r="7" fill="#ef9b35"/><text x="760" y="98" font-size="14" fill="#31485e">Gas gradient — kecil</text>
            <circle cx="742" cy="122" r="7" fill="#1976d2"/><text x="760" y="128" font-size="14" fill="#31485e">Oil gradient</text>
            <rect x="735" y="144" width="14" height="14" fill="#19a58b"/><text x="760" y="157" font-size="14" fill="#31485e">Water — lebih besar</text>
            <text x="250" y="438" font-size="14" font-weight="700" fill="#526b80">QC: gunakan TVDSS, buang station supercharge/bad seal, regresi tiap fluid column</text>
            """,
        )
        st.code(
            """TVDSS / kedalaman
bertambah ke bawah
       ↓
       │   ● gas stations
       │      ●
       │         ●          garis gas: gradient kecil
       │
       │      ○ oil stations
       │          ○
       │              ○
       │                 X  ← perpotongan oil–water = FWL
       │              □
       │          □          garis water: gradient lebih besar
       │      □ water stations
       └────────────────────────────→ Pressure (psi)

       slope tiap garis = dp/dz = fluid gradient""",
            language="text",
        )
        st.markdown(
            r"""
            **Cara menggambar dan menjelaskan:**
            1. Buat sumbu horizontal pressure dan sumbu vertikal TVDSS/depth.
            2. Plot station yang lolos QC, lalu tarik regresi terpisah untuk gas, oil, dan water.
            3. Tulis $p=a+Gz$ dan $G=dp/dz$. Gradient kecil menunjukkan density lebih ringan.
            4. Perpotongan regresi oil–water memberi estimasi $z_{FWL}=(a_o-a_w)/(G_w-G_o)$.
            5. Beri catatan bahwa FWL adalah level zero capillary pressure dan dapat berbeda dari OWC log karena transition zone.
            6. Hubungkan ke PVT: density, bubble/dew point, viscosity, B, GOR, dan contamination sample dipakai untuk mengecek konsistensi fluid column.
            """
        )

    with st.expander("D · Gambar dan tabel whole core vs sidewall core"):
        render_diagram(
            "Tiga cara memperoleh sample batuan",
            "Perhatikan kapan sample diambil, ukurannya, kontinuitas, dan potensi damage.",
            "CORE",
            "0 0 1100 420",
            """
            <rect x="35" y="45" width="315" height="325" rx="18" fill="#eef6fd" stroke="#bfd8ec" stroke-width="2"/>
            <text x="58" y="82" font-size="20" font-weight="800" fill="#154d78">WHOLE CORE</text>
            <rect x="130" y="115" width="120" height="205" rx="55" fill="#c99a67" stroke="#815c39" stroke-width="4"/>
            <path d="M145 145 Q190 120 235 145 M145 190 Q190 165 235 190 M145 235 Q190 210 235 235 M145 280 Q190 255 235 280" fill="none" stroke="#8a623e" stroke-width="3"/>
            <text x="58" y="345" font-size="14" fill="#496276">Kontinu · volume besar · RCA/SCAL</text>
            <rect x="392" y="45" width="315" height="325" rx="18" fill="#fff6ea" stroke="#ebd0a7" stroke-width="2"/>
            <text x="415" y="82" font-size="20" font-weight="800" fill="#8a5417">PERCUSSION</text>
            <rect x="475" y="105" width="100" height="235" rx="45" fill="#d6dde5" stroke="#718195" stroke-width="4"/>
            <line x1="630" y1="160" x2="555" y2="160" stroke="#d26d35" stroke-width="7"/><polygon points="555,160 575,148 575,172" fill="#d26d35"/>
            <path d="M560 153 l42 -15 l12 32 l-50 8 z" fill="#c99a67" stroke="#815c39" stroke-width="3"/>
            <text x="415" y="365" font-size="14" fill="#6f5a42">Cepat · sample kecil · risiko remuk</text>
            <rect x="750" y="45" width="315" height="325" rx="18" fill="#edf9f5" stroke="#b9dfd2" stroke-width="2"/>
            <text x="773" y="82" font-size="20" font-weight="800" fill="#18725d">ROTARY SIDEWALL</text>
            <rect x="830" y="105" width="100" height="235" rx="45" fill="#d6dde5" stroke="#718195" stroke-width="4"/>
            <line x1="1005" y1="185" x2="910" y2="185" stroke="#1a9b7e" stroke-width="9"/><circle cx="910" cy="185" r="15" fill="#f1c06e" stroke="#8d652c" stroke-width="4"/>
            <rect x="930" y="168" width="58" height="34" rx="14" fill="#c99a67" stroke="#815c39" stroke-width="3"/>
            <text x="773" y="365" font-size="14" fill="#446d63">Mini plug · kualitas &gt; percussion</text>
            <text x="550" y="402" text-anchor="middle" font-size="14" font-weight="700" fill="#526b80">Semua sample harus depth-match, diberi orientasi, dipreservasi, dan dicatat recovery-nya.</text>
            """,
        )
        st.code(
            """WHOLE CORE                         SIDEWALL CORE

bit + core barrel                  dinding borehole
       ↓                                 │
  ┌───────────┐                     tool │──→ [plug kecil]
  │███████████│  silinder kontinu        │
  │███████████│                     titik│──→ [plug kecil]
  │███████████│                     depth│
  └───────────┘                          │

besar, kontinu, mahal               kecil, depth-selective, cepat
baik untuk RCA + SCAL               percussion atau rotary""",
            language="text",
        )
        st.markdown(
            """
            **Isi tabel tangan:** whole core diambil menggunakan core barrel selama drilling; volumenya besar dan relatif kontinu sehingga struktur, heterogeneity, RCA, dan SCAL lebih representatif, tetapi mahal serta menambah rig time. Percussion sidewall menembakkan sample kecil dari dinding lubang; cepat tetapi sample dapat remuk. Rotary sidewall mengebor mini plug sehingga kualitasnya lebih baik daripada percussion, tetapi ukuran dan kontinuitasnya tetap di bawah whole core.
            """
        )

    with st.expander("E · Gambar casing–cement–formation dan posisi LOT"):
        render_diagram(
            "Casing shoe, cement sheath, dan lokasi LOT",
            "LOT menguji formasi tepat di bawah shoe—bukan cement di sembarang kedalaman.",
            "WELL INTEGRITY",
            "0 0 1000 520",
            """
            <defs><marker id="a3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#d24646"/></marker></defs>
            <rect x="0" y="0" width="1000" height="520" fill="#f8fbff"/>
            <path d="M0 80 H380 V470 H0 Z M620 80 H1000 V470 H620 Z" fill="#d2ad7d"/>
            <path d="M380 80 H430 V370 H570 V80 H620 V470 H380 Z" fill="#b6c3cf"/>
            <rect x="430" y="60" width="140" height="315" fill="#f7fafc" stroke="#3d5368" stroke-width="7"/>
            <path d="M430 375 L465 410 H535 L570 375" fill="#5e7184" stroke="#3d5368" stroke-width="4"/>
            <path d="M465 410 V485 M535 410 V485" stroke="#7d6040" stroke-width="4" stroke-dasharray="8 6"/>
            <line x1="500" y1="25" x2="500" y2="330" stroke="#d24646" stroke-width="7" marker-end="url(#a3)"/>
            <text x="525" y="45" font-size="16" font-weight="800" fill="#b23333">Pump pressure</text>
            <text x="445" y="115" font-size="18" font-weight="800" fill="#31485e">CASING</text>
            <text x="315" y="210" font-size="16" font-weight="800" fill="#596e81">CEMENT</text>
            <text x="720" y="210" font-size="16" font-weight="800" fill="#815d36">FORMATION</text>
            <line x1="575" y1="390" x2="750" y2="390" stroke="#176bce" stroke-width="3"/>
            <text x="765" y="397" font-size="18" font-weight="800" fill="#176bce">CASING SHOE</text>
            <rect x="625" y="440" width="290" height="52" rx="12" fill="#eaf4fc" stroke="#b9d5e9"/>
            <text x="645" y="462" font-size="14" font-weight="700" fill="#37546b">FIT: berhenti di target pressure</text>
            <text x="645" y="482" font-size="14" font-weight="700" fill="#b34747">LOT: lanjut sampai slope berubah</text>
            <text x="55" y="505" font-size="13" fill="#5b7185">Urutan: cement → WOC → drill out shoe → pressure test → FIT/LOT → tentukan safe pressure window</text>
            """,
        )
        st.code(
            """             pressure pump
                  ↓
            ┌──────────┐
            │  CASING  │
formation   │ │      │ │   formation
████████████│C│      │C│████████████   C = cement sheath
████████████│C│      │C│████████████
            │C│      │C│
            └─┴──┬───┴─┘  ← casing shoe
                 │
                 │ open hole di bawah shoe
                 ▼
          pressure dinaikkan bertahap
          sampai target FIT / leak-off LOT""",
            language="text",
        )
        st.markdown(
            """
            **Penjelasan:** casing menahan beban dan pressure, sedangkan cement mengisolasi annulus agar tidak terjadi crossflow. Setelah cement mencapai strength dan shoe track dibersihkan, pressure dinaikkan di bawah casing shoe. FIT berhenti pada target pressure untuk membuktikan integritas; LOT diteruskan sampai slope pressure–volume berubah sebagai indikasi leak-off. Hasilnya menentukan safe mud-weight dan pressure window.
            """
        )

    with st.expander("F · Gambar perforation tunnel"):
        render_diagram(
            "Perforation menghubungkan wellbore dan reservoir",
            "Charge menembus casing, cement, crushed zone, lalu membuat tunnel ke batuan yang belum rusak.",
            "COMPLETION",
            "0 0 1100 420",
            """
            <defs><marker id="a4" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#19a079"/></marker></defs>
            <rect x="0" y="0" width="230" height="420" fill="#e8eef4"/>
            <rect x="230" y="0" width="95" height="420" fill="#708294"/>
            <rect x="325" y="0" width="115" height="420" fill="#bbc7d1"/>
            <rect x="440" y="0" width="660" height="420" fill="#d2ad7d"/>
            <rect x="440" y="0" width="90" height="420" fill="#ad865f" opacity=".7"/>
            <text x="80" y="45" font-size="18" font-weight="800" fill="#31485e">WELLBORE</text><text x="235" y="45" font-size="16" font-weight="800" fill="white">CASING</text><text x="332" y="45" font-size="16" font-weight="800" fill="#31485e">CEMENT</text><text x="700" y="45" font-size="18" font-weight="800" fill="#6c4b2d">RESERVOIR</text>
            <circle cx="165" cy="205" r="40" fill="#e66b3c" stroke="#a83c21" stroke-width="5"/><text x="165" y="212" text-anchor="middle" font-size="14" font-weight="800" fill="white">CHARGE</text>
            <path d="M205 205 L270 170 L365 175 L465 190 L830 205 L465 220 L365 235 L270 240 Z" fill="#f3b64e" stroke="#c17d19" stroke-width="4"/>
            <path d="M530 175 L820 205 L530 235 Z" fill="#f8d17d"/>
            <line x1="870" y1="205" x2="560" y2="205" stroke="#19a079" stroke-width="7" marker-end="url(#a4)"/>
            <text x="730" y="180" font-size="16" font-weight="800" fill="#15785d">cleanup flow</text>
            <line x1="480" y1="85" x2="480" y2="340" stroke="#8e5d36" stroke-width="2" stroke-dasharray="7 6"/><text x="490" y="105" font-size="13" fill="#714c30">crushed zone</text>
            <line x1="270" y1="300" x2="365" y2="300" stroke="#176bce" stroke-width="4"/><text x="250" y="330" font-size="13" fill="#31536c">entrance hole</text>
            <line x1="365" y1="355" x2="830" y2="355" stroke="#176bce" stroke-width="4"/><text x="500" y="383" font-size="14" font-weight="700" fill="#31536c">penetration / tunnel length</text>
            """,
        )
        st.code(
            """wellbore        casing      cement       reservoir
   │              │            │        █████████████
   │   charge  *──┼────────────┼══════▶ █ tunnel ████
   │              │            │        █████████████
   │  ◀──────── aliran setelah perforasi (underbalance)
   │
   └─ shot lain diputar menurut phasing, mis. 60° / 90° / 120°""",
            language="text",
        )
        st.markdown(
            """
            Tulis tiga label utama: **shot density** = jumlah lubang per ft; **phasing** = distribusi sudut charge; **penetration** = panjang tunnel menembus formation. Pada underbalanced perforating, wellbore pressure dibuat lebih rendah sehingga fluida mengalir masuk dan membantu membersihkan debris/crushed zone. Depth correlation dan cement quality harus benar agar zona yang ditembak memang zona target.
            """
        )

    with st.expander("G · Gambar DST string dan timeline pressure event"):
        render_diagram(
            "DST string dan urutan event yang dibaca gauge",
            "Kolom kiri menunjukkan alat; kolom kanan menunjukkan hubungan valve event dengan pressure response.",
            "DST TOOL",
            "0 0 1200 650",
            """
            <rect x="30" y="35" width="390" height="575" rx="18" fill="#eef6fd" stroke="#bdd7eb" stroke-width="2"/>
            <text x="55" y="72" font-size="20" font-weight="800" fill="#164f7b">DOWNHOLE STRING</text>
            <line x1="220" y1="95" x2="220" y2="560" stroke="#405b72" stroke-width="9"/>
            <g fill="#176bce" stroke="white" stroke-width="3"><rect x="145" y="115" width="150" height="48" rx="10"/><rect x="145" y="185" width="150" height="48" rx="10"/><rect x="145" y="255" width="150" height="48" rx="10"/><rect x="145" y="325" width="150" height="48" rx="10"/></g>
            <text x="220" y="145" text-anchor="middle" font-size="13" font-weight="800" fill="white">REVERSE VALVE</text><text x="220" y="215" text-anchor="middle" font-size="13" font-weight="800" fill="white">TESTER VALVE</text><text x="220" y="285" text-anchor="middle" font-size="13" font-weight="800" fill="white">P / T GAUGE</text><text x="220" y="355" text-anchor="middle" font-size="13" font-weight="800" fill="white">JARS + SAFETY</text>
            <path d="M95 415 Q220 365 345 415 L325 475 Q220 430 115 475 Z" fill="#e4a73f" stroke="#9a6419" stroke-width="4"/><text x="220" y="430" text-anchor="middle" font-size="14" font-weight="800" fill="#5f3e13">PACKER</text>
            <rect x="70" y="485" width="300" height="75" rx="8" fill="#d0aa79"/><path d="M70 500 H370 M70 520 H370 M70 540 H370" stroke="#9b744a" stroke-width="3"/><text x="220" y="530" text-anchor="middle" font-size="14" font-weight="800" fill="#5d4227">TEST INTERVAL</text>
            <rect x="455" y="35" width="710" height="575" rx="18" fill="#ffffff" stroke="#c8dae8" stroke-width="2"/>
            <text x="485" y="72" font-size="20" font-weight="800" fill="#164f7b">PRESSURE EVENT TIMELINE</text>
            <line x1="500" y1="520" x2="1125" y2="520" stroke="#405b72" stroke-width="3"/>
            <line x1="500" y1="520" x2="500" y2="110" stroke="#405b72" stroke-width="3"/>
            <text x="1085" y="555" font-size="15" font-weight="700" fill="#405b72">TIME →</text><text x="465" y="115" font-size="15" font-weight="700" fill="#405b72">P ↑</text>
            <polyline points="500,470 585,270 625,235 655,425 735,360 765,210 800,185 835,430 945,340 980,165 1045,145 1080,260 1120,465" fill="none" stroke="#176bce" stroke-width="6"/>
            <g fill="#0e4f82"><circle cx="585" cy="270" r="7"/><circle cx="655" cy="425" r="7"/><circle cx="765" cy="210" r="7"/><circle cx="835" cy="430" r="7"/><circle cx="980" cy="165" r="7"/><circle cx="1120" cy="465" r="7"/></g>
            <text x="520" y="490" font-size="12" fill="#536b7f">run in</text><text x="575" y="222" font-size="12" fill="#536b7f">hydrostatic</text><text x="625" y="450" font-size="12" fill="#b64f3d">flow-1</text><text x="720" y="190" font-size="12" fill="#19795f">shut-in-1</text><text x="805" y="455" font-size="12" fill="#b64f3d">flow-2</text><text x="930" y="135" font-size="12" fill="#19795f">shut-in-2</text><text x="1050" y="490" font-size="12" fill="#536b7f">release / out</text>
            <rect x="500" y="580" width="625" height="1" fill="#dce7ef"/>
            <text x="500" y="600" font-size="13" font-weight="700" fill="#526b80">QC: initial ≈ final hydrostatic · smooth flow/buildup · event log sinkron · dua gauge konsisten</text>
            """,
        )
        st.code(
            """SURFACE: flowhead → choke → separator → flare / tank
                       │
                   drill pipe
                       │
             reverse circulation valve
                       │
                  tester valve   ← buka = flow, tutup = buildup
                       │
              pressure / temperature gauge
                       │
                 safety joint + jars
                       │
                ┌──── PACKER ────┐  ← isolasi interval
formation  █████│ perforated zone│█████
                └──── anchor ────┘

TIME →  Run in → Set packer → FLOW-1 → SHUT-IN-1 → FLOW-2 → SHUT-IN-2 → Release → Pull out
P     → hydrostatic   bump       turun      pulih        turun       pulih       hydrostatic""",
            language="text",
        )
        st.markdown(
            """
            **Narasi gambar:** saat tool masuk, gauge membaca hydrostatic mud pressure. Setting packer dapat membuat pressure bump. Valve dibuka untuk initial flow, lalu ditutup untuk initial buildup. Siklus diulang dengan final flow dan final shut-in yang lebih representatif. Setelah packer dilepas, gauge kembali membaca hydrostatic pressure. Bandingkan initial/final hydrostatic dan pastikan flow/buildup curve halus sebelum interpretasi.
            """
        )

    with st.expander("H · Diagram integrasi static model dan dynamic validation"):
        render_diagram(
            "Closed-loop reservoir characterization",
            "Interpretasi tidak berhenti di satu hasil: data statik dan data dinamik harus saling menguji.",
            "INTEGRATION",
            "0 0 1200 430",
            """
            <defs><marker id="a5" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#39749f"/></marker></defs>
            <rect x="35" y="55" width="220" height="310" rx="18" fill="#eef6fd" stroke="#bcd7eb" stroke-width="2"/>
            <text x="58" y="90" font-size="18" font-weight="800" fill="#164f7b">STATIC INPUTS</text>
            <g font-size="14" fill="#405b72"><text x="65" y="135">● Drilling / mud log</text><text x="65" y="172">● Logging</text><text x="65" y="209">● Core RCA / SCAL</text><text x="65" y="246">● RFT gradients</text><text x="65" y="283">● PVT sample</text><text x="65" y="320">● Completion geometry</text></g>
            <line x1="255" y1="210" x2="340" y2="210" stroke="#39749f" stroke-width="5" marker-end="url(#a5)"/>
            <rect x="350" y="95" width="210" height="230" rx="18" fill="#0d4f82"/>
            <text x="455" y="137" text-anchor="middle" font-size="20" font-weight="800" fill="white">STATIC MODEL</text><text x="455" y="177" text-anchor="middle" font-size="14" fill="#c9e4f7">ϕ · k · Sw · facies</text><text x="455" y="207" text-anchor="middle" font-size="14" fill="#c9e4f7">contact · fluid · h</text><text x="455" y="260" text-anchor="middle" font-size="13" fill="#88bde3">→ test design</text><text x="455" y="285" text-anchor="middle" font-size="13" fill="#88bde3">→ expected response</text>
            <line x1="560" y1="210" x2="645" y2="210" stroke="#39749f" stroke-width="5" marker-end="url(#a5)"/>
            <rect x="655" y="95" width="210" height="230" rx="18" fill="#176bce"/>
            <text x="760" y="137" text-anchor="middle" font-size="20" font-weight="800" fill="white">DST DATA</text><text x="760" y="177" text-anchor="middle" font-size="14" fill="#d7ecfb">P(t) · q(t) · T(t)</text><text x="760" y="207" text-anchor="middle" font-size="14" fill="#d7ecfb">valve event · sample</text><text x="760" y="260" text-anchor="middle" font-size="13" fill="#a9d1ef">→ QC chart</text><text x="760" y="285" text-anchor="middle" font-size="13" fill="#a9d1ef">→ PTA / matching</text>
            <line x1="865" y1="210" x2="950" y2="210" stroke="#39749f" stroke-width="5" marker-end="url(#a5)"/>
            <rect x="960" y="95" width="205" height="230" rx="18" fill="#168b75"/>
            <text x="1062" y="137" text-anchor="middle" font-size="20" font-weight="800" fill="white">DYNAMIC MODEL</text><text x="1062" y="177" text-anchor="middle" font-size="14" fill="#d8f3ed">kh · skin · pᵢ</text><text x="1062" y="207" text-anchor="middle" font-size="14" fill="#d8f3ed">regime · boundary</text><text x="1062" y="260" text-anchor="middle" font-size="13" fill="#b9e8dc">→ completion decision</text><text x="1062" y="285" text-anchor="middle" font-size="13" fill="#b9e8dc">→ development plan</text>
            <path d="M1060 342 C1060 405 450 405 450 335" fill="none" stroke="#d16b42" stroke-width="5" stroke-dasharray="10 7" marker-end="url(#a5)"/>
            <text x="760" y="400" text-anchor="middle" font-size="14" font-weight="800" fill="#b65734">MISMATCH → cek QC → revisi asumsi → ulangi model</text>
            """,
        )
        st.code(
            """DRILLING ─┐
LOGGING  ──┼─→ STATIC MODEL ─→ TEST DESIGN ─→ DST DATA ─→ PTA / MATCH ─→ DYNAMIC MODEL
CORE     ──┤      │               │             │            │               │
RFT/PVT  ─┘      ϕ, k, Sw,       rate, time,    P(t), q(t),  kh, skin,       keputusan:
                 contact, fluid   gauge, safety  sample       boundary        completion/
                                                                              development

Jika hasil DST tidak cocok dengan static model → cek QC, ubah asumsi/model, lalu iterasi.""",
            language="text",
        )
        st.markdown(
            """
            **Kesimpulan yang dapat ditulis:** log, core, RFT, dan PVT membangun dugaan awal mengenai batuan serta fluida. DST memberikan eksperimen dinamis untuk menguji dugaan itu. Bila pressure response tidak cocok, penyebabnya dapat berupa data buruk, completion effect, heterogeneity, fracture, atau boundary yang belum dimasukkan. Karena itu interpretasi well testing bersifat iteratif dan multidisiplin.
            """
        )

    st.warning("Tuliskan ulang dengan bahasa sendiri. Diagram boleh disederhanakan lagi, tetapi pertahankan arah panah, label alat, jenis data, serta hubungan sebab–akibatnya. Bila notasi/urutan video kuliah berbeda, utamakan versi dosen.")

with tab_pr2:
    st.markdown(
        """
        <div class="guide-hero">
          <div class="guide-eyebrow">PR 2 · Penurunan Diffusivity Equation</div>
          <h2>Mass conservation + Darcy + equation of state</h2>
          <p>Penurunan berikut menunjukkan bagaimana keseimbangan massa, hukum aliran, dan sifat kompresibilitas digabung menjadi persamaan difusivitas radial satu fasa.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Ini adalah penurunan standar untuk fluida slightly compressible, reservoir homogen-isotropik, aliran radial, dan temperatur konstan. Cocokkan simbol, tanda, serta urutan aljabar dengan video segmen 4–5 karena dosen dapat memakai konvensi arah radial atau definisi velocity yang berbeda.")

    with st.expander("A · Persamaan 1 — Kekekalan massa", expanded=True):
        st.markdown(
            r"""
            Ambil control volume annulus berjari-jari $r$ sampai $r+dr$, ketebalan reservoir $h$, dan sudut penuh $2\pi$. Prinsip dasarnya:

            **massa masuk − massa keluar = laju akumulasi massa di pori.**

            Luas aliran radial pada radius $r$ adalah $A_r=2\pi rh$. Jika $v_r$ didefinisikan positif ke arah luar, mass flux adalah $\rho v_r$ dan bentuk diferensial konservasi massa menjadi:
            """
        )
        render_diagram(
            "Annulus control volume r sampai r+dr",
            "Massa masuk di r, massa keluar di r+dr; selisihnya sama dengan akumulasi di dalam cincin.",
            "MASS BALANCE",
            "0 0 1000 380",
            """
            <rect x="60" y="70" width="880" height="220" fill="#f3e3c4" stroke="#c8a866" stroke-width="2"/>
            <text x="80" y="60" font-size="15" font-weight="700" fill="#7a5a25">RESERVOIR, ketebalan h</text>
            <line x1="945" y1="70" x2="945" y2="290" stroke="#7a5a25" stroke-width="2"/>
            <line x1="935" y1="70" x2="955" y2="70" stroke="#7a5a25" stroke-width="2"/>
            <line x1="935" y1="290" x2="955" y2="290" stroke="#7a5a25" stroke-width="2"/>
            <text x="963" y="185" font-size="16" font-weight="800" fill="#7a5a25">h</text>
            <rect x="455" y="70" width="55" height="220" fill="#9fc6ea" stroke="#2e6ca8" stroke-width="3"/>
            <text x="482" y="330" text-anchor="middle" font-size="14" font-weight="800" fill="#154d78">control volume</text>
            <line x1="455" y1="345" x2="455" y2="300" stroke="#2e6ca8" stroke-width="2" stroke-dasharray="4 3"/>
            <line x1="510" y1="345" x2="510" y2="300" stroke="#2e6ca8" stroke-width="2" stroke-dasharray="4 3"/>
            <text x="455" y="360" text-anchor="middle" font-size="15" font-weight="800" fill="#31485e">r</text>
            <text x="510" y="360" text-anchor="middle" font-size="15" font-weight="800" fill="#31485e">r+dr</text>
            <line x1="270" y1="180" x2="450" y2="180" stroke="#b3541e" stroke-width="6" marker-end="url(#pr2arrow)"/>
            <text x="270" y="150" font-size="14" font-weight="700" fill="#8a3f14">ṁ masuk = ρvᵣ|ᵣ · 2πrh</text>
            <line x1="515" y1="180" x2="700" y2="180" stroke="#19795f" stroke-width="6" marker-end="url(#pr2arrow2)"/>
            <text x="515" y="150" font-size="14" font-weight="700" fill="#125c46">ṁ keluar = [ρvᵣ] pada r+dr · 2π(r+dr)h</text>
            <text x="482" y="215" text-anchor="middle" font-size="12" font-weight="700" fill="#154d78">∂(φρ)/∂t</text>
            <text x="482" y="232" text-anchor="middle" font-size="12" font-weight="700" fill="#154d78">· 2πr·dr·h</text>
            <defs>
              <marker id="pr2arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#b3541e"/></marker>
              <marker id="pr2arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#19795f"/></marker>
            </defs>
            <line x1="60" y1="345" x2="920" y2="345" stroke="#405b72" stroke-width="2" marker-end="url(#pr2arrow3)"/>
            <defs><marker id="pr2arrow3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#405b72"/></marker></defs>
            <text x="920" y="368" text-anchor="end" font-size="14" font-weight="700" fill="#405b72">r bertambah →</text>
            <text x="65" y="368" font-size="14" font-weight="700" fill="#405b72">← sumur</text>
            """,
        )
        st.latex(r"-\frac{1}{r}\frac{\partial}{\partial r}\left(r\rho v_r\right)=\frac{\partial(\phi\rho)}{\partial t}")
        st.markdown(
            r"""
            Tanda minus muncul karena net inflow menyebabkan akumulasi. Bila velocity didefinisikan positif menuju sumur, tanda dapat terlihat berbeda tetapi hasil akhir fisiknya sama selama konvensinya konsisten.

            **Makna tiap suku:**
            - $r\rho v_r$ menyatakan mass flow radial setelah geometri silinder diperhitungkan;
            - operator $\frac{1}{r}\frac{\partial}{\partial r}(r\cdot)$ adalah divergence radial;
            - $\phi\rho$ adalah massa fluida per bulk volume;
            - $\partial(\phi\rho)/\partial t$ adalah perubahan storage terhadap waktu.
            """
        )

    with st.expander("B · Persamaan 2 — Darcy equation"):
        st.markdown("Untuk aliran radial satu fasa tanpa gravity term eksplisit pada arah horizontal:")
        st.latex(r"v_r=-\frac{k}{\mu}\frac{\partial p}{\partial r}")
        st.markdown(
            r"""
            - $k$ mengukur kemampuan batuan mengalirkan fluida.
            - $\mu$ adalah tahanan viscous fluida.
            - Gradien pressure adalah driving force.
            - Tanda minus menyatakan fluida mengalir dari pressure tinggi menuju pressure rendah.

            Substitusi Darcy ke mass conservation memberi:
            """
        )
        st.latex(r"\frac{1}{r}\frac{\partial}{\partial r}\left(r\rho\frac{k}{\mu}\frac{\partial p}{\partial r}\right)=\frac{\partial(\phi\rho)}{\partial t}")
        st.markdown("Bentuk ini masih umum: density, porosity, permeability, dan viscosity belum semuanya dianggap konstan.")

    with st.expander("C · Persamaan 3 — Equation of state dan compressibility"):
        st.markdown(
            """
            Untuk fluida slightly compressible pada temperatur konstan, perubahan density terhadap pressure dinyatakan oleh fluid compressibility:
            """
        )
        st.latex(r"c_f=\frac{1}{\rho}\left(\frac{\partial\rho}{\partial p}\right)_T")
        st.latex(r"\frac{\partial\rho}{\partial t}=\rho c_f\frac{\partial p}{\partial t}")
        st.markdown("Pore volume juga berubah terhadap pressure. Definisikan pore/formation compressibility:")
        st.latex(r"c_\phi=\frac{1}{\phi}\frac{\partial\phi}{\partial p}")
        st.markdown("Dengan product rule:")
        st.latex(r"\frac{\partial(\phi\rho)}{\partial t}=\rho\frac{\partial\phi}{\partial t}+\phi\frac{\partial\rho}{\partial t}=\phi\rho(c_\phi+c_f)\frac{\partial p}{\partial t}")
        st.latex(r"c_t=c_\phi+c_f")
        st.markdown("Untuk sistem multiphase, $c_t$ dan storage term lebih kompleks karena melibatkan saturation dan compressibility tiap fase. PR ini memakai single-phase slightly compressible.")

    with st.expander("D · Menggabungkan ketiga persamaan sampai diffusivity equation"):
        st.markdown(
            r"""
            Mulai dari hasil substitusi Darcy. Ambil asumsi $k$, $\mu$, dan ketebalan konstan; abaikan produk gradien density dengan gradien pressure yang merupakan orde kecil untuk slightly compressible liquid. Density kemudian dapat dicoret pada kedua sisi:
            """
        )
        st.latex(r"\frac{k\rho}{\mu}\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial p}{\partial r}\right)=\phi\rho c_t\frac{\partial p}{\partial t}")
        st.latex(r"\boxed{\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial p}{\partial r}\right)=\frac{\phi\mu c_t}{k}\frac{\partial p}{\partial t}}")
        st.markdown("Jika bagian kiri diekspansi:")
        st.latex(r"\boxed{\frac{\partial^2p}{\partial r^2}+\frac{1}{r}\frac{\partial p}{\partial r}=\frac{\phi\mu c_t}{k}\frac{\partial p}{\partial t}}")
        st.markdown(r"Definisikan hydraulic diffusivity $\eta$:")
        st.latex(r"\eta=\frac{k}{\phi\mu c_t},\qquad \frac{\partial p}{\partial t}=\eta\left(\frac{\partial^2p}{\partial r^2}+\frac{1}{r}\frac{\partial p}{\partial r}\right)")
        st.markdown("Dalam field units yang dipakai aplikasi, koefisien konversi menghasilkan definisi dimensionless time:")
        st.latex(r"t_D=\frac{0.0002637\,k\,t}{\phi\mu c_t r_w^2}")
        eta_field = 0.0002637 * k / (phi * mu * ct)
        st.markdown(
            f"**Angka nyata dari sidebar saat ini:** k = {k:g} mD, ϕ = {phi:.3f}, μ = {mu:g} cP, "
            f"cₜ = {ct:.2e} psi⁻¹ → η = 0.0002637·k/(ϕμcₜ) ≈ **{eta_field:,.3f} ft²/jam**. Angka inilah "
            f"yang menentukan seberapa cepat gangguan pressure merambat sebelum waktu lapangan $t$ diubah "
            f"menjadi $t_D$."
        )

    with st.expander("E · Makna fisik, asumsi, dan sanity check"):
        st.markdown(
            r"""
            **Makna diffusivity:** pressure disturbance menyebar karena kombinasi transmissibility dan storage. Nilai $k$ besar mempercepat penyebaran pressure; $\phi$, $\mu$, atau $c_t$ besar memperlambatnya.

            **Asumsi yang harus ditulis:** single phase; slightly compressible; isothermal; Darcy flow; homogen dan isotropik; ketebalan konstan; aliran radial; tidak ada source/sink di dalam domain selain sumur; gravity dan capillary diabaikan; properties dianggap konstan atau variasinya kecil.

            **Sanity check kuantitatif — pengaruh menggandakan satu parameter (parameter lain tetap):**
            - $k\uparrow \Rightarrow \eta\uparrow$: pressure communication lebih cepat.
            - $\mu\uparrow \Rightarrow \eta\downarrow$: fluida lebih sulit bergerak.
            - $c_t\uparrow$ atau $\phi\uparrow \Rightarrow$ storage lebih besar sehingga respons melambat.
            - Operator $1/r$ membedakan geometri radial dari linear Cartesian.

            **Kesalahan umum:** kehilangan faktor $r$ pada divergence; mencampur mass rate dan volumetric rate; salah tanda Darcy; mencoret density tanpa menyatakan slightly compressible; menganggap $c_t=c_f$ tanpa rock compressibility; mencampur field units dan SI tanpa conversion factor.
            """
        )
        eta_base = 0.0002637 * k / (phi * mu * ct)
        sensitivity_table = pd.DataFrame(
            [
                ("k (permeability)", "×2", eta_base * 2.0, "η naik proporsional — komunikasi pressure makin cepat"),
                ("ϕ (porosity)", "×2", eta_base / 2.0, "η turun — storage per volume makin besar, respons melambat"),
                ("μ (viscosity)", "×2", eta_base / 2.0, "η turun — fluida makin sulit mengalir"),
                ("cₜ (total compressibility)", "×2", eta_base / 2.0, "η turun — storage elastik makin besar"),
            ],
            columns=["Parameter digandakan", "Faktor", "η baru (ft²/jam)", "Interpretasi"],
        )
        st.dataframe(
            sensitivity_table.style.format({"η baru (ft²/jam)": "{:,.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"η dasar (parameter sidebar saat ini) = {eta_base:,.3f} ft²/jam. Karena η hanya bergantung pada k/(ϕμcₜ), menggandakan k selalu menggandakan η, sedangkan menggandakan ϕ, μ, atau cₜ selalu membelah η menjadi setengahnya — pola ini berlaku untuk kombinasi parameter berapa pun, bukan hanya angka pada sidebar.")

    st.markdown("### Urutan tulisan tangan PR 2")
    st.markdown(
        r"""
        1. Gambar annular control volume $r$ sampai $r+dr$ dan beri label $2\pi rh$.
        2. Tulis “mass in − mass out = accumulation”, lalu turunkan bentuk divergence radial.
        3. Tulis Darcy equation dan jelaskan tanda minus.
        4. Substitusikan Darcy ke mass balance sebelum membuat asumsi penyederhanaan.
        5. Turunkan storage term memakai product rule pada $\phi\rho$.
        6. Definisikan $c_f$, $c_\phi$, dan $c_t$.
        7. Coret $\rho$, kumpulkan koefisien, lalu beri kotak pada radial diffusivity equation.
        8. Definisikan diffusivity $\eta$ dan tulis interpretasi pengaruh $k$, $\phi$, $\mu$, dan $c_t$.
        9. Akhiri dengan daftar asumsi dan hubungkan ke $t_D$ yang dipakai PR 3.
        """
    )

with tab_guide:
    st.markdown(
        """
        <div class="guide-hero">
          <div class="guide-eyebrow">PR 3 · Code Python Well Testing</div>
          <h2>Panduan memahami model Laplace–Bessel–Gaver–Stehfest</h2>
          <p>Ringkasan ini menghubungkan persamaan, algoritma, parameter, dan bentuk kurva pada aplikasi. Gunakan sebagai kerangka belajar; penjelasan akhir tetap sebaiknya ditulis ulang dengan kata-kata sendiri.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Kesesuaian output tugas</div>', unsafe_allow_html=True)
    output_cols = st.columns(4)
    output_items = [
        ("01", "Cartesian", "p<sub>wf</sub> terhadap waktu linear pada Dashboard."),
        ("02", "Semi-log", "p<sub>wf</sub> terhadap waktu dengan sumbu-x logaritmik."),
        ("03", "Log–log pressure", "Pressure drop Δp pada diagnostic plot."),
        ("04", "Log–log derivative", "dΔp/dln(t), dapat ditampilkan sendiri atau bersama pressure."),
    ]
    for col, (idx, title, desc) in zip(output_cols, output_items):
        col.markdown(
            f'<div class="guide-card"><div class="guide-card-index">OUTPUT {idx}</div><div class="guide-card-title">{title}</div><div class="guide-card-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Parameter aktif, nilai saat ini, dan interpretasinya"):
        parameter_summary = pd.DataFrame(
            [
                ("k", "Permeability", f"{k:g} mD", "Mengontrol kemampuan batuan mengalirkan fluida."),
                ("ϕ", "Porosity", f"{phi:.3f}", "Fraksi volume pori; harus berada antara 0 dan 1."),
                ("cₜ", "Total compressibility", f"{ct:.2e} psi⁻¹", "Penyimpanan elastik total batuan dan fluida."),
                ("C", "Wellbore storage", "0 bbl/psi", "Dikunci nol sesuai asumsi no wellbore storage."),
                ("S", "Skin", "0", "Dikunci nol sesuai asumsi no skin."),
                ("h", "Net thickness", f"{h:g} ft", "Ketebalan interval produktif."),
                ("q", "Constant rate", f"{q:g} STB/hari", "Laju produksi drawdown yang dijaga konstan."),
                ("t", "Real time", f"{t_min:g}–{t_max:g} jam", "Rentang waktu simulasi di ruang nyata."),
                ("tᴅ", "Dimensionless time", f"{result['dimensionless_time'][0]:.2e}–{result['dimensionless_time'][-1]:.2e}", "Waktu yang telah dinormalisasi oleh sifat reservoir."),
                ("Cᴅ", "Dimensionless storage", "0", "Nol karena C = 0."),
                ("pᴅ", "Dimensionless pressure", f"{result['dimensionless_pressure'][-1]:.5f} (akhir)", "Hasil inversi Laplace sebelum konversi ke psi."),
            ],
            columns=["Simbol", "Parameter", "Nilai saat ini", "Interpretasi"],
        )
        st.dataframe(parameter_summary, use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="guide-note"><strong>Random tetapi masuk akal:</strong> tombol “Generate skenario realistis” mengacak k, ϕ, cₜ, μ, B, h, rᵥ, q, dan pᵢ di dalam rentang lapangan yang wajar. C, Cᴅ, dan S tidak diacak karena ketiganya ditetapkan nol oleh model tugas.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Alur fisika dan matematika")
    concept_cols = st.columns(4)
    concept_items = [
        ("01", "Real space", "Mulai dari diffusivity equation radial dan kondisi awal seragam."),
        ("02", "Laplace transform", "Turunan terhadap waktu diubah menjadi perkalian oleh u."),
        ("03", "Bessel solution", "Selesaikan ODE radial dan terapkan kondisi batas sumur serta infinity."),
        ("04", "Inverse transform", "Gaver–Stehfest mengembalikan p̄ᴅ(u) menjadi pᴅ(tᴅ)."),
    ]
    for col, (idx, title, desc) in zip(concept_cols, concept_items):
        col.markdown(
            f'<div class="guide-card"><div class="guide-card-index">STEP {idx}</div><div class="guide-card-title">{title}</div><div class="guide-card-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Backend solver — dari input UI sampai menjadi grafik")
    st.markdown(
        """
        Solver pada aplikasi ini **bukan model AI** dan bukan curve fitting. Ia adalah solver numerik deterministik: input reservoir yang sama selalu menghasilkan output yang sama. `app.py` mengurus antarmuka, state, tabel, dan Plotly; `pressure_model.py` mengurus persamaan fisika, Bessel function, inversi Gaver–Stehfest, konversi satuan, serta derivative.
        """
    )
    st.markdown(
        """
        <div class="backend-flow">
          <div class="backend-card"><div class="backend-index">BACKEND 01</div><div class="backend-title">Validate & normalize</div><div class="backend-desc">Periksa input positif dan ubah waktu field menjadi tᴅ.</div></div>
          <div class="backend-card"><div class="backend-index">BACKEND 02</div><div class="backend-title">Laplace–Bessel</div><div class="backend-desc">Bangun titik u dan evaluasi p̄ᴅ(u) dengan rasio K₀/K₁.</div></div>
          <div class="backend-card"><div class="backend-index">BACKEND 03</div><div class="backend-title">Gaver–Stehfest</div><div class="backend-desc">Jumlahkan Vⱼp̄ᴅ(uⱼ) untuk memperoleh pᴅ(tᴅ).</div></div>
          <div class="backend-card"><div class="backend-index">BACKEND 04</div><div class="backend-title">Field output</div><div class="backend-desc">Konversi ke Δp, pᵥf, derivative, tabel, dan grafik.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    backend_map = pd.DataFrame(
        [
            ("app.py", "Sidebar dan session_state", "Membaca k, ϕ, cₜ, μ, B, h, rᵥ, q, pᵢ, t, dan N."),
            ("app.py", "ReservoirInputs", "Mengemas input agar nama dan satuannya eksplisit."),
            ("pressure_model.py", "dimensionless_time", "Mengubah time field menjadi tᴅ."),
            ("pressure_model.py", "stehfest_weights", "Menghitung bobot Vⱼ untuk N genap."),
            ("pressure_model.py", "laplace_wellbore_pressure", "Mengevaluasi solusi Bessel pada Laplace space."),
            ("pressure_model.py", "invert_pressure_gaver_stehfest", "Melakukan inversi untuk seluruh tᴅ secara vectorized."),
            ("pressure_model.py", "simulate_drawdown", "Menghasilkan Δp, pᵥf, derivative, plateau, dan Cᴅ."),
            ("app.py", "Plotly + DataFrame", "Menyajikan tiga chart, tabel, serta CSV."),
        ],
        columns=["Lokasi", "Komponen", "Tanggung jawab"],
    )
    st.dataframe(backend_map, use_container_width=True, hide_index=True)

    with st.expander("Backend A · UI membentuk time array dan paket input", expanded=True):
        st.code(
            """# app.py — HIGHLIGHT: satu sumber input untuk seluruh solver
inputs = ReservoirInputs(
    permeability_md=k,
    porosity=phi,
    total_compressibility_psi_inv=ct,
    viscosity_cp=mu,
    formation_volume_factor_rb_stb=bo,
    thickness_ft=h,
    wellbore_radius_ft=rw,
    rate_stb_day=q,
    initial_pressure_psi=pi,
)

# Geometric spacing wajib untuk rentang log yang lebar.
time_hours = np.geomspace(t_min, t_max, points)
result = simulate_drawdown(time_hours, inputs, n_terms=n_terms)""",
            language="python",
        )
        st.markdown(
            """
            `ReservoirInputs` adalah frozen dataclass: setelah dibuat, paket parameter tidak berubah diam-diam di tengah perhitungan. `np.geomspace` dipilih karena well test mencakup banyak dekade waktu; linear spacing akan membuang terlalu banyak titik di late-time dan terlalu sedikit di early-time.
            """
        )

    with st.expander("Backend B · Konversi waktu field menjadi dimensionless time"):
        st.code(
            """def dimensionless_time(time_hours, inputs):
    return (
        0.0002637
        * inputs.permeability_md
        * np.asarray(time_hours, dtype=float)
        / (
            inputs.porosity
            * inputs.viscosity_cp
            * inputs.total_compressibility_psi_inv
            * inputs.wellbore_radius_ft**2
        )
    )""",
            language="python",
        )
        st.latex(r"t_D=\frac{0.0002637\,k\,t}{\phi\mu c_t r_w^2}")
        st.markdown(
            r"""
            **Highlight penting:** $k$ dan t berada di pembilang karena permeability tinggi atau waktu lebih lama membuat disturbance menjangkau lebih jauh. $\phi$, $\mu$, $c_t$, dan $r_w^2$ berada di penyebut karena storage/viscous resistance/radius scale memperlambat atau menormalkan respons. Konstanta 0.0002637 mengonsistenkan field units yang digunakan.
            """
        )

    with st.expander("Backend C · Menghitung bobot Gaver–Stehfest Vⱼ"):
        st.code(
            """def stehfest_weights(n_terms=12):
    if n_terms < 6 or n_terms > 18 or n_terms % 2:
        raise ValueError("n_terms must be an even integer")

    half = n_terms // 2
    weights = np.zeros(n_terms)
    for k in range(1, n_terms + 1):
        total = 0.0
        for j in range((k + 1) // 2, min(k, half) + 1):
            numerator = (j**half) * factorial(2 * j)
            denominator = (
                factorial(half-j) * factorial(j) * factorial(j-1)
                * factorial(k-j) * factorial(2*j-k)
            )
            total += numerator / denominator
        # HIGHLIGHT: tanda bergantian menyebabkan cancellation.
        weights[k-1] = ((-1) ** (k + half)) * total
    return weights""",
            language="python",
        )
        st.markdown(
            """
            N wajib genap. Bobot dapat sangat besar dan tandanya bergantian; pᴅ muncul dari cancellation presisi antara suku positif dan negatif. Karena itu menambah N tanpa batas tidak menjamin hasil lebih baik. UI membatasi pilihan praktis 8–14, sedangkan backend menolak nilai di luar 6–18.
            """
        )

    with st.expander("Backend D · Solusi Bessel di Laplace space dan alasan memakai kve"):
        st.code(
            """def laplace_wellbore_pressure(u):
    u_array = np.asarray(u, dtype=float)
    if np.any(u_array <= 0):
        raise ValueError("Laplace variable u must be positive")

    root_u = np.sqrt(u_array)

    # HIGHLIGHT: scaled Bessel menghindari exponential underflow.
    bessel_ratio = kve(0, root_u) / kve(1, root_u)

    return bessel_ratio / np.power(u_array, 1.5)""",
            language="python",
        )
        st.latex(r"\bar p_D(u)=\frac{K_0(\sqrt u)}{u^{3/2}K_1(\sqrt u)}")
        st.markdown(
            r"""
            `scipy.special.kve(v, x)` menghitung scaled modified Bessel $e^xK_v(x)$. Karena pembilang dan penyebut memiliki exponential scale yang sama, faktor tersebut saling habis dalam rasio. Hasil fisiknya tetap $K_0/K_1$, tetapi komputasi tidak mudah menjadi 0/0 ketika $\sqrt u$ besar.
            """
        )

    with st.expander("Backend E · Inversi seluruh waktu secara vectorized"):
        st.code(
            """def invert_pressure_gaver_stehfest(dimensionless_time, n_terms=12):
    td = np.atleast_1d(np.asarray(dimensionless_time, dtype=float))
    weights = stehfest_weights(n_terms)                 # shape (N,)
    indices = np.arange(1, n_terms + 1, dtype=float)    # shape (N,)

    # HIGHLIGHT: broadcasting membentuk satu grid Laplace untuk semua waktu.
    u = np.log(2.0) * indices[None, :] / td[:, None]    # shape (M, N)
    pbar = laplace_wellbore_pressure(u)                 # shape (M, N)

    pd = (np.log(2.0) / td) * np.sum(
        weights[None, :] * pbar, axis=1
    )                                                   # shape (M,)
    return pd""",
            language="python",
        )
        st.markdown(
            """
            Jika ada M titik waktu dan N suku Stehfest, `u` adalah matriks M×N. Setiap baris adalah satu $t_D$, setiap kolom adalah satu $u_j$. NumPy broadcasting menghindari loop Python per waktu sehingga solver lebih cepat dan struktur matematikanya terlihat langsung. Kompleksitas utamanya sekitar O(MN), kecil untuk default M=160 dan N=12.
            """
        )

    with st.expander("Backend F · Konversi pᴅ ke pressure field dan derivative"):
        st.code(
            """td = dimensionless_time(time, inputs)
pd = invert_pressure_gaver_stehfest(td, n_terms=n_terms)

pressure_scale = (
    141.2 * inputs.rate_stb_day
    * inputs.formation_volume_factor_rb_stb
    * inputs.viscosity_cp
    / (inputs.permeability_md * inputs.thickness_ft)
)

# HIGHLIGHT: dimensionless → psi → flowing pressure.
delta_pressure = pressure_scale * pd
flowing_pressure = inputs.initial_pressure_psi - delta_pressure

# HIGHLIGHT: derivative numerik terhadap ln(t), bukan terhadap t linear.
derivative = np.gradient(delta_pressure, np.log(time), edge_order=2)
plateau = 0.5 * pressure_scale""",
            language="python",
        )
        st.markdown(
            r"""
            Pressure scale $141.2qB\mu/(kh)$ mengubah pᴅ menjadi psi. Drawdown didefinisikan $\Delta p=p_i-p_{wf}$, sehingga $p_{wf}=p_i-\Delta p$. Derivative dihitung terhadap $\ln t$ agar plateau radial mudah dikenali. Pada model ideal ini plateau teoritisnya $\tfrac12$ pressure scale.
            """
        )

    with st.expander("Backend G · Guardrail, error path, dan validasi hasil"):
        st.markdown(
            r"""
            **Guardrail yang sudah ada:**
            - time harus array 1-D, minimal tiga titik, seluruhnya positif;
            - semua rock/fluid/well inputs harus positif;
            - u dan tᴅ harus finite serta positif;
            - N harus genap dan berada dalam rentang aman;
            - UI menghentikan perhitungan jika $t_{max}\le t_{min}$;
            - UI memberi warning bila $p_{wf}\le0$ psi.

            **Validasi berlapis:** unit test membandingkan bobot/solusi; late-time approximation memeriksa tren; N=8/10/12/14 dapat dibandingkan; derivative plateau diperiksa terhadap $0.5×$ pressure scale; dan Streamlit AppTest memastikan seluruh tab dapat dirender tanpa exception.

            **Alur error:** backend melempar `ValueError`/`FloatingPointError`; `app.py` menangkapnya, menampilkan pesan yang dapat dibaca pengguna, lalu `st.stop()` agar grafik salah tidak diteruskan.
            """
        )

    with st.expander("1 · Mengapa konstanta Bessel berbentuk seperti itu?", expanded=True):
        st.markdown(
            "Setelah transformasi Laplace, persamaan difusivitas radial menjadi ODE modified Bessel. Solusi umumnya adalah kombinasi fungsi $I_0$ dan $K_0$:"
        )
        st.latex(r"\frac{1}{r_D}\frac{d}{dr_D}\left(r_D\frac{d\bar p_D}{dr_D}\right)-u\bar p_D=0")
        st.latex(r"\bar p_D=A\,I_0(\sqrt{u}\,r_D)+B\,K_0(\sqrt{u}\,r_D)")
        st.markdown(
            r"""
            - $I_0$ membesar secara eksponensial ketika $r_D\to\infty$. Agar pressure disturbance tetap nol di infinity, koefisien **A harus nol**.
            - $K_0$ meluruh menuju nol, sehingga sesuai untuk reservoir infinite.
            - Kondisi laju konstan pada dinding sumur menentukan koefisien B. Identitas turunan $dK_0(x)/dx=-K_1(x)$ menghasilkan faktor $K_1$ pada penyebut.
            - Jadi “konstanta Bessel” bukan angka tebakan; nilainya muncul dari **initial condition + inner boundary + outer boundary**.
            """
        )
        st.latex(r"\bar p_D(u)=\frac{K_0(\sqrt{u})}{u^{3/2}K_1(\sqrt{u})}")

    with st.expander("2 · Bagaimana Gaver–Stehfest menginversi Laplace?"):
        st.markdown(
            "Gaver–Stehfest mendekati integral inversi Laplace menggunakan sejumlah evaluasi fungsi di sumbu real positif. Untuk setiap $t_D$, aplikasi mengambil N titik Laplace berikut:"
        )
        st.latex(r"u_j=\frac{j\ln 2}{t_D},\qquad j=1,2,\ldots,N")
        st.latex(r"p_D(t_D)\approx\frac{\ln2}{t_D}\sum_{j=1}^{N}V_j\,\bar p_D(u_j)")
        st.markdown(
            r"""
            1. Pilih N genap; aplikasi menggunakan pilihan 8–14 dan default 12.
            2. Evaluasi solusi Bessel $\bar p_D(u_j)$ di setiap titik.
            3. Kalikan dengan bobot $V_j$ yang bertanda positif-negatif secara bergantian.
            4. Jumlahkan seluruh suku; cancellation antar-suku merekonstruksi nilai waktu nyata.

            N yang lebih besar **tidak selalu lebih akurat**, karena bobot menjadi sangat besar dan sensitif terhadap round-off. Karena itu rasio Bessel dievaluasi dengan fungsi terskala dan N dibatasi.
            """
        )

    with st.expander("3 · Mengapa infinite boundary menghasilkan nol?"):
        st.markdown(
            """
            Nol yang dimaksud bukan berarti tekanan reservoir nol. Variabel $p_D$ adalah **pressure disturbance/pressure drop yang dinormalisasi**. Sangat jauh dari sumur, gangguan belum terasa sehingga tekanannya masih sama dengan tekanan awal; akibatnya pressure drop menuju nol:
            """
        )
        st.latex(r"\lim_{r_D\to\infty}p_D(r_D,t_D)=0")
        boundary_table = pd.DataFrame(
            [
                ("Infinite acting", r"$p_D\to0$ saat $r_D\to\infty$", "Tidak ada respons boundary selama waktu pengamatan."),
                ("Closed / no-flow", r"$\partial p_D/\partial r_D=0$ pada $r_{eD}$", "Derivative naik pada late time karena fluida terkurung."),
                ("Constant pressure", "$p_D=0$ pada radius luar finite", "Derivative turun menuju nol karena pressure support."),
                ("Leaky / mixed", "Hubungan pressure dan flux pada boundary", "Respons berada di antara closed dan constant-pressure."),
            ],
            columns=["Model boundary", "Kondisi matematis", "Ciri respons"],
        )
        st.dataframe(boundary_table, use_container_width=True, hide_index=True)
        st.caption("Jadi boundary lain tidak sekadar diberi ‘angka selain 0’; yang berubah adalah jenis kondisi matematisnya: nilai pressure, gradien/flux, atau kombinasi keduanya.")

    with st.expander("4 · Bagaimana Laplace mengubah real space ke Laplace space?"):
        st.markdown("Transformasi Laplace mengintegralkan fungsi waktu dengan kernel eksponensial. Istilah ‘real space’ pada tugas sebaiknya dipahami sebagai domain waktu nyata; koordinat radialnya sendiri tidak ditransformasikan:")
        st.latex(r"\bar f(u)=\mathcal{L}\{f(t)\}=\int_0^\infty e^{-ut}f(t)\,dt")
        st.markdown(
            "Keuntungan utamanya adalah turunan waktu berubah menjadi bentuk aljabar. Dengan initial pressure disturbance nol:"
        )
        st.latex(r"\mathcal{L}\left\{\frac{\partial p_D}{\partial t_D}\right\}=u\bar p_D-p_D(r_D,0)=u\bar p_D")
        st.markdown(
            "Karena itu PDE yang melibatkan radius dan waktu berubah menjadi ODE yang hanya melibatkan radius. ODE ini lebih mudah diselesaikan dengan modified Bessel functions. Setelah kondisi batas diterapkan, Gaver–Stehfest mengembalikan solusi ke domain waktu."
        )

    with st.expander("5 · Kombinasi model yang umum pada well testing"):
        model_table = pd.DataFrame(
            [
                ("Wellbore storage + skin", "Early time", "Unit-slope storage lalu transisi skin."),
                ("Homogeneous radial flow", "Middle time", "Derivative plateau; dasar estimasi kh."),
                ("Dual porosity / naturally fractured", "Transition", "Dip/valley derivative akibat interporosity flow."),
                ("Finite-conductivity fracture", "Early–middle", "Bilinear atau linear flow sebelum radial flow."),
                ("Composite reservoir", "Middle–late", "Mobilitas/storativitas berubah antar-zona."),
                ("Single sealing fault", "Late time", "Derivative cenderung naik menuju dua kali plateau."),
                ("Constant-pressure boundary", "Late time", "Pressure derivative turun menuju nol."),
                ("Closed reservoir", "Late time", "Boundary-dominated/pseudosteady response."),
            ],
            columns=["Kombinasi model", "Region waktu", "Ciri umum"],
        )
        st.dataframe(model_table, use_container_width=True, hide_index=True)
        st.markdown(
            "Model pada aplikasi ini sengaja mengambil kasus dasar paling bersih: **no storage + no skin + homogeneous reservoir + infinite boundary**. Karena itu respons utama yang diharapkan adalah infinite-acting radial flow tanpa distorsi early-time dan late-time."
        )

    with st.expander("6 · Cara membaca empat output grafik"):
        st.markdown(
            r"""
            - **Cartesian $p_{wf}$ vs t:** memperlihatkan tekanan aktual dan total drawdown dengan jelas, tetapi early-time terkompres di dekat sumbu-y.
            - **Semi-log $p_{wf}$ vs t:** log pada sumbu waktu membuka rentang waktu yang lebar. Bagian radial-flow ideal mendekati garis lurus.
            - **Log–log pressure:** memakai $\Delta p=p_i-p_{wf}$, bukan tekanan absolut, agar perubahan skala terlihat jelas.
            - **Log–log derivative:** derivative terhadap $\ln(t)$ menonjolkan perubahan rezim aliran. Plateau menandakan radial flow.
            - Tombol **Pressure / Derivative / Keduanya** pada Dashboard membantu memisahkan atau menumpuk kurva diagnostik.
            """
        )
        st.latex(r"\frac{d\Delta p}{d\ln t}=t\frac{d\Delta p}{dt}")

    with st.expander("7 · Mengapa parameter random-nya masih masuk akal?"):
        random_ranges = pd.DataFrame(
            [
                ("k", "50–220 mD", "Log-uniform", "Mencakup reservoir moderate hingga good quality tanpa mendominasi nilai tinggi."),
                ("ϕ", "0.13–0.27", "Uniform", "Rentang porosity clastic/carbonate produktif yang masih realistis."),
                ("cₜ", "0.8×10⁻⁵–2.8×10⁻⁵ psi⁻¹", "Uniform", "Orde slightly-compressible reservoir system."),
                ("μ", "0.7–2.2 cP", "Uniform", "Fluida liquid ringan sampai moderately viscous."),
                ("B", "1.05–1.32 rb/STB", "Uniform", "Formation volume factor liquid yang moderat."),
                ("h", "40–110 ft", "Uniform", "Net thickness yang cukup untuk test responsif."),
                ("rᵥ", "0.25–0.50 ft", "Uniform", "Radius wellbore tipikal completion."),
                ("q", "250–700 STB/hari", "Uniform", "Rate drawdown terukur tanpa sengaja dibuat ekstrem."),
                ("pᵢ", "3,000–4,800 psi", "Uniform", "Initial pressure tipikal reservoir subsurface."),
            ],
            columns=["Parameter", "Rentang generator", "Distribusi", "Alasan"],
        )
        st.dataframe(random_ranges, use_container_width=True, hide_index=True)
        st.markdown(
            r"""
            Permeability diacak secara **log-uniform** karena k secara alamiah dapat berubah beberapa orde dan distribusinya sering lebih representatif pada skala log. Parameter lain memakai uniform dalam rentang sempit untuk membuat skenario demonstrasi yang stabil.

            Random bukan berarti bebas. Setiap kombinasi tetap diperiksa secara fisik melalui $p_{wf}=p_i-\Delta p$. Jika $p_{wf}\le0$, aplikasi memberi warning karena solusi matematis masih dapat dihitung tetapi skenario produksinya tidak realistis. C dan S selalu nol karena merupakan asumsi model, sedangkan t/titik numerik tetap dikontrol pengguna agar perbandingan skenario konsisten.
            """
        )

    with st.expander("8 · Pseudocode, validasi numerik, dan keterbatasan PR 3"):
        st.code(
            """for setiap t:
    hitung tD dari k, phi, mu, ct, rw
    for j = 1 ... N:
        uj = j * ln(2) / tD
        hitung pbarD(uj) dari rasio scaled Bessel K0/K1
        hitung term_j = Vj * pbarD(uj)
    pD(tD) = ln(2) / tD * sum(term_j)
    delta_p = 141.2 * q * B * mu / (k * h) * pD
    pwf = pi - delta_p
hitung derivative numerik terhadap ln(t)
plot dan ekspor tabel""",
            language="text",
        )
        st.markdown(
            r"""
            **Validasi yang sebaiknya dijelaskan:**
            - Ulangi perhitungan dengan N = 8, 10, 12, dan 14. Hasil yang sehat relatif konvergen; perbedaan besar menandakan cancellation/round-off atau rentang $t_D$ ekstrem.
            - Pastikan $t>0$, N genap, properties positif, dan urutan time monoton.
            - Bandingkan late-time $p_D$ dengan kecenderungan solusi radial yang linear terhadap $\ln(t_D)$.
            - Derivative radial ideal mendekati $\tfrac12$ dari pressure scale dimensionless; dalam field pressure, plateau sekitar $70.6qB\mu/(kh)$.
            - Periksa $p_{wf}=p_i-\Delta p$, satuan field, serta apakah drawdown dan rate masuk akal.
            - Gunakan scaled Bessel function agar $K_0$ dan $K_1$ tidak sama-sama underflow pada argumen besar; exponential scale saling menghilangkan dalam rasionya.

            **Keterbatasan:** Gaver–Stehfest tidak ideal untuk fungsi yang sangat oscillatory, discontinuous, atau membutuhkan presisi ekstrem. Model ini juga tidak mencakup wellbore storage, skin, fracture, multiphase, non-Darcy flow, changing rate, finite boundary, anisotropy, partial penetration, maupun heterogeneity. Karena itu grafik adalah baseline teoritis, bukan pengganti history matching data lapangan.
            """
        )

    st.markdown("### Jawaban lengkap untuk penjelasan tulisan tangan PR 3")
    st.markdown(
        """
        Bagian ini menyatukan seluruh jawaban dalam urutan yang dapat langsung dijadikan struktur tulisan tangan. Baca sebagai satu cerita: **pressure disturbance dibuat di sumur → diffusivity equation menjelaskan penyebarannya → Laplace menyederhanakan waktu → Bessel menyelesaikan geometri radial → Gaver–Stehfest mengembalikan solusi ke waktu nyata → hasil dikonversi menjadi pressure lapangan dan grafik.**
        """
    )

    with st.expander("Jawaban 1 · Asumsi fisik dan definisi setiap parameter", expanded=True):
        st.markdown(
            """
            **Gambaran fisiknya:** terdapat satu sumur vertikal yang menembus penuh reservoir berbentuk sangat luas. Sebelum produksi, seluruh reservoir memiliki pressure seragam $p_i$. Pada $t=0$, sumur diproduksikan dengan rate konstan q. Pressure di dekat sumur turun, lalu gangguan pressure merambat radial ke luar.

            **Asumsi model:**
            1. sumur vertikal dan fully penetrating;
            2. reservoir homogen dan isotropik—k serta ϕ sama di semua lokasi;
            3. fluida satu fasa, slightly compressible, viscosity konstan, dan isothermal;
            4. Darcy flow berlaku; tidak ada turbulent/non-Darcy flow;
            5. rate q berubah sebagai step dari nol menjadi konstan;
            6. initial pressure seragam;
            7. outer boundary infinite sehingga disturbance belum mencapai batas;
            8. no wellbore storage: C=0 dan Cᴅ=0;
            9. no skin: S=0;
            10. gravity, capillary, fracture, heterogeneity, dan partial penetration diabaikan.

            **Arti parameter utama:**
            - **k (mD):** permeability; kemudahan batuan meloloskan fluida.
            - **ϕ:** porosity; fraksi bulk volume yang menjadi ruang pori.
            - **cₜ (psi⁻¹):** total compressibility batuan+fluida; besarnya storage elastik.
            - **μ (cP):** viscosity; tahanan internal fluida terhadap aliran.
            - **B (rb/STB):** formation volume factor; penghubung volume reservoir dan surface.
            - **h (ft):** net thickness yang berkontribusi terhadap aliran.
            - **rᵥ (ft):** wellbore radius; panjang acuan untuk nondimensionalization.
            - **q (STB/day):** production rate konstan.
            - **pᵢ dan pᵥf (psi):** initial pressure dan flowing bottomhole pressure.
            - **Δp=pᵢ−pᵥf:** pressure drop akibat produksi.
            - **t dan tᴅ:** waktu lapangan dan dimensionless time.
            - **pᴅ:** dimensionless pressure; respons yang belum memiliki satuan psi.
            - **C, Cᴅ, S:** wellbore storage, dimensionless storage, dan skin; semuanya nol pada tugas.
            - **rᴅ=r/rᵥ:** dimensionless radius.
            - **u:** variabel Laplace; bukan waktu, tetapi koordinat pada Laplace space.
            - **N:** jumlah suku genap dalam inversi Gaver–Stehfest.

            **Kalimat inti:** k mempercepat komunikasi pressure, sedangkan ϕ, μ, dan cₜ memperlambatnya. q, B, μ, k, dan h mengontrol besar drawdown dalam psi.
            """
        )

    with st.expander("Jawaban 2 · Penurunan diffusivity equation dimensionless"):
        st.markdown("Mulai dari radial diffusivity equation dimensional yang berasal dari mass conservation + Darcy + compressibility:")
        st.latex(r"\frac{\partial^2p}{\partial r^2}+\frac{1}{r}\frac{\partial p}{\partial r}=\frac{\phi\mu c_t}{k}\frac{\partial p}{\partial t}")
        st.markdown("Definisikan variabel dimensionless:")
        st.latex(r"r_D=\frac{r}{r_w},\qquad t_D=\frac{0.0002637\,k\,t}{\phi\mu c_t r_w^2}")
        st.latex(r"p_D=\frac{kh}{141.2\,qB\mu}(p_i-p)")
        st.markdown(
            r"""
            Gunakan chain rule: turunan terhadap r membawa faktor $1/r_w$, sedangkan turunan terhadap t membawa faktor yang berasal dari definisi $t_D$. Pressure scale pada pᴅ muncul di semua suku dan dapat dicoret. Faktor rock/fluid juga terserap ke $t_D$. Hasilnya tidak lagi membawa satuan lapangan:
            """
        )
        st.latex(r"\boxed{\frac{1}{r_D}\frac{\partial}{\partial r_D}\left(r_D\frac{\partial p_D}{\partial r_D}\right)=\frac{\partial p_D}{\partial t_D}}")
        st.markdown(
            """
            **Mengapa dimensionless berguna?** Satu solusi matematika dapat dipakai untuk banyak reservoir. Perbedaan k, ϕ, μ, cₜ, rᵥ, q, B, dan h baru dimasukkan saat mengubah waktu serta pressure ke/dari bentuk dimensionless.

            **Kalimat inti:** nondimensionalization bukan menghilangkan fisika; ia mengumpulkan sifat reservoir ke dalam kelompok skala tᴅ dan pᴅ agar persamaannya universal.
            """
        )

    with st.expander("Jawaban 3 · Transformasi Laplace mengubah PDE menjadi ODE"):
        st.markdown(
            """
            PDE sulit karena pᴅ bergantung pada radius **dan** waktu. Transformasi Laplace hanya diterapkan pada variabel waktu tᴅ; radius tetap berada di real space.
            """
        )
        st.latex(r"\bar p_D(r_D,u)=\int_0^\infty e^{-ut_D}p_D(r_D,t_D)\,dt_D")
        st.markdown("Sifat yang paling penting adalah transformasi turunan waktu:")
        st.latex(r"\mathcal L\left\{\frac{\partial p_D}{\partial t_D}\right\}=u\bar p_D-p_D(r_D,0)")
        st.markdown("Initial pressure disturbance nol, sehingga $p_D(r_D,0)=0$. Maka bagian kanan menjadi $u\bar p_D$ dan PDE berubah menjadi:")
        st.latex(r"\boxed{\frac{1}{r_D}\frac{d}{dr_D}\left(r_D\frac{d\bar p_D}{dr_D}\right)-u\bar p_D=0}")
        st.markdown(
            """
            Sekarang tidak ada turunan terhadap waktu. Persamaan hanya berupa ODE radial dengan parameter u. Setiap nilai u mewakili satu evaluasi di Laplace space.

            **Analogi sederhana:** transformasi Laplace mengubah masalah “bagaimana kurva berubah sepanjang waktu” menjadi sekumpulan masalah aljabar/radial yang lebih mudah diselesaikan. Setelah itu inverse Laplace diperlukan untuk kembali ke waktu nyata.
            """
        )

    with st.expander("Jawaban 4 · Solusi I₀/K₀ dan alasan I₀ dieliminasi"):
        st.markdown("ODE tersebut adalah modified Bessel equation orde nol. Solusi umumnya:")
        st.latex(r"\bar p_D=A\,I_0(\sqrt u\,r_D)+B\,K_0(\sqrt u\,r_D)")
        bessel_behavior = pd.DataFrame(
            [
                ("I₀", "Membesar kira-kira seperti eˣ/√x", "Tidak cocok di reservoir infinite karena pressure disturbance akan meledak pada rᴅ→∞"),
                ("K₀", "Meluruh kira-kira seperti e⁻ˣ/√x", "Cocok karena disturbance harus menghilang jauh dari sumur"),
            ],
            columns=["Fungsi", "Perilaku saat argumen besar", "Konsekuensi fisik"],
        )
        st.dataframe(bessel_behavior, use_container_width=True, hide_index=True)
        st.markdown(
            r"""
            Outer boundary infinite meminta $\bar p_D\to0$ saat $r_D\to\infty$. Karena I₀ justru tumbuh tanpa batas, satu-satunya cara memenuhi boundary tersebut adalah **A=0**. Jadi solusi yang tersisa:
            """
        )
        st.latex(r"\bar p_D=B\,K_0(\sqrt u\,r_D)")
        st.markdown("**Kalimat inti:** I₀ dibuang bukan karena fungsi itu salah, tetapi karena tidak sesuai dengan kondisi fisik reservoir infinite.")

    with st.expander("Jawaban 5 · Kondisi batas laju konstan dan bentuk p̄ᴅ(u)"):
        st.markdown(
            """
            Tiga kondisi diperlukan:
            1. **Initial:** sebelum produksi, disturbance nol: $p_D(r_D,0)=0$.
            2. **Outer infinite:** jauh dari sumur, disturbance nol: $p_D(∞,t_D)=0$.
            3. **Inner constant rate:** flux dimensionless di dinding sumur rᴅ=1 bernilai step konstan.
            """
        )
        st.latex(r"-\left.\frac{\partial p_D}{\partial r_D}\right|_{r_D=1}=1")
        st.markdown("Transformasi Laplace dari step konstan adalah 1/u:")
        st.latex(r"-\left.\frac{d\bar p_D}{dr_D}\right|_{r_D=1}=\frac{1}{u}")
        st.markdown("Karena $dK_0(x)/dx=-K_1(x)$, turunan solusi yang tersisa adalah:")
        st.latex(r"\frac{d\bar p_D}{dr_D}=-B\sqrt u\,K_1(\sqrt u\,r_D)")
        st.markdown("Substitusi rᴅ=1 ke boundary flux memberi:")
        st.latex(r"B\sqrt u\,K_1(\sqrt u)=\frac1u\quad\Rightarrow\quad B=\frac{1}{u^{3/2}K_1(\sqrt u)}")
        st.markdown("Pressure Laplace di wellbore rᴅ=1 akhirnya:")
        st.latex(r"\boxed{\bar p_D(u)=\frac{K_0(\sqrt u)}{u^{3/2}K_1(\sqrt u)}}")
        st.markdown("**Kalimat inti:** K₀ berasal dari outer infinite boundary, sedangkan K₁ dan u³ᐟ² pada penyebut muncul saat constant-rate inner boundary diterapkan.")

    with st.expander("Jawaban 6 · Titik uⱼ, bobot Vⱼ, cancellation, dan pemilihan N"):
        st.markdown("Untuk setiap tᴅ, Gaver–Stehfest memilih N titik Laplace:")
        st.latex(r"u_j=\frac{j\ln2}{t_D},\qquad j=1,2,\ldots,N")
        st.markdown("Lalu evaluasi solusi Bessel di setiap titik dan jumlahkan:")
        st.latex(r"p_D(t_D)\approx\frac{\ln2}{t_D}\sum_{j=1}^{N}V_j\bar p_D(u_j)")
        st.markdown(
            """
            **Makna langkahnya:**
            1. satu tᴅ menghasilkan N nilai uⱼ;
            2. masing-masing uⱼ dimasukkan ke solusi Bessel p̄ᴅ;
            3. setiap hasil dikalikan bobot Vⱼ;
            4. seluruh suku dijumlahkan untuk merekonstruksi pᴅ pada waktu tersebut;
            5. proses diulang untuk semua titik waktu.

            **Cancellation:** Vⱼ bergantian positif dan negatif serta dapat bernilai sangat besar. Hasil akhir justru berasal dari selisih angka-angka besar tersebut. Ini normal, tetapi sensitif terhadap round-off.

            **Mengapa N genap?** Rumus Stehfest dibangun untuk jumlah suku genap. Aplikasi memakai N=8,10,12,14; default 12 adalah kompromi accuracy dan numerical stability. N lebih besar belum tentu lebih baik karena cancellation makin ekstrem.

            **Cara mengecek:** hitung ulang dengan beberapa N. Jika kurva hampir sama, hasil cukup stabil. Jika berubah besar, rentang tᴅ atau precision perlu diperiksa.
            """
        )
        st.markdown("Bobot Vⱼ bukan angka yang dipilih sembarangan — bentuk tertutupnya:")
        st.latex(
            r"V_j=(-1)^{j+N/2}\sum_{m=\lceil (j+1)/2\rceil}^{\min(j,N/2)}"
            r"\frac{m^{N/2}(2m)!}{(N/2-m)!\,m!\,(m-1)!\,(j-m)!\,(2m-j)!}"
        )
        st.markdown("Contoh konkret untuk N=8 — angka ini dihitung langsung oleh `stehfest_weights(8)`, bukan ditulis manual, agar sesuai dengan yang sungguh-sungguh dipakai solver:")
        n_demo = 8
        demo_weights = stehfest_weights(n_demo)
        demo_table = pd.DataFrame(
            {
                "j": np.arange(1, n_demo + 1),
                "Vj": demo_weights,
                "tanda": np.where(demo_weights >= 0, "+", "−"),
            }
        )
        st.dataframe(demo_table.style.format({"Vj": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.markdown("Perhatikan pola selang-seling tanda dan lonjakan magnitude di tengah (j=4–5) — itulah sumber cancellation yang disebutkan di atas.")

        growth_ns = [8, 10, 12, 14, 16, 18]
        growth_table = pd.DataFrame(
            {
                "N": growth_ns,
                "max |Vj|": [float(np.max(np.abs(stehfest_weights(n)))) for n in growth_ns],
            }
        )
        st.markdown(
            "**Kenapa N lebih besar tidak otomatis lebih akurat:** magnitude Vⱼ terbesar tumbuh sangat cepat "
            "terhadap N, padahal hasil akhir pᴅ tetap bernilai orde satuan. Artinya semakin besar N, semakin "
            "besar pula suku-suku yang harus saling meniadakan — presisi floating point ikut terkuras:"
        )
        st.dataframe(growth_table.style.format({"max |Vj|": "{:.3e}"}), use_container_width=True, hide_index=True)
        st.caption("Tabel Vⱼ, uⱼ, dan p̄ᴅ(uⱼ) yang dihitung langsung untuk skenario aktif tersedia di tab 'Metode numerik' → expander 'Lihat bukti perhitungan numerik Gaver–Stehfest'.")

    with st.expander("Jawaban 7 · Mengubah hasil inversi menjadi Δp dan pᵥf"):
        st.markdown("Gaver–Stehfest menghasilkan pᴅ tanpa satuan. Konversi ke pressure field:")
        st.latex(r"\Delta p=\frac{141.2\,qB\mu}{kh}\,p_D")
        st.latex(r"p_{wf}=p_i-\Delta p")
        st.markdown(
            fr"""
            Untuk skenario aktif, solver memperoleh pᴅ akhir **{result['dimensionless_pressure'][-1]:.5f}**, drawdown maksimum **{max_drop:,.2f} psi**, dan pᵥf akhir **{final_pwf:,.2f} psi** dari pᵢ **{pi:,.0f} psi**.

            **Arah pengaruh parameter:**
            - q, B, atau μ naik → pressure scale naik → drawdown lebih besar;
            - k atau h naik → kemampuan alir naik → drawdown lebih kecil;
            - pᵢ menggeser pressure absolut, tetapi tidak mengubah pᴅ.

            Derivative dihitung sebagai $d\Delta p/d\ln t=t\,d\Delta p/dt$. Untuk radial flow ideal, derivative mendekati plateau setengah pressure scale.
            """
        )

    with st.expander("Jawaban 8 · Cara menjelaskan dan menganotasi empat grafik"):
        graph_explanation = pd.DataFrame(
            [
                ("pᵥf vs t Cartesian", "x linear, y linear", "Pressure turun cepat di awal lalu lebih lambat", "Tandai pᵢ, pᵥf akhir, dan total drawdown"),
                ("pᵥf vs t semi-log", "x log, y linear", "Radial-flow response menjadi hampir straight line terhadap log time", "Tandai interval straight line dan slope"),
                ("Δp log–log", "x log, y log", "Memperlihatkan magnitude pressure change lintas dekade", "Bandingkan bentuk/slope dengan derivative"),
                ("Derivative log–log", "x log, y log", "Plateau horizontal = infinite-acting radial flow", "Tandai plateau; tidak ada late-time upturn/downturn"),
            ],
            columns=["Grafik", "Skala", "Yang terlihat", "Anotasi tulisan tangan"],
        )
        st.dataframe(graph_explanation, use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Arti absence of boundary effect:** selama rentang waktu simulasi, pressure disturbance belum merasakan outer boundary. Karena itu derivative tidak menunjukkan late-time upturn seperti closed/no-flow boundary dan tidak turun menuju nol seperti constant-pressure boundary. “Infinite” bukan berarti reservoir benar-benar tak berujung; artinya reservoir **berperilaku infinite selama waktu test**.

            Pada model no storage/no skin, early-time unit-slope storage dan skin transition memang tidak muncul. Karena tᴅ minimum skenario dapat sudah cukup besar, kurva derivative bisa langsung sangat dekat plateau.
            """
        )

        st.markdown("#### Bagaimana derivative berubah bentuk kalau ada boundary")
        st.markdown(
            """
            Aplikasi ini hanya memodelkan **infinite-acting** (plateau datar tanpa akhir), tetapi soal ujian
            sering meminta membedakan plateau itu dari tiga respons boundary yang paling umum. Kurva di bawah
            adalah **sketsa skematik** bentuk derivative log–log — bukan hasil solver — untuk melatih mata
            mengenali pola sebelum bertemu data lapangan.
            """
        )
        t_schematic = np.geomspace(0.05, 200, 400)
        t_boundary = 3.0
        plateau_level = 1.0
        transition = 1.0 / (1.0 + (t_boundary / t_schematic) ** 3)
        inf_acting_curve = np.full_like(t_schematic, plateau_level)
        sealing_fault_curve = plateau_level * (1.0 + transition)
        constant_pressure_curve = plateau_level * np.clip(1.0 - transition, 1e-3, None)
        closed_reservoir_curve = np.where(
            t_schematic <= t_boundary, plateau_level, plateau_level * (t_schematic / t_boundary)
        )
        boundary_fig = go.Figure()
        boundary_series = [
            ("Infinite-acting (dipakai aplikasi)", inf_acting_curve, "#176bce", "solid"),
            ("Sealing fault → menuju 2×plateau", sealing_fault_curve, "#b3541e", "dash"),
            ("Constant-pressure boundary → menuju 0", constant_pressure_curve, "#19a58b", "dash"),
            ("Closed reservoir (PSS) → unit slope", closed_reservoir_curve, "#7a3fa0", "dash"),
        ]
        for name, curve, color, dash in boundary_series:
            boundary_fig.add_trace(
                go.Scatter(x=t_schematic, y=curve, mode="lines", name=name, line=dict(color=color, width=3, dash=dash))
            )
        boundary_fig.add_vline(
            x=t_boundary, line_dash="dot", line_color="#8ea1b7",
            annotation_text="mulai terasa boundary", annotation_font=dict(size=10, color="#596b80"),
        )
        boundary_fig.update_layout(**chart_layout("Sketsa derivative: infinite-acting vs tiga jenis boundary", "waktu (skala log, ilustratif)", "derivative (skala log, ilustratif)", log_x=True, log_y=True))
        boundary_fig.update_layout(height=380)
        st.plotly_chart(boundary_fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG)
        st.markdown(
            """
            **Cara membaca sketsa ini di ujian:**
            - **Plateau tetap datar** selamanya → infinite-acting radial flow, persis seperti model di aplikasi ini.
            - **Naik menuju kira-kira dua kali plateau** → sealing fault (batas no-flow tunggal); faktor 2 muncul karena fault bekerja seperti sumur bayangan (image well) yang menggandakan efek drawdown.
            - **Turun menuju nol** → constant-pressure boundary (misal kontak aquifer aktif atau gas cap kuat) yang menahan penurunan pressure lebih lanjut.
            - **Naik dengan kemiringan satu (unit slope)** pada log–log setelah boundary time → reservoir tertutup memasuki pseudosteady-state, Δp mulai linear terhadap waktu.
            - Waktu saat kurva mulai menyimpang dari plateau berkorelasi dengan **jarak** ke boundary — semakin cepat menyimpang, semakin dekat boundary tersebut ke sumur.
            """
        )

    with st.expander("Jawaban 9 · Keterbatasan model dan kombinasi model nyata"):
        limitation_table = pd.DataFrame(
            [
                ("Wellbore storage + skin", "Early-time unit slope lalu transition", "Model tugas mengunci C=S=0"),
                ("Hydraulically fractured well", "Bilinear/linear flow sebelum radial", "Butuh fracture conductivity dan half-length"),
                ("Dual porosity", "Derivative valley/dip", "Butuh storativity dan interporosity parameter"),
                ("Sealing fault", "Late-time derivative menuju sekitar 2× plateau", "Butuh jarak dan orientasi fault"),
                ("Constant-pressure boundary", "Derivative turun menuju nol", "Butuh finite radius/boundary condition"),
                ("Closed reservoir", "Boundary-dominated atau pseudosteady response", "Butuh drainage geometry/area"),
                ("Multiphase/non-Darcy", "Rate-dependent dan mobility kompleks", "Tidak dapat diwakili single-phase Darcy"),
                ("Changing rate", "Superposition/convolution diperlukan", "Model tugas memakai satu constant rate"),
            ],
            columns=["Model nyata", "Ciri respons", "Mengapa belum ada di aplikasi"],
        )
        st.dataframe(limitation_table, use_container_width=True, hide_index=True)
        st.markdown(
            """
            **Keterbatasan numerik:** Gaver–Stehfest sensitif terhadap round-off, tidak ideal untuk fungsi oscillatory/discontinuous, dan N besar dapat memperburuk cancellation. Scaled Bessel mengurangi underflow tetapi tidak menghilangkan seluruh keterbatasan floating-point.

            **Kesimpulan akhir yang dapat ditulis:** model ini adalah baseline teoritis paling bersih untuk memahami radial diffusivity, Laplace–Bessel solution, dan numerical inversion. Data lapangan harus melalui QC, rate-history reconstruction, model selection, sensitivity, serta history matching. Hasil satu model tidak boleh dianggap unik tanpa dukungan log, core, PVT, completion, dan geologi.
            """
        )

    st.warning("Gunakan struktur sembilan jawaban ini sebagai panduan, lalu tulis ulang dengan bahasa sendiri. Yang dinilai bukan panjang rumusnya, tetapi apakah kamu dapat menjelaskan hubungan antara asumsi fisik, kondisi batas, solusi Bessel, inversi numerik, dan bentuk kurva.")

with tab_data:
    st.markdown("### Tabel hasil transformasi dan inversi")
    st.dataframe(
        df.style.format({"t (jam)": "{:.5g}", "tD": "{:.5e}", "CD": "{:.1f}", "pD": "{:.6f}", "Δp (psi)": "{:.3f}", "pwf (psi)": "{:.3f}", "dΔp/dln(t) (psi)": "{:.3f}"}),
        use_container_width=True,
        height=480,
    )
    st.download_button(
        "↓ Unduh hasil CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="radial_pta_results.csv",
        mime="text/csv",
    )
