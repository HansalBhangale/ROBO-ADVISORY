"""
============================================================================
ROBO-ADVISORY — STREAMLIT DASHBOARD
============================================================================
Unified interactive dashboard integrating all modules:
  Step 1: Risk Profiling (user questionnaire)
  Step 2: Risk Score Prediction
  Step 3: Stock Analysis (Technical + Fundamental)
  Step 4: Portfolio Allocation
  Step 5: Portfolio Report & Visualization

Run: streamlit run app.py
============================================================================
"""

import os
import sys
import glob
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MongoDB (optional — gracefully degrades)
try:
    from db import MongoDB
    _db = MongoDB()
except Exception:
    _db = None

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(BASE_DIR, 'risk prediction', 'risk_tolerance_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'technical analysis', 'scaler_v2.pkl')
ENSEMBLE_DIR = os.path.join(BASE_DIR, 'technical analysis', 'ensemble_models')
ENSEMBLE_INFO_PATH = os.path.join(BASE_DIR, 'technical analysis', 'sp500_ensemble_info.pkl')
ENSEMBLE_INFO_PATH_OLD = os.path.join(BASE_DIR, 'technical analysis', 'nifty50_integrated_ensemble_info.pkl')
FUND_DIR = os.path.join(BASE_DIR, 'fundamental analysis')
FUND_META_PATH = os.path.join(FUND_DIR, 'best_model_meta.json')

# ============================================================================
# CONSTANTS
# ============================================================================
SEQ_LENGTH = 60
MAX_HORIZON = 360
TECHNICAL_WEIGHT = 0.4
FUNDAMENTAL_WEIGHT = 0.6

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'ABBV',
    'MRK', 'PEP', 'KO', 'COST', 'LLY', 'WMT', 'BAC', 'CRM',
    'NFLX', 'AMD', 'ORCL', 'ADBE', 'NOW', 'UBER',
]

STOCK_NAMES = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft', 'AMZN': 'Amazon',
    'GOOGL': 'Alphabet', 'META': 'Meta Platforms', 'NVDA': 'NVIDIA',
    'TSLA': 'Tesla', 'BRK-B': 'Berkshire Hathaway', 'UNH': 'UnitedHealth',
    'JNJ': 'Johnson & Johnson', 'JPM': 'JPMorgan Chase', 'V': 'Visa',
    'PG': 'Procter & Gamble', 'MA': 'Mastercard', 'HD': 'Home Depot',
    'ABBV': 'AbbVie', 'MRK': 'Merck', 'PEP': 'PepsiCo',
    'KO': 'Coca-Cola', 'COST': 'Costco', 'LLY': 'Eli Lilly',
    'WMT': 'Walmart', 'BAC': 'Bank of America', 'CRM': 'Salesforce',
    'NFLX': 'Netflix', 'AMD': 'AMD', 'ORCL': 'Oracle',
    'ADBE': 'Adobe', 'NOW': 'ServiceNow', 'UBER': 'Uber',
}


# ============================================================================
# STUB CLASSES (for pickle deserialization)
# ============================================================================
import sklearn.preprocessing
class PCABasedRiskScorer:
    def __init__(self, df=None):
        self.df = df; self.pca = None; self.scaler = sklearn.preprocessing.StandardScaler(); self.feature_names = []
    def create_risk_score(self): pass
    def get_feature_loadings(self): pass

class EmpiricalCorrelationScorer:
    def __init__(self, df=None):
        self.df = df; self.weights = {}; self.correlations = {}
    def calculate_empirical_weights(self): pass
    def normalize_to_percentile(self, series): pass
    def create_risk_score(self): pass


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Robo-Advisory Portfolio System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS — PREMIUM FINANCE THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ===== KEYFRAME ANIMATIONS ===== */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(56,239,125,0.15), 0 4px 20px rgba(0,0,0,0.3); }
        50% { box-shadow: 0 0 30px rgba(56,239,125,0.25), 0 8px 30px rgba(0,0,0,0.4); }
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(102,126,234,0.3); }
        50% { border-color: rgba(240,147,251,0.5); }
    }
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    @keyframes tickerScroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes lineSweep {
        0% { width: 0; }
        100% { width: 100%; }
    }

    /* ===== GLOBAL ===== */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #0a0a1a;
    }

    /* ===== ANIMATED HEADER ===== */
    .main-header {
        background: linear-gradient(-45deg, #0f0c29, #1a0533, #302b63, #0f2027, #203a43);
        background-size: 400% 400%;
        animation: gradientShift 12s ease infinite;
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(102,126,234,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(240,147,251,0.06) 0%, transparent 40%),
                    radial-gradient(circle at 50% 80%, rgba(56,239,125,0.04) 0%, transparent 40%);
        pointer-events: none;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #F7971E 0%, #FFD200 40%, #38ef7d 100%);
        background-size: 200% auto;
        animation: gradientShift 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(160,174,192,0.9);
        font-size: 1rem;
        margin-top: 0.4rem;
        letter-spacing: 0.5px;
    }
    .main-header .subtitle-line {
        height: 2px;
        background: linear-gradient(90deg, #667eea, #f093fb, #38ef7d);
        border-radius: 2px;
        margin-top: 1rem;
        animation: lineSweep 2s ease-out forwards, gradientShift 3s ease infinite;
        background-size: 200% auto;
    }

    /* ===== STOCK TICKER MARQUEE ===== */
    .ticker-wrap {
        overflow: hidden;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 10px 0;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .ticker-content {
        display: inline-flex;
        animation: tickerScroll 30s linear infinite;
        gap: 2rem;
        white-space: nowrap;
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #a0aec0;
    }
    .ticker-item .tick-name { color: #fff; font-weight: 600; }
    .ticker-item .tick-up { color: #38ef7d; }
    .ticker-item .tick-down { color: #ff6b6b; }

    /* ===== GLASSMORPHISM CARDS ===== */
    .step-card {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeSlideUp 0.6s ease-out;
    }
    .step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(102,126,234,0.15);
        border-color: rgba(102,126,234,0.3);
    }
    .step-card h3 {
        background: linear-gradient(135deg, #FFD200, #F7971E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    /* ===== METRIC CARDS WITH GLOW ===== */
    .metric-card {
        background: linear-gradient(145deg, rgba(15,32,39,0.9), rgba(32,58,67,0.9), rgba(44,83,100,0.8));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.4rem 1rem;
        text-align: center;
        color: white;
        border: 1px solid rgba(56,239,125,0.15);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        animation: scaleIn 0.5s ease-out, pulseGlow 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
        animation: shimmer 3s infinite;
    }
    .metric-card:hover {
        transform: translateY(-3px) scale(1.02);
        border-color: rgba(56,239,125,0.4);
        box-shadow: 0 8px 30px rgba(56,239,125,0.2);
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: countUp 0.8s ease-out;
    }
    .metric-card .label {
        color: rgba(160,174,192,0.8);
        font-size: 0.8rem;
        margin-top: 0.4rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ===== WORKFLOW PIPELINE ===== */
    .workflow-step {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 700;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    .step-active {
        background: linear-gradient(135deg, #FFD200, #F7971E);
        color: #1a1a2e;
        box-shadow: 0 4px 15px rgba(255,210,0,0.3);
    }
    .step-done {
        background: rgba(56,239,125,0.15);
        color: #38ef7d;
        border: 1px solid rgba(56,239,125,0.4);
    }
    .step-pending {
        background: rgba(255,255,255,0.04);
        color: rgba(160,174,192,0.6);
        border: 1px solid rgba(255,255,255,0.08);
        animation: borderGlow 3s ease-in-out infinite;
    }
    .step-pending:hover {
        background: rgba(255,255,255,0.08);
        color: #a0aec0;
        transform: scale(1.05);
    }
    .step-arrow {
        color: rgba(102,126,234,0.5);
        font-size: 1.2rem;
        margin: 0 0.3rem;
        animation: float 2s ease-in-out infinite;
    }

    /* ===== MODULE CARDS ===== */
    .module-card {
        background: rgba(22, 33, 62, 0.5);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1.8rem;
        color: white;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .module-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        border-radius: 18px 18px 0 0;
    }
    .module-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
    }
    .module-card.risk::before { background: linear-gradient(90deg, #667eea, #764ba2); }
    .module-card.tech::before { background: linear-gradient(90deg, #11998e, #38ef7d); }
    .module-card.fund::before { background: linear-gradient(90deg, #F7971E, #FFD200); }
    .module-card:hover.risk { border-color: rgba(102,126,234,0.3); box-shadow: 0 20px 50px rgba(102,126,234,0.1); }
    .module-card:hover.tech { border-color: rgba(56,239,125,0.3); box-shadow: 0 20px 50px rgba(56,239,125,0.1); }
    .module-card:hover.fund { border-color: rgba(255,210,0,0.3); box-shadow: 0 20px 50px rgba(255,210,0,0.1); }
    .module-card h3 {
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }
    .module-card.risk h3 { background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .module-card.tech h3 { background: linear-gradient(135deg, #11998e, #38ef7d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .module-card.fund h3 { background: linear-gradient(135deg, #F7971E, #FFD200); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .module-card p { color: rgba(160,174,192,0.85); font-size: 0.9rem; line-height: 1.6; }
    .module-card .module-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }
    .module-card .module-stat {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #a0aec0;
        margin-top: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a, #0f0c29, #1a1a2e);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: white; }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(102,126,234,0.2);
    }

    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }

    /* ===== TABLE STYLING ===== */
    .dataframe { font-size: 0.85rem; }

    /* ===== SECTION HEADERS ===== */
    .stApp h3 {
        animation: fadeSlideUp 0.5s ease-out;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
        animation: fadeSlideUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    .section-header::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 16px 0 0 16px;
    }
    .section-header.risk::before { background: linear-gradient(180deg, #667eea, #764ba2); }
    .section-header.analysis::before { background: linear-gradient(180deg, #11998e, #38ef7d); }
    .section-header.portfolio::before { background: linear-gradient(180deg, #F7971E, #FFD200); }
    .section-header.backtest::before { background: linear-gradient(180deg, #667eea, #38ef7d); }
    .section-header .section-icon {
        font-size: 1.8rem;
        animation: float 3s ease-in-out infinite;
    }
    .section-header .section-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: white;
        letter-spacing: -0.3px;
    }
    .section-header .section-subtitle {
        font-size: 0.8rem;
        color: rgba(160,174,192,0.7);
        margin-top: 2px;
    }
    .section-header.risk .section-title { background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .section-header.analysis .section-title { background: linear-gradient(135deg, #11998e, #38ef7d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .section-header.portfolio .section-title { background: linear-gradient(135deg, #F7971E, #FFD200); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .section-header.backtest .section-title { background: linear-gradient(135deg, #667eea, #38ef7d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* ===== SECTION DIVIDERS ===== */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), rgba(240,147,251,0.2), transparent);
        margin: 2rem 0;
        border: none;
    }

    /* ===== ANALYSIS COLUMN HEADERS ===== */
    .analysis-col-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.8rem 1.2rem;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .analysis-col-header.tech { border-left: 3px solid #38ef7d; }
    .analysis-col-header.fund { border-left: 3px solid #FFD200; }
    .analysis-col-header span { font-weight: 700; color: white; font-size: 0.95rem; }

    /* ===== HOLDINGS HEADER ===== */
    .holdings-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,210,0,0.3);
    }

    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(240,147,251,0.2)) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(102,126,234,0.35), rgba(240,147,251,0.35)) !important;
        border-color: rgba(240,147,251,0.5) !important;
        box-shadow: 0 4px 20px rgba(102,126,234,0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* ===== DISCLAIMER ===== */
    .disclaimer {
        background: rgba(255,107,107,0.05);
        border: 1px solid rgba(255,107,107,0.15);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: rgba(160,174,192,0.7);
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ANIMATED HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🤖 Robo-Advisory Portfolio System</h1>
    <p>AI-Powered Investment Portfolio Generation — Powered by LSTM, Random Forest & Deep Learning</p>
    <div class="subtitle-line"></div>
</div>
""", unsafe_allow_html=True)

# Stock ticker marquee
st.markdown("""
<div class="ticker-wrap">
    <div class="ticker-content">
        <span class="ticker-item"><span class="tick-name">AAPL</span> <span class="tick-up">▲ 2.3%</span></span>
        <span class="ticker-item"><span class="tick-name">MSFT</span> <span class="tick-up">▲ 1.1%</span></span>
        <span class="ticker-item"><span class="tick-name">NVDA</span> <span class="tick-down">▼ 0.8%</span></span>
        <span class="ticker-item"><span class="tick-name">GOOGL</span> <span class="tick-up">▲ 0.5%</span></span>
        <span class="ticker-item"><span class="tick-name">AMZN</span> <span class="tick-up">▲ 1.7%</span></span>
        <span class="ticker-item"><span class="tick-name">TSLA</span> <span class="tick-down">▼ 2.1%</span></span>
        <span class="ticker-item"><span class="tick-name">META</span> <span class="tick-up">▲ 3.2%</span></span>
        <span class="ticker-item"><span class="tick-name">JPM</span> <span class="tick-up">▲ 0.9%</span></span>
        <span class="ticker-item"><span class="tick-name">V</span> <span class="tick-up">▲ 0.4%</span></span>
        <span class="ticker-item"><span class="tick-name">KO</span> <span class="tick-up">▲ 0.2%</span></span>
        <span class="ticker-item"><span class="tick-name">AMD</span> <span class="tick-down">▼ 1.5%</span></span>
        <span class="ticker-item"><span class="tick-name">NFLX</span> <span class="tick-up">▲ 2.8%</span></span>
        <span class="ticker-item"><span class="tick-name">AAPL</span> <span class="tick-up">▲ 2.3%</span></span>
        <span class="ticker-item"><span class="tick-name">MSFT</span> <span class="tick-up">▲ 1.1%</span></span>
        <span class="ticker-item"><span class="tick-name">NVDA</span> <span class="tick-down">▼ 0.8%</span></span>
        <span class="ticker-item"><span class="tick-name">GOOGL</span> <span class="tick-up">▲ 0.5%</span></span>
        <span class="ticker-item"><span class="tick-name">AMZN</span> <span class="tick-up">▲ 1.7%</span></span>
        <span class="ticker-item"><span class="tick-name">TSLA</span> <span class="tick-down">▼ 2.1%</span></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR — STEP 1: RISK PROFILING QUESTIONNAIRE
# ============================================================================

user_email = ""
user_name = ""

with st.sidebar:
    st.markdown("""<div style='text-align:center;margin-bottom:0.8rem;'>
        <span style='font-size:1.6rem;'>🔐</span>
        <span style='font-size:1.1rem;font-weight:800;color:white;'> User Login</span>
    </div>""", unsafe_allow_html=True)

    user_email_input = st.text_input("Email", placeholder="your@email.com", key="user_email_input")
    user_password_input = st.text_input("Password", type="password", placeholder="••••••••", key="user_password_input")
    user_name_input  = st.text_input("Name (for Registration)",  placeholder="Your Name",  key="user_name_input")

    if st.button("🔑 Login / Register", use_container_width=True):
        if user_email_input and user_password_input and _db and _db.is_connected():
            user, error = _db.get_or_create_user(user_email_input, user_password_input, user_name_input)
            if error:
                st.error(f"❌ {error}")
            elif user:
                st.session_state["logged_in_email"] = user_email_input
                st.session_state["logged_in_name"] = user.get('name', user_email_input)
                st.success(f"Welcome, **{st.session_state['logged_in_name']}**!")
                
        elif not user_email_input or not user_password_input:
            st.warning("Please enter both email and password.")
        elif not _db or not _db.is_connected():
            st.info("📦 MongoDB not connected — portfolios won't be saved.")
            
    # Persist login state
    if "logged_in_email" in st.session_state:
        user_email = st.session_state["logged_in_email"]
        user_name = st.session_state["logged_in_name"]
        st.success(f"Logged in as: {user_email}")

    st.markdown("---")
    st.markdown("## 📋 Step 1: Investor Profile")
    st.markdown("---")

    st.markdown("#### 👤 Personal Information")
    age = st.slider("Age", 18, 85, 35, help="Your current age")

    age_cl_map = {(18,24): 1, (25,34): 2, (35,44): 3, (45,54): 4, (55,64): 5, (65,85): 6}
    agecl = 3
    for (lo, hi), v in age_cl_map.items():
        if lo <= age <= hi:
            agecl = v; break

    education = st.selectbox("Education Level", [
        "Less than High School",
        "High School / GED",
        "Some College",
        "Bachelor's Degree",
        "Master's Degree",
        "Doctoral / Professional Degree"
    ], index=3)
    educ_map = {"Less than High School": 8, "High School / GED": 12,
                "Some College": 14, "Bachelor's Degree": 16,
                "Master's Degree": 17, "Doctoral / Professional Degree": 17}
    educ = educ_map[education]

    occupation = st.selectbox("Occupation", [
        "Employee / Salaried",
        "Self-Employed / Business Owner",
        "Retired",
        "Not Working / Student"
    ], index=0)
    occ_map = {"Employee / Salaried": (1, 1), "Self-Employed / Business Owner": (1, 2),
               "Retired": (4, 4), "Not Working / Student": (3, 3)}
    occat1, occat2 = occ_map[occupation]

    st.markdown("---")
    st.markdown("#### 💰 Financial Information")

    income_level = st.selectbox("Annual Income", [
        "Under $25,000",
        "$25,000 – $50,000",
        "$50,000 – $100,000",
        "$100,000 – $250,000",
        "$250,000 – $500,000",
        "Over $500,000"
    ], index=3)
    inc_map = {"Under $25,000": 1, "$25,000 – $50,000": 2, "$50,000 – $100,000": 3,
               "$100,000 – $250,000": 4, "$250,000 – $500,000": 5, "Over $500,000": 6}
    inccat = inc_map[income_level]

    net_worth = st.selectbox("Net Worth", [
        "Under $50,000",
        "$50,000 – $150,000",
        "$150,000 – $500,000",
        "$500,000 – $1M",
        "Over $1M"
    ], index=2)
    nw_map = {"Under $50,000": 1, "$50,000 – $150,000": 2, "$150,000 – $500,000": 3,
              "$500,000 – $1M": 4, "Over $1M": 5}
    nwcat = nw_map[net_worth]

    assets_level = st.selectbox("Total Assets", [
        "Under $50,000",
        "$50,000 – $150,000",
        "$150,000 – $500,000",
        "$500,000 – $1M",
        "Over $1M",
        "Over $5M"
    ], index=2)
    asset_map = {"Under $50,000": 1, "$50,000 – $150,000": 2, "$150,000 – $500,000": 3,
                 "$500,000 – $1M": 4, "Over $1M": 5, "Over $5M": 6}
    assetcat = asset_map[assets_level]

    st.markdown("---")
    st.markdown("#### 📈 Investment Experience")

    has_emergency = st.toggle("Emergency Savings Fund", value=True)
    has_savings = st.toggle("Savings / Checking Accounts", value=True)
    has_mutual_funds = st.toggle("Mutual Fund Investments", value=False)
    has_retirement = st.toggle("Retirement Accounts (401k/IRA)", value=True)

    # Percentile estimates based on selections
    inc_pctile = min(95, max(10, inccat * 18))
    nw_pctile = min(95, max(10, nwcat * 20))

    st.markdown("---")
    st.markdown("#### 💵 Portfolio Settings")
    capital = st.number_input("Investment Capital ($)", min_value=1000, max_value=10_000_000,
                              value=100_000, step=5000, format="%d")
    horizon = st.selectbox("Investment Horizon", [90, 180, 270, 360], index=1,
                          format_func=lambda x: f"{x} days ({x//30} months)")

    # Build feature dict
    user_features = {
        'EDUC': educ, 'EMERGSAV': int(has_emergency), 'HSAVFIN': int(has_savings),
        'HNMMF': int(has_mutual_funds), 'HRETQLIQ': int(has_retirement),
        'NWCAT': nwcat, 'INCCAT': inccat, 'ASSETCAT': assetcat,
        'NINCCAT': max(1, inccat - 1), 'NINC2CAT': max(1, inccat - 1),
        'NWPCTLECAT': nw_pctile, 'INCPCTLECAT': inc_pctile,
        'NINCPCTLECAT': max(10, inc_pctile - 10),
        'INCQRTCAT': min(4, max(1, inccat // 2 + 1)),
        'NINCQRTCAT': min(4, max(1, (inccat - 1) // 2 + 1)),
        'AGE': age, 'AGECL': agecl, 'OCCAT1': occat1, 'OCCAT2': occat2
    }

    if user_email:
        generate_btn = st.button("🚀 Generate Portfolio", type="primary", use_container_width=True)
    else:
        st.info("🔐 Please Login / Register above to generate a portfolio.")
        generate_btn = False


# ============================================================================
# HELPER FUNCTIONS (imported from generate_portfolio logic)
# ============================================================================
@st.cache_resource
def load_risk_model():
    with open(RISK_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_technical_ensemble():
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    class _DAL(tf.keras.losses.Loss):
        def __init__(self, direction_weight=0.3, name='direction_aware_loss', reduction='sum_over_batch_size'):
            super().__init__(name=name, reduction=reduction)
            self.direction_weight = direction_weight
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            dp = tf.reduce_mean(tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32))
            return (1 - self.direction_weight) * mse + self.direction_weight * dp * 100
        def get_config(self):
            c = super().get_config(); c.update({'direction_weight': self.direction_weight}); return c

    scaler = joblib.load(SCALER_PATH)
    info_path = ENSEMBLE_INFO_PATH if os.path.exists(ENSEMBLE_INFO_PATH) else ENSEMBLE_INFO_PATH_OLD
    info = joblib.load(info_path)
    weights = np.array(info['weights'])

    models = []
    for mf in sorted(glob.glob(os.path.join(ENSEMBLE_DIR, 'model_*.keras'))):
        try:
            m = tf.keras.models.load_model(mf, custom_objects={'DirectionAwareLoss': _DAL}, compile=False)
        except Exception:
            m = tf.keras.models.load_model(mf, compile=False)
        models.append(m)
    return models, weights, scaler

@st.cache_resource
def load_fundamental_model():
    if FUND_DIR not in sys.path:
        sys.path.insert(0, FUND_DIR)
    from score import load_best_model, engineer_features, predictions_to_scores
    from prepare import prepare_dataset
    model, meta = load_best_model()
    return model, meta, engineer_features, predictions_to_scores, prepare_dataset


def predict_risk(features):
    rd = load_risk_model()
    model = rd['model']
    feat_names = rd['features']
    fv = [features.get(f, 0) for f in feat_names]
    score = float(np.clip(model.predict(np.array([fv]))[0], 0, 100))
    if score <= 20: cat = 'Conservative'
    elif score <= 35: cat = 'Conservative-Moderate'
    elif score <= 50: cat = 'Moderate'
    elif score <= 70: cat = 'Moderate-Aggressive'
    else: cat = 'Aggressive'
    return score, cat


class _Indicators:
    @staticmethod
    def rsi(s, p=14):
        d=s.diff(); g=(d.where(d>0,0)).ewm(alpha=1/p,adjust=False).mean()
        l=(-d.where(d<0,0)).ewm(alpha=1/p,adjust=False).mean(); return 100-(100/(1+g/l))
    @staticmethod
    def adx(h,l,c,p=14):
        pd_=h.diff().where(h.diff()>0,0); md=l.diff().where(l.diff()<0,0).abs()
        tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr=tr.ewm(alpha=1/p,adjust=False).mean()
        pi_=100*(pd_.ewm(alpha=1/p,adjust=False).mean()/atr)
        mi=100*(md.ewm(alpha=1/p,adjust=False).mean()/atr)
        dx=(abs(pi_-mi)/(pi_+mi+1e-8))*100; return dx.ewm(alpha=1/p,adjust=False).mean()
    @staticmethod
    def natr(h,l,c,p=14):
        tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        return (tr.ewm(alpha=1/p,adjust=False).mean()/c)*100
    @staticmethod
    def obv_slope(c,v,p=14): return (np.sign(c.diff())*v).fillna(0).cumsum().diff(p)
    @staticmethod
    def dist_sma(c,p=50): s=c.rolling(p).mean(); return (c-s)/s
    @staticmethod
    def macd(c,f=12,s=26,sg=9):
        m=c.ewm(span=f,adjust=False).mean()-c.ewm(span=s,adjust=False).mean()
        return m-m.ewm(span=sg,adjust=False).mean()
    @staticmethod
    def roc(c,p=10): return ((c-c.shift(p))/c.shift(p))*100
    @staticmethod
    def vol_ratio(v,p=20): return v/v.rolling(p).mean()
    @staticmethod
    def bb_pos(c,p=20,s=2):
        sm=c.rolling(p).mean(); rs=c.rolling(p).std()
        bw=(sm+rs*s)-(sm-rs*s); bw=bw.replace(0,np.nan); return (c-(sm-rs*s))/bw


def run_technical_analysis(progress_bar, status_text, horizon_days):
    import yfinance as yf
    from scipy.stats import norm

    models, weights, scaler = load_technical_ensemble()

    status_text.text("📈 Downloading S&P 500 index data...")
    sp = yf.download('^GSPC', period='2y', interval='1d', progress=False)
    if isinstance(sp.columns, pd.MultiIndex): sp.columns = sp.columns.get_level_values(0)

    # Market baseline
    try:
        sp10 = yf.download('^GSPC', period='10y', interval='1d', progress=False)
        if isinstance(sp10.columns, pd.MultiIndex): sp10.columns = sp10.columns.get_level_values(0)
        mret = float(sp10['Close'].pct_change(252).mean() * 100)
        mvol = float(sp10['Close'].pct_change().dropna().std() * np.sqrt(252) * 100)
    except Exception:
        mret, mvol = 10.0, 15.0

    results = []
    total = len(STOCK_UNIVERSE)
    for i, ticker in enumerate(STOCK_UNIVERSE):
        progress_bar.progress((i + 1) / total, f"Analyzing {ticker}...")
        status_text.text(f"🔍 Technical: {STOCK_NAMES.get(ticker, ticker)} ({i+1}/{total})")
        try:
            df = yf.download(ticker, period='2y', interval='1d', progress=False)
            if len(df) < 150: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            price = float(df['Close'].iloc[-1])

            df['RSI']=_Indicators.rsi(df['Close']); df['ADX']=_Indicators.adx(df['High'],df['Low'],df['Close'])
            df['NATR']=_Indicators.natr(df['High'],df['Low'],df['Close'])
            df['OBV_Slope']=_Indicators.obv_slope(df['Close'],df['Volume'])
            df['Dist_SMA']=_Indicators.dist_sma(df['Close']); df['MACD']=_Indicators.macd(df['Close'])
            df['ROC']=_Indicators.roc(df['Close']); df['Vol_Ratio']=_Indicators.vol_ratio(df['Volume'])
            df['BB_Position']=_Indicators.bb_pos(df['Close'])

            # Market features
            ma = sp.reindex(df.index, method='ffill')
            mf = pd.DataFrame(index=ma.index)
            for p in [20,50,200]: mf[f'am{p}']=(ma['Close']>ma['Close'].rolling(p).mean()).astype(float)
            mf['market_strength']=mf[[f'am{p}' for p in [20,50,200]]].mean(axis=1)
            mf['market_momentum']=ma['Close'].pct_change(20)*100
            r_=ma['Close'].pct_change()
            mf['market_volatility']=r_.rolling(20).std()*np.sqrt(252)*100
            mf['market_trend']=r_.rolling(20).mean()/(r_.rolling(20).std()+1e-8)
            df=pd.concat([df,mf[['market_strength','market_momentum','market_volatility','market_trend']]],axis=1)
            df['relative_strength']=df['Close'].pct_change(20)-ma['Close'].pct_change(20)
            sr,mr=df['Close'].pct_change(),ma['Close'].pct_change()
            df['beta']=sr.rolling(60).cov(mr)/(mr.rolling(60).var()+1e-8)
            df.dropna(inplace=True)

            if len(df) < SEQ_LENGTH: continue

            fcols=['RSI','ADX','NATR','OBV_Slope','Dist_SMA','MACD','ROC','Vol_Ratio','BB_Position',
                   'market_strength','market_momentum','market_volatility','market_trend','relative_strength','beta']
            inp = np.array([scaler.transform(df[fcols].tail(SEQ_LENGTH))])
            preds = np.array([m.predict(inp, verbose=0) for m in models])
            mean_f = np.average(preds, axis=0, weights=weights)[0]
            std_f = np.std(preds, axis=0)[0]

            eff = min(horizon_days, MAX_HORIZON)
            raw_ret = float(mean_f[eff-1]); pred_std = float(std_f[eff-1])
            tf_ = eff/365.0
            natr_v = float(df['NATR'].iloc[-1])
            total_unc = float(np.sqrt((natr_v*np.sqrt(eff))**2 + (mvol*np.sqrt(tf_))**2 + pred_std**2))
            fz = float((raw_ret - mret*tf_) / total_unc) if total_unc > 0 else 0.0

            adx_v = float(df['ADX'].iloc[-1]); rsi_v = float(df['RSI'].iloc[-1])
            conf = 1.0
            if adx_v < 20: conf *= 0.7
            elif adx_v > 40: conf *= 1.1
            if rsi_v > 70 and raw_ret > 0: conf *= 0.8
            elif rsi_v < 30 and raw_ret < 0: conf *= 0.8
            if pred_std > 5: conf *= 0.85
            conf = min(conf * 1.05, 1.2)

            base_sc = norm.cdf(fz) * 100
            final_sc = float(np.clip(50 + (base_sc - 50) * conf, 0, 100))

            if final_sc > 75: sig = 'STRONG BUY'
            elif final_sc > 65: sig = 'BUY'
            elif final_sc > 55: sig = 'WEAK BUY'
            elif final_sc > 45: sig = 'NEUTRAL'
            elif final_sc > 35: sig = 'WEAK SELL'
            elif final_sc > 25: sig = 'SELL'
            else: sig = 'STRONG SELL'

            results.append({
                'ticker': ticker, 'name': STOCK_NAMES.get(ticker, ticker),
                'current_price': round(price, 2), 'technical_score': round(final_sc, 2),
                'predicted_return': round(raw_ret, 2), 'uncertainty': round(pred_std, 2),
                'confidence': round(conf, 2), 'signal': sig,
                'rsi': round(rsi_v, 2), 'adx': round(adx_v, 2), 'natr': round(natr_v, 2),
            })
        except Exception:
            continue
    return results


def run_fundamental_analysis(progress_bar, status_text):
    try:
        model, meta, engineer_features, predictions_to_scores, prepare_dataset = load_fundamental_model()
        status_text.text("📊 Loading SEC EDGAR fundamental data...")
        df = prepare_dataset()
        df = engineer_features(df)
        latest = df.sort_values("filed_date").groupby("ticker").last().reset_index()
        X = latest[meta['feature_cols']].values
        raw = model.predict(X)
        latest = latest.copy()
        latest['fund_score'] = predictions_to_scores(raw)
        latest['fund_signal'] = latest['fund_score'].apply(
            lambda s: "BUY" if s >= meta['buy_threshold'] else ("SELL" if s < meta['sell_threshold'] else "HOLD"))
        scores = {}
        for _, r in latest.iterrows():
            scores[r['ticker']] = {
                'fundamental_score': round(float(r['fund_score']), 2),
                'fundamental_signal': r['fund_signal'],
            }
        progress_bar.progress(1.0, "Fundamental analysis complete!")
        return scores
    except Exception as e:
        status_text.text(f"⚠️ Fundamental analysis unavailable: {str(e)[:80]}")
        return {}


def combine_scores(tech_results, fund_scores):
    combined = []
    for r in tech_results:
        fd = fund_scores.get(r['ticker'], {})
        fs = fd.get('fundamental_score', None)
        if fs is not None:
            cs = TECHNICAL_WEIGHT * r['technical_score'] + FUNDAMENTAL_WEIGHT * fs
            src = 'BOTH'
        else:
            cs = r['technical_score'] * 0.85
            src = 'TECH_ONLY'
        cs = round(float(np.clip(cs, 0, 100)), 2)
        if cs > 75: csig = 'STRONG BUY'
        elif cs > 65: csig = 'BUY'
        elif cs > 55: csig = 'WEAK BUY'
        elif cs > 45: csig = 'NEUTRAL'
        elif cs > 35: csig = 'WEAK SELL'
        elif cs > 25: csig = 'SELL'
        else: csig = 'STRONG SELL'
        entry = {**r, 'fundamental_score': fs if fs else None,
                 'fundamental_signal': fd.get('fundamental_signal', 'N/A'),
                 'combined_score': cs, 'combined_signal': csig, 'score_source': src}
        combined.append(entry)
    combined.sort(key=lambda x: x['combined_score'], reverse=True)
    return combined


# Known stable/defensive stocks (low beta, dividend payers)
DEFENSIVE_STOCKS = {'KO', 'PG', 'JNJ', 'PEP', 'MRK', 'ABBV', 'MCD', 'WMT', 'COST', 'HD',
                     'CL', 'SO', 'DUK', 'MMC', 'UNH', 'V', 'MA', 'ADBE', 'NOW'}
# Known high-growth/momentum stocks (high beta)
GROWTH_STOCKS = {'NVDA', 'TSLA', 'AMD', 'NFLX', 'META', 'AMZN', 'UBER', 'CRM', 'SHOP', 'CRWD', 'PANW', 'SQ'}


def allocate_portfolio(stock_results, risk_score, capital):
    if risk_score <= 20: params = {'cat': 'Conservative', 'conc': 0.10, 'min_sc': 60, 'max_natr': 2.0, 'max_beta': 1.2}
    elif risk_score <= 35: params = {'cat': 'Cons-Moderate', 'conc': 0.15, 'min_sc': 55, 'max_natr': 2.5, 'max_beta': 1.4}
    elif risk_score <= 50: params = {'cat': 'Moderate', 'conc': 0.20, 'min_sc': 50, 'max_natr': 3.5, 'max_beta': 1.8}
    elif risk_score <= 70: params = {'cat': 'Mod-Aggressive', 'conc': 0.25, 'min_sc': 45, 'max_natr': 5.0, 'max_beta': 2.5}
    else: params = {'cat': 'Aggressive', 'conc': 0.30, 'min_sc': 45, 'max_natr': 999, 'max_beta': 999}

    max_eq = min(0.95, (risk_score / 100) ** 0.8)

    # --- RISK-AWARE STOCK FILTERING ---
    qual = [s for s in stock_results if s['combined_score'] >= 45
            and s['combined_signal'] in ['NEUTRAL','WEAK BUY','BUY','STRONG BUY']]

    # Apply volatility & beta filters based on risk profile
    risk_filtered = []
    for s in qual:
        natr = s.get('natr', 1.0)
        ticker = s['ticker']

        # Conservative: exclude high-volatility stocks, prefer defensive
        if risk_score <= 35:
            if natr > params['max_natr']:
                continue
            if ticker in GROWTH_STOCKS and risk_score <= 20:
                continue  # Ultra-conservative skip growth entirely
            # Boost defensive stocks for conservative
            if ticker in DEFENSIVE_STOCKS:
                s = {**s, 'combined_score': min(100, s['combined_score'] * 1.15)}

        # Aggressive: boost high-growth/momentum stocks
        elif risk_score >= 70:
            if ticker in GROWTH_STOCKS:
                s = {**s, 'combined_score': min(100, s['combined_score'] * 1.10)}
            elif ticker in DEFENSIVE_STOCKS:
                s = {**s, 'combined_score': s['combined_score'] * 0.90}  # Slight penalty

        risk_filtered.append(s)

    filt = [s for s in risk_filtered if s['combined_score'] >= params['min_sc']]
    filt.sort(key=lambda x: x['combined_score'], reverse=True)

    if len(filt) < 3:
        return {'allocations': [], 'cash_pct': 100.0, 'cash_amt': capital,
                'eq_pct': 0.0, 'eq_amt': 0.0, 'port_ret': 0.0, 'port_unc': 0.0, 'sharpe': 0.0, 'params': params}

    scores = np.array([s['combined_score'] for s in filt])
    w = (scores / 100.0) ** 2.0
    w /= w.sum()

    # Risk adjustments
    for i, s in enumerate(filt):
        pu = max(0.5, 1 - (s['uncertainty']/100)*(1-risk_score/100))
        ps = 1.1 if s['combined_signal']=='STRONG BUY' and risk_score>=50 else (0.95 if s['combined_signal'] in ['WEAK BUY','NEUTRAL'] else 1.0)
        pc = min(1.2, 0.8 + 0.4*s['confidence'])
        w[i] *= pu * ps * pc
    if w.sum() > 0: w /= w.sum()

    # Concentration limits
    cl = params['conc']
    for _ in range(10):
        ex = np.maximum(w - cl, 0)
        if ex.sum() == 0: break
        w = np.minimum(w, cl)
        if w.sum() > 0:
            rem = w < cl
            if rem.sum() > 0: w[rem] += ex.sum() * (w[rem] / w[rem].sum())
    if w.sum() > 0: w /= w.sum()

    mask = w >= 0.05
    w *= mask; filt = [s for s, m in zip(filt, mask) if m]; w = w[mask]
    if w.sum() > 0: w /= w.sum()

    eq_amt = capital * max_eq
    cash_amt = capital - eq_amt

    allocs = []
    for i, s in enumerate(filt):
        sw = w[i] * max_eq; sa = capital * sw
        allocs.append({**s, 'weight_pct': round(sw*100, 2), 'capital': round(sa, 2),
                       'shares': int(sa / s['current_price']) if s['current_price'] > 0 else 0})

    rets = np.array([s['predicted_return'] for s in filt])
    uncs = np.array([s['uncertainty'] for s in filt])
    pr = float(np.sum(w * rets)); pu = float(np.sqrt(np.sum((w * uncs)**2)))

    return {
        'allocations': allocs, 'cash_pct': round((1-max_eq)*100, 2), 'cash_amt': round(cash_amt, 2),
        'eq_pct': round(max_eq*100, 2), 'eq_amt': round(eq_amt, 2),
        'port_ret': round(pr, 2), 'port_unc': round(pu, 2), 'sharpe': round(pr/(pu+1e-8), 2), 'params': params
    }


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================
def run_backtest(allocations, capital, lookback_months=12):
    """Backtest portfolio using historical prices."""
    import yfinance as yf

    tickers = [a['ticker'] for a in allocations]
    weights = {a['ticker']: a['weight_pct'] / 100.0 for a in allocations}
    cash_weight = 1.0 - sum(weights.values())

    # Download historical prices
    period = f"{lookback_months}mo"
    prices = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval='1d', progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) > 20:
                prices[t] = df['Close']
        except Exception:
            pass

    # S&P 500 benchmark
    try:
        sp = yf.download('^GSPC', period=period, interval='1d', progress=False)
        if isinstance(sp.columns, pd.MultiIndex): sp.columns = sp.columns.get_level_values(0)
        sp_prices = sp['Close']
    except Exception:
        sp_prices = None

    if len(prices) < 2:
        return None

    # Align dates
    price_df = pd.DataFrame(prices)
    price_df.dropna(inplace=True)
    if len(price_df) < 20:
        return None

    # Calculate daily returns
    returns = price_df.pct_change().dropna()

    # Portfolio daily returns (weighted)
    port_daily = pd.Series(0.0, index=returns.index)
    for t in returns.columns:
        if t in weights:
            port_daily += returns[t] * weights[t]
    # Cash earns ~4.5% annual (money market)
    port_daily += cash_weight * (0.045 / 252)

    # Equity curve
    port_cumulative = (1 + port_daily).cumprod() * capital

    # Benchmark
    if sp_prices is not None:
        sp_aligned = sp_prices.reindex(returns.index, method='ffill').dropna()
        sp_ret = sp_aligned.pct_change().dropna()
        common_idx = port_daily.index.intersection(sp_ret.index)
        sp_cumulative = (1 + sp_ret.loc[common_idx]).cumprod() * capital
        port_cumulative = port_cumulative.loc[common_idx]
    else:
        sp_cumulative = None
        common_idx = port_daily.index

    # Metrics
    total_ret = float((port_cumulative.iloc[-1] / capital - 1) * 100)
    ann_ret = float(total_ret * (252 / len(common_idx)))
    daily_vol = float(port_daily.loc[common_idx].std())
    ann_vol = float(daily_vol * np.sqrt(252) * 100)
    sharpe = float(ann_ret / (ann_vol + 1e-8))

    # Max drawdown
    running_max = port_cumulative.cummax()
    drawdown = (port_cumulative - running_max) / running_max * 100
    max_dd = float(drawdown.min())

    # Benchmark metrics
    if sp_cumulative is not None and len(sp_cumulative) > 1:
        bench_ret = float((sp_cumulative.iloc[-1] / capital - 1) * 100)
        alpha = total_ret - bench_ret
    else:
        bench_ret = 0.0
        alpha = 0.0

    return {
        'port_curve': port_cumulative,
        'bench_curve': sp_cumulative,
        'total_return': round(total_ret, 2),
        'ann_return': round(ann_ret, 2),
        'ann_volatility': round(ann_vol, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd, 2),
        'bench_return': round(bench_ret, 2),
        'alpha': round(alpha, 2),
        'days': len(common_idx),
    }


# ============================================================================
# MAIN CONTENT — TABBED NAVIGATION
# ============================================================================
tab_gen, tab_models, tab_history = st.tabs([
    "🚀 Portfolio Generator",
    "🔧 Model Management",
    "📜 Portfolio History"
])

# ============================================================================
# TAB 1: PORTFOLIO GENERATOR
# ============================================================================
with tab_gen:
  if generate_btn:
    # --- Step 2: Risk Score ---
    with st.container():
        st.markdown("""<div class="section-header risk">
            <span class="section-icon">🎯</span>
            <div><div class="section-title">Step 2: Risk Score Prediction</div>
            <div class="section-subtitle">RandomForest model analyzing your financial profile</div></div>
        </div>""", unsafe_allow_html=True)
        with st.spinner("Calculating risk score..."):
            risk_score, risk_cat = predict_risk(user_features)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{risk_score:.1f}</div>
                <div class="label">Risk Score (0-100)</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            color = '#38ef7d' if risk_score <= 35 else ('#FFD200' if risk_score <= 65 else '#ff6b6b')
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{risk_cat}</div>
                <div class="label">Risk Category</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            max_eq = min(0.95, (risk_score / 100) ** 0.8)
            st.markdown(f"""<div class="metric-card">
                <div class="value">{max_eq*100:.0f}%</div>
                <div class="label">Max Equity Exposure</div>
            </div>""", unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score,
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': color},
                'bgcolor': '#1a1a2e',
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(56,239,125,0.13)'},
                    {'range': [20, 35], 'color': 'rgba(56,239,125,0.07)'},
                    {'range': [35, 50], 'color': 'rgba(255,210,0,0.07)'},
                    {'range': [50, 70], 'color': 'rgba(255,210,0,0.13)'},
                    {'range': [70, 100], 'color': 'rgba(255,107,107,0.13)'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.8, 'value': risk_score}
            },
            title={'text': 'Risk Tolerance', 'font': {'color': 'white', 'size': 16}},
            number={'font': {'color': 'white', 'size': 32}}
        ))
        fig_gauge.update_layout(
            height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'},
            margin=dict(l=30, r=30, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Step 3: Stock Analysis ---
    st.markdown("""<div class="section-header analysis">
        <span class="section-icon">📊</span>
        <div><div class="section-title">Step 3: Stock Analysis</div>
        <div class="section-subtitle">LSTM Ensemble + Random Forest scoring S&P 500 universe</div></div>
    </div>""", unsafe_allow_html=True)
    col_t, col_f = st.columns(2)

    with col_t:
        st.markdown('<div class="analysis-col-header tech"><span>📈 Technical Analysis</span></div>', unsafe_allow_html=True)
        tech_progress = st.progress(0, "Starting technical analysis...")
        tech_status = st.empty()

    tech_results = run_technical_analysis(tech_progress, tech_status, horizon)
    tech_status.text(f"✅ Scored {len(tech_results)} stocks")

    with col_f:
        st.markdown('<div class="analysis-col-header fund"><span>📋 Fundamental Analysis</span></div>', unsafe_allow_html=True)
        fund_progress = st.progress(0, "Starting fundamental analysis...")
        fund_status = st.empty()

    fund_scores = run_fundamental_analysis(fund_progress, fund_status)
    fund_status.text(f"✅ Scored {len(fund_scores)} companies")

    # Combine
    combined = combine_scores(tech_results, fund_scores)

    # Show scores table
    if combined:
        df_scores = pd.DataFrame([{
            'Stock': s['name'], 'Ticker': s['ticker'],
            'Technical': s['technical_score'],
            'Fundamental': s['fundamental_score'] if s['fundamental_score'] else '—',
            'Combined': s['combined_score'],
            'Signal': s['combined_signal'],
            'Pred. Return': f"{s['predicted_return']:+.1f}%",
        } for s in combined[:15]])
        st.dataframe(df_scores, use_container_width=True, hide_index=True)

        # Score distribution chart
        fig_scores = go.Figure()
        tickers = [s['ticker'] for s in combined[:15]]
        tech_s = [s['technical_score'] for s in combined[:15]]
        fund_s = [s['fundamental_score'] if s['fundamental_score'] else 0 for s in combined[:15]]
        comb_s = [s['combined_score'] for s in combined[:15]]

        fig_scores.add_trace(go.Bar(name='Technical', x=tickers, y=tech_s,
                                    marker_color='#667eea', opacity=0.85,
                                    marker_line=dict(width=0)))
        fig_scores.add_trace(go.Bar(name='Fundamental', x=tickers, y=fund_s,
                                    marker_color='#f093fb', opacity=0.85,
                                    marker_line=dict(width=0)))
        fig_scores.add_trace(go.Scatter(name='Combined', x=tickers, y=comb_s,
                                        mode='lines+markers',
                                        line=dict(color='#FFD200', width=3.5, shape='spline'),
                                        marker=dict(size=9, color='#FFD200',
                                                   line=dict(width=2, color='#0a0a1a'))))
        fig_scores.update_layout(
            title=dict(text='<b>Stock Scores Comparison</b>', font=dict(size=16, color='white')),
            barmode='group', template='plotly_dark',
            height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=1.12, font=dict(size=12)),
            margin=dict(l=40, r=20, t=60, b=40),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Score'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)')
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Step 4 & 5: Allocation & Portfolio ---
    st.markdown("""<div class="section-header portfolio">
        <span class="section-icon">💼</span>
        <div><div class="section-title">Step 4 & 5: Portfolio Allocation</div>
        <div class="section-subtitle">Risk-aware selection with volatility filtering & concentration limits</div></div>
    </div>""", unsafe_allow_html=True)
    portfolio = allocate_portfolio(combined, risk_score, capital)

    if portfolio['allocations']:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{portfolio['eq_pct']:.0f}%</div>
                <div class="label">Equity Allocation</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{portfolio['cash_pct']:.0f}%</div>
                <div class="label">Cash Reserve</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{portfolio['port_ret']:+.1f}%</div>
                <div class="label">Expected Return</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{portfolio['sharpe']:.2f}</div>
                <div class="label">Sharpe Ratio</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Portfolio pie chart & holdings table side by side
        col_pie, col_tbl = st.columns([1, 2])

        with col_pie:
            labels = [a['name'][:15] for a in portfolio['allocations']] + ['Cash']
            values = [a['weight_pct'] for a in portfolio['allocations']] + [portfolio['cash_pct']]
            _palette = ['#667eea','#38ef7d','#f093fb','#FFD200','#ff6b6b','#11998e','#F7971E',
                       '#764ba2','#43e97b','#fa709a','#fee140','#30cfd0','#a18cd1']
            colors = _palette[:len(labels)-1] + ['rgba(80,80,80,0.6)']

            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.55,
                marker=dict(colors=colors, line=dict(color='#0a0a1a', width=2)),
                textinfo='label+percent', textposition='outside',
                textfont=dict(size=11, color='rgba(200,210,220,0.9)')
            ))
            fig_pie.update_layout(
                title='Portfolio Allocation', template='plotly_dark',
                height=450, paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False, margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_tbl:
            st.markdown('<div class="holdings-title">📋 Holdings</div>', unsafe_allow_html=True)
            df_port = pd.DataFrame([{
                'Stock': a['name'], 'Score': a['combined_score'],
                'Signal': a['combined_signal'],
                'Weight': f"{a['weight_pct']:.1f}%",
                'Capital': f"${a['capital']:,.0f}",
                'Shares': a['shares'],
                'Return': f"{a['predicted_return']:+.1f}%",
            } for a in portfolio['allocations']])
            st.dataframe(df_port, use_container_width=True, hide_index=True, height=380)

        # Capital allocation bar
        fig_bar = go.Figure()
        _bar_palette = ['#667eea','#38ef7d','#f093fb','#FFD200','#ff6b6b','#11998e','#F7971E',
                       '#764ba2','#43e97b','#fa709a','#fee140','#30cfd0','#a18cd1']
        for idx, a in enumerate(portfolio['allocations']):
            fig_bar.add_trace(go.Bar(
                name=a['ticker'], x=[a['capital']], y=['Portfolio'],
                orientation='h', text=f"  {a['ticker']} ${a['capital']:,.0f}  ",
                textposition='inside', textfont=dict(size=10, color='white', family='JetBrains Mono'),
                marker_color=_bar_palette[idx % len(_bar_palette)],
                marker_line=dict(width=0)
            ))
        fig_bar.add_trace(go.Bar(
            name='Cash', x=[portfolio['cash_amt']], y=['Portfolio'],
            orientation='h', marker_color='rgba(80,80,80,0.6)',
            text=f"  Cash ${portfolio['cash_amt']:,.0f}  ", textposition='inside',
            textfont=dict(size=10, color='rgba(200,210,220,0.8)', family='JetBrains Mono'),
            marker_line=dict(width=0)
        ))
        fig_bar.update_layout(
            barmode='stack', template='plotly_dark', height=100,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title=dict(text='<b>Capital Distribution</b>', font=dict(size=14, color='white')),
            xaxis=dict(title='', tickformat='$,.0f', gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(title='', showticklabels=False),
            margin=dict(l=10, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Download CSV
        csv_data = pd.DataFrame([{
            'Ticker': a['ticker'], 'Stock': a['name'], 'Price': a['current_price'],
            'Technical Score': a['technical_score'],
            'Fundamental Score': a.get('fundamental_score', ''),
            'Combined Score': a['combined_score'], 'Signal': a['combined_signal'],
            'Weight (%)': a['weight_pct'], 'Capital ($)': a['capital'], 'Shares': a['shares'],
            'Predicted Return (%)': a['predicted_return'], 'Uncertainty (%)': a['uncertainty'],
        } for a in portfolio['allocations']])
        csv_data.loc[len(csv_data)] = ['CASH', 'Cash Reserve', '', '', '', '', 'N/A',
                                        portfolio['cash_pct'], portfolio['cash_amt'], '', 0, 0]

        st.download_button("📥 Download Portfolio CSV", csv_data.to_csv(index=False),
                          "portfolio.csv", "text/csv", use_container_width=True)

        # --- Step 6: BACKTESTING ---
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div class="section-header backtest">
            <span class="section-icon">📈</span>
            <div><div class="section-title">Step 6: Historical Backtest</div>
            <div class="section-subtitle">How would this portfolio have performed over the past 12 months?</div></div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Running backtest on historical data..."):
            bt = run_backtest(portfolio['allocations'], capital, lookback_months=12)

        if bt:
            # Backtest metrics
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            with bc1:
                clr = '#38ef7d' if bt['total_return'] >= 0 else '#ff6b6b'
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="background:{clr};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{bt['total_return']:+.1f}%</div>
                    <div class="label">Portfolio Return</div>
                </div>""", unsafe_allow_html=True)
            with bc2:
                clr2 = '#38ef7d' if bt['bench_return'] >= 0 else '#ff6b6b'
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="background:{clr2};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{bt['bench_return']:+.1f}%</div>
                    <div class="label">S&P 500 Return</div>
                </div>""", unsafe_allow_html=True)
            with bc3:
                aclr = '#38ef7d' if bt['alpha'] >= 0 else '#ff6b6b'
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="background:{aclr};-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{bt['alpha']:+.1f}%</div>
                    <div class="label">Alpha (vs S&P)</div>
                </div>""", unsafe_allow_html=True)
            with bc4:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{bt['max_drawdown']:.1f}%</div>
                    <div class="label">Max Drawdown</div>
                </div>""", unsafe_allow_html=True)
            with bc5:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{bt['sharpe']:.2f}</div>
                    <div class="label">Backtest Sharpe</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Equity curve chart
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt['port_curve'].index, y=bt['port_curve'].values,
                name='Your Portfolio', mode='lines',
                line=dict(color='#38ef7d', width=2.5),
                fill='tozeroy', fillcolor='rgba(56,239,125,0.08)'
            ))
            if bt['bench_curve'] is not None:
                fig_bt.add_trace(go.Scatter(
                    x=bt['bench_curve'].index, y=bt['bench_curve'].values,
                    name='S&P 500', mode='lines',
                    line=dict(color='#667eea', width=2, dash='dash')
                ))
            # Starting capital line
            fig_bt.add_hline(y=capital, line_dash='dot', line_color='rgba(255,255,255,0.3)',
                           annotation_text=f'Initial: ${capital:,.0f}')
            fig_bt.update_layout(
                title=f'Portfolio Performance — Past {bt["days"]} Trading Days',
                template='plotly_dark', height=420,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title='Portfolio Value ($)', tickformat='$,.0f'),
                xaxis=dict(title=''),
                legend=dict(orientation='h', y=1.12),
                margin=dict(l=60, r=20, t=60, b=40),
                hovermode='x unified'
            )
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.info("⚠️ Insufficient historical data for backtesting.")

        # --- Save portfolio to MongoDB ---
        if _db and _db.is_connected() and user_email:
            try:
                clean_bt = None
                if bt:
                    # Remove pandas Series (not serializable) before saving to DB
                    clean_bt = {k: v for k, v in bt.items() if k not in ['port_curve', 'bench_curve']}

                _db.save_portfolio(
                    email=user_email,
                    portfolio_data=portfolio,
                    risk_score=risk_score,
                    risk_category=risk_cat,
                    capital=capital,
                    allocations=portfolio['allocations'],
                    backtest=clean_bt,
                )
                st.success("💾 Portfolio saved to your account!")
            except Exception as e:
                st.warning(f"⚠️ Could not save portfolio: {e}")

    else:
        st.warning("⚠️ Insufficient qualified stocks for portfolio construction.")

    # Disclaimer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="disclaimer">
        ⚠️ <b>Disclaimer</b>: This is an AI-generated advisory for educational purposes.
        Past performance does not guarantee future results. Please consult a qualified
        financial advisor before making investment decisions.
    </div>""", unsafe_allow_html=True)

  else:
    # Landing page when not generating
    st.markdown("""
    <div class="step-card">
        <h3>🔄 How It Works</h3>
        <p style="color: rgba(200,210,220,0.9);">Complete your investor profile in the sidebar, then click <b style="color:#FFD200;">Generate Portfolio</b> to run the full AI pipeline:</p>
        <br>
        <p style="text-align: center;">
        <span class="workflow-step step-pending">1️⃣ Risk Profiling</span>
        <span class="step-arrow">→</span>
        <span class="workflow-step step-pending">2️⃣ Risk Score</span>
        <span class="step-arrow">→</span>
        <span class="workflow-step step-pending">3️⃣ Stock Analysis</span>
        <span class="step-arrow">→</span>
        <span class="workflow-step step-pending">4️⃣ Allocation</span>
        <span class="step-arrow">→</span>
        <span class="workflow-step step-pending">5️⃣ Portfolio</span>
        <span class="step-arrow">→</span>
        <span class="workflow-step step-pending">6️⃣ Backtest</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="module-card risk">
            <span class="module-icon">🎯</span>
            <h3>Risk Prediction</h3>
            <p>RandomForest model trained on Federal Reserve Survey of Consumer Finances data. Maps 19 demographic & financial features to a risk tolerance score (0–100).</p>
            <span class="module-stat">R² = 0.926 · MAE = 3.66</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="module-card tech">
            <span class="module-icon">📈</span>
            <h3>Technical Analysis</h3>
            <p>LSTM + Multi-Head Attention ensemble (5 models) trained on S&P 500 stocks. Predicts multi-horizon returns using 15 technical indicators.</p>
            <span class="module-stat">74.8% Dir Acc · r=0.329</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="module-card fund">
            <span class="module-icon">📊</span>
            <h3>Fundamental Analysis</h3>
            <p>Random Forest on SEC EDGAR XBRL filings. Scores companies on 20 fundamental ratios — margins, growth, leverage, and quality composites.</p>
            <span class="module-stat">IC = 0.085 · Sharpe = 0.445</span>
        </div>""", unsafe_allow_html=True)

# ============================================================================
# TAB 2: MODEL MANAGEMENT
# ============================================================================
with tab_models:
    st.markdown("""<div class="section-header analysis">
        <span class="section-icon">🔧</span>
        <div><div class="section-title">Model Management</div>
        <div class="section-subtitle">Retrain models on live data & track versions</div></div>
    </div>""", unsafe_allow_html=True)

    col_tech, col_fund = st.columns(2)

    with col_tech:
        st.markdown('<div class="analysis-col-header tech"><span>📈 Technical Model (LSTM Ensemble)</span></div>', unsafe_allow_html=True)

        tech_mode = st.radio("Retraining Mode", ["Incremental (fast ~30min)", "Full (slow ~3hr)"],
                             key="tech_mode", horizontal=True)
        tech_stocks = st.slider("Max Stocks", 5, 80, 30, key="tech_stocks")

        if st.button("🔄 Retrain Technical Model", use_container_width=True, key="btn_retrain_tech"):
            mode = "incremental" if "Incremental" in tech_mode else "full"
            with st.spinner(f"Retraining technical model ({mode})..."):
                try:
                    from retrain import retrain_technical
                    result = retrain_technical(mode=mode, max_stocks=tech_stocks)
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"✅ Retrained! Version: **{result['version']}**")
                        if _db and _db.is_connected():
                            _db.save_model_version("technical", result["version"],
                                                   result.get("metrics", {}), result.get("path", ""))
                except Exception as e:
                    st.error(f"❌ Retraining failed: {e}")

    with col_fund:
        st.markdown('<div class="analysis-col-header fund"><span>📊 Fundamental Model (Random Forest)</span></div>', unsafe_allow_html=True)
        st.info("Fundamental model always does a full retrain (~2 min). It re-downloads SEC EDGAR filings.")

        if st.button("🔄 Retrain Fundamental Model", use_container_width=True, key="btn_retrain_fund"):
            with st.spinner("Retraining fundamental model..."):
                try:
                    from retrain import retrain_fundamental
                    result = retrain_fundamental()
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"✅ Retrained! Version: **{result['version']}** (IC={result.get('val_ic', 0):.4f})")
                        if _db and _db.is_connected():
                            _db.save_model_version("fundamental", result["version"],
                                                   result.get("metrics", {}), result.get("path", ""))
                except Exception as e:
                    st.error(f"❌ Retraining failed: {e}")

    # Model version history
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📦 Model Version History")

    try:
        from retrain import list_model_versions
        versions = list_model_versions()
        if versions:
            vdf = pd.DataFrame([{
                "Version": v["version"],
                "Type": v["type"].title(),
                "Mode": v["mode"].title(),
                "Trained At": v["trained_at"][:19].replace("T", " ") if v["trained_at"] else "—",
            } for v in versions])
            st.dataframe(vdf, use_container_width=True, hide_index=True)
        else:
            st.info("No model versions saved yet. Retrain a model to create the first version.")
    except Exception:
        st.info("No model versions found. Retrain a model to get started.")

    # MongoDB model versions
    if _db and _db.is_connected():
        st.markdown("##### ☁️ Cloud-Tracked Versions (MongoDB)")
        db_versions = _db.get_model_versions(limit=10)
        if db_versions:
            dbvdf = pd.DataFrame([{
                "Version": v.get("version", "—"),
                "Type": v.get("model_type", "—").title(),
                "Trained At": str(v.get("trained_at", "—"))[:19],
            } for v in db_versions])
            st.dataframe(dbvdf, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: PORTFOLIO HISTORY
# ============================================================================
with tab_history:
    st.markdown("""<div class="section-header portfolio">
        <span class="section-icon">📜</span>
        <div><div class="section-title">Portfolio History</div>
        <div class="section-subtitle">View, compare, and rebalance your previous portfolios</div></div>
    </div>""", unsafe_allow_html=True)

    if not user_email:
        st.info("🔑 Enter your email in the sidebar to view your portfolio history.")
    elif not _db or not _db.is_connected():
        st.warning("📦 MongoDB not connected. Configure your `.env` file to enable portfolio storage.")
    else:
        portfolios = _db.get_user_portfolios(user_email, limit=10)
        if not portfolios:
            st.info("No portfolios saved yet. Generate a portfolio in the **🚀 Portfolio Generator** tab.")
        else:
            st.markdown(f"**{len(portfolios)}** portfolio(s) found for `{user_email}`")

            for idx, p in enumerate(portfolios):
                status_emoji = {"active": "🟢", "archived": "📦", "rebalanced": "🔄"}.get(p.get("status"), "⚪")
                created = str(p.get("created_at", ""))[:16]
                metrics = p.get("portfolio_metrics", {})
                ret_str = f"{metrics.get('port_ret', 0):+.1f}%" if metrics.get('port_ret') else "—"
                sharpe_str = f"{metrics.get('sharpe', 0):.2f}" if metrics.get('sharpe') else "—"

                with st.expander(f"{status_emoji} Portfolio — {created} | Return: {ret_str} | Sharpe: {sharpe_str} | Status: {p.get('status', '?').title()}"):
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    with pm1:
                        st.metric("Capital", f"${p.get('capital', 0):,.0f}")
                    with pm2:
                        st.metric("Risk Score", f"{p.get('risk_score', 0):.1f}")
                    with pm3:
                        st.metric("Equity %", f"{metrics.get('eq_pct', 0):.0f}%")
                    with pm4:
                        st.metric("Status", p.get("status", "—").title())

                    allocs = p.get("allocations", [])
                    if allocs:
                        adf = pd.DataFrame([{
                            "Ticker": a["ticker"],
                            "Name": a.get("name", ""),
                            "Weight": f"{a.get('weight_pct', 0):.1f}%",
                            "Capital": f"${a.get('capital', 0):,.0f}",
                            "Shares": a.get("shares", 0),
                            "Entry Price": f"${a.get('entry_price', 0):,.2f}",
                            "Signal": a.get("signal", "—"),
                        } for a in allocs])
                        st.dataframe(adf, use_container_width=True, hide_index=True)

                    bt = p.get("backtest")
                    if bt:
                        st.caption(f"Backtest: Return {bt.get('total_return', 0):+.1f}% | Alpha {bt.get('alpha', 0):+.1f}% | Sharpe {bt.get('sharpe', 0):.2f}")

                    if p.get("status") == "active":
                        st.markdown("---")
                        if st.button(f"🔄 Analyze & Rebalance", key=f"reb_{p['_id']}", use_container_width=True):
                            st.info(f"Rebalancing portfolio from {created}.")
                            
                            # Real Rebalancing Logic
                            with st.spinner("1️⃣ Running Technical Analysis..."):
                                tech_res = run_technical_analysis(st.progress(0), st.empty(), horizon_days=90)
                            with st.spinner("2️⃣ Running Fundamental Analysis..."):
                                fund_res = run_fundamental_analysis(st.progress(0), st.empty())
                            with st.spinner("3️⃣ Calculating Optimal Weights..."):
                                combined = combine_scores(tech_res, fund_res)
                                new_port = allocate_portfolio(combined, p.get("risk_score", 50), p.get("capital", 100000))
                            
                            if not new_port['allocations']:
                                st.warning("Not enough qualified stocks to rebalance right now.")
                            else:
                                st.markdown("#### ⚖️ Proposed Rebalancing Trades")
                                
                                old_allocs = {a['ticker']: a for a in allocs}
                                new_allocs = {a['ticker']: a for a in new_port['allocations']}
                                
                                all_tickers = set(old_allocs.keys()).union(set(new_allocs.keys()))
                                reb_data = []
                                
                                for t in all_tickers:
                                    old_w = old_allocs.get(t, {}).get("weight_pct", 0.0)
                                    new_w = new_allocs.get(t, {}).get("weight_pct", 0.0)
                                    
                                    if new_w > old_w + 1.0:
                                        action = "BUY"
                                    elif new_w < old_w - 1.0:
                                        action = "SELL"
                                    else:
                                        action = "HOLD"
                                        
                                    if action != "HOLD" or old_w > 0:
                                        reb_data.append({
                                            "Ticker": t,
                                            "Current Weight": f"{old_w:.1f}%",
                                            "Target Weight": f"{new_w:.1f}%",
                                            "Target Capital": f"${new_allocs.get(t, {}).get('capital', 0):,.0f}",
                                            "Action": action
                                        })
                                
                                r_df = pd.DataFrame(reb_data).sort_values("Action")
                                st.dataframe(r_df, use_container_width=True, hide_index=True)
                                
                                if st.button("Confirm Rebalance & Update Portfolio", key=f"conf_{p['_id']}"):
                                    if _db and _db.is_connected():
                                        _db.update_portfolio_in_place(
                                            portfolio_id=str(p["_id"]),
                                            portfolio_data=new_port,
                                            allocations=new_port['allocations'],
                                            backtest=None
                                        )
                                        st.success("✅ Portfolio successfully rebalanced and updated! Reload the page to see changes.")
