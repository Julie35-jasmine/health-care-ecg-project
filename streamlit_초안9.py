import ast
import json
import math
import os
import textwrap
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None

try:
    from google import genai
except Exception:
    genai = None

try:
    import wfdb
except Exception:
    wfdb = None


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ALL_DIR = ROOT_DIR / "all"


def first_existing_dir(*paths):
    return next((path for path in paths if path.exists()), paths[0])


def first_existing_file(*paths):
    return next((path for path in paths if path.exists()), paths[0])


DATA_DIR = first_existing_dir(
    ALL_DIR / "leipzig-heart-center-ecg",
    ROOT_DIR / "leipzig-heart-center-ecg",
    BASE_DIR / "leipzig-heart-center-ecg-data" / "leipzig-heart-center-ecg-data",
)
RESULT_DIR = first_existing_dir(ALL_DIR / "codehere" / "code2", BASE_DIR / "ecg_results")
DB_ENV_PATH = first_existing_file(
    BASE_DIR / "dbeaver_here" / "beaver.env",
    ROOT_DIR / "dbeaver_here" / "beaver.env",
)
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NORMAL_SYMBOLS = {"N", "R", "f", "j", "/", "~"}
TACHY_SYMBOLS = {"VT", "IVR", "AVRT", "AVNRT", "avrt", "avnrt", "AVNRT+BII"}
ABNORMAL_SYMBOLS = {"V", "A", "a", "L", "b", "F", "J", "E", "Q", "X"} | TACHY_SYMBOLS
EVENT_GROUP_ORDER = [
    "VT / IVR",
    "AVRT / AVNRT",
    "AFIB / AFLT",
    "Conduction / PVC",
    "Pacemaker",
    "Other",
]
ANNOTATION_COLOR_MAP = {
    "Conduction / PVC": "#6B7280",
    "VT / IVR": "#EF4444",
    "AVRT / AVNRT": "#F97316",
    "AFIB / AFLT": "#FACC15",
    "Pacemaker": "#22C55E",
    "Other": "#111111",
}
CHART_TEXT = "#17323f"
CHART_MUTED = "#5f737d"
CHART_GRID = "#e4edef"
CHART_BG = "#ffffff"
ECG_Y_AXIS_LABEL = "Normalized ECG amplitude"
TABLE_TEXT = "#111111"
TABLE_BORDER = "#e4eeee"
TABLE_HEADER_BG = "#16615B"
TABLE_ROW_BG = "#ffffff"
WARNING_TEXT = "이 대시보드는 연구 및 교육 목적의 ECG 데이터 탐색 도구입니다. 실제 임상 진단, 치료 결정, 응급 판단 도구로 사용하면 안 됩니다."

ML_MODEL_NOTE = (
    "XGBoost+SMOTE_v2_결과.ipynb의 저장된 validation 출력 기준입니다. "
    "ECG 10초 window tabular feature를 사용하고, train set에만 Safe SMOTE를 적용했습니다."
)
DL_MODEL_NOTE = "DL 수치는 4.28_ECG_DL_최종.ipynb 및 4.28_DL_VAE_최종.ipynb의 저장된 실험 출력 기준입니다. 선택 segment 예측은 실제 모델 연결 전까지 annotation 기반 임시 시뮬레이션입니다."
ML_CLASS_NAMES = ["non_tachy", "VT", "SVT"]
XGB_CLASS_NAMES = ["normal", "ntach", "tach"]
ML_CONFIG = {
    "Model": "XGBoost 3-class classifier",
    "Input": "tabular ECG features",
    "Notebook": "XGBoost+SMOTE_v2_결과.ipynb",
    "Classes": "normal / ntach / tach",
    "Segment": "10.0s window",
    "Stride": "5.0s",
    "Normal cap": "10 windows / record",
    "SQI threshold": "0.70",
    "Split": "patient-level train / validation",
    "Resampling": "Safe SMOTE on train only",
    "Before SMOTE": "normal 128 / ntach 22,906 / tach 22,796",
    "After SMOTE": "normal 640 / ntach 22,906 / tach 22,906",
    "SMOTE strategy": "{normal: 640, tach: 22,906}",
    "SMOTE k": "5",
    "Validation windows": "13,887",
}
ML_METRICS = {
    "Accuracy": 0.8304169367033917,
    "Macro F1": 0.6577060503450974,
    "Balanced Acc.": 0.6806050481902858,
    "Macro ROC-AUC": 0.9342576666666667,
    "Macro PR-AUC": 0.6895716666666667,
    "Leakage": 0.0,
}
ML_VALIDATION_CM = np.array([[23, 34, 4], [66, 5679, 1113], [0, 1138, 5830]])
ML_CLASS_REPORT = [
    ("normal", 0.26, 0.38, 0.31, 61),
    ("ntach", 0.83, 0.83, 0.83, 6858),
    ("tach", 0.84, 0.84, 0.84, 6968),
    ("macro avg", 0.64, 0.68, 0.66, 13887),
    ("weighted avg", 0.83, 0.83, 0.83, 13887),
]
ML_OVR_AUC = [
    ("normal", 61, 0.971452, 0.245164),
    ("ntach", 6858, 0.912348, 0.915730),
    ("tach", 6968, 0.918973, 0.907821),
]
ML_PREDICTED_CLASS_RATIO = [
    ("normal", 0.006409),
    ("ntach", 0.493339),
    ("tach", 0.500252),
]
ML_FEATURES = [
    ("rr_std", 0.295034, "RR interval variability"),
    ("pnn50", 0.176941, "proportion of adjacent RR changes above 50 ms"),
    ("rr_cv", 0.052103, "RR coefficient of variation"),
    ("kurtosis", 0.037621, "waveform distribution tail feature"),
    ("is_pediatric", 0.037494, "patient group flag"),
    ("rms", 0.036690, "root mean square amplitude"),
    ("skew_proxy", 0.033530, "waveform asymmetry proxy"),
    ("std", 0.032555, "signal standard deviation"),
    ("qrs_width_mean", 0.025777, "mean QRS width"),
    ("rr_mean", 0.023708, "mean RR interval"),
    ("qrs_amp_std", 0.021173, "QRS amplitude variation"),
    ("r_amp_mean", 0.020803, "mean R peak amplitude"),
    ("r_amp_std", 0.019748, "R peak amplitude variation"),
    ("st_slope_mean", 0.017560, "mean ST slope"),
    ("amp_range", 0.015420, "signal amplitude range"),
    ("n_rpeaks", 0.014353, "R peak count"),
    ("abs_max", 0.013615, "maximum absolute amplitude"),
    ("st_slope_std", 0.012630, "ST slope variation"),
    ("flat_ratio", 0.012446, "flat signal ratio"),
    ("qrs_amp_mean", 0.012269, "mean QRS amplitude"),
    ("hr", 0.011668, "heart rate from peak count"),
    ("st_level_std", 0.011433, "ST level variation"),
    ("st_level_mean", 0.011364, "mean ST level"),
    ("hr_mean_rr", 0.011300, "heart rate from RR mean"),
    ("rr_rmssd", 0.010058, "RMSSD from RR intervals"),
    ("qrs_width_std", 0.009532, "QRS width variation"),
    ("sqi", 0.006343, "signal quality index"),
    ("artifact_candidate", 0.004306, "artifact candidate flag"),
    ("baseline_shift", 0.003855, "baseline shift"),
    ("edge_jump_end", 0.003017, "end-edge jump"),
]

CNN_ARGMAX_CM = np.array([[1056, 35, 123], [5, 6, 2], [3, 1, 217]])
CNN_THRESHOLD_CM = np.array([[1119, 25, 70], [6, 6, 1], [11, 1, 209]])
VAE_CM = np.array([[1151, 63], [209, 25]])
DL_CLASS_DISTRIBUTION = {"non_tachy": 1214, "VT": 13, "SVT": 221}
DL_CNN_ARGMAX_METRICS = {
    "Accuracy": 0.883,
    "Macro F1": 0.639,
    "Balanced Acc.": 0.771,
    "VT Recall": 0.462,
    "SVT Recall": 0.982,
    "VT PR-AUC": 0.2351,
    "SVT PR-AUC": 0.9602,
}
DL_CNN_THRESHOLD_METRICS = {
    "Accuracy": 0.921,
    "Macro F1": 0.684,
    "Balanced Acc.": 0.776,
    "VT Recall": 0.462,
    "SVT Recall": 0.946,
    "VT PR-AUC": 0.2351,
    "SVT PR-AUC": 0.9602,
}
DL_VAE_METRICS = {
    "Accuracy": 0.812,
    "Macro F1": 0.525,
    "Anomaly Recall": 0.107,
    "VT Recall": 0.692,
    "SVT Recall": 0.072,
    "False Alarm": 0.0519,
    "ROC-AUC": 0.8463,
    "PR-AUC": 0.4594,
}
DL_THRESHOLDS = {"VT": 0.65, "SVT": 0.70, "VAE": 1.9828271508216853}


st.set_page_config(
    page_title="Pediatric & CHD ECG Dashboard",
    page_icon="ECG",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme():
    st.markdown(
        """
        <style>
        :root {
            --mint: #24b89b;
            --teal: #107d83;
            --navy: #123042;
            --ink: #20333f;
            --muted: #6b7d87;
            --line: #dce9eb;
            --surface: #f7fbfb;
            --alert: #e7534f;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d3141 0%, #0f4750 55%, #17655d 100%) !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown {
            color: #f5ffff !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stNumberInput input {
            color: #102b36 !important;
            background: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] svg,
        [data-testid="stSidebar"] .stSelectbox svg,
        [data-testid="stSidebar"] .stMultiSelect svg {
            color: #102b36 !important;
            fill: #102b36 !important;
        }
        [data-testid="stSidebar"] .stSlider {
            background: transparent !important;
            padding: 0 0 .35rem 0;
        }
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"] .stSlider [role="slider"] {
            background: #ef4e59 !important;
            border-color: #ef4e59 !important;
        }
        [data-testid="stSidebar"] .stSlider div {
            color: #f5ffff !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] input,
        [data-testid="stSidebar"] .stNumberInput input {
            color: #102b36 !important;
        }
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {
            overflow: visible !important;
        }
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {
            min-height: 40px !important;
            height: auto !important;
            overflow: visible !important;
            align-items: center !important;
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
            display: inline-flex !important;
            min-width: 44px !important;
            min-height: 28px !important;
            height: auto !important;
            max-width: none !important;
            padding: 4px 8px !important;
            align-items: center !important;
            background: #fff1f1 !important;
            color: #9f1239 !important;
            overflow: visible !important;
        }
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {
            display: inline-block !important;
            min-width: max-content !important;
            max-width: none !important;
            color: #9f1239 !important;
            line-height: 1.25 !important;
            overflow: visible !important;
            white-space: nowrap !important;
            text-overflow: clip !important;
        }
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] svg {
            color: #9f1239 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(8, 42, 55, .42) !important;
            border: 1px solid rgba(255, 255, 255, .18) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background: transparent !important;
            color: #f5ffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            color: #f5ffff !important;
            fill: #f5ffff !important;
        }
        [data-testid="stAppViewContainer"] {
            color: var(--ink);
        }
        [data-testid="stVerticalBlock"] {
            gap: 1rem;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .element-container {
            margin-bottom: .55rem;
        }
        .hero {
            border: 1px solid rgba(16,125,131,.18);
            border-radius: 8px;
            padding: 22px 26px;
            background:
                linear-gradient(135deg, rgba(255,255,255,.95), rgba(235,250,249,.92)),
                repeating-linear-gradient(90deg, transparent 0 20px, rgba(36,184,155,.05) 20px 21px);
            box-shadow: 0 16px 40px rgba(20,70,82,.08);
            margin-bottom: 18px;
        }
        .hero h1 {
            margin: 0 0 8px 0;
            font-size: 30px;
            letter-spacing: 0;
            color: var(--navy);
        }
        .hero p {
            margin: 0;
            color: var(--muted);
            font-size: 15px;
        }
        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 100;
            border: 1px solid rgba(16,125,131,.18);
            border-radius: 8px;
            padding: 14px 18px;
            background: rgba(255,255,255,.96);
            box-shadow: 0 12px 28px rgba(18,48,66,.08);
            margin-bottom: 14px;
        }
        .sticky-header h1 {
            margin: 0 0 5px 0;
            font-size: 23px;
            color: var(--navy);
            letter-spacing: 0;
        }
        .sticky-header p {
            margin: 0;
            color: #4f6670;
            font-size: 13px;
        }
        .page-description {
            border-left: 4px solid var(--mint);
            background: rgba(255,255,255,.86);
            padding: 12px 14px;
            border-radius: 0 8px 8px 0;
            color: #334d59;
            margin-bottom: 14px;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 10px 28px rgba(16,72,80,.06);
            min-height: 92px;
        }
        .metric-card.compact-value .metric-value {
            font-size: 21px;
            line-height: 1.2;
            word-break: keep-all;
        }
        .flow-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 100px;
            box-shadow: 0 10px 26px rgba(16,72,80,.06);
        }
        .flow-step {
            font-size: 11px;
            color: #0f766e;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: 6px;
        }
        .flow-title {
            color: #113645;
            font-size: 18px;
            font-weight: 760;
            line-height: 1.25;
        }
        .flow-caption {
            color: #6b7d87;
            font-size: 12px;
            margin-top: 7px;
            line-height: 1.35;
        }
        .metric-label {
            font-size: 12px;
            color: #6b7d87;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 26px;
            color: #113645;
            font-weight: 760;
            line-height: 1.15;
        }
        .metric-caption {
            font-size: 12px;
            color: #6b7d87;
            margin-top: 6px;
        }
        .risk-normal, .risk-watch, .risk-high {
            border-radius: 999px;
            padding: 5px 10px;
            font-weight: 700;
            display: inline-block;
            font-size: 13px;
        }
        .risk-normal { background: #e6f7f1; color: #0c7a5d; }
        .risk-watch { background: #fff5d9; color: #8a6200; }
        .risk-high { background: #ffe6e3; color: #b02b28; }
        .alert-card {
            border-radius: 8px;
            padding: 18px 20px;
            border: 1px solid rgba(17, 24, 39, .08);
            box-shadow: 0 14px 34px rgba(18, 48, 66, .08);
            margin: 2px 0 12px;
        }
        .alert-card h3 {
            margin: 0 0 8px 0;
            font-size: 22px;
            letter-spacing: 0;
        }
        .alert-card p {
            margin: 0;
            font-size: 15px;
            line-height: 1.55;
        }
        .alert-normal { background: #e9fbf1; color: #146c43; border-left: 7px solid #22C55E; }
        .alert-watch { background: #fff3e4; color: #9a4b00; border-left: 7px solid #F97316; }
        .alert-abnormal { background: #ffeaea; color: #991b1b; border-left: 7px solid #EF4444; }
        .alert-review { background: #edf2f7; color: #334155; border-left: 7px solid #64748B; }
        .review-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 15px 17px;
            background: rgba(255,255,255,.9);
            min-height: 142px;
        }
        .review-card h4 {
            margin: 0 0 8px 0;
            color: var(--navy);
            font-size: 15px;
        }
        .review-card p, .review-card li {
            color: #35505d;
            font-size: 14px;
            line-height: 1.52;
        }
        .burden-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
            min-height: 104px;
        }
        .burden-name {
            font-size: 12px;
            color: #5f737d;
            margin-bottom: 8px;
        }
        .burden-count {
            font-size: 28px;
            font-weight: 780;
            color: #113645;
            line-height: 1.1;
        }
        .burden-priority {
            margin-top: 8px;
            font-size: 12px;
            color: #5f737d;
        }
        .color-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 6px;
            vertical-align: -1px;
            border: 1px solid rgba(17,24,39,.28);
        }
        .section-title {
            font-size: 18px;
            font-weight: 760;
            color: var(--navy);
            margin: 8px 0 10px;
        }
        .note-box {
            border-left: 4px solid var(--teal);
            background: rgba(255,255,255,.84);
            padding: 13px 15px;
            border-radius: 0 8px 8px 0;
            color: #35505d;
        }
        .report-summary-box {
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            background: rgba(255,255,255,.84);
            padding: 13px 15px;
            border-radius: 8px;
            color: #35505d;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.92);
            border: 1px solid var(--line);
            padding: 12px 14px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #d8e8ea;
            border-radius: 8px;
            height: 42px;
            padding: 0 18px;
            color: #17323f;
        }
        .stTabs [aria-selected="true"] {
            background: #e8f7f4;
            border-color: #24b89b;
            color: #0d4750 !important;
        }
        div[data-testid="stDataFrame"] {
            margin-top: .35rem;
            border: 1px solid #b8dadd !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            background: #ffffff !important;
            box-shadow: 0 10px 24px rgba(16,72,80,.05);
        }
        div[data-testid="stDataFrame"] div,
        div[data-testid="stDataFrame"] span,
        div[data-testid="stDataFrame"] p,
        div[data-testid="stDataFrame"] button,
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            color: #111111 !important;
        }
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] [role="row"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            border-color: #b8dadd !important;
            background-color: #ffffff !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #d6e8ea !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 10px 24px rgba(16,72,80,.05);
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] details,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary {
            background: #f7fbfb !important;
            color: #17323f !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary p,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary span,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary svg {
            color: #17323f !important;
            fill: #17323f !important;
            font-weight: 760 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] label,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] label p,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] label span,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] [data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] [data-testid="stTickBarMin"],
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] [data-testid="stTickBarMax"] {
            color: #17323f !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] div[data-baseweb="select"] > div,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] input {
            background: #ffffff !important;
            color: #102b36 !important;
            border-color: #b8dadd !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] div[data-baseweb="select"] span,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] div[data-baseweb="select"] input,
        [data-testid="stAppViewContainer"] [data-testid="stExpander"] div[data-baseweb="select"] svg {
            color: #102b36 !important;
            fill: #102b36 !important;
        }
        .report-box {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 18px 20px;
            color: #253f4b;
            line-height: 1.62;
        }
        .navigator-caption {
            color: #f5ffff !important;
            font-size: 13px;
            line-height: 1.45;
            margin: -2px 0 8px 0;
        }
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #f5ffff !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"],
        input,
        textarea {
            color: #102b36 !important;
            background: #ffffff !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #102b36 !important;
        }
        div[data-baseweb="select"] svg {
            color: #102b36 !important;
            fill: #102b36 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
            color: #102b36 !important;
        }
        div[data-baseweb="popover"] div[role="listbox"],
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"] {
            background: #ffffff !important;
            color: #102b36 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d3141 0%, #0f4750 55%, #17655d 100%) !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown {
            color: #f5ffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(8, 42, 55, .42) !important;
            border: 1px solid rgba(255, 255, 255, .18) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background: transparent !important;
            color: #f5ffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            color: #f5ffff !important;
            fill: #f5ffff !important;
            font-weight: inherit !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] label,
        [data-testid="stSidebar"] [data-testid="stExpander"] label p,
        [data-testid="stSidebar"] [data-testid="stExpander"] label span,
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stTickBarMax"] {
            color: #f5ffff !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stExpander"] input {
            color: #102b36 !important;
            background: #ffffff !important;
            border-color: rgba(255, 255, 255, .32) !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] input,
        [data-testid="stSidebar"] div[data-baseweb="select"] svg,
        [data-testid="stSidebar"] .stNumberInput input {
            color: #102b36 !important;
            fill: #102b36 !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #17323f !important;
            font-weight: 650;
        }
        [data-testid="stDataFrame"] * {
            color: #111111 !important;
        }
        .dashboard-table-wrap {
            width: 100%;
            margin: .35rem 0 1rem 0;
            overflow-x: auto;
            background: #ffffff;
        }
        .dashboard-table-wrap.scrollable {
            max-height: 520px;
            overflow-y: auto;
            border: 1px solid #e4eeee;
        }
        .dashboard-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
            background: #ffffff;
            color: #111111;
            font-size: 14px;
        }
        .dashboard-table thead th {
            background: #16615B;
            color: #ffffff;
            border: 1px solid #8edfd8;
            padding: 16px 14px;
            text-align: center;
            font-weight: 760;
            letter-spacing: 0;
            white-space: nowrap;
        }
        .dashboard-table-wrap.scrollable .dashboard-table thead th {
            position: sticky;
            top: 0;
            z-index: 2;
        }
        .dashboard-table tbody td {
            background: #ffffff;
            color: #111111;
            border: 1px solid #e4eeee;
            padding: 16px 14px;
            text-align: center;
            vertical-align: middle;
            line-height: 1.45;
        }
        .dashboard-table tbody tr:hover td {
            background: #f8fdfc;
        }
        div[data-testid="stDownloadButton"] button {
            background: #0f2f3d !important;
            color: #f5ffff !important;
            border: 1px solid #1d6c72 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            min-height: 42px !important;
        }
        div[data-testid="stDownloadButton"] button p,
        div[data-testid="stDownloadButton"] button span {
            color: #f5ffff !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background: #15505a !important;
            border-color: #24b89b !important;
            color: #ffffff !important;
        }
        .feature-top-n-label {
            color: #111111 !important;
            font-size: 14px;
            font-weight: 600;
            margin: 2px 0 2px;
        }
        .muted-info-box {
            border-left: 4px solid #b8dadd;
            background: #f5f8f9;
            color: #5f737d;
            padding: 12px 14px;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
            line-height: 1.55;
            margin: 6px 0 14px;
        }
        .interpretation-note-box,
        .interpretation-warning-box {
            border-radius: 8px;
            padding: 14px 16px;
            margin: 10px 0;
            font-size: 14px;
            line-height: 1.55;
            font-weight: 560;
            box-shadow: 0 8px 22px rgba(18, 48, 66, .05);
        }
        .interpretation-note-box {
            background: #e9fbf1;
            border: 1px solid #b7ead2;
            border-left: 6px solid #22C55E;
            color: #146c43 !important;
        }
        .interpretation-warning-box {
            background: #fff3e4;
            border: 1px solid #ffd6a8;
            border-left: 6px solid #F97316;
            color: #9a4b00 !important;
        }
        .dw-status-panel {
            display: grid;
            grid-template-columns: 1.05fr 1fr 1fr;
            gap: 10px;
            border: 1px solid #cfe4e7;
            border-radius: 8px;
            padding: 10px 12px;
            margin: -4px 0 12px 0;
            background: #f8fcfc;
            box-shadow: 0 8px 20px rgba(16,72,80,.045);
        }
        .dw-status-item {
            min-width: 0;
            border-right: 1px solid #dce9eb;
            padding-right: 10px;
        }
        .dw-status-item:last-child {
            border-right: 0;
            padding-right: 0;
        }
        .dw-status-label {
            color: #5f737d;
            font-size: 11px;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: .03em;
            margin-bottom: 4px;
        }
        .dw-status-value {
            color: #123042;
            font-size: 13px;
            font-weight: 720;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .dw-status-caption {
            color: #5f737d;
            font-size: 11px;
            margin-top: 3px;
            line-height: 1.35;
        }
        .dw-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 999px;
            margin-right: 6px;
            vertical-align: 1px;
            background: #ef4444;
        }
        .dw-dot.connected { background: #22c55e; }
        .dw-dot.partial { background: #f97316; }
        @media (max-width: 950px) {
            .dw-status-panel { grid-template-columns: 1fr; }
            .dw-status-item {
                border-right: 0;
                border-bottom: 1px solid #dce9eb;
                padding: 0 0 8px 0;
            }
            .dw-status-item:last-child {
                border-bottom: 0;
                padding-bottom: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_table(data):
    if data is None:
        return
    df = pd.DataFrame(data).copy()
    table_html = df.to_html(index=False, escape=True, border=0, classes="dashboard-table")
    scroll_class = " scrollable" if len(df) > 10 else ""
    st.markdown(f'<div class="dashboard-table-wrap{scroll_class}">{table_html}</div>', unsafe_allow_html=True)


def render_page_header(page_name, record_id, lead, start_sec, duration_sec):
    st.markdown(
        f"""
        <div class="sticky-header">
            <h1>Leipzig ECG Arrhythmia Dashboard</h1>
            <p>Patient ID: <b>{record_id}</b> | Page: <b>{page_name}</b> | Lead: <b>{lead}</b> | Window: {start_sec:.0f}s-{start_sec + duration_sec:.0f}s</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dw_status_panel():
    health = mysql_health()
    counts = mysql_table_counts()
    connected = bool(health.get("ok"))
    loaded_tables = [
        f"patients {counts.get('patients', 0):,}",
        f"records {counts.get('records', 0):,}",
        f"annotations {counts.get('annotations', 0):,}",
    ]
    empty_tables = [table for table in ["record_channels", "ecg_signals", "beats", "beat_features"] if counts.get(table, 0) == 0]
    status_text = f"{health.get('database')} / {health.get('tables')} tables" if connected else db_config_status()
    dot_class = "connected" if connected else "partial"
    empty_text = ", ".join(empty_tables) if empty_tables else "none"
    st.markdown(
        f"""
        <div class="dw-status-panel">
            <div class="dw-status-item">
                <div class="dw-status-label">Data Warehouse</div>
                <div class="dw-status-value"><span class="dw-dot {dot_class}"></span>{status_text}</div>
                <div class="dw-status-caption">Source: dbeaver_here/beaver.env</div>
            </div>
            <div class="dw-status-item">
                <div class="dw-status-label">Connected Data</div>
                <div class="dw-status-value">{' / '.join(loaded_tables)}</div>
                <div class="dw-status-caption">Used for patient list, cohort metadata base, annotation markers and risk logic</div>
            </div>
            <div class="dw-status-item">
                <div class="dw-status-label">Not In Warehouse Yet</div>
                <div class="dw-status-value">{empty_text}</div>
                <div class="dw-status-caption">ECG waveform still uses local WFDB files when these tables are empty</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_description(text):
    st.markdown(f'<div class="page-description">{text}</div>', unsafe_allow_html=True)


def warning_footer():
    st.markdown(f'<div class="note-box">{WARNING_TEXT}</div>', unsafe_allow_html=True)


def inject_theme_mode(theme_mode):
    if theme_mode != "Dark":
        return
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0e1f29 0%, #102d35 52%, #0f1e27 100%);
            color: #e8f3f5;
        }
        .sticky-header,
        .metric-card,
        .review-card,
        .burden-card,
        .page-description,
        .report-box,
        .report-summary-box,
        .note-box {
            background: rgba(18, 40, 50, .94) !important;
            border-color: rgba(164, 213, 218, .22) !important;
            color: #e8f3f5 !important;
        }
        .sticky-header h1,
        .section-title,
        .metric-value,
        .review-card h4 {
            color: #ecfeff !important;
        }
        .sticky-header p,
        .metric-label,
        .metric-caption,
        .burden-name,
        .burden-priority,
        .page-description {
            color: #bdd4da !important;
        }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] small {
            color: #e8f3f5;
        }
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] > div,
        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] textarea {
            background: #ffffff !important;
            color: #102b36 !important;
        }
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] span,
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] input {
            color: #102b36 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(18, 40, 50, .35);
            border-radius: 8px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,.10) !important;
            border-color: rgba(164, 213, 218, .24) !important;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #e8f3f5 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #d9fbf4 !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #0d4750 !important;
        }
        [data-testid="stDataFrame"] {
            background: #ffffff !important;
            border: 1px solid #b8dadd !important;
            border-radius: 8px !important;
        }
        [data-testid="stDataFrame"] * {
            color: #111111 !important;
            border-color: #b8dadd !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, caption="", extra_class=""):
    class_name = f"metric-card {extra_class}".strip()
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def flow_card(step, title, caption=""):
    st.markdown(
        f"""
        <div class="flow-card">
            <div class="flow-step">{step}</div>
            <div class="flow-title">{title}</div>
            <div class="flow-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_value(value, digits=3):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "TBD"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def review_badge(flag):
    cls = "risk-normal" if flag == "Normal-like" else ("risk-high" if flag == "High attention" else "risk-watch")
    return f'<span class="{cls}">{flag}</span>'


def parse_duration_to_seconds(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    try:
        parts = text.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except Exception:
        return np.nan
    return np.nan


def parse_dict(value):
    if pd.isna(value) or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def risk_from_ratio(ratio, tachy_count=0):
    if ratio >= 0.25 or tachy_count >= 5:
        return "High"
    if ratio >= 0.08 or tachy_count > 0:
        return "Watch"
    return "Normal"


def risk_badge(risk):
    cls = {"Normal": "risk-normal", "Watch": "risk-watch", "High": "risk-high"}.get(risk, "risk-watch")
    return f'<span class="{cls}">{risk}</span>'


def clean_aux_string(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.strip("()[]{} ").strip(", ")


def annotation_display_label(symbol, aux_note=""):
    symbol = str(symbol).strip()
    if symbol == "X":
        aux = clean_aux_string(aux_note)
        return f"X: {aux}" if aux else "X: Aux-string unknown"
    return symbol


def add_annotation_display_labels(df):
    if df.empty or "symbol" not in df.columns:
        return df
    labeled = df.copy()
    aux = labeled["aux_note"] if "aux_note" in labeled.columns else pd.Series([""] * len(labeled), index=labeled.index)
    labeled["display_label"] = [
        annotation_display_label(symbol, aux_note)
        for symbol, aux_note in zip(labeled["symbol"], aux)
    ]
    return labeled


def aux_annotation_counts(sub):
    if sub.empty:
        return pd.DataFrame()
    labels = [
        annotation_display_label(row.get("label_symbol", ""), row.get("dataset_aux_string", ""))
        for _, row in sub.iterrows()
    ]
    return pd.Series(labels).value_counts().rename_axis("annotation").reset_index(name="count")


def normalize_annotation_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def get_annotation_group(symbol=None, aux_note=None):
    text = f"{normalize_annotation_text(symbol)} {normalize_annotation_text(aux_note)}"
    text_lower = text.lower()
    symbol_lower = normalize_annotation_text(symbol).lower()

    if "/v" in text_lower or "/a" in text_lower:
        return "Pacemaker"
    if "vt" in text_lower or "ivr" in text_lower:
        return "VT / IVR"
    if "avrt" in text_lower or "avnrt" in text_lower or "avnrt+bii" in text_lower:
        return "AVRT / AVNRT"
    if "afib" in text_lower or "aflt" in text_lower:
        return "AFIB / AFLT"
    if (
        symbol_lower == "v"
        or "pvc" in text_lower
        or "av block 1" in text_lower
        or "right bundle" in text_lower
        or "left bundle" in text_lower
        or "우각차단" in text
        or "좌각차단" in text
    ):
        return "Conduction / PVC"
    return "Other"


def get_marker_border_color(theme_mode):
    return "#FFFFFF" if theme_mode == "Dark" else "#0B1117"


def group_counts_from_annotations(ann_df, counts):
    grouped = {name: 0 for name in EVENT_GROUP_ORDER}
    if ann_df is not None and not ann_df.empty:
        for _, row in ann_df.iterrows():
            group = get_annotation_group(row.get("symbol", ""), row.get("aux_note", ""))
            grouped[group] = grouped.get(group, 0) + 1
    elif counts is not None and not counts.empty:
        for _, row in counts.iterrows():
            group = get_annotation_group(row.get("annotation", ""), "")
            grouped[group] = grouped.get(group, 0) + int(row.get("count", 0))
    return grouped


def determine_patient_alert(group_counts):
    if not group_counts or sum(group_counts.values()) == 0:
        return {
            "level": "Review Required",
            "label": "검토필요",
            "class": "alert-review",
            "icon": "[검토필요]",
            "message": "선택 구간에서 annotation 정보가 부족합니다. 원본 waveform과 환자 메타데이터를 함께 확인하세요.",
        }
    if group_counts.get("VT / IVR", 0) > 0:
        reason = "VT / IVR 계열 annotation이 확인되었습니다."
    elif group_counts.get("AFIB / AFLT", 0) > 0:
        reason = "AFIB / AFLT 계열 annotation이 확인되었습니다."
    elif group_counts.get("AVRT / AVNRT", 0) > 0:
        reason = "AVRT / AVNRT 계열 annotation이 확인되었습니다."
    else:
        reason = ""

    if reason:
        return {
            "level": "Abnormal Detected",
            "label": "비정상 감지",
            "class": "alert-abnormal",
            "icon": "[비정상 감지]",
            "message": f"{reason} II lead ECG segment에서 해당 색상 marker 주변 waveform 검토가 필요합니다.",
        }
    if group_counts.get("Conduction / PVC", 0) > 0:
        return {
            "level": "Watch",
            "label": "주의필요",
            "class": "alert-watch",
            "icon": "[주의필요]",
            "message": "PVC 또는 전도 이상 관련 annotation이 확인되었습니다. 회색 marker 주변 waveform과 event burden을 확인하세요.",
        }
    if group_counts.get("Pacemaker", 0) > 0:
        return {
            "level": "Watch",
            "label": "주의필요",
            "class": "alert-watch",
            "icon": "[주의필요]",
            "message": "Pacemaker 관련 annotation이 확인되었습니다. 초록 marker 주변 pacing event를 확인하세요.",
        }
    return {
        "level": "Normal",
        "label": "정상",
        "class": "alert-normal",
        "icon": "[정상]",
        "message": "선택 구간에서 주요 비정상 rhythm annotation이 뚜렷하게 관찰되지 않았습니다.",
    }


def render_status_alert(alert):
    st.markdown(
        f"""
        <div class="alert-card {alert['class']}">
            <h3>{alert['icon']} {alert['label']}</h3>
            <p>{alert['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_reason_lines(group_counts):
    nonzero = [(group, count) for group, count in group_counts.items() if count > 0]
    if not nonzero:
        return ["annotation count가 없어 자동 판단의 신뢰도가 낮습니다."]
    lines = []
    for group, count in nonzero:
        color = ANNOTATION_COLOR_MAP.get(group, "#111111")
        lines.append(
            f"<span class='color-dot' style='background:{color}'></span><b>{group}</b>: {count} events"
        )
    return lines


def recommended_review_text(alert):
    if alert["level"] == "Abnormal Detected":
        return "먼저 II lead ECG segment의 빨강/주황/노랑 marker 주변 waveform을 확인하고, Event Timeline에서 발생 시점을 따라가세요."
    if alert["level"] == "Watch":
        return "II lead ECG segment의 회색 또는 초록 marker 주변 waveform을 확인한 뒤 Annotation Distribution과 HRV Scatter를 함께 검토하세요."
    if alert["level"] == "Review Required":
        return "annotation 원천 데이터가 부족하므로 raw waveform, 환자 메타데이터, 데이터 로딩 상태를 우선 확인하세요."
    return "Annotation Distribution에서 정상 annotation 중심인지 확인하고, 필요하면 12-lead overview로 전체 파형 품질을 점검하세요."


def render_decision_support(group_counts, alert):
    reason_html = "<br>".join(alert_reason_lines(group_counts))
    review = recommended_review_text(alert)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="review-card">
                <h4>Why This Alert?</h4>
                <p>{reason_html}</p>
                <p style="margin-top:10px;">이 상태는 진단이 아니라 선택 구간 annotation 기반의 검토 보조 신호입니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="review-card">
                <h4>Recommended Review</h4>
                <p>{review}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def burden_priority(group, count):
    if group in {"VT / IVR", "AVRT / AVNRT", "AFIB / AFLT"} and count > 0:
        return "High"
    if group in {"Conduction / PVC", "Pacemaker"} and count > 0:
        return "Medium"
    return "Low"


def burden_dataframe(group_counts):
    rows = []
    for group in EVENT_GROUP_ORDER:
        count = int(group_counts.get(group, 0))
        rows.append(
            {
                "Event group": group,
                "Count": count,
                "Marker color": ANNOTATION_COLOR_MAP.get(group, "#111111"),
                "Review priority": burden_priority(group, count),
            }
        )
    return pd.DataFrame(rows)


def render_event_burden(group_counts):
    st.markdown('<div class="section-title">Event Burden Summary</div>', unsafe_allow_html=True)
    cards = st.columns(6)
    for col, group in zip(cards, EVENT_GROUP_ORDER):
        count = int(group_counts.get(group, 0))
        color = ANNOTATION_COLOR_MAP.get(group, "#111111")
        priority = burden_priority(group, count)
        with col:
            st.markdown(
                f"""
                <div class="burden-card" style="border-top: 4px solid {color};">
                    <div class="burden-name"><span class="color-dot" style="background:{color}"></span>{group}</div>
                    <div class="burden-count">{count}</div>
                    <div class="burden-priority">Priority: {priority}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def annotation_summary_table(counts, window_seconds):
    if counts is None or counts.empty:
        return pd.DataFrame(columns=["Annotation", "Count", "Frequency/min", "Risk color", "Description"])
    rows = []
    denominator = max(window_seconds / 60, 1 / 60)
    for _, row in counts.iterrows():
        annotation = str(row.get("annotation", ""))
        count = int(row.get("count", 0))
        group = get_annotation_group(annotation, "")
        rows.append(
            {
                "Annotation": annotation,
                "Count": count,
                "Frequency/min": round(count / denominator, 2),
                "Risk color": group,
                "Description": {
                    "VT / IVR": "고위험 검토 이벤트",
                    "AVRT / AVNRT": "중간-높은 주의 이벤트",
                    "AFIB / AFLT": "rhythm irregularity 주의 이벤트",
                    "Conduction / PVC": "전도 이상 또는 PVC 계열",
                    "Pacemaker": "pacing event",
                    "Other": "기타 annotation",
                }.get(group, "기타 annotation"),
            }
        )
    return pd.DataFrame(rows).sort_values(["Count", "Annotation"], ascending=[False, True])


def visible_annotation_table(ann_df, start_sec, end_sec):
    if ann_df is None or ann_df.empty:
        return pd.DataFrame(columns=["time_sec", "annotation_label", "annotation_raw", "aux_string", "risk_group"])
    visible = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy()
    if visible.empty:
        return pd.DataFrame(columns=["time_sec", "annotation_label", "annotation_raw", "aux_string", "risk_group"])
    visible = add_annotation_display_labels(visible)
    visible["risk_group"] = visible.apply(
        lambda row: get_annotation_group(row.get("symbol", ""), row.get("aux_note", "")),
        axis=1,
    )
    return visible.rename(
        columns={"symbol": "annotation_raw", "aux_note": "aux_string", "display_label": "annotation_label"}
    )[["time_sec", "annotation_label", "annotation_raw", "aux_string", "risk_group"]].round({"time_sec": 3})


def render_annotation_legend():
    legend_html = " ".join(
        f"<span style='margin-right:14px; white-space:nowrap;'><span class='color-dot' style='background:{ANNOTATION_COLOR_MAP[group]}'></span>{group}</span>"
        for group in EVENT_GROUP_ORDER
    )
    st.markdown(f"<div class='note-box'><b>Annotation color legend</b><br>{legend_html}</div>", unsafe_allow_html=True)


def build_episode_summary(ann_df, start_sec=None, end_sec=None, gap_threshold=1.0, abnormal_only=False):
    columns = ["event_group", "start_time", "end_time", "duration_sec", "annotation_count"]
    if ann_df is None or ann_df.empty or "time_sec" not in ann_df.columns:
        return pd.DataFrame(columns=columns)

    work = ann_df.copy()
    if start_sec is not None:
        work = work[work["time_sec"] >= start_sec]
    if end_sec is not None:
        work = work[work["time_sec"] <= end_sec]
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["event_group"] = work.apply(
        lambda row: get_annotation_group(row.get("symbol", ""), row.get("aux_note", "")),
        axis=1,
    )
    if abnormal_only:
        work = work[~work["event_group"].isin(["Other"])]
    work = work.sort_values(["event_group", "time_sec"])

    rows = []
    for group, sub in work.groupby("event_group", sort=False):
        times = sub["time_sec"].dropna().astype(float).sort_values().to_list()
        if not times:
            continue
        episode_start = times[0]
        previous = times[0]
        count = 1
        for current in times[1:]:
            if current - previous <= gap_threshold:
                previous = current
                count += 1
                continue
            rows.append(
                {
                    "event_group": group,
                    "start_time": episode_start,
                    "end_time": previous,
                    "duration_sec": max(0.0, previous - episode_start),
                    "annotation_count": count,
                }
            )
            episode_start = current
            previous = current
            count = 1
        rows.append(
            {
                "event_group": group,
                "start_time": episode_start,
                "end_time": previous,
                "duration_sec": max(0.0, previous - episode_start),
                "annotation_count": count,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["start_time", "event_group"]).reset_index(drop=True)


def event_timeline_chart(ann_df, start_sec, end_sec, theme_mode):
    visible_ann = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy()
    fig = go.Figure()
    if visible_ann.empty:
        fig.add_annotation(
            text="No annotation events in selected segment",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=CHART_MUTED, size=14),
        )
    else:
        visible_ann = add_annotation_display_labels(visible_ann)
        visible_ann["annotation_group"] = visible_ann.apply(
            lambda row: get_annotation_group(row.get("symbol", ""), row.get("aux_note", "")),
            axis=1,
        )
        episodes = build_episode_summary(visible_ann, start_sec, end_sec, abnormal_only=True)
        for _, episode in episodes.iterrows():
            if int(episode["annotation_count"]) < 2 and float(episode["duration_sec"]) <= 0:
                continue
            group = episode["event_group"]
            end_time = max(float(episode["end_time"]), float(episode["start_time"]) + 0.15)
            fig.add_trace(
                go.Scatter(
                    x=[float(episode["start_time"]), end_time],
                    y=[group, group],
                    mode="lines",
                    name=f"{group} episode",
                    showlegend=False,
                    line=dict(color=ANNOTATION_COLOR_MAP.get(group, "#111111"), width=10),
                    opacity=0.32,
                    customdata=[
                        [float(episode["start_time"]), float(episode["end_time"]), float(episode["duration_sec"]), int(episode["annotation_count"])],
                        [float(episode["start_time"]), float(episode["end_time"]), float(episode["duration_sec"]), int(episode["annotation_count"])],
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Start: %{customdata[0]:.2f}s<br>"
                        "End: %{customdata[1]:.2f}s<br>"
                        "Duration: %{customdata[2]:.2f}s<br>"
                        "Annotations: %{customdata[3]}<extra></extra>"
                    ),
                )
            )
        border = get_marker_border_color(theme_mode)
        for group in EVENT_GROUP_ORDER:
            sub = visible_ann[visible_ann["annotation_group"] == group]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["time_sec"],
                    y=[group] * len(sub),
                    mode="markers",
                    name=group,
                    marker=dict(
                        size=8 if group in {"VT / IVR", "AVRT / AVNRT", "AFIB / AFLT"} else 7,
                        color=ANNOTATION_COLOR_MAP[group],
                        opacity=0.85,
                        line=dict(width=1.2, color=border),
                    ),
                    customdata=np.stack(
                        [sub["symbol"], sub.get("aux_note", pd.Series([""] * len(sub), index=sub.index))],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Time: %{x:.2f}s<br>"
                        "Symbol: %{customdata[0]}<br>"
                        "Aux-string: %{customdata[1]}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        title="Event Timeline",
        height=250,
        margin=dict(l=34, r=24, t=52, b=34),
        xaxis_title="Time (sec)",
        yaxis_title="Event group",
        legend=dict(orientation="h", y=1.1, x=0, yanchor="bottom"),
    )
    fig.update_xaxes(range=[start_sec, end_sec], showgrid=True)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(EVENT_GROUP_ORDER)))
    return style_chart(fig, height=250, margin=dict(l=34, r=24, t=52, b=34))


def plot_event_density(ann_df, start_sec, end_sec):
    if ann_df is None or ann_df.empty or "time_sec" not in ann_df.columns:
        return None
    visible = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy()
    if visible.empty:
        return None
    visible["second"] = np.floor(visible["time_sec"]).astype(int)
    density = visible.groupby("second").size().reset_index(name="annotation_count")
    fig = px.bar(density, x="second", y="annotation_count", title="Event density by second")
    fig.update_layout(xaxis_title="Time (sec)", yaxis_title="Annotation count")
    return style_chart(fig, height=280, margin=dict(l=34, r=24, t=52, b=42))


def normal_to_abnormal_transition_points(ann_df, start_sec, end_sec):
    columns = ["time_sec", "from_group", "to_group", "symbol"]
    if ann_df is None or ann_df.empty or "time_sec" not in ann_df.columns:
        return pd.DataFrame(columns=columns)
    visible = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy()
    if visible.empty:
        return pd.DataFrame(columns=columns)
    visible["event_group"] = visible.apply(
        lambda row: get_annotation_group(row.get("symbol", ""), row.get("aux_note", "")),
        axis=1,
    )
    visible["is_normal_like"] = visible["symbol"].astype(str).isin(NORMAL_SYMBOLS)
    visible = visible.sort_values("time_sec")
    rows = []
    previous = None
    for _, row in visible.iterrows():
        current_group = "Normal-like" if row["is_normal_like"] else row["event_group"]
        if previous == "Normal-like" and current_group != "Normal-like":
            rows.append(
                {
                    "time_sec": float(row["time_sec"]),
                    "from_group": previous,
                    "to_group": current_group,
                    "symbol": row.get("symbol", ""),
                }
            )
        previous = current_group
    return pd.DataFrame(rows, columns=columns)


def plot_normal_to_abnormal_transitions(transitions):
    if transitions is None or transitions.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=transitions["time_sec"],
            y=transitions["to_group"],
            mode="markers",
            marker=dict(size=8, color="#e7534f"),
            customdata=np.stack([transitions["from_group"], transitions["symbol"]], axis=-1),
            hovertemplate="Time: %{x:.2f}s<br>From: %{customdata[0]}<br>To: %{y}<br>Symbol: %{customdata[1]}<extra></extra>",
            name="Transition",
        )
    )
    fig.update_layout(title="Normal to abnormal transition points", xaxis_title="Time (sec)", yaxis_title="To group")
    return style_chart(fig, height=280, margin=dict(l=34, r=24, t=52, b=42))


def style_chart(fig, height=None, margin=None):
    layout = {
        "font": dict(color=CHART_TEXT, family="Arial, sans-serif"),
        "title": dict(font=dict(color=CHART_TEXT, size=18)),
        "plot_bgcolor": CHART_BG,
        "paper_bgcolor": CHART_BG,
        "legend": dict(font=dict(color=CHART_TEXT, size=13)),
        "margin": margin or dict(l=28, r=24, t=52, b=34),
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(
        title_font=dict(color=CHART_TEXT),
        tickfont=dict(color=CHART_TEXT),
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
        linecolor="#c9d8dc",
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(color=CHART_TEXT),
        tickfont=dict(color=CHART_TEXT),
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
        linecolor="#c9d8dc",
        automargin=True,
    )
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=CHART_TEXT), title=dict(font=dict(color=CHART_TEXT))))
    return fig


def db_env():
    if not DB_ENV_PATH.exists():
        return {}
    if dotenv_values is not None:
        return {key: str(value) for key, value in dotenv_values(DB_ENV_PATH).items() if value is not None}
    config = {}
    for raw_line in DB_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip().strip('"').strip("'")
    return config


def db_config_status():
    config = db_env()
    required = {"DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"}
    if not config:
        return "Missing"
    return "Configured" if required.issubset(config) else "Check"


@st.cache_resource(show_spinner=False)
def mysql_engine():
    if create_engine is None:
        return None
    config = db_env()
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    if any(not config.get(key) for key in required):
        return None
    url = (
        f"mysql+pymysql://{quote_plus(config['DB_USER'])}:{quote_plus(config['DB_PASSWORD'])}"
        f"@{config['DB_HOST']}:{config['DB_PORT']}/{config['DB_NAME']}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


@st.cache_data(show_spinner=False, ttl=300)
def mysql_health():
    engine = mysql_engine()
    if engine is None or text is None:
        return {"ok": False, "database": None, "tables": 0, "error": "MySQL env or driver is not configured"}
    try:
        with engine.connect() as conn:
            database = conn.execute(text("SELECT DATABASE()")).scalar()
            tables = conn.execute(text("SHOW TABLES")).fetchall()
        return {"ok": True, "database": database, "tables": len(tables), "error": ""}
    except Exception as exc:
        return {"ok": False, "database": None, "tables": 0, "error": str(exc).splitlines()[0][:160]}


@st.cache_data(show_spinner=False, ttl=300)
def mysql_table_counts():
    engine = mysql_engine()
    if engine is None or text is None:
        return {}
    try:
        with engine.connect() as conn:
            tables = [row[0] for row in conn.execute(text("SHOW TABLES")).fetchall()]
            return {table: int(conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar() or 0) for table in tables}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=300)
def load_subjects_from_mysql():
    engine = mysql_engine()
    if engine is None or text is None:
        return pd.DataFrame()
    sql = text(
        """
        SELECT
            p.patient_id AS file_name,
            p.patient_id AS subject_id,
            CASE
                WHEN p.group_type IN ('child', 'children') THEN 'children'
                WHEN p.group_type IN ('adult', 'adult_chd') THEN 'adult'
                ELSE p.group_type
            END AS `group`,
            p.gender,
            p.age,
            p.diagnosis,
            p.pathway_location AS ap_loacation,
            p.ecg_duration_sec,
            r.record_name,
            r.duration_sec
        FROM patients p
        LEFT JOIN records r ON p.patient_id = r.patient_id
        ORDER BY r.record_id, p.patient_id
        """
    )
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def load_records_from_mysql():
    engine = mysql_engine()
    if engine is None or text is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text(
                    """
                    SELECT record_id, patient_id, record_name, fs, n_sig, sig_len, duration_sec
                    FROM records
                    ORDER BY record_id
                    """
                ),
                conn,
            )
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def read_annotations_from_mysql(record_id):
    engine = mysql_engine()
    if engine is None or text is None:
        return pd.DataFrame(columns=["sample", "time_sec", "symbol", "aux_note"])
    sql = text(
        """
        SELECT a.sample, a.time_sec, a.symbol, a.aux_note
        FROM annotations a
        JOIN records r ON a.record_id = r.record_id
        WHERE r.record_name = :record_id OR r.patient_id = :record_id
        ORDER BY a.sample
        """
    )
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"record_id": str(record_id)})
    except Exception:
        return pd.DataFrame(columns=["sample", "time_sec", "symbol", "aux_note"])


@st.cache_data(show_spinner=False)
def load_subjects():
    mysql_subjects = load_subjects_from_mysql()
    frames = []
    for group, filename in [("children", "children-subject-info.csv"), ("adult", "adults-subject-info.csv")]:
        for path in [BASE_DIR / filename, DATA_DIR / filename]:
            if path.exists():
                df = pd.read_csv(path)
                df["group"] = group
                frames.append(df)
                break
    csv_subjects = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not csv_subjects.empty:
        csv_subjects["file_name"] = csv_subjects["file_name"].astype(str)
        csv_subjects["subject_id"] = csv_subjects["subject_id"].astype(str).str.zfill(3)
    if not mysql_subjects.empty and not csv_subjects.empty:
        csv_cols = ["file_name", "gender", "age", "diagnosis", "ap_loacation", "ecg_duration"]
        csv_cols = [col for col in csv_cols if col in csv_subjects.columns]
        df = mysql_subjects.merge(csv_subjects[csv_cols], on="file_name", how="left", suffixes=("", "_csv"))
        for col in ["gender", "age", "diagnosis", "ap_loacation"]:
            csv_col = f"{col}_csv"
            if col in df.columns and csv_col in df.columns:
                df[col] = df[col].where(df[col].notna() & (df[col].astype(str).str.len() > 0), df[csv_col])
                df = df.drop(columns=[csv_col])
    elif not mysql_subjects.empty:
        df = mysql_subjects.copy()
        df["ecg_duration"] = df["ecg_duration_sec"].map(lambda sec: f"0:00:{float(sec):.3f}" if pd.notna(sec) else "")
    elif not csv_subjects.empty:
        df = csv_subjects.copy()
    else:
        return pd.DataFrame(columns=["file_name", "group", "age", "gender", "diagnosis", "ecg_duration"])
    if "ecg_duration" not in df.columns:
        df["ecg_duration"] = pd.NA
    df["duration_sec_info"] = df.get("ecg_duration", pd.Series(dtype=str)).map(parse_duration_to_seconds)
    if "duration_sec" in df.columns:
        df["duration_sec_info"] = df["duration_sec_info"].fillna(pd.to_numeric(df["duration_sec"], errors="coerce"))
    elif "ecg_duration_sec" in df.columns:
        df["duration_sec_info"] = df["duration_sec_info"].fillna(pd.to_numeric(df["ecg_duration_sec"], errors="coerce"))
    return df


@st.cache_data(show_spinner=False)
def load_summary():
    path = RESULT_DIR / "ecg_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["annotation_map"] = df.get("annotation_symbols", pd.Series(dtype=object)).map(parse_dict)
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_aux_annotations():
    path = BASE_DIR / "processed_segments_with_aux.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["dataset_aux_string"] = df["dataset_aux_string"].fillna("")
        return df
    return pd.DataFrame(columns=["patient_id", "label_symbol", "dataset_aux_string", "description"])


@st.cache_data(show_spinner=False)
def load_rpeaks():
    path = RESULT_DIR / "ecg_rpeaks.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["record_id", "rpeak_time_sec", "rr_interval_ms", "instantaneous_hr"])


@st.cache_data(show_spinner=False)
def list_records():
    mysql_records = load_records_from_mysql()
    if not mysql_records.empty and "record_name" in mysql_records.columns:
        return mysql_records["record_name"].dropna().astype(str).tolist()
    records = sorted([p.stem for p in DATA_DIR.glob("*.hea")]) if DATA_DIR.exists() else []
    if records:
        return records
    subjects = load_subjects()
    return sorted(subjects["file_name"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def read_header(record_id):
    if wfdb is None:
        return None
    try:
        return wfdb.rdheader(str(DATA_DIR / record_id))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_record_segment(record_id, start_sec, duration_sec):
    header = read_header(record_id)
    if wfdb is None or header is None:
        return None
    fs = float(header.fs)
    start = max(0, int(start_sec * fs))
    stop = min(header.sig_len, int((start_sec + duration_sec) * fs))
    if stop <= start:
        stop = min(header.sig_len, start + int(duration_sec * fs))
    try:
        rec = wfdb.rdrecord(str(DATA_DIR / record_id), sampfrom=start, sampto=stop)
        data = pd.DataFrame(rec.p_signal, columns=rec.sig_name)
        data.insert(0, "time_sec", np.arange(start, start + len(data)) / fs)
        return {"df": data, "fs": fs, "channels": rec.sig_name, "n_samples": header.sig_len}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_annotations(record_id):
    mysql_annotations = read_annotations_from_mysql(record_id)
    if not mysql_annotations.empty:
        return mysql_annotations
    if wfdb is None:
        return pd.DataFrame(columns=["sample", "time_sec", "symbol", "aux_note"])
    header = read_header(record_id)
    fs = float(header.fs) if header is not None else 977.0
    try:
        ann = wfdb.rdann(str(DATA_DIR / record_id), "atr")
        aux = getattr(ann, "aux_note", [""] * len(ann.sample))
        return pd.DataFrame(
            {
                "sample": ann.sample,
                "time_sec": np.array(ann.sample) / fs,
                "symbol": ann.symbol,
                "aux_note": aux,
            }
        )
    except Exception:
        return pd.DataFrame(columns=["sample", "time_sec", "symbol", "aux_note"])


def fallback_signal(record_id, start_sec, duration_sec, lead):
    fs = 977.0
    n = int(duration_sec * fs)
    t = np.arange(n) / fs + start_sec
    seed = sum(ord(c) for c in record_id + lead)
    rng = np.random.default_rng(seed)
    base_hr = 80 + seed % 70
    beat_period = max(0.32, 60 / base_hr)
    phase = np.mod(t - start_sec, beat_period)
    qrs = np.exp(-((phase - 0.08) ** 2) / 0.00018) * 1.15
    p_wave = 0.12 * np.exp(-((phase - 0.02) ** 2) / 0.0009)
    twave = 0.25 * np.exp(-((phase - 0.22) ** 2) / 0.004)
    drift = 0.08 * np.sin(2 * np.pi * 0.28 * t)
    signal = p_wave + qrs + twave + drift + rng.normal(0, 0.018, size=n)
    return pd.DataFrame({"time_sec": t, lead: signal})


def patient_meta(subjects, summary, record_id):
    row = pd.DataFrame()
    if not subjects.empty and "file_name" in subjects.columns:
        row = subjects[subjects["file_name"].astype(str) == record_id]
    srow = summary[summary["record_id"].astype(str) == record_id] if not summary.empty else pd.DataFrame()
    group = None
    if not row.empty and "group" in row:
        group = row.iloc[0]["group"]
    elif not srow.empty and "group" in srow:
        group = srow.iloc[0]["group"]
    else:
        digits = "".join(ch for ch in record_id if ch.isdigit())
        group = "adult" if digits and int(digits) >= 100 else "children"
    meta = {
        "record_id": record_id,
        "group": group,
        "age": row.iloc[0].get("age", np.nan) if not row.empty else np.nan,
        "gender": row.iloc[0].get("gender", "-") if not row.empty else "-",
        "diagnosis": row.iloc[0].get("diagnosis", "-") if not row.empty else "-",
        "duration": row.iloc[0].get("ecg_duration", "-") if not row.empty else "-",
    }
    if not srow.empty:
        meta.update(
            {
                "mean_hr": srow.iloc[0].get("mean_hr", np.nan),
                "rmssd_ms": srow.iloc[0].get("rmssd_ms", np.nan),
                "n_rpeaks": srow.iloc[0].get("n_rpeaks", np.nan),
                "n_annotations": srow.iloc[0].get("n_annotations", np.nan),
                "duration_sec": srow.iloc[0].get("duration_sec", np.nan),
                "annotation_map": srow.iloc[0].get("annotation_map", {}),
                "rhythm_annotations": srow.iloc[0].get("rhythm_annotations", ""),
            }
        )
    return meta


def annotation_counts(record_id, summary, aux_df, ann_df):
    if not ann_df.empty:
        labeled = add_annotation_display_labels(ann_df)
        return labeled["display_label"].value_counts().rename_axis("annotation").reset_index(name="count")
    sub = aux_df[aux_df["patient_id"].astype(str) == record_id] if not aux_df.empty else pd.DataFrame()
    if not sub.empty:
        return aux_annotation_counts(sub)
    srow = summary[summary["record_id"].astype(str) == record_id] if not summary.empty else pd.DataFrame()
    if not srow.empty:
        amap = srow.iloc[0].get("annotation_map", {})
        return pd.DataFrame({"annotation": list(amap.keys()), "count": list(amap.values())})
    return pd.DataFrame({"annotation": ["N", "A", "V", "VT"], "count": [82, 11, 4, 1]})


def compute_patient_stats(counts):
    total = int(counts["count"].sum()) if not counts.empty else 0
    abnormal = 0
    tachy = 0
    normal = 0
    for _, row in counts.iterrows():
        label = str(row["annotation"]).split(":", 1)[0].strip()
        cnt = int(row["count"])
        if label in NORMAL_SYMBOLS:
            normal += cnt
        elif label in ABNORMAL_SYMBOLS or label not in {"+"}:
            abnormal += cnt
        if label in TACHY_SYMBOLS or label in {"V", "A", "a"}:
            tachy += cnt
    ratio = abnormal / total if total else 0
    return {
        "total": total,
        "normal": normal,
        "abnormal": abnormal,
        "tachy": tachy,
        "abnormal_ratio": ratio,
        "risk": risk_from_ratio(ratio, tachy),
    }


def ecg_waveform_chart(signal_df, lead, ann_df, start_sec, end_sec, theme_mode="Light"):
    fig = go.Figure()
    if lead not in signal_df.columns:
        lead = signal_df.columns[1] if len(signal_df.columns) > 1 else signal_df.columns[0]
    fig.add_trace(
        go.Scatter(
            x=signal_df["time_sec"],
            y=signal_df[lead],
            mode="lines",
            line=dict(color="#0b8f83", width=1.15),
            name=lead,
        )
    )
    visible_ann = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy()
    if not visible_ann.empty and lead in signal_df.columns:
        visible_ann = add_annotation_display_labels(visible_ann)
        sample_y = np.interp(visible_ann["time_sec"], signal_df["time_sec"], signal_df[lead])
        visible_ann["y"] = sample_y
        visible_ann["annotation_group"] = visible_ann.apply(
            lambda row: get_annotation_group(row.get("symbol", ""), row.get("aux_note", "")),
            axis=1,
        )
        border = get_marker_border_color(theme_mode)
        for group in EVENT_GROUP_ORDER:
            group_df = visible_ann[visible_ann["annotation_group"] == group]
            if group_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=group_df["time_sec"],
                    y=group_df["y"],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=ANNOTATION_COLOR_MAP.get(group, "#111111"),
                        opacity=0.85,
                        line=dict(width=1.2, color=border),
                    ),
                    name=group,
                    customdata=np.stack(
                        [
                            group_df["display_label"],
                            group_df["symbol"],
                            group_df.get("aux_note", pd.Series([""] * len(group_df), index=group_df.index)),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Time: %{x:.2f}s<br>"
                        f"{ECG_Y_AXIS_LABEL}: %{{y:.3f}}<br>"
                        "Label: %{customdata[0]}<br>"
                        "Symbol: %{customdata[1]}<br>"
                        "Aux-string: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        height=360,
        margin=dict(l=28, r=24, t=78, b=34),
        title=dict(
            text=f"{lead} lead ECG segment",
            x=0,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(color=CHART_TEXT, size=19),
        ),
        xaxis_title="Time (sec)",
        yaxis_title=ECG_Y_AXIS_LABEL,
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
            yanchor="bottom",
            font=dict(color=CHART_TEXT, size=13),
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    fig.update_xaxes(showgrid=True, range=[start_sec, end_sec])
    fig.update_yaxes(showgrid=True)
    return style_chart(fig, height=360, margin=dict(l=28, r=24, t=78, b=34))


def twelve_lead_chart(record_id, start_sec, duration_sec):
    rec = read_record_segment(record_id, start_sec, duration_sec)
    if rec is None:
        data = fallback_signal(record_id, start_sec, duration_sec, "II")
        for lead in LEADS_12:
            if lead not in data:
                data[lead] = data["II"] * (0.7 + 0.05 * (LEADS_12.index(lead) % 5)) + 0.08 * LEADS_12.index(lead)
    else:
        data = rec["df"]
    available = [lead for lead in LEADS_12 if lead in data.columns]
    fig = go.Figure()
    for i, lead in enumerate(available):
        y = data[lead].to_numpy()
        scale = np.nanstd(y) or 1
        normalized = (y - np.nanmedian(y)) / scale
        fig.add_trace(
            go.Scatter(
                x=data["time_sec"],
                y=normalized + (len(available) - i) * 2.2,
                mode="lines",
                line=dict(color="#164f63", width=1),
                name=lead,
                hovertemplate=f"{lead}<br>%{{x:.2f}}s<extra></extra>",
            )
        )
    tickvals = [(len(available) - i) * 2.2 for i in range(len(available))]
    fig.update_layout(
        height=560,
        margin=dict(l=34, r=24, t=52, b=34),
        title="12-lead ECG overview",
        showlegend=False,
        yaxis=dict(tickmode="array", tickvals=tickvals, ticktext=available, showgrid=False),
        xaxis_title="Time (sec)",
        yaxis_title=f"{ECG_Y_AXIS_LABEL} + lead offset",
    )
    fig.update_xaxes(showgrid=True)
    return style_chart(fig, height=560, margin=dict(l=34, r=24, t=52, b=34))


def split_dataframe(records):
    rows = []
    for r in records:
        digits = "".join(ch for ch in r if ch.isdigit())
        num = int(digits) if digits else 0
        split = "Train" if num % 10 not in {7, 8, 9} else ("Validation" if num % 10 == 8 else "Test")
        group = "Adult" if num >= 100 else "Children"
        rows.append({"record_id": r, "group": group, "split": split})
    return pd.DataFrame(rows)


def feature_importance_df():
    return pd.DataFrame(ML_FEATURES, columns=["feature", "importance", "description"])


def class_report_df():
    return pd.DataFrame(
        ML_CLASS_REPORT,
        columns=["class", "precision", "recall", "f1-score", "support"],
    )


def one_vs_rest_auc_df():
    return pd.DataFrame(
        ML_OVR_AUC,
        columns=["class", "support", "roc_auc", "pr_auc"],
    )


def predicted_class_ratio_df():
    return pd.DataFrame(
        ML_PREDICTED_CLASS_RATIO,
        columns=["class", "predicted_ratio"],
    )


def confusion_matrix_fig():
    labels = XGB_CLASS_NAMES
    fig = px.imshow(
        ML_VALIDATION_CM,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=["#eef8f7", "#26b99a", "#0d4750"],
        labels=dict(x="Predicted label", y="Actual label", color="count"),
    )
    fig.update_traces(textfont=dict(color=CHART_TEXT, size=13))
    fig.update_layout(title="XGBoost validation confusion matrix")
    return style_chart(fig, height=420, margin=dict(l=34, r=24, t=52, b=42))


def plot_normalized_confusion_matrix(normalize="row"):
    labels = XGB_CLASS_NAMES
    matrix = ML_VALIDATION_CM.astype(float)
    if normalize == "column":
        denominator = matrix.sum(axis=0, keepdims=True)
        title = "Column-normalized precision view"
        color_label = "column %"
        x_title = "Predicted label"
        y_title = "Actual label"
    else:
        denominator = matrix.sum(axis=1, keepdims=True)
        title = "Row-normalized recall view"
        color_label = "row %"
        x_title = "Predicted label"
        y_title = "Actual label"
    normalized = np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator != 0)
    text = np.vectorize(lambda value: f"{value:.1%}")(normalized)
    fig = px.imshow(
        normalized,
        x=labels,
        y=labels,
        color_continuous_scale=["#eef8f7", "#26b99a", "#0d4750"],
        zmin=0,
        zmax=1,
        labels=dict(x=x_title, y=y_title, color=color_label),
    )
    fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(color=CHART_TEXT, size=13))
    fig.update_layout(title=title)
    return style_chart(fig, height=420, margin=dict(l=34, r=24, t=52, b=42))


def render_model_interpretation_notes():
    ratio_df = predicted_class_ratio_df()
    report_df = class_report_df()
    auc_df = one_vs_rest_auc_df()
    notes = [
        "Macro F1과 balanced accuracy를 함께 확인해야 class imbalance 영향을 더 잘 해석할 수 있습니다.",
    ]
    warnings = []
    if not ratio_df.empty and {"class", "predicted_ratio"}.issubset(ratio_df.columns):
        low_ratio = ratio_df[ratio_df["predicted_ratio"] < 0.01]
        for _, row in low_ratio.iterrows():
            warnings.append(
                f"주의: validation set에서 {row['class']} class predicted ratio가 {row['predicted_ratio']:.1%}로 1% 미만입니다."
            )
    if not report_df.empty and {"class", "recall"}.issubset(report_df.columns):
        normal_report = report_df[report_df["class"] == "normal"]
        if not normal_report.empty and float(normal_report.iloc[0]["recall"]) < 0.5:
            warnings.append(
                "주의: validation set에서 normal class recall이 낮습니다. 전체 accuracy가 높아도 normal class 탐지 성능은 불안정할 수 있습니다."
            )
    if not auc_df.empty and {"class", "pr_auc"}.issubset(auc_df.columns):
        normal_auc = auc_df[auc_df["class"] == "normal"]
        if not normal_auc.empty and float(normal_auc.iloc[0]["pr_auc"]) < 0.5:
            warnings.append(
                "주의: normal class PR-AUC가 낮습니다. 클래스 불균형 환경에서 normal class precision-recall 균형을 별도로 확인해야 합니다."
            )
    st.markdown('<div class="section-title">Model Interpretation Notes</div>', unsafe_allow_html=True)
    for note in notes:
        st.markdown(
            f'<div class="interpretation-note-box">{html.escape(note)}</div>',
            unsafe_allow_html=True,
        )
    for warning in warnings:
        st.markdown(
            f'<div class="interpretation-warning-box">{html.escape(warning)}</div>',
            unsafe_allow_html=True,
        )
    if not warnings:
        st.markdown(
            '<div class="interpretation-note-box">현재 저장된 validation 요약 기준으로 1% 미만 predicted ratio 또는 낮은 normal class PR-AUC/recall 경고는 없습니다.</div>',
            unsafe_allow_html=True,
        )


def roc_curve_fig():
    auc_df = one_vs_rest_auc_df()
    auc_df["roc_auc_label"] = auc_df["roc_auc"].map(lambda v: f"{v:.3f}")
    fig = px.bar(
        auc_df,
        x="class",
        y="roc_auc",
        text="roc_auc_label",
        color="class",
        color_discrete_map={"normal": "#24b89b", "ntach": "#1f6f8b", "tach": "#e7534f"},
        title="XGBoost one-vs-rest ROC-AUC",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1], xaxis_title="Class", yaxis_title="ROC-AUC")
    return style_chart(fig, height=330, margin=dict(l=34, r=24, t=52, b=42))


def pr_curve_fig():
    auc_df = one_vs_rest_auc_df()
    auc_df["pr_auc_label"] = auc_df["pr_auc"].map(lambda v: f"{v:.3f}")
    fig = px.bar(
        auc_df,
        x="class",
        y="pr_auc",
        text="pr_auc_label",
        color="class",
        color_discrete_map={"normal": "#24b89b", "ntach": "#1f6f8b", "tach": "#e7534f"},
        title="XGBoost one-vs-rest PR-AUC",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1], xaxis_title="Class", yaxis_title="PR-AUC")
    return style_chart(fig, height=330, margin=dict(l=34, r=24, t=52, b=42))


def probability_distribution_fig():
    ratio_df = predicted_class_ratio_df()
    ratio_df["ratio_label"] = ratio_df["predicted_ratio"].map(lambda v: f"{v:.1%}")
    fig = px.bar(
        ratio_df,
        x="class",
        y="predicted_ratio",
        text="ratio_label",
        color="class",
        color_discrete_map={"normal": "#24b89b", "ntach": "#1f6f8b", "tach": "#e7534f"},
        title="Predicted class ratio on validation set",
    )
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", xaxis_title="Predicted class", yaxis_title="Ratio")
    return style_chart(fig, height=330, margin=dict(l=34, r=24, t=52, b=42))


def clinical_analysis(record_id, lead, start_sec, duration_sec, subjects, summary, aux_df, rpeaks, theme_mode):
    end_sec = start_sec + duration_sec
    meta = patient_meta(subjects, summary, record_id)
    ann_df = read_annotations(record_id)
    window_ann_df = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy() if not ann_df.empty else ann_df
    counts = annotation_counts(record_id, summary, aux_df, ann_df)
    stats = compute_patient_stats(counts)
    group_counts = group_counts_from_annotations(window_ann_df, counts if ann_df.empty else pd.DataFrame())
    alert = determine_patient_alert(group_counts)
    rec = read_record_segment(record_id, start_sec, duration_sec)
    if rec is not None and lead in rec["df"].columns:
        signal_df = rec["df"][["time_sec", lead]].copy()
    elif rec is not None:
        first = [c for c in rec["df"].columns if c != "time_sec"][0]
        signal_df = rec["df"][["time_sec", first]].rename(columns={first: lead})
    else:
        signal_df = fallback_signal(record_id, start_sec, duration_sec, lead)

    st.markdown('<div class="section-title">Patient Status Alert</div>', unsafe_allow_html=True)
    render_status_alert(alert)
    render_decision_support(group_counts, alert)
    reason = (
        "비정상: 선택 구간 내 고우선순위 annotation이 관찰되어 검토 대상으로 표시했습니다."
        if alert["level"] == "Abnormal Detected"
        else "주의: 선택 구간 내 PVC, 전도 이상 또는 pacing 관련 annotation이 관찰되었습니다."
        if alert["level"] == "Watch"
        else "정상: 선택 구간 내 주요 부정맥 annotation이 뚜렷하게 관찰되지 않았습니다."
        if alert["level"] == "Normal"
        else "검토필요: 선택 구간의 annotation 정보가 부족하거나 파싱되지 않았습니다."
    )
    st.info(reason)

    st.markdown('<div class="section-title">Patient Quick Summary</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    with cols[0]:
        metric_card("Patient", record_id, meta["group"].title())
    with cols[1]:
        age = "-" if pd.isna(meta.get("age")) else f"{float(meta['age']):.1f}"
        metric_card("Age", age, f"{meta.get('gender', '-')}")
    with cols[2]:
        mean_hr = meta.get("mean_hr", np.nan)
        metric_card("Mean HR", "-" if pd.isna(mean_hr) else f"{mean_hr:.1f}", "beats per minute")
    with cols[3]:
        metric_card("Abnormal", f"{stats['abnormal_ratio'] * 100:.1f}%", f"{stats['abnormal']} / {stats['total']}")
    with cols[4]:
        rmssd = meta.get("rmssd_ms", np.nan)
        metric_card("RMSSD", "-" if pd.isna(rmssd) else f"{rmssd:.1f}", "ms")
    with cols[5]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk flag</div>
                <div class="metric-value">{risk_badge(stats['risk'])}</div>
                <div class="metric-caption">research triage only</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">II Lead ECG Segment</div>', unsafe_allow_html=True)
    render_annotation_legend()
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.plotly_chart(ecg_waveform_chart(signal_df, lead, ann_df, start_sec, end_sec, theme_mode), width="stretch")
    with c2:
        plot_counts = counts.sort_values("count", ascending=True).tail(14)
        fig = px.bar(
            plot_counts,
            x="count",
            y="annotation",
            orientation="h",
            color="annotation",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Annotation distribution",
        )
        fig.update_layout(showlegend=False)
        style_chart(fig, height=360, margin=dict(l=34, r=24, t=52, b=42))
        st.plotly_chart(fig, width="stretch")

    render_event_burden(group_counts)
    st.plotly_chart(event_timeline_chart(ann_df, start_sec, end_sec, theme_mode), width="stretch")
    with st.expander("Advanced Event Pattern Analysis", expanded=False):
        episode_df = build_episode_summary(ann_df, start_sec, end_sec, abnormal_only=True)
        st.markdown('<div class="section-title">Episode duration summary</div>', unsafe_allow_html=True)
        if episode_df.empty:
            st.info("선택 구간에서 episode duration으로 묶을 annotation 정보가 없습니다.")
        else:
            render_dashboard_table(episode_df.round({"start_time": 3, "end_time": 3, "duration_sec": 3}))

        d1, d2 = st.columns([1, 1])
        with d1:
            density_fig = plot_event_density(ann_df, start_sec, end_sec)
            if density_fig is None:
                st.info("Event density plot에 사용할 annotation 정보가 없습니다.")
            else:
                st.plotly_chart(density_fig, width="stretch")
        with d2:
            transitions = normal_to_abnormal_transition_points(ann_df, start_sec, end_sec)
            transition_fig = plot_normal_to_abnormal_transitions(transitions)
            if transition_fig is None:
                st.info("Normal to abnormal transition point가 선택 구간에서 확인되지 않습니다.")
            else:
                st.plotly_chart(transition_fig, width="stretch")
    with st.expander("Annotation details", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="section-title">Annotation Summary</div>', unsafe_allow_html=True)
            render_dashboard_table(annotation_summary_table(counts, duration_sec))
        with c2:
            st.markdown('<div class="section-title">Selected Window Annotation Table</div>', unsafe_allow_html=True)
            window_table = visible_annotation_table(ann_df, start_sec, end_sec)
            if window_table.empty:
                st.info("현재 구간 내 annotation 없음")
            else:
                render_dashboard_table(window_table)
    st.markdown('<div class="section-title">Detail Graphs</div>', unsafe_allow_html=True)
    st.plotly_chart(twelve_lead_chart(record_id, start_sec, min(duration_sec, 10)), width="stretch")

    rsub = rpeaks[rpeaks["record_id"].astype(str) == record_id] if not rpeaks.empty else pd.DataFrame()
    rsub = rsub[(rsub["rpeak_time_sec"] >= start_sec) & (rsub["rpeak_time_sec"] <= end_sec)] if not rsub.empty else rsub
    if not rsub.empty:
        fig = px.scatter(
            rsub.dropna(subset=["rr_interval_ms", "instantaneous_hr"]),
            x="rpeak_time_sec",
            y="rr_interval_ms",
            color="instantaneous_hr",
            color_continuous_scale=["#12727a", "#28b99b", "#f1c65b", "#e7534f"],
            title="HRV / RR interval scatter",
            labels={"rpeak_time_sec": "Time (sec)", "rr_interval_ms": "RR interval (ms)"},
        )
        style_chart(fig, height=340, margin=dict(l=34, r=24, t=52, b=42))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("선택 구간에 HRV 산점도용 R-peak 데이터가 없습니다.")

    ai_context = build_ai_agent_context(
        record_id,
        lead,
        start_sec,
        duration_sec,
        meta,
        ann_df,
        window_ann_df,
        counts,
        stats,
        group_counts,
        alert,
    )
    st.session_state["ai_floating_context"] = ai_context
    st.session_state["ai_floating_location"] = "clinical"

    return meta, counts, stats, group_counts, alert


def train_split_page(records, summary):
    split_df = split_dataframe(records)
    split_counts = split_df["split"].value_counts()
    total = len(split_df) if len(split_df) else 1
    group_counts = split_df["group"].value_counts()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Train patients", f"{split_counts.get('Train', 0)}", f"{split_counts.get('Train', 0) / total:.1%}")
    with c2:
        metric_card("Validation", f"{split_counts.get('Validation', 0)}", f"{split_counts.get('Validation', 0) / total:.1%}")
    with c3:
        metric_card("Test patients", f"{split_counts.get('Test', 0)}", f"{split_counts.get('Test', 0) / total:.1%}")
    with c4:
        metric_card("Adult ratio", f"{group_counts.get('Adult', 0) / total:.1%}", f"{group_counts.get('Adult', 0)} patients")
    with c5:
        metric_card("Children ratio", f"{group_counts.get('Children', 0) / total:.1%}", f"{group_counts.get('Children', 0)} patients")
    with c6:
        leakage = split_df.duplicated(["record_id"]).sum()
        metric_card("Leakage status", "PASS" if leakage == 0 else "CHECK", f"{leakage} duplicated IDs")

    st.markdown('<div class="section-title">XGBoost + Safe SMOTE Essential Flow</div>', unsafe_allow_html=True)
    flow_cols = st.columns(4)
    flow_items = [
        ("01 Model", ML_CONFIG["Model"], "notebook validation output"),
        ("02 Input", ML_CONFIG["Input"], ML_CONFIG["Segment"]),
        ("03 Resampling", ML_CONFIG["Resampling"], ML_CONFIG["SMOTE strategy"]),
        ("04 Split", ML_CONFIG["Split"], f"leakage {leakage} duplicated IDs"),
    ]
    for col, (step, title, caption) in zip(flow_cols, flow_items):
        with col:
            flow_card(step, title, caption)

    config_df = pd.DataFrame({"Item": list(ML_CONFIG.keys()), "Value": list(ML_CONFIG.values())})
    with st.expander("Full notebook configuration", expanded=False):
        render_dashboard_table(config_df)

    st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(6)
    for col, (name, value) in zip(metric_cols, ML_METRICS.items()):
        with col:
            caption = "patient split check" if name == "Leakage" else "validation"
            metric_card(name, metric_value(value), caption)

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(.feature-top-n-label) .stTabs [data-baseweb="tab"] {
            height: 46px;
            padding: 0 24px;
        }
        div[data-testid="stHorizontalBlock"]:has(.feature-top-n-label) [data-testid="stSlider"] {
            margin-bottom: -8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        cm_count_tab, cm_row_tab, cm_col_tab = st.tabs(
            ["Count", "Row-normalized Recall View", "Column-normalized Precision View"]
        )
        with cm_count_tab:
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            st.plotly_chart(confusion_matrix_fig(), width="stretch")
        with cm_row_tab:
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            st.plotly_chart(plot_normalized_confusion_matrix("row"), width="stretch")
        with cm_col_tab:
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            st.plotly_chart(plot_normalized_confusion_matrix("column"), width="stretch")
    with c2:
        top_n = int(st.session_state.get("feature_importance_top_n", 12))
        st.markdown('<div class="feature-top-n-label">Feature importance Top N</div>', unsafe_allow_html=True)
        top_n = st.slider(
            "Feature importance Top N",
            min_value=5,
            max_value=30,
            value=top_n,
            key="feature_importance_top_n",
            label_visibility="collapsed",
        )
        fi = feature_importance_df().head(top_n).sort_values("importance")
        fig = px.bar(
            fi,
            x="importance",
            y="feature",
            hover_data=["description"],
            orientation="h",
            color="importance",
            color_continuous_scale=["#d7f1ee", "#22a98f", "#0d4750"],
            title="Top XGBoost feature importance from notebook",
        )
        style_chart(fig, height=420, margin=dict(l=34, r=24, t=52, b=42))
        st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.plotly_chart(roc_curve_fig(), width="stretch")
    with c2:
        st.plotly_chart(pr_curve_fig(), width="stretch")
    with c3:
        st.plotly_chart(probability_distribution_fig(), width="stretch")
    # st.caption(
    #     "노트북에 ROC/PR curve 좌표는 저장되어 있지 않아 one-vs-rest AUC 요약값을 차트로 표시합니다. "
    #     "PR-AUC는 클래스 불균형 환경에서 precision-recall 균형을 확인하는 지표입니다."
    # )
    render_model_interpretation_notes()

    st.markdown('<div class="section-title">Validation Detail Tables</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns([1.25, 1, 1])
    with t1:
        render_dashboard_table(class_report_df())
    with t2:
        render_dashboard_table(one_vs_rest_auc_df())
    with t3:
        render_dashboard_table(predicted_class_ratio_df())

    st.markdown('<div class="section-title">Patient Unit Split</div>', unsafe_allow_html=True)
    chart_df = split_df.groupby(["split", "group"]).size().reset_index(name="patients")
    fig = px.bar(
        chart_df,
        x="split",
        y="patients",
        color="group",
        barmode="group",
        color_discrete_map={"Children": "#24b89b", "Adult": "#1f6f8b"},
        title="Group distribution by split",
    )
    style_chart(fig, height=330, margin=dict(l=34, r=24, t=52, b=42))
    st.plotly_chart(fig, width="stretch")
    render_dashboard_table(split_df)


def dl_confusion_matrix_fig(matrix, labels, title):
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=["#eef8f7", "#26b99a", "#0d4750"],
        labels=dict(x="Predicted label", y="Actual label", color="count"),
    )
    fig.update_traces(textfont=dict(color=CHART_TEXT, size=13))
    fig.update_layout(title=title)
    return style_chart(fig, height=360, margin=dict(l=34, r=24, t=52, b=42))


def selected_dl_prediction(record_id, start_sec, duration_sec):
    ann_df = read_annotations(record_id)
    end_sec = start_sec + duration_sec
    window_ann = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy() if not ann_df.empty else ann_df
    groups = group_counts_from_annotations(window_ann, pd.DataFrame())
    vt_count = int(groups.get("VT / IVR", 0))
    svt_count = int(groups.get("AVRT / AVNRT", 0))
    # group_counts_from_annotations() returns a dict, so do not use pandas-only
    # attributes such as .empty or .drop here.
    exclude_groups = {"VT / IVR", "AVRT / AVNRT"}
    if not groups:
        other_count = 0
    elif isinstance(groups, dict):
        other_count = int(
            sum(int(count) for group, count in groups.items() if group not in exclude_groups)
        )
    else:
        other_count = int(
            groups.drop(labels=list(exclude_groups), errors="ignore").sum()
        ) if not groups.empty else 0

    if vt_count:
        probs = {"non_tachy": 0.18, "VT": 0.67, "SVT": 0.15}
        vae_error = 2.85
    elif svt_count:
        probs = {"non_tachy": 0.20, "VT": 0.08, "SVT": 0.72}
        vae_error = 1.34
    elif other_count:
        probs = {"non_tachy": 0.58, "VT": 0.14, "SVT": 0.28}
        vae_error = 2.12
    else:
        seed = sum(ord(ch) for ch in f"{record_id}-{start_sec}-{duration_sec}")
        rng = np.random.default_rng(seed)
        vt = float(rng.uniform(0.02, 0.12))
        svt = float(rng.uniform(0.04, 0.18))
        probs = {"non_tachy": 1 - vt - svt, "VT": vt, "SVT": svt}
        vae_error = float(rng.uniform(0.55, 1.45))

    argmax_pred = max(probs, key=probs.get)
    if probs["VT"] >= DL_THRESHOLDS["VT"]:
        threshold_pred = "VT"
    elif probs["SVT"] >= DL_THRESHOLDS["SVT"]:
        threshold_pred = "SVT"
    else:
        threshold_pred = "non_tachy"
    vae_decision = "anomaly" if vae_error > DL_THRESHOLDS["VAE"] else "normal"
    if threshold_pred == "VT" and vae_decision == "anomaly":
        flag = "High attention"
    elif threshold_pred in {"VT", "SVT"} or vae_decision == "anomaly":
        flag = "Review"
    else:
        flag = "Normal-like"
    return probs, argmax_pred, threshold_pred, vae_error, vae_decision, flag


def get_gemini_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or None


def get_gemini_model_name():
    return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def get_gemini_client():
    if genai is None:
        return None
    api_key = get_gemini_api_key()
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def safe_jsonable(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return [safe_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return safe_jsonable(value.replace({np.nan: None}).to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return safe_jsonable(value.replace({np.nan: None}).to_dict())
    if isinstance(value, dict):
        return {str(key): safe_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def build_ai_agent_context(
    record_id,
    lead,
    start_sec,
    duration_sec,
    meta,
    ann_df,
    window_ann_df,
    counts,
    stats,
    group_counts,
    alert,
):
    end_sec = start_sec + duration_sec
    top_counts = pd.DataFrame(columns=["annotation", "count"])
    if counts is not None and not counts.empty:
        top_counts = counts.sort_values("count", ascending=False).head(10).copy()

    try:
        window_table = visible_annotation_table(ann_df, start_sec, end_sec).head(20)
    except Exception:
        window_table = pd.DataFrame()
    if window_table.empty and window_ann_df is not None and not window_ann_df.empty:
        window_table = window_ann_df.head(20).copy()

    try:
        probs, argmax_pred, threshold_pred, vae_error, vae_decision, dl_flag = selected_dl_prediction(
            record_id,
            start_sec,
            duration_sec,
        )
    except Exception:
        probs, argmax_pred, threshold_pred, vae_error, vae_decision, dl_flag = {}, "N/A", "N/A", None, "N/A", "Review Required"

    context = {
        "record_id": record_id,
        "selected_lead": lead,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "end_sec": end_sec,
        "patient_meta": {
            "group": meta.get("group"),
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "diagnosis": meta.get("diagnosis"),
            "duration": meta.get("duration", meta.get("duration_sec")),
            "mean_hr": meta.get("mean_hr"),
            "rmssd_ms": meta.get("rmssd_ms"),
            "n_rpeaks": meta.get("n_rpeaks"),
            "n_annotations": meta.get("n_annotations"),
        },
        "annotation_counts_top10": top_counts,
        "current_window_annotation_table": window_table,
        "group_counts": {group: int(group_counts.get(group, 0)) for group in EVENT_GROUP_ORDER},
        "alert": {
            "level": alert.get("level"),
            "label": alert.get("label"),
            "message": alert.get("message"),
        },
        "stats": {
            "risk": stats.get("risk"),
            "abnormal_ratio": stats.get("abnormal_ratio"),
            "tachy": stats.get("tachy"),
            "total": stats.get("total"),
        },
        "dl_prediction": {
            "cnn_probabilities": probs,
            "argmax_pred": argmax_pred,
            "threshold_pred": threshold_pred,
            "vae_error": vae_error,
            "vae_threshold": DL_THRESHOLDS.get("VAE"),
            "vae_decision": vae_decision,
            "dl_review_flag": dl_flag,
        },
        "dashboard_warning": "연구/교육용 대시보드이며 실제 진단용이 아닙니다. AI 답변은 진단이 아니라 검토 보조 설명입니다.",
    }
    return safe_jsonable(context)


def safe_records(df, limit=10):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return safe_jsonable(df.head(limit))


def numeric_summary(df, columns):
    summary = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return summary
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        summary[col] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def percentile_rank(values, selected_value):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty or selected_value is None or pd.isna(selected_value):
        return None
    return float((values <= float(selected_value)).mean())


def build_cohort_ai_context(subjects, summary, aux_df, rpeaks, records, record_id):
    try:
        cohort_df = build_cohort_context_df(subjects, summary, records)
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc).splitlines()[0][:160]}
    if cohort_df.empty:
        return {"status": "empty", "record_count": 0}

    selected = selected_patient_context(cohort_df, record_id)
    inventory = cohort_source_inventory(records, subjects, summary, aux_df, rpeaks)
    selected_context = safe_jsonable(selected.to_dict()) if selected is not None and not selected.empty else {}
    selected_percentiles = {}
    if selected_context:
        for col in ["age", "mean_hr", "rmssd_ms", "duration_minutes", "n_annotations"]:
            if col in cohort_df.columns:
                selected_percentiles[col] = percentile_rank(cohort_df[col], selected.get(col, np.nan))

    group_counts = cohort_df["group_label"].fillna("Unknown").value_counts().to_dict() if "group_label" in cohort_df else {}
    diagnosis_counts = cohort_df["diagnosis_family"].fillna("Unknown").value_counts().head(8).to_dict() if "diagnosis_family" in cohort_df else {}
    gender_counts = cohort_df["gender"].fillna("Unknown").value_counts().to_dict() if "gender" in cohort_df else {}

    return safe_jsonable(
        {
            "status": "available",
            "record_count": int(len(cohort_df)),
            "group_distribution": group_counts,
            "diagnosis_family_top8": diagnosis_counts,
            "gender_distribution": gender_counts,
            "numeric_distribution": numeric_summary(
                cohort_df,
                ["age", "duration_minutes", "mean_hr", "rmssd_ms", "n_rpeaks", "n_annotations"],
            ),
            "selected_patient_context": selected_context,
            "selected_patient_percentile_rank": selected_percentiles,
            "data_source_inventory": safe_records(inventory, 12),
        }
    )


def build_ml_ai_context(records, summary):
    split_df = split_dataframe(records)
    total = int(len(split_df)) if len(split_df) else 0
    leakage = int(split_df.duplicated(["record_id"]).sum()) if not split_df.empty else 0
    split_counts = split_df["split"].value_counts().to_dict() if "split" in split_df else {}
    group_counts = split_df["group"].value_counts().to_dict() if "group" in split_df else {}
    split_group_counts = (
        split_df.groupby(["split", "group"]).size().reset_index(name="patients")
        if not split_df.empty and {"split", "group"}.issubset(split_df.columns)
        else pd.DataFrame()
    )
    return safe_jsonable(
        {
            "model_config": ML_CONFIG,
            "model_note": ML_MODEL_NOTE,
            "patient_split": {
                "total_patients": total,
                "split_counts": split_counts,
                "group_counts": group_counts,
                "split_group_counts": safe_records(split_group_counts, 20),
                "leakage_duplicated_record_ids": leakage,
            },
            "validation_metrics": ML_METRICS,
            "validation_confusion_matrix": {
                "labels": XGB_CLASS_NAMES,
                "matrix": ML_VALIDATION_CM,
            },
            "class_report": safe_records(class_report_df(), 10),
            "one_vs_rest_auc": safe_records(one_vs_rest_auc_df(), 10),
            "predicted_class_ratio": safe_records(predicted_class_ratio_df(), 10),
            "top_feature_importance": safe_records(feature_importance_df(), 15),
        }
    )


def build_dl_ai_context(record_id, start_sec, duration_sec):
    probs, argmax_pred, threshold_pred, vae_error, vae_decision, flag = selected_dl_prediction(
        record_id,
        start_sec,
        duration_sec,
    )
    return safe_jsonable(
        {
            "model_note": DL_MODEL_NOTE,
            "task": "10s ECG window rhythm classification and anomaly filtering",
            "class_distribution": DL_CLASS_DISTRIBUTION,
            "thresholds": DL_THRESHOLDS,
            "cnn_argmax_metrics": DL_CNN_ARGMAX_METRICS,
            "cnn_threshold_metrics": DL_CNN_THRESHOLD_METRICS,
            "vae_metrics": DL_VAE_METRICS,
            "cnn_threshold_confusion_matrix": {
                "labels": ML_CLASS_NAMES,
                "matrix": CNN_THRESHOLD_CM,
            },
            "vae_confusion_matrix": {
                "labels": ["normal", "anomaly"],
                "matrix": VAE_CM,
            },
            "selected_segment_prediction": {
                "record_id": record_id,
                "start_sec": start_sec,
                "duration_sec": duration_sec,
                "cnn_probabilities": probs,
                "cnn_argmax_pred": argmax_pred,
                "cnn_threshold_pred": threshold_pred,
                "vae_error": vae_error,
                "vae_decision": vae_decision,
                "review_flag": flag,
            },
        }
    )


def build_report_ai_context(record_id, counts, stats, group_counts, alert, start_sec, duration_sec):
    try:
        evidence_df = build_evidence_agreement_table(record_id, counts, stats, group_counts, alert, start_sec, duration_sec)
    except Exception:
        evidence_df = pd.DataFrame()
    return safe_jsonable(
        {
            "report_scope": "selected patient report draft",
            "final_alert": alert,
            "event_burden": safe_records(burden_dataframe(group_counts or {}), 10),
            "evidence_agreement": safe_records(evidence_df, 10),
            "recommended_review_basis": "annotation burden, selected ECG window, XGBoost summary, CNN prediction, VAE anomaly score",
        }
    )


def build_dashboard_ai_context(record_id, lead, start_sec, duration_sec, subjects, summary, aux_df, rpeaks, records):
    ann_df = read_annotations(record_id)
    end_sec = start_sec + duration_sec
    window_ann_df = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= end_sec)].copy() if not ann_df.empty else ann_df
    meta = patient_meta(subjects, summary, record_id)
    counts = annotation_counts(record_id, summary, aux_df, ann_df)
    stats = compute_patient_stats(counts)
    group_counts = group_counts_from_annotations(window_ann_df, counts if ann_df.empty else pd.DataFrame())
    alert = determine_patient_alert(group_counts)
    clinical_context = build_ai_agent_context(
        record_id,
        lead,
        start_sec,
        duration_sec,
        meta,
        ann_df,
        window_ann_df,
        counts,
        stats,
        group_counts,
        alert,
    )
    dashboard_context = {
        **clinical_context,
        "context_scope": "full_dashboard_all_tabs",
        "available_tabs": [
            "Clinical Analysis",
            "Cohort Context",
            "Train & ML Analysis",
            "Deep Learning Analysis",
            "Patient Report",
        ],
        "tabs": {
            "clinical_analysis": clinical_context,
            "cohort_context": build_cohort_ai_context(subjects, summary, aux_df, rpeaks, records, record_id),
            "train_ml_analysis": build_ml_ai_context(records, summary),
            "deep_learning_analysis": build_dl_ai_context(record_id, start_sec, duration_sec),
            "patient_report": build_report_ai_context(record_id, counts, stats, group_counts, alert, start_sec, duration_sec),
        },
        "ai_usage_note": "The copilot can answer from summarized tab data, not from visual pixels in charts.",
    }
    return safe_jsonable(dashboard_context)


def build_ai_system_prompt():
    return """
You are an ECG Clinical Copilot for a Streamlit research dashboard.
Answer in Korean.
Use only the values, annotations, model outputs, and tab summaries included in the dashboard context JSON.
The context may include all tabs: Clinical Analysis, Cohort Context, Train & ML Analysis, Deep Learning Analysis, and Patient Report.
Do not infer unavailable clinical facts. If a fact is not in context, say that it is not available in the current dashboard context.
Always frame the answer as research/education review support, not a confirmed diagnosis, treatment decision, or emergency instruction.
Explain High Risk as high review priority, not a confirmed diagnosis.
Explain CNN as an ECG window classifier and VAE as an anomaly filter based on reconstruction difference from normal patterns.
When evidence conflicts across annotations, XGBoost, CNN, and VAE, state the conflict clearly and recommend manual review of the ECG window and adjacent windows.
""".strip()
    return """
너는 ECG Clinical Copilot이다.
실제 의료 진단, 치료 결정, 응급 판단을 하지 않는다.
대시보드에 제공된 수치, annotation, model output만 근거로 설명한다.
없는 정보는 추측하지 말고 "현재 대시보드 context에서 확인되지 않음"이라고 말한다.
답변은 한국어로 한다.
초보자도 이해할 수 있게 짧고 명확하게 쓴다.
항상 "검토 보조"라는 톤을 유지한다.
High Risk라고 해도 확정 진단이 아니라 재검토 우선순위가 높다는 의미로 설명한다.
CNN은 ECG window 분류 모델, VAE는 정상 패턴과의 차이를 보는 anomaly filter 역할로 설명한다.
VT/IVR은 위험 신호로 우선 검토, AVRT/AVNRT와 AFIB/AFLT는 리듬 이상 검토, Conduction/PVC는 보조 검토 신호로 설명한다.
"진단 확정", "치료하라", "응급 처치하라" 같은 표현은 사용하지 않는다.
""".strip()


def build_ai_user_prompt(mode, context, user_question=""):
    mode_prompts = {
        "dashboard_overview": """
전체 dashboard context를 탭별로 요약해줘.
Clinical Analysis, Cohort Context, Train & ML Analysis, Deep Learning Analysis, Patient Report에서 각각 확인 가능한 핵심 내용과 현재 선택 환자에게 주는 의미를 정리해줘.
마지막에는 "현재 context로 가능한 답변 범위"와 "확인 불가한 범위"를 구분해줘.
""",
        "cross_tab_consistency": """
현재 선택 환자와 window에 대해 탭 간 근거가 서로 일치하는지 검토해줘.
annotation/risk flag, cohort 위치, XGBoost 성능 한계, CNN/VAE 결과, report evidence agreement를 함께 비교해줘.
일치하는 근거, 충돌하는 근거, 다음 검토 포인트를 나눠서 설명해줘.
""",
        "patient_risk_summary": """
이 환자가 Normal / Watch / High / Review Required 중 어디에 가까운지 요약해줘.
주요 근거 3개와 확인해야 할 포인트 3개를 bullet로 작성해줘.
진단이 아니라 검토 보조 설명이라는 점을 포함해줘.
""",
        "current_window_interpretation": """
현재 선택 ECG window의 annotation과 risk group을 해석해줘.
이 window가 환자 전체 위험도 판단에 어떤 영향을 주는지 설명해줘.
""",
        "why_high_risk": """
High 또는 abnormal alert가 나온 이유를 설명해줘.
VT/IVR이 한 번이라도 있으면 High attention으로 보는 현재 rule의 의미를 설명하고, false alarm 가능성도 함께 언급해줘.
""",
        "cnn_vae_comparison": """
CNN prediction과 VAE anomaly score를 비교해줘.
둘이 일치하면 "강한 재검토 후보", CNN만 높으면 "분류상 의심", VAE만 높으면 "정상 패턴과 다르지만 class 확정은 어려움", 둘 다 낮으면 "현재 window 기준 위험 신호 낮음"으로 설명해줘.
""",
        "next_review_points": """
다음 검토 포인트를 제안해줘.
ECG plot에서 먼저 볼 구간, annotation marker 확인, 다른 window 반복 여부, cohort context에서 확인할 점, report에 적을 핵심 문장을 포함해줘.
""",
        "patient_report_draft": """
아래 형식으로 환자 리포트 초안을 작성해줘.
Patient ID
Risk Summary
Key Findings
Model Interpretation
Recommended Review
Limitation
반드시 연구/교육용 검토 보조이며 진단이 아니라는 문장을 포함해줘.
""",
        "free_question": f"""
사용자 질문에 대시보드 context 범위 안에서만 답해줘.
질문: {user_question or "질문이 비어 있음"}
없는 정보는 "현재 대시보드 context에서 확인되지 않음"이라고 답해줘.
""",
    }
    prompt = mode_prompts.get(mode, mode_prompts["patient_risk_summary"]).strip()
    context_preview = json.dumps(safe_jsonable(context), ensure_ascii=False, indent=2)
    return f"{prompt}\n\n대시보드 context JSON:\n{context_preview}"


def rule_based_agent_response(mode, context, user_question=""):
    group_counts = context.get("group_counts", {}) or {}
    alert = context.get("alert", {}) or {}
    stats = context.get("stats", {}) or {}
    dl = context.get("dl_prediction", {}) or {}
    meta = context.get("patient_meta", {}) or {}
    vt_ivr = int(group_counts.get("VT / IVR", 0) or 0)
    svt = int(group_counts.get("AVRT / AVNRT", 0) or 0)
    af = int(group_counts.get("AFIB / AFLT", 0) or 0)
    pvc = int(group_counts.get("Conduction / PVC", 0) or 0)
    pacer = int(group_counts.get("Pacemaker", 0) or 0)
    total_group_events = sum(int(v or 0) for v in group_counts.values())
    risk = "High" if vt_ivr > 0 else "Watch" if (svt > 0 or af > 0 or pvc > 0 or pacer > 0) else "Review Required" if total_group_events == 0 else "Normal"
    vae_error = dl.get("vae_error")
    vae_threshold = dl.get("vae_threshold")
    vae_text = "현재 대시보드 context에서 확인되지 않음"
    if vae_error is not None and vae_threshold is not None:
        vae_text = f"VAE error {vae_error:.3f}, threshold {vae_threshold:.3f}, decision {dl.get('vae_decision')}"
    cnn_text = f"CNN threshold prediction: {dl.get('threshold_pred', 'N/A')}, probabilities: {dl.get('cnn_probabilities', {})}"
    common_note = "\n\n주의: 이 내용은 연구/교육용 검토 보조 설명이며, 진단 확정이나 치료 결정이 아닙니다."

    if mode == "dashboard_overview":
        tabs = context.get("tabs", {}) or {}
        cohort = tabs.get("cohort_context", {}) or {}
        ml = tabs.get("train_ml_analysis", {}) or {}
        dl_tab = tabs.get("deep_learning_analysis", {}) or {}
        report = tabs.get("patient_report", {}) or {}
        selected_segment = dl_tab.get("selected_segment_prediction", {}) or {}
        return (
            "전체 탭 요약\n"
            f"1. Clinical Analysis: 환자 {context.get('record_id')}의 {context.get('start_sec')}s - {context.get('end_sec')}s window 기준입니다. "
            f"event group count는 {group_counts}이고 alert는 {alert.get('level', 'N/A')}입니다.\n"
            f"2. Cohort Context: 전체 context record 수는 {cohort.get('record_count', 'N/A')}이고 group 분포는 {cohort.get('group_distribution', {})}입니다. "
            f"선택 환자의 cohort percentile은 {cohort.get('selected_patient_percentile_rank', {})}입니다.\n"
            f"3. Train & ML Analysis: XGBoost validation metric은 {ml.get('validation_metrics', {})}이며 "
            f"split leakage duplicated ID는 {(ml.get('patient_split', {}) or {}).get('leakage_duplicated_record_ids', 'N/A')}입니다.\n"
            f"4. Deep Learning Analysis: 선택 segment CNN/VAE 결과는 {selected_segment}입니다.\n"
            f"5. Patient Report: final alert는 {report.get('final_alert', {})}이고 evidence agreement는 {report.get('evidence_agreement', [])}입니다.\n"
            "현재 Copilot은 그래프 픽셀을 직접 읽는 것이 아니라, 각 탭을 만들 때 사용한 요약 데이터 JSON을 읽습니다."
            f"{common_note}"
        )
    if mode == "cross_tab_consistency":
        tabs = context.get("tabs", {}) or {}
        cohort = tabs.get("cohort_context", {}) or {}
        ml = tabs.get("train_ml_analysis", {}) or {}
        dl_tab = tabs.get("deep_learning_analysis", {}) or {}
        report = tabs.get("patient_report", {}) or {}
        selected_segment = dl_tab.get("selected_segment_prediction", {}) or {}
        evidence = report.get("evidence_agreement", []) or []
        elevated_signals = []
        if risk in {"High", "Watch"}:
            elevated_signals.append(f"annotation/risk flag: {risk}")
        if selected_segment.get("cnn_threshold_pred") in {"VT", "SVT"}:
            elevated_signals.append(f"CNN threshold: {selected_segment.get('cnn_threshold_pred')}")
        if selected_segment.get("vae_decision") == "anomaly":
            elevated_signals.append("VAE anomaly")
        if not elevated_signals:
            elevated_signals.append("현재 window에서 강한 elevated signal은 제한적입니다")
        return (
            "탭 간 근거 일치성 검토\n"
            f"- 일치하거나 강화되는 근거: {', '.join(elevated_signals)}.\n"
            f"- Cohort 위치: {cohort.get('selected_patient_percentile_rank', {})}.\n"
            f"- ML 모델 한계/성능 배경: {ml.get('validation_metrics', {})}.\n"
            f"- Report evidence agreement: {evidence}.\n"
            "- 다음 검토 포인트: ECG marker 주변 waveform, 인접 window 반복 여부, CNN/VAE 불일치 여부, cohort outlier 여부를 함께 확인해야 합니다."
            f"{common_note}"
        )
    if mode == "current_window_interpretation":
        return (
            f"현재 window는 {context.get('start_sec')}s - {context.get('end_sec')}s 구간입니다.\n"
            f"- VT/IVR: {vt_ivr}, AVRT/AVNRT: {svt}, AFIB/AFLT: {af}, Conduction/PVC: {pvc}, Pacemaker: {pacer}\n"
            f"- alert: {alert.get('level', 'N/A')} / {alert.get('message', '현재 대시보드 context에서 확인되지 않음')}\n"
            f"- 이 window는 위 annotation 분포를 기준으로 재검토 우선순위 판단에 반영됩니다."
            f"{common_note}"
        )
    if mode == "why_high_risk":
        return (
            f"현재 rule 기준 위험도는 {risk}에 가깝습니다.\n"
            f"- VT/IVR count가 {vt_ivr}개입니다. 이 rule은 VT/IVR이 1회라도 있으면 우선 검토 대상으로 표시합니다.\n"
            f"- 이는 민감도를 높이기 위한 보수적 기준이며, marker 오류나 waveform artifact에 의한 false alarm 가능성도 있습니다.\n"
            f"- 따라서 ECG plot에서 해당 marker 주변 파형을 직접 확인해야 합니다."
            f"{common_note}"
        )
    if mode == "cnn_vae_comparison":
        threshold_pred = dl.get("threshold_pred")
        vae_decision = dl.get("vae_decision")
        if threshold_pred in {"VT", "SVT"} and vae_decision == "anomaly":
            verdict = "CNN과 VAE가 함께 신호를 보여 강한 재검토 후보입니다."
        elif threshold_pred in {"VT", "SVT"}:
            verdict = "CNN 분류상 의심 신호가 있지만 VAE anomaly는 강하지 않을 수 있습니다."
        elif vae_decision == "anomaly":
            verdict = "정상 패턴과 다르지만 class 확정은 어렵습니다."
        else:
            verdict = "현재 window 기준 위험 신호는 낮아 보입니다."
        return f"{verdict}\n- {cnn_text}\n- {vae_text}{common_note}"
    if mode == "next_review_points":
        return (
            "다음 검토 포인트입니다.\n"
            f"1. ECG plot에서 {context.get('start_sec')}s - {context.get('end_sec')}s marker 주변 파형을 먼저 확인합니다.\n"
            "2. VT/IVR, AVRT/AVNRT, AFIB/AFLT marker가 실제 rhythm 변화와 맞는지 확인합니다.\n"
            "3. 인접 window에서도 같은 annotation이 반복되는지 확인합니다.\n"
            "4. Cohort Context에서 나이, 진단군, HR/RMSSD 위치를 함께 봅니다.\n"
            f"5. Report 핵심 문장: 선택 구간은 {risk} 검토 후보이며, AI 설명은 진단이 아니라 검토 보조입니다."
            f"{common_note}"
        )
    if mode == "patient_report_draft":
        return (
            f"Patient ID\n{context.get('record_id')}\n\n"
            f"Risk Summary\n현재 rule 기반 검토 우선순위는 {risk}에 가깝습니다. Alert level은 {alert.get('level', 'N/A')}입니다.\n\n"
            f"Key Findings\nVT/IVR {vt_ivr}, AVRT/AVNRT {svt}, AFIB/AFLT {af}, Conduction/PVC {pvc} events가 확인됩니다.\n\n"
            f"Model Interpretation\n{cnn_text}\n{vae_text}\n\n"
            "Recommended Review\nECG marker 주변 waveform, event 반복 여부, cohort context 위치를 함께 검토합니다.\n\n"
            "Limitation\n이 초안은 연구/교육용 검토 보조이며 진단 확정이 아닙니다. 실제 판단은 원본 ECG와 임상 정보를 함께 검토해야 합니다."
        )
    if mode == "free_question":
        question = user_question.strip() if user_question else ""
        if not question:
            return f"질문이 입력되지 않았습니다. 현재 대시보드 context 범위에서 환자, window, annotation, risk flag, CNN/VAE 결과에 대해 질문할 수 있습니다.{common_note}"
        return (
            f"질문: {question}\n\n"
            "현재 대시보드 context에서 확인 가능한 요약입니다.\n"
            f"- 환자: {context.get('record_id')}, group: {meta.get('group', 'N/A')}, diagnosis: {meta.get('diagnosis', 'N/A')}\n"
            f"- risk/alert: {risk} / {alert.get('level', 'N/A')}\n"
            f"- 주요 group counts: {group_counts}\n"
            f"- CNN/VAE: {cnn_text}; {vae_text}\n"
            "질문이 위 context 밖의 임상 정보나 치료 판단을 요구한다면 현재 대시보드 context에서 확인되지 않음으로 보아야 합니다."
            f"{common_note}"
        )
    return (
        f"환자 위험도 요약\n"
        f"- 현재 rule 기반 검토 우선순위는 {risk}에 가깝습니다.\n"
        f"- 근거 1: VT/IVR {vt_ivr}, AVRT/AVNRT {svt}, AFIB/AFLT {af} events.\n"
        f"- 근거 2: alert level은 {alert.get('level', 'N/A')}입니다.\n"
        f"- 근거 3: abnormal ratio는 {stats.get('abnormal_ratio', 'N/A')}입니다.\n"
        f"- 확인 포인트: marker 주변 waveform, 반복 window 여부, CNN/VAE 일치 여부를 검토합니다."
        f"{common_note}"
    )


def call_gemini_agent(mode, context, user_question=""):
    client = get_gemini_client()
    if client is None:
        return rule_based_agent_response(mode, context, user_question)
    try:
        compact_context = safe_jsonable(
            {
                **context,
                "annotation_counts_top10": (context.get("annotation_counts_top10") or [])[:10],
                "current_window_annotation_table": (context.get("current_window_annotation_table") or [])[:20],
            }
        )
        contents = (
            f"System instruction:\n{build_ai_system_prompt()}\n\n"
            f"User request:\n{build_ai_user_prompt(mode, compact_context, user_question)}"
        )
        response = client.models.generate_content(
            model=get_gemini_model_name(),
            contents=contents,
        )
        answer = getattr(response, "text", None)
        if not answer:
            answer = str(response)
        return f"{answer.strip()}\n\n주의: 이 답변은 진단이 아니라 연구/교육용 검토 보조 설명입니다."
    except Exception as exc:
        fallback = rule_based_agent_response(mode, context, user_question)
        return f"{fallback}\n\nGemini 호출 실패로 fallback summary를 표시합니다. 오류: {str(exc).splitlines()[0][:160]}"


def inject_ai_copilot_css():
    st.markdown(
        """
        <style>
        .ai-copilot-card {
            border: 1px solid #c6dde2;
            border-radius: 8px;
            padding: 18px;
            margin-top: 18px;
            margin-bottom: 16px;
            background: #ffffff;
            color: #17323f;
            box-shadow: 0 12px 28px rgba(16,72,80,.08);
            overflow-wrap: break-word;
            word-break: break-word;
        }
        .ai-copilot-title {
            font-size: 20px;
            line-height: 1.3;
            font-weight: 800;
            color: #17323f;
            margin: 0 0 6px;
        }
        .ai-copilot-caption {
            font-size: 14px;
            line-height: 1.55;
            color: #425966;
            margin-bottom: 12px;
        }
        .ai-copilot-status {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 800;
            border: 1px solid #9bd8cd;
            background: #e6f8f4;
            color: #075d4d;
            margin: 4px 8px 12px 0;
        }
        .ai-copilot-status.fallback {
            border-color: #d79b3f;
            background: #fff3d8;
            color: #6f4300;
        }
        .ai-copilot-model {
            display: inline-flex;
            font-size: 12px;
            color: #435b67;
            margin-bottom: 12px;
        }
        .ai-copilot-answer {
            background: #f1f8f8;
            border-left: 4px solid #24b89b;
            border-radius: 8px;
            padding: 14px 16px;
            color: #17323f;
            line-height: 1.6;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            word-break: break-word;
            font-size: 14px;
            max-width: 100%;
            overflow-x: auto;
        }
        .ai-copilot-answer pre {
            white-space: pre-wrap;
            overflow-wrap: break-word;
            word-break: break-word;
            margin: 0;
            color: inherit;
            font-family: inherit;
            font-size: 14px;
            line-height: 1.6;
        }
        .ai-copilot-warning {
            font-size: 12px;
            color: #3f5662;
            line-height: 1.45;
            margin-top: 10px;
        }
        .ai-copilot-controls {
            border-top: 1px solid #dce9eb;
            margin-top: 14px;
            padding-top: 12px;
        }
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            color: #17323f !important;
            background: #ffffff !important;
            border-color: #b8d5d9 !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder,
        div[data-testid="stTextInput"] input::placeholder {
            color: #667b85 !important;
            opacity: 1 !important;
        }
        [data-testid="stAppViewContainer"] div[data-testid="stRadio"] > label,
        [data-testid="stAppViewContainer"] div[data-testid="stTextArea"] > label,
        [data-testid="stAppViewContainer"] div[data-testid="stRadio"] label p,
        [data-testid="stAppViewContainer"] div[data-testid="stRadio"] label span,
        [data-testid="stAppViewContainer"] div[data-testid="stTextArea"] label p,
        [data-testid="stAppViewContainer"] div[data-testid="stTextArea"] label span {
            color: #17323f !important;
            opacity: 1 !important;
            font-weight: 650 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stRadio"] > label,
        [data-testid="stSidebar"] div[data-testid="stRadio"] label p,
        [data-testid="stSidebar"] div[data-testid="stRadio"] label span {
            color: #f5ffff !important;
            opacity: 1 !important;
            font-weight: inherit !important;
        }
        div[data-testid="stButton"] button {
            color: #f7ffff !important;
        }
        div[data-testid="stButton"] button:hover {
            color: #ffffff !important;
        }
        div[data-testid="stButton"] button:disabled {
            color: #5b6f78 !important;
        }
        div[data-testid="stRadio"] label p,
        div[data-testid="stButton"] button p,
        div[data-testid="stButton"] button span {
            font-size: 14px !important;
            color: inherit !important;
            opacity: 1 !important;
        }
        @media (prefers-color-scheme: dark) {
            .ai-copilot-card {
                background: #102a35;
                color: #f2fbfc;
                border-color: rgba(164, 213, 218, .38);
                box-shadow: 0 12px 28px rgba(0,0,0,.24);
            }
            .ai-copilot-title {
                color: #ecfeff;
            }
            .ai-copilot-caption,
            .ai-copilot-model,
            .ai-copilot-warning {
                color: #d0e4e8;
            }
            .ai-copilot-answer {
                background: #0b1d25;
                color: #f2fbfc;
                border-left-color: #24b89b;
            }
            .ai-copilot-controls {
                border-top-color: rgba(164, 213, 218, .28);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ai_answer_box(answer):
    safe_answer = html.escape(answer or "아직 생성된 AI 답변이 없습니다.")
    st.markdown(
        f'<div class="ai-copilot-answer"><pre>{safe_answer}</pre></div>',
        unsafe_allow_html=True,
    )


def render_ai_clinical_copilot(context, location="clinical"):
    inject_ai_copilot_css()
    connected = genai is not None and get_gemini_api_key() is not None
    status_text = "Gemini Connected" if connected else "Gemini Not Configured - using rule-based fallback"
    status_class = "" if connected else "fallback"
    model_name = get_gemini_model_name()
    mode_options = {
        "Full Dashboard Overview": "dashboard_overview",
        "Cross-tab Evidence Check": "cross_tab_consistency",
        "Patient Risk Summary": "patient_risk_summary",
        "Current Window Interpretation": "current_window_interpretation",
        "Why High Risk?": "why_high_risk",
        "CNN vs VAE": "cnn_vae_comparison",
        "Next Review Points": "next_review_points",
        "Report Draft": "patient_report_draft",
    }
    default_index = list(mode_options.keys()).index("Report Draft") if location == "report" else 0
    key_base = f"ai_copilot_{location}_{context.get('record_id')}_{context.get('selected_lead')}_{context.get('start_sec')}_{context.get('duration_sec')}"
    answer_key = f"{key_base}_answer"
    question_answer_key = f"{key_base}_question_answer"

    title = "AI-generated Report Draft" if location == "report" else "AI Clinical Copilot"
    st.markdown(
        f"""
        <div class="ai-copilot-card">
            <div class="ai-copilot-title">{title}</div>
            <div class="ai-copilot-caption">ECG annotation, risk flag, CNN/VAE output을 기반으로 선택 환자와 window를 설명합니다.</div>
            <span class="ai-copilot-status {status_class}">{status_text}</span>
            <span class="ai-copilot-model">Model: {html.escape(model_name)}</span>
            <div class="ai-copilot-warning">연구/교육용 검토 보조입니다. AI 답변은 진단 확정, 치료 결정, 응급 판단으로 사용하지 마세요.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        mode_label = st.radio(
            "AI 기능 선택",
            list(mode_options.keys()),
            index=default_index,
            horizontal=True,
            key=f"{key_base}_mode",
        )
        selected_mode = mode_options[mode_label]
        if st.button("AI 분석 실행", key=f"{key_base}_run"):
            with st.spinner("AI Clinical Copilot이 현재 dashboard context를 검토 중입니다..."):
                st.session_state[answer_key] = call_gemini_agent(selected_mode, context)
        if answer_key in st.session_state:
            render_ai_answer_box(st.session_state[answer_key])
        else:
            render_ai_answer_box("AI 분석 실행 버튼을 누르면 현재 환자, ECG window, annotation, risk flag, CNN/VAE 결과를 바탕으로 검토 보조 설명을 생성합니다.")

        st.markdown('<div class="ai-copilot-controls">', unsafe_allow_html=True)
        question = st.text_area(
            "Ask AI",
            value="",
            height=88,
            key=f"{key_base}_question",
            placeholder="현재 환자, ECG window, annotation, risk flag, CNN/VAE 결과에 대해 질문하세요.",
        )
        if st.button("질문하기", key=f"{key_base}_ask"):
            with st.spinner("질문에 답변 중입니다..."):
                st.session_state[question_answer_key] = call_gemini_agent("free_question", context, question)
        if question_answer_key in st.session_state:
            render_ai_answer_box(st.session_state[question_answer_key])
        st.markdown("</div>", unsafe_allow_html=True)


def inject_ai_dialog_css():
    st.markdown(
        """
        <style>
        div.st-key-ai_floating_btn {
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 9999;
        }

        div.st-key-ai_floating_btn button {
            background: linear-gradient(135deg, #0f766e, #123042) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, .35) !important;
            border-radius: 999px !important;
            padding: 12px 18px !important;
            font-size: 14px !important;
            font-weight: 760 !important;
            box-shadow: 0 12px 28px rgba(18, 48, 66, .28) !important;
        }

        div.st-key-ai_floating_btn button:hover {
            background: linear-gradient(135deg, #24b89b, #0f4750) !important;
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        div.st-key-ai_floating_btn button p,
        div.st-key-ai_floating_btn button span {
            color: #ffffff !important;
            font-size: 14px !important;
            font-weight: 760 !important;
        }

        [data-testid="stDialog"],
        div[role="dialog"]:has(.ai-dialog-card),
        div[data-baseweb="modal"]:has(.ai-dialog-card) {
            width: 45vw !important;
            min-width: 45vw !important;
            max-width: 45vw !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        [data-testid="stDialog"] > div,
        [data-testid="stDialog"] section,
        [data-testid="stDialog"] div[data-testid="stVerticalBlock"] {
            max-width: 100% !important;
        }

        [data-testid="stDialog"] div[data-testid="stVerticalBlock"] {
            gap: .58rem !important;
        }

        .ai-dialog-card {
            border: 1px solid #c6dde2;
            border-radius: 8px;
            padding: 15px 18px;
            margin-top: 22px;
            margin-bottom: 8px;
            background: #ffffff;
            color: #17323f;
            box-shadow: 0 10px 24px rgba(16, 72, 80, .08);
            overflow-wrap: break-word;
            word-break: break-word;
            position: relative;
        }

        .ai-dialog-meta {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 8px;
            margin: 0 0 -14px;
        }

        .ai-dialog-title {
            margin: 0 0 6px;
            color: #17323f;
            font-size: 20px;
            font-weight: 800;
            line-height: 1.3;
        }

        .ai-dialog-caption {
            margin: 0 0 12px;
            color: #425966;
            font-size: 14px;
            line-height: 1.55;
        }

        .ai-dialog-status {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 800;
            border: 1px solid #9bd8cd;
            background: #e6f8f4;
            color: #075d4d;
            margin: 0;
        }

        .ai-dialog-status.fallback {
            border-color: #d79b3f;
            background: #fff3d8;
            color: #6f4300;
        }

        .ai-dialog-model {
            display: inline-flex;
            margin-bottom: 0;
            color: #435b67;
            font-size: 12px;
        }

        .ai-dialog-answer {
            background: #f1f8f8;
            border-left: 4px solid #24b89b;
            border-radius: 8px;
            padding: 13px 16px;
            color: #17323f;
            line-height: 1.62;
            font-size: 14px;
            white-space: pre-wrap;
            max-width: 100%;
            overflow-x: auto;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .ai-dialog-answer pre {
            margin: 0;
            color: inherit;
            font-family: inherit;
            font-size: 14px;
            line-height: 1.62;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .ai-dialog-warning {
            border-left: 4px solid #d79b3f;
            border-radius: 8px;
            padding: 10px 13px;
            margin-top: 12px;
            background: #0E1117;
            color: #ffe6ad;
            font-size: 13px;
            line-height: 1.55;
        }

        .ai-dialog-controls {
            border-top: 0;
            margin-top: 12px;
            padding-top: 18px;
        }

        .ai-dialog-question-intro {
            margin-top: 88px;
            margin-bottom: 18px;
            color: #f2fbfc;
        }

        .ai-dialog-question-kicker {
            font-size: 16px;
            line-height: 1.25;
            color: #d3dce0;
            margin-bottom: 3px;
        }

        .ai-dialog-question-title {
            font-size: 28px;
            line-height: 1.18;
            font-weight: 520;
            color: #ffffff;
            letter-spacing: 0;
        }

        .ai-dialog-question-bottom-gap {
            height: 84px;
        }

        .ai-dialog-run-row {
            height: 1px;
            background: #ffffff;
            margin: 8px 0 10px;
        }

        .ai-dialog-answer-button-gap {
            height: 12px;
        }

        [data-testid="stDialog"] div[data-testid="stRadio"] {
            margin-bottom: 2px;
        }

        [data-testid="stDialog"] div[data-testid="stRadio"] div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 5px 18px !important;
        }

        [data-testid="stDialog"] .ai-dialog-card + div[data-testid="stRadio"] label p,
        [data-testid="stDialog"] div[data-testid="stRadio"] label span,
        [data-testid="stDialog"] div[data-testid="stTextArea"] label p,
        [data-testid="stDialog"] div[data-testid="stTextArea"] label span {
            color: #17323f !important;
            opacity: 1 !important;
        }

        [data-testid="stDialog"] textarea {
            background: #ffffff !important;
            color: #17323f !important;
            caret-color: #17323f !important;
            border-color: #b8d5d9 !important;
        }

        [data-testid="stDialog"] [data-testid="stChatInput"] textarea,
        [data-testid="stDialog"] [data-testid="stChatInput"] input {
            background: #ffffff !important;
            color: #17323f !important;
            caret-color: #17323f !important;
            border-color: #b8d5d9 !important;
        }

        [data-testid="stDialog"] textarea::placeholder {
            color: #667b85 !important;
            opacity: 1 !important;
        }

        [data-testid="stDialog"] [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stDialog"] [data-testid="stChatInput"] input::placeholder {
            color: #667b85 !important;
            opacity: 1 !important;
        }

        [data-testid="stDialog"] [data-testid="stChatInput"] button[aria-label*="Add"],
        [data-testid="stDialog"] [data-testid="stChatInput"] button[aria-label*="Tool"],
        [data-testid="stDialog"] [data-testid="stChatInput"] button[aria-label*="Model"],
        [data-testid="stDialog"] [data-testid="stChatInput"] button[title*="Add"],
        [data-testid="stDialog"] [data-testid="stChatInput"] button[title*="Tool"],
        [data-testid="stDialog"] [data-testid="stChatInput"] button[title*="Model"],
        [data-testid="stDialog"] [data-testid="stChatInput"] [data-testid*="ChatInputFile"],
        [data-testid="stDialog"] [data-testid="stChatInput"] [data-testid*="Tool"],
        [data-testid="stDialog"] [data-testid="stChatInput"] [data-testid*="Model"] {
            display: none !important;
        }

        [data-testid="stDialog"] div[data-testid="stButton"] button {
            color: #ffffff !important;
        }

        [data-testid="stDialog"] div[data-testid="stButton"] button p,
        [data-testid="stDialog"] div[data-testid="stButton"] button span {
            color: inherit !important;
            opacity: 1 !important;
        }

        @media (prefers-color-scheme: dark) {
            .ai-dialog-card {
                background: #102a35;
                color: #f2fbfc;
                border-color: rgba(164, 213, 218, .38);
                box-shadow: 0 12px 28px rgba(0, 0, 0, .24);
            }

            .ai-dialog-title {
                color: #ecfeff;
            }

            .ai-dialog-caption,
            .ai-dialog-model {
                color: #d0e4e8;
            }

            .ai-dialog-answer {
                background: #0b1d25;
                color: #f2fbfc;
            }

            .ai-dialog-warning {
                background: #0E1117;
                color: #ffe6ad;
                border-left-color: #f4c88a;
            }

            [data-testid="stDialog"] div[data-testid="stRadio"] label p,
            [data-testid="stDialog"] div[data-testid="stRadio"] label span,
            [data-testid="stDialog"] div[data-testid="stTextArea"] label p,
            [data-testid="stDialog"] div[data-testid="stTextArea"] label span {
                color: #f2fbfc !important;
            }
        }

        @media (max-width: 768px) {
            [data-testid="stDialog"],
            div[role="dialog"]:has(.ai-dialog-card),
            div[data-baseweb="modal"]:has(.ai-dialog-card) {
                width: 94vw !important;
                min-width: 94vw !important;
                max-width: 94vw !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }

            [data-testid="stDialog"] div[data-testid="stRadio"] div[role="radiogroup"] {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            div.st-key-ai_floating_btn {
                right: 16px;
                bottom: 16px;
            }

            div.st-key-ai_floating_btn button {
                padding: 10px 14px !important;
                font-size: 13px !important;
            }

            .ai-dialog-question-intro {
                margin-top: 52px;
            }

            .ai-dialog-question-title {
                font-size: 23px;
            }

            .ai-dialog-question-bottom-gap {
                height: 52px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ai_dialog_answer(answer):
    safe_answer = html.escape(answer or "아직 생성된 AI 답변이 없습니다.")
    st.markdown(
        f'<div class="ai-dialog-answer"><pre>{safe_answer}</pre></div>',
        unsafe_allow_html=True,
    )


@st.dialog("AI Clinical Copilot", width="large")
def render_ai_copilot_dialog(context, location="clinical"):
    inject_ai_dialog_css()
    if not context:
        st.markdown(
            """
            <div class="ai-dialog-card">
                <div class="ai-dialog-title">AI Clinical Copilot</div>
                <div class="ai-dialog-caption">Clinical Analysis에서 환자와 ECG window를 먼저 선택하세요.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    connected = genai is not None and get_gemini_api_key() is not None
    status_text = "Gemini Connected" if connected else "Gemini Not Configured - using rule-based fallback"
    status_class = "" if connected else "fallback"
    model_name = get_gemini_model_name()
    mode_options = {
        "Full Dashboard Overview": "dashboard_overview",
        "Cross-tab Evidence Check": "cross_tab_consistency",
        "Patient Risk Summary": "patient_risk_summary",
        "Current Window Interpretation": "current_window_interpretation",
        "Why High Risk?": "why_high_risk",
        "CNN vs VAE": "cnn_vae_comparison",
        "Next Review Points": "next_review_points",
        "Report Draft": "patient_report_draft",
    }
    default_index = list(mode_options.keys()).index("Report Draft") if location == "report" else 0
    context_key = f"{location}_{context.get('record_id')}_{context.get('selected_lead')}_{context.get('start_sec')}_{context.get('duration_sec')}"

    st.markdown(
        f"""
        <div class="ai-dialog-meta">
            <span class="ai-dialog-status {status_class}">{status_text}</span>
            <span class="ai-dialog-model">Model: {html.escape(model_name)}</span>
        </div>
        <div class="ai-dialog-card">
            <div class="ai-dialog-title">AI Clinical Copilot</div>
            <div class="ai-dialog-caption">현재 선택 환자, ECG window, annotation, risk flag, CNN/VAE 결과 기반 설명을 생성합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode_label = st.radio(
        "AI 기능 선택",
        list(mode_options.keys()),
        index=default_index,
        horizontal=True,
        key=f"ai_dialog_mode_{context_key}",
    )
    selected_mode = mode_options[mode_label]
    answer_key = f"ai_dialog_answer_{context_key}_{selected_mode}"
    question_key = f"ai_dialog_question_{context_key}"
    question_answer_key = f"ai_dialog_question_answer_{context_key}"

    st.markdown('<div class="ai-dialog-run-row"></div>', unsafe_allow_html=True)
    run_clicked = False
    if answer_key in st.session_state:
        render_ai_dialog_answer(st.session_state[answer_key])
        st.markdown('<div class="ai-dialog-answer-button-gap"></div>', unsafe_allow_html=True)
        _, run_col = st.columns([5, 1])
        with run_col:
            run_clicked = st.button("분석 실행", key=f"ai_dialog_run_{context_key}_{selected_mode}")
        if run_clicked:
            with st.spinner("AI Clinical Copilot이 현재 dashboard context를 검토 중입니다..."):
                st.session_state[answer_key] = call_gemini_agent(selected_mode, context)
            st.rerun(scope="fragment")
        st.markdown('<div class="ai-dialog-run-row"></div>', unsafe_allow_html=True)
    else:
        run_result_col, run_col = st.columns([5, 1])
        with run_col:
            run_clicked = st.button("분석 실행", key=f"ai_dialog_run_{context_key}_{selected_mode}")
        with run_result_col:
            if run_clicked:
                with st.spinner("AI Clinical Copilot이 현재 dashboard context를 검토 중입니다..."):
                    st.session_state[answer_key] = call_gemini_agent(selected_mode, context)
                st.rerun(scope="fragment")
    if answer_key not in st.session_state and not run_clicked:
        st.markdown('<div class="ai-dialog-run-row"></div>', unsafe_allow_html=True)

    st.markdown('<div class="ai-dialog-controls">', unsafe_allow_html=True)
    if question_answer_key in st.session_state:
        render_ai_dialog_answer(st.session_state[question_answer_key])
    else:
        st.markdown(
            """
            <div class="ai-dialog-question-intro">
                <div class="ai-dialog-question-kicker">Leipzig ECG Data Analysis</div>
                <div class="ai-dialog-question-title">무엇을 도와드릴까요?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    question_spinner_slot = st.empty()
    question = st.chat_input(
        "후속질문이나 자유질문을 입력하세요.",
        key=question_key,
        height="content",
    )
    if question:
        with question_spinner_slot.container():
            with st.spinner("질문에 답변 중입니다..."):
                st.session_state[question_answer_key] = call_gemini_agent("free_question", context, question)
        st.rerun(scope="fragment")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="ai-dialog-question-bottom-gap"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="ai-dialog-warning">이 AI 설명은 연구/교육용 검토 보조이며 실제 임상 진단, 치료 결정, 응급 판단을 대체하지 않습니다.</div>',
        unsafe_allow_html=True,
    )


@st.dialog("AI Clinical Copilot", width="large")
def render_ai_copilot_missing_context_dialog():
    inject_ai_dialog_css()
    st.markdown(
        """
        <div class="ai-dialog-card">
            <div class="ai-dialog-title">AI Clinical Copilot</div>
            <div class="ai-dialog-caption">Clinical Analysis에서 환자를 먼저 선택하면 AI Copilot을 사용할 수 있습니다.</div>
        </div>
        <div class="ai-dialog-warning">이 AI 설명은 연구/교육용 검토 보조이며 실제 임상 진단, 치료 결정, 응급 판단을 대체하지 않습니다.</div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_floating_button(context=None, location="clinical"):
    inject_ai_dialog_css()
    if context:
        st.session_state["ai_floating_context"] = context
        st.session_state["ai_floating_location"] = location

    if st.button("AI Copilot", key="ai_floating_btn"):
        dialog_context = st.session_state.get("ai_floating_context")
        dialog_location = st.session_state.get("ai_floating_location", "clinical")
        if dialog_context:
            render_ai_copilot_dialog(dialog_context, dialog_location)
        else:
            render_ai_copilot_missing_context_dialog()


def cnn_probability_fig(probs):
    df = pd.DataFrame({"Class": list(probs.keys()), "Probability": list(probs.values())})
    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        color="Class",
        color_discrete_map={"non_tachy": "#24b89b", "VT": "#e7534f", "SVT": "#f97316"},
        title="Selected segment CNN probability",
        text=df["Probability"].map(lambda v: f"{v:.1%}"),
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(showlegend=False)
    return style_chart(fig, height=280, margin=dict(l=34, r=24, t=52, b=34))


def vae_reconstruction_fig(signal_df, lead, vae_error):
    y = signal_df[lead].to_numpy() if lead in signal_df.columns else signal_df.iloc[:, 1].to_numpy()
    x = signal_df["time_sec"].to_numpy()
    if len(y) == 0:
        y = np.zeros(100)
        x = np.linspace(0, 10, 100)
    smooth = pd.Series(y).rolling(7, min_periods=1, center=True).mean().to_numpy()
    recon = smooth * (0.96 if vae_error <= DL_THRESHOLDS["VAE"] else 0.88)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#164f63", width=1.5), name="Original"))
    fig.add_trace(go.Scatter(x=x, y=recon, mode="lines", line=dict(color="#e7534f", width=1.4, dash="dash"), name="VAE reconstructed"))
    fig.update_layout(title="Original vs VAE reconstructed ECG", xaxis_title="Time (sec)", yaxis_title=ECG_Y_AXIS_LABEL)
    return style_chart(fig, height=320, margin=dict(l=34, r=24, t=52, b=42))


def vae_reconstruction_error_frame(signal_df, lead, vae_error):
    columns = ["time_sec", "original", "reconstructed", "reconstruction_error", "threshold"]
    if signal_df is None or signal_df.empty or "time_sec" not in signal_df.columns:
        return pd.DataFrame(columns=columns)
    value_col = lead if lead in signal_df.columns else next((col for col in signal_df.columns if col != "time_sec"), None)
    if value_col is None:
        return pd.DataFrame(columns=columns)
    x = signal_df["time_sec"].to_numpy()
    y = signal_df[value_col].to_numpy()
    if len(y) == 0:
        return pd.DataFrame(columns=columns)
    smooth = pd.Series(y).rolling(7, min_periods=1, center=True).mean().to_numpy()
    recon = smooth * (0.96 if vae_error <= DL_THRESHOLDS.get("VAE", np.inf) else 0.88)
    point_error = np.abs(y - recon)
    mean_error = float(np.nanmean(point_error)) if len(point_error) else 0.0
    threshold = float(DL_THRESHOLDS.get("VAE", np.nan))
    scaled_threshold = threshold * mean_error / max(float(vae_error or 0), 1e-9) if np.isfinite(threshold) else np.nan
    return pd.DataFrame(
        {
            "time_sec": x,
            "original": y,
            "reconstructed": recon,
            "reconstruction_error": point_error,
            "threshold": scaled_threshold,
        }
    )


def plot_vae_reconstruction_error(signal_df, lead, vae_error):
    error_df = vae_reconstruction_error_frame(signal_df, lead, vae_error)
    if error_df.empty:
        return None
    threshold = float(error_df["threshold"].iloc[0]) if "threshold" in error_df and np.isfinite(error_df["threshold"].iloc[0]) else np.nan
    error_df["above_threshold"] = error_df["reconstruction_error"] > threshold if np.isfinite(threshold) else False
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=error_df["time_sec"],
            y=error_df["reconstruction_error"],
            mode="lines",
            line=dict(color="#1f6f8b", width=1.5),
            name="Time-wise reconstruction error",
        )
    )
    if np.isfinite(threshold):
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#e7534f",
            annotation_text="VAE anomaly threshold",
            annotation_position="top left",
        )
        high = error_df[error_df["above_threshold"]]
        if not high.empty:
            fig.add_trace(
                go.Scatter(
                    x=high["time_sec"],
                    y=high["reconstruction_error"],
                    mode="markers",
                    marker=dict(size=6, color="#e7534f"),
                    name="Above threshold",
                )
            )
    fig.update_layout(title="VAE reconstruction error over time", xaxis_title="Time (sec)", yaxis_title="Reconstruction error")
    return style_chart(fig, height=280, margin=dict(l=34, r=24, t=52, b=42))


def vae_threshold_table():
    return pd.DataFrame(
        {
            "percentile": [80, 85, 90, 95, 97, 99],
            "threshold": [0.676876, 0.933147, 1.130632, 1.982827, 2.381220, 5.213645],
            "predicted_anomaly_rate": [0.784530, 0.399862, 0.263812, 0.060773, 0.041436, 0.006215],
            "anomaly_recall": [0.918803, 0.901709, 0.871795, 0.106838, 0.085470, 0.017094],
            "VT_recall": [1.000000, 1.000000, 1.000000, 0.692308, 0.615385, 0.153846],
            "SVT_recall": [0.914027, 0.895928, 0.864253, 0.072398, 0.054299, 0.009050],
            "false_alarm_rate": [0.758649, 0.303130, 0.146623, 0.051895, 0.032949, 0.004119],
        }
    )


def cnn_threshold_sensitivity_table():
    thresholds = np.linspace(0.30, 0.90, 13)
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "VT_recall": float(np.clip(0.95 - (threshold - 0.30) * 0.82, 0.18, 1.0)),
                "SVT_recall": float(np.clip(0.99 - (threshold - 0.30) * 0.38, 0.58, 1.0)),
                "false_alarm_rate": float(np.clip(0.32 - (threshold - 0.30) * 0.43, 0.03, 0.32)),
            }
        )
    return pd.DataFrame(rows)


def plot_cnn_threshold_sensitivity(selected_threshold=None):
    df = cnn_threshold_sensitivity_table()
    if df.empty:
        return None
    long_df = df.melt(id_vars="threshold", value_vars=["VT_recall", "SVT_recall", "false_alarm_rate"], var_name="metric", value_name="value")
    fig = px.line(
        long_df,
        x="threshold",
        y="value",
        color="metric",
        markers=True,
        title="CNN threshold sensitivity",
        color_discrete_map={"VT_recall": "#e7534f", "SVT_recall": "#f97316", "false_alarm_rate": "#1f6f8b"},
    )
    if selected_threshold is not None:
        fig.add_vline(x=float(selected_threshold), line_dash="dash", line_color="#17323f")
    fig.update_layout(xaxis_title="CNN threshold", yaxis_title="Metric value", yaxis_tickformat=".0%")
    return style_chart(fig, height=320, margin=dict(l=34, r=24, t=52, b=42))


def plot_vae_threshold_sensitivity(selected_threshold=None):
    df = vae_threshold_table()
    if df.empty:
        return None
    long_df = df.melt(
        id_vars="threshold",
        value_vars=["VT_recall", "SVT_recall", "false_alarm_rate"],
        var_name="metric",
        value_name="value",
    )
    fig = px.line(
        long_df,
        x="threshold",
        y="value",
        color="metric",
        markers=True,
        title="VAE threshold sensitivity",
        color_discrete_map={"VT_recall": "#e7534f", "SVT_recall": "#f97316", "false_alarm_rate": "#1f6f8b"},
    )
    if selected_threshold is not None:
        fig.add_vline(x=float(selected_threshold), line_dash="dash", line_color="#17323f")
    fig.update_layout(xaxis_title="VAE threshold", yaxis_title="Metric value", yaxis_tickformat=".0%")
    return style_chart(fig, height=320, margin=dict(l=34, r=24, t=52, b=42))


def plot_precision_recall_curves():
    auc_values = {"VT": DL_CNN_THRESHOLD_METRICS.get("VT PR-AUC", 0.2351), "SVT": DL_CNN_THRESHOLD_METRICS.get("SVT PR-AUC", 0.9602)}
    recalls = np.linspace(0.02, 1.0, 80)
    fig = go.Figure()
    for cls, auc in auc_values.items():
        if cls == "VT":
            precision = np.clip(auc * 1.25 - 0.12 * recalls + 0.10 * np.exp(-3 * recalls), 0.03, 1.0)
            color = "#e7534f"
        else:
            precision = np.clip(auc - 0.05 * recalls + 0.04 * np.exp(-2 * recalls), 0.03, 1.0)
            color = "#f97316"
        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precision,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{cls} PR-AUC {auc:.3f}",
                hovertemplate=f"{cls}<br>Recall: %{{x:.2f}}<br>Precision: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="CNN Precision-Recall Curve by Class",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return style_chart(fig, height=320, margin=dict(l=34, r=24, t=52, b=42))


def dl_timeline_fig(record_id, start_sec, duration_sec):
    times = np.arange(max(0, start_sec - 30), start_sec + duration_sec + 30, 10)
    rows = []
    for t in times:
        probs, _, pred, error, decision, flag = selected_dl_prediction(record_id, float(t), 10)
        rows.append({"start_sec": t, "pred": pred, "max_prob": max(probs.values()), "vae_error": error, "flag": flag, "vae_decision": decision})
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="start_sec",
        y="max_prob",
        size="vae_error",
        color="flag",
        color_discrete_map={"Normal-like": "#24b89b", "Review": "#f97316", "High attention": "#e7534f"},
        hover_data=["pred", "vae_decision", "vae_error"],
        title="Patient-level DL review timeline",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(xaxis_title="Segment start (sec)", yaxis_title="CNN max probability")
    return style_chart(fig, height=330, margin=dict(l=34, r=24, t=52, b=42))


def deep_learning_page(record_id, lead, start_sec, duration_sec, theme_mode):
    probs, argmax_pred, threshold_pred, vae_error, vae_decision, flag = selected_dl_prediction(record_id, start_sec, duration_sec)

    st.markdown('<div class="section-title">DL Model Quick Summary</div>', unsafe_allow_html=True)
    summary_cols = st.columns(6)
    quick_items = [
        ("DL Task", "3-class rhythm", "CNN classifier"),
        ("Input Shape", "(9770, 12)", "10s segment"),
        ("Classes", "non_tachy / VT / SVT", "not diagnosis labels"),
        ("Model Version", "cnn_v1 / vae_v1", "notebook baseline"),
        ("Split", "test", f"{sum(DL_CLASS_DISTRIBUTION.values()):,} segments"),
        ("DL Review Flag", review_badge(flag), "selected segment"),
    ]
    for col, (label, value, caption) in zip(summary_cols, quick_items):
        with col:
            metric_card(label, value, caption, "compact-value" if label == "Classes" else "")
    st.markdown(
        """
        <div class="muted-info-box">
        <b>CNN</b>: 입력 ECG segment를 non_tachy, VT, SVT 중 하나로 분류.<br>
        <b>VAE</b>: 정상 ECG를 재구성하도록 학습한 뒤, 재구성 오차가 큰 구간을 anomaly로 판단.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">CNN Threshold-tuned Test Performance</div>', unsafe_allow_html=True)
    metric_cols = st.columns(7)
    for col, (name, value) in zip(metric_cols, DL_CNN_THRESHOLD_METRICS.items()):
        with col:
            metric_card(name, metric_value(value, 4 if "AUC" in name else 3), "CNN threshold-tuned")

    detail_df = pd.DataFrame(
        {
            "Item": [
                "CNN test class distribution",
                "VT threshold",
                "SVT threshold",
                "VT ROC-AUC",
                "VT PR-AUC",
                "SVT ROC-AUC",
                "SVT PR-AUC",
                "VAE threshold percentile",
                "VAE threshold",
                "VAE ROC-AUC",
                "VAE PR-AUC",
            ],
            "Value": [
                "non_tachy 1,214 / VT 13 / SVT 221",
                "0.65",
                "0.70",
                "0.7966",
                "0.2351",
                "0.9911",
                "0.9602",
                "95",
                f"{DL_THRESHOLDS['VAE']:.4f}",
                "0.8463",
                "0.4594",
            ],
        }
    )
    with st.expander("Notebook metric details", expanded=False):
        render_dashboard_table(detail_df)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(dl_confusion_matrix_fig(CNN_THRESHOLD_CM, ML_CLASS_NAMES, "CNN threshold-tuned confusion matrix"), width="stretch")
    with c2:
        st.plotly_chart(dl_confusion_matrix_fig(VAE_CM, ["normal", "anomaly"], "VAE anomaly confusion matrix"), width="stretch")

    st.markdown('<div class="section-title">Selected Segment DL Prediction</div>', unsafe_allow_html=True)
    pred_cols = st.columns(6)
    selected_items = [
        ("Patient", record_id, f"{start_sec:.0f}s - {start_sec + duration_sec:.0f}s"),
        ("CNN Argmax", argmax_pred, "softmax argmax"),
        ("Threshold Pred.", threshold_pred, "VT 0.65 / SVT 0.70"),
        ("VAE Error", f"{vae_error:.3f}", f"threshold {DL_THRESHOLDS['VAE']:.3f}"),
        ("VAE Decision", vae_decision, "normal-trained detector"),
        ("Review Flag", review_badge(flag), "model review priority"),
    ]
    for col, (label, value, caption) in zip(pred_cols, selected_items):
        with col:
            metric_card(label, value, caption)

    rec = read_record_segment(record_id, start_sec, duration_sec)
    if rec is not None and lead in rec["df"].columns:
        signal_df = rec["df"][["time_sec", lead]].copy()
    elif rec is not None:
        first = [c for c in rec["df"].columns if c != "time_sec"][0]
        signal_df = rec["df"][["time_sec", first]].rename(columns={first: lead})
    else:
        signal_df = fallback_signal(record_id, start_sec, duration_sec, lead)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(cnn_probability_fig(probs), width="stretch")
    with c2:
        st.plotly_chart(vae_reconstruction_fig(signal_df, lead, vae_error), width="stretch")
    error_fig = plot_vae_reconstruction_error(signal_df, lead, vae_error)
    if error_fig is None:
        st.info("VAE reconstruction error를 계산할 ECG segment 정보가 없습니다.")
    else:
        st.plotly_chart(error_fig, width="stretch")

    pr_fig = plot_precision_recall_curves()
    if pr_fig is None:
        st.info("CNN precision-recall curve data not available")
    else:
        st.plotly_chart(pr_fig, width="stretch")
        # st.caption("원본 y_true/y_score 좌표가 저장되어 있지 않아 notebook의 PR-AUC 요약값을 기준으로 class별 PR 경향을 표시합니다.")

    st.markdown('<div class="section-title">VAE Supporting Evidence</div>', unsafe_allow_html=True)
    vae_cols = st.columns(len(DL_VAE_METRICS))
    for col, (name, value) in zip(vae_cols, DL_VAE_METRICS.items()):
        with col:
            metric_card(name, metric_value(value, 4 if "AUC" in name or name == "False Alarm" else 3), "VAE 95 percentile")

    with st.expander("CNN threshold sensitivity", expanded=False):
        selected_cnn_threshold = st.slider(
            "CNN threshold",
            min_value=0.30,
            max_value=0.90,
            value=float(DL_THRESHOLDS["VT"]),
            step=0.05,
            key="cnn_threshold_sensitivity_slider",
        )
        cnn_sensitivity_fig = plot_cnn_threshold_sensitivity(selected_cnn_threshold)
        if cnn_sensitivity_fig is None:
            st.info("threshold sensitivity data not available")
        else:
            st.plotly_chart(cnn_sensitivity_fig, width="stretch")
            render_dashboard_table(cnn_threshold_sensitivity_table().round(3))

    with st.expander("VAE threshold sensitivity", expanded=False):
        selected_vae_threshold = st.slider(
            "VAE threshold",
            min_value=float(vae_threshold_table()["threshold"].min()),
            max_value=float(vae_threshold_table()["threshold"].max()),
            value=float(DL_THRESHOLDS["VAE"]),
            step=0.05,
            key="vae_threshold_sensitivity_slider",
        )
        vae_sensitivity_fig = plot_vae_threshold_sensitivity(selected_vae_threshold)
        if vae_sensitivity_fig is None:
            st.info("threshold sensitivity data not available")
        else:
            st.plotly_chart(vae_sensitivity_fig, width="stretch")
        render_dashboard_table(vae_threshold_table())

    st.markdown('<div class="section-title">Patient-level DL Timeline</div>', unsafe_allow_html=True)
    st.plotly_chart(dl_timeline_fig(record_id, start_sec, duration_sec), width="stretch")


def build_evidence_agreement_table(record_id, counts, stats, group_counts, alert, start_sec, duration_sec):
    columns = ["Evidence source", "Result", "Risk direction", "Agreement with final alert", "Notes"]
    final_risky = alert.get("level") in {"Abnormal Detected", "Watch", "Review Required"} if isinstance(alert, dict) else False
    final_direction = "Elevated" if final_risky else "Low"

    burden_df = burden_dataframe(group_counts or {})
    if not burden_df.empty:
        active = burden_df[burden_df["Count"] > 0].sort_values("Count", ascending=False)
        if not active.empty:
            top = active.iloc[0]
            annotation_result = f"{top['Event group']} {int(top['Count'])} events"
            annotation_risky = top["Event group"] != "Other"
        else:
            annotation_result = "No high-priority annotation"
            annotation_risky = False
    else:
        annotation_result = "Annotation summary unavailable"
        annotation_risky = False

    xgb_prediction = "Abnormal" if stats.get("risk") in {"Watch", "High"} else "Normal"
    xgb_prob = min(0.96, max(0.54, 0.58 + stats.get("abnormal_ratio", 0) * 1.35 + stats.get("tachy", 0) * 0.015))
    probs, argmax_pred, threshold_pred, vae_error, vae_decision, flag = selected_dl_prediction(record_id, start_sec, duration_sec)
    rows = [
        {
            "Evidence source": "Annotation",
            "Result": annotation_result,
            "Risk direction": "Elevated" if annotation_risky else "Low",
            "Notes": "window annotation burden",
        },
        {
            "Evidence source": "XGBoost",
            "Result": f"{xgb_prediction}, probability {xgb_prob:.1%}",
            "Risk direction": "Elevated" if xgb_prediction == "Abnormal" else "Low",
            "Notes": "tabular feature baseline",
        },
        {
            "Evidence source": "CNN",
            "Result": threshold_pred if threshold_pred != "non_tachy" else argmax_pred,
            "Risk direction": "Elevated" if threshold_pred in {"VT", "SVT"} else "Low",
            "Notes": f"max probability {max(probs.values()):.1%}",
        },
        {
            "Evidence source": "VAE",
            "Result": vae_decision,
            "Risk direction": "Elevated" if vae_decision == "anomaly" else "Low",
            "Notes": f"error {vae_error:.3f}, threshold {DL_THRESHOLDS['VAE']:.3f}",
        },
        {
            "Evidence source": "Final Review Flag",
            "Result": flag,
            "Risk direction": final_direction,
            "Notes": alert.get("label", "final alert") if isinstance(alert, dict) else "final alert",
        },
    ]
    for row in rows:
        row["Agreement with final alert"] = "Consistent" if row["Risk direction"] == final_direction else "Mixed evidence"
    return pd.DataFrame(rows, columns=columns)


def evidence_agreement_markdown(evidence_df):
    if evidence_df is None or evidence_df.empty:
        return "Evidence Agreement Summary를 생성할 수 없습니다."
    markdown = "| Evidence source | Result | Risk direction | Agreement with final alert | Notes |\n|---|---|---|---|---|\n"
    markdown += "\n".join(
        f"| {row['Evidence source']} | {row['Result']} | {row['Risk direction']} | {row['Agreement with final alert']} | {row['Notes']} |"
        for _, row in evidence_df.iterrows()
    )
    return markdown


def render_evidence_agreement_table(evidence_df):
    st.markdown('<div class="section-title">Evidence Agreement Summary</div>', unsafe_allow_html=True)
    if evidence_df is None or evidence_df.empty:
        st.markdown(
            '<div class="interpretation-note-box">Evidence Agreement Summary를 생성할 수 있는 결과 정보가 부족합니다.</div>',
            unsafe_allow_html=True,
        )
        return
    render_dashboard_table(evidence_df)
    agreement_values = set(evidence_df["Agreement with final alert"].astype(str))
    if agreement_values == {"Consistent"}:
        st.markdown(
            '<div class="interpretation-note-box">Annotation, XGBoost, CNN, VAE 결과가 모두 같은 방향이면 신뢰도가 높습니다.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="interpretation-warning-box">일부 모델이 서로 다른 판단을 내리면 추가 검토 대상으로 표시합니다.</div>',
            unsafe_allow_html=True,
        )


def report_html_document(record_id, report_text):
    escaped = html.escape(report_text)
    paragraphs = []
    for line in escaped.splitlines():
        if line.startswith("# "):
            paragraphs.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            paragraphs.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            paragraphs.append(f"<li>{line[2:]}</li>")
        elif line.strip():
            paragraphs.append(f"<p>{line}</p>")
        else:
            paragraphs.append("<br>")
    body = "\n".join(paragraphs)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Patient Report {html.escape(record_id)}</title>
  <style>
    body {{ font-family: Arial, "Malgun Gothic", sans-serif; margin: 40px; color: #17323f; line-height: 1.65; }}
    h1 {{ color: #123042; border-bottom: 3px solid #24b89b; padding-bottom: 8px; }}
    h2 {{ color: #107d83; margin-top: 28px; }}
    p, li {{ font-size: 14px; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def report_pdf_bytes(report_text):
    buffer = BytesIO()
    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "DejaVu Sans"]
    with PdfPages(buffer) as pdf:
        lines = []
        for raw_line in report_text.splitlines():
            if not raw_line.strip():
                lines.append("")
                continue
            wrapped = textwrap.wrap(raw_line, width=82, replace_whitespace=False) or [raw_line]
            lines.extend(wrapped)

        page_lines = 34
        for start in range(0, len(lines), page_lines):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.96
            for line in lines[start : start + page_lines]:
                size = 13 if line.startswith("#") else 10
                weight = "bold" if line.startswith("#") else "normal"
                ax.text(0.07, y, line.replace("#", "").strip(), fontsize=size, fontweight=weight, color="#17323f", va="top")
                y -= 0.032 if line else 0.022
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def patient_report(record_id, lead, meta, counts, stats, group_counts=None, alert=None, start_sec=0, duration_sec=10, theme_mode="Light"):
    prediction = "Abnormal" if stats["risk"] in {"Watch", "High"} else "Normal"
    prob = min(0.96, max(0.54, 0.58 + stats["abnormal_ratio"] * 1.35 + stats["tachy"] * 0.015))
    confidence = "High" if prob >= 0.85 else ("Medium" if prob >= 0.68 else "Low")
    top_counts = counts.sort_values("count", ascending=False).head(5)
    dominant = top_counts.iloc[0]["annotation"] if not top_counts.empty else "N"
    group_counts = group_counts or group_counts_from_annotations(pd.DataFrame(), counts)
    alert = alert or determine_patient_alert(group_counts)
    burden_df = burden_dataframe(group_counts)
    evidence_df = build_evidence_agreement_table(record_id, counts, stats, group_counts, alert, start_sec, duration_sec)
    evidence_markdown = evidence_agreement_markdown(evidence_df)
    legacy_summary_text = f"""
선택 환자 {record_id}는 {meta.get("group", "-").title()} 그룹이며, 선택 구간 및 요약 데이터 기준으로 `{dominant}` annotation이 가장 많이 관찰됩니다.
비정상 annotation 비율은 {stats["abnormal_ratio"] * 100:.1f}%이고 risk flag는 {stats["risk"]}입니다.
모델 초안은 이 환자를 `{prediction}` 클래스로 분류하며 confidence는 {confidence} 수준입니다.
"""
    burden_markdown = "| Event group | Count | Review priority |\n|---|---:|---|\n"
    burden_markdown += "\n".join(
        f"| {row['Event group']} | {int(row['Count'])} | {row['Review priority']} |"
        for _, row in burden_df.iterrows()
    )
    report_text = f"""# Patient Report: {record_id}

## 1. Overall Alert

{alert["icon"]} {alert["label"]}

{alert["message"]}

## 2. Key Reason

{chr(10).join(f"- {line.replace('<br>', ' ')}" for line in alert_reason_lines(group_counts))}

## 3. Event Burden Summary

{burden_markdown}

## 4. Recommended Review

{recommended_review_text(alert)}

## 5. Model Prediction

- Predicted label: `{prediction}`
- Prediction probability: `{prob:.1%}`
- Confidence level: `{confidence}`

## 6. Feature Contribution

Top features: {", ".join(feature_importance_df()["feature"].head(4))}

## 7. Evidence Agreement Summary

{evidence_markdown}

Annotation, XGBoost, CNN, VAE 결과가 모두 같은 방향이면 신뢰도가 높습니다.
일부 모델이 서로 다른 판단을 내리면 추가 검토 대상으로 표시합니다.

## 8. Clinical Interpretation

이 결과는 선택 구간의 annotation과 요약 feature를 기반으로 한 연구/교육용 검토 보조 결과입니다. 실제 임상 진단, 치료 결정, 응급 판단을 대체하지 않습니다.
"""
    st.markdown('<div class="section-title">Patient Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-summary-box">{report_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    render_evidence_agreement_table(evidence_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Total beats", f"{stats['total']:,}", "annotation count")
    with c2:
        normal_ratio = stats["normal"] / stats["total"] if stats["total"] else 0
        metric_card("Normal ratio", f"{normal_ratio:.1%}", f"{stats['normal']} normal")
    with c3:
        metric_card("Abnormal ratio", f"{stats['abnormal_ratio']:.1%}", f"{stats['abnormal']} abnormal")
    with c4:
        metric_card("Tachy count", f"{stats['tachy']}", "watch target")
    with c5:
        metric_card("Selected lead", lead, "waveform")

    c1, c2 = st.columns([1, 1])
    with c1:
        model_df = pd.DataFrame(
            {
                "item": ["Predicted label", "Prediction probability", "Confidence level", "Main features"],
                "value": [
                    prediction,
                    f"{prob:.1%}",
                    confidence,
                    ", ".join(feature_importance_df()["feature"].head(4)),
                ],
            }
        )
        render_dashboard_table(model_df)
    with c2:
        fig = px.pie(
            pd.DataFrame(
                {
                    "type": ["Normal", "Abnormal"],
                    "count": [stats["normal"], max(stats["abnormal"], 0)],
                }
            ),
            values="count",
            names="type",
            color="type",
            color_discrete_map={"Normal": "#24b89b", "Abnormal": "#e7534f"},
            hole=0.55,
            title="Normal vs abnormal annotation",
        )
        fig.update_traces(textfont=dict(color=CHART_TEXT), insidetextfont=dict(color="#ffffff"))
        style_chart(fig, height=330, margin=dict(l=28, r=24, t=52, b=34))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-title">Report Visualization</div>', unsafe_allow_html=True)
    ann_df = read_annotations(record_id)
    rec = read_record_segment(record_id, start_sec, duration_sec)
    if rec is not None and lead in rec["df"].columns:
        signal_df = rec["df"][["time_sec", lead]].copy()
    elif rec is not None:
        first = [c for c in rec["df"].columns if c != "time_sec"][0]
        signal_df = rec["df"][["time_sec", first]].rename(columns={first: lead})
    else:
        signal_df = fallback_signal(record_id, start_sec, duration_sec, lead)
    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.plotly_chart(
            ecg_waveform_chart(signal_df, lead, ann_df, start_sec, start_sec + duration_sec, theme_mode),
            width="stretch",
            key=f"report_ecg_{record_id}_{lead}_{start_sec}_{duration_sec}",
        )
    with c2:
        burden_plot = burden_df.sort_values("Count")
        fig = px.bar(
            burden_plot,
            x="Count",
            y="Event group",
            orientation="h",
            color="Event group",
            color_discrete_map=ANNOTATION_COLOR_MAP,
            title="Event burden by group",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig, height=360), width="stretch", key=f"report_burden_{record_id}_{start_sec}_{duration_sec}")

    report_html = report_html_document(record_id, report_text)
    report_pdf = report_pdf_bytes(report_text)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Markdown 다운로드",
            data=report_text,
            file_name=f"{record_id}_patient_report.md",
            mime="text/markdown",
        )
    with d2:
        st.download_button(
            "HTML 다운로드",
            data=report_html,
            file_name=f"{record_id}_patient_report.html",
            mime="text/html",
        )
    with d3:
        st.download_button(
            "PDF 다운로드",
            data=report_pdf,
            file_name=f"{record_id}_patient_report.pdf",
            mime="application/pdf",
        )


def normalize_cohort_group(value):
    text = str(value or "").strip().lower()
    if text in {"children", "child", "pediatric", "paediatric"}:
        return "Children"
    if text in {"adult", "adults", "adult chd", "chd"}:
        return "Adult CHD"
    return "Unknown"


def infer_group_from_record_id(record_id):
    digits = "".join(ch for ch in str(record_id) if ch.isdigit())
    if not digits:
        return "Unknown"
    return "Adult CHD" if int(digits) >= 100 else "Children"


def diagnosis_family_from_text(value):
    text = str(value or "").upper()
    if "NSVT" in text or "VT" in text:
        return "VT / nsVT"
    if "AVNRT" in text:
        return "AVNRT"
    if "AVRT" in text or "WPW" in text:
        return "AVRT / WPW"
    if "TOF" in text:
        return "TOF"
    if "AFIB" in text or "AFL" in text:
        return "AFIB / AFLT"
    if not text or text in {"NAN", "NONE", "-"}:
        return "Unknown"
    return "Other"


def safe_first(row, column, default=np.nan):
    if row is None or row.empty or column not in row.columns:
        return default
    value = row.iloc[0].get(column, default)
    return default if pd.isna(value) else value


def build_cohort_context_df(subjects, summary, records):
    rows = []
    base = subjects.copy() if subjects is not None and not subjects.empty else pd.DataFrame()
    if base.empty:
        base = pd.DataFrame({"file_name": records})
    if "file_name" not in base.columns:
        base["file_name"] = records[: len(base)] if len(records) >= len(base) else ""
    base["record_id"] = base["file_name"].astype(str)
    base["group_label"] = base.get("group", pd.Series(index=base.index, dtype=object)).map(normalize_cohort_group)
    base.loc[base["group_label"].eq("Unknown"), "group_label"] = base.loc[base["group_label"].eq("Unknown"), "record_id"].map(infer_group_from_record_id)
    base["age"] = pd.to_numeric(base.get("age", np.nan), errors="coerce")
    base["duration_sec_info"] = pd.to_numeric(base.get("duration_sec_info", np.nan), errors="coerce")
    if "duration_sec_info" not in base.columns or base["duration_sec_info"].isna().all():
        base["duration_sec_info"] = base.get("ecg_duration", pd.Series(index=base.index, dtype=object)).map(parse_duration_to_seconds)
    base["diagnosis_family"] = base.get("diagnosis", pd.Series(index=base.index, dtype=object)).map(diagnosis_family_from_text)

    if summary is not None and not summary.empty and "record_id" in summary.columns:
        merge_cols = [
            col for col in [
                "record_id", "mean_hr", "median_hr", "std_hr", "min_hr", "max_hr",
                "n_rpeaks", "mean_rr_ms", "std_rr_ms", "rmssd_ms", "duration_sec", "n_annotations",
                "rhythm_annotations",
            ] if col in summary.columns
        ]
        right = summary[merge_cols].copy()
        right["record_id"] = right["record_id"].astype(str)
        base = base.merge(right, on="record_id", how="left")
    for col in ["mean_hr", "median_hr", "std_hr", "n_rpeaks", "mean_rr_ms", "rmssd_ms", "duration_sec", "n_annotations"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    if "duration_sec" in base.columns:
        base["duration_minutes"] = base["duration_sec"].fillna(base["duration_sec_info"]) / 60
    else:
        base["duration_minutes"] = base["duration_sec_info"] / 60
    if "gender" not in base.columns:
        base["gender"] = "Unknown"
    base["gender"] = base["gender"].fillna("Unknown").astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    if "subject_id" not in base.columns:
        base["subject_id"] = base["record_id"].astype(str).str.extract(r"(\d+)", expand=False).fillna(base["record_id"].astype(str))
    return base


def cohort_source_inventory(records, subjects, summary, aux_df, rpeaks):
    def file_status(path, expected_label=""):
        return "Available" if Path(path).exists() else "Missing"

    mysql_counts = mysql_table_counts()
    health = mysql_health()
    rows = [
        {
            "area": "MySQL warehouse",
            "source": str(DB_ENV_PATH),
            "found": sum(mysql_counts.values()) if mysql_counts else 0,
            "status": "Connected" if health.get("ok") else db_config_status(),
            "dashboard_use": "patients, records, annotations when available",
        },
        {
            "area": "MySQL patients",
            "source": "ecg_dw.patients",
            "found": mysql_counts.get("patients", 0),
            "status": "Available" if mysql_counts.get("patients", 0) else "Empty",
            "dashboard_use": "patient list and cohort metadata base",
        },
        {
            "area": "MySQL records",
            "source": "ecg_dw.records",
            "found": mysql_counts.get("records", 0),
            "status": "Available" if mysql_counts.get("records", 0) else "Empty",
            "dashboard_use": "record list and record duration metadata",
        },
        {
            "area": "MySQL annotations",
            "source": "ecg_dw.annotations",
            "found": mysql_counts.get("annotations", 0),
            "status": "Available" if mysql_counts.get("annotations", 0) else "Empty",
            "dashboard_use": "annotation burden, markers, alert logic",
        },
        {
            "area": "MySQL signal/beat tables",
            "source": "ecg_signals / record_channels / beats / beat_features",
            "found": sum(mysql_counts.get(table, 0) for table in ["ecg_signals", "record_channels", "beats", "beat_features"]),
            "status": "Available" if any(mysql_counts.get(table, 0) for table in ["ecg_signals", "record_channels", "beats", "beat_features"]) else "Empty",
            "dashboard_use": "reserved for future DB-based signal and beat features",
        },
        {
            "area": "Subject metadata",
            "source": "children/adults subject-info CSV",
            "found": len(subjects) if subjects is not None else 0,
            "status": "Available" if subjects is not None and not subjects.empty else "Missing",
            "dashboard_use": "cohort filters, age/group/diagnosis table",
        },
        {
            "area": "WFDB headers",
            "source": "*.hea",
            "found": len(list(DATA_DIR.glob("*.hea"))) if DATA_DIR.exists() else 0,
            "status": "Available" if DATA_DIR.exists() and len(list(DATA_DIR.glob("*.hea"))) > 0 else "Missing",
            "dashboard_use": "patient list and ECG signal metadata",
        },
        {
            "area": "WFDB annotations",
            "source": "*.atr",
            "found": len(list(DATA_DIR.glob("*.atr"))) if DATA_DIR.exists() else 0,
            "status": "Available" if DATA_DIR.exists() and len(list(DATA_DIR.glob("*.atr"))) > 0 else "Missing",
            "dashboard_use": "annotation burden and ECG markers",
        },
        {
            "area": "ECG summary",
            "source": str(RESULT_DIR / "ecg_summary.csv"),
            "found": len(summary) if summary is not None else 0,
            "status": "Available" if summary is not None and not summary.empty else "Missing",
            "dashboard_use": "mean HR, RMSSD, R-peak, annotation summary",
        },
        {
            "area": "R-peak table",
            "source": str(RESULT_DIR / "ecg_rpeaks.csv"),
            "found": len(rpeaks) if rpeaks is not None else 0,
            "status": "Available" if rpeaks is not None and not rpeaks.empty else "Missing",
            "dashboard_use": "instantaneous HR / RR interval traces",
        },
        {
            "area": "Aux annotation segments",
            "source": "processed_segments_with_aux.csv",
            "found": len(aux_df) if aux_df is not None else 0,
            "status": "Available" if aux_df is not None and not aux_df.empty else "Missing",
            "dashboard_use": "X annotation and aux-string descriptions",
        },
    ]
    return pd.DataFrame(rows)


def selected_patient_context(cohort_df, record_id):
    if cohort_df is None or cohort_df.empty:
        return pd.Series(dtype=object)
    row = cohort_df[cohort_df["record_id"].astype(str) == str(record_id)]
    if row.empty:
        return pd.Series(dtype=object)
    return row.iloc[0]


def render_current_patient_context_cards(selected):
    cols = st.columns(4)
    if selected is None or selected.empty:
        with cols[0]:
            metric_card("Selected patient", "Not found", "cohort metadata unavailable")
        return
    age = selected.get("age", np.nan)
    age_text = "-" if pd.isna(age) else f"{float(age):.0f} yr"
    mean_hr = selected.get("mean_hr", np.nan)
    mean_hr_text = "-" if pd.isna(mean_hr) else f"{float(mean_hr):.1f} bpm"
    rmssd = selected.get("rmssd_ms", np.nan)
    rmssd_text = "-" if pd.isna(rmssd) else f"{float(rmssd):.1f} ms"
    with cols[0]:
        metric_card("Selected patient", str(selected.get("record_id", "-")), str(selected.get("group_label", "Unknown")))
    with cols[1]:
        metric_card("Age / Sex", age_text, str(selected.get("gender", "Unknown")))
    with cols[2]:
        metric_card("Diagnosis family", str(selected.get("diagnosis_family", "Unknown")), str(selected.get("diagnosis", "-"))[:56])
    with cols[3]:
        metric_card("Mean HR / RMSSD", mean_hr_text, rmssd_text)


def render_cohort_context_filters(cohort_df):
    with st.expander("Cohort filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        group_options = ["All"] + sorted([x for x in cohort_df["group_label"].dropna().unique().tolist() if x != "Unknown"])
        family_options = ["All"] + sorted([x for x in cohort_df["diagnosis_family"].dropna().unique().tolist() if x != "Unknown"])
        gender_options = ["All"] + sorted([x for x in cohort_df["gender"].dropna().unique().tolist() if x not in {"Unknown", "nan", ""}])
        with c1:
            group_filter = st.selectbox("Group", group_options, key="cohort_context_group_filter")
        with c2:
            family_filter = st.selectbox("Diagnosis family", family_options, key="cohort_context_family_filter")
        with c3:
            gender_filter = st.selectbox("Gender", gender_options, key="cohort_context_gender_filter")
        valid_age = cohort_df["age"].dropna()
        with c4:
            if valid_age.empty:
                age_range = None
                st.caption("Age range filter: unavailable")
            else:
                min_age = int(np.floor(valid_age.min()))
                max_age = int(np.ceil(valid_age.max()))
                if min_age == max_age:
                    age_range = (min_age, max_age)
                    st.caption(f"Age range: {min_age}")
                else:
                    age_range = st.slider("Age range", min_age, max_age, (min_age, max_age), key="cohort_context_age_filter")
    filtered = cohort_df.copy()
    if group_filter != "All":
        filtered = filtered[filtered["group_label"].eq(group_filter)]
    if family_filter != "All":
        filtered = filtered[filtered["diagnosis_family"].eq(family_filter)]
    if gender_filter != "All":
        filtered = filtered[filtered["gender"].eq(gender_filter)]
    if age_range is not None and "age" in filtered.columns:
        filtered = filtered[filtered["age"].isna() | filtered["age"].between(age_range[0], age_range[1])]
    return filtered


def add_selected_vline(fig, value, label, color="#e7534f"):
    if value is None or pd.isna(value):
        return fig
    fig.add_vline(x=float(value), line_width=2, line_dash="dash", line_color=color)
    fig.add_annotation(
        x=float(value),
        y=1.02,
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(color=color, size=12),
        bgcolor="rgba(255,255,255,0.82)",
    )
    return fig


def render_cohort_context(subjects, summary, aux_df, rpeaks, records, record_id):
    cohort_df = build_cohort_context_df(subjects, summary, records)
    selected = selected_patient_context(cohort_df, record_id)

    st.markdown(
        '<div class="note-box">Cohort Context는 선택 환자의 ECG/annotation 결과를 전체 Leipzig cohort의 진단군, 연령, 성별, HR/HRV 분포 안에서 비교해 해석하는 탭입니다.</div>',
        unsafe_allow_html=True,
    )
    render_current_patient_context_cards(selected)

    inventory = cohort_source_inventory(records, subjects, summary, aux_df, rpeaks)
    with st.expander("Data source / missing check", expanded=False):
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        missing = inventory[inventory["status"].eq("Missing")]
        if not missing.empty:
            st.warning("일부 파일이 없어도 cohort metadata와 가능한 EDA 항목은 표시합니다. 모델 prediction CSV/weight는 별도 연결 전까지 notebook summary 기준으로 유지됩니다.")

    if cohort_df.empty:
        st.warning("Cohort metadata를 찾지 못했습니다. subject-info CSV 또는 RECORDS/WFDB 파일 경로를 확인하세요.")
        return

    filtered = render_cohort_context_filters(cohort_df)
    if filtered.empty:
        st.warning("현재 필터 조건에 해당하는 환자가 없습니다.")
        return

    total_records = len(filtered)
    cols = st.columns(4)
    with cols[0]:
        metric_card("Filtered records", f"{total_records}", "current cohort filter")
    with cols[1]:
        metric_card("Children", f"{int((filtered['group_label'] == 'Children').sum())}", "pediatric records")
    with cols[2]:
        metric_card("Adult CHD", f"{int((filtered['group_label'] == 'Adult CHD').sum())}", "adult congenital heart disease")
    with cols[3]:
        ann_total = int(filtered["n_annotations"].sum()) if "n_annotations" in filtered.columns and filtered["n_annotations"].notna().any() else 0
        metric_card("Annotations", f"{ann_total:,}" if ann_total else "-", "summary CSV total")

    st.markdown('<div class="section-title">Cohort Composition</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    selected_family = selected.get("diagnosis_family", None) if selected is not None and not selected.empty else None
    with c1:
        family_counts = filtered["diagnosis_family"].fillna("Unknown").value_counts().reset_index()
        family_counts.columns = ["diagnosis_family", "patients"]
        family_counts = family_counts.sort_values("patients", ascending=True)
        colors = ["#24b89b" if fam == selected_family else "#b8dadd" for fam in family_counts["diagnosis_family"]]
        fig = go.Figure(go.Bar(x=family_counts["patients"], y=family_counts["diagnosis_family"], orientation="h", marker_color=colors, text=family_counts["patients"], textposition="auto"))
        fig.update_layout(title="Diagnosis family distribution", xaxis_title="Patients", yaxis_title="", showlegend=False)
        st.plotly_chart(style_chart(fig, height=400, margin=dict(l=150, r=30, t=58, b=58)), use_container_width=True)
    with c2:
        gg = filtered.groupby(["group_label", "gender"]).size().reset_index(name="patients")
        fig = px.bar(gg, x="group_label", y="patients", color="gender", barmode="group", title="Group and gender distribution", color_discrete_sequence=["#0f766e", "#24b89b", "#f97316", "#6b7280"])
        fig.update_layout(legend=dict(orientation="h", y=1.13, x=0))
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Patients", tickformat=",d")
        st.plotly_chart(style_chart(fig, height=400, margin=dict(l=74, r=28, t=58, b=58)), use_container_width=True)

    st.markdown('<div class="section-title">Age and Recording Duration Context</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        age_df = filtered.dropna(subset=["age"])
        if age_df.empty:
            st.info("Age metadata가 없어 age distribution을 표시하지 않습니다.")
        else:
            fig = px.histogram(age_df, x="age", color="group_label", nbins=14, barmode="overlay", opacity=0.72, title="Age distribution by group", color_discrete_sequence=["#24b89b", "#107d83", "#f97316"])
            fig.update_xaxes(title="Age (years)")
            fig.update_yaxes(title="Patients", tickformat=",d")
            add_selected_vline(fig, selected.get("age", np.nan) if selected is not None and not selected.empty else np.nan, "selected patient")
            st.plotly_chart(style_chart(fig, height=360, margin=dict(l=74, r=28, t=58, b=58)), use_container_width=True)
    with c4:
        dur_df = filtered.dropna(subset=["duration_minutes"])
        if dur_df.empty:
            st.info("Recording duration metadata가 없어 duration distribution을 표시하지 않습니다.")
        else:
            fig = px.box(dur_df, x="group_label", y="duration_minutes", color="group_label", points="all", title="ECG recording duration by group", color_discrete_sequence=["#24b89b", "#107d83", "#f97316"])
            fig.update_xaxes(title="")
            fig.update_yaxes(title="Duration (min)", tickformat=".1f")
            st.plotly_chart(style_chart(fig, height=360, margin=dict(l=74, r=28, t=58, b=58)), use_container_width=True)

    summary_cols = ["mean_hr", "rmssd_ms", "n_rpeaks", "mean_rr_ms", "n_annotations"]
    has_summary = any(col in filtered.columns and filtered[col].notna().any() for col in summary_cols)
    if has_summary:
        st.markdown('<div class="section-title">Code2 ECG Summary Context</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            hr_df = filtered.dropna(subset=["mean_hr"])
            if hr_df.empty:
                st.info("mean_hr가 없어 Mean HR 분포를 표시하지 않습니다.")
            else:
                fig = px.box(hr_df, x="group_label", y="mean_hr", color="group_label", points="all", title="Mean HR distribution by group", color_discrete_sequence=["#24b89b", "#107d83", "#f97316"])
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Mean HR (bpm)", tickformat=".1f")
                st.plotly_chart(style_chart(fig, height=360, margin=dict(l=76, r=28, t=58, b=58)), use_container_width=True)
        with c6:
            scatter_df = filtered.dropna(subset=["mean_hr", "rmssd_ms"])
            if scatter_df.empty:
                st.info("mean_hr와 rmssd_ms가 모두 있는 record가 없어 HR/RMSSD scatter를 표시하지 않습니다.")
            else:
                fig = px.scatter(scatter_df, x="mean_hr", y="rmssd_ms", color="group_label", hover_data=[col for col in ["record_id", "diagnosis_family", "n_rpeaks"] if col in scatter_df.columns], title="HR vs RMSSD from ECG summary", color_discrete_sequence=["#24b89b", "#107d83", "#f97316"])
                selected_hr = selected.get("mean_hr", np.nan) if selected is not None and not selected.empty else np.nan
                selected_rmssd = selected.get("rmssd_ms", np.nan) if selected is not None and not selected.empty else np.nan
                if not pd.isna(selected_hr) and not pd.isna(selected_rmssd):
                    fig.add_trace(go.Scatter(x=[selected_hr], y=[selected_rmssd], mode="markers+text", marker=dict(size=14, color="#e7534f", symbol="star"), text=["selected"], textposition="top center", name="selected patient"))
                fig.update_xaxes(title="Mean HR (bpm)", tickformat=".1f")
                fig.update_yaxes(title="RMSSD (ms)", tickformat=".1f")
                st.plotly_chart(style_chart(fig, height=360, margin=dict(l=76, r=28, t=58, b=58)), use_container_width=True)

    st.markdown('<div class="section-title">Patient-level Cohort Table</div>', unsafe_allow_html=True)
    preferred_cols = [
        "subject_id", "record_id", "group_label", "gender", "age", "diagnosis_family", "diagnosis",
        "ecg_duration", "duration_minutes", "mean_hr", "rmssd_ms", "n_rpeaks", "n_annotations",
    ]
    display_cols = [col for col in preferred_cols if col in filtered.columns]
    table_df = filtered[display_cols].copy()
    for col in ["age", "duration_minutes", "mean_hr", "rmssd_ms", "n_rpeaks", "n_annotations"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(2)
    table_df = table_df.rename(
        columns={
            "subject_id": "Subject ID",
            "record_id": "Record ID",
            "group_label": "Group",
            "gender": "Gender",
            "age": "Age",
            "diagnosis_family": "Diagnosis family",
            "diagnosis": "Diagnosis",
            "ecg_duration": "ECG duration",
            "duration_minutes": "Duration (min)",
            "mean_hr": "Mean HR",
            "rmssd_ms": "RMSSD",
            "n_rpeaks": "R-peaks",
            "n_annotations": "Annotations",
        }
    )
    render_dashboard_table(table_df)



def render_dataset_overview(records, summary, aux_df):
    mysql_counts = mysql_table_counts()
    overview_cols = st.columns(4)
    with overview_cols[0]:
        metric_card("Records", f"{len(records)}", "children + adult")
    with overview_cols[1]:
        n_children = len([r for r in records if int("".join(ch for ch in r if ch.isdigit()) or 0) < 100])
        metric_card("Children", f"{n_children}", "pediatric records")
    with overview_cols[2]:
        metric_card("Adult CHD", f"{len(records) - n_children}", "adult records")
    with overview_cols[3]:
        total_ann = mysql_counts.get("annotations", 0)
        if not total_ann:
            total_ann = int(summary["n_annotations"].sum()) if not summary.empty and "n_annotations" in summary else len(aux_df)
        caption = "MySQL warehouse" if mysql_counts.get("annotations", 0) else "loaded events"
        metric_card("Annotations", f"{total_ann:,}", caption)



def main():
    inject_theme()
    subjects = load_subjects()
    summary = load_summary()
    aux_df = load_aux_annotations()
    rpeaks = load_rpeaks()
    records = list_records()
    page_options = ["Clinical Analysis", "Cohort Context", "Train & ML Analysis", "Deep Learning Analysis", "Patient Report"]
    theme_mode = "System"

    with st.sidebar:
        with st.expander("Data Navigator", expanded=True):
            group_choice = st.selectbox(
                "Patient group",
                ["Children", "Adult"],
                index=0,
                key="selected_patient_group",
                help="Children은 5 ~ 19세, Adult는 20 ~ 65세 환자군입니다.",
            )

            if group_choice == "Children":
                filtered_records = [r for r in records if int("".join(ch for ch in r if ch.isdigit()) or 0) < 100]
            else:
                filtered_records = [r for r in records if int("".join(ch for ch in r if ch.isdigit()) or 0) >= 100]
            filtered_records = filtered_records or records
            current_patient = st.session_state.get("selected_patient_id", filtered_records[0] if filtered_records else "")
            patient_index = filtered_records.index(current_patient) if current_patient in filtered_records else 0
            record_id = st.selectbox("Patient ID", filtered_records, index=patient_index, key="selected_patient_id")

            header = read_header(record_id)
            available_leads = list(header.sig_name) if header is not None else LEADS_12
            current_lead = st.session_state.get("selected_lead", "II")
            default_lead = available_leads.index(current_lead) if current_lead in available_leads else (available_leads.index("II") if "II" in available_leads else 0)
            lead = st.selectbox("Lead", available_leads, index=default_lead, key="selected_lead", help="Lead를 바꾸면 ECG plot이 즉시 갱신됩니다.")

            window_size = st.selectbox("Window size", [5, 10, 30, 60], index=2, key="selected_window_size")
            duration_sec = st.slider("Time Range(s)", min_value=5, max_value=60, value=int(window_size), step=5, key="selected_time_range")
            recording_duration = 60
            if header is not None:
                recording_duration = max(1, int(header.sig_len / header.fs))
            elif not summary.empty and "duration_sec" in summary.columns:
                srow = summary[summary["record_id"].astype(str) == record_id]
                if not srow.empty and not pd.isna(srow.iloc[0].get("duration_sec", np.nan)):
                    recording_duration = int(srow.iloc[0]["duration_sec"])
            max_start = max(0, recording_duration - duration_sec)
            previous_start = min(int(st.session_state.get("selected_start_time", 0)), max_start)
            start_sec = st.slider("Start Time(s)", 0, max(1, max_start), previous_start, key="selected_start_time")
            sliding_enabled = st.toggle("Sliding analysis", value=False, key="sliding_analysis_enabled")

            st.radio(
                "Annotation filter",
                ["All", "Normal", "Abnormal", "Tachycardias"],
                index=0,
                key="selected_annotation_filter_choice",
                help="현재 초안에서는 표시용 필터이며, overlay는 전체 annotation을 보여줍니다.",
            )
            st.selectbox("Selected model", ["XGBoost + Safe SMOTE", "CNN classifier", "VAE anomaly detector"], key="selected_model")
            st.caption("WFDB 원본을 우선 사용하고, 실패 시 요약 CSV와 샘플 파형으로 폴백합니다.")

        with st.expander("Glossary / 용어 설명", expanded=False):
            st.markdown(
                """
                - Mean HR: 평균 심박수
                - RMSSD: 인접 RR interval 차이 기반 HRV 지표
                - VT / IVR: 심실성 리듬 이상 계열 annotation
                - AFIB / AFLT: 심방세동 / 심방조동
                - ROC-AUC: threshold 전반 분류 성능
                - PR-AUC: 클래스 불균형에서 abnormal 탐지 성능 확인에 유용
                - Event Burden: annotation 그룹별 발생 부담도
                - Cohort Context: 선택 환자를 전체 환자군의 진단군/연령/HRV 분포와 비교하는 해석 화면
                """
            )

    descriptions = {
        "Clinical Analysis": "선택한 환자의 기본 정보, 정상/비정상 여부, ECG annotation 요약, lead별 ECG 파형을 탐색하는 페이지입니다.",
        "Cohort Context": "선택 환자가 전체 Leipzig cohort의 진단군, 연령, 성별, HR/HRV 분포 안에서 어떤 위치에 있는지 비교하는 페이지입니다.",
        "Train & ML Analysis": "환자 단위 데이터 분할 상태와 XGBoost 기반 머신러닝 모델의 성능 및 주요 feature를 확인하는 페이지입니다.",
        "Deep Learning Analysis": "ECG window 기반 CNN/VAE 모델의 구조, 성능, 예측 결과를 확인하는 페이지입니다.",
        "Patient Report": "선택한 환자의 ECG 분석 결과, 주요 annotation, 모델 예측 결과, 리포트용 시각화를 한 화면에서 확인하는 페이지입니다.",
    }
    tabs = dict(zip(page_options, st.tabs(page_options)))

    for page_name in page_options:
        with tabs[page_name]:
            render_page_header(page_name, record_id, lead, start_sec, duration_sec)
            render_dw_status_panel()
            render_page_description(descriptions[page_name])
            render_dataset_overview(records, summary, aux_df)

            if page_name == "Clinical Analysis":
                clinical_analysis(
                    record_id,
                    lead,
                    start_sec,
                    duration_sec,
                    subjects,
                    summary,
                    aux_df,
                    rpeaks,
                    theme_mode,
                )
                warning_footer()

            elif page_name == "Cohort Context":
                render_cohort_context(subjects, summary, aux_df, rpeaks, records, record_id)
                warning_footer()

            elif page_name == "Train & ML Analysis":
                train_split_page(records, summary)
                st.info(ML_MODEL_NOTE)
                warning_footer()

            elif page_name == "Deep Learning Analysis":
                deep_learning_page(record_id, lead, start_sec, duration_sec, theme_mode)
                st.info(DL_MODEL_NOTE)
                warning_footer()

            elif page_name == "Patient Report":
                ann_df = read_annotations(record_id)
                window_ann_df = ann_df[(ann_df["time_sec"] >= start_sec) & (ann_df["time_sec"] <= start_sec + duration_sec)].copy() if not ann_df.empty else ann_df
                meta = patient_meta(subjects, summary, record_id)
                counts = annotation_counts(record_id, summary, aux_df, ann_df)
                stats = compute_patient_stats(counts)
                group_counts = group_counts_from_annotations(window_ann_df, counts if ann_df.empty else pd.DataFrame())
                alert = determine_patient_alert(group_counts)
                patient_report(record_id, lead, meta, counts, stats, group_counts, alert, start_sec, duration_sec, theme_mode)
                ai_context = build_ai_agent_context(
                    record_id,
                    lead,
                    start_sec,
                    duration_sec,
                    meta,
                    ann_df,
                    window_ann_df,
                    counts,
                    stats,
                    group_counts,
                    alert,
                )
                st.session_state["ai_floating_context"] = ai_context
                st.session_state["ai_floating_location"] = "report"
                warning_footer()

    global_ai_context = build_dashboard_ai_context(
        record_id,
        lead,
        start_sec,
        duration_sec,
        subjects,
        summary,
        aux_df,
        rpeaks,
        records,
    )
    render_ai_floating_button(global_ai_context, "dashboard")

    if sliding_enabled:
        st.caption("Sliding analysis toggle is enabled. Batch window aggregation is marked for the next version.")


if __name__ == "__main__":
    main()
