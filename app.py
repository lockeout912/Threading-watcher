import math
import base64
from pathlib import Path

import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# LOGO LOADER
# -----------------------------
def load_logo(path: str):
    try:
        p = Path(path)
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        pass
    return None

LOGO_B64 = load_logo("edge15logo.jpg")

# -----------------------------
# STYLE
# -----------------------------
st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(0,255,180,0.06), transparent 24%),
            radial-gradient(circle at 88% 12%, rgba(0,140,255,0.07), transparent 26%),
            linear-gradient(180deg, #040812 0%, #07111d 55%, #091521 100%);
        color: #f4f8ff;
    }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #09111d 0%, #0c1525 100%); }

    .header-wrap {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 0.20rem;
        padding: 6px 0 2px 0;
    }

    .header-logo {
        width: 78px;
        height: 78px;
        object-fit: contain;
        border-radius: 18px;
        padding: 6px;
        background:
            radial-gradient(circle at center, rgba(255,255,255,0.06), transparent 70%),
            linear-gradient(180deg, rgba(14,22,38,0.95), rgba(8,14,24,0.95));
        border: 1px solid rgba(113,150,225,0.20);
        box-shadow:
            0 12px 24px rgba(0,0,0,0.22),
            0 0 24px rgba(0,140,255,0.10),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .header-text-wrap {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 1000;
        color: #ffffff;
        margin-bottom: 0.10rem;
        line-height: 1.02;
        text-shadow: 0 0 22px rgba(0,180,255,0.12);
    }

    .subtle {
        color: #9cb1d3;
        font-size: 0.90rem;
        margin-bottom: 0.85rem;
    }

    .hero-card {
        background:
            radial-gradient(circle at top center, rgba(255,255,255,0.03), transparent 36%),
            linear-gradient(180deg, rgba(18,27,46,0.98), rgba(9,15,26,0.98));
        border: 1px solid rgba(101,139,215,0.22);
        border-radius: 20px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow:
            0 14px 24px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .command-bar {
        background: linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.18);
        border-radius: 16px;
        padding: 11px 12px;
        margin-bottom: 12px;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .command-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(110px, 1fr));
        gap: 8px;
    }

    .command-pill {
        border-radius: 14px;
        padding: 8px 10px;
        background: linear-gradient(180deg, rgba(21,31,52,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.18);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.12),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .command-label {
        font-size: 0.64rem;
        font-weight: 1000;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #8ea6d1;
        margin-bottom: 3px;
    }

    .command-value {
        font-size: 0.95rem;
        font-weight: 1000;
        color: #f4f8ff;
        line-height: 1.05;
    }

    .market-banner {
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-weight: 1000;
        letter-spacing: 0.4px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .banner-bull {
        background:
            radial-gradient(circle at center, rgba(77,240,165,0.12), transparent 60%),
            linear-gradient(180deg, rgba(8,35,24,0.98), rgba(8,22,18,0.98));
        border-color: rgba(77,240,165,0.18);
        color: #4df0a5;
    }

    .banner-bear {
        background:
            radial-gradient(circle at center, rgba(255,111,142,0.12), transparent 60%),
            linear-gradient(180deg, rgba(44,10,18,0.98), rgba(24,8,12,0.98));
        border-color: rgba(255,111,142,0.18);
        color: #ff6f8e;
    }

    .banner-neutral {
        background:
            radial-gradient(circle at center, rgba(255,216,106,0.10), transparent 60%),
            linear-gradient(180deg, rgba(44,34,8,0.98), rgba(22,18,8,0.98));
        border-color: rgba(255,216,106,0.16);
        color: #ffd86a;
    }

    .section-shell {
        background: linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .section-title {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 1000;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        color: #a9c4ff;
        background: linear-gradient(180deg, rgba(21,31,52,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.16);
        margin-bottom: 10px;
    }

    .ticker-line {
        font-size: 1.55rem;
        font-weight: 1000;
        margin-bottom: 0.35rem;
        letter-spacing: 1px;
    }

    .price-box {
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 8px 14px rgba(0,0,0,0.16);
        position: relative;
        overflow: hidden;
    }

    .price-box-bull {
        background:
            radial-gradient(circle at center, rgba(77,240,165,0.12), transparent 60%),
            linear-gradient(180deg, rgba(8,35,24,0.98), rgba(8,22,18,0.98));
        border-color: rgba(77,240,165,0.18);
    }

    .price-box-bear {
        background:
            radial-gradient(circle at center, rgba(255,111,142,0.12), transparent 60%),
            linear-gradient(180deg, rgba(44,10,18,0.98), rgba(24,8,12,0.98));
        border-color: rgba(255,111,142,0.18);
    }

    .price-box-neutral {
        background:
            radial-gradient(circle at center, rgba(255,216,106,0.10), transparent 60%),
            linear-gradient(180deg, rgba(44,34,8,0.98), rgba(22,18,8,0.98));
        border-color: rgba(255,216,106,0.16);
    }

    .hero-price {
        font-size: 3.0rem;
        font-weight: 1000;
        line-height: 0.95;
        margin: 0;
    }

    .hero-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 12px;
    }

    .day-change-box { text-align: right; }

    .day-change-label {
        color: #8ea6d1;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .day-change-value {
        font-size: 1.05rem;
        font-weight: 1000;
    }

    .green { color: #4df0a5; }
    .red { color: #ff6f8e; }
    .gold { color: #ffd86a; }
    .blue { color: #8dc8ff; }
    .white { color: #f4f8ff; }

    .signal-box {
        font-size: 0.98rem;
        font-weight: 1000;
        padding: 9px 12px;
        border-radius: 12px;
        margin-top: 0.35rem;
        margin-bottom: 0.55rem;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 16px rgba(0,0,0,0.14);
        animation: pulse-glow 2.6s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%   { box-shadow: 0 8px 16px rgba(0,0,0,0.14); }
        50%  { box-shadow: 0 10px 20px rgba(0,0,0,0.18), 0 0 16px rgba(255,255,255,0.04); }
        100% { box-shadow: 0 8px 16px rgba(0,0,0,0.14); }
    }

    .signal-green {
        background: rgba(77,240,165,0.12);
        border-color: rgba(77,240,165,0.28);
        color: #4df0a5;
    }

    .signal-red {
        background: rgba(255,111,142,0.12);
        border-color: rgba(255,111,142,0.28);
        color: #ff6f8e;
    }

    .signal-gold {
        background: rgba(255,216,106,0.12);
        border-color: rgba(255,216,106,0.28);
        color: #ffd86a;
    }

    .mode-chip {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 1000;
        margin-right: 7px;
        margin-bottom: 5px;
        background: linear-gradient(180deg, rgba(21,31,52,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.18);
        box-shadow:
            0 7px 14px rgba(0,0,0,0.11),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .feed-shell {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        border-radius: 14px;
        background: linear-gradient(180deg, rgba(11,19,33,0.98), rgba(8,15,27,0.98));
        border: 1px solid rgba(96,126,194,0.18);
        padding: 9px 0;
        margin-bottom: 12px;
        box-shadow: 0 10px 18px rgba(0,0,0,0.16);
    }

    .feed-text {
        display: inline-block;
        padding-left: 100%;
        animation: ticker-scroll 26s linear infinite;
        font-size: 0.88rem;
        font-weight: 1000;
        letter-spacing: 0.2px;
    }

    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .tiny-note {
        color: #7f95b8;
        font-size: 0.78rem;
    }

    .pulse-card {
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
        background:
            radial-gradient(circle at top right, rgba(141,200,255,0.10), transparent 42%),
            linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pulse-label {
        font-size: 0.68rem;
        color: #8ea6d1;
        font-weight: 1000;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .pulse-value {
        font-size: 1.08rem;
        font-weight: 1000;
        margin-bottom: 2px;
    }

    .gauge-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .gauge-title {
        font-size: 0.68rem;
        color: #8ea6d1;
        font-weight: 1000;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .gauge-track {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.35);
    }

    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00e676 0%, #ffd740 55%, #ff5252 100%);
        box-shadow: 0 0 16px rgba(141,200,255,0.18);
        animation: gauge-breathe 2.8s ease-in-out infinite;
    }

    @keyframes gauge-breathe {
        0%   { filter: brightness(1); }
        50%  { filter: brightness(1.12); }
        100% { filter: brightness(1); }
    }

    .gauge-caption {
        display: flex;
        justify-content: space-between;
        margin-top: 7px;
        font-size: 0.76rem;
        color: #8ea6d1;
        font-weight: 900;
    }

    .go-shell {
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
        text-align: center;
        font-weight: 1000;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .go-green {
        background:
            radial-gradient(circle at center, rgba(77,240,165,0.12), transparent 60%),
            linear-gradient(180deg, rgba(8,35,24,0.98), rgba(8,22,18,0.98));
        border-color: rgba(77,240,165,0.18);
        color: #4df0a5;
    }

    .go-red {
        background:
            radial-gradient(circle at center, rgba(255,111,142,0.12), transparent 60%),
            linear-gradient(180deg, rgba(44,10,18,0.98), rgba(24,8,12,0.98));
        border-color: rgba(255,111,142,0.18);
        color: #ff6f8e;
    }

    .go-gold {
        background:
            radial-gradient(circle at center, rgba(255,216,106,0.10), transparent 60%),
            linear-gradient(180deg, rgba(44,34,8,0.98), rgba(22,18,8,0.98));
        border-color: rgba(255,216,106,0.16);
        color: #ffd86a;
    }

    .go-label {
        font-size: 0.70rem;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 5px;
    }

    .go-value {
        font-size: 1.30rem;
        line-height: 1.1;
    }

    .radar-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .radar-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 7px;
        margin-top: 8px;
    }

    .radar-dot {
        height: 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.25);
    }

    .radar-on {
        background: linear-gradient(180deg, rgba(141,200,255,0.95), rgba(58,132,255,0.88));
        box-shadow: 0 0 12px rgba(58,132,255,0.35);
    }

    .radar-hot {
        background: linear-gradient(180deg, rgba(255,216,106,0.95), rgba(255,111,142,0.90));
        box-shadow: 0 0 12px rgba(255,111,142,0.35);
    }

    .mini-alert {
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 10px;
        background: linear-gradient(180deg, rgba(15,24,40,0.98), rgba(9,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.14);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.12),
            inset 0 1px 0 rgba(255,255,255,0.02);
        font-size: 0.90rem;
    }

    .commentary-shell {
        background:
            radial-gradient(circle at top right, rgba(141,200,255,0.08), transparent 40%),
            linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
        min-height: 240px;
    }

    .commentary-title {
        font-size: 0.72rem;
        color: #8ea6d1;
        font-weight: 1000;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .commentary-text {
        color: #f4f8ff;
        font-size: 1.00rem;
        line-height: 1.55;
        font-weight: 500;
        white-space: pre-wrap;
    }

    .rolling-item {
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.90rem;
    }
    .rolling-item:last-child { border-bottom: none; }

    .scanner-shell {
        background: linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 10px 18px rgba(0,0,0,0.16), inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .scan-row {
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        display: grid;
        grid-template-columns: 1.1fr 1.2fr 0.9fr 1fr 1fr;
        gap: 10px;
        align-items: center;
        box-shadow: 0 8px 14px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .scan-green {
        background:
            radial-gradient(circle at center, rgba(77,240,165,0.10), transparent 58%),
            linear-gradient(180deg, rgba(8,35,24,0.94), rgba(8,22,18,0.94));
        border-color: rgba(77,240,165,0.20);
    }

    .scan-yellow {
        background:
            radial-gradient(circle at center, rgba(255,216,106,0.10), transparent 58%),
            linear-gradient(180deg, rgba(44,34,8,0.94), rgba(22,18,8,0.94));
        border-color: rgba(255,216,106,0.20);
    }

    .scan-red {
        background:
            radial-gradient(circle at center, rgba(255,111,142,0.10), transparent 58%),
            linear-gradient(180deg, rgba(44,10,18,0.94), rgba(24,8,12,0.94));
        border-color: rgba(255,111,142,0.20);
    }

    .scan-symbol {
        font-weight: 1000;
        letter-spacing: 0.8px;
        font-size: 1.0rem;
    }

    .scan-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 1000;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(180deg, rgba(21,31,52,0.86), rgba(12,18,31,0.86));
        color: #f4f8ff;
        width: fit-content;
    }

    .scan-pill-green { border-color: rgba(77,240,165,0.28); color: #4df0a5; }
    .scan-pill-yellow { border-color: rgba(255,216,106,0.28); color: #ffd86a; }
    .scan-pill-red { border-color: rgba(255,111,142,0.28); color: #ff6f8e; }

    .scan-metric {
        font-size: 0.92rem;
        font-weight: 900;
        color: #f4f8ff;
    }

    .scan-sub {
        font-size: 0.72rem;
        font-weight: 900;
        color: #8ea6d1;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15,24,40,0.98), rgba(9,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.14);
        border-radius: 14px;
        padding: 7px 9px;
        box-shadow:
            0 8px 14px rgba(0,0,0,0.12),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    div[data-testid="stMetricLabel"] > div {
        color: #8ea6d1 !important;
        font-weight: 800;
        letter-spacing: 0.4px;
        font-size: 0.76rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f8ff !important;
        font-weight: 1000 !important;
        font-size: 1.45rem !important;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid rgba(118,157,232,0.20);
        background: linear-gradient(180deg, #15213b 0%, #0d1728 100%);
        color: #f7fbff;
        font-weight: 1000;
        padding: 0.52rem 0.9rem;
        box-shadow:
            0 8px 14px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    div[data-baseweb="select"] > div,
    div[role="radiogroup"] { background: transparent; }

    @media (max-width: 1100px) {
        .command-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
        .hero-price { font-size: 2.6rem; }
        .main-title { font-size: 2.25rem; }
        .scan-row { grid-template-columns: 1fr; }
        .header-wrap { gap: 12px; }
        .header-logo { width: 64px; height: 64px; }
    }
</style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HELPERS
# -----------------------------
def fmt_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"

def safe_get(sig, key, default="N/A"):
    try:
        return sig.get(key, default)
    except Exception:
        return default

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def bias_color_class(bias: str):
    if bias == "BULLISH":
        return "green"
    if bias == "BEARISH":
        return "red"
    return "gold"

def signal_class(signal_text: str):
    s = str(signal_text).upper()
    if "CALL" in s or "BULL" in s or "LONG" in s:
        return "signal-green"
    if "PUT" in s or "BEAR" in s or "SHORT" in s:
        return "signal-red"
    return "signal-gold"

def price_box_class(bias: str):
    if bias == "BULLISH":
        return "price-box price-box-bull"
    if bias == "BEARISH":
        return "price-box price-box-bear"
    return "price-box price-box-neutral"

def calc_day_change(df):
    try:
        if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
            return None, None
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None, None
        first_price = float(closes.iloc[0])
        last_price = float(closes.iloc[-1])
        if first_price == 0:
            return None, None
        change = last_price - first_price
        pct = (change / first_price) * 100
        return change, pct
    except Exception:
        return None, None

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def feed_colorize(items):
    out = []
    for item in items:
        t = str(item).lower()
        if any(x in t for x in ["bull", "calls", "above", "surge", "long", "breakout"]):
            out.append(f'<span class="green">▲ {item}</span>')
        elif any(x in t for x in ["bear", "puts", "below", "breakdown", "short", "fade"]):
            out.append(f'<span class="red">▼ {item}</span>')
        elif any(x in t for x in ["chop", "warning", "battle zone", "neutral"]):
            out.append(f'<span class="gold">• {item}</span>')
        else:
            out.append(f'<span class="blue">• {item}</span>')
    return " &nbsp;&nbsp; | &nbsp;&nbsp; ".join(out)

def market_banner_class(bias):
    if bias == "BULLISH":
        return "market-banner banner-bull"
    if bias == "BEARISH":
        return "market-banner banner-bear"
    return "market-banner banner-neutral"

def go_class(go_signal):
    if go_signal == "GO LONG":
        return "go-shell go-green"
    if go_signal == "GO SHORT":
        return "go-shell go-red"
    return "go-shell go-gold"

def calc_market_condition(sig, bias, day_pct):
    pressure = safe_float(safe_get(sig, "pressure", 0), 0) or 0
    state = str(safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))).upper()
    rsi = safe_float(safe_get(sig, "rsi"), 50) or 50

    if "BREAKOUT" in state and pressure >= 75:
        return "Volatility Expansion"
    if "CHOP" in state:
        return "Range Chop"
    if bias == "BULLISH" and pressure >= 70 and rsi >= 55:
        return "Momentum Trend Day"
    if bias == "BEARISH" and pressure >= 70 and rsi <= 45:
        return "Momentum Sell Pressure"
    if day_pct is not None and abs(day_pct) >= 2.0:
        return "High Velocity Session"
    return "Balanced Rotation"

def calc_market_pulse(sig, bias):
    pressure = safe_float(safe_get(sig, "pressure", 0), 0) or 0
    price = safe_float(safe_get(sig, "price"))
    ema9 = safe_float(safe_get(sig, "ema9"))
    vwap = safe_float(safe_get(sig, "vwap"))
    state = str(safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))).upper()

    above_ema = price is not None and ema9 is not None and price >= ema9
    above_vwap = price is not None and vwap is not None and price >= vwap

    if bias == "BULLISH" and pressure >= 80 and above_ema and above_vwap:
        return "Strong Buyers"
    if bias == "BEARISH" and pressure >= 80 and (not above_ema) and (not above_vwap):
        return "Heavy Sellers"
    if "CHOP" in state:
        return "Balanced Tape"
    if pressure >= 60:
        return "Trend Expansion"
    return "Developing Pressure"

def calc_go_signal(sig, bias):
    price = safe_float(safe_get(sig, "price"))
    calls_above = safe_float(safe_get(sig, "calls_favored_above"))
    puts_below = safe_float(safe_get(sig, "puts_favored_below"))
    pressure = safe_float(safe_get(sig, "pressure", 50), 50) or 50
    state = str(safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))).upper()

    if price is None:
        return "WAIT"

    if bias == "BULLISH" and calls_above is not None and price >= calls_above and pressure >= 60:
        return "GO LONG"

    if bias == "BEARISH" and puts_below is not None and price <= puts_below and pressure >= 60:
        return "GO SHORT"

    if "CHOP" in state:
        return "WAIT"

    return "WAIT"

def build_pressure_gauge(value):
    try:
        v = max(0, min(int(float(value)), 100))
    except Exception:
        v = 0

    st.markdown(
        f"""
<div class="gauge-shell">
    <div class="gauge-title">Pressure Gauge</div>
    <div class="gauge-track">
        <div class="gauge-fill" style="width:{v}%;"></div>
    </div>
    <div class="gauge-caption">
        <span>0</span>
        <span>{v}/100</span>
        <span>100</span>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

def calc_volatility_score(sig, day_pct):
    pressure = safe_float(safe_get(sig, "pressure", 50), 50) or 50
    rsi = abs((safe_float(safe_get(sig, "rsi"), 50) or 50) - 50) * 2
    price = safe_float(safe_get(sig, "price"))
    chop_low = safe_float(safe_get(sig, "chop_low"))
    chop_high = safe_float(safe_get(sig, "chop_high"))

    score = 0.0
    score += min(40, pressure * 0.35)
    score += min(20, rsi * 0.25)

    if day_pct is not None:
        score += min(25, abs(day_pct) * 4)

    if price and chop_low is not None and chop_high is not None and price != 0:
        width_pct = abs(chop_high - chop_low) / price * 100
        score += min(15, width_pct * 8)

    return max(0, min(100, int(round(score))))

def build_volatility_radar(vol_score):
    total_dots = 10
    on_dots = max(0, min(total_dots, math.ceil((vol_score / 100) * total_dots)))

    dots = []
    for i in range(total_dots):
        cls = "radar-dot"
        if i < on_dots:
            cls += " radar-hot" if vol_score >= 70 else " radar-on"
        dots.append(f'<div class="{cls}"></div>')

    radar_label = "HOT" if vol_score >= 70 else "ELEVATED" if vol_score >= 45 else "STABLE"
    tone_cls = "red" if vol_score >= 70 else "gold" if vol_score >= 45 else "blue"

    st.markdown(
        f"""
<div class="radar-shell">
    <div class="gauge-title">Volatility Radar</div>
    <div class="pulse-value {tone_cls}">{radar_label}</div>
    <div class="tiny-note">Volatility score: {vol_score}/100</div>
    <div class="radar-grid">
        {''.join(dots)}
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

def build_market_pulse(pulse_text):
    tone_cls = "green"
    pulse_upper = str(pulse_text).upper()
    if "SELL" in pulse_upper or "DISTRIBUTION" in pulse_upper:
        tone_cls = "red"
    elif "BALANCED" in pulse_upper:
        tone_cls = "gold"

    st.markdown(
        f"""
<div class="pulse-card">
    <div class="pulse-label">Market Pulse</div>
    <div class="pulse-value {tone_cls}">{pulse_text}</div>
    <div class="tiny-note">Live read from pressure, bias, structure, VWAP and EMA alignment.</div>
</div>
        """,
        unsafe_allow_html=True
    )

def build_go_signal(go_signal, sound_armed=False):
    subtitle = "Signal armed" if sound_armed else "Visual signal only"
    st.markdown(
        f"""
<div class="{go_class(go_signal)}">
    <div class="go-label">Go / No Go Signal</div>
    <div class="go-value">{go_signal}</div>
    <div class="tiny-note">{subtitle}</div>
</div>
        """,
        unsafe_allow_html=True
    )

def build_commentary_box(text):
    safe_text = text if text else "No commentary available."
    safe_text = safe_text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
<div class="commentary-shell">
    <div class="commentary-title">Commentary</div>
    <div class="commentary-text">{safe_text}</div>
</div>
        """,
        unsafe_allow_html=True
    )

def classify_scan_color(sig: dict) -> tuple[str, str, str]:
    bias = str(safe_get(sig, "bias", "NEUTRAL")).upper()
    state = str(safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))).upper()
    pressure = safe_float(safe_get(sig, "pressure", 0), 0) or 0

    if ("BEAR" in state or "BREAKDOWN" in state) and pressure >= 60:
        return ("scan-red", "scan-pill-red", "RED")

    if ("BULL" in state or "BREAKOUT" in state) and pressure >= 60 and bias == "BULLISH":
        return ("scan-green", "scan-pill-green", "GREEN")

    if bias == "BEARISH" and pressure >= 65:
        return ("scan-red", "scan-pill-red", "RED")
    if bias == "BULLISH" and pressure >= 65:
        return ("scan-green", "scan-pill-green", "GREEN")

    return ("scan-yellow", "scan-pill-yellow", "YELLOW")

def render_scanner_row(ticker: str, sig: dict):
    row_cls, pill_cls, label = classify_scan_color(sig)

    price = fmt_num(safe_get(sig, "price"))
    state = safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))
    bias = safe_get(sig, "bias", "NEUTRAL")
    pressure = safe_get(sig, "pressure", "N/A")
    signal = safe_get(sig, "signal", "N/A")

    st.markdown(
        f"""
<div class="scan-row {row_cls}">
  <div>
    <div class="scan-symbol">{ticker}</div>
    <div class="scan-sub">PRICE</div>
    <div class="scan-metric">${price}</div>
  </div>

  <div>
    <div class="scan-pill {pill_cls}">{label}</div>
    <div class="scan-sub" style="margin-top:6px;">STATE</div>
    <div class="scan-metric">{state}</div>
  </div>

  <div>
    <div class="scan-sub">BIAS</div>
    <div class="scan-metric">{bias}</div>
    <div class="scan-sub" style="margin-top:6px;">PRESSURE</div>
    <div class="scan-metric">{pressure}/100</div>
  </div>

  <div>
    <div class="scan-sub">SIGNAL</div>
    <div class="scan-metric">{signal}</div>
  </div>

  <div>
    <div class="scan-sub">TRIGGERS</div>
    <div class="scan-metric">↑ {fmt_num(safe_get(sig,"calls_favored_above"))} / ↓ {fmt_num(safe_get(sig,"puts_favored_below"))}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# HEADER
# -----------------------------
if LOGO_B64:
    st.markdown(
        f"""
        <div class="header-wrap">
            <img class="header-logo" src="data:image/jpeg;base64,{LOGO_B64}" alt="Lockout Signals Logo">
            <div class="header-text-wrap">
                <div class="main-title">Lockout Signals</div>
                <div class="subtle">Command center upgrade • premium UI layer • backend brain untouched</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown('<div class="main-title">Lockout Signals</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Command center upgrade • premium UI layer • backend brain untouched</div>', unsafe_allow_html=True)

# Autorefresh (if available)
try:
    from streamlit import st_autorefresh
except Exception:
    st_autorefresh = None

top1, top2, top3, top4, top5 = st.columns([1.05, 1.05, 1.05, 1.05, 2.8])
with top1:
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()
with top2:
    show_charts = st.toggle("Show Charts", value=False)
with top3:
    sound_alerts = st.toggle("Sound Alerts", value=False)
with top4:
    auto_scan = st.toggle("Auto Scan", value=False)
with top5:
    st.markdown(
        '<div class="tiny-note">Safe mode engaged: display upgrades only. Signal generation and backend logic remain untouched.</div>',
        unsafe_allow_html=True
    )

control1, control2, control3 = st.columns([2, 2, 2])
with control1:
    selected_ticker = st.selectbox(
        "Select Ticker",
        [
            "SPY", "QQQ", "IWM", "DIA",
            "AAPL", "NVDA", "TSLA", "AMD", "META",
            "AMZN", "MSFT", "NBIS", "NFLX", "PLTR",
            "MSTR", "COIN", "SOFI", "HOOD", "INTC",
            "MU", "AVGO", "SMCI", "ARM", "BABA",
            "XOM", "OXY", "USO"
        ],
        index=0
    )
with control2:
    mode = st.radio("Mode", ["Aggressive", "Full Send"], horizontal=True, index=0)
with control3:
    scan_every = st.selectbox("Auto Scan Interval", [15, 30, 60], index=1)

if auto_scan and st_autorefresh is not None:
    st_autorefresh(interval=int(scan_every * 1000), key="lockout_autoscan")
elif auto_scan and st_autorefresh is None:
    st.info("Auto Scan needs a newer Streamlit build. You can still use Scan Now.")

# -----------------------------
# WATCHLIST SCANNER (COLOR CODED)
# -----------------------------
st.markdown('<div class="scanner-shell">', unsafe_allow_html=True)
section_title("Watchlist Scanner")

watch_default = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "AMD", "OXY", "USO"]
watchlist = st.multiselect(
    "Watchlist",
    options=[
        "SPY", "QQQ", "IWM", "DIA",
        "AAPL", "NVDA", "TSLA", "AMD", "META",
        "AMZN", "MSFT", "NBIS", "NFLX", "PLTR",
        "MSTR", "COIN", "SOFI", "HOOD", "INTC",
        "MU", "AVGO", "SMCI", "ARM", "BABA",
        "XOM", "OXY", "USO"
    ],
    default=watch_default
)

scan_now = st.button("Scan Now", use_container_width=True)
should_scan = scan_now or auto_scan

if should_scan:
    if not watchlist:
        st.warning("Add at least 1 ticker to the watchlist.")
    else:
        for t in watchlist:
            s = generate_signal(t, interval="5m")
            if isinstance(s, dict) and "error" in s:
                st.markdown(
                    f"""
<div class="scan-row scan-yellow">
  <div>
    <div class="scan-symbol">{t}</div>
    <div class="scan-sub">STATUS</div>
    <div class="scan-metric">No data</div>
  </div>
  <div>
    <div class="scan-pill scan-pill-yellow">YELLOW</div>
    <div class="scan-sub" style="margin-top:6px;">ERROR</div>
    <div class="scan-metric">{s.get("error","Unknown error")}</div>
  </div>
  <div><div class="scan-sub">BIAS</div><div class="scan-metric">N/A</div></div>
  <div><div class="scan-sub">SIGNAL</div><div class="scan-metric">N/A</div></div>
  <div><div class="scan-sub">TRIGGERS</div><div class="scan-metric">N/A</div></div>
</div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                render_scanner_row(t, s)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# DATA (BRAIN CALLS — untouched)
# -----------------------------
sig = generate_signal(selected_ticker, interval="5m")
if isinstance(sig, dict) and "error" in sig:
    st.error(sig["error"])
    st.stop()

df_day = get_data(selected_ticker, interval="5m")
_, day_pct = calc_day_change(df_day)

bias = safe_get(sig, "bias", "NEUTRAL")
bias_cls = bias_color_class(bias)
state = safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))
price_text = fmt_num(safe_get(sig, "price"))
pressure_value = safe_get(sig, "pressure", 0)
session_status = safe_get(sig, "session_status", "N/A")
signal_text = safe_get(sig, "signal", "N/A")
rsi_text = fmt_num(safe_get(sig, "rsi"))

if day_pct is None:
    day_text = "N/A"
    day_text_class = "white"
else:
    sign = "+" if day_pct >= 0 else ""
    day_text = f"{sign}{day_pct:.2f}%"
    day_text_class = "green" if day_pct >= 0 else "red"

market_condition = calc_market_condition(sig, bias, day_pct)
market_pulse = calc_market_pulse(sig, bias)
go_signal = calc_go_signal(sig, bias)
vol_score = calc_volatility_score(sig, day_pct)

commentary = safe_get(sig, "commentary", "")
if not commentary:
    commentary = f"{selected_ticker} is reading {state} with {bias} bias. Signal: {signal_text}."

feed_items = safe_get(sig, "feed", [])
if isinstance(feed_items, list) and feed_items:
    feed_html = feed_colorize(feed_items[:6])
else:
    feed_html = f'<span class="blue">• {selected_ticker} tactical feed online</span>'

# -----------------------------
# COMMAND BAR
# -----------------------------
st.markdown(
    f"""
<div class="command-bar">
    <div class="command-grid">
        <div class="command-pill">
            <div class="command-label">Symbol</div>
            <div class="command-value">{selected_ticker}</div>
        </div>
        <div class="command-pill">
            <div class="command-label">Price</div>
            <div class="command-value">${price_text}</div>
        </div>
        <div class="command-pill">
            <div class="command-label">State</div>
            <div class="command-value">{state}</div>
        </div>
        <div class="command-pill">
            <div class="command-label">Pressure</div>
            <div class="command-value">{pressure_value}/100</div>
        </div>
        <div class="command-pill">
            <div class="command-label">Session</div>
            <div class="command-value">{session_status}</div>
        </div>
    </div>
</div>
    """,
    unsafe_allow_html=True
)

st.markdown(f'<div class="{market_banner_class(bias)}">MARKET CONDITION: {market_condition}</div>', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="feed-shell">
    <div class="feed-text">{feed_html} &nbsp;&nbsp; | &nbsp;&nbsp; {feed_html}</div>
</div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HERO
# -----------------------------
st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.markdown(f'<div class="ticker-line {bias_cls}">{selected_ticker} • 5M Tactical Brain</div>', unsafe_allow_html=True)

st.markdown(f'<div class="{price_box_class(bias)}">', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="hero-row">
    <div class="hero-price {bias_cls}">${price_text}</div>
    <div class="day-change-box">
        <div class="day-change-label">Day %</div>
        <div class="day-change-value {day_text_class}">{day_text}</div>
    </div>
</div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="signal-box {signal_class(signal_text)}">{signal_text}</div>', unsafe_allow_html=True)

chip_html = f"""
<span class="mode-chip">{bias}</span>
<span class="mode-chip">{state}</span>
<span class="mode-chip">Heat: {safe_get(sig, "heat", "N/A")}</span>
<span class="mode-chip">Conviction: {safe_get(sig, "conviction", "N/A")}</span>
<span class="mode-chip">Mode: {mode.upper()}</span>
"""
st.markdown(chip_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# BODY LAYOUT (center = commentary)
# -----------------------------
left, center, right = st.columns([1.55, 1.35, 0.90])

with left:
    build_go_signal(go_signal, sound_armed=sound_alerts)
    build_market_pulse(market_pulse)
    build_pressure_gauge(pressure_value)
    build_volatility_radar(vol_score)

with center:
    build_commentary_box(commentary)

    st.markdown('<div class="mini-alert">', unsafe_allow_html=True)
    st.markdown(
        f"<strong>Trade Readiness:</strong> {go_signal} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<strong>Mode:</strong> {mode.upper()} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<strong>Alerts:</strong> {'ARMED' if sound_alerts else 'OFF'}",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Signal Stack")
    st.metric("Bias", bias)
    st.metric("State", state)
    st.metric("RSI", rsi_text)
    st.metric("Session", session_status)
    st.metric("Pressure", f"{pressure_value}/100")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# MAIN DATA PANELS
# -----------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Bias + Structure")
    a1, a2 = st.columns(2)
    with a1:
        st.metric("Bias", bias)
        st.metric("RSI", fmt_num(safe_get(sig, "rsi")))
        st.metric("EMA9", fmt_num(safe_get(sig, "ema9")))
        st.metric("VWAP", fmt_num(safe_get(sig, "vwap")))
    with a2:
        st.metric("State", state)
        st.metric("EMA20", fmt_num(safe_get(sig, "ema20")))
        st.metric("Session", session_status)
        st.metric("Price", price_text)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Battle Zones + Targets")
    b1, b2 = st.columns(2)
    with b1:
        st.metric("Calls Favored Above", fmt_num(safe_get(sig, "calls_favored_above")))
        st.metric("Chop Zone", f'{fmt_num(safe_get(sig, "chop_low"))} - {fmt_num(safe_get(sig, "chop_high"))}')
        st.metric("Invalidation", fmt_num(safe_get(sig, "invalidation", safe_get(sig, "invalid"))))
        st.metric("Likely Up", fmt_num(safe_get(sig, "likely_up", safe_get(sig, "likely"))))
        st.metric("Likely Down", fmt_num(safe_get(sig, "likely_down")))
    with b2:
        st.metric("Puts Favored Below", fmt_num(safe_get(sig, "puts_favored_below")))
        st.metric("Warning Line", fmt_num(safe_get(sig, "warning_line")))
        st.metric("Stretch Up", fmt_num(safe_get(sig, "stretch_up", safe_get(sig, "stretch"))))
        st.metric("Stretch Down", fmt_num(safe_get(sig, "stretch_down")))
        st.metric("Pressure", f'{safe_get(sig, "pressure", "N/A")}/100')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Session Levels")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("PM High", fmt_num(safe_get(sig, "pm_high")))
    with s2:
        st.metric("PM Low", fmt_num(safe_get(sig, "pm_low")))
    with s3:
        st.metric("OR High", fmt_num(safe_get(sig, "or_high")))
    with s4:
        st.metric("OR Low", fmt_num(safe_get(sig, "or_low")))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# OPTIONAL FEED DETAILS
# -----------------------------
if isinstance(feed_items, list) and len(feed_items) > 0:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Rolling Feed")
    for item in feed_items[:6]:
        st.markdown(f'<div class="rolling-item">• {item}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# CHART
# -----------------------------
if show_charts:
    with st.expander(f"Open {selected_ticker} Chart Deck"):
        df_chart = get_data(selected_ticker, interval="5m")
        if df_chart is None or getattr(df_chart, "empty", True):
            st.warning("No chart data available.")
        else:
            df_chart = add_indicators(df_chart)
            cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df_chart.columns]
            if cols:
                st.line_chart(df_chart[cols], use_container_width=True)
            else:
                st.line_chart(df_chart[["Close"]], use_container_width=True)