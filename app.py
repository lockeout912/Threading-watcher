import math
import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(0,255,180,0.06), transparent 24%),
            radial-gradient(circle at 88% 12%, rgba(0,140,255,0.07), transparent 26%),
            linear-gradient(180deg, #040812 0%, #07111d 55%, #091521 100%);
        color: #f4f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09111d 0%, #0c1525 100%);
    }

    .main-title {
        font-size: 2.9rem;
        font-weight: 1000;
        color: #ffffff;
        margin-bottom: 0.15rem;
        line-height: 1.02;
        text-shadow: 0 0 22px rgba(0,180,255,0.12);
    }

    .subtle {
        color: #9cb1d3;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .hero-card {
        background:
            radial-gradient(circle at top center, rgba(255,255,255,0.03), transparent 36%),
            linear-gradient(180deg, rgba(18,27,46,0.98), rgba(9,15,26,0.98));
        border: 1px solid rgba(101,139,215,0.22);
        border-radius: 24px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow:
            0 18px 30px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .command-bar {
        background:
            linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.18);
        border-radius: 20px;
        padding: 14px 16px;
        margin-bottom: 14px;
        box-shadow:
            0 12px 22px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .command-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 10px;
    }

    .command-pill {
        border-radius: 16px;
        padding: 10px 12px;
        background: linear-gradient(180deg, rgba(21,31,52,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.18);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .command-label {
        font-size: 0.68rem;
        font-weight: 1000;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #8ea6d1;
        margin-bottom: 4px;
    }

    .command-value {
        font-size: 1.02rem;
        font-weight: 1000;
        color: #f4f8ff;
        line-height: 1.05;
    }

    .market-banner {
        border-radius: 18px;
        padding: 13px 16px;
        margin-bottom: 14px;
        font-weight: 1000;
        letter-spacing: 0.5px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.15),
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
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow:
            0 12px 22px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .section-title {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 1000;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #a9c4ff;
        background: linear-gradient(180deg, rgba(21,31,52,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.16);
        margin-bottom: 12px;
    }

    .ticker-line {
        font-size: 1.7rem;
        font-weight: 1000;
        margin-bottom: 0.45rem;
        letter-spacing: 1px;
    }

    .price-box {
        border-radius: 20px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 10px 18px rgba(0,0,0,0.18);
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
        font-size: 3.5rem;
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

    .day-change-box {
        text-align: right;
    }

    .day-change-label {
        color: #8ea6d1;
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .day-change-value {
        font-size: 1.2rem;
        font-weight: 1000;
    }

    .green { color: #4df0a5; }
    .red { color: #ff6f8e; }
    .gold { color: #ffd86a; }
    .blue { color: #8dc8ff; }
    .white { color: #f4f8ff; }

    .signal-box {
        font-size: 1.08rem;
        font-weight: 1000;
        padding: 11px 14px;
        border-radius: 14px;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 18px rgba(0,0,0,0.15);
        animation: pulse-glow 2.6s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%   { box-shadow: 0 8px 18px rgba(0,0,0,0.15); }
        50%  { box-shadow: 0 10px 22px rgba(0,0,0,0.20), 0 0 18px rgba(255,255,255,0.04); }
        100% { box-shadow: 0 8px 18px rgba(0,0,0,0.15); }
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
        padding: 8px 13px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 1000;
        margin-right: 8px;
        margin-bottom: 6px;
        background: linear-gradient(180deg, rgba(21,31,52,0.96), rgba(12,18,31,0.96));
        border: 1px solid rgba(113,150,225,0.18);
        box-shadow:
            0 8px 16px rgba(0,0,0,0.12),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .feed-shell {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(11,19,33,0.98), rgba(8,15,27,0.98));
        border: 1px solid rgba(96,126,194,0.18);
        padding: 11px 0;
        margin-bottom: 16px;
        box-shadow: 0 12px 22px rgba(0,0,0,0.18);
    }

    .feed-text {
        display: inline-block;
        padding-left: 100%;
        animation: ticker-scroll 26s linear infinite;
        font-size: 0.95rem;
        font-weight: 1000;
        letter-spacing: 0.2px;
    }

    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .tiny-note {
        color: #7f95b8;
        font-size: 0.82rem;
    }

    .ladder-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 16px;
        padding: 12px;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .ladder-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 2px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.94rem;
        font-weight: 900;
    }

    .ladder-row:last-child {
        border-bottom: none;
    }

    .ladder-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .ladder-tag-up { color: #4df0a5; }
    .ladder-tag-mid { color: #ffd86a; }
    .ladder-tag-down { color: #ff6f8e; }

    .pulse-card {
        border-radius: 18px;
        padding: 16px;
        margin-bottom: 12px;
        background:
            radial-gradient(circle at top right, rgba(141,200,255,0.10), transparent 42%),
            linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pulse-label {
        font-size: 0.72rem;
        color: #8ea6d1;
        font-weight: 1000;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .pulse-value {
        font-size: 1.25rem;
        font-weight: 1000;
        margin-bottom: 3px;
    }

    .gauge-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .gauge-title {
        font-size: 0.72rem;
        color: #8ea6d1;
        font-weight: 1000;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .gauge-track {
        width: 100%;
        height: 14px;
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
        margin-top: 8px;
        font-size: 0.8rem;
        color: #8ea6d1;
        font-weight: 900;
    }

    .prob-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .prob-row {
        margin-bottom: 12px;
    }

    .prob-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 0.9rem;
        font-weight: 1000;
    }

    .prob-track {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }

    .prob-fill-up {
        height: 100%;
        background: linear-gradient(90deg, rgba(77,240,165,0.55), #4df0a5);
        border-radius: 999px;
    }

    .prob-fill-down {
        height: 100%;
        background: linear-gradient(90deg, rgba(255,111,142,0.55), #ff6f8e);
        border-radius: 999px;
    }

    .go-shell {
        border-radius: 18px;
        padding: 16px;
        margin-bottom: 14px;
        text-align: center;
        font-weight: 1000;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 12px 20px rgba(0,0,0,0.18),
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
        font-size: 0.74rem;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 6px;
    }

    .go-value {
        font-size: 1.5rem;
        line-height: 1.1;
    }

    .radar-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .radar-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        margin-top: 10px;
    }

    .radar-dot {
        height: 18px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.25);
    }

    .radar-on {
        background: linear-gradient(180deg, rgba(141,200,255,0.95), rgba(58,132,255,0.88));
        box-shadow: 0 0 14px rgba(58,132,255,0.35);
    }

    .radar-hot {
        background: linear-gradient(180deg, rgba(255,216,106,0.95), rgba(255,111,142,0.90));
        box-shadow: 0 0 14px rgba(255,111,142,0.35);
    }

    .mini-alert {
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 12px;
        background: linear-gradient(180deg, rgba(15,24,40,0.98), rgba(9,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.14);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .mini-alert strong {
        color: #f4f8ff;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15,24,40,0.98), rgba(9,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.14);
        border-radius: 16px;
        padding: 10px 12px;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    div[data-testid="stMetricLabel"] > div {
        color: #8ea6d1 !important;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f8ff !important;
        font-weight: 1000 !important;
    }

    div[data-testid="stButton"] > button {
        border-radius: 14px;
        border: 1px solid rgba(118,157,232,0.20);
        background: linear-gradient(180deg, #15213b 0%, #0d1728 100%);
        color: #f7fbff;
        font-weight: 1000;
        padding: 0.60rem 1rem;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    div[data-baseweb="select"] > div,
    div[role="radiogroup"] {
        background: transparent;
    }

    @media (max-width: 1100px) {
        .command-grid {
            grid-template-columns: repeat(2, minmax(120px, 1fr));
        }
        .hero-price {
            font-size: 2.9rem;
        }
        .main-title {
            font-size: 2.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
def fmt_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"

def safe_get(sig, key, default="N/A"):
    return sig.get(key, default)

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
        if df is None or df.empty or "Close" not in df.columns:
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
        t = item.lower()
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

def calc_trade_probabilities(sig, bias):
    pressure = safe_float(safe_get(sig, "pressure", 50), 50) or 50
    rsi = safe_float(safe_get(sig, "rsi"), 50) or 50
    price = safe_float(safe_get(sig, "price"))
    ema9 = safe_float(safe_get(sig, "ema9"))
    vwap = safe_float(safe_get(sig, "vwap"))

    score = 50.0

    if bias == "BULLISH":
        score += 12
    elif bias == "BEARISH":
        score -= 12

    score += (pressure - 50) * 0.35
    score += (rsi - 50) * 0.25

    if price is not None and ema9 is not None:
        score += 6 if price >= ema9 else -6

    if price is not None and vwap is not None:
        score += 6 if price >= vwap else -6

    score = max(5, min(95, round(score)))
    return score, 100 - score

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

def build_probabilities(up_pct, down_pct):
    st.markdown(
        f"""
        <div class="prob-shell">
            <div class="gauge-title">Trade Probability</div>

            <div class="prob-row">
                <div class="prob-head">
                    <span class="green">Upside Probability</span>
                    <span class="green">{up_pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill-up" style="width:{up_pct}%;"></div>
                </div>
            </div>

            <div class="prob-row" style="margin-bottom:0;">
                <div class="prob-head">
                    <span class="red">Downside Probability</span>
                    <span class="red">{down_pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill-down" style="width:{down_pct}%;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def build_target_ladder(sig):
    calls_above = fmt_num(safe_get(sig, "calls_favored_above"))
    likely_up = fmt_num(safe_get(sig, "likely_up", safe_get(sig, "likely")))
    stretch_up = fmt_num(safe_get(sig, "stretch_up", safe_get(sig, "stretch")))
    warning_line = fmt_num(safe_get(sig, "warning_line"))
    likely_down = fmt_num(safe_get(sig, "likely_down"))
    stretch_down = fmt_num(safe_get(sig, "stretch_down"))

    st.markdown(
        f"""
        <div class="ladder-shell">
            <div class="gauge-title">Target Ladder</div>

            <div class="ladder-row">
                <div class="ladder-left"><span class="ladder-tag-up">↑</span><span>Stretch Up</span></div>
                <div class="ladder-tag-up">{stretch_up}</div>
            </div>

            <div class="ladder-row">
                <div class="ladder-left"><span class="ladder-tag-up">↑</span><span>Likely Up</span></div>
                <div class="ladder-tag-up">{likely_up}</div>
            </div>

            <div class="ladder-row">
                <div class="ladder-left"><span class="ladder-tag-mid">—</span><span>Calls Favored Above</span></div>
                <div class="ladder-tag-mid">{calls_above}</div>
            </div>

            <div class="ladder-row">
                <div class="ladder-left"><span class="ladder-tag-mid">•</span><span>Warning Line</span></div>
                <div class="ladder-tag-mid">{warning_line}</div>
            </div>

            <div class="ladder-row">
                <div class="ladder-left"><span class="ladder-tag-down">↓</span><span>Likely Down</span></div>
                <div class="ladder-tag-down">{likely_down}</div>
            </div>

            <div class="ladder-row">
                <div class="ladder-left"><span class="ladder-tag-down">↓</span><span>Stretch Down</span></div>
                <div class="ladder-tag-down">{stretch_down}</div>
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

    st.markdown(
        f"""
        <div class="radar-shell">
            <div class="gauge-title">Volatility Radar</div>
            <div class="pulse-value {'red' if vol_score >= 70 else 'gold' if vol_score >= 45 else 'blue'}">{radar_label}</div>
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
    pulse_upper = pulse_text.upper()
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

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">Lockout Signals</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Command center upgrade • premium UI layer • backend brain untouched</div>', unsafe_allow_html=True)

top1, top2, top3, top4 = st.columns([1.2, 1.2, 1.2, 2.6])
with top1:
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()
with top2:
    show_charts = st.toggle("Show Charts", value=False)
with top3:
    sound_alerts = st.toggle("Sound Alerts", value=False)
with top4:
    st.markdown(
        '<div class="tiny-note">Safe mode engaged: this upgrade changes display only. Brain logic and signal generation stay exactly as-is.</div>',
        unsafe_allow_html=True
    )

control1, control2 = st.columns([2, 2])
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
    mode = st.radio(
        "Mode",
        ["Aggressive", "Full Send"],
        horizontal=True,
        index=0
    )

# -----------------------------
# DATA
# -----------------------------
sig = generate_signal(selected_ticker, interval="5m")

if "error" in sig:
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
up_prob, down_prob = calc_trade_probabilities(sig, bias)
go_signal = calc_go_signal(sig, bias)
vol_score = calc_volatility_score(sig, day_pct)

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

st.markdown(
    f'<div class="{market_banner_class(bias)}">MARKET CONDITION: {market_condition}</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    f'<div class="ticker-line {bias_cls}">{selected_ticker} • 5M Tactical Brain</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    f'<div class="signal-box {signal_class(signal_text)}">{signal_text}</div>',
    unsafe_allow_html=True
)

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
# BODY LAYOUT
# -----------------------------
left, center, right = st.columns([1.15, 1.15, 0.95])

with left:
    build_go_signal(go_signal, sound_armed=sound_alerts)
    build_market_pulse(market_pulse)
    build_pressure_gauge(pressure_value)
    build_volatility_radar(vol_score)

with center:
    build_target_ladder(sig)
    build_probabilities(up_prob, down_prob)

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

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Commentary")
    commentary = safe_get(sig, "commentary", "")
    if commentary:
        st.write(commentary)
    else:
        st.write("No commentary available.")
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
        st.write(f"• {item}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# CHART
# -----------------------------
if show_charts:
    with st.expander(f"Open {selected_ticker} Chart Deck"):
        df_chart = get_data(selected_ticker, interval="5m")
        if df_chart is None or df_chart.empty:
            st.warning("No chart data available.")
        else:
            df_chart = add_indicators(df_chart)
            cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df_chart.columns]
            if cols:
                st.line_chart(df_chart[cols], use_container_width=True)
            else:
                st.line_chart(df_chart[["Close"]], use_container_width=True)