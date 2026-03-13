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

    .day-change-box {
        text-align: right;
    }

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
        padding: 16px;
        margin-bottom: 12px;
        background:
            radial-gradient(circle at top right, rgba(141,200,255,0.10), transparent 42%),
            linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.03);
        min-height: 112px;
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

    .commentary-card {
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        background:
            radial-gradient(circle at top right, rgba(77,240,165,0.06), transparent 42%),
            linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.16);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.14),
            inset 0 1px 0 rgba(255,255,255,0.03);
        min-height: 150px;
    }

    .commentary-text {
        color: #f4f8ff;
        font-size: 0.98rem;
        line-height: 1.55;
        font-weight: 500;
    }

    .gauge-shell {
        background: linear-gradient(180deg, rgba(12,18,31,0.96), rgba(9,13,24,0.96));
        border: 1px solid rgba(101,139,215,0.16);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        min-height: 110px;
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
        padding: 18px;
        margin-bottom: 12px;
        text-align: center;
        font-weight: 1000;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 10px 18px rgba(0,0,0,0.16),
            inset 0 1px 0 rgba(255,255,255,0.03);
        min-height: 120px;
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
        padding: 16px;
        margin-bottom: 12px;
        min-height: 120px;
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
        padding: 14px 16px;
        margin-bottom: 12px;
        background: linear-gradient(180deg, rgba(15,24,40,0.98), rgba(9,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.14);
        box-shadow:
            0 8px 14px rgba(0,0,0,0.12),
            inset 0 1px 0 rgba(255,255,255,0.02);
        font-size: 0.95rem;
        min-height: 120px;
    }

    .rolling-item {
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.90rem;
    }

    .