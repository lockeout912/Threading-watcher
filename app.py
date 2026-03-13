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
        border: 1px solid rgba(