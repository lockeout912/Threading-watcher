import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# STYLING
# ----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #05070d 0%, #0a1220 100%);
        color: #e8eefc;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: #f5f7ff;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #8ea3c7;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .panel {
        background: rgba(13, 19, 34, 0.92);
        border: 1px solid rgba(120, 150, 220, 0.18);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 0 28px rgba(0, 0, 0, 0.35);
        margin-bottom: 18px;
    }

    .panel h3 {
        margin-top: 0;
        margin-bottom: 0.9rem;
        font-size: 1.25rem;
        color: #ffffff;
    }

    .section-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #7f92b2;
        margin-top: 0.4rem;
        margin-bottom: 0.45rem;
        font-weight: 700;
    }

    .metric-card {
        background: rgba(19, 28, 49, 0.95);
        border: 1px solid rgba(126, 154, 214, 0.12);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #8ea3c7;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.18rem;
        font-weight: 700;
    }

    .bull-box {
        background: linear-gradient(135deg, rgba(6, 58, 31, 0.95), rgba(10, 90, 49, 0.82));
        border: 1px solid rgba(45, 220, 130, 0.35);
        border-radius: 14px;
        padding: 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        color: #d9ffea;
        font-weight: 700;
    }

    .bear-box {
        background: linear-gradient(135deg, rgba(66, 10, 16, 0.95), rgba(108, 17, 28, 0.82));
        border: 1px solid rgba(255, 93, 115, 0.35);
        border-radius: 14px;
        padding: 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        color: #ffe1e5;
        font-weight: 700;
    }

    .chop-box {
        background: linear-gradient(135deg, rgba(58, 48, 13, 0.95), rgba(96, 80, 17, 0.82));
        border: 1px solid rgba(255, 210, 77, 0.35);
        border-radius: 14px;
        padding: 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        color: #fff3c7;
        font-weight: 700;
    }

    .commentary-box {
        background: rgba(16, 24, 42, 0.95);
        border-left: 4px solid #4da3ff;
        border-radius: 10px;
        padding: 12px 14px;
        color: #d7e4ff;
        margin-top: 8px;
    }

    .footer-note {
        color: #7083a4;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid rgba(120, 150, 220, 0.24);
        background: linear-gradient(180deg, #121b2f 0%, #0d1525 100%);
        color: #f5f7ff;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(140, 175, 255, 0.55);
        color: white;
    }

    .divider-space {
        height: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="main-title">LOCKOUT SIGNALS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Battle Map V2 • ORB + Premarket + Bias Zones • Space Force Edition</div>',
    unsafe_allow_html=True
)

top1, top2, top3 = st.columns([1, 1, 2])

with top1:
    if st.button("Refresh Now"):
        st.rerun()

with top2:
    if st.button("Reload Board"):
        st.rerun()

with top3:
    st.markdown('<div class="footer-note">Use after 9:35 AM ET for full Opening Range logic</div>', unsafe_allow_html=True)

tickers = ["SPY", "QQQ"]

# ----------------------------
# HELPERS
# ----------------------------
def metric_box(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def fmt_level(x):
    return f"{x:.2f}" if x is not None else "N/A"

# ----------------------------
# MAIN PANELS
# ----------------------------
for ticker in tickers:
    sig = generate_signal(ticker)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"<h3>{ticker} Command Deck</h3>", unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        metric_box("Price", f"${sig['price']:.2f}")
    with row1_col2:
        metric_box("Bias", sig["bias"])
    with row1_col3:
        metric_box("Market State", sig["market_state"])

    with row2_col1:
        metric_box("Pressure", f"{sig['pressure']}/100")
    with row2_col2:
        metric_box("Heat", sig["heat"])
    with row2_col3:
        metric_box("Signal", sig["signal"])

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-label">Core Structure</div>', unsafe_allow_html=True)
        metric_box("RSI", f"{sig['rsi']:.2f}")
        metric_box("EMA9", fmt_level(sig["ema9"]))
        metric_box("VWAP", fmt_level(sig["vwap"]))
        metric_box("Premarket High", fmt_level(sig["pm_high"]))
        metric_box("Premarket Low", fmt_level(sig["pm_low"]))

        st.markdown('<div class="section-label">Opening Range</div>', unsafe_allow_html=True)
        metric_box("OR High", fmt_level(sig["or_high"]))
        metric_box("OR Low", fmt_level(sig["or_low"]))

    with right:
        st.markdown('<div class="section-label">Battle Zones</div>', unsafe_allow_html=True)
        metric_box("Calls Favored Above", fmt_level(sig["calls_favored_above"]))
        metric_box("Puts Favored Below", fmt_level(sig["puts_favored_below"]))
        metric_box("Chop Zone", f"{sig['chop_low']:.2f} - {sig['chop_high']:.2f}")
        metric_box("Warning Line", fmt_level(sig["warning_line"]))
        metric_box("Invalidation", fmt_level(sig["invalidation"]))

    target_left, target_right = st.columns(2)

    with target_left:
        st.markdown('<div class="section-label">Upside Targets</div>', unsafe_allow_html=True)
        metric_box("Likely Up", fmt_level(sig["likely_up"]))
        metric_box("Stretch Up", fmt_level(sig["stretch_up"]))

    with target_right:
        st.markdown('<div class="section-label">Downside Targets</div>', unsafe_allow_html=True)
        metric_box("Likely Down", fmt_level(sig["likely_down"]))
        metric_box("Stretch Down", fmt_level(sig["stretch_down"]))

    # Signal state banner
    price = sig["price"]
    calls_level = sig["calls_favored_above"]
    puts_level = sig["puts_favored_below"]

    if price > calls_level:
        st.markdown(
            f'<div class="bull-box">BULL CONTROL — Calls favored while price holds above {calls_level:.2f}</div>',
            unsafe_allow_html=True
        )
    elif price < puts_level:
        st.markdown(
            f'<div class="bear-box">BEAR CONTROL — Puts favored while price stays below {puts_level:.2f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chop-box">CHOP ZONE — No-man’s-land between {puts_level:.2f} and {calls_level:.2f}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-label">AI Commentary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="commentary-box">{sig["commentary"]}</div>', unsafe_allow_html=True)

    if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
        df = get_data(ticker)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "VWAP"] if c in df.columns]
            st.line_chart(df[cols], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)