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
        background:
            radial-gradient(circle at top left, rgba(37,62,112,0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(0,204,255,0.12), transparent 22%),
            linear-gradient(180deg, #04060c 0%, #08111d 40%, #0b1422 100%);
        color: #edf4ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .command-header {
        padding: 10px 0 6px 0;
        margin-bottom: 12px;
    }

    .title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: 1.8px;
        color: #f7fbff;
        text-shadow: 0 0 16px rgba(95, 180, 255, 0.18);
    }

    .subtitle {
        color: #9ab0d8;
        font-size: 1rem;
        margin-top: 4px;
    }

    .top-chip-bar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    .top-chip {
        background: rgba(17, 26, 45, 0.95);
        border: 1px solid rgba(111, 147, 223, 0.2);
        color: #eaf3ff;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 700;
        box-shadow: 0 0 18px rgba(0, 0, 0, 0.22);
    }

    .live-feed {
        background: linear-gradient(90deg, rgba(9,16,30,0.96), rgba(17,28,51,0.96));
        border: 1px solid rgba(101, 145, 255, 0.22);
        border-radius: 16px;
        padding: 12px 14px;
        margin: 6px 0 18px 0;
        box-shadow: 0 0 24px rgba(0,0,0,0.28);
    }

    .live-feed-title {
        color: #7fd4ff;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .panel {
        background: rgba(10, 17, 31, 0.94);
        border: 1px solid rgba(100, 135, 210, 0.16);
        border-radius: 20px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 0 34px rgba(0, 0, 0, 0.34);
    }

    .panel-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 12px;
        letter-spacing: 0.6px;
    }

    .section-label {
        color: #8da4cd;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 800;
        margin: 8px 0 8px 0;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(15,25,45,0.97), rgba(11,19,35,0.97));
        border: 1px solid rgba(115, 151, 230, 0.12);
        border-radius: 15px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #86a0cb;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 4px;
        font-weight: 700;
    }

    .metric-value {
        color: #f5f9ff;
        font-size: 1.12rem;
        font-weight: 800;
    }

    .hero-banner-bull {
        background: linear-gradient(135deg, rgba(7,65,37,0.96), rgba(13,100,58,0.82));
        border: 1px solid rgba(60, 240, 148, 0.38);
        border-radius: 15px;
        padding: 14px;
        color: #e1ffef;
        font-weight: 800;
        margin: 10px 0 12px 0;
        box-shadow: 0 0 18px rgba(14, 118, 63, 0.16);
    }

    .hero-banner-bear {
        background: linear-gradient(135deg, rgba(76,12,22,0.96), rgba(126,22,38,0.82));
        border: 1px solid rgba(255, 97, 117, 0.36);
        border-radius: 15px;
        padding: 14px;
        color: #ffe3e8;
        font-weight: 800;
        margin: 10px 0 12px 0;
        box-shadow: 0 0 18px rgba(140, 26, 45, 0.16);
    }

    .hero-banner-chop {
        background: linear-gradient(135deg, rgba(72,60,15,0.96), rgba(115,94,18,0.82));
        border: 1px solid rgba(255, 214, 92, 0.34);
        border-radius: 15px;
        padding: 14px;
        color: #fff2ca;
        font-weight: 800;
        margin: 10px 0 12px 0;
        box-shadow: 0 0 18px rgba(154, 118, 20, 0.14);
    }

    .commentary-box {
        background: rgba(13, 22, 39, 0.95);
        border-left: 4px solid #56afff;
        border-radius: 12px;
        padding: 12px 14px;
        color: #dce8ff;
        margin-top: 8px;
        line-height: 1.5;
    }

    .feed-item {
        background: rgba(12, 20, 36, 0.88);
        border: 1px solid rgba(96, 129, 199, 0.12);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #e8f0ff;
        font-size: 0.95rem;
    }

    .small-note {
        color: #7086ad;
        font-size: 0.84rem;
        margin-top: 0.15rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid rgba(120, 155, 230, 0.20);
        background: linear-gradient(180deg, #15213a 0%, #0f1728 100%);
        color: #f5f9ff;
        font-weight: 800;
        padding: 0.52rem 0.95rem;
        box-shadow: 0 0 10px rgba(56, 102, 196, 0.08);
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(132, 176, 255, 0.50);
        color: white;
    }

    div[data-baseweb="select"] > div {
        background: rgba(14, 22, 38, 0.96);
        border-radius: 12px;
        border: 1px solid rgba(102, 139, 215, 0.20);
    }
</style>
""", unsafe_allow_html=True)

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


def banner_html(price, calls_level, puts_level, likely_up, likely_down):
    if price > calls_level:
        return f'<div class="hero-banner-bull">🚀 BULL CONTROL — Calls favored while price holds above {calls_level:.2f} • continuation zone toward {likely_up:.2f}</div>'
    if price < puts_level:
        return f'<div class="hero-banner-bear">🩸 BEAR CONTROL — Puts favored while price stays below {puts_level:.2f} • continuation zone toward {likely_down:.2f}</div>'
    return f'<div class="hero-banner-chop">🟡 CHOP ZONE — No-man’s-land between {puts_level:.2f} and {calls_level:.2f}. Wait for clean expansion.</div>'


# ----------------------------
# SIDEBAR / CONTROLS
# ----------------------------
with st.sidebar:
    st.markdown("## Command Controls")
    selected_ticker = st.selectbox("Primary Ticker", ["SPY", "QQQ", "IWM", "DIA", "AAPL", "NVDA"], index=0)
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m"], index=1)
    show_secondary = st.toggle("Show Secondary Ticker", value=True)
    secondary_ticker = st.selectbox("Secondary Ticker", ["QQQ", "SPY", "IWM", "DIA", "AAPL", "NVDA"], index=0)
    auto_refresh = st.toggle("Auto Refresh 30s", value=False)

if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='30'>",
        unsafe_allow_html=True
    )

tickers = [selected_ticker]
if show_secondary and secondary_ticker != selected_ticker:
    tickers.append(secondary_ticker)

# ----------------------------
# HEADER
# ----------------------------
primary_sig = generate_signal(selected_ticker, interval=interval)

st.markdown('<div class="command-header">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="title-row">
        <div>
            <div class="main-title">LOCKOUT SIGNALS COMMAND CENTER</div>
            <div class="subtitle">Titanic 2077 • Battle Map V3 • ORB + Premarket + Bias Zones + Rolling Feed</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

chip_session = primary_sig.get("session_status", "N/A") if "error" not in primary_sig else "N/A"
chip_state = primary_sig.get("market_state", "N/A") if "error" not in primary_sig else "N/A"
chip_price = f"{primary_sig.get('price', 0):.2f}" if "error" not in primary_sig else "N/A"

st.markdown(
    f"""
    <div class="top-chip-bar">
        <div class="top-chip">Ticker: {selected_ticker}</div>
        <div class="top-chip">Timeframe: {interval}</div>
        <div class="top-chip">Session: {chip_session}</div>
        <div class="top-chip">Market State: {chip_state}</div>
        <div class="top-chip">Price: {chip_price}</div>
    </div>
    """,
    unsafe_allow_html=True
)

if "error" not in primary_sig:
    st.markdown(
        f"""
        <div class="live-feed">
            <div class="live-feed-title">Live Signal Bar</div>
            <div>
                🚨 {selected_ticker}: {primary_sig['signal']} &nbsp;&nbsp;|&nbsp;&nbsp;
                ⚠ Warning: {primary_sig['warning_line']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                🛑 Invalidation: {primary_sig['invalidation']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                🎯 Likely Up: {primary_sig['likely_up']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                🎯 Likely Down: {primary_sig['likely_down']:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

top_btn1, top_btn2 = st.columns([1, 5])
with top_btn1:
    if st.button("Refresh Now"):
        st.rerun()
with top_btn2:
    st.markdown('<div class="small-note">Opening Range logic becomes most useful after 9:35 AM ET.</div>', unsafe_allow_html=True)

# ----------------------------
# MAIN PANELS
# ----------------------------
for ticker in tickers:
    sig = generate_signal(ticker, interval=interval)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{ticker} Tactical Deck</div>', unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    with row1_col1:
        metric_box("Price", f"${sig['price']:.2f}")
    with row1_col2:
        metric_box("Bias", sig["bias"])
    with row1_col3:
        metric_box("Market State", sig["market_state"])
    with row1_col4:
        metric_box("Conviction", sig["conviction"])

    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    with row2_col1:
        metric_box("Signal", sig["signal"])
    with row2_col2:
        metric_box("Pressure", f"{sig['pressure']}/100")
    with row2_col3:
        metric_box("Heat", sig["heat"])
    with row2_col4:
        metric_box("Session", sig["session_status"])

    st.markdown(
        banner_html(
            sig["price"],
            sig["calls_favored_above"],
            sig["puts_favored_below"],
            sig["likely_up"],
            sig["likely_down"]
        ),
        unsafe_allow_html=True
    )

    left, mid, right = st.columns([1, 1, 1])

    with left:
        st.markdown('<div class="section-label">Market Structure</div>', unsafe_allow_html=True)
        metric_box("RSI", f"{sig['rsi']:.2f}")
        metric_box("EMA9", fmt_level(sig["ema9"]))
        metric_box("EMA20", fmt_level(sig["ema20"]))
        metric_box("VWAP", fmt_level(sig["vwap"]))

        st.markdown('<div class="section-label">Premarket</div>', unsafe_allow_html=True)
        metric_box("Premarket High", fmt_level(sig["pm_high"]))
        metric_box("Premarket Low", fmt_level(sig["pm_low"]))

    with mid:
        st.markdown('<div class="section-label">Opening Range</div>', unsafe_allow_html=True)
        metric_box("OR High", fmt_level(sig["or_high"]))
        metric_box("OR Low", fmt_level(sig["or_low"]))

        st.markdown('<div class="section-label">Battle Zones</div>', unsafe_allow_html=True)
        metric_box("Calls Favored Above", fmt_level(sig["calls_favored_above"]))
        metric_box("Puts Favored Below", fmt_level(sig["puts_favored_below"]))
        metric_box("Chop Zone", f"{sig['chop_low']:.2f} - {sig['chop_high']:.2f}")
        metric_box("Warning Line", fmt_level(sig["warning_line"]))
        metric_box("Invalidation", fmt_level(sig["invalidation"]))

    with right:
        st.markdown('<div class="section-label">Targets</div>', unsafe_allow_html=True)
        metric_box("Likely Up", fmt_level(sig["likely_up"]))
        metric_box("Stretch Up", fmt_level(sig["stretch_up"]))
        metric_box("Likely Down", fmt_level(sig["likely_down"]))
        metric_box("Stretch Down", fmt_level(sig["stretch_down"]))

        st.markdown('<div class="section-label">Rolling Signal Feed</div>', unsafe_allow_html=True)
        for item in sig["feed"]:
            st.markdown(f'<div class="feed-item">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">AI Commentary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="commentary-box">{sig["commentary"]}</div>', unsafe_allow_html=True)

    cbtn1, cbtn2 = st.columns([1, 4])
    with cbtn1:
        if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
            df = get_data(ticker, interval=interval)
            if df.empty:
                st.warning("No chart data available.")
            else:
                df = add_indicators(df)
                cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df.columns]
                st.line_chart(df[cols], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)