import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals Nuclear Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# STYLE: SPY NUCLEAR TERMINAL
# -------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(0,255,153,0.08), transparent 18%),
            radial-gradient(circle at top right, rgba(0,170,255,0.10), transparent 18%),
            radial-gradient(circle at bottom right, rgba(255,196,0,0.06), transparent 20%),
            linear-gradient(180deg, #020406 0%, #07101a 50%, #08111b 100%);
        color: #f3f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .terminal-title {
        font-size: 2.65rem;
        font-weight: 900;
        letter-spacing: 2px;
        color: #f7fbff;
        text-shadow: 0 0 24px rgba(0, 220, 255, 0.18);
        margin-bottom: 0.15rem;
    }

    .terminal-subtitle {
        color: #8fa8d2;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .ticker-tape {
        background: linear-gradient(90deg, rgba(8,18,30,0.98), rgba(12,24,41,0.98));
        border: 1px solid rgba(99, 144, 255, 0.18);
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 18px;
        overflow: hidden;
        white-space: nowrap;
        box-shadow: 0 0 24px rgba(0,0,0,0.28);
    }

    .ticker-scroll {
        display: inline-block;
        padding-left: 100%;
        animation: ticker-scroll 22s linear infinite;
        font-weight: 800;
        color: #dff1ff;
    }

    @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 14px;
    }

    .chip {
        background: rgba(15, 24, 40, 0.96);
        border: 1px solid rgba(110, 150, 230, 0.18);
        color: #edf5ff;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        box-shadow: 0 0 14px rgba(0,0,0,0.22);
    }

    .signal-superbar {
        background: linear-gradient(90deg, rgba(5,17,30,0.99), rgba(13,31,53,0.99));
        border: 1px solid rgba(0, 194, 255, 0.22);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(0,0,0,0.30);
    }

    .signal-superbar-title {
        color: #62d7ff;
        font-size: 0.78rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .deck {
        background: rgba(9, 16, 29, 0.96);
        border: 1px solid rgba(95, 131, 205, 0.14);
        border-radius: 22px;
        padding: 18px;
        margin-bottom: 24px;
        box-shadow: 0 0 34px rgba(0,0,0,0.34);
    }

    .deck-title {
        color: #ffffff;
        font-size: 1.42rem;
        font-weight: 900;
        letter-spacing: 0.7px;
        margin-bottom: 12px;
    }

    .section-label {
        color: #8ba5cf;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 1.35px;
        font-weight: 900;
        margin: 10px 0 8px 0;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(16,25,44,0.98), rgba(10,17,31,0.98));
        border: 1px solid rgba(118, 156, 232, 0.10);
        border-radius: 15px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #8ca6d3;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 4px;
        font-weight: 800;
    }

    .metric-value {
        color: #f7fbff;
        font-size: 1.16rem;
        font-weight: 900;
    }

    .hero-bull {
        background: linear-gradient(135deg, rgba(0,72,43,0.98), rgba(7,116,67,0.86));
        border: 1px solid rgba(62, 255, 169, 0.38);
        border-radius: 16px;
        padding: 14px;
        color: #dfffee;
        font-weight: 900;
        margin: 10px 0 14px 0;
        box-shadow: 0 0 24px rgba(0, 180, 98, 0.16);
    }

    .hero-bear {
        background: linear-gradient(135deg, rgba(82,8,21,0.98), rgba(138,18,37,0.86));
        border: 1px solid rgba(255, 90, 120, 0.36);
        border-radius: 16px;
        padding: 14px;
        color: #ffe4e8;
        font-weight: 900;
        margin: 10px 0 14px 0;
        box-shadow: 0 0 24px rgba(154, 20, 48, 0.16);
    }

    .hero-chop {
        background: linear-gradient(135deg, rgba(72,56,7,0.98), rgba(120,95,11,0.86));
        border: 1px solid rgba(255, 213, 73, 0.34);
        border-radius: 16px;
        padding: 14px;
        color: #fff4cf;
        font-weight: 900;
        margin: 10px 0 14px 0;
        box-shadow: 0 0 24px rgba(159, 125, 18, 0.14);
    }

    .quick-tape-box {
        background: linear-gradient(180deg, rgba(15,24,42,0.98), rgba(11,18,31,0.98));
        border: 1px solid rgba(98, 137, 210, 0.10);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .quick-tape-title {
        color: #83c6ff;
        font-size: 0.78rem;
        letter-spacing: 1.2px;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .quick-tape-value {
        color: #ffffff;
        font-size: 1.24rem;
        font-weight: 900;
    }

    .commentary-box {
        background: rgba(13, 22, 39, 0.96);
        border-left: 4px solid #4fc2ff;
        border-radius: 12px;
        padding: 12px 14px;
        color: #deebff;
        line-height: 1.5;
        margin-top: 8px;
    }

    .feed-item {
        background: rgba(11, 19, 33, 0.92);
        border: 1px solid rgba(96, 131, 198, 0.12);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #eef5ff;
        font-size: 0.92rem;
    }

    .small-note {
        color: #7388af;
        font-size: 0.84rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid rgba(118, 157, 232, 0.20);
        background: linear-gradient(180deg, #14213b 0%, #0d1828 100%);
        color: #f7fbff;
        font-weight: 900;
        padding: 0.54rem 0.96rem;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(148, 191, 255, 0.48);
        color: white;
    }

    div[data-baseweb="select"] > div {
        background: rgba(14, 22, 38, 0.96);
        border-radius: 12px;
        border: 1px solid rgba(101, 139, 215, 0.18);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
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

def quick_box(title: str, value: str):
    st.markdown(
        f"""
        <div class="quick-tape-box">
            <div class="quick-tape-title">{title}</div>
            <div class="quick-tape-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def fmt(x):
    return f"{x:.2f}" if x is not None else "N/A"

def make_banner(sig):
    price = sig["price"]
    calls = sig["calls_favored_above"]
    puts = sig["puts_favored_below"]
    likely_up = sig["likely_up"]
    likely_down = sig["likely_down"]

    if price > calls:
        return f'<div class="hero-bull">🚀 BULL CONTROL — CALLS FAVORED ABOVE {calls:.2f} → continuation path toward {likely_up:.2f}</div>'
    elif price < puts:
        return f'<div class="hero-bear">🩸 BEAR CONTROL — PUTS FAVORED BELOW {puts:.2f} → continuation path toward {likely_down:.2f}</div>'
    return f'<div class="hero-chop">🟡 CHOP ZONE — WAITING FOR EXPANSION BETWEEN {puts:.2f} AND {calls:.2f}</div>'

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Nuclear Controls")
    selected_ticker = st.selectbox(
        "Primary Ticker",
        ["SPY", "QQQ", "IWM", "DIA", "AAPL", "NVDA", "TSLA"],
        index=0
    )

    interval = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "30m"],
        index=1
    )

    show_secondary = st.toggle("Show Secondary Ticker", value=True)

    secondary_ticker = st.selectbox(
        "Secondary Ticker",
        ["QQQ", "SPY", "IWM", "DIA", "AAPL", "NVDA", "TSLA"],
        index=0
    )

    auto_refresh = st.toggle("Auto Refresh 30s", value=False)

if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='30'>",
        unsafe_allow_html=True
    )

tickers = [selected_ticker]
if show_secondary and secondary_ticker != selected_ticker:
    tickers.append(secondary_ticker)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
primary_sig = generate_signal(selected_ticker, interval=interval)

st.markdown('<div class="terminal-title">LOCKOUT SIGNALS NUCLEAR TERMINAL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="terminal-subtitle">High-speed SPY tape • ORB + Premarket + Bias Zones + Rolling Signal Feed</div>',
    unsafe_allow_html=True
)

session_text = primary_sig.get("session_status", "N/A") if "error" not in primary_sig else "N/A"
state_text = primary_sig.get("market_state", "N/A") if "error" not in primary_sig else "N/A"
conviction_text = primary_sig.get("conviction", "N/A") if "error" not in primary_sig else "N/A"
price_text = f"{primary_sig.get('price', 0):.2f}" if "error" not in primary_sig else "N/A"

st.markdown(
    f"""
    <div class="chip-row">
        <div class="chip">Ticker: {selected_ticker}</div>
        <div class="chip">Timeframe: {interval}</div>
        <div class="chip">Session: {session_text}</div>
        <div class="chip">State: {state_text}</div>
        <div class="chip">Conviction: {conviction_text}</div>
        <div class="chip">Live Price: {price_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)

if "error" not in primary_sig:
    ticker_tape_text = "   ✦   ".join(
        [
            f"{selected_ticker} {primary_sig['signal']}",
            f"WARNING {primary_sig['warning_line']:.2f}",
            f"INVALIDATION {primary_sig['invalidation']:.2f}",
            f"LIKELY UP {primary_sig['likely_up']:.2f}",
            f"LIKELY DOWN {primary_sig['likely_down']:.2f}",
            f"STATE {primary_sig['market_state']}",
            f"HEAT {primary_sig['heat']}",
        ]
    )
    st.markdown(
        f"""
        <div class="ticker-tape">
            <div class="ticker-scroll">⚡ {ticker_tape_text} ⚡ {ticker_tape_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="signal-superbar">
            <div class="signal-superbar-title">Primary Signal Bar</div>
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

top_a, top_b = st.columns([1, 4])
with top_a:
    if st.button("Refresh Now"):
        st.rerun()
with top_b:
    st.markdown('<div class="small-note">Use after 9:35 AM ET for full Opening Range breakout logic.</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MAIN DECKS
# -------------------------------------------------
for ticker in tickers:
    sig = generate_signal(ticker, interval=interval)

    st.markdown('<div class="deck">', unsafe_allow_html=True)
    st.markdown(f'<div class="deck-title">{ticker} Tactical Nuclear Deck</div>', unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        quick_box("Price", f"${sig['price']:.2f}")
    with top2:
        quick_box("Bias", sig["bias"])
    with top3:
        quick_box("Market State", sig["market_state"])
    with top4:
        quick_box("Conviction", sig["conviction"])

    st.markdown(make_banner(sig), unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<div class="section-label">Tape Read</div>', unsafe_allow_html=True)
        metric_box("Signal", sig["signal"])
        metric_box("Pressure", f"{sig['pressure']}/100")
        metric_box("Heat", sig["heat"])
        metric_box("Session", sig["session_status"])

        st.markdown('<div class="section-label">Market Structure</div>', unsafe_allow_html=True)
        metric_box("RSI", f"{sig['rsi']:.2f}")
        metric_box("EMA9", fmt(sig["ema9"]))
        if "ema20" in sig:
            metric_box("EMA20", fmt(sig["ema20"]))
        metric_box("VWAP", fmt(sig["vwap"]))

    with col2:
        st.markdown('<div class="section-label">Battle Zones</div>', unsafe_allow_html=True)
        metric_box("Calls Favored Above", fmt(sig["calls_favored_above"]))
        metric_box("Puts Favored Below", fmt(sig["puts_favored_below"]))
        metric_box("Chop Zone", f"{sig['chop_low']:.2f} - {sig['chop_high']:.2f}")
        metric_box("Warning Line", fmt(sig["warning_line"]))
        metric_box("Invalidation", fmt(sig["invalidation"]))

        st.markdown('<div class="section-label">Session Levels</div>', unsafe_allow_html=True)
        metric_box("Premarket High", fmt(sig["pm_high"]))
        metric_box("Premarket Low", fmt(sig["pm_low"]))
        metric_box("OR High", fmt(sig["or_high"]))
        metric_box("OR Low", fmt(sig["or_low"]))

    with col3:
        st.markdown('<div class="section-label">Targets</div>', unsafe_allow_html=True)
        metric_box("Likely Up", fmt(sig["likely_up"]))
        metric_box("Stretch Up", fmt(sig["stretch_up"]))
        metric_box("Likely Down", fmt(sig["likely_down"]))
        metric_box("Stretch Down", fmt(sig["stretch_down"]))

        st.markdown('<div class="section-label">Rolling Feed</div>', unsafe_allow_html=True)
        for item in sig.get("feed", []):
            st.markdown(f'<div class="feed-item">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">AI Commentary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="commentary-box">{sig["commentary"]}</div>', unsafe_allow_html=True)

    chart_expander = st.expander(f"Open {ticker} Chart Deck")
    with chart_expander:
        df = get_data(ticker, interval=interval)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df.columns]
            st.line_chart(df[cols], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)