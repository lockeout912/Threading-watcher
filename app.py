import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="LOCKOUT NUCLEAR TERMINAL",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 12% 10%, rgba(0,255,170,0.12), transparent 18%),
            radial-gradient(circle at 88% 10%, rgba(0,180,255,0.14), transparent 20%),
            radial-gradient(circle at 50% 100%, rgba(255,0,120,0.08), transparent 24%),
            linear-gradient(180deg, #010306 0%, #050b14 45%, #08111d 100%);
        color: #f5fbff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08101c 0%, #0b1422 100%);
    }

    .mega-title {
        font-size: 3.0rem;
        font-weight: 1000;
        letter-spacing: 2px;
        color: #ffffff;
        text-shadow:
            0 0 8px rgba(0,255,255,0.25),
            0 0 18px rgba(0,180,255,0.18),
            0 0 30px rgba(0,255,170,0.10);
        margin-bottom: 0.12rem;
        line-height: 1.0;
    }

    .mega-subtitle {
        color: #9bc0ff;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 16px;
    }

    .chip {
        background: linear-gradient(180deg, rgba(19,29,50,0.98), rgba(11,19,35,0.98));
        border: 1px solid rgba(113,150,225,0.18);
        color: #f1f8ff;
        padding: 9px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 900;
        box-shadow: 0 0 14px rgba(0,0,0,0.24);
    }

    .marquee-wrap {
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(7,16,30,0.98), rgba(12,28,48,0.98));
        border: 1px solid rgba(0, 195, 255, 0.25);
        padding: 10px 0;
        margin-bottom: 16px;
        box-shadow: 0 0 24px rgba(0,0,0,0.30);
    }

    .marquee {
        display: inline-block;
        padding-left: 100%;
        animation: marquee 20s linear infinite;
        color: #eff8ff;
        font-weight: 900;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }

    @keyframes marquee {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
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
        background: linear-gradient(180deg, rgba(8,15,28,0.96), rgba(10,18,31,0.96));
        border: 1px solid rgba(95,131,205,0.16);
        border-radius: 24px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow:
            0 0 28px rgba(0,0,0,0.34),
            inset 0 0 0 1px rgba(255,255,255,0.02);
    }

    .deck-title {
        color: #ffffff;
        font-size: 1.48rem;
        font-weight: 1000;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
    }

    .section-label {
        color: #88b1ff;
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 1000;
        margin: 12px 0 8px 0;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(14,23,42,0.98), rgba(9,17,30,0.98));
        border: 1px solid rgba(120,160,240,0.12);
        border-radius: 16px;
        padding: 13px 14px;
        margin-bottom: 10px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.01);
    }

    .metric-label {
        color: #8ba7d6;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 5px;
        font-weight: 900;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.22rem;
        font-weight: 1000;
    }

    .hero-bull {
        background: linear-gradient(135deg, rgba(0,77,46,0.98), rgba(7,116,67,0.88));
        border: 1px solid rgba(55,255,166,0.40);
        border-radius: 18px;
        padding: 16px;
        margin: 12px 0;
        color: #ddffed;
        font-weight: 1000;
        box-shadow: 0 0 22px rgba(0,180,98,0.18);
    }

    .hero-bear {
        background: linear-gradient(135deg, rgba(84,8,22,0.98), rgba(153,15,42,0.88));
        border: 1px solid rgba(255,91,121,0.38);
        border-radius: 18px;
        padding: 16px;
        margin: 12px 0;
        color: #ffe4e9;
        font-weight: 1000;
        box-shadow: 0 0 22px rgba(180,18,57,0.16);
    }

    .hero-chop {
        background: linear-gradient(135deg, rgba(88,71,7,0.98), rgba(156,122,8,0.88));
        border: 1px solid rgba(255,219,78,0.36);
        border-radius: 18px;
        padding: 16px;
        margin: 12px 0;
        color: #fff4cc;
        font-weight: 1000;
        box-shadow: 0 0 22px rgba(188,146,14,0.14);
    }

    .hero-price {
        font-size: 2.7rem;
        font-weight: 1000;
        line-height: 1;
        color: #ffffff;
        text-shadow: 0 0 16px rgba(0,190,255,0.10);
    }

    .hero-state {
        font-size: 1rem;
        font-weight: 900;
        color: #9dc1ff;
        margin-top: 4px;
    }

    .quick-box {
        border-radius: 18px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .quick-green {
        background: linear-gradient(180deg, rgba(0,62,39,0.95), rgba(0,105,68,0.88));
        border: 1px solid rgba(55,255,166,0.30);
        color: #e6fff1;
    }

    .quick-red {
        background: linear-gradient(180deg, rgba(70,10,18,0.95), rgba(130,18,35,0.88));
        border: 1px solid rgba(255,91,121,0.30);
        color: #ffe6eb;
    }

    .quick-gold {
        background: linear-gradient(180deg, rgba(72,58,8,0.95), rgba(126,99,11,0.88));
        border: 1px solid rgba(255,219,78,0.30);
        color: #fff5d2;
    }

    .quick-title {
        font-size: 0.78rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-weight: 1000;
        opacity: 0.9;
        margin-bottom: 6px;
    }

    .quick-value {
        font-size: 1.45rem;
        font-weight: 1000;
    }

    .commentary-box {
        background: linear-gradient(180deg, rgba(12,22,38,0.98), rgba(10,17,30,0.98));
        border-left: 5px solid #43c2ff;
        border-radius: 12px;
        padding: 14px;
        color: #e1ecff;
        line-height: 1.55;
        margin-top: 8px;
    }

    .feed-item {
        background: linear-gradient(180deg, rgba(12,20,35,0.94), rgba(10,17,30,0.94));
        border: 1px solid rgba(96,131,198,0.12);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: #eff6ff;
        font-size: 0.94rem;
        font-weight: 700;
    }

    .small-note {
        color: #7388af;
        font-size: 0.84rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 14px;
        border: 1px solid rgba(118,157,232,0.22);
        background: linear-gradient(180deg, #14213c 0%, #0c1626 100%);
        color: #f7fbff;
        font-weight: 1000;
        padding: 0.56rem 1rem;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(148,191,255,0.50);
        color: white;
    }

    div[data-baseweb="select"] > div {
        background: rgba(14,22,38,0.96);
        border-radius: 12px;
        border: 1px solid rgba(101,139,215,0.18);
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

def fmt(x):
    return f"{x:.2f}" if x is not None else "N/A"

def banner_html(sig):
    price = sig["price"]
    calls = sig["calls_favored_above"]
    puts = sig["puts_favored_below"]
    likely_up = sig["likely_up"]
    likely_down = sig["likely_down"]

    if price > calls:
        return f'<div class="hero-bull">🚀 BULL CONTROL ACTIVE • CALLS FAVORED ABOVE {calls:.2f} • NEXT LIKELY PUSH {likely_up:.2f}</div>'
    elif price < puts:
        return f'<div class="hero-bear">🩸 BEAR CONTROL ACTIVE • PUTS FAVORED BELOW {puts:.2f} • NEXT LIKELY PUSH {likely_down:.2f}</div>'
    return f'<div class="hero-chop">🟡 CHOP / NEUTRAL TAPE • ACTIVE BATTLE ZONE {puts:.2f} - {calls:.2f}</div>'

def quick_box(title: str, value: str, tone: str):
    tone_class = {
        "green": "quick-box quick-green",
        "red": "quick-box quick-red",
        "gold": "quick-box quick-gold",
    }.get(tone, "quick-box quick-gold")

    st.markdown(
        f"""
        <div class="{tone_class}">
            <div class="quick-title">{title}</div>
            <div class="quick-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Nuclear Controls")
    selected_ticker = st.selectbox("Primary Ticker", ["SPY", "QQQ", "IWM", "DIA", "AAPL", "NVDA", "TSLA"], index=0)
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m"], index=1)
    show_secondary = st.toggle("Show Secondary Ticker", value=True)
    secondary_ticker = st.selectbox("Secondary Ticker", ["QQQ", "SPY", "IWM", "DIA", "AAPL", "NVDA", "TSLA"], index=0)
    auto_refresh = st.toggle("Auto Refresh 30s", value=False)

if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='30'>", unsafe_allow_html=True)

tickers = [selected_ticker]
if show_secondary and secondary_ticker != selected_ticker:
    tickers.append(secondary_ticker)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
primary_sig = generate_signal(selected_ticker, interval=interval)

st.markdown('<div class="mega-title">LOCKOUT SIGNALS • SPY NUCLEAR TERMINAL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mega-subtitle">Neon tape • scrolling signals • bright zones • premium war-room build</div>',
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
        <div class="chip">Market State: {state_text}</div>
        <div class="chip">Conviction: {conviction_text}</div>
        <div class="chip">Live Price: {price_text}</div>
    </div>
    """,
    unsafe_allow_html=True
)

if "error" not in primary_sig:
    tape_text = " ✦ ".join([
        f"{selected_ticker} {primary_sig['signal']}",
        f"CALLS>{primary_sig['calls_favored_above']:.2f}",
        f"PUTS<{primary_sig['puts_favored_below']:.2f}",
        f"WARNING {primary_sig['warning_line']:.2f}",
        f"INVALIDATION {primary_sig['invalidation']:.2f}",
        f"LIKELY UP {primary_sig['likely_up']:.2f}",
        f"LIKELY DOWN {primary_sig['likely_down']:.2f}",
        f"STATE {primary_sig['market_state']}",
        f"HEAT {primary_sig['heat']}",
    ])
    st.markdown(
        f"""
        <div class="marquee-wrap">
            <div class="marquee">⚡ {tape_text} ⚡ {tape_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="signal-superbar">
            <div class="signal-superbar-title">Primary Signal Command Line</div>
            <div>
                🚨 {primary_sig['signal']} &nbsp;&nbsp;|&nbsp;&nbsp;
                ⚠ Warning {primary_sig['warning_line']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                🛑 Invalidation {primary_sig['invalidation']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                🎯 Likely Up {primary_sig['likely_up']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                🎯 Likely Down {primary_sig['likely_down']:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

refresh_col, note_col = st.columns([1, 4])
with refresh_col:
    if st.button("Refresh Now"):
        st.rerun()
with note_col:
    st.markdown('<div class="small-note">Opening Range logic is strongest after 9:35 AM ET.</div>', unsafe_allow_html=True)

# -------------------------------------------------
# DECKS
# -------------------------------------------------
for ticker in tickers:
    sig = generate_signal(ticker, interval=interval)

    st.markdown('<div class="deck">', unsafe_allow_html=True)
    st.markdown(f'<div class="deck-title">{ticker} Tactical Deck</div>', unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    top_row = st.columns([1.2, 1, 1, 1])
    with top_row[0]:
        st.markdown(f'<div class="hero-price">${sig["price"]:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-state">{ticker} • {sig["market_state"]}</div>', unsafe_allow_html=True)
    with top_row[1]:
        quick_box("Bias", sig["bias"], "green" if sig["bias"] == "BULLISH" else "red" if sig["bias"] == "BEARISH" else "gold")
    with top_row[2]:
        quick_box("Heat", sig["heat"], "green" if sig["heat"] == "HOT" else "gold")
    with top_row[3]:
        quick_box("Conviction", sig["conviction"], "green" if "HIGH" in sig["conviction"] else "gold")

    st.markdown(banner_html(sig), unsafe_allow_html=True)

    left, middle, right = st.columns([1, 1, 1])

    with left:
        st.markdown('<div class="section-label">Tape Read</div>', unsafe_allow_html=True)
        metric_box("Signal", sig["signal"])
        metric_box("Pressure", f"{sig['pressure']}/100")
        metric_box("Session", sig["session_status"])

        st.markdown('<div class="section-label">Core Structure</div>', unsafe_allow_html=True)
        metric_box("RSI", f"{sig['rsi']:.2f}")
        metric_box("EMA9", fmt(sig["ema9"]))
        if "ema20" in sig:
            metric_box("EMA20", fmt(sig["ema20"]))
        metric_box("VWAP", fmt(sig["vwap"]))

    with middle:
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

    with right:
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

    with st.expander(f"Open {ticker} Chart Deck"):
        df = get_data(ticker, interval=interval)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df.columns]
            st.line_chart(df[cols], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)