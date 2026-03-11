import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals Command Center",
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
            radial-gradient(circle at top left, rgba(0,255,180,0.08), transparent 22%),
            radial-gradient(circle at top right, rgba(0,165,255,0.10), transparent 24%),
            radial-gradient(circle at bottom center, rgba(255,0,128,0.06), transparent 25%),
            linear-gradient(180deg, #020306 0%, #050913 40%, #07101a 100%);
        color: #f4f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1220 0%, #0c1525 100%);
    }

    .page-title {
        font-size: 3rem;
        font-weight: 1000;
        line-height: 1.02;
        color: #ffffff;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
        text-shadow:
            0 0 12px rgba(255,255,255,0.06),
            0 0 18px rgba(0,180,255,0.10);
    }

    .page-subtitle {
        font-size: 1rem;
        color: #9eb6dc;
        margin-bottom: 1rem;
    }

    .chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }

    .chip {
        padding: 9px 14px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(22,31,52,0.95), rgba(12,18,31,0.95));
        border: 1px solid rgba(118,152,220,0.18);
        color: #ecf4ff;
        font-weight: 800;
        font-size: 0.82rem;
        box-shadow:
            0 6px 18px rgba(0,0,0,0.24),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .marquee-shell {
        background: linear-gradient(180deg, rgba(10,18,31,0.98), rgba(11,20,36,0.98));
        border: 1px solid rgba(90, 140, 255, 0.18);
        border-radius: 18px;
        overflow: hidden;
        margin-bottom: 18px;
        box-shadow:
            0 10px 30px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .marquee-track {
        white-space: nowrap;
        overflow: hidden;
        padding: 12px 0;
    }

    .marquee-text {
        display: inline-block;
        padding-left: 100%;
        animation: marquee 22s linear infinite;
        font-weight: 900;
        color: #7cffc3;
        font-size: 0.98rem;
        letter-spacing: 0.5px;
        text-shadow: 0 0 12px rgba(0,255,180,0.14);
    }

    @keyframes marquee {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .deck {
        border-radius: 28px;
        padding: 18px;
        margin-bottom: 24px;
        background:
            linear-gradient(180deg, rgba(11,18,31,0.98), rgba(7,13,23,0.98));
        border: 1px solid rgba(96,126,194,0.18);
        box-shadow:
            0 16px 36px rgba(0,0,0,0.34),
            inset 0 1px 0 rgba(255,255,255,0.03),
            inset 0 -1px 0 rgba(255,255,255,0.02);
    }

    .deck-title {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 1000;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
    }

    .hero-shell {
        border-radius: 26px;
        padding: 20px 18px 18px 18px;
        margin-bottom: 16px;
        background:
            radial-gradient(circle at top center, rgba(0,255,170,0.06), transparent 32%),
            linear-gradient(180deg, rgba(18,27,46,0.98), rgba(9,15,26,0.98));
        border: 1px solid rgba(98,129,194,0.20);
        box-shadow:
            0 18px 42px rgba(0,0,0,0.34),
            inset 0 1px 0 rgba(255,255,255,0.04),
            inset 0 -1px 0 rgba(255,255,255,0.02);
    }

    .hero-topline {
        color: #b5c8ea;
        font-size: 0.85rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 8px;
        text-align: center;
    }

    .hero-price {
        font-size: 4.0rem;
        line-height: 1;
        font-weight: 1000;
        text-align: center;
        margin-bottom: 6px;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255,255,255,0.06);
    }

    .hero-price-bull {
        color: #31f095;
        text-shadow:
            0 0 14px rgba(49,240,149,0.20),
            0 0 28px rgba(49,240,149,0.10);
    }

    .hero-price-bear {
        color: #ff5b78;
        text-shadow:
            0 0 14px rgba(255,91,120,0.20),
            0 0 28px rgba(255,91,120,0.10);
    }

    .hero-price-neutral {
        color: #ffd45e;
        text-shadow:
            0 0 14px rgba(255,212,94,0.18),
            0 0 28px rgba(255,212,94,0.08);
    }

    .hero-signal {
        text-align: center;
        font-size: 2rem;
        font-weight: 1000;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }

    .hero-signal-bull { color: #31f095; }
    .hero-signal-bear { color: #ff5b78; }
    .hero-signal-neutral { color: #ffd45e; }

    .pill-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 12px;
    }

    .pill {
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 900;
        font-size: 0.92rem;
        background: linear-gradient(180deg, rgba(20,29,49,0.96), rgba(12,18,30,0.96));
        border: 1px solid rgba(114,149,220,0.18);
        box-shadow:
            0 10px 22px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pill-green {
        color: #dfffee;
        border-color: rgba(55,255,166,0.30);
        box-shadow:
            0 10px 22px rgba(0,0,0,0.22),
            0 0 12px rgba(0,255,170,0.08),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pill-red {
        color: #ffe5ea;
        border-color: rgba(255,91,121,0.28);
        box-shadow:
            0 10px 22px rgba(0,0,0,0.22),
            0 0 12px rgba(255,91,121,0.08),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .pill-gold {
        color: #fff3cc;
        border-color: rgba(255,219,78,0.25);
        box-shadow:
            0 10px 22px rgba(0,0,0,0.22),
            0 0 12px rgba(255,219,78,0.06),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .expected-block {
        text-align: center;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .expected-title {
        color: #b3c5e5;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .expected-line {
        font-size: 1rem;
        font-weight: 900;
        color: #f5fbff;
        line-height: 1.4;
    }

    .expected-green { color: #33ef98; }
    .expected-red { color: #ff6e88; }
    .expected-gold { color: #ffd76a; }

    .subtext {
        text-align: center;
        color: #c7d5eb;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 10px;
    }

    .section-label {
        color: #88b1ff;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 1000;
        margin: 8px 0 8px 0;
    }

    .sunken-wrap {
        border-radius: 20px;
        padding: 12px;
        background: linear-gradient(180deg, rgba(5,9,16,0.98), rgba(8,14,24,0.98));
        border: 1px solid rgba(64, 89, 142, 0.16);
        box-shadow:
            inset 0 10px 18px rgba(0,0,0,0.34),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(18,27,45,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(110,145,220,0.12);
        border-radius: 16px;
        padding: 10px 12px;
        margin-bottom: 8px;
        box-shadow:
            0 10px 20px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .metric-label {
        color: #8da6d3;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        font-weight: 900;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.02rem;
        font-weight: 1000;
    }

    .metric-green .metric-value { color: #33ef98; }
    .metric-red .metric-value { color: #ff6e88; }
    .metric-gold .metric-value { color: #ffd76a; }
    .metric-blue .metric-value { color: #91c9ff; }

    .feed-item {
        background: linear-gradient(180deg, rgba(17,25,42,0.96), rgba(10,16,28,0.96));
        border: 1px solid rgba(100,132,199,0.12);
        border-radius: 14px;
        padding: 9px 11px;
        margin-bottom: 7px;
        color: #edf5ff;
        font-size: 0.88rem;
        font-weight: 800;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .commentary-box {
        background: linear-gradient(180deg, rgba(14,24,40,0.98), rgba(9,15,26,0.98));
        border-left: 5px solid #43c2ff;
        border-radius: 14px;
        padding: 14px;
        color: #e1ecff;
        line-height: 1.55;
        box-shadow:
            0 12px 24px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .small-note {
        color: #748aad;
        font-size: 0.84rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 15px;
        border: 1px solid rgba(118,157,232,0.20);
        background: linear-gradient(180deg, #15213b 0%, #0d1728 100%);
        color: #f7fbff;
        font-weight: 1000;
        padding: 0.58rem 1rem;
        box-shadow:
            0 10px 18px rgba(0,0,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.03);
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
def fmt(x):
    return f"{x:.2f}" if x is not None else "N/A"

def metric_box(label: str, value: str, tone: str = "blue"):
    tone_class = {
        "green": "metric-green",
        "red": "metric-red",
        "gold": "metric-gold",
        "blue": "metric-blue",
    }.get(tone, "metric-blue")

    st.markdown(
        f"""
        <div class="metric-card {tone_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def hero_price_class(bias):
    if bias == "BULLISH":
        return "hero-price hero-price-bull"
    if bias == "BEARISH":
        return "hero-price hero-price-bear"
    return "hero-price hero-price-neutral"

def hero_signal_class(bias):
    if bias == "BULLISH":
        return "hero-signal hero-signal-bull"
    if bias == "BEARISH":
        return "hero-signal hero-signal-bear"
    return "hero-signal hero-signal-neutral"

def bias_tone(bias):
    if bias == "BULLISH":
        return "pill pill-green"
    if bias == "BEARISH":
        return "pill pill-red"
    return "pill pill-gold"

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Command Controls")
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

st.markdown('<div class="page-title">Lockout Signals • Command Center</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Premium live terminal • raised cards • sunken panels • giant hero price • mobile-first</div>',
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
    tape_text = " • ".join([
        f"{selected_ticker} {primary_sig['signal']}",
        f"BIAS {primary_sig['bias']}",
        f"CALLS>{primary_sig['calls_favored_above']:.2f}",
        f"PUTS<{primary_sig['puts_favored_below']:.2f}",
        f"WARNING {primary_sig['warning_line']:.2f}",
        f"INVALID {primary_sig['invalidation']:.2f}",
        f"STATE {primary_sig['market_state']}",
        f"HEAT {primary_sig['heat']}",
    ])
    st.markdown(
        f"""
        <div class="marquee-shell">
            <div class="marquee-track">
                <div class="marquee-text">⚡ {tape_text} ⚡ {tape_text}</div>
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
    st.markdown('<div class="small-note">Opening Range logic becomes strongest after 9:35 AM ET.</div>', unsafe_allow_html=True)

# -------------------------------------------------
# DECKS
# -------------------------------------------------
for ticker in tickers:
    sig = generate_signal(ticker, interval=interval)

    st.markdown('<div class="deck">', unsafe_allow_html=True)
    st.markdown(f'<div class="deck-title">{ticker}</div>', unsafe_allow_html=True)

    if "error" in sig:
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    # HERO
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-topline">{ticker} • {interval} tactical brain</div>
            <div class="{hero_price_class(sig['bias'])}">{sig['price']:.2f}</div>
            <div class="{hero_signal_class(sig['bias'])}">{sig['signal']}</div>

            <div class="pill-row">
                <div class="{bias_tone(sig['bias'])}">🎯 {sig['bias']}</div>
                <div class="pill pill-green">🛰 {sig['market_state']}</div>
                <div class="pill pill-gold">📊 SCORE: {sig['pressure']}/100</div>
                <div class="pill pill-gold">🔥 {sig['heat']}</div>
            </div>

            <div class="expected-block">
                <div class="expected-title">Expected Move (from here)</div>
                <div class="expected-line">
                    <span class="expected-green">LIKELY {sig['likely_up']:.2f}</span>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <span class="expected-gold">WARNING {sig['warning_line']:.2f}</span>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <span class="expected-red">INVALID {sig['invalidation']:.2f}</span>
                </div>
            </div>

            <div class="subtext">{sig["commentary"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # MID LAYOUT
    left, middle, right = st.columns([1, 1, 1])

    with left:
        st.markdown('<div class="section-label">Bias + Structure</div>', unsafe_allow_html=True)
        st.markdown('<div class="sunken-wrap">', unsafe_allow_html=True)
        metric_box("Bias", sig["bias"], "green" if sig["bias"] == "BULLISH" else "red" if sig["bias"] == "BEARISH" else "gold")
        metric_box("Signal", sig["signal"], "blue")
        metric_box("Market State", sig["market_state"], "blue")
        metric_box("RSI", f"{sig['rsi']:.2f}", "blue")
        metric_box("EMA9", fmt(sig["ema9"]), "blue")
        if "ema20" in sig:
            metric_box("EMA20", fmt(sig["ema20"]), "blue")
        metric_box("VWAP", fmt(sig["vwap"]), "gold")
        st.markdown('</div>', unsafe_allow_html=True)

    with middle:
        st.markdown('<div class="section-label">Battle Zones</div>', unsafe_allow_html=True)
        st.markdown('<div class="sunken-wrap">', unsafe_allow_html=True)
        metric_box("Calls Favored Above", fmt(sig["calls_favored_above"]), "green")
        metric_box("Puts Favored Below", fmt(sig["puts_favored_below"]), "red")
        metric_box("Chop Zone", f"{sig['chop_low']:.2f} - {sig['chop_high']:.2f}", "gold")
        metric_box("Warning Line", fmt(sig["warning_line"]), "gold")
        metric_box("Invalidation", fmt(sig["invalidation"]), "red")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Targets + Session</div>', unsafe_allow_html=True)
        st.markdown('<div class="sunken-wrap">', unsafe_allow_html=True)
        metric_box("Likely Up", fmt(sig["likely_up"]), "green")
        metric_box("Stretch Up", fmt(sig["stretch_up"]), "green")
        metric_box("Likely Down", fmt(sig["likely_down"]), "red")
        metric_box("Stretch Down", fmt(sig["stretch_down"]), "red")
        metric_box("Session", sig["session_status"], "blue")
        metric_box("Conviction", sig["conviction"], "blue")
        st.markdown('</div>', unsafe_allow_html=True)

    # SESSION LEVELS
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        metric_box("Premarket High", fmt(sig["pm_high"]), "blue")
    with s2:
        metric_box("Premarket Low", fmt(sig["pm_low"]), "blue")
    with s3:
        metric_box("OR High", fmt(sig["or_high"]), "blue")
    with s4:
        metric_box("OR Low", fmt(sig["or_low"]), "blue")

    # FEED + COMMENTARY
    feed_col, comm_col = st.columns([1, 2])

    with feed_col:
        st.markdown('<div class="section-label">Rolling Feed</div>', unsafe_allow_html=True)
        for item in sig.get("feed", []):
            st.markdown(f'<div class="feed-item">{item}</div>', unsafe_allow_html=True)

    with comm_col:
        st.markdown('<div class="section-label">Commentary</div>', unsafe_allow_html=True)
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