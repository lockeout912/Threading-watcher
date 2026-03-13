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
        margin-bottom: 0.15rem;
        line-height: 1.02;
        text-shadow: 0 0 20px rgba(0,180,255,0.10);
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
        border-radius: 22px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow:
            0 18px 30px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,0.04);
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
        font-size: 1.55rem;
        font-weight: 1000;
        margin-bottom: 0.45rem;
        letter-spacing: 1px;
    }

    .price-box {
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 10px 18px rgba(0,0,0,0.18);
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
        font-size: 3.2rem;
        font-weight: 1000;
        line-height: 0.95;
        margin: 0;
    }

    .green { color: #4df0a5; }
    .red { color: #ff6f8e; }
    .gold { color: #ffd86a; }
    .blue { color: #8dc8ff; }
    .white { color: #f4f8ff; }

    .signal-box {
        font-size: 1.05rem;
        font-weight: 1000;
        padding: 11px 14px;
        border-radius: 14px;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 18px rgba(0,0,0,0.15);
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
        animation: ticker-scroll 22s linear infinite;
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

def pressure_bar(value):
    try:
        v = max(0, min(int(float(value)), 100))
    except Exception:
        v = 0
    st.progress(v / 100.0, text=f"Momentum Pressure: {v}/100")

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def feed_colorize(items):
    out = []
    for item in items:
        t = item.lower()
        if any(x in t for x in ["bull", "calls", "above", "surge", "long"]):
            out.append(f'<span class="green">▲ {item}</span>')
        elif any(x in t for x in ["bear", "puts", "below", "breakdown", "short"]):
            out.append(f'<span class="red">▼ {item}</span>')
        elif any(x in t for x in ["chop", "warning", "battle zone"]):
            out.append(f'<span class="gold">• {item}</span>')
        else:
            out.append(f'<span class="blue">• {item}</span>')
    return " &nbsp;&nbsp; | &nbsp;&nbsp; ".join(out)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">Lockout Signals</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Single-ticker tactical board • upgraded UI layer • backend brain untouched</div>', unsafe_allow_html=True)

top1, top2, top3 = st.columns([1.25, 1.4, 2.5])
with top1:
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()
with top2:
    show_charts = st.toggle("Show Charts", value=False)
with top3:
    st.markdown('<div class="tiny-note">Aggressive / Full Send are display modes for now so we don’t mess with the signal brain.</div>', unsafe_allow_html=True)

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

if day_pct is None:
    day_text = "Day %: N/A"
else:
    sign = "+" if day_pct >= 0 else ""
    day_text = f"{sign}{day_pct:.2f}%"

feed_items = safe_get(sig, "feed", [])
if isinstance(feed_items, list) and feed_items:
    feed_html = feed_colorize(feed_items[:6])
else:
    feed_html = f'<span class="blue">• {selected_ticker} tactical feed online</span>'

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

hero_top_left, hero_top_right = st.columns([3, 1])
with hero_top_left:
    st.markdown(
        f'<div class="ticker-line {bias_cls}">{selected_ticker} • 5M Tactical Brain</div>',
        unsafe_allow_html=True
    )
with hero_top_right:
    st.metric("Day %", day_text)

st.markdown(f'<div class="{price_box_class(bias)}">', unsafe_allow_html=True)
price_left, price_right = st.columns([3, 1])
with price_left:
    st.markdown(
        f'<div class="hero-price {bias_cls}">${fmt_num(safe_get(sig, "price"))}</div>',
        unsafe_allow_html=True
    )
with price_right:
        st.markdown(
            f"""
            <div class="mode-chip {'green' if mode == 'Aggressive' else 'gold'}">{mode.upper()}</div>
            """,
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="signal-box {signal_class(safe_get(sig, "signal", "N/A"))}">{safe_get(sig, "signal", "N/A")}</div>',
    unsafe_allow_html=True
)

chip_html = f"""
<span class="mode-chip">{bias}</span>
<span class="mode-chip">{safe_get(sig, "market_state", safe_get(sig, "regime", "N/A"))}</span>
<span class="mode-chip">Heat: {safe_get(sig, "heat", "N/A")}</span>
<span class="mode-chip">Conviction: {safe_get(sig, "conviction", "N/A")}</span>
"""
st.markdown(chip_html, unsafe_allow_html=True)

pressure_bar(safe_get(sig, "pressure", 0))

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# BODY LAYOUT
# -----------------------------
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Bias + Structure")

    a1, a2 = st.columns(2)
    with a1:
        st.metric("Bias", bias)
        st.metric("RSI", fmt_num(safe_get(sig, "rsi")))
        st.metric("EMA9", fmt_num(safe_get(sig, "ema9")))
        st.metric("VWAP", fmt_num(safe_get(sig, "vwap")))
    with a2:
        st.metric("State", safe_get(sig, "market_state", safe_get(sig, "regime", "N/A")))
        st.metric("EMA20", fmt_num(safe_get(sig, "ema20")))
        st.metric("Session", safe_get(sig, "session_status", "N/A"))
        st.metric("Price", fmt_num(safe_get(sig, "price")))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    section_title("Commentary")
    commentary = safe_get(sig, "commentary", "")
    if commentary:
        st.write(commentary)
    else:
        st.write("No commentary available.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
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