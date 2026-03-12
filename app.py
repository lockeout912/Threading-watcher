import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(
    page_title="Lockout Signals",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# SIMPLE SAFE STYLING
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #050913 0%, #08111d 100%);
        color: #f4f8ff;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 900;
        color: white;
        margin-bottom: 0.15rem;
    }

    .subtle {
        color: #9cb1d3;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .hero-card {
        background: linear-gradient(180deg, rgba(18,27,46,0.98), rgba(9,15,26,0.98));
        border: 1px solid rgba(101,139,215,0.18);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.22);
    }

    .section-card {
        background: linear-gradient(180deg, rgba(16,24,40,0.98), rgba(10,16,28,0.98));
        border: 1px solid rgba(101,139,215,0.14);
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.18);
    }

    .ticker-line {
        font-size: 1.4rem;
        font-weight: 900;
        margin-bottom: 0.4rem;
    }

    .price-line {
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.6rem;
    }

    .green { color: #4df0a5; }
    .red { color: #ff6f8e; }
    .gold { color: #ffd86a; }
    .blue { color: #8dc8ff; }

    .signal-box {
        font-size: 1.15rem;
        font-weight: 900;
        padding: 10px 12px;
        border-radius: 12px;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .signal-green {
        background: rgba(77,240,165,0.12);
        border: 1px solid rgba(77,240,165,0.28);
        color: #4df0a5;
    }

    .signal-red {
        background: rgba(255,111,142,0.12);
        border: 1px solid rgba(255,111,142,0.28);
        color: #ff6f8e;
    }

    .signal-gold {
        background: rgba(255,216,106,0.12);
        border: 1px solid rgba(255,216,106,0.28);
        color: #ffd86a;
    }

    .tiny-note {
        color: #7f95b8;
        font-size: 0.82rem;
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

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">Lockout Signals</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Stable tactical board • service-first rollback</div>', unsafe_allow_html=True)

top1, top2, top3 = st.columns([1, 1, 3])
with top1:
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()
with top2:
    show_charts = st.toggle("Show Charts", value=False)
with top3:
    st.markdown('<div class="tiny-note">This version is intentionally simple and stable so your people can actually use it.</div>', unsafe_allow_html=True)

tickers = ["SPY", "QQQ"]

# -----------------------------
# MAIN TICKER PANELS
# -----------------------------
for ticker in tickers:
    sig = generate_signal(ticker, interval="5m")

    st.markdown('<div class="hero-card">', unsafe_allow_html=True)

    if "error" in sig:
        st.markdown(f'<div class="ticker-line">{ticker}</div>', unsafe_allow_html=True)
        st.error(sig["error"])
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    bias = safe_get(sig, "bias", "NEUTRAL")
    bias_cls = bias_color_class(bias)

    # day move from chart data
    df_day = get_data(ticker, interval="5m")
    _, day_pct = calc_day_change(df_day)
    if day_pct is None:
        day_text = "Day %: N/A"
    else:
        sign = "+" if day_pct >= 0 else ""
        day_text = f"{sign}{day_pct:.2f}%"

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            f'<div class="ticker-line {bias_cls}">{ticker} • 5M Tactical Brain</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="price-line {bias_cls}">${fmt_num(safe_get(sig, "price"))}</div>',
            unsafe_allow_html=True
        )
    with h2:
        st.metric("Day %", day_text)

    st.markdown(
        f'<div class="signal-box {signal_class(safe_get(sig, "signal", "N/A"))}">{safe_get(sig, "signal", "N/A")}</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Bias", bias)
    with p2:
        st.metric("State", safe_get(sig, "market_state", safe_get(sig, "regime", "N/A")))
    with p3:
        st.metric("Heat", safe_get(sig, "heat", "N/A"))
    with p4:
        st.metric("Conviction", safe_get(sig, "conviction", "N/A"))

    pressure_bar(safe_get(sig, "pressure", 0))

    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Bias + Structure")
        a1, a2 = st.columns(2)
        with a1:
            st.metric("RSI", fmt_num(safe_get(sig, "rsi")))
            st.metric("EMA9", fmt_num(safe_get(sig, "ema9")))
            st.metric("VWAP", fmt_num(safe_get(sig, "vwap")))
        with a2:
            st.metric("EMA20", fmt_num(safe_get(sig, "ema20")))
            st.metric("Session", safe_get(sig, "session_status", "N/A"))
            st.metric("Price", fmt_num(safe_get(sig, "price")))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Commentary")
        commentary = safe_get(sig, "commentary", "")
        if commentary:
            st.write(commentary)
        else:
            st.write("No commentary available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Battle Zones + Targets")
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
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Session Levels")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("PM High", fmt_num(safe_get(sig, "pm_high")))
        with s2:
            st.metric("PM Low", fmt_num(safe_get(sig, "pm_low")))
        with s3:
            st.metric("OR High", fmt_num(safe_get(sig, "or_high")))
        with s4:
            st.metric("OR Low", fmt_num(safe_get(sig, "or_low")))
        st.markdown("</div>", unsafe_allow_html=True)

    feed = safe_get(sig, "feed", [])
    if isinstance(feed, list) and len(feed) > 0:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Rolling Feed")
        for item in feed[:5]:
            st.write(f"• {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    if show_charts:
        with st.expander(f"Open {ticker} Chart Deck"):
            df_chart = get_data(ticker, interval="5m")
            if df_chart is None or df_chart.empty:
                st.warning("No chart data available.")
            else:
                df_chart = add_indicators(df_chart)
                cols = [c for c in ["Close", "EMA9", "EMA20", "VWAP"] if c in df_chart.columns]
                if cols:
                    st.line_chart(df_chart[cols], use_container_width=True)
                else:
                    st.line_chart(df_chart[["Close"]], use_container_width=True)

    st.divider()