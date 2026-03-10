import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(page_title="Lockout Signals", layout="wide")

st.title("Lockout Signals • SPY / QQQ")
st.caption("Battle Map V2 • ORB + Premarket + Bias Zones")

top1, top2 = st.columns([1, 1])

with top1:
    if st.button("Refresh Now"):
        st.rerun()

with top2:
    st.write("Use after 9:35 AM ET for full Opening Range logic")

tickers = ["SPY", "QQQ"]

for ticker in tickers:
    st.subheader(ticker)

    sig = generate_signal(ticker)

    if "error" in sig:
        st.error(sig["error"])
        continue

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"**Price:** ${sig['price']:.2f}")
        st.write(f"**Bias:** {sig['bias']}")
        st.write(f"**Signal:** {sig['signal']}")
        st.write(f"**Market State:** {sig['market_state']}")
        st.write(f"**Pressure:** {sig['pressure']}/100")
        st.write(f"**Heat:** {sig['heat']}")

    with c2:
        st.write(f"**RSI:** {sig['rsi']:.2f}")
        st.write(f"**EMA9:** {sig['ema9']:.2f}")
        st.write(f"**VWAP:** {sig['vwap']:.2f}")
        st.write(f"**Premarket High:** {sig['pm_high']:.2f}" if sig["pm_high"] is not None else "**Premarket High:** N/A")
        st.write(f"**Premarket Low:** {sig['pm_low']:.2f}" if sig["pm_low"] is not None else "**Premarket Low:** N/A")

    st.markdown("### Opening Range")
    st.write(f"**OR High:** {sig['or_high']:.2f}" if sig["or_high"] is not None else "**OR High:** Waiting for 9:35 ET")
    st.write(f"**OR Low:** {sig['or_low']:.2f}" if sig["or_low"] is not None else "**OR Low:** Waiting for 9:35 ET")

    st.markdown("### Battle Zones")
    st.write(f"**Calls favored above:** {sig['calls_favored_above']:.2f}")
    st.write(f"**Puts favored below:** {sig['puts_favored_below']:.2f}")
    st.write(f"**Chop zone:** {sig['chop_low']:.2f} - {sig['chop_high']:.2f}")
    st.write(f"**Warning line:** {sig['warning_line']:.2f}")
    st.write(f"**Invalidation:** {sig['invalidation']:.2f}")

    st.markdown("### Targets")
    st.write(f"**Likely up:** {sig['likely_up']:.2f}")
    st.write(f"**Stretch up:** {sig['stretch_up']:.2f}")
    st.write(f"**Likely down:** {sig['likely_down']:.2f}")
    st.write(f"**Stretch down:** {sig['stretch_down']:.2f}")

    st.markdown("### Commentary")
    st.info(sig["commentary"])

    if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
        df = get_data(ticker)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "VWAP"] if c in df.columns]
            st.line_chart(df[cols])

    st.divider()