import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(page_title="Lockout Signals", layout="wide")

st.title("Lockout Signals • SPY / QQQ")
st.caption("Battle Map v1 • live price, bias, trade map, and chart")

tickers = ["SPY", "QQQ"]

for ticker in tickers:
    st.subheader(ticker)

    sig = generate_signal(ticker)

    if "error" in sig:
        st.error(sig["error"])
        continue

    st.write(f"**Price:** ${sig['price']:.2f}")
    st.write(f"**Bias:** {sig['bias']}")
    st.write(f"**Signal:** {sig['signal']}")
    st.write(f"**Pressure:** {sig['pressure']}/100")
    st.write(f"**Heat:** {sig['heat']}")
    st.write(f"**RSI:** {sig['rsi']:.2f}")
    st.write(f"**EMA9:** {sig['ema9']:.2f}")
    st.write(f"**VWAP:** {sig['vwap']:.2f}")

    # -------- Trade Map --------
    price = sig["price"]
    ema = sig["ema9"]
    vwap = sig["vwap"]

    calls_level = max(ema, vwap)
    puts_level = min(ema, vwap)

    range_size = abs(ema - vwap) * 2
    if range_size < 0.50:
        range_size = price * 0.003  # fallback so targets aren't too tiny

    likely_up = price + range_size
    likely_down = price - range_size
    invalidation = puts_level - (range_size / 2)
    warning_line = (calls_level + puts_level) / 2

    st.markdown("### Trade Map")
    st.write(f"**Calls favored above:** {calls_level:.2f}")
    st.write(f"**Puts favored below:** {puts_level:.2f}")
    st.write(f"**Chop zone:** {puts_level:.2f} - {calls_level:.2f}")
    st.write(f"**Warning line:** {warning_line:.2f}")
    st.write(f"**Likely upside:** {likely_up:.2f}")
    st.write(f"**Likely downside:** {likely_down:.2f}")
    st.write(f"**Invalidation:** {invalidation:.2f}")

    # -------- Commentary --------
    if price > calls_level:
        st.info(
            f"Bulls have control above {calls_level:.2f}. "
            f"Calls are favored while price holds above that zone. "
            f"Watch for stall or rejection as price approaches {likely_up:.2f}."
        )
    elif price < puts_level:
        st.info(
            f"Bears have control below {puts_level:.2f}. "
            f"Puts are favored while price stays below that zone. "
            f"Watch for bounce attempts or stall near {likely_down:.2f}."
        )
    else:
        st.info(
            f"Price is in chop between {puts_level:.2f} and {calls_level:.2f}. "
            f"No-man's-land. Wait for a clean break before getting aggressive."
        )

    if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
        df = get_data(ticker)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "VWAP"] if c in df.columns]
            st.line_chart(df[cols])

    st.divider()