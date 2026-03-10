import streamlit as st
from agent import generate_signal, get_data, add_indicators

st.set_page_config(page_title="Lockout Signals", layout="wide")

st.title("Lockout Signals • SPY / QQQ")
st.caption("Battle Map v1 • live price, trade map, and chart")

tickers = ["SPY", "QQQ"]

for ticker in tickers:
    st.subheader(ticker)

    sig = generate_signal(ticker)

    if "error" in sig:
        st.error(sig["error"])
        continue

    price = sig["price"]
    ema = sig["ema9"]
    vwap = sig["vwap"]

    st.write(f"Price: ${price:.2f}")
    st.write(f"Bias: {sig['bias']}")
    st.write(f"Signal: {sig['signal']}")
    st.write(f"Pressure: {sig['pressure']}/100")
    st.write(f"Heat: {sig['heat']}")
    st.write(f"RSI: {sig['rsi']:.2f}")
    st.write(f"EMA9: {ema:.2f}")
    st.write(f"VWAP: {vwap:.2f}")

    calls_level = max(ema, vwap)
    puts_level = min(ema, vwap)

    range_size = abs(ema - vwap) * 2
    if range_size < 0.5:
        range_size = price * 0.003

    likely_up = price + range_size
    likely_down = price - range_size
    warning_line = (calls_level + puts_level) / 2
    invalidation = puts_level - (range_size / 2)

    st.markdown("## Trade Map")
    st.write(f"Calls favored above: {calls_level:.2f}")
    st.write(f"Puts favored below: {puts_level:.2f}")
    st.write(f"Chop zone: {puts_level:.2f} - {calls_level:.2f}")
    st.write(f"Warning line: {warning_line:.2f}")
    st.write(f"Likely upside: {likely_up:.2f}")
    st.write(f"Likely downside: {likely_down:.2f}")
    st.write(f"Invalidation: {invalidation:.2f}")

    if price > calls_level:
        st.success(f"BULL CONTROL — calls favored toward {likely_up:.2f}")
    elif price < puts_level:
        st.error(f"BEAR CONTROL — puts favored toward {likely_down:.2f}")
    else:
        st.warning("CHOP ZONE — wait for breakout")

    if st.button(f"Show Chart {ticker}", key=f"chart_{ticker}"):
        df = get_data(ticker)
        if df.empty:
            st.warning("No chart data available.")
        else:
            df = add_indicators(df)
            cols = [c for c in ["Close", "EMA9", "VWAP"] if c in df.columns]
            st.line_chart(df[cols])

    st.divider()