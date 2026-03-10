import time
import requests
import pandas as pd
import numpy as np

print("agent.py loaded")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_yahoo_chart(ticker: str, interval: str = "5m", range_: str = "5d", prepost: bool = True) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "interval": interval,
        "range": range_,
        "includePrePost": "true" if prepost else "false",
        "events": "div,splits",
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()

        data = r.json()
        result = data["chart"]["result"][0]

        timestamps = result.get("timestamp", [])
        quote = result["indicators"]["quote"][0]

        if not timestamps:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        })

        df["Datetime"] = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York")
        df = df.set_index("Datetime")

        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        df["Volume"] = df["Volume"].fillna(0)

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"fetch_yahoo_chart error for {ticker} {interval} {range_}: {e}")
        return pd.DataFrame()


def get_data(ticker, interval="5m", period="5d"):
    attempts = [
        {"interval": interval, "range": period},
        {"interval": "5m", "range": "1d"},
        {"interval": "15m", "range": "5d"},
        {"interval": "30m", "range": "1mo"},
        {"interval": "1d", "range": "3mo"},
    ]

    for attempt in attempts:
        for retry in range(3):
            df = fetch_yahoo_chart(
                ticker=ticker,
                interval=attempt["interval"],
                range_=attempt["range"],
                prepost=True
            )

            if not df.empty:
                df = df.sort_index()
                print(
                    f"{ticker}: loaded {len(df)} rows using interval={attempt['interval']} range={attempt['range']} retry={retry+1}"
                )
                return df

            time.sleep(1)

    print(f"All data attempts failed for {ticker}")
    return pd.DataFrame()


def add_indicators(df):
    if df.empty:
        return df

    df = df.copy()

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tpv = (typical_price * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    df["VWAP"] = cum_tpv / cum_vol

    df["Range"] = df["High"] - df["Low"]
    df["AvgRange10"] = df["Range"].rolling(10).mean()

    return df


def build_battle_map(df, ticker="SPY"):
    df = add_indicators(df)
    latest = df.iloc[-1]
    current = float(latest["Close"])

    ema9 = float(latest["EMA9"]) if pd.notna(latest["EMA9"]) else current
    vwap = float(latest["VWAP"]) if pd.notna(latest["VWAP"]) else current
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 50.0

    # Use recent bars to define a live battlefield
    recent = df.tail(12).copy()  # roughly last hour on 5m
    recent_high = float(recent["High"].max())
    recent_low = float(recent["Low"].min())
    range_size = max(recent_high - recent_low, 0.01)

    # Control zones
    bull_control = max(vwap, ema9)
    bear_control = min(vwap, ema9)

    # A small no-man's-land / chop band around the midpoint
    midpoint = (recent_high + recent_low) / 2
    chop_half_band = max(range_size * 0.10, current * 0.0015)

    chop_low = midpoint - chop_half_band
    chop_high = midpoint + chop_half_band

    # Direction bias
    if current > bull_control and ema9 >= vwap:
        bias = "BULLISH"
    elif current < bear_control and ema9 <= vwap:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    # Pressure
    pressure = 50
    if rsi > 60:
        pressure += 15
    elif rsi < 40:
        pressure -= 15

    if current > ema9:
        pressure += 10
    else:
        pressure -= 10

    if current > vwap:
        pressure += 10
    else:
        pressure -= 10

    pressure = max(0, min(100, pressure))
    heat = "HOT" if pressure >= 70 else "WARM" if pressure >= 45 else "COLD"

    # Regime
    avg_range = float(latest["AvgRange10"]) if pd.notna(latest["AvgRange10"]) else range_size / 3
    if range_size < current * 0.004:
        regime = "CHOP"
    elif avg_range > 0 and latest["Range"] > avg_range * 1.35:
        regime = "EXPANSION"
    else:
        regime = "TREND"

    # Levels
    calls_favored_above = bull_control
    puts_favored_below = bear_control
    warning_line = ema9 if bias == "BULLISH" else vwap if bias == "BEARISH" else midpoint
    invalidation = recent_low if bias == "BULLISH" else recent_high if bias == "BEARISH" else midpoint

    # Targets
    likely_up = current + range_size * 0.5
    stretch_up = current + range_size * 1.0
    likely_down = current - range_size * 0.5
    stretch_down = current - range_size * 1.0

    if bias == "BULLISH":
        signal = f"CALLS FAVORED above {calls_favored_above:.2f}"
    elif bias == "BEARISH":
        signal = f"PUTS FAVORED below {puts_favored_below:.2f}"
    else:
        signal = "NO CLEAR EDGE - CHOP / WAIT"

    # Commentary
    if bias == "BULLISH":
        commentary = (
            f"Bulls have the ball above {calls_favored_above:.2f}. "
            f"Watch for stall near {likely_up:.2f}. "
            f"Warning if price slips under {warning_line:.2f}. "
            f"Long thesis is in trouble below {invalidation:.2f}."
        )
    elif bias == "BEARISH":
        commentary = (
            f"Bears control below {puts_favored_below:.2f}. "
            f"Watch for stall near {likely_down:.2f}. "
            f"Warning if price reclaims {warning_line:.2f}. "
            f"Short thesis is in trouble above {invalidation:.2f}."
        )
    else:
        commentary = (
            f"Price is in no-man's-land between {chop_low:.2f} and {chop_high:.2f}. "
            f"Wait for a clean break above {calls_favored_above:.2f} or below {puts_favored_below:.2f}."
        )

    return {
        "ticker": ticker,
        "price": current,
        "bias": bias,
        "signal": signal,
        "pressure": pressure,
        "heat": heat,
        "rsi": rsi,
        "ema9": ema9,
        "vwap": vwap,
        "regime": regime,
        "calls_favored_above": calls_favored_above,
        "puts_favored_below": puts_favored_below,
        "chop_low": chop_low,
        "chop_high": chop_high,
        "warning_line": warning_line,
        "invalidation": invalidation,
        "likely_up": likely_up,
        "stretch_up": stretch_up,
        "likely_down": likely_down,
        "stretch_down": stretch_down,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "commentary": commentary,
    }


def generate_signal(ticker="SPY"):
    try:
        df = get_data(ticker)

        if df.empty:
            return {"error": f"No data for {ticker}"}

        return build_battle_map(df, ticker=ticker)

    except Exception as e:
        print(f"generate_signal error for {ticker}: {e}")
        return {"error": f"Signal error: {str(e)}"}