import time
import requests
import pandas as pd
import numpy as np

print("agent.py loaded")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def interval_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "1d": 86400,
    }
    return mapping.get(interval, 300)


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

        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        df["Volume"] = df["Volume"].fillna(0)

        return df[required]

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

    return df


def generate_signal(ticker="SPY"):
    try:
        df = get_data(ticker)

        if df.empty:
            return {"error": f"No data for {ticker}"}

        df = add_indicators(df)
        latest = df.iloc[-1]

        price = float(latest["Close"])
        ema9 = float(latest["EMA9"]) if pd.notna(latest["EMA9"]) else price
        vwap = float(latest["VWAP"]) if pd.notna(latest["VWAP"]) else price
        rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 50.0

        if price > ema9 and ema9 > vwap:
            bias = "BULLISH"
            signal = f"CALLS BIAS @ {price:.2f}"
        elif price < ema9 and ema9 < vwap:
            bias = "BEARISH"
            signal = f"PUTS BIAS @ {price:.2f}"
        else:
            bias = "NEUTRAL"
            signal = "No Clear Signal"

        pressure = 50
        if rsi > 60:
            pressure += 20
        elif rsi < 40:
           