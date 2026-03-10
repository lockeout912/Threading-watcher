import time
import yfinance as yf
import pandas as pd
import numpy as np

print("agent.py loaded")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def fetch_history(ticker, interval, period, prepost=True):
    """
    More reliable Yahoo fetch path for Streamlit Cloud.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(interval=interval, period=period, prepost=prepost, auto_adjust=False)
        df = flatten_columns(df)

        if df is None or df.empty:
            return pd.DataFrame()

        return df
    except Exception as e:
        print(f"fetch_history error for {ticker} interval={interval} period={period}: {e}")
        return pd.DataFrame()


def get_data(ticker, interval="5m", period="5d"):
    """
    Fetch market data with retries + fallbacks.
    """
    attempts = [
        {"interval": interval, "period": period},
        {"interval": "5m", "period": "1d"},
        {"interval": "15m", "period": "5d"},
        {"interval": "30m", "period": "1mo"},
        {"interval": "1d", "period": "3mo"},
    ]

    for attempt in attempts:
        for retry in range(3):
            try:
                df = fetch_history(
                    ticker=ticker,
                    interval=attempt["interval"],
                    period=attempt["period"],
                    prepost=True
                )

                if df.empty:
                    time.sleep(1)
                    continue

                required = ["Open", "High", "Low", "Close", "Volume"]
                missing = [col for col in required if col not in df.columns]
                if missing:
                    print(f"{ticker}: missing required columns {missing}")
                    time.sleep(1)
                    continue

                df = df.dropna(subset=required).copy()
                if df.empty:
                    print(f"{ticker}: empty after subset dropna")
                    time.sleep(1)
                    continue

                df = df.sort_index()
                print(
                    f"{ticker}: loaded {len(df)} rows using interval={attempt['interval']} period={attempt['period']} retry={retry+1}"
                )
                return df

            except Exception as e:
                print(f"get_data error for {ticker} with {attempt} retry={retry+1}: {e}")
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
            pressure -= 20

        pressure = max(0, min(100, pressure))
        heat = "HOT" if pressure >= 70 else "WARM" if pressure >= 45 else "COLD"

        return {
            "ticker": ticker,
            "price": price,
            "bias": bias,
            "signal": signal,
            "pressure": pressure,
            "heat": heat,
            "rsi": rsi,
            "ema9": ema9,
            "vwap": vwap,
        }

    except Exception as e:
        print(f"generate_signal error for {ticker}: {e}")
        return {"error": f"Signal error: {str(e)}"}