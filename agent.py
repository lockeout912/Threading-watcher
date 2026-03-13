import time
import requests
import pandas as pd
import numpy as np

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_yahoo_chart(
    ticker: str,
    interval: str = "5m",
    range_: str = "5d",
    prepost: bool = True
) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "interval": interval,
        "range": range_,
        "includePrePost": "true" if prepost else "false",
        "events": "div,splits",
    }

    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()

        data = response.json()
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

        df["Datetime"] = pd.to_datetime(
            timestamps, unit="s", utc=True
        ).tz_convert("America/New_York")

        df = df.set_index("Datetime")
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        df["Volume"] = df["Volume"].fillna(0)

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"fetch_yahoo_chart error for {ticker} {interval} {range_}: {e}")
        return pd.DataFrame()


def default_range_for_interval(interval: str) -> str:
    mapping = {
        "1m": "1d",
        "2m": "1d",
        "5m": "5d",
        "15m": "5d",
        "30m": "1mo",
        "60m": "3mo",
        "1d": "6mo",
    }
    return mapping.get(interval, "5d")


def get_data(ticker: str, interval: str = "5m", period: str | None = None) -> pd.DataFrame:
    if period is None:
        period = default_range_for_interval(interval)

    attempts = [
        {"interval": interval, "range": period},
        {"interval": "5m", "range": "1d"},
        {"interval": "15m", "range": "5d"},
        {"interval": "30m", "range": "1mo"},
        {"interval": "1d", "range": "6mo"},
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
                    f"{ticker}: loaded {len(df)} rows using "
                    f"interval={attempt['interval']} range={attempt['range']} retry={retry + 1}"
                )
                return df

            time.sleep(1)

    print(f"All data attempts failed for {ticker}")
    return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

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

    df["BarRange"] = df["High"] - df["Low"]
    df["AvgRange10"] = df["BarRange"].rolling(10).mean()

    df["PriceChange"] = df["Close"].diff()
    df["VolumeAvg10"] = df["Volume"].rolling(10).mean()

    return df


def _today_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    today = pd.Timestamp.now(tz="America/New_York").date()
    return df[df.index.date == today].copy()


def get_premarket_levels(df: pd.DataFrame):
    if df.empty:
        return None, None

    today_df = _today_slice(df)
    if today_df.empty:
        return None, None

    pre = today_df.between_time("04:00", "09:29")
    if pre.empty:
        return None, None

    pm_high = float(pre["High"].max())
    pm_low = float(pre["Low"].min())
    return pm_high, pm_low


def get_opening_range(df: pd.DataFrame):
    if df.empty:
        return None, None

    today_df = _today_slice(df)
    if today_df.empty:
        return None, None

    orb = today_df.between_time("09:30", "09:34")
    if orb.empty:
        return None, None

    or_high = float(orb["High"].max())
    or_low = float(orb["Low"].min())
    return or_high, or_low


def get_session_status(df: pd.DataFrame) -> str:
    if df.empty:
        return "NO DATA"

    now = pd.Timestamp.now(tz="America/New_York").time()

    if now >= pd.Timestamp("04:00").time() and now < pd.Timestamp("09:30").time():
        return "PREMARKET"
    if now >= pd.Timestamp("09:30").time() and now <= pd.Timestamp("16:00").time():
        return "REGULAR SESSION"
    return "AFTER HOURS"


def get_market_state(price, vwap, ema9, or_high, or_low, avg_range, pm_high, pm_low):
    if or_high is not None and or_low is not None:
        if price > or_high and price > vwap:
            return "BULL BREAKOUT"
        if price < or_low and price < vwap:
            return "BEAR BREAKOUT"
        if or_low <= price <= or_high:
            return "OPENING RANGE CHOP"

    if pm_high is not None and price > pm_high and price > vwap:
        return "PREMARKET BREAKOUT UP"
    if pm_low is not None and price < pm_low and price < vwap:
        return "PREMARKET BREAKDOWN"

    if abs(price - vwap) <= max(price * 0.001, 0.25):
        return "VWAP CHOP"

    if price > ema9 and ema9 > vwap:
        return "BULL TREND"

    if price < ema9 and ema9 < vwap:
        return "BEAR TREND"

    if avg_range > 0:
        return "DEVELOPING"

    return "NEUTRAL"


def classify_signal_strength(pressure: int, market_state: str) -> str:
    if "BREAKOUT" in market_state and pressure >= 70:
        return "HIGH CONVICTION"
    if pressure >= 60:
        return "MEDIUM CONVICTION"
    if "CHOP" in market_state:
        return "LOW CONVICTION"
    return "DEVELOPING"


def rolling_feed(df: pd.DataFrame, signal_pack: dict) -> list[str]:
    events = []

    price = signal_pack["price"]
    calls = signal_pack["calls_favored_above"]
    puts = signal_pack["puts_favored_below"]
    state = signal_pack["market_state"]
    warning = signal_pack["warning_line"]
    or_high = signal_pack["or_high"]
    or_low = signal_pack["or_low"]

    if "BULL" in state:
        events.append(f"🚀 {signal_pack['ticker']} bulls in control above {calls:.2f}")
    elif "BEAR" in state:
        events.append(f"🩸 {signal_pack['ticker']} bears in control below {puts:.2f}")
    else:
        events.append(f"🛰️ {signal_pack['ticker']} inside active battle zone")

    if or_high is not None and price > or_high:
        events.append(f"⚡ Price above OR High {or_high:.2f}")
    if or_low is not None and price < or_low:
        events.append(f"⚠️ Price below OR Low {or_low:.2f}")

    if price > calls:
        events.append(f"✅ Calls favored while holding above {calls:.2f}")
    elif price < puts:
        events.append(f"✅ Puts favored while holding below {puts:.2f}")
    else:
        events.append(f"🟡 Chop zone active around {warning:.2f}")

    if signal_pack["pressure"] >= 70:
        events.append("🔥 Momentum pressure elevated")
    elif signal_pack["pressure"] <= 40:
        events.append("❄️ Momentum pressure weak")

    latest = df.iloc[-1]
    if pd.notna(latest.get("VolumeAvg10")) and latest["Volume"] > latest["VolumeAvg10"] * 1.5:
        events.append("📣 Volume surge detected")

    return events[:6]


def build_battle_map(df: pd.DataFrame, ticker: str = "SPY") -> dict:
    df = add_indicators(df)
    latest = df.iloc[-1]

    price = float(latest["Close"])
    ema9 = float(latest["EMA9"]) if pd.notna(latest["EMA9"]) else price
    ema20 = float(latest["EMA20"]) if pd.notna(latest["EMA20"]) else price
    vwap = float(latest["VWAP"]) if pd.notna(latest["VWAP"]) else price
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 50.0
    avg_range = float(latest["AvgRange10"]) if pd.notna(latest["AvgRange10"]) else 0.0

    pm_high, pm_low = get_premarket_levels(df)
    or_high, or_low = get_opening_range(df)
    session_status = get_session_status(df)

    if price > ema9 and ema9 > vwap:
        bias = "BULLISH"
    elif price < ema9 and ema9 < vwap:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    pressure = 50

    if rsi > 60:
        pressure += 15
    elif rsi < 40:
        pressure -= 15

    if price > ema9:
        pressure += 10
    else:
        pressure -= 10

    if price > vwap:
        pressure += 10
    else:
        pressure -= 10

    if price > ema20:
        pressure += 5
    else:
        pressure -= 5

    pressure = max(0, min(100, pressure))
    heat = "HOT" if pressure >= 70 else "WARM" if pressure >= 45 else "COLD"

    market_state = get_market_state(price, vwap, ema9, or_high, or_low, avg_range, pm_high, pm_low)
    conviction = classify_signal_strength(pressure, market_state)

    calls_favored_above = max(ema9, vwap)
    puts_favored_below = min(ema9, vwap)

    if or_high is not None:
        calls_favored_above = max(calls_favored_above, or_high)
    if or_low is not None:
        puts_favored_below = min(puts_favored_below, or_low)

    if or_high is not None and or_low is not None and or_low < or_high:
        chop_low = or_low
        chop_high = or_high
    else:
        chop_low = puts_favored_below
        chop_high = calls_favored_above

    if or_high is not None and or_low is not None:
        range_size = max(or_high - or_low, price * 0.003)
    elif pm_high is not None and pm_low is not None:
        range_size = max(pm_high - pm_low, price * 0.003)
    else:
        range_size = max(abs(ema9 - vwap) * 2, price * 0.003)

    warning_line = (chop_low + chop_high) / 2

    if bias == "BULLISH":
        invalidation = min(vwap, or_low) if or_low is not None else vwap
    elif bias == "BEARISH":
        invalidation = max(vwap, or_high) if or_high is not None else vwap
    else:
        invalidation = warning_line

    likely_up = price + (range_size * 0.75)
    stretch_up = price + (range_size * 1.50)
    likely_down = price - (range_size * 0.75)
    stretch_down = price - (range_size * 1.50)

    if market_state == "BULL BREAKOUT":
        signal = f"CALLS FAVORED above {calls_favored_above:.2f}"
    elif market_state == "BEAR BREAKOUT":
        signal = f"PUTS FAVORED below {puts_favored_below:.2f}"
    elif "CHOP" in market_state:
        signal = "CHOP / WAIT FOR BREAKOUT"
    elif bias == "BULLISH":
        signal = f"CALLS BIAS above {calls_favored_above:.2f}"
    elif bias == "BEARISH":
        signal = f"PUTS BIAS below {puts_favored_below:.2f}"
    else:
        signal = "NO CLEAR EDGE"

    if market_state == "BULL BREAKOUT":
        commentary = (
            f"{ticker} is above the opening range and above VWAP. "
            f"Bulls have control while price holds above {calls_favored_above:.2f}. "
            f"Primary continuation zone is {likely_up:.2f}, with stretch potential near {stretch_up:.2f}. "
            f"Warning if momentum fades back under {warning_line:.2f}."
        )
    elif market_state == "BEAR BREAKOUT":
        commentary = (
            f"{ticker} is below the opening range and below VWAP. "
            f"Bears have control while price stays under {puts_favored_below:.2f}. "
            f"Primary continuation zone is {likely_down:.2f}, with stretch potential near {stretch_down:.2f}. "
            f"Warning if price reclaims {warning_line:.2f}."
        )
    elif "CHOP" in market_state:
        commentary = (
            f"{ticker} is trading inside the active battle zone between {chop_low:.2f} and {chop_high:.2f}. "
            f"This is chop until proven otherwise. "
            f"Wait for a clean break above {calls_favored_above:.2f} or below {puts_favored_below:.2f}."
        )
    elif bias == "BULLISH":
        commentary = (
            f"{ticker} has bullish structure with price above EMA9 and VWAP. "
            f"Calls are favored while price holds above {calls_favored_above:.2f}. "
            f"First push zone is {likely_up:.2f}."
        )
    elif bias == "BEARISH":
        commentary = (
            f"{ticker} has bearish structure with price below EMA9 and VWAP. "
            f"Puts are favored while price stays below {puts_favored_below:.2f}. "
            f"First downside push zone is {likely_down:.2f}."
        )
    else:
        commentary = (
            f"{ticker} is in a mixed state. Use {calls_favored_above:.2f} as the upside trigger "
            f"and {puts_favored_below:.2f} as the downside trigger."
        )

    pack = {
        "ticker": ticker,
        "price": price,
        "bias": bias,
        "signal": signal,
        "pressure": pressure,
        "heat": heat,
        "rsi": rsi,
        "ema9": ema9,
        "ema20": ema20,
        "vwap": vwap,
        "market_state": market_state,
        "conviction": conviction,
        "session_status": session_status,
        "pm_high": pm_high,
        "pm_low": pm_low,
        "or_high": or_high,
        "or_low": or_low,
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
        "commentary": commentary,
    }

    pack["feed"] = rolling_feed(df, pack)
    return pack


def generate_signal(ticker: str = "SPY", interval: str = "5m") -> dict:
    try:
        df = get_data(ticker, interval=interval)

        if df.empty:
            return {"error": f"No data for {ticker}"}

        return build_battle_map(df, ticker=ticker)

    except Exception as e:
        print(f"generate_signal error for {ticker}: {e}")
        return {"error": f"Signal error: {str(e)}"}


# ============================================================
# SCANNER + ALERTS LAYER (APP-SAFE) ✅ DOES NOT TOUCH THE BRAIN
# ============================================================

DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "NVDA", "TSLA", "AMD", "META",
    "AMZN", "MSFT", "NFLX", "PLTR",
    "MSTR", "COIN", "SOFI", "HOOD",
    "XOM", "OXY", "USO",
]


class AlertState:
    """
    Minimal in-memory state so alerts only fire on changes/crossings.
    In Streamlit you can keep one of these in st.session_state.
    """
    def __init__(self):
        self.last = {}  # ticker -> snapshot dict
        self.last_sent_at = {}  # (ticker, alert_type) -> epoch seconds

    def get(self, ticker: str) -> dict | None:
        return self.last.get(ticker)

    def set(self, ticker: str, snap: dict):
        self.last[ticker] = snap

    def can_send(self, ticker: str, alert_type: str, cooldown_sec: int) -> bool:
        key = (ticker, alert_type)
        now = time.time()
        last_time = self.last_sent_at.get(key, 0)
        if now - last_time >= cooldown_sec:
            self.last_sent_at[key] = now
            return True
        return False


def _snapshot(sig: dict) -> dict:
    """Create a tiny comparable snapshot (so we don't store huge objects)."""
    return {
        "price": float(sig.get("price", 0) or 0),
        "market_state": str(sig.get("market_state", "")),
        "bias": str(sig.get("bias", "")),
        "pressure": int(sig.get("pressure", 0) or 0),
        "calls_favored_above": float(sig.get("calls_favored_above", 0) or 0),
        "puts_favored_below": float(sig.get("puts_favored_below", 0) or 0),
        "warning_line": float(sig.get("warning_line", 0) or 0),
        "signal": str(sig.get("signal", "")),
        "session_status": str(sig.get("session_status", "")),
        "conviction": str(sig.get("conviction", "")),
        "heat": str(sig.get("heat", "")),
    }


def _crossed_up(prev_price: float, price: float, level: float) -> bool:
    if level is None:
        return False
    return prev_price <= level < price


def _crossed_down(prev_price: float, price: float, level: float) -> bool:
    if level is None:
        return False
    return prev_price >= level > price


def evaluate_alerts(
    sig: dict,
    prev_snap: dict | None,
    *,
    cooldown_sec: int = 300
) -> list[dict]:
    """
    Returns a list of alert events (dicts). Designed to be simple and safe.
    """
    if not sig or "error" in sig:
        return []

    ticker = sig.get("ticker", "UNKNOWN")
    snap = _snapshot(sig)

    alerts: list[dict] = []

    # First run: no previous snapshot, don't spam alerts
    if not prev_snap:
        return alerts

    prev_price = float(prev_snap.get("price", 0) or 0)
    price = snap["price"]

    prev_state = str(prev_snap.get("market_state", ""))
    state = snap["market_state"]

    prev_pressure = int(prev_snap.get("pressure", 0) or 0)
    pressure = snap["pressure"]

    calls = snap["calls_favored_above"]
    puts = snap["puts_favored_below"]
    warning = snap["warning_line"]

    # 1) State change alerts (big ones)
    if state != prev_state:
        if "BULL BREAKOUT" == state:
            alerts.append({
                "ticker": ticker,
                "type": "STATE_BULL_BREAKOUT",
                "title": f"{ticker} BULL BREAKOUT",
                "message": f"State flipped to BULL BREAKOUT. Price {price:.2f} above OR/VWAP trigger. Calls favored above {calls:.2f}.",
                "cooldown_sec": cooldown_sec,
            })
        elif "BEAR BREAKOUT" == state:
            alerts.append({
                "ticker": ticker,
                "type": "STATE_BEAR_BREAKOUT",
                "title": f"{ticker} BEAR BREAKOUT",
                "message": f"State flipped to BEAR BREAKOUT. Price {price:.2f} below OR/VWAP trigger. Puts favored below {puts:.2f}.",
                "cooldown_sec": cooldown_sec,
            })

    # 2) Trigger-cross alerts (the practical “do something” pings)
    if _crossed_up(prev_price, price, calls):
        alerts.append({
            "ticker": ticker,
            "type": "CROSS_CALLS_ABOVE",
            "title": f"{ticker} crossed CALLS trigger",
            "message": f"Price crossed UP through Calls Favored Above {calls:.2f} → now {price:.2f}.",
            "cooldown_sec": cooldown_sec,
        })

    if _crossed_down(prev_price, price, puts):
        alerts.append({
            "ticker": ticker,
            "type": "CROSS_PUTS_BELOW",
            "title": f"{ticker} crossed PUTS trigger",
            "message": f"Price crossed DOWN through Puts Favored Below {puts:.2f} → now {price:.2f}.",
            "cooldown_sec": cooldown_sec,
        })

    # 3) Warning-line reclaim/loss (chop + fakeout detector)
    if warning and warning > 0:
        if _crossed_up(prev_price, price, warning):
            alerts.append({
                "ticker": ticker,
                "type": "CROSS_WARNING_UP",
                "title": f"{ticker} reclaimed warning line",
                "message": f"Price reclaimed warning line {warning:.2f} → now {price:.2f}.",
                "cooldown_sec": cooldown_sec,
            })
        if _crossed_down(prev_price, price, warning):
            alerts.append({
                "ticker": ticker,
                "type": "CROSS_WARNING_DOWN",
                "title": f"{ticker} lost warning line",
                "message": f"Price lost warning line {warning:.2f} → now {price:.2f}.",
                "cooldown_sec": cooldown_sec,
            })

    # 4) Pressure spike (momentum pickup)
    if prev_pressure < 70 <= pressure:
        alerts.append({
            "ticker": ticker,
            "type": "PRESSURE_SPIKE",
            "title": f"{ticker} pressure spike",
            "message": f"Pressure crossed into HOT zone: {prev_pressure}/100 → {pressure}/100. Heat={snap['heat']} Conviction={snap['conviction']}.",
            "cooldown_sec": cooldown_sec,
        })

    return alerts


def scan_watchlist(
    tickers: list[str] | None = None,
    *,
    interval: str = "5m",
    state: AlertState | None = None,
    per_ticker_delay_sec: float = 0.25,
    cooldown_sec: int = 300
) -> dict:
    """
    Runs generate_signal across a list and returns:
      {
        "signals": {ticker: sig_dict},
        "alerts": [alert_event_dicts...]
      }

    This function DOES NOT change how generate_signal works.
    It only orchestrates and compares snapshots.
    """
    if tickers is None:
        tickers = DEFAULT_WATCHLIST

    if state is None:
        state = AlertState()

    results: dict = {"signals": {}, "alerts": []}

    for t in tickers:
        t = str(t).upper().strip()
        if not t:
            continue

        sig = generate_signal(t, interval=interval)
        results["signals"][t] = sig

        if "error" not in sig:
            prev = state.get(t)
            prev_snap = prev if prev else None
            new_snap = _snapshot(sig)

            # Evaluate
            alert_events = evaluate_alerts(sig, prev_snap, cooldown_sec=cooldown_sec)

            # Apply cooldown gating
            gated = []
            for ev in alert_events:
                a_type = ev.get("type", "UNKNOWN")
                a_cd = int(ev.get("cooldown_sec", cooldown_sec))
                if state.can_send(t, a_type, a_cd):
                    gated.append(ev)

            results["alerts"].extend(gated)

            # Store snapshot
            state.set(t, new_snap)

        time.sleep(per_ticker_delay_sec)

    return results


def send_webhook_alert(webhook_url: str, alert_event: dict) -> bool:
    """
    Simple generic webhook sender.
    Works for many services if you adjust payload format.
    """
    try:
        payload = {
            "text": f"**{alert_event.get('title','ALERT')}**\n{alert_event.get('message','')}"
        }
        r = requests.post(webhook_url, json=payload, timeout=12)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"send_webhook_alert error: {e}")
        return False


def format_alert_compact(alert_event: dict) -> str:
    ticker = alert_event.get("ticker", "UNK")
    title = alert_event.get("title", "ALERT")
    msg = alert_event.get("message", "")
    return f"{ticker} | {title} — {msg}"