# ---------------------------
# TREND
# ---------------------------
def get_trend(df):
    ema20 = df["ema20"].iloc[-1]
    ema50 = df["ema50"].iloc[-1]
    ema200 = df["ema200"].iloc[-1]
    price = df["close"].iloc[-1]

    if price > ema20 > ema50 > ema200:
        direction = "UPTREND"
        strength = "strong"
        alignment = "20>50>200"
    elif price < ema20 < ema50 < ema200:
        direction = "DOWNTREND"
        strength = "strong"
        alignment = "20<50<200"
    else:
        direction = "SIDEWAYS"
        strength = "weak"
        alignment = "mixed"

    return {
        "direction": direction,
        "strength": strength,
        "ema_alignment": alignment
    }


# ---------------------------
# RSI
# ---------------------------
def get_rsi_features(df):
    rsi = df["rsi"].iloc[-1]
    prev_rsi = df["rsi"].iloc[-2]

    if rsi > 70:
        zone = "overbought"
    elif rsi < 30:
        zone = "oversold"
    elif rsi > 55:
        zone = "bullish"
    elif rsi < 45:
        zone = "bearish"
    else:
        zone = "neutral"

    momentum = "accelerating" if rsi > prev_rsi else "falling"

    return {
        "value": round(rsi, 2),
        "zone": zone,
        "momentum": momentum
    }


# ---------------------------
# MACD
# ---------------------------
def get_macd_features(df):
    macd = df["macd"].iloc[-1]
    signal = df["macd_signal"].iloc[-1]
    hist = df["macd_hist"].iloc[-1]

    prev_macd = df["macd"].iloc[-2]
    prev_signal = df["macd_signal"].iloc[-2]

    # crossover detection
    if prev_macd < prev_signal and macd > signal:
        signal_type = "bullish_crossover"
        bars_since = 1
    elif prev_macd > prev_signal and macd < signal:
        signal_type = "bearish_crossover"
        bars_since = 1
    else:
        signal_type = "bullish" if macd > signal else "bearish"
        bars_since = 3  # approximate fallback

    histogram_state = "expanding" if abs(hist) > abs(df["macd_hist"].iloc[-2]) else "contracting"

    return {
        "signal": signal_type,
        "histogram": histogram_state,
        "bars_since": bars_since
    }


# ---------------------------
# ATR
# ---------------------------
def get_atr_features(df):
    atr = df["atr"].iloc[-1]
    prev_atr = df["atr"].iloc[-2]

    if atr > prev_atr:
        state = "expanding"
    else:
        state = "contracting"

    if atr < df["atr"].rolling(50).mean().iloc[-1]:
        volatility = "low"
    else:
        volatility = "moderate"

    return {
        "state": state,
        "volatility": volatility
    }

