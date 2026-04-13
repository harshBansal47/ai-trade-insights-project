import pandas as pd
from src.core.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------
# TREND
# ---------------------------
def get_trend(df: pd.DataFrame) -> dict:
    ema20  = df["ema20"].iloc[-1]
    ema50  = df["ema50"].iloc[-1]
    ema200 = df["ema200"].iloc[-1]
    price  = df["close"].iloc[-1]

    if price > ema20 > ema50 > ema200:
        direction = "UPTREND"
        strength  = "strong"
        alignment = "20>50>200"
    elif price < ema20 < ema50 < ema200:
        direction = "DOWNTREND"
        strength  = "strong"
        alignment = "20<50<200"
    else:
        direction = "SIDEWAYS"
        # EMAs aligned but price not confirming → moderate, else weak
        strength  = "moderate" if ema20 > ema50 > ema200 or ema20 < ema50 < ema200 else "weak"
        alignment = "mixed"

    return {
        "direction":     direction,
        "strength":      strength,
        "ema_alignment": alignment,
    }


# ---------------------------
# RSI
# ---------------------------
def get_rsi_features(df: pd.DataFrame) -> dict:
    rsi_val  = df["rsi"].iloc[-1]
    prev_rsi = df["rsi"].iloc[-2]

    if rsi_val > 70:
        zone = "overbought"
    elif rsi_val < 30:
        zone = "oversold"
    elif rsi_val > 55:
        zone = "bullish"
    elif rsi_val < 45:
        zone = "bearish"
    else:
        zone = "neutral"

    momentum = "accelerating" if rsi_val > prev_rsi else "falling"

    return {
        "value":    round(rsi_val, 2),
        "zone":     zone,
        "momentum": momentum,
    }


# ---------------------------
# MACD — helpers
# ---------------------------
def _count_bars_since_crossover(df: pd.DataFrame, lookback: int = 50) -> int | None:
    """
    Walks back through macd_hist sign changes to find how many bars ago
    the last crossover occurred. Returns None if not found within lookback.
    """
    hist         = df["macd_hist"].values
    current_sign = hist[-1] > 0

    for i in range(2, min(lookback + 1, len(hist))):
        if (hist[-i] > 0) != current_sign:
            return i - 1

    return None


def get_macd_features(df: pd.DataFrame) -> dict:
    macd_val    = df["macd"].iloc[-1]
    signal_val  = df["macd_signal"].iloc[-1]
    hist        = df["macd_hist"].iloc[-1]
    prev_macd   = df["macd"].iloc[-2]
    prev_signal = df["macd_signal"].iloc[-2]
    prev_hist   = df["macd_hist"].iloc[-2]

    # Crossover detection
    if prev_macd < prev_signal and macd_val > signal_val:
        signal_type = "bullish_crossover"
        bars_since  = 1
    elif prev_macd > prev_signal and macd_val < signal_val:
        signal_type = "bearish_crossover"
        bars_since  = 1
    else:
        signal_type = "bullish" if macd_val > signal_val else "bearish"
        bars_since  = _count_bars_since_crossover(df)   # int or None

    histogram_state = "expanding" if abs(hist) > abs(prev_hist) else "contracting"

    return {
        "signal":     signal_type,
        "histogram":  histogram_state,
        "bars_since": bars_since,
    }


# ---------------------------
# ATR
# ---------------------------
def get_atr_features(df: pd.DataFrame) -> dict:
    atr_val  = df["atr"].iloc[-1]
    prev_atr = df["atr"].iloc[-2]
    atr_ma50 = df["atr_ma50"].iloc[-1]   # precomputed in add_indicators

    state = "expanding" if atr_val > prev_atr else "contracting"

    if pd.notna(atr_ma50) and atr_ma50 > 0:
        ratio = atr_val / atr_ma50
        if ratio < 0.8:
            volatility = "low"
        elif ratio > 1.5:
            volatility = "high"
        else:
            volatility = "moderate"
    else:
        volatility = "unknown"

    return {
        "state":      state,
        "volatility": volatility,
    }


# ---------------------------
# BOLLINGER BANDS
# ---------------------------
def get_bollinger_features(df: pd.DataFrame) -> dict:
    price  = df["close"].iloc[-1]
    upper  = df["bb_upper"].iloc[-1]
    middle = df["bb_middle"].iloc[-1]
    lower  = df["bb_lower"].iloc[-1]

    band_width = upper - lower

    # %B — 0.0 = at lower band, 1.0 = at upper band
    percent_b = (price - lower) / band_width if band_width > 0 else 0.5

    if price > upper:
        position = "above_upper"
    elif price < lower:
        position = "below_lower"
    elif price > middle:
        position = "upper_half"
    else:
        position = "lower_half"

    # Squeeze — current width below its own 20-bar rolling mean
    all_widths  = df["bb_upper"] - df["bb_lower"]
    width_ma20  = all_widths.rolling(20).mean().iloc[-1]
    squeeze     = bool(band_width < width_ma20) if pd.notna(width_ma20) and width_ma20 > 0 else False

    return {
        "position":   position,
        "percent_b":  round(percent_b, 3),
        "squeeze":    squeeze,
        "band_width": round(band_width, 6),
    }