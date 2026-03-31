
"""
Technical Indicators Module

All functions take a pandas DataFrame with columns:
['open', 'high', 'low', 'close', 'volume']

Returns the same DataFrame with added indicator columns.
"""

import pandas as pd


# =========================
# EMA
# =========================
def ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    return df[column].ewm(span=period, adjust=False).mean()


# =========================
# RSI
# =========================
def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    delta = df[column].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================
# MACD
# =========================
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close"
):
    ema_fast = ema(df, fast, column)
    ema_slow = ema(df, slow, column)

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram
    })


# =========================
# ATR
# =========================
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


# =========================
# BOLLINGER BANDS
# =========================
def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: int = 2,
    column: str = "close"
):
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return pd.DataFrame({
        "bb_middle": sma,
        "bb_upper": upper_band,
        "bb_lower": lower_band
    })


# =========================
# MASTER FUNCTION
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema_20"] = ema(df, 20)
    df["ema_50"] = ema(df, 50)
    df["ema_200"] = ema(df, 200)

    # RSI
    df["rsi_14"] = rsi(df, 14)

    # MACD
    macd_df = macd(df)
    df = pd.concat([df, macd_df], axis=1)

    # ATR
    df["atr_14"] = atr(df, 14)

    # Bollinger Bands
    bb_df = bollinger_bands(df)
    df = pd.concat([df, bb_df], axis=1)

    return df
