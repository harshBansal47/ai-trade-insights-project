
from constants import MODE_TIMEFRAMES
from helpers.indicators import add_indicators
from helpers.input_json_helper import build_global_signals, build_timeframe_features



def generate_trade_payload(symbol, mode, all_dfs, indicator_calculator):
    """
    symbol: str
    mode: str
    all_dfs: dict -> {"1m": df, "5m": df, ...}
    indicator_calculator: your existing module
    """

    # 1. Select timeframes
    selected_tfs = MODE_TIMEFRAMES.get(mode, [])

    # 2. Filter dfs
    dfs = {tf: all_dfs[tf] for tf in selected_tfs if tf in all_dfs}

    # 3. Apply indicators
    dfs_with_indicators = {}
    for tf, df in dfs.items():
        df = add_indicators(df)
        dfs_with_indicators[tf] = df

    # 4. Build feature-engineered payload
    timeframe_data = {}
    for tf, df in dfs_with_indicators.items():
        timeframe_data[tf] = build_timeframe_features(df)

    # 5. Global signals
    signals = build_global_signals(timeframe_data)

    # 6. Get current price (from lowest tf)
    base_tf = selected_tfs[0]
    price = dfs_with_indicators[base_tf]["close"].iloc[-1]

    # 7. Final payload
    payload = {
        "symbol": symbol,
        "mode": mode,
        "price": float(price),
        "timeframes": timeframe_data,
        "signals": signals
    }

    return payload