from src.helpers.data_loader import process_symbol_multi_timeframe


def test_raw_data_collection():
    symbol = "BTCUSDT"
    timeframes = ["1m", "5m", "15m"]
    raw_data = process_symbol_multi_timeframe(symbol, timeframes, fill_gaps=False)
    print(raw_data)

test_raw_data_collection()