
from src.helpers.data_loader import process_symbol_multi_timeframe
from src.core.redis import redis_client

def test_raw_data_collection():
    redis_client.ping()
    symbol = "BTCUSDT"
    timeframes = ["1m", "5m", "15m"]
    raw_data = process_symbol_multi_timeframe(symbol, timeframes, fill_gaps=False)
    print(raw_data)

test_raw_data_collection()