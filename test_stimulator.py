from src.services.analysis import assemble_report
from src.helpers.data_loader import _init_worker, process_symbol_multi_timeframe
from src.core.redis import get_redis_client

def test_raw_data_collection():
    redis = get_redis_client()
    redis.ping()
    symbol = "BTCUSDT"
    timeframes = ["1m", "5m", "15m"]
    _init_worker()
    raw_data = process_symbol_multi_timeframe(symbol, timeframes, fill_gaps=False)
    mode_value = "SCALPER"
    report = assemble_report(
        symbol=symbol,
        mode=mode_value,
        raw_data=raw_data
    )
    print(report)

test_raw_data_collection()