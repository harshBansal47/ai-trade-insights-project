from src.agentic_workflow.chain import get_signal_chain
from src.services.analysis import assemble_report
from src.helpers.data_loader import _init_worker, process_symbol_multi_timeframe
from src.core.redis import get_redis_client



def build_test_pipeline():
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
    signal = get_signal_chain().run_safe(report)
    if signal is not None:
            report["ai_signal"] = signal.model_dump()
    return report


    
build_test_pipeline()