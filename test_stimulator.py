import asyncio

from src.helpers.data_loader import process_symbol_multi_timeframe
from src.constants import MODE_TIMEFRAMES
from src.services.analysis import Mode, run_analysis



async def test_stimulator():
    coin = "BTCUSDT"
    mode = Mode.SCALPER
    mode_timeframes = MODE_TIMEFRAMES[mode.value]

    raw_data  = process_symbol_multi_timeframe(
        symbol=coin, timeframes=mode_timeframes
    )
    print(f"Raw data for {coin} in {mode.value} mode:")



asyncio.run(test_stimulator())
