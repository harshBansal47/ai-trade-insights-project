import asyncio
from src.core.redis import get_redis_client
from src.helpers.data_loader import CoinsDataLoader, process_symbol_multi_timeframe
from src.constants import MODE_TIMEFRAMES
from src.services.analysis import Mode, run_analysis




async def test_coin_data_loader():

    redis_client =await get_redis_client()
    coin = "BTCUSDT"
    mode = Mode.SCALPER
    mode_timeframes = MODE_TIMEFRAMES[mode.value]
    data =await CoinsDataLoader(redis_client=redis_client).process_symbol(
        symbol=coin, timeframe=mode_timeframes[0]
    )
    print(f"Data for {coin} in {mode.value} mode:")
    print(data)   




# async def test_stimulator():
#     coin = "BTCUSDT"
#     mode = Mode.SCALPER
#     mode_timeframes = MODE_TIMEFRAMES[mode.value]

#     raw_data  = process_symbol_multi_timeframe(
#         symbol=coin, timeframes=mode_timeframes
#     )
#     print(f"Raw data for {coin} in {mode.value} mode:")



# asyncio.run(test_stimulator())
asyncio.run(test_coin_data_loader())
