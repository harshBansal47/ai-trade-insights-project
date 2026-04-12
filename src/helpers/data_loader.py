import os
import time
import redis
import pandas as pd
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import orjson

from src.core.logger import setup_logger
from src.core.redis import get_redis_client

logger = setup_logger(__name__)

BASE_DIR_HIST_DATA = "/home/harsh-bansal/Downloads/hist-coins-data"

TIMEFRAME_SECONDS: Dict[str, int] = {
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
}

_loads = orjson.loads


# ---------------------------------------------------------------------------
# CoinsDataLoader
# ---------------------------------------------------------------------------
class CoinsDataLoader:

    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis_client = redis_client

    def process_symbol(
        self,
        symbol: str,
        timeframe: str,
        fill_gaps: bool = True,
    ) -> Optional[pd.DataFrame]:

        logger.info(f"[PROCESS] {symbol} | {timeframe}")

        filepath = self.get_coins_file_path(symbol, timeframe)

        df_file = self._load_from_file(filepath)
        df_redis = self._load_from_redis(symbol, timeframe)
        df_file.sort_values(by="timestamp", inplace=True) if df_file is not None else None
        df_redis.sort_values(by="timestamp", inplace=True) if df_redis is not None else None
        df_file = self.validate_dataframe_recency(df_file,max_delay_seconds=90000) if df_file is not None else None
        df_redis = self.validate_dataframe_recency(df_redis) if df_redis is not None else None


        df = self._merge(df_file, df_redis)

        if df is None or df.empty:
            logger.warning(f"[NO DATA] {symbol} | {timeframe}")
            return None

        if fill_gaps:
            df = self._fill_time_gaps(df, timeframe)

        logger.info(f"[SUCCESS] {symbol} | {timeframe} | rows={len(df)}")
        return df

    @staticmethod
    def get_coins_file_path(coin: str, timeframe: str) -> str:
        return os.path.join(BASE_DIR_HIST_DATA, coin, f"{coin}_{timeframe}.parquet")

    @staticmethod
    def _load_from_file(filepath: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_parquet(filepath)
            return df if not df.empty else None
        except Exception as exc:
            logger.error(f"[FILE ERROR] {filepath} | {exc}")
            return None

    def _load_from_redis(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:

        key = f"{symbol}:{timeframe}"

        try:
            raw_items = self.redis_client.lrange(key, 0, -1)

            if not raw_items:
                return None

            rows = [_loads(item) for item in raw_items if item]

            df = pd.DataFrame(rows)

            if "timestamp" not in df.columns:
                logger.warning(f"[REDIS ERROR] Missing timestamp | {key}")
                return None

            return df

        except Exception as exc:
            logger.error(f"[REDIS ERROR] {key} | {exc}", exc_info=True)
            return None

    @staticmethod
    def _merge(df_file, df_redis):
        frames = [f for f in (df_file, df_redis) if f is not None]

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = df["timestamp"].astype(int)
        df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def _fill_time_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:

        step = TIMEFRAME_SECONDS.get(timeframe)

        if not step:
            logger.warning(f"[GAP SKIP] Unknown timeframe {timeframe}")
            return df

        df = df.copy()
        df["timestamp"] = df["timestamp"].astype(int)
        df = df.set_index("timestamp")

        full_index = range(df.index.min(), df.index.max() + step, step)
        df = df.reindex(full_index)

        price_cols = ["open", "high", "low", "close"]

        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)

        return df.reset_index().rename(columns={"index": "timestamp"})
    

    def validate_dataframe_recency(
            self,
    df: pd.DataFrame,
    max_delay_seconds: int = 120,  # 2 minutes
) -> Optional[pd.DataFrame]:
        try:
            latest_ts = int(df["timestamp"].iloc[-1])
            current_ts = int(time.time())

            delay = current_ts - latest_ts

            logger.info(
                f"[DF VALIDATION] latest={latest_ts} current={current_ts} delay={delay}s"
            )

            if delay > max_delay_seconds:
                logger.warning(
                    f"[DF STALE] delay={delay}s > allowed={max_delay_seconds}s"
                )
                return None

            return df

        except Exception as e:
            logger.error(f"[DF VALIDATION ERROR] {e}", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# WORKER FUNCTIONS (TOP LEVEL ONLY)
# ---------------------------------------------------------------------------


def _worker(symbol: str, timeframe: str, fill_gaps: bool):
    logger.info(f"[WORKER START] {symbol} | {timeframe}")

    try:
        redis_client = get_redis_client()

        loader = CoinsDataLoader(redis_client)
        df = loader.process_symbol(symbol, timeframe, fill_gaps)

        if df is None:
            return {
                "status": "skipped",
                "symbol": symbol,
                "timeframe": timeframe,
            }

        return {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "df":df
        }

    except Exception as e:
        logger.error(f"[WORKER ERROR] {symbol} | {timeframe} | {e}", exc_info=True)
        return {
            "status": "error",
            "symbol": symbol,
            "timeframe": timeframe,
            "error": str(e),
        }



def _worker_unpack(args):
    return _worker(*args)

# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------


def process_symbol_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    fill_gaps: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:

    logger.info(f"[MULTI TF START] {symbol} | {timeframes}")

    if not timeframes:
        return []

    args = [(symbol, tf, fill_gaps) for tf in timeframes]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker_unpack, args))

    logger.info(f"[MULTI TF DONE] {symbol}")

    return results
