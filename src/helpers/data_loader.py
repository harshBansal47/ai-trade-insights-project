import os
import logging
import redis
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import orjson

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR_HIST_DATA = "/home/harsh-bansal/Downloads/hist-coins-data"
       

TIMEFRAME_SECONDS: Dict[str, int] = {
    "5s":  5,
    "15s": 15,
    "30s": 30,
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1_800,
    "1h":  3_600,
    "4h":  14_400
}

_loads = orjson.loads

# ---------------------------------------------------------------------------
# CoinsDataLoader
# ---------------------------------------------------------------------------
class CoinsDataLoader:


    def __init__(self, redis: redis.Redis) -> None:
        self.redis_client = redis

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process_symbol(
        self,
        symbol: str,
        timeframe: str,
        fill_gaps: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load, merge, and optionally gap-fill OHLCV data for one symbol/timeframe.

        Returns None when no data is available from any source.
        """
        filepath = self.get_coins_file_path(symbol, timeframe)

        # 1. Load from each source independently
        df_file  = self._load_from_file(filepath)
        df_redis = self._load_from_redis(symbol, timeframe, self.redis_config)

        # 2. Merge (file rows are base; Redis rows overwrite on duplicate timestamps)
        df = self._merge(df_file, df_redis)
        if df is None or df.empty:
            return None

        # 3. Optionally fill temporal gaps
        if fill_gaps:
            df = self._fill_time_gaps(df, timeframe)

        return df

    # ------------------------------------------------------------------
    # Path helper
    # ------------------------------------------------------------------
    @staticmethod
    def get_coins_file_path(coin: str, timeframe: str) -> str:
        return os.path.join(BASE_DIR_HIST_DATA, coin, f"{coin}_{timeframe}.parquet")

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    @staticmethod
    def _load_from_file(filepath: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_parquet(filepath)
            return df if not df.empty else None
        except Exception as exc:
            logger.error("[ParquetLoad] Failed for %s: %s", filepath, exc)
            return None

    @staticmethod
    def _load_from_redis(
        symbol: str,
        timeframe: str,
        redis_client: redis.Redis,
    ) -> Optional[pd.DataFrame]:
        key = f"{symbol}:{timeframe}"
        try:
            client = redis_client.build_client()
            raw_items: List[bytes] = client.lrange(key, 0, -1)
            if not raw_items:
                return None

            rows = [_loads(raw) for raw in raw_items if raw]
            if not rows:
                return None

            df = pd.DataFrame(rows)
            if "timestamp" not in df.columns:
                logger.warning("[RedisLoad] 'timestamp' column missing for key %s", key)
                return None

            return df
        except Exception as exc:
            logger.error("[RedisLoad] Failed for %s: %s", key, exc)
            return None

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(
        df_file: Optional[pd.DataFrame],
        df_redis: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """
        Combine file and Redis frames.

        Ordering: [df_file, df_redis] → keep="last" means Redis wins on
        duplicate timestamps. This is intentional: Redis holds the most
        recent ticks and should always take precedence over the parquet
        snapshot.
        """
        frames = [f for f in (df_file, df_redis) if f is not None]
        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = df["timestamp"].astype(int)
        df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Gap filling
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_time_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        step = TIMEFRAME_SECONDS.get(timeframe)
        if not step:
            logger.warning("[FillGaps] Unknown timeframe '%s' — skipping gap fill.", timeframe)
            return df

        df = df.copy()
        df["timestamp"] = df["timestamp"].astype(int)
        df = df.set_index("timestamp")

        full_index = range(df.index.min(), df.index.max() + step, step)
        df = df.reindex(full_index)

        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        if price_cols:
            # Forward-fill only — no bfill to avoid look-ahead bias
            df[price_cols] = df[price_cols].ffill()

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)

        return df.reset_index().rename(columns={"index": "timestamp"})


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------
def _worker(
    symbol: str,
    timeframe: str,
    redis_client: redis.Redis,
    fill_gaps: bool = True,
) -> Dict[str, Any]:
    """
    Executed inside a subprocess spawned by ProcessPoolExecutor.

    A new CoinsDataLoader (and therefore a new Redis connection) is
    created here — inside the subprocess — so nothing that cannot be
    pickled crosses the process boundary.
    """
    loader = CoinsDataLoader(redis_client)
    df = loader.process_symbol(symbol, timeframe, fill_gaps)

    if df is None:
        return {
            "status": "skipped",
            "symbol": symbol,
            "timeframe": timeframe,
            "reason": "No data found",
        }

    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": len(df),
        "df": df,
    }


# ---------------------------------------------------------------------------
# Public API: process one symbol across multiple timeframes in parallel
# ---------------------------------------------------------------------------
def process_symbol_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    redis_client: redis.Redis,
    fill_gaps: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if not timeframes:
        return []

    args = [(symbol, tf, redis_client, fill_gaps) for tf in timeframes]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda a: _worker(*a), args))

    return results