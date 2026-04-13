# src/agents/memory.py

"""
Signal Memory
-------------
Redis-backed memory that stores the last N signals per symbol+mode.
Used to give the LLM historical context so it can detect signal flips,
sustained trends, and avoid contradicting a recent strong signal without
sufficient new evidence.

Falls back to an in-process dict if Redis is unavailable.

Schema per key  (Redis hash):
  Key   : signal_memory:{symbol}:{mode}
  Field : timestamp ISO string
  Value : serialised SignalOutput JSON

Additional sorted set for ordered retrieval:
  Key   : signal_memory_index:{symbol}:{mode}
  Score : Unix timestamp
  Member: ISO timestamp (links back to hash field)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import redis

from src.core.config  import settings
from src.core.logger  import setup_logger

logger = setup_logger(__name__)

_MEMORY_TTL_SECONDS = 60 * 60 * 24 * 7   # 7 days
_MAX_HISTORY        = 10                  # signals stored per symbol+mode
_KEY_PREFIX         = "signal_memory"
_INDEX_PREFIX       = "signal_memory_index"


# ---------------------------------------------------------------------------
# Redis client — lazy singleton
# ---------------------------------------------------------------------------

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        _redis_client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _redis_client.ping()
        logger.info("[MEMORY] Redis connected")
        return _redis_client
    except Exception as exc:
        logger.warning(f"[MEMORY] Redis unavailable — using in-process fallback: {exc}")
        return None


# ---------------------------------------------------------------------------
# In-process fallback
# ---------------------------------------------------------------------------

_local_store: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_signal(symbol: str, mode: str, signal: dict) -> None:
    """
    Persist a signal dict to memory.

    Parameters
    ----------
    symbol : e.g. "BTCUSDT"
    mode   : e.g. "SWING"
    signal : model_dump() output of SignalOutput
    """
    ts  = datetime.now(timezone.utc).isoformat()
    payload = json.dumps({**signal, "_saved_at": ts}, default=str)

    r = _get_redis()
    if r:
        _redis_save(r, symbol, mode, ts, payload)
    else:
        _local_save(symbol, mode, payload)

    logger.info(f"[MEMORY] Saved signal for {symbol}|{mode} at {ts}")


def get_history(
    symbol: str,
    mode:   str,
    limit:  int = 5,
) -> list[dict]:
    """
    Retrieve the last `limit` signals for symbol+mode, most recent first.

    Returns
    -------
    List of signal dicts (may be empty if no history exists).
    """
    r = _get_redis()
    if r:
        return _redis_get_history(r, symbol, mode, limit)
    return _local_get_history(symbol, mode, limit)


def get_last_signal(symbol: str, mode: str) -> Optional[dict]:
    """Return the single most recent signal or None."""
    history = get_history(symbol, mode, limit=1)
    return history[0] if history else None


def build_memory_context(symbol: str, mode: str, limit: int = 3) -> str:
    """
    Build a plain-text memory context string for injection into the LLM prompt.

    Returns a formatted string summarising the last `limit` signals,
    or an empty string if no history exists.
    """
    history = get_history(symbol, mode, limit=limit)

    if not history:
        return ""

    lines = [f"RECENT SIGNAL HISTORY for {symbol} [{mode}] (most recent first):"]

    for i, entry in enumerate(history, 1):
        saved_at = entry.get("_saved_at", "unknown time")
        signal   = entry.get("signal",     "UNKNOWN")
        conf     = entry.get("confidence", 0.0)
        trend    = entry.get("dominant_trend", "?")
        fake     = entry.get("fake_signal_warning", False)
        headline = entry.get("headline", "")

        lines.append(
            f"  [{i}] {saved_at} | {signal} | conf={conf:.2f} | "
            f"trend={trend} | fake_warning={fake} | {headline}"
        )

    lines.append(
        "\nNote: If a strong signal was issued recently and market conditions "
        "have not materially changed, maintain consistency unless new evidence "
        "clearly justifies a signal flip."
    )

    return "\n".join(lines)


def clear_history(symbol: str, mode: str) -> None:
    """Delete all stored signals for symbol+mode (useful for testing)."""
    r = _get_redis()
    if r:
        try:
            r.delete(_hash_key(symbol, mode))
            r.delete(_index_key(symbol, mode))
            logger.info(f"[MEMORY] Cleared Redis history for {symbol}|{mode}")
        except Exception as exc:
            logger.error(f"[MEMORY] Clear error: {exc}")
    else:
        key = _local_key(symbol, mode)
        if key in _local_store:
            del _local_store[key]
            logger.info(f"[MEMORY] Cleared local history for {symbol}|{mode}")


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

def _hash_key(symbol: str, mode: str) -> str:
    return f"{_KEY_PREFIX}:{symbol.upper()}:{mode.upper()}"


def _index_key(symbol: str, mode: str) -> str:
    return f"{_INDEX_PREFIX}:{symbol.upper()}:{mode.upper()}"


def _redis_save(
    r:       redis.Redis,
    symbol:  str,
    mode:    str,
    ts:      str,
    payload: str,
) -> None:
    try:
        hkey  = _hash_key(symbol, mode)
        ikey  = _index_key(symbol, mode)
        score = datetime.fromisoformat(ts).timestamp()

        pipe = r.pipeline()
        pipe.hset(hkey, ts, payload)
        pipe.expire(hkey, _MEMORY_TTL_SECONDS)
        pipe.zadd(ikey, {ts: score})
        pipe.expire(ikey, _MEMORY_TTL_SECONDS)
        pipe.execute()

        # Prune oldest beyond _MAX_HISTORY
        _redis_prune(r, symbol, mode)

    except Exception as exc:
        logger.error(f"[MEMORY] Redis save error: {exc}", exc_info=True)


def _redis_prune(r: redis.Redis, symbol: str, mode: str) -> None:
    """Remove entries beyond _MAX_HISTORY from oldest to newest."""
    try:
        ikey  = _index_key(symbol, mode)
        hkey  = _hash_key(symbol, mode)
        count = r.zcard(ikey)

        if count > _MAX_HISTORY:
            excess  = count - _MAX_HISTORY
            oldest  = r.zrange(ikey, 0, excess - 1)         # oldest timestamps
            pipe    = r.pipeline()
            for ts in oldest:
                pipe.hdel(hkey, ts)
            pipe.zrem(ikey, *oldest)
            pipe.execute()
            logger.debug(f"[MEMORY] Pruned {excess} old entries for {symbol}|{mode}")

    except Exception as exc:
        logger.error(f"[MEMORY] Redis prune error: {exc}", exc_info=True)


def _redis_get_history(
    r:      redis.Redis,
    symbol: str,
    mode:   str,
    limit:  int,
) -> list[dict]:
    try:
        ikey = _index_key(symbol, mode)
        hkey = _hash_key(symbol, mode)

        # Get last `limit` timestamps from sorted set (most recent last → reverse)
        ts_list = r.zrevrange(ikey, 0, limit - 1)
        if not ts_list:
            return []

        pipe    = r.pipeline()
        for ts in ts_list:
            pipe.hget(hkey, ts)
        payloads = pipe.execute()

        results = []
        for payload in payloads:
            if payload:
                try:
                    results.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
        return results

    except Exception as exc:
        logger.error(f"[MEMORY] Redis get error: {exc}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# In-process fallback backend
# ---------------------------------------------------------------------------

def _local_key(symbol: str, mode: str) -> str:
    return f"{symbol.upper()}:{mode.upper()}"


def _local_save(symbol: str, mode: str, payload: str) -> None:
    key = _local_key(symbol, mode)
    _local_store.setdefault(key, [])
    _local_store[key].insert(0, json.loads(payload))   # most recent first
    _local_store[key] = _local_store[key][:_MAX_HISTORY]


def _local_get_history(symbol: str, mode: str, limit: int) -> list[dict]:
    key = _local_key(symbol, mode)
    return _local_store.get(key, [])[:limit]