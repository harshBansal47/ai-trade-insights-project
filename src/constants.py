

MODE_TIMEFRAMES = {
    "SCALPER": ["15s", "30s", "1m", "3m", "5m", "15m", "30m", "1h"],
    "SWING": ["15m", "30m", "1h", "2h", "4h"],
    "CONSERVATIVE": ["1h", "2h", "4h"]  # ideally add 1D later
}

TF_WEIGHTS = {
    "SCALPER": {
        "1m": 0.25,
        "3m": 0.2,
        "5m": 0.2,
        "15m": 0.2,
        "1h": 0.15
    },
    "SWING": {
        "15m": 0.1,
        "1h": 0.3,
        "4h": 0.35,
        "1d": 0.25
    },
    "CONSERVATIVE": {
        "1h": 0.2,
        "4h": 0.3,
        "1d": 0.3,
        "1w": 0.2
    }
}