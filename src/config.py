from __future__ import annotations

import pandas as pd

FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
HORIZON = 168
N_FOLDS = 5
ZERO_LAG_THRESHOLD = 2

DEFAULT_SERVED_MODEL = "hybrid"
DEFAULT_FALLBACK_MODEL = "naive"
DEFAULT_LEADERBOARD_MODELS = ("naive", "lightgbm", "hybrid")

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

