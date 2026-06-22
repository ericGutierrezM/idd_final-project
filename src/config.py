from __future__ import annotations

import pandas as pd

FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
HORIZON = 168
N_FOLDS = 5
ZERO_LAG_THRESHOLD = 2

DEFAULT_SERVED_MODEL = "hybrid"
DEFAULT_FALLBACK_MODEL = "naive"
DEFAULT_LEADERBOARD_MODELS = ("naive", "histgbm", "hybrid")

# HistGradientBoostingRegressor — sklearn's native gradient boosting.
# Poisson loss is appropriate for non-negative count data (analogous to
# LightGBM Tweedie). Handles NaN features natively, no extra C library needed.
HISTGBM_PARAMS = {
    "loss": "poisson",
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 42,
}

