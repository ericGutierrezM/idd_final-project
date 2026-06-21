from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ZERO_LAG_THRESHOLD

HOLIDAYS = pd.to_datetime(
    [
        "2021-01-01",
        "2021-01-06",
        "2021-04-02",
        "2021-04-05",
        "2021-05-01",
        "2021-06-24",
        "2021-08-15",
        "2021-09-11",
        "2021-09-24",
        "2021-10-12",
        "2021-11-01",
        "2021-12-06",
        "2021-12-08",
        "2021-12-25",
        "2021-12-26",
        "2022-01-01",
        "2022-01-06",
    ]
)

LGBM_FEATURES = [
    "hour",
    "dayofweek",
    "trend",
    "is_weekend",
    "is_holiday",
    "zero_mask",
    "lag_168",
    "lag_336",
    "roll_168_mean",
    "roll_336_mean",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["hour"] = featured["time"].dt.hour
    featured["dayofweek"] = featured["time"].dt.dayofweek
    featured["week"] = featured["time"].dt.isocalendar().week.astype(int)
    featured["trend"] = np.arange(len(featured))
    featured["is_weekend"] = (featured["dayofweek"] >= 4).astype(int)
    featured["is_holiday"] = featured["time"].dt.normalize().isin(HOLIDAYS).astype(int)

    featured["lag_168"] = featured["orders"].shift(168)
    featured["lag_336"] = featured["orders"].shift(336)
    featured["roll_168_mean"] = featured["orders"].shift(168).rolling(168).mean()
    featured["roll_336_mean"] = featured["orders"].shift(168).rolling(336).mean()
    featured["zero_mask"] = (
        featured["lag_168"].fillna(999) <= ZERO_LAG_THRESHOLD
    ).astype(int)
    return featured

