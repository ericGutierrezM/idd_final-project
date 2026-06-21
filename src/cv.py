from __future__ import annotations

import pandas as pd

from src.config import HORIZON, N_FOLDS


def get_cv_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    last = df["time"].max()
    for k in range(n_folds, 0, -1):
        val_end = last - pd.Timedelta(hours=(k - 1) * HORIZON)
        val_start = val_end - pd.Timedelta(hours=HORIZON - 1)
        train_end = val_start - pd.Timedelta(hours=1)
        train = df[df["time"] <= train_end].copy()
        val = df[(df["time"] >= val_start) & (df["time"] <= val_end)].copy()
        if len(val) == HORIZON and len(train) >= HORIZON * 2:
            folds.append((train, val))
    return folds

