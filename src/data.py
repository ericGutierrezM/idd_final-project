from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_and_prepare_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    full_index = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    df = df.set_index("time").reindex(full_index).rename_axis("time").reset_index()
    df["orders"] = df["orders"].fillna(0.0)
    df["city"] = df["city"].ffill()
    return df

