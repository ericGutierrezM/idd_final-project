from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Optional

from src.config import HISTGBM_PARAMS, HORIZON
from src.features import MODEL_FEATURES, add_features


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    aliases = {
        "naive": "naive",
        "histgbm": "histgbm",
        "hist": "histgbm",
        "hybrid": "hybrid",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: naive, histgbm, hybrid.")
    return aliases[normalized]


def predict_naive(train: pd.DataFrame, val: Optional[pd.DataFrame] = None, horizon: int = HORIZON) -> np.ndarray:
    return np.maximum(train.tail(horizon)["orders"].values, 0)


def fit_final_histgbm(train_df: pd.DataFrame) -> HistGradientBoostingRegressor:
    train_ready = train_df.dropna(subset=MODEL_FEATURES)
    x_train = train_ready[MODEL_FEATURES]
    y_train = train_ready["orders"]
    model = HistGradientBoostingRegressor(**HISTGBM_PARAMS)
    model.fit(x_train, y_train)
    return model


def predict_histgbm(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    train_ready = train.dropna(subset=MODEL_FEATURES)
    x_train = train_ready[MODEL_FEATURES]
    y_train = train_ready["orders"]

    combined = pd.concat([train, val]).reset_index(drop=True)
    combined = add_features(combined)
    x_val = combined.loc[combined["time"].isin(val["time"].values), MODEL_FEATURES]

    model = HistGradientBoostingRegressor(**HISTGBM_PARAMS)
    model.fit(x_train, y_train)
    return np.maximum(model.predict(x_val), 0)


def predict_hybrid(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    naive_preds = predict_naive(train, val)
    histgbm_preds = predict_histgbm(train, val)
    mask = val["zero_mask"].values.astype(bool)
    return np.where(mask, naive_preds, histgbm_preds)


def predict_future_histgbm(model: HistGradientBoostingRegressor, future_rows: pd.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(future_rows[MODEL_FEATURES]), 0)


def select_model_predictions(
    selected_model: str,
    train_df: pd.DataFrame,
    future_rows: pd.DataFrame,
) -> np.ndarray:
    model_name = normalize_model_name(selected_model)
    if model_name == "naive":
        return predict_naive(train_df, horizon=len(future_rows))
    if model_name == "histgbm":
        model = fit_final_histgbm(train_df)
        return predict_future_histgbm(model, future_rows)

    # hybrid: zero_mask hours → naive, active hours → histgbm
    naive_preds = predict_naive(train_df, horizon=len(future_rows))
    model = fit_final_histgbm(train_df)
    histgbm_preds = predict_future_histgbm(model, future_rows)
    mask = future_rows["zero_mask"].values.astype(bool)
    return np.where(mask, naive_preds, histgbm_preds)
