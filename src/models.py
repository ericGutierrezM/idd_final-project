from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import HORIZON, LGBM_PARAMS
from src.features import LGBM_FEATURES, add_features


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    aliases = {
        "naive": "naive",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "hybrid": "hybrid",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model '{model_name}'.")
    return aliases[normalized]


def predict_naive(train: pd.DataFrame, val: pd.DataFrame | None = None, horizon: int = HORIZON) -> np.ndarray:
    return np.maximum(train.tail(horizon)["orders"].values, 0)


def fit_final_lgbm(train_df: pd.DataFrame) -> lgb.LGBMRegressor:
    train_ready = train_df.dropna(subset=LGBM_FEATURES)
    x_train = train_ready[LGBM_FEATURES]
    y_train = train_ready["orders"]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(x_train, y_train)
    return model


def predict_lgbm(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    train_ready = train.dropna(subset=LGBM_FEATURES)
    x_train = train_ready[LGBM_FEATURES]
    y_train = train_ready["orders"]

    combined = pd.concat([train, val]).reset_index(drop=True)
    combined = add_features(combined)
    x_val = combined.loc[combined["time"].isin(val["time"].values), LGBM_FEATURES]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(x_train, y_train)
    return np.maximum(model.predict(x_val), 0)


def predict_hybrid(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    naive_preds = predict_naive(train, val)
    lgbm_preds = predict_lgbm(train, val)
    mask = val["zero_mask"].values.astype(bool)
    return np.where(mask, naive_preds, lgbm_preds)


def predict_future_lgbm(model: lgb.LGBMRegressor, future_rows: pd.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(future_rows[LGBM_FEATURES]), 0)


def select_model_predictions(
    selected_model: str,
    train_df: pd.DataFrame,
    future_rows: pd.DataFrame,
) -> np.ndarray:
    model_name = normalize_model_name(selected_model)
    if model_name == "naive":
        return predict_naive(train_df, horizon=len(future_rows))
    if model_name == "lightgbm":
        model = fit_final_lgbm(train_df)
        return predict_future_lgbm(model, future_rows)

    naive_preds = predict_naive(train_df, horizon=len(future_rows))
    model = fit_final_lgbm(train_df)
    lgbm_preds = predict_future_lgbm(model, future_rows)
    mask = future_rows["zero_mask"].values.astype(bool)
    return np.where(mask, naive_preds, lgbm_preds)

