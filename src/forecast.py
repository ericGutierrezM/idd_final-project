from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_SERVED_MODEL,
    FORECAST_END,
    FORECAST_START,
)
from src.data import load_and_prepare_data
from src.features import add_features, LGBM_FEATURES
from src.models import normalize_model_name, select_model_predictions
from src.validation import (
    roundtrip_validate_csv,
    run_official_checker,
    validate_predictions_format,
)


def build_forecast_frame(history_df: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    forecast_index = pd.date_range(FORECAST_START, FORECAST_END, freq="h")
    city = history_df["city"].iloc[-1] if not history_df["city"].isna().all() else "BCN"
    forecast_df = pd.DataFrame({"time": forecast_index, "orders": np.nan, "city": city})
    return forecast_index, forecast_df


def _future_feature_rows(raw_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_index, forecast_df = build_forecast_frame(raw_history)
    full = pd.concat([raw_history, forecast_df]).reset_index(drop=True)
    full = add_features(full)
    future_rows = full[full["time"].isin(forecast_index)].copy()
    return future_rows, pd.DataFrame({"time": forecast_index})


def generate_forecast(
    input_path: str | Path,
    model_name: str = DEFAULT_SERVED_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    output_path: str | Path | None = None,
    run_checker: bool = False,
    test_mock_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_model = normalize_model_name(model_name)
    fallback = normalize_model_name(fallback_model)

    raw_history = load_and_prepare_data(input_path)
    featured_history = add_features(raw_history)
    future_rows, forecast_times = _future_feature_rows(raw_history)

    future_has_nan = future_rows[LGBM_FEATURES].isna().any().any()
    used_fallback = False
    served_model = selected_model
    fallback_reason = None

    try:
        preds = select_model_predictions(selected_model, featured_history, future_rows)
    except Exception as exc:
        if fallback != "naive":
            raise
        used_fallback = True
        served_model = fallback
        fallback_reason = str(exc)
        preds = select_model_predictions(fallback, featured_history, future_rows)

    predictions = pd.DataFrame(
        {
            "time": pd.to_datetime(forecast_times["time"]).astype("datetime64[ns]"),
            "preds": np.asarray(preds, dtype="float64"),
        }
    )
    validate_predictions_format(predictions)

    if output_path is not None:
        predictions.to_csv(output_path, index=False)
        roundtrip_validate_csv(output_path)

    if run_checker:
        if test_mock_path is None:
            raise ValueError("test_mock_path is required when run_checker=True")
        run_official_checker(predictions, test_mock_path)

    run_meta: dict[str, object] = {
        "selected_model": selected_model,
        "served_model": served_model,
        "fallback_model": fallback,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "forecast_start": str(FORECAST_START),
        "forecast_end": str(FORECAST_END),
        "row_count": len(predictions),
        "future_feature_nan": bool(future_has_nan),
    }
    return predictions, run_meta

