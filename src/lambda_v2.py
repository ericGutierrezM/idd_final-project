import logging
import os
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype, is_float_dtype
from sklearn.ensemble import HistGradientBoostingRegressor

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ASSIGNMENT_FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
ASSIGNMENT_FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
HORIZON = 168

DEFAULT_BUCKET = "eric-gutierrez-moreno-bucket-idd"

HISTGBM_PARAMS = {
    "loss": "poisson",
    "max_iter": 500,
    "learning_rate": 0.05,
    "random_state": 42,
}

MODEL_FEATURES = [
    "month",
    "hour",
    "dayofweek",
    "is_weekend",
    "part_of_day",
    "lag_168",
    "roll_168_mean"
]

TMP_DIR = Path(os.getenv("LAMBDA_TMP_DIR", tempfile.gettempdir()))
TRAIN_LOCAL_PATH = TMP_DIR / "train_data.csv"
PREDICTIONS_LOCAL_PATH = TMP_DIR / "predictions.csv"


def _env_or_default(event: Dict[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    value = event.get(key)
    if value is not None:
        return str(value)
    env_value = os.getenv(key.upper())
    if env_value is not None:
        return env_value
    return default


def _load_config(event: Dict[str, Any]) -> Dict[str, Any]:
    input_bucket = _env_or_default(event, "input_bucket", DEFAULT_BUCKET)
    train_data_key = _env_or_default(event, "train_data_key", "train_data.csv")
    output_latest_key = _env_or_default(event, "output_latest_key", "predictions.csv")
    output_snapshot_prefix = _env_or_default(event, "output_snapshot_prefix", "snapshots/")
    forecast_start = _env_or_default(event, "forecast_start")
    forecast_end = _env_or_default(event, "forecast_end")

    parsed_forecast_start = None if forecast_start is None else pd.Timestamp(forecast_start)
    parsed_forecast_end = None if forecast_end is None else pd.Timestamp(forecast_end)
    if (parsed_forecast_start is None) != (parsed_forecast_end is None):
        raise ValueError("forecast_start and forecast_end must be provided together")

    snapshot_prefix = output_snapshot_prefix if str(output_snapshot_prefix).endswith("/") else "{}/".format(output_snapshot_prefix)

    return {
        "input_bucket": str(input_bucket),
        "train_data_key": str(train_data_key),
        "output_latest_key": str(output_latest_key),
        "output_snapshot_prefix": snapshot_prefix,
        "forecast_start": parsed_forecast_start,
        "forecast_end": parsed_forecast_end,
    }

def get_part_of_day(hour: int) -> int:
    if 5 <= hour < 11:
        return 0 
    elif 11 <= hour < 14:
        return 1
    elif 14 <= hour < 18:
        return 2 
    else:
        return 3

def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    full_index = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    df = df.set_index("time").reindex(full_index).rename_axis("time").reset_index()
    df["orders"] = df["orders"].fillna(0.0)
    df["city"] = df["city"].ffill()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    
    featured["month"] = featured["time"].dt.month
    featured["hour"] = featured["time"].dt.hour
    featured["dayofweek"] = featured["time"].dt.dayofweek
    featured["is_weekend"] = featured["dayofweek"].isin([5, 6]).astype(int)
    featured["part_of_day"] = featured["hour"].apply(get_part_of_day)
    
    featured["lag_168"] = featured["orders"].shift(168)
    featured["roll_168_mean"] = featured["orders"].shift(168).rolling(168).mean()
    
    return featured

def determine_forecast_window(
    history_df: pd.DataFrame,
    forecast_start: Optional[pd.Timestamp],
    forecast_end: Optional[pd.Timestamp],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if forecast_start is not None and forecast_end is not None:
        start = forecast_start
        end = forecast_end
    else:
        latest_time = pd.Timestamp(history_df["time"].max())
        next_hour = latest_time + pd.Timedelta(hours=1)
        start = next_hour.normalize()
        end = start + pd.Timedelta(hours=HORIZON - 1)

    expected_end = start + pd.Timedelta(hours=HORIZON - 1)
    if end != expected_end:
        raise ValueError(
            "Forecast window must span exactly {} hours; got {} to {}".format(HORIZON, start, end)
        )
    if start.hour != 0 or end.hour != 23:
        raise ValueError("Forecast window must run from 00:00 to 23:00")
    if start.dayofweek != 0 or end.dayofweek != 6:
        raise ValueError("Forecast window must run from Monday to Sunday")
    return start, end


def build_forecast_frame(
    history_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    forecast_index = pd.date_range(forecast_start, forecast_end, freq="h")
    city = history_df["city"].iloc[-1] if not history_df["city"].isna().all() else "BCN"
    return pd.DataFrame({"time": forecast_index, "orders": np.nan, "city": city})

def build_future_rows(
    raw_history: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    forecast_df = build_forecast_frame(raw_history, forecast_start, forecast_end)
    forecast_index = forecast_df["time"]
    full = pd.concat([raw_history, forecast_df]).reset_index(drop=True)
    full = add_features(full)
    return full[full["time"].isin(forecast_index)].copy()

def validate_predictions_format(
    predictions: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> None:
    if len(predictions) != HORIZON:
        raise AssertionError(f"Expected {HORIZON} rows, got {len(predictions)}")
    if list(predictions.columns) != ["time", "preds"]:
        raise AssertionError("Output must contain exactly 'time' and 'preds' columns")
    if not is_float_dtype(predictions["preds"]):
        raise AssertionError("preds must be float64")
    if not is_datetime64_ns_dtype(predictions["time"]):
        raise AssertionError("time must be datetime64[ns]")
    if predictions["time"].min() != forecast_start:
        raise AssertionError("Wrong start")
    if predictions["time"].max() != forecast_end:
        raise AssertionError("Wrong end")

def generate_forecast(
    input_path: Path,
    output_path: Path,
    forecast_start: Optional[pd.Timestamp],
    forecast_end: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    raw_history = load_and_prepare_data(input_path)
    resolved_forecast_start, resolved_forecast_end = determine_forecast_window(raw_history, forecast_start, forecast_end)
    featured_history = add_features(raw_history)
    future_rows = build_future_rows(raw_history, resolved_forecast_start, resolved_forecast_end)

    train_ready = featured_history.dropna(subset=MODEL_FEATURES)
    model = HistGradientBoostingRegressor(**HISTGBM_PARAMS)
    model.fit(train_ready[MODEL_FEATURES], train_ready["orders"])

    raw_preds = model.predict(future_rows[MODEL_FEATURES])
    safe_preds = np.maximum(raw_preds, 0)

    predictions = pd.DataFrame({
        "time": pd.date_range(resolved_forecast_start, resolved_forecast_end, freq="h").astype("datetime64[ns]"),
        "preds": np.asarray(safe_preds, dtype="float64"),
    })

    validate_predictions_format(predictions, resolved_forecast_start, resolved_forecast_end)
    predictions.to_csv(output_path, index=False)
    
    return {
        "row_count": len(predictions),
        "forecast_start": resolved_forecast_start.isoformat(),
        "forecast_end": resolved_forecast_end.isoformat(),
    }

def lambda_handler(event: Optional[Dict[str, Any]], context: Any) -> Dict[str, Any]:
    event = event or {}
    try:
        config = _load_config(event)
        
        s3 = boto3.client("s3")

        LOGGER.info(f"Downloading {config['train_data_key']} from {config['input_bucket']}")
        s3.download_file(config["input_bucket"], config["train_data_key"], str(TRAIN_LOCAL_PATH))

        LOGGER.info("Generating predictions...")
        run_meta = generate_forecast(
            input_path=TRAIN_LOCAL_PATH,
            output_path=PREDICTIONS_LOCAL_PATH,
            forecast_start=config["forecast_start"],
            forecast_end=config["forecast_end"],
        )

        LOGGER.info(f"Uploading {config['output_latest_key']} to {config['input_bucket']}")
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), config["input_bucket"], config["output_latest_key"])
        
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_key = f"{config['output_snapshot_prefix']}{run_ts}_predictions.csv"
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), config["input_bucket"], snapshot_key)

        result = {
            "status": "success",
            "row_count": run_meta["row_count"],
            "forecast_start": run_meta["forecast_start"],
            "forecast_end": run_meta["forecast_end"],
            "output_latest_key": f"s3://{config['input_bucket']}/{config['output_latest_key']}",
            "output_snapshot_key": f"s3://{config['input_bucket']}/{snapshot_key}",
        }
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as exc:
        LOGGER.error(f"Forecast run failed: {str(exc)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
