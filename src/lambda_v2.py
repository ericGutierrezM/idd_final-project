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

FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
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

def build_forecast_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    forecast_index = pd.date_range(FORECAST_START, FORECAST_END, freq="h")
    city = history_df["city"].iloc[-1] if not history_df["city"].isna().all() else "BCN"
    return pd.DataFrame({"time": forecast_index, "orders": np.nan, "city": city})

def build_future_rows(raw_history: pd.DataFrame) -> pd.DataFrame:
    forecast_df = build_forecast_frame(raw_history)
    forecast_index = forecast_df["time"]
    full = pd.concat([raw_history, forecast_df]).reset_index(drop=True)
    full = add_features(full)
    return full[full["time"].isin(forecast_index)].copy()

def validate_predictions_format(predictions: pd.DataFrame) -> None:
    if len(predictions) != HORIZON:
        raise AssertionError(f"Expected {HORIZON} rows, got {len(predictions)}")
    if list(predictions.columns) != ["time", "preds"]:
        raise AssertionError("Output must contain exactly 'time' and 'preds' columns")
    if not is_float_dtype(predictions["preds"]):
        raise AssertionError("preds must be float64")
    if not is_datetime64_ns_dtype(predictions["time"]):
        raise AssertionError("time must be datetime64[ns]")

def generate_forecast(input_path: Path, output_path: Path) -> int:
    raw_history = load_and_prepare_data(input_path)
    featured_history = add_features(raw_history)
    future_rows = build_future_rows(raw_history)

    train_ready = featured_history.dropna(subset=MODEL_FEATURES)
    model = HistGradientBoostingRegressor(**HISTGBM_PARAMS)
    model.fit(train_ready[MODEL_FEATURES], train_ready["orders"])

    raw_preds = model.predict(future_rows[MODEL_FEATURES])
    safe_preds = np.maximum(raw_preds, 0)

    predictions = pd.DataFrame({
        "time": pd.date_range(FORECAST_START, FORECAST_END, freq="h").astype("datetime64[ns]"),
        "preds": np.asarray(safe_preds, dtype="float64"),
    })

    validate_predictions_format(predictions)
    predictions.to_csv(output_path, index=False)
    
    return len(predictions)

def lambda_handler(event: Optional[Dict[str, Any]], context: Any) -> Dict[str, Any]:
    try:
        bucket_name = os.getenv("INPUT_BUCKET", DEFAULT_BUCKET)
        train_key = os.getenv("TRAIN_DATA_KEY", "train_data.csv")
        output_key = os.getenv("OUTPUT_LATEST_KEY", "predictions.csv")
        
        s3 = boto3.client("s3")

        LOGGER.info(f"Downloading {train_key} from {bucket_name}")
        s3.download_file(bucket_name, train_key, str(TRAIN_LOCAL_PATH))

        LOGGER.info("Generating predictions...")
        row_count = generate_forecast(
            input_path=TRAIN_LOCAL_PATH,
            output_path=PREDICTIONS_LOCAL_PATH
        )

        LOGGER.info(f"Uploading {output_key} to {bucket_name}")
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), bucket_name, output_key)
        
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_key = f"snapshots/{run_ts}_predictions.csv"
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), bucket_name, snapshot_key)

        result = {
            "status": "success",
            "row_count": row_count,
            "output_latest_key": f"s3://{bucket_name}/{output_key}"
        }
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as exc:
        LOGGER.error(f"Forecast run failed: {str(exc)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}