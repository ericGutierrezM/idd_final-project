from __future__ import annotations

import importlib.util
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype, is_float_dtype
from sklearn.ensemble import HistGradientBoostingRegressor

# Set up Lambda-friendly logging so each run can report success/failure details to CloudWatch.
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Keep the assignment submission window available as an explicit override/reference point.
ASSIGNMENT_FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
ASSIGNMENT_FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
HORIZON = 168
ZERO_LAG_THRESHOLD = 2

# Keep the gradient boosting hyperparameters in one place so model setup is easy to inspect.
HISTGBM_PARAMS = {
    "loss": "poisson",
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 42,
}

# Store the holiday calendar used for the holiday feature engineering step.
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

# These are the feature columns the model expects during training and prediction.
MODEL_FEATURES = [
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

# Lambda writes temporary files into its ephemeral temp directory before uploading to S3.
TMP_DIR = Path(os.getenv("LAMBDA_TMP_DIR", tempfile.gettempdir()))
TRAIN_LOCAL_PATH = TMP_DIR / "train_data.csv"
TEST_MOCK_LOCAL_PATH = TMP_DIR / "test_data_mock.csv"
PREDICTIONS_LOCAL_PATH = TMP_DIR / "predictions.csv"


def _env_or_default(event: Dict[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    # Read configuration from the event first, then fall back to environment variables.
    # This makes the same code work in AWS and in local/manual test runs.
    value = event.get(key)
    if value is not None:
        return str(value)
    env_value = os.getenv(key.upper())
    if env_value is not None:
        return env_value
    return default


def _bool_env_or_default(event: Dict[str, Any], key: str, default: bool = False) -> bool:
    # Same idea as _env_or_default, but specialized for boolean flags like RUN_OFFICIAL_CHECKER.
    value = event.get(key)
    if value is not None:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    env_value = os.getenv(key.upper())
    if env_value is None:
        return default
    return env_value.strip().lower() in {"1", "true", "yes", "y"}


def _normalize_prefix(prefix: str) -> str:
    # Ensure S3 snapshot prefixes always end with "/" so timestamped files land in the right folder.
    return prefix if prefix.endswith("/") else "{}{}".format(prefix, "/")


def _load_config(event: Dict[str, Any]) -> Dict[str, Any]:
    # Collect all runtime configuration values without hardcoding any bucket or key paths.
    input_bucket = _env_or_default(event, "input_bucket")
    train_data_key = _env_or_default(event, "train_data_key")
    test_mock_key = _env_or_default(event, "test_mock_key")
    output_latest_key = _env_or_default(event, "output_latest_key")
    output_snapshot_prefix = _env_or_default(event, "output_snapshot_prefix")
    forecast_start = _env_or_default(event, "forecast_start")
    forecast_end = _env_or_default(event, "forecast_end")
    run_official_checker = _bool_env_or_default(event, "run_official_checker", False)

    # Fail early if the Lambda is missing the minimum configuration it needs to run.
    required = {
        "input_bucket": input_bucket,
        "train_data_key": train_data_key,
        "output_latest_key": output_latest_key,
        "output_snapshot_prefix": output_snapshot_prefix,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError("Missing required Lambda config: {}".format(", ".join(missing)))

    # Optional overrides let us reproduce the assignment's exact grading week when needed.
    parsed_forecast_start = None if forecast_start is None else pd.Timestamp(forecast_start)
    parsed_forecast_end = None if forecast_end is None else pd.Timestamp(forecast_end)
    if (parsed_forecast_start is None) != (parsed_forecast_end is None):
        raise ValueError("forecast_start and forecast_end must be provided together")

    # Return one normalized config object for the rest of the handler to use.
    return {
        "input_bucket": str(input_bucket),
        "train_data_key": str(train_data_key),
        "test_mock_key": None if test_mock_key is None else str(test_mock_key),
        "output_latest_key": str(output_latest_key),
        "output_snapshot_prefix": _normalize_prefix(str(output_snapshot_prefix)),
        "forecast_start": parsed_forecast_start,
        "forecast_end": parsed_forecast_end,
        "run_official_checker": run_official_checker,
    }


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    # Load the training CSV, parse timestamps, and make sure the time series is hourly and ordered.
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Fill any missing hourly rows so downstream lag/rolling features have a consistent timeline.
    full_index = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    df = df.set_index("time").reindex(full_index).rename_axis("time").reset_index()
    df["orders"] = df["orders"].fillna(0.0)
    df["city"] = df["city"].ffill()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create the calendar and lag-based features used by the hybrid forecast model.
    featured = df.copy()
    featured["hour"] = featured["time"].dt.hour
    featured["dayofweek"] = featured["time"].dt.dayofweek
    featured["trend"] = np.arange(len(featured))
    featured["is_weekend"] = (featured["dayofweek"] >= 4).astype(int)
    featured["is_holiday"] = featured["time"].dt.normalize().isin(HOLIDAYS).astype(int)
    featured["lag_168"] = featured["orders"].shift(168)
    featured["lag_336"] = featured["orders"].shift(336)
    featured["roll_168_mean"] = featured["orders"].shift(168).rolling(168).mean()
    featured["roll_336_mean"] = featured["orders"].shift(168).rolling(336).mean()
    featured["zero_mask"] = (featured["lag_168"].fillna(999) <= ZERO_LAG_THRESHOLD).astype(int)
    return featured


def determine_forecast_window(
    history_df: pd.DataFrame,
    forecast_start: Optional[pd.Timestamp],
    forecast_end: Optional[pd.Timestamp],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Use explicit overrides when provided; otherwise forecast the next full Monday-Sunday week
    # after the latest observed training timestamp.
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


def build_forecast_frame(history_df: pd.DataFrame, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp) -> pd.DataFrame:
    # Create empty future rows covering the target 168-hour prediction horizon.
    forecast_index = pd.date_range(forecast_start, forecast_end, freq="h")
    city = history_df["city"].iloc[-1] if not history_df["city"].isna().all() else "BCN"
    return pd.DataFrame({"time": forecast_index, "orders": np.nan, "city": city})


def build_future_rows(raw_history: pd.DataFrame, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp) -> pd.DataFrame:
    # Append the future horizon to the historical data so the same feature logic can be reused.
    forecast_df = build_forecast_frame(raw_history, forecast_start, forecast_end)
    forecast_index = forecast_df["time"]
    full = pd.concat([raw_history, forecast_df]).reset_index(drop=True)
    full = add_features(full)
    return full[full["time"].isin(forecast_index)].copy()


def fit_histgbm(train_df: pd.DataFrame) -> HistGradientBoostingRegressor:
    # Train the machine learning component on rows where all engineered features are available.
    train_ready = train_df.dropna(subset=MODEL_FEATURES)
    model = HistGradientBoostingRegressor(**HISTGBM_PARAMS)
    model.fit(train_ready[MODEL_FEATURES], train_ready["orders"])
    return model


def predict_naive(train_df: pd.DataFrame, horizon: int = HORIZON) -> np.ndarray:
    # Baseline forecast: reuse the most recent horizon of observed values and clamp below zero.
    return np.maximum(train_df.tail(horizon)["orders"].values, 0)


def predict_hybrid(train_df: pd.DataFrame, future_rows: pd.DataFrame) -> np.ndarray:
    # Combine the naive baseline with the trained model.
    # When recent lag values are near zero, the code trusts the naive forecast; otherwise it uses HistGBM.
    naive_preds = predict_naive(train_df, horizon=len(future_rows))
    model = fit_histgbm(train_df)
    hist_preds = np.maximum(model.predict(future_rows[MODEL_FEATURES]), 0)
    mask = future_rows["zero_mask"].values.astype(bool)
    return np.where(mask, naive_preds, hist_preds)


def validate_predictions_format(predictions: pd.DataFrame, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp) -> None:
    # Enforce the required 168-hour output contract before anything gets written or uploaded.
    if len(predictions) != HORIZON:
        raise AssertionError("Expected {} rows, got {}".format(HORIZON, len(predictions)))
    if list(predictions.columns) != ["time", "preds"]:
        raise AssertionError("Wrong columns")
    if not is_float_dtype(predictions["preds"]):
        raise AssertionError("preds must be float64")
    if not is_datetime64_ns_dtype(predictions["time"]):
        raise AssertionError("time dtype: {}".format(predictions["time"].dtype))
    if predictions["preds"].isna().sum() != 0:
        raise AssertionError("NaN in preds")
    if predictions["time"].isna().sum() != 0:
        raise AssertionError("NaN in time")
    if predictions["time"].min() != forecast_start:
        raise AssertionError("Wrong start")
    if predictions["time"].max() != forecast_end:
        raise AssertionError("Wrong end")


def roundtrip_validate_csv(output_path: Path) -> None:
    # Reload the saved CSV once to confirm it still has the right shape and types after serialization.
    check = pd.read_csv(output_path, parse_dates=["time"])
    check["time"] = check["time"].astype("datetime64[ns]")
    if not is_float_dtype(check["preds"]):
        raise AssertionError("Round-trip preds must be float64")
    if len(check) != HORIZON:
        raise AssertionError("Round-trip row count mismatch")


def run_official_checker(predictions: pd.DataFrame, test_mock_path: Path) -> None:
    # Optionally run the course-provided checker script for an additional format validation pass.
    checker_file = Path(__file__).resolve().parent.parent / "check_output_format.py"
    if not checker_file.exists():
        raise FileNotFoundError("Checker not found at {}".format(checker_file))

    spec = importlib.util.spec_from_file_location("check_output_format", checker_file)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load checker from {}".format(checker_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_output_format(predictions, str(test_mock_path))


def generate_forecast(
    input_path: Path,
    output_path: Path,
    forecast_start: Optional[pd.Timestamp],
    forecast_end: Optional[pd.Timestamp],
    run_checker: bool,
    test_mock_path: Optional[Path],
) -> Dict[str, Any]:
    # This helper keeps forecasting logic separate from AWS I/O inside lambda_handler.
    raw_history = load_and_prepare_data(input_path)
    resolved_forecast_start, resolved_forecast_end = determine_forecast_window(raw_history, forecast_start, forecast_end)
    featured_history = add_features(raw_history)
    future_rows = build_future_rows(raw_history, resolved_forecast_start, resolved_forecast_end)

    # Build the final predictions table in the exact two-column format the downstream pipeline expects.
    predictions = pd.DataFrame(
        {
            "time": pd.date_range(resolved_forecast_start, resolved_forecast_end, freq="h").astype("datetime64[ns]"),
            "preds": np.asarray(predict_hybrid(featured_history, future_rows), dtype="float64"),
        }
    )

    validate_predictions_format(predictions, resolved_forecast_start, resolved_forecast_end)
    predictions.to_csv(output_path, index=False)
    roundtrip_validate_csv(output_path)

    # The official checker is optional so the same code can be used in lighter-weight runs too.
    if run_checker:
        if test_mock_path is None:
            raise ValueError("test_mock_path is required when run_official_checker=true")
        if resolved_forecast_start != ASSIGNMENT_FORECAST_START or resolved_forecast_end != ASSIGNMENT_FORECAST_END:
            raise ValueError(
                "The official checker expects the assignment window; provide matching forecast_start/forecast_end overrides"
            )
        run_official_checker(predictions, test_mock_path)

    # Return a small metadata summary so the handler can report what happened.
    return {
        "selected_model": "hybrid",
        "row_count": len(predictions),
        "forecast_start": resolved_forecast_start.isoformat(),
        "forecast_end": resolved_forecast_end.isoformat(),
    }


def lambda_handler(event: Optional[Dict[str, Any]], context: Any) -> Dict[str, Any]:
    # This is the AWS Lambda entry point.
    # AWS invokes this function, and it orchestrates config loading, S3 I/O, forecasting, and uploads.
    event = event or {}
    try:
        # Read runtime configuration from the event or environment variables so paths are not hardcoded.
        config = _load_config(event)
        s3 = boto3.client("s3")

        # Download the latest training data from S3 into Lambda's temporary local storage.
        s3.download_file(config["input_bucket"], config["train_data_key"], str(TRAIN_LOCAL_PATH))

        # If requested, also download the mock test file used by the official output checker.
        test_mock_path = None
        if config["run_official_checker"]:
            test_mock_key = config["test_mock_key"]
            if not test_mock_key:
                raise ValueError("test_mock_key is required when run_official_checker=true")
            s3.download_file(config["input_bucket"], test_mock_key, str(TEST_MOCK_LOCAL_PATH))
            test_mock_path = TEST_MOCK_LOCAL_PATH

        # Run the forecast pipeline and write the predictions CSV locally.
        run_meta = generate_forecast(
            input_path=TRAIN_LOCAL_PATH,
            output_path=PREDICTIONS_LOCAL_PATH,
            forecast_start=config["forecast_start"],
            forecast_end=config["forecast_end"],
            run_checker=bool(config["run_official_checker"]),
            test_mock_path=test_mock_path,
        )

        # Build a timestamped S3 key so every run keeps a historical snapshot of its predictions.
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_key = "{}{}_predictions.csv".format(config["output_snapshot_prefix"], run_ts)

        # Upload the same file twice:
        # 1) to a fixed "latest" location for easy access
        # 2) to a timestamped snapshot location for audit/history tracking
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), config["input_bucket"], config["output_latest_key"])
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), config["input_bucket"], snapshot_key)

        # Return a structured success response that can be logged or inspected in AWS.
        result = {
            "status": "success",
            "selected_model": run_meta["selected_model"],
            "row_count": run_meta["row_count"],
            "forecast_start": run_meta["forecast_start"],
            "forecast_end": run_meta["forecast_end"],
            "output_latest_key": config["output_latest_key"],
            "output_snapshot_key": snapshot_key,
        }
        LOGGER.info("Forecast run completed: %s", json.dumps(result))
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as exc:
        # If anything goes wrong, log the full error and return a 500 response for visibility.
        LOGGER.error("Forecast run failed: %s", str(exc), exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
