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

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

FORECAST_START = pd.Timestamp("2022-01-24 00:00:00")
FORECAST_END = pd.Timestamp("2022-01-30 23:00:00")
HORIZON = 168
ZERO_LAG_THRESHOLD = 2

HISTGBM_PARAMS = {
    "loss": "poisson",
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 42,
}

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

TMP_DIR = Path(os.getenv("LAMBDA_TMP_DIR", tempfile.gettempdir()))
TRAIN_LOCAL_PATH = TMP_DIR / "train_data.csv"
TEST_MOCK_LOCAL_PATH = TMP_DIR / "test_data_mock.csv"
PREDICTIONS_LOCAL_PATH = TMP_DIR / "predictions.csv"


def _env_or_default(event: Dict[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    value = event.get(key)
    if value is not None:
        return str(value)
    env_value = os.getenv(key.upper())
    if env_value is not None:
        return env_value
    return default


def _bool_env_or_default(event: Dict[str, Any], key: str, default: bool = False) -> bool:
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
    return prefix if prefix.endswith("/") else "{}{}".format(prefix, "/")


def _load_config(event: Dict[str, Any]) -> Dict[str, Any]:
    input_bucket = _env_or_default(event, "input_bucket")
    train_data_key = _env_or_default(event, "train_data_key")
    test_mock_key = _env_or_default(event, "test_mock_key")
    output_latest_key = _env_or_default(event, "output_latest_key")
    output_snapshot_prefix = _env_or_default(event, "output_snapshot_prefix")
    run_official_checker = _bool_env_or_default(event, "run_official_checker", False)

    required = {
        "input_bucket": input_bucket,
        "train_data_key": train_data_key,
        "output_latest_key": output_latest_key,
        "output_snapshot_prefix": output_snapshot_prefix,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError("Missing required Lambda config: {}".format(", ".join(missing)))

    return {
        "input_bucket": str(input_bucket),
        "train_data_key": str(train_data_key),
        "test_mock_key": None if test_mock_key is None else str(test_mock_key),
        "output_latest_key": str(output_latest_key),
        "output_snapshot_prefix": _normalize_prefix(str(output_snapshot_prefix)),
        "run_official_checker": run_official_checker,
    }


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


def fit_histgbm(train_df: pd.DataFrame) -> HistGradientBoostingRegressor:
    train_ready = train_df.dropna(subset=MODEL_FEATURES)
    model = HistGradientBoostingRegressor(**HISTGBM_PARAMS)
    model.fit(train_ready[MODEL_FEATURES], train_ready["orders"])
    return model


def predict_naive(train_df: pd.DataFrame, horizon: int = HORIZON) -> np.ndarray:
    return np.maximum(train_df.tail(horizon)["orders"].values, 0)


def predict_hybrid(train_df: pd.DataFrame, future_rows: pd.DataFrame) -> np.ndarray:
    naive_preds = predict_naive(train_df, horizon=len(future_rows))
    model = fit_histgbm(train_df)
    hist_preds = np.maximum(model.predict(future_rows[MODEL_FEATURES]), 0)
    mask = future_rows["zero_mask"].values.astype(bool)
    return np.where(mask, naive_preds, hist_preds)


def validate_predictions_format(predictions: pd.DataFrame) -> None:
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
    if predictions["time"].min() != FORECAST_START:
        raise AssertionError("Wrong start")
    if predictions["time"].max() != FORECAST_END:
        raise AssertionError("Wrong end")


def roundtrip_validate_csv(output_path: Path) -> None:
    check = pd.read_csv(output_path, parse_dates=["time"])
    check["time"] = check["time"].astype("datetime64[ns]")
    if not is_float_dtype(check["preds"]):
        raise AssertionError("Round-trip preds must be float64")
    if len(check) != HORIZON:
        raise AssertionError("Round-trip row count mismatch")


def run_official_checker(predictions: pd.DataFrame, test_mock_path: Path) -> None:
    checker_file = Path(__file__).resolve().parent.parent / "check_output_format.py"
    if not checker_file.exists():
        raise FileNotFoundError("Checker not found at {}".format(checker_file))

    spec = importlib.util.spec_from_file_location("check_output_format", checker_file)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load checker from {}".format(checker_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_output_format(predictions, str(test_mock_path))


def generate_forecast(input_path: Path, output_path: Path, run_checker: bool, test_mock_path: Optional[Path]) -> Dict[str, Any]:
    raw_history = load_and_prepare_data(input_path)
    featured_history = add_features(raw_history)
    future_rows = build_future_rows(raw_history)

    predictions = pd.DataFrame(
        {
            "time": pd.date_range(FORECAST_START, FORECAST_END, freq="h").astype("datetime64[ns]"),
            "preds": np.asarray(predict_hybrid(featured_history, future_rows), dtype="float64"),
        }
    )

    validate_predictions_format(predictions)
    predictions.to_csv(output_path, index=False)
    roundtrip_validate_csv(output_path)

    if run_checker:
        if test_mock_path is None:
            raise ValueError("test_mock_path is required when run_official_checker=true")
        run_official_checker(predictions, test_mock_path)

    return {
        "selected_model": "hybrid",
        "row_count": len(predictions),
    }


def lambda_handler(event: Optional[Dict[str, Any]], context: Any) -> Dict[str, Any]:
    event = event or {}
    try:
        config = _load_config(event)
        s3 = boto3.client("s3")

        s3.download_file(config["input_bucket"], config["train_data_key"], str(TRAIN_LOCAL_PATH))

        test_mock_path = None
        if config["run_official_checker"]:
            test_mock_key = config["test_mock_key"]
            if not test_mock_key:
                raise ValueError("test_mock_key is required when run_official_checker=true")
            s3.download_file(config["input_bucket"], test_mock_key, str(TEST_MOCK_LOCAL_PATH))
            test_mock_path = TEST_MOCK_LOCAL_PATH

        run_meta = generate_forecast(
            input_path=TRAIN_LOCAL_PATH,
            output_path=PREDICTIONS_LOCAL_PATH,
            run_checker=bool(config["run_official_checker"]),
            test_mock_path=test_mock_path,
        )

        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_key = "{}{}_predictions.csv".format(config["output_snapshot_prefix"], run_ts)

        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), config["input_bucket"], config["output_latest_key"])
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), config["input_bucket"], snapshot_key)

        result = {
            "status": "success",
            "selected_model": run_meta["selected_model"],
            "row_count": run_meta["row_count"],
            "output_latest_key": config["output_latest_key"],
            "output_snapshot_key": snapshot_key,
        }
        LOGGER.info("Forecast run completed: %s", json.dumps(result))
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as exc:
        LOGGER.error("Forecast run failed: %s", str(exc), exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
