from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

from src.config import DEFAULT_FALLBACK_MODEL, DEFAULT_SERVED_MODEL
from src.forecast import generate_forecast

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

TMP_DIR = Path(os.getenv("LAMBDA_TMP_DIR", tempfile.gettempdir()))
TRAIN_LOCAL_PATH = TMP_DIR / "train_data.csv"
TEST_MOCK_LOCAL_PATH = TMP_DIR / "test_data_mock.csv"
PREDICTIONS_LOCAL_PATH = TMP_DIR / "predictions.csv"


def _env_or_default(event: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = event.get(key)
    if value is not None:
        return str(value)
    env_key = key.upper()
    env_value = os.getenv(env_key)
    if env_value is not None:
        return env_value
    return default


def _bool_env_or_default(event: dict[str, Any], key: str, default: bool = False) -> bool:
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
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _load_config(event: dict[str, Any]) -> dict[str, Any]:
    env_name = _env_or_default(event, "env", "test")
    input_bucket = _env_or_default(event, "input_bucket")
    train_data_key = _env_or_default(event, "train_data_key")
    test_mock_key = _env_or_default(event, "test_mock_key")
    output_latest_key = _env_or_default(event, "output_latest_key")
    output_snapshot_prefix = _env_or_default(event, "output_snapshot_prefix")
    model_name = _env_or_default(event, "model_name", DEFAULT_SERVED_MODEL)
    fallback_model = _env_or_default(event, "fallback_model", DEFAULT_FALLBACK_MODEL)
    run_official_checker = _bool_env_or_default(event, "run_official_checker", False)

    required = {
        "input_bucket": input_bucket,
        "train_data_key": train_data_key,
        "output_latest_key": output_latest_key,
        "output_snapshot_prefix": output_snapshot_prefix,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required Lambda config: {', '.join(missing)}")

    return {
        "env": env_name,
        "input_bucket": input_bucket,
        "train_data_key": train_data_key,
        "test_mock_key": test_mock_key,
        "output_latest_key": output_latest_key,
        "output_snapshot_prefix": _normalize_prefix(output_snapshot_prefix),
        "model_name": model_name,
        "fallback_model": fallback_model,
        "run_official_checker": run_official_checker,
    }


def lambda_handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    event = event or {}
    try:
        config = _load_config(event)
        s3 = boto3.client("s3")

        LOGGER.info("Starting forecast run with config: %s", json.dumps({k: v for k, v in config.items() if k != "input_bucket"}))

        bucket = config["input_bucket"]
        s3.download_file(bucket, config["train_data_key"], str(TRAIN_LOCAL_PATH))

        test_mock_path: str | None = None
        if config["run_official_checker"]:
            test_mock_key = config["test_mock_key"]
            if not test_mock_key:
                raise ValueError("test_mock_key is required when run_official_checker=true")
            s3.download_file(bucket, test_mock_key, str(TEST_MOCK_LOCAL_PATH))
            test_mock_path = str(TEST_MOCK_LOCAL_PATH)

        predictions, run_meta = generate_forecast(
            input_path=str(TRAIN_LOCAL_PATH),
            model_name=str(config["model_name"]),
            fallback_model=str(config["fallback_model"]),
            output_path=str(PREDICTIONS_LOCAL_PATH),
            run_checker=bool(config["run_official_checker"]),
            test_mock_path=test_mock_path,
        )

        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_key = f"{config['output_snapshot_prefix']}{run_ts}_predictions.csv"

        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), bucket, str(config["output_latest_key"]))
        s3.upload_file(str(PREDICTIONS_LOCAL_PATH), bucket, snapshot_key)

        result = {
            "status": "success",
            "env": config["env"],
            "input_bucket": bucket,
            "selected_model": run_meta["selected_model"],
            "served_model": run_meta["served_model"],
            "used_fallback": run_meta["used_fallback"],
            "row_count": run_meta["row_count"],
            "output_latest_key": config["output_latest_key"],
            "output_snapshot_key": snapshot_key,
        }
        LOGGER.info("Forecast run completed: %s", json.dumps(result))
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as e:
        LOGGER.error("Forecast run failed: %s", str(e), exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
