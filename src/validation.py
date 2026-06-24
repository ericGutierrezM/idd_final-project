from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype, is_float_dtype

from src.config import FORECAST_END, FORECAST_START, HORIZON


def validate_predictions_format(predictions: pd.DataFrame) -> None:
    if len(predictions) != HORIZON:
        raise AssertionError(f"Expected {HORIZON} rows, got {len(predictions)}")
    if list(predictions.columns) != ["time", "preds"]:
        raise AssertionError("Wrong columns")
    if not is_float_dtype(predictions["preds"]):
        raise AssertionError("preds must be float64")
    if not is_datetime64_ns_dtype(predictions["time"]):
        raise AssertionError(f"time dtype: {predictions['time'].dtype}")
    if predictions["preds"].isna().sum() != 0:
        raise AssertionError("NaN in preds")
    if predictions["time"].isna().sum() != 0:
        raise AssertionError("NaN in time")
    if predictions["time"].min() != FORECAST_START:
        raise AssertionError("Wrong start")
    if predictions["time"].max() != FORECAST_END:
        raise AssertionError("Wrong end")


def roundtrip_validate_csv(output_path: Union[str, Path]) -> pd.DataFrame:
    check = pd.read_csv(output_path, parse_dates=["time"])
    check["time"] = check["time"].astype("datetime64[ns]")
    if not is_float_dtype(check["preds"]):
        raise AssertionError("Round-trip preds must be float64")
    if len(check) != HORIZON:
        raise AssertionError("Round-trip row count mismatch")
    return check


def run_official_checker(
    predictions: pd.DataFrame,
    test_mock_path: Union[str, Path],
    checker_path: Optional[Union[str, Path]] = None,
) -> None:
    checker_file = Path(checker_path) if checker_path else Path(__file__).resolve().parent.parent / "check_output_format.py"
    if not checker_file.exists():
        raise FileNotFoundError(f"Checker not found at {checker_file}")

    spec = importlib.util.spec_from_file_location("check_output_format", checker_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load checker from {checker_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_output_format(predictions, str(test_mock_path))
