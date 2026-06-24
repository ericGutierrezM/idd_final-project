from __future__ import annotations

import numpy as np
from typing import Dict


def smape(actual, pred) -> float:
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = np.abs(actual) + np.abs(pred)
    values = np.zeros_like(denom, dtype=float)
    nonzero = denom != 0
    values[nonzero] = 2 * np.abs(actual[nonzero] - pred[nonzero]) / denom[nonzero]
    return np.mean(values) * 100


def mse(actual, pred) -> float:
    return np.mean((np.array(actual) - np.array(pred)) ** 2)


def evaluate(actual, pred) -> Dict[str, float]:
    pred_clipped = np.maximum(pred, 0)
    return {
        "mse": mse(actual, pred_clipped),
        "smape": smape(actual, pred_clipped),
    }
