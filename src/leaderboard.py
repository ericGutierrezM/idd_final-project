from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_LEADERBOARD_MODELS
from src.cv import get_cv_folds
from src.data import load_and_prepare_data
from src.features import add_features
from src.metrics import evaluate
from src.models import normalize_model_name, predict_hybrid, predict_histgbm, predict_naive


MODEL_PREDICTORS = {
    "naive": predict_naive,
    "histgbm": predict_histgbm,
    "hybrid": predict_hybrid,
}


def compute_leaderboard(
    input_path: str | Path,
    output_path: str | Path | None = None,
    models: tuple[str, ...] = DEFAULT_LEADERBOARD_MODELS,
) -> pd.DataFrame:
    raw_history = load_and_prepare_data(input_path)
    featured_history = add_features(raw_history)
    folds = get_cv_folds(featured_history)

    rows: list[dict[str, object]] = []
    for model_name in models:
        normalized = normalize_model_name(model_name)
        predictor = MODEL_PREDICTORS[normalized]
        scores = [
            evaluate(val["orders"].values, predictor(train, val))
            for train, val in folds
        ]
        avg_mse = sum(score["mse"] for score in scores) / len(scores)
        avg_smape = sum(score["smape"] for score in scores) / len(scores)
        rows.append(
            {
                "model": normalized,
                "avg_mse": avg_mse,
                "avg_smape": avg_smape,
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("avg_smape").reset_index(drop=True)
    leaderboard["is_champion"] = False
    if not leaderboard.empty:
        leaderboard.loc[0, "is_champion"] = True

    if output_path is not None:
        leaderboard.to_csv(output_path, index=False)
    return leaderboard

