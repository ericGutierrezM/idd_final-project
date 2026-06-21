from __future__ import annotations

import argparse
import json

from src.config import DEFAULT_FALLBACK_MODEL, DEFAULT_SERVED_MODEL
from src.forecast import generate_forecast
from src.leaderboard import compute_leaderboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast_parser = subparsers.add_parser("forecast", help="Generate the weekly forecast CSV")
    forecast_parser.add_argument("--input", required=True, help="Path to training CSV")
    forecast_parser.add_argument("--output", required=True, help="Path to output predictions CSV")
    forecast_parser.add_argument("--model", default=DEFAULT_SERVED_MODEL, help="Served model name")
    forecast_parser.add_argument("--fallback", default=DEFAULT_FALLBACK_MODEL, help="Fallback model name")
    forecast_parser.add_argument("--run-checker", action="store_true", help="Run the official format checker")
    forecast_parser.add_argument("--test-mock", help="Path to test_data_mock.csv")

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Recompute offline model leaderboard")
    leaderboard_parser.add_argument("--input", required=True, help="Path to training CSV")
    leaderboard_parser.add_argument("--output", required=True, help="Path to output leaderboard CSV")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "forecast":
        predictions, run_meta = generate_forecast(
            input_path=args.input,
            model_name=args.model,
            fallback_model=args.fallback,
            output_path=args.output,
            run_checker=args.run_checker,
            test_mock_path=args.test_mock,
        )
        print(f"Saved forecast with {len(predictions)} rows to {args.output}")
        print(json.dumps(run_meta, indent=2))
        return 0

    leaderboard = compute_leaderboard(input_path=args.input, output_path=args.output)
    print(leaderboard.to_string(index=False))
    champion = leaderboard.loc[leaderboard["is_champion"], "model"].iloc[0]
    print(f"\nChampion (lowest SMAPE): {champion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
