# Lambda Handoff Notes

This folder is the runtime code for the AWS Lambda submission path.

We are assuming the classroom Lambda environment should be treated as **Python 3.8**.

## What this code is for

The Lambda should:

1. read `train_data.csv` from S3
2. generate the 168-hour forecast for `2022-01-24 00:00:00` through `2022-01-30 23:00:00`
3. save `predictions.csv` back to S3

This matches the class pattern shown in the AWS slides:

- S3 as storage
- Lambda as compute
- pandas/sklearn provided through layers
- zip upload for our code

## Main Lambda entrypoint

Handler:

```text
src.aws_lambda_handler.lambda_handler
```

Main flow:

- `src/aws_lambda_handler.py`
  - AWS entrypoint
  - downloads files from S3 to Lambda temp storage
  - calls forecast code
  - uploads output CSV back to S3
- `src/forecast.py`
  - runs the forecast workflow
- `src/data.py`, `src/features.py`, `src/models.py`, `src/validation.py`
  - reusable forecasting logic

## Expected AWS setup

Lambda:

- Runtime: `Python 3.8`
- Handler: `src.aws_lambda_handler.lambda_handler`
- Timeout: give it enough time to train and write output
- Layers: classroom `pandas` and `sklearn` layers

S3:

- upload `train_data.csv`
- optionally upload `test_data_mock.csv` if testing the official checker path

IAM role:

- permission to read from and write to the class S3 bucket
- CloudWatch logging permission

## Environment variables / event values

The handler can read config either from the test event or from Lambda environment variables.

Minimum required values:

- `INPUT_BUCKET`
- `TRAIN_DATA_KEY`
- `OUTPUT_LATEST_KEY`
- `OUTPUT_SNAPSHOT_PREFIX`

Optional values:

- `TEST_MOCK_KEY`
- `MODEL_NAME`
- `FALLBACK_MODEL`
- `RUN_OFFICIAL_CHECKER`
- `ENV`

## Recommended test event

```json
{
  "env": "test",
  "input_bucket": "your-bucket-name",
  "train_data_key": "train_data.csv",
  "test_mock_key": "test_data_mock.csv",
  "output_latest_key": "predictions/predictions.csv",
  "output_snapshot_prefix": "predictions/history/",
  "model_name": "hybrid",
  "fallback_model": "naive",
  "run_official_checker": true
}
```

If your classmate wants a simpler first test, set:

```json
{
  "run_official_checker": false
}
```

and keep the rest the same.

## How to package the code

From the repo root:

```powershell
.\scripts\build_lambda_zip.ps1
```

This creates:

```text
build/lambda_deployment.zip
```

The zip currently contains only:

- `src/`
- `check_output_format.py`

That is the intended runtime package.

## Suggested upload flow for a classmate

1. Open the class Lambda in AWS.
2. Confirm the runtime is `Python 3.8`.
3. Confirm the `pandas` and `sklearn` layers are attached.
4. Upload `build/lambda_deployment.zip`.
5. Set the handler to `src.aws_lambda_handler.lambda_handler`.
6. Upload `train_data.csv` to the team bucket.
7. Upload `test_data_mock.csv` too if they want to test the checker path.
8. Add the environment variables or paste the test event above.
9. Run a manual test invoke.
10. Check CloudWatch logs and the S3 output path.

## What success should look like

The Lambda response should return a JSON body with fields like:

- `status`
- `selected_model`
- `served_model`
- `used_fallback`
- `row_count`
- `output_latest_key`
- `output_snapshot_key`

The output CSV in S3 should:

- have exactly 168 rows
- contain columns `time` and `preds`
- cover `2022-01-24 00:00:00` through `2022-01-30 23:00:00`

## Important caveat

We adjusted the repo to be Python 3.8-friendly, but we have **not** yet tested this inside a real local Python 3.8 interpreter or inside the classroom Lambda itself.

So if something fails in AWS, the most likely causes are:

- the classroom `sklearn` layer version is older than expected
- the Lambda handler/zip root is configured incorrectly
- S3 keys or permissions are wrong

## Best first fallback if upload fails

If the `hybrid` path has an issue in AWS, try:

- keep the same code package
- set `model_name` to `naive`
- keep `fallback_model` as `naive`

That gives the simplest possible cloud run while preserving the same S3/Lambda architecture.
