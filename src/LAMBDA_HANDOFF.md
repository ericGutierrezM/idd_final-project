# Lambda Handoff Notes

This folder is the runtime code for the AWS Lambda submission path.

We are assuming the classroom Lambda environment should be treated as **Python 3.8**.

This `src/` folder is intentionally simplified for handoff:

- one deployment file for AWS
- one short handoff note
- no extra leaderboard, CLI, or research-time modules

Use this handler:

```text
src.lambda_app.lambda_handler
```

That file contains only the logic needed to:

- read `train_data.csv` from S3
- build features
- run one model
- validate the forecast CSV
- write `predictions.csv` back to S3

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
src.lambda_app.lambda_handler
```

Main file:

- `src/lambda_app.py`
  - AWS entrypoint
  - S3 download and upload logic
  - feature engineering
  - one deployed model
  - output validation

## Expected AWS setup

Lambda:

- Runtime: `Python 3.8`
- Handler: `src.lambda_app.lambda_handler`
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
- `RUN_OFFICIAL_CHECKER`

## Recommended test event

```json
{
  "input_bucket": "your-bucket-name",
  "train_data_key": "train_data.csv",
  "test_mock_key": "test_data_mock.csv",
  "output_latest_key": "predictions/predictions.csv",
  "output_snapshot_prefix": "predictions/history/",
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
5. Set the handler to `src.lambda_app.lambda_handler`.
6. Upload `train_data.csv` to the team bucket.
7. Upload `test_data_mock.csv` too if they want to test the checker path.
8. Add the environment variables or paste the test event above.
9. Run a manual test invoke.
10. Check CloudWatch logs and the S3 output path.

## What success should look like

The Lambda response should return a JSON body with fields like:

- `status`
- `selected_model`
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

If something fails in AWS, first debug the packaging, handler path, and S3 permissions before changing the model logic.
