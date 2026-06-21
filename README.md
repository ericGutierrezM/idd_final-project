# IDD Final Project - Deploying a Forecasting Model for Glovo

**Team:** Samuel Fraley, Corneel Moons, Eric Gutierrez Moreno  
**Due date:** June 28, 2026

## Project Overview

This project builds a cloud-deployable forecasting workflow for Glovo hourly order demand. The business goal is to predict the number of orders expected in each hourly slot for one city so operations can plan courier capacity for the upcoming week.

For the assignment, every Sunday at `23:59` the system should generate forecasts for the next `168` hours, from Monday `00:00` through Sunday `23:00`.

## Current Status

We now have two layers in the project:

- `notebooks/`
  - research and analysis artifacts
  - EDA, model comparison, justification of the champion model
- `src/`
  - extracted production-oriented Python modules and CLIs
  - reusable logic for data prep, feature engineering, validation, forecasting, and offline model monitoring

The strongest current model remains:

- `Hybrid (zero_mask + LightGBM Tweedie)`

The current production-serving idea is:

- serve `Hybrid` as the champion model
- use `Naive` as the runtime fallback if the main model fails
- recompute an offline leaderboard for `Naive`, `LightGBM`, and `Hybrid` to monitor whether the champion should change

## What We Have Done

### Data and EDA

- Explored `data/train_data.csv`
- Identified strong daily and weekly seasonality
- Confirmed the data covers one city, Barcelona (`BCN`)
- Found three 6-hour early-morning gaps and handled them in preprocessing
- Documented the strong share of zero-order hours and upward trend over time

### Modeling

- Built a naive seasonal baseline
- Compared multiple model families in the notebook
- Used walk-forward validation aligned with the real business problem
- Evaluated models on `MSE` and `SMAPE`
- Selected `Hybrid` as the current champion based on lowest average `SMAPE`

### Production Refactor

We extracted notebook logic into a new `src/` package:

- [src/data.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/data.py)
  - load and prepare the training data
- [src/features.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/features.py)
  - holiday list and engineered features
- [src/models.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/models.py)
  - `Naive`, `LightGBM`, and `Hybrid` forecasting logic
- [src/forecast.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/forecast.py)
  - production forecast generation with fallback support
- [src/leaderboard.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/leaderboard.py)
  - offline model leaderboard recomputation
- [src/validation.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/validation.py)
  - assignment-format and CSV round-trip checks
- [src/cli.py](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/src/cli.py)
  - local CLI entrypoints

### Verified Workflows

Forecast CLI:

```bash
.\.venv\Scripts\python.exe -m src.cli forecast --input data\train_data.csv --output data\predictions.csv --model hybrid --fallback naive --run-checker --test-mock data\test_data_mock.csv
```

Leaderboard CLI:

```bash
.\.venv\Scripts\python.exe -m src.cli leaderboard --input data\train_data.csv --output data\model_leaderboard.csv
```

Verified so far:

- forecast output matches the existing notebook output exactly
- output passes the official `check_output_format.py` checker
- fallback to `Naive` works if the champion path fails
- offline leaderboard keeps `Hybrid` as the champion

## Project Structure

```text
data/
  train_data.csv
  test_data_mock.csv
  predictions.csv
notebooks/
  EDA.ipynb
  modeling.ipynb
  modeling_improved.ipynb
project-guidelines/
  IDD_ASSIGNMENT.pdf
  final_project.py
src/
  cli.py
  config.py
  cv.py
  data.py
  features.py
  forecast.py
  leaderboard.py
  metrics.py
  models.py
  validation.py
check_output_format.py
CLOUD_DEPLOY_STRATEGY.md
```

## Deliverables

### 1. Code

- Must run in the cloud
- Must train a model and generate predictions

### 2. Video

- Max 10 minutes
- Walkthrough of code and cloud setup
- Ideally shows the cloud job generating the output file

### 3. Predictions CSV stored in S3

Required format:

| Column | Type | Description |
|--------|------|-------------|
| `time` | `datetime64[ns]` | Hour start for the prediction |
| `preds` | `float64` | Forecast order count for that hour |

Requirements:

- exactly `168` rows
- time range from `2022-01-24 00:00:00` to `2022-01-30 23:00:00`
- no missing values

### 4. Production Deployment Considerations PDF

The PDF should cover:

- team roles and ways of working
- rough estimates and how they were agreed
- end-to-end architecture
- data flow from storage to forecasting to user consumption
- dev/test/prod environment design
- build and monitoring of the ML model
- end-user interaction with the output
- success KPIs

## Cloud Production Idea

Our current recommended cloud architecture is:

- `S3`
  - store input data and prediction outputs
- `Lambda`
  - run the weekly forecasting job
- `EventBridge`
  - trigger the run every Sunday night
- `CloudWatch`
  - logging, monitoring, and alerting
- optional `CloudFormation`
  - define infrastructure reproducibly

### Planned forecast flow

1. `train_data.csv` is stored in S3
2. `EventBridge` triggers the weekly forecast job
3. `Lambda` loads the training data from S3
4. the refactored Python forecast code retrains the champion model on all available history
5. the job generates the next `168` hourly predictions
6. output-format checks run
7. `predictions.csv` is written back to S3
8. logs and failures are captured in CloudWatch

### Why this design

This is the simplest architecture that still aligns well with the course:

- one champion production model
- one runtime fallback model
- offline leaderboard monitoring for reevaluation
- cloud-native scheduling and storage

If Lambda packaging becomes difficult because of dependencies like `lightgbm`, the fallback compute plan is:

- container-based Lambda
- or small batch compute on EC2 / SageMaker

The architecture story stays the same even if the compute service changes.

## What We Plan To Do Next

### Immediate next steps

- add an AWS-specific handler that reads training data from S3 and writes outputs back to S3
- decide whether to deploy with plain Lambda, Lambda container image, or a small batch fallback
- define `dev`, `test`, and `prod` S3 prefixes or paths

### Submission-focused next steps

- create the cloud architecture diagram for the PDF
- map teammate names to team roles in the production-considerations write-up
- prepare the 5-page PDF section using the current deployment strategy
- prepare the video demo flow:
  - show input data in S3
  - show forecast trigger or schedule
  - show prediction artifact in S3
  - show the forecast output or simple dashboard

### Nice-to-have next steps

- save dated forecast snapshots for reproducibility
- write run metadata alongside each forecast
- add a simple dashboard or reporting view for planners
- track leaderboard outputs over time to monitor champion changes

## AWS Setup Notes

- **S3 bucket naming:** `<name>-<surname>-bucket-idd`
- **Lambda naming:** `<name>-<surname>-forecasting-lambda`
- **Runtime:** Python 3.x
- **Login:** `https://idd-class.signin.aws.amazon.com/console`

Upload these before testing the cloud path:

- `train_data.csv`
- `test_data_mock.csv`

## Output Format Check

```python
from check_output_format import check_output_format

check_output_format(predictions, "data/test_data_mock.csv")
```

## Related Project Notes

- deployment strategy: [CLOUD_DEPLOY_STRATEGY.md](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/CLOUD_DEPLOY_STRATEGY.md)
- assignment materials: [project-guidelines](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/project-guidelines)
- current research notebook: [notebooks/modeling_improved.ipynb](C:/Users/sffra/Projects/BSE%202025-2026/idd_final-project/notebooks/modeling_improved.ipynb)

## AI Policy

AI is allowed with disclosure, validation, and shared prompt history. All AI-generated suggestions should be reviewed and adapted by the team before submission.
