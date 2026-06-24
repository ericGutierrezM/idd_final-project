# IDD Final Project - Deploying a Forecasting Model for Glovo

**Team:** Samuel Fraley, Corneel Moons, Eric Gutierrez Moreno  
**Due date:** June 28, 2026

## Project Overview

This project builds a cloud-deployable forecasting workflow for Glovo hourly order demand. The business goal is to predict the number of orders expected in each hourly slot for one city so operations can plan courier capacity for the upcoming week.

For the assignment, every Sunday at `23:59` the system should generate forecasts for the next `168` hours, from Monday `00:00` through Sunday `23:00`.

## Instructor Deployment Constraints (updated June 2026)

Gal.la clarified the cloud deliverable requirements:

- **sklearn only** — no LightGBM in the Lambda; the class layer only packages sklearn
- **Python 3.8** — the pre-deployed Lambda and layer run Python 3.8, not 3.14
- **Zip + layer deployment** — the sklearn layer is already deployed; we package our code as a zip, not a container
- **Accuracy is not the focus for the cloud piece** — the goal is code that runs and produces predictions; model quality is Javier's notebook deliverable
- **Fit under 15 minutes** — the full train + predict cycle must complete within the Lambda timeout

This creates a clean split between two deliverables:

| Deliverable | Owner | Model | Environment |
|---|---|---|---|
| Cloud forecast job | Gal.la / cloud track | sklearn-based (zip + layer) | AWS Lambda Python 3.8 |
| Modeling notebook | Javier | Hybrid (zero_mask + LightGBM Tweedie) | Local notebook |

## Current Status

We now have two layers in the project:

- `notebooks/`
  - research and analysis artifacts
  - EDA, model comparison, justification of the champion model
- `src/`
  - extracted production-oriented Python modules and CLIs
  - reusable logic for data prep, feature engineering, validation, forecasting, and offline model monitoring

**Notebook champion:** `Hybrid (zero_mask + LightGBM Tweedie)` — 17.93% SMAPE, beats seasonal naive by 1.30 pp

**Cloud-served model:** sklearn-based alternative (Ridge or similar hybrid routing) — LightGBM is not available in the deployed Lambda layer

The current cloud-serving plan:

- serve a sklearn model (e.g. `Hybrid (zero_mask + Ridge)`) as the Lambda model
- use `Naive` as the runtime fallback if the sklearn model fails
- the notebook leaderboard (Naive / LightGBM / Hybrid) remains the accuracy reference

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

The sklearn layer is already deployed in the class account (Python 3.8). Our code is packaged as a zip and attached to that Lambda — no container needed.

LightGBM is only used in the local notebook. It is not part of the cloud deployment path.

## Python 3.8 compatibility notes

For the Lambda deployment path, the repo is now pinned to a Python 3.8-compatible stack:

- `numpy>=1.24.4,<1.25.0`
- `pandas>=1.5.3,<1.6.0`
- `scikit-learn>=1.3.2,<1.4.0`

The served cloud model path in `src/` uses sklearn APIs that are available in that range, and the type hints were kept Python 3.8-safe.

## Build the Lambda zip

Package only the runtime code with:

```powershell
.\scripts\build_lambda_zip.ps1
```

That creates `build/lambda_deployment.zip` with:

- `src/`
- `check_output_format.py`

## What We Plan To Do Next

### Immediate next steps

- implement and test a sklearn-based Lambda model (e.g. `Hybrid (zero_mask + Ridge)`) locally on Python 3.8
- fix any Python 3.8 incompatibilities in `src/` (e.g. `str | None` union syntax needs `Optional[str]` or `from __future__ import annotations`)
- make the LightGBM import in `src/models.py` conditional so the zip does not fail at import time
- package `src/` as a zip and deploy to the class Lambda
- define `dev`, `test`, and `prod` S3 prefixes

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
