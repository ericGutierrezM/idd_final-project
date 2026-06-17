# IDD Final Project — Deploying a Forecasting Model for Glovo

**Team:** Samuel Fraley, Corneel Moons, Eric Gutierrez Moreno
**Due date:** June 28, 2026

## Project Overview

Deploy an AWS cloud solution that trains a model and generates hourly order volume predictions using Glovo data. The goal is to forecast how many orders are expected per hour for one city, so operations can plan courier staffing in advance.

Every Sunday at 23:59, the model should forecast all 168 hours of the upcoming week (Monday 00:00 through Sunday 23:00).

## Deliverables

Each of the four deliverables is equally weighted within the final project (which is 60% of the module grade).

### 1. Code
- Runs in the cloud (any cloud-based compute environment).
- Must train a model and generate predictions.

### 2. Video (max 10 minutes)
- Walkthrough of code and cloud setup.
- Ideally runs the code live on the cloud and generates the output file.
- If the job takes longer than 10 min, caching is fine — no need to show the full run.
- Can be split into two clips if needed.

### 3. Predictions CSV — stored in S3
Format:

| Column | Type | Description |
|--------|------|-------------|
| `time` | `datetime64[ns]` | Hour start for the prediction |
| `preds` | `float64` | Forecast order count for that hour |

- Exactly **168 rows**: `2022-01-24 00:00:00` through `2022-01-30 23:00:00`, inclusive.
- No missing values.
- Validate with `check_output_format(predictions, path_to_test_data)` from `check_output_format.py`.

### 4. Production Deployment Considerations (PDF, max 5 pages)
Cover the following:
- Team(s), roles, interactions, and ways of working
- Rough effort estimates per feature/part and how they were agreed
- Architecture components of the end-to-end solution
- Data used and data flows (storage → forecasting tool → visualization)
- Environments to be used (dev, staging, prod, etc.)
- How the ML model will be built and monitored
- How end-users will interact with the solution (visualization tool, dashboard, etc.)
- Success KPIs

## Data

- **Training data:** `data/train_data.csv` — columns: `time` (hourly timestamp), `orders` (count), `city`
- **Mock test data:** `data/test_data_mock.csv` — use to validate output format and merge logic during development

## EDA Requirements

Explore `data/train_data.csv` and document:
- Trends, seasonality, and cycles
- Outliers
- How findings inform modelling choices

## Modelling Requirements

- Train a **naive baseline**
- Train **at least two additional model families** beyond the baseline
- Validate using a **walk-forward strategy** that simulates the real production scenario (forecast next week using only data available up to that Sunday — no data leakage)
- Evaluate each model on **MSE** and **SMAPE**
- Select and justify a champion model

## Output Format Check

```python
from check_output_format import check_output_format

check_output_format(predictions, "data/test_data_mock.csv")
```

Re-read the saved CSV after writing and verify dtypes and row count before submitting.

## AWS Infrastructure

- **S3 bucket naming:** `<name>-<surname>-bucket-idd`
  - Upload `train_data.csv` and `test_data_mock.csv` before testing Lambda
- **Lambda function naming:** `<name>-<surname>-forecasting-lambda`
  - Runtime: Python 3.x
  - Layers: pandas, sklearn
  - Execution role: assigned by instructor
- **Login:** https://idd-class.signin.aws.amazon.com/console

## EDA Findings

**Dataset:** 8,550 rows, single city — Barcelona (BCN), covering **2021-02-01 to 2022-01-23 23:00:00**.

### Data Quality
- No NaN values in any column
- `time` column is stored as string and must be converted to datetime
- Three 6-hour gaps (00:00–05:00) on: 2021-02-15, 2021-06-07, 2021-10-18

### Order Distribution
- Right-skewed / log-normal shape
- Over 25% of hours have 0 orders
- Median: 30, Mean: 73, Max: 939
- 8.33% of observations are outliers by IQR (712 slots)

### Seasonality (Prophet decomposition)
- **Trend:** Clear upward trend from ~65 to ~95 orders/hour over the year
- **Weekly:** Fri/Sat/Sun significantly higher; Mon–Thu lower
- **Daily:** Low overnight → lunch peak (~1pm) → afternoon dip (~5pm) → dinner peak (~8:30pm) → low again
- **Yearly:** Not modeled — data covers less than one full year

### Modelling Implications
Strong weekly and daily seasonality makes this well-suited for models that can capture both cycles (e.g. Prophet, SARIMA, tree-based models with time features).

## AI Policy

AI is allowed with disclosure, validation, and shared prompt history.
