# Cloud Deploy Strategy

## Purpose

This document translates our current best forecasting workflow in [notebooks/modeling_improved.ipynb](/abs/path/notebooks/modeling_improved.ipynb) into a realistic cloud deployment plan for the final project PDF and implementation work.

It addresses each required assignment topic directly:

- team roles, interactions, and ways of working
- rough estimates and how they were agreed
- end-to-end architecture
- data used and data flow
- environments
- how the ML model is built and monitored
- how end-users consume the solution
- success KPIs

It also highlights practical options, tradeoffs, and next steps from our current notebook-based state.

## Current State

Our strongest modeling asset today is [notebooks/modeling_improved.ipynb](/abs/path/notebooks/modeling_improved.ipynb), which:

- uses walk-forward validation aligned to the business problem
- compares multiple model families
- selects a champion model using SMAPE
- currently chooses `Hybrid (zero_mask + LightGBM Tweedie)` as the champion
- generates a valid `168`-row predictions file with `time` and `preds`
- saves output to `data/predictions.csv`
- runs the official output-format checks

Current limitation: the logic is still notebook-first. For cloud deployment, we need to turn notebook code into reusable Python functions/scripts plus a deployable runtime.

## Recommended Deployment Direction

## Recommendation

For this class project, the best balance of simplicity, course alignment, and realism is:

- `S3` for input/output storage
- `Lambda` for scheduled weekly forecasting
- `EventBridge` to trigger the weekly run every Sunday night
- `CloudWatch` for logs, metrics, and alerts
- optional `CloudFormation` for infrastructure setup
- optional simple dashboard layer using `Streamlit`, `QuickSight`, or even a second CSV/report in S3

## Why this is the best fit

This matches the course slides closely and keeps the architecture understandable in a 10-minute video. The dataset is small enough that weekly retraining plus prediction should be feasible without heavy infrastructure, as long as we package dependencies carefully.

## Main deployment decision

There are two viable ways to deploy from the current notebook:

### Option A: Lambda runs training and inference each week

How it works:

- Lambda loads `train_data.csv` from S3
- retrains the chosen hybrid model on all available history
- generates the next `168` hours of predictions
- writes predictions CSV to S3
- logs success/failure to CloudWatch

Pros:

- simple architecture
- closest to class hands-on material
- easy to demo
- no persistent server needed

Cons:

- packaging `lightgbm` can be annoying in plain zip deployments
- Lambda memory/time limits may become an issue if the code grows
- notebook code must be refactored carefully into a lean inference script

### Option B: Batch-style training job plus lightweight inference step

How it works:

- a scheduled batch job retrains the model and saves artifacts to S3
- Lambda only loads the saved artifact and produces the weekly predictions

Pros:

- cleaner separation between training and inference
- more realistic MLOps structure
- easier to scale later

Cons:

- more moving parts
- more setup than the course likely requires

## Recommended final choice

Start with `Option A` for the submission unless packaging becomes a blocker. If Lambda packaging becomes painful, fall back to a small EC2, SageMaker notebook job, or container-based run while keeping the same architecture story in the PDF.

## From Notebook to Cloud

## Refactor plan

We should move from notebook code to production code in four steps:

1. Extract reusable logic from the notebook into Python modules.
2. Create one script for training plus prediction generation.
3. Add a cloud entrypoint that reads from S3 and writes predictions back to S3.
4. Add deployment config and monitoring.

## Suggested code structure

Possible structure:

```text
src/
  data_loader.py
  features.py
  metrics.py
  models.py
  predict_week.py
  aws_lambda_handler.py
scripts/
  run_local_forecast.py
```

What moves out of the notebook first:

- feature engineering
- walk-forward validation helpers
- hybrid model training logic
- forecast window generation
- output-format validation

## Assignment Questions: Proposed Answers and Next Steps

## 1. Teams, roles, interactions, and ways of working

### What the assignment wants

It wants proof that we are applying Agile and project-organization concepts from the course, not just building a model.

### Proposed solution

For a 2-3 person team, define roles by responsibility area rather than rigid job title:

- `Model owner`
  - owns EDA, feature engineering, model comparison, champion selection
- `Cloud/deployment owner`
  - owns AWS setup, S3, Lambda, IAM, EventBridge, CloudWatch
- `Integration/quality owner`
  - owns output validation, testing, documentation, video/demo flow

If the team has only two people, one person can combine `model owner` and `integration/quality owner`.

### Ways of working

- use Git branches and pull requests or at least peer review before merging
- work in short increments
- agree on a shared "definition of done" for each task
- keep architecture and modeling decisions documented in Markdown
- do quick syncs before merging changes to notebook/code/cloud setup

### Interactions

- modeling changes should be communicated before deployment changes
- cloud owner should define expected input/output file paths early
- quality owner should run the output-format checker before release

### Next steps

- write down actual teammate names against the roles
- add a short "ways of working" paragraph for the final PDF
- decide whether the team will use Scrum-style weekly tasks or a lighter Kanban flow

## 2. Rough estimates for each feature/part and how they were agreed

### What the assignment wants

It wants a lightweight estimation process, not a fake corporate planning exercise.

### Proposed solution

Use simple team estimates in person-hours and explain they were agreed through discussion after reviewing the assignment and current repo state.

Suggested rough estimates:

- EDA review and write-up: `4-6 hours`
- modeling improvement and validation: `8-12 hours`
- notebook-to-script refactor: `4-8 hours`
- AWS setup and deployment: `6-10 hours`
- testing and output validation: `3-5 hours`
- dashboard/reporting layer: `2-5 hours`
- video prep and recording: `3-4 hours`
- final PDF write-up: `4-6 hours`

### How to explain agreement

Good wording for the PDF:

"We estimated effort collaboratively after reviewing the assignment requirements, current notebook maturity, and AWS setup tasks. Estimates were rough, intended for prioritization rather than precision, and were updated as implementation risks became clearer."

### Next steps

- replace these estimates with team-specific numbers if needed
- record which tasks are already complete versus still open
- add one sentence explaining that estimates were refined after the champion model was selected

## 3. Architecture components of the end-to-end solution

### What the assignment wants

A clear system view from raw data to forecast output and user consumption.

### Proposed solution

Recommended end-to-end architecture:

- `S3 input bucket`
  - stores `train_data.csv`, `test_data_mock.csv`, optional config files
- `Forecasting compute`
  - Lambda function running weekly training + prediction
- `Model logic package`
  - extracted Python code from the notebook
- `S3 output location`
  - stores weekly predictions CSV and optional plots/summary files
- `EventBridge scheduler`
  - triggers the job every Sunday night
- `CloudWatch`
  - logs, monitoring, alerting
- `Visualization layer`
  - simple dashboard, report, or CSV consumption

### Text architecture diagram

```text
S3 raw data
  -> Lambda forecasting job
  -> model retraining + 168-hour forecast generation
  -> predictions.csv written to S3
  -> dashboard/report reads predictions
CloudWatch monitors the job and EventBridge schedules it weekly
```

### Alternative solution

If Lambda packaging fails, swap the compute box:

- Lambda -> EC2 scheduled script
- Lambda -> SageMaker notebook job
- Lambda -> container-based batch run

Everything else can stay mostly the same.

### Next steps

- create a simple diagram image for the final PDF
- confirm exact bucket/folder names
- decide whether to keep one bucket with prefixes or separate input/output buckets

## 4. Data used and data flows

### What the assignment wants

Explain what data is used, where it lives, how it moves, and how it reaches both the forecasting tool and the user.

### Proposed solution

Current data:

- `data/train_data.csv`
  - historical hourly orders for one city
- `data/test_data_mock.csv`
  - development-only validation for output format and checker integration

Production-style data flow:

1. historical training data is uploaded to `S3`
2. scheduled forecast job reads training data from `S3`
3. job parses dates, rebuilds features, retrains the champion model
4. job creates the next `168` hourly timestamps
5. job generates predictions and validates output structure
6. job writes `predictions.csv` to `S3`
7. dashboard/report/user downloads or visualizes the predictions

### What to say about visualization

Possible solutions:

- `simplest`: planners download the CSV from S3
- `better`: a small Streamlit dashboard reads the latest predictions file
- `AWS-native`: QuickSight visualizes actual vs predicted and future demand curve

### Recommended positioning

For the final PDF, say:

- the operational artifact is the weekly prediction CSV in S3
- a lightweight dashboard is the preferred consumption layer for planners

### Next steps

- define the exact S3 path for outputs, for example `s3://bucket-idd/forecasts/weekly/predictions.csv`
- decide whether to save dated forecast snapshots for reproducibility
- optionally save a second CSV with metadata such as run timestamp and champion model

## 5. Environments to be used

### What the assignment wants

A clear explanation of dev, test/staging, and prod thinking.

### Proposed solution

Use three logical environments even if they are lightweight:

- `Dev`
  - local notebooks and scripts
  - used for feature work, validation, and debugging
- `Test / UAT`
  - cloud dry-run environment using `test_data_mock.csv` and sample outputs
  - validates IAM permissions, S3 paths, Lambda execution, and output format
- `Prod`
  - real scheduled weekly run using the official training data and production output path

### Why this matters

This shows that we understood the course point that data/ML systems should not go directly from notebook to production without checks.

### Practical note

For the project, these environments may share one AWS account but use different prefixes:

- `dev/`
- `test/`
- `prod/`

### Next steps

- create separate S3 prefixes or folders for each environment
- define separate output paths so test runs never overwrite prod outputs
- decide whether to keep separate Lambda configs or one Lambda with environment variables

## 6. How the ML model will be built and monitored

### What the assignment wants

This is the MLOps section: training, deployment, monitoring, quality tracking, and retraining.

### Proposed solution for build

Weekly build flow:

1. EventBridge triggers the run
2. Lambda reads training data from S3
3. feature engineering is rebuilt exactly as in the validated notebook
4. hybrid model is retrained on all available history
5. predictions are generated for the next week
6. output-format checks run
7. results are saved to S3

### Proposed solution for monitoring

Monitor two categories:

- `pipeline health`
  - did the job run
  - did it finish on time
  - did it produce exactly 168 rows
  - were there nulls or schema issues
- `model health`
  - weekly SMAPE once actuals are available
  - prediction distribution checks
  - share of zero predictions
  - sudden shifts in forecast level versus recent history

### Alert ideas

- Lambda failure alert
- missing output file alert
- wrong row count alert
- unusually high/low forecast mean alert
- accuracy degradation alert once actuals are known

### Retraining logic

For the class project, weekly retraining is acceptable because:

- the data is time-series and evolves over time
- the business process itself is weekly
- the training set is small enough to retrain cheaply

### Important implementation note

The current champion hybrid model depends only on the chosen serving logic, not on Prophet at inference time. That is helpful because Prophet does not need to be in the production path if we only deploy the hybrid winner.

### Next steps

- log run metadata: timestamp, model version, data window, row count
- save each forecast with a date-stamped filename
- add a simple post-run validation summary
- define one rollback plan, for example falling back to the seasonal naive forecast if the main run fails

## 7. How end-users will use the solution

### What the assignment wants

Describe how a planner or operations stakeholder actually consumes the forecast.

### Proposed solution

Target end-user:

- Glovo operations planner responsible for weekly courier capacity planning

Usage flow:

1. every Sunday night the system creates the upcoming week forecast
2. Monday morning the planner opens the latest forecast
3. they review hourly demand by day
4. they use the forecast to decide courier staffing levels

### Possible consumption channels

- `Option 1: CSV in S3`
  - easiest to implement
  - good enough for submission
- `Option 2: simple dashboard`
  - shows next-week hourly forecast as line chart and daily totals
  - more user-friendly
- `Option 3: API`
  - overkill for this assignment unless already easy to build

### Recommended final stance

Say that the primary operational output is the CSV, while a simple dashboard is the intended user-facing layer for future improvement.

### Next steps

- decide whether to build a minimal dashboard or just mock the interface in the PDF
- if building one, show:
  - hourly line chart
  - daily totals
  - latest run timestamp
  - optional comparison with previous week

## 8. Success KPIs

### What the assignment wants

Both technical and business indicators of success.

### Proposed KPI set

Model KPIs:

- `SMAPE` on holdout/rolling validation
- `MSE` for comparison with assignment requirements
- forecast bias over time

Pipeline KPIs:

- weekly run success rate
- forecast generated before business deadline
- percentage of runs producing valid 168-row outputs

Business-facing KPIs:

- reduction in manual forecasting effort
- planner adoption of the forecast output
- reduction in under-staffing or over-staffing risk

### Recommended concrete targets

Possible targets for the PDF:

- SMAPE better than seasonal naive baseline
- 100% valid output format on production runs
- forecast available before Monday planning starts
- zero manual intervention in normal weekly runs

### Next steps

- align the KPI wording with the notebook results
- include one sentence that business impact would ideally be measured against operational staffing outcomes, but that such data is outside the scope of the class dataset

## Risks and Mitigations

## Risk 1: Notebook code is not production-ready

Mitigation:

- refactor notebook logic into scripts/modules
- keep the notebook as analysis, not deployment code

## Risk 2: Lambda dependency packaging

Mitigation:

- deploy only the champion model dependencies
- use Lambda container image or layer if plain zip becomes difficult
- keep fallback compute option ready

## Risk 3: No reproducible forecast snapshots

Mitigation:

- save dated outputs
- log model version and run timestamp

## Risk 4: Main model fails at runtime

Mitigation:

- fall back to seasonal naive forecast
- alert through CloudWatch

## Risk 5: Environment mismatch

Mitigation:

- use environment variables for bucket names and paths
- test with a dedicated `test/` prefix before prod

## Recommended Next Implementation Steps

## Immediate

1. Extract notebook logic into a scriptable Python module.
2. Create a local command that reproduces the forecast CSV from the module, not the notebook.
3. Decide on deployment path: `Lambda zip`, `Lambda container`, or fallback batch compute.

## Short-term

4. Create the AWS S3 folder structure for `dev`, `test`, and `prod`.
5. Build a Lambda handler that:
   - reads training data from S3
   - runs forecast generation
   - writes predictions to S3
6. Add CloudWatch logging and a basic success/failure alert.

## Submission-focused

7. Turn the architecture and role sections above into the 5-page PDF narrative.
8. Create one simple architecture diagram.
9. Prepare a demo flow for the video:
   - show S3 input
   - trigger or show scheduled Lambda
   - show output file in S3
   - show forecast preview/dashboard

## Best PDF Framing

If we want the final PDF section to align tightly with the course slides, the strongest framing is:

- Agile explains how we organized the team and estimates.
- DevOps explains how we package, schedule, test, and deploy the solution.
- DataOps explains how we protect data quality and job reliability.
- MLOps explains how we retrain, validate, monitor, and maintain the forecast model.
- Cloud explains where the solution runs and how users access the results.

That framing answers the assignment directly and shows that the deployment strategy is grounded in the course concepts, not added as an afterthought.
