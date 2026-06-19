import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final project: hourly demand forecasts for Glovo

    ## Your brief
    You have just joined **Glovo** as someone who owns **demand forecasting** for local delivery. The ops lead pulls you aside: each city runs on **hourly slots**. For every hour they must decide roughly **how many couriers** to have on the road—and that headcount should follow **how many orders** they expect in that hour. Schedule **too many** riders and you burn payroll; **too few** and deliveries slip, restaurants complain, and customers churn.

    Until now the team has patched it together with **spreadsheets and gut feel**: looking at last week, nudging numbers up or down before each wave of planning. That worked when there were fewer cities and calmer growth. It does not scale. Ops is blunt: *“We need a number we can defend for every hour of the week”*

    Your first deliverable: for **one city** (the dataset is already a single market), produce **trusted hourly order forecasts** that they could plug into staffing. You are not building the full rostering product yet—you are giving them the **demand curve** the product will sit on top of.

    ## What you must predict
    - **Target:** number of orders per hour (the `orders` column in the training file).
    - **Horizon:** a full **upcoming week**: 168 consecutive hours (7 × 24), from **Monday 00:00** through **Sunday 23:00** in the same timezone as the data.
    - **Concrete forecast window** (this assignment): timestamps from **2022-01-24 00:00:00** through **2022-01-30 23:00:00**, **both inclusive**.
    - **Business context:** in operations, planning is refreshed **once a week** (after **Sunday 23:59**), and the next delivery covers **all hours of the following week**.

    ## Data
    - **Path:** `data/train_data.csv`
    - **Columns:** `time` (hourly timestamp), `orders` (count), `city`. For this task the series is for a **single city**.
    - Use **`data/test_data_mock.csv`** to sanity-check shape, dtypes, and the merge logic (see below). The real holdout may differ; the format checker still applies.

    ## EDA and modelling
    Use the methods from the course: explore the data, justify your modelling choices, compare **several** approaches, and evaluate them out-of-sample with **MSE** and **SMAPE**. Say which model you would ship.

    ## Required output (CSV)
    Your submission must be a single CSV with **exactly** **168 rows** and **2 columns**:

    | Column | Meaning | dtype |
    |--------|---------|--------|
    | `time` | hour start for the prediction | `datetime64[ns]` |
    | `preds` | forecast order count for that hour | `float64` |

    Requirements:
    - One row per hour from **2022-01-24 00:00:00** to **2022-01-30 23:00:00**, inclusive.
    - **No missing** values in either column.
    - Every `time` in your file must appear in the test frame used for evaluation (the checker **inner-merges** on `time`).

    ## Format checker (repository)
    Use **`check_output_format(predictions, path_to_test_data)`** from `check_output_format.py`:
    - `predictions`: your `DataFrame` with columns `time` and `preds`.
    - `path_to_test_data`: path to a test CSV with at least `time` and `orders` for the evaluation hours (e.g. `data/test_data_mock.csv` while developing).

    The helper asserts row count, columns, dtypes, no nulls, and that every `time` in your file aligns with the test file. It prints **MSE** against `orders` for that test file; compute **SMAPE** yourself where you need it beyond that.

    ## Before you submit
    - Run the checker on your final `DataFrame` and on a written CSV if you load from disk.
    - Re-read the CSV in pandas and confirm dtypes and row count **after** saving (no integer `preds` dtype by accident, no duplicate `time` rows).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grading criteria

    ### EDA
    - Identifies and describes the main patterns in the data.
    - Explains how those findings shape the modelling choices.

    ### Modelling
    - Trains a **naive baseline**.
    - Trains **at least two additional model families** beyond the baseline.
    - Applies **correct time-series validation**, simulating the production scenario.
    - Selects and justifies a champion model.

    ### Forecast performance
    - Graded on the **SMAPE** of the submitted CSV against the held-out week.

    ### Code and explanation
    - The notebook tells a clear story: EDA → modelling decisions → results → conclusion.
    - Code is readable and does not require the reader to guess what a cell does.
    """)
    return


if __name__ == "__main__":
    app.run()
