import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from pathlib import Path

    HORIZON = 12
    VIZ_OBS = 120
    y_df = pd.read_csv(
        Path(__file__).parent / "data" / "02_bayes_product.csv",
        index_col="date", parse_dates=True,
    )
    y = y_df["value"]
    n_train = len(y) - HORIZON
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    return (
        HORIZON,
        VIZ_OBS,
        mo,
        n_train,
        np,
        pd,
        plt,
        sm,
        y,
        y_df,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(
    HORIZON,
    VIZ_OBS,
    mo,
    n_train,
    np,
    pd,
    plt,
    sm,
    y,
    y_df,
    y_test,
    y_train,
):
    _start = y_df.index.min().strftime("%b %Y")
    _end = y_df.index.max().strftime("%b %Y")
    mo.md(
        f"**Data loaded:** `y` — {len(y_df)} monthly observations ({_start}–{_end}). "
        f"Train: {n_train} obs, test: {HORIZON} obs. "
        f"`np`, `pd`, `plt`, `sm` available."
    )
    _ = np.array([]), pd.Series(), plt.rcParams, sm.__version__, VIZ_OBS
    _ = y, y_train, y_test
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 02: The Brief

    Six months ago you were forecasting grid load for the Department of Energy of Stochastia.
    Your capacity report circulated further than you expected — **Bayes Capital** reached out shortly after and offered you a role
    on their quantitative research team.

    Your first week. Your manager drops a dataset on your desk:

    > *"We're analyzing a new financial product to sell to one of our clients.
    > Our client wants to understand what is the expected performance of the product, as well as
    > our confidence around it.
    > Take a look at the data I shared with you. I need a twelve-month forecast with a 95% prediction interval at each step. Use what you know."*

    The data is in `y`. You should use the last 12 months as hold out for evaluation.

    Your task:
    1. Explore the data. What are the main patterns?
    2. Fit a simple OLS linear model that captures the structure identified in your EDA. Forecast 12 months ahead with 95% prediction intervals at each step,
    and plot your forecast against the held-out test period.
    3. Interpret the results of the models: the coefficients, the point estimates and the CIs. What do you see?
    """)
    return


@app.cell
def _():
    # Your code here
    return


if __name__ == "__main__":
    app.run()
