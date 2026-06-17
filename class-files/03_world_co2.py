import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    _df = pd.read_csv(
        Path(__file__).parent / "data" / "03_world_co2.csv",
        index_col="year",
    )
    co2 = _df["co2"].dropna()
    population = _df["population"]
    co2_per_capita = _df["co2_per_capita"].dropna()
    return co2, co2_per_capita, mo, np, pd, plt, population


@app.cell(hide_code=True)
def _(co2, co2_per_capita, population):
    print(f"co2:            {int(co2.index.min())}–{int(co2.index.max())}  ({len(co2)} obs)")
    print(f"population:     {int(population.index.min())}–{int(population.index.max())}  ({len(population)} obs)")
    print(f"co2_per_capita: {int(co2_per_capita.index.min())}–{int(co2_per_capita.index.max())}  ({len(co2_per_capita)} obs)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 03: A new chapter — forecasting world CO2

    The structured-products book at **Bayes Capital** has been good to you:
    after some work there the head of research has stopped second-guessing your model choices. But you
    keep catching yourself wondering whether forecasting bond yields and equity
    indices is the most useful thing you could be doing with what you've
    learned.

    Then a recruiter from the **International Energy Agency** in Paris reaches
    out. Someone there saw your analysis and remembered the analyst who
    insisted on inspecting the series before letting the software pick orders.
    They have a position open on the modelling team, and they're skipping
    straight to a take-home test:

    > *"Annual world CO2 emissions, 1960 through 2025. We need a forecast to
    > 2050 and a defensible argument for why we should believe it. Our outlook
    > report goes to the climate negotiation in November and the policy team
    > needs to know whether the curve bends — and if so, when."*

    The data on your desk:

    - `co2` — annual World CO2 emissions 1960–2025 (OWID / Global Carbon
      Project; 2025 is provisional).
    - `population` — World population 1960–2050, historical through 2024 and
      UN WPP 2024 medium-variant projections to 2050.
    - `co2_per_capita` — tonnes CO2 per person, computed, 1960–2025.

    Unlike anything you ever shipped at Bayes, **there is no held-out truth
    here.** You are forecasting the actual future.

    You will:

    1. Explore the series.
    2. Fit a couple of ARIMA specifications and compare their forecasts to 2050.
    3. Pick a model and answer the IEA's question.

    *Optional:* you may use `population` and `co2_per_capita` if they sharpen
    your argument (for example total emissions vs per-capita), not only `co2`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore the data

    Plot the series and any transformations of it that help you decide what
    to model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference: snippets (commented — copy and adapt)

    ```python
    # --- First-order differences ---
    # y is a pd.Series (e.g. co2). .diff() loses the first row; dropna() drops NaN.
    # d1 = y.diff().dropna()

    # --- Log scale, then first-order differences (needs y > 0 everywhere) ---
    # import numpy as np
    # log_y = np.log(y)
    # d1_log = log_y.diff().dropna()

    # --- Undo log on predictions (if you fitted on log(y)) ---
    # np.exp maps forecasts and CIs back to the original scale (quick shortcut).
    # y_hat_level = np.exp(y_hat_log)
    # ci_level = np.exp(ci_log)

    # --- pmdarima: import, fit, predict (orders in ARIMA(...); training series in fit(...)) ---
    # from pmdarima import ARIMA
    # h = 25  # example: years ahead
    # model = ARIMA(order=(1, 1, 1))
    # model.fit(y_train)  # pass a 1d array-like or Series
    # y_hat, ci = model.predict(n_periods=h, return_conf_int=True, alpha=0.05)
    #
    # # Build a Series for the helper (example):
    # future = np.arange(y_train.index[-1] + 1, y_train.index[-1] + 1 + h)
    # forecast_series = pd.Series(y_hat, index=future)
    # plot_real_and_forecast(y_train, forecast_series, ylabel="MtCO2")
    ```
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit ARIMA models and forecast to 2050

    Fit a couple of ARIMA specifications, plot the point forecasts on the same
    axis, and pick the one you'd defend.
    """)
    return


@app.cell
def _(co2, np, pd, plt):
    def plot_real_and_forecast(
        history: pd.Series,
        forecast: pd.Series,
        *,
        ax=None,
        title=None,
        ylabel=None,
        show_history=True,
        forecast_label="Forecast",
        forecast_color="#DD8452",
    ):
        """Line plot: observed history vs forecast. `history` and `forecast` are pd.Series (index = time)."""
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))
        cutoff = history.index[-1]
        ax.axvline(cutoff, color="grey", linestyle=":", linewidth=1.2, zorder=1)
        if show_history:
            ax.plot(
                history.index,
                history.values,
                color="#4C72B0",
                marker=".",
                markersize=5,
                label="Observed",
                zorder=2,
            )
        ax.plot(
            forecast.index,
            forecast.values,
            color=forecast_color,
            marker=".",
            markersize=5,
            linestyle="--",
            linewidth=2.0,
            label=forecast_label,
            zorder=2,
        )
        ax.legend(loc="best", fontsize=9)
        if title is None:
            title = f"Observed vs forecast (cutoff: {cutoff})"

        ax.set_title(title)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return ax

    future_dates = np.arange(co2.index[-1] + 1, 2051)
    dummy_preds = pd.Series([co2.iloc[-1]]*len(future_dates), index=future_dates)
    plot_real_and_forecast(co2, dummy_preds)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Recommendation

    1. Do emissions peak — and if so, when?
    2. What could make your forecast wrong?
    """)
    return


if __name__ == "__main__":
    app.run()
