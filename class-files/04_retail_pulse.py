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

    sales_data = pd.read_csv(
        Path(__file__).parent / "data" / "04_retail_pulse.csv",
        parse_dates=["date"],
    )
    sales_data = sales_data.set_index("date")
    sales_data.index.freq = "MS"
    sales = sales_data["sales"]
    return mo, np, pd, plt, sales, sales_data


@app.cell(hide_code=True)
def _(mo, np, pd, plt, sales, sales_data):
    _start = pd.Timestamp(sales.index[0]).strftime("%b %Y")
    _end = pd.Timestamp(sales.index[-1]).strftime("%b %Y")
    mo.md(
        f"**Data loaded:** `sales` — {len(sales_data)} monthly observations"
        f" ({_start}–{_end}),"
        f" mean {np.mean(sales):,.0f} units/month."
        f" (`plt`, `pd`, `np` are available in your cells.)"
    )
    _ = plt.rcParams
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 04: Retail Pulse — forecasting OTC allergy tablet demand

    After a couple years at the IEA, you're ready for a new challenge.
    You're considering buying an apartment and with the current prices you'd need
    a second job to afford it. So when a recruiter from the pharmaceutical
    industry reaches out, you jump at the opportunity to apply your data
    skills in a new context.

    You joined the procurement analytics team at **Medistor**, a mid-size
    generics distributor that supplies independent pharmacies across the country.
    Your first assignment is the top-selling OTC product in the allergy
    category: a private-label antihistamine that has been quietly gaining
    shelf space over the past decade.

    The category manager, **Elena Voss**, drops a CSV on your desk:

    > "Sixteen years of monthly sales. We need a 12-month forward forecast —
    > point estimate and a 95% interval — to feed into next year's procurement
    > plan. The spring peak is what keeps me up at night: if we under-call it
    > we lose shelf space, if we over-call it we write off expired stock.
    > Make sure the model earns its keep on that month."

    The data:

    - `sales` — monthly units sold, January 2008 through December 2023
      (192 months). Integer counts. Clean series; no missing values.

    You will:

    1. Explore the series — trend, seasonal pattern, and what they suggest
       about the right modelling approach.
    2. Fit a seasonal `ARIMA` with **pmdarima**, hold out the last 12 months,
       and evaluate on the spring peak.
    3. Refit on all data and deliver the 12-month forecast with intervals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore the series

    Plot the raw series and any transformations that help you characterise it.
    Decompose or compute rolling statistics if that helps.

    What seasonal pattern is present? Is there a trend, and what shape does it
    take? Which calendar month is the spring peak? What does the seasonal
    structure suggest about the right modelling approach — and does that
    suggestion change depending on what you think is driving the seasonality?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference: snippets (commented — copy and adapt)

    Monthly index: `sales.index` is `DatetimeIndex` with `freq="MS"`.

    ```python
    # --- Log scale (sales > 0) and back-transform ---
    # import numpy as np
    # log_y = np.log(sales)
    # y_hat_units = np.exp(y_hat_log)
    # ci_lower_units = np.exp(ci_log[:, 0])
    # ci_upper_units = np.exp(ci_log[:, 1])

    # --- pmdarima seasonal ARIMA ("airline" on log scale): fit / predict ---
    # from pmdarima import ARIMA
    # train = log_y.iloc[:-12]   # example: hold out last year
    # fitted = ARIMA(
    #     order=(0, 1, 1),
    #     seasonal_order=(0, 1, 1, 12),
    #     enforce_stationarity=True,
    #     enforce_invertibility=True,
    #     suppress_warnings=True,
    # ).fit(train)
    # y_hat_log, ci_log = fitted.predict(n_periods=12, return_conf_int=True, alpha=0.05)
    #
    # # Forecast timestamps (first month after training ends):
    # import pandas as pd
    # future = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    # forecast_series = pd.Series(y_hat_log, index=future)
    ```
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _(np, pd, plt, sales):
    def plot_history_forecast(
        history,
        forecast,
        *,
        interval_lower=None,
        interval_upper=None,
        ax=None,
        title=None,
        ylabel=None,
        show_history=True,
        forecast_label="Forecast",
        forecast_color="#D55E00",
        cutoff=None,
    ):
        """Plot `history` and `forecast` (pd.Series). Optional interval bounds on the forecast index."""
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))
        vline_at = cutoff if cutoff is not None else history.index[-1]
        ax.axvline(vline_at, color="grey", linestyle=":", linewidth=1.2, zorder=1)
        if show_history:
            ax.plot(
                history.index,
                history.values,
                color="#4C72B0",
                lw=2,
                label="history",
                zorder=2,
            )
        ax.plot(
            forecast.index,
            forecast.values,
            color=forecast_color,
            lw=2,
            marker="o",
            markersize=4,
            linestyle="--",
            label=forecast_label,
            zorder=2,
        )
        if interval_lower is not None and interval_upper is not None:
            ax.fill_between(
                forecast.index,
                interval_lower.loc[forecast.index].values,
                interval_upper.loc[forecast.index].values,
                color=forecast_color,
                alpha=0.15,
                label="interval",
            )
        ax.legend(loc="best", fontsize=9)
        if title is not None:
            ax.set_title(title)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return ax

    train = sales.iloc[:-12]
    future = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    level = float(train.iloc[-1])
    dummy = pd.Series(np.full(12, level), index=future)
    dummy_lower = pd.Series(np.full(12, level * 0.92), index=future)
    dummy_upper = pd.Series(np.full(12, level * 1.08), index=future)
    plot_history_forecast(
        train,
        dummy,
        interval_lower=dummy_lower,
        interval_upper=dummy_upper,
        ylabel="units",
        title="Example: naive flat extension with illustrative ±8% band",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit seasonal ARIMA and evaluate on a held-out period

    Hold out the last 12 months. Fit an `ARIMA` model from **pmdarima** on
    the remaining data (same sklearn-style pattern as elsewhere: orders in
    `ARIMA(...)`, training series in `fit(...)`), then produce 12-step-ahead
    forecasts on the held-out window.

    Defend the specification you chose. What is the MAE on the held-out window?
    Does the model capture the spring peak adequately?
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Deliver the forecast

    Refit the model on all 192 months, then forecast 12 months ahead from the
    end of the series. Extract a 95% prediction interval.

    Present the full 12-month forecast table (point estimate and interval for
    each month). Call out the spring peak month specifically: what is the
    point forecast and how wide is the interval? Is the model precise enough
    to be useful for procurement planning?
    """)
    return


@app.cell
def _():
    # Your code here
    return


if __name__ == "__main__":
    app.run()
