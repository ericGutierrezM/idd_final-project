import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    df = pd.read_csv(Path(__file__).parent / "data" / "01_grid_load.csv", index_col=0, parse_dates=True)
    return df, np, pd, plt


@app.cell(hide_code=True)
def _(df, mo, np, pd, plt):
    _start = pd.Timestamp(df.index.min()).strftime("%b %Y")
    _end = pd.Timestamp(df.index.max()).strftime("%b %Y")
    mo.md(
        f"**Data loaded:** `df` — {len(df)} observations ({_start}–{_end}). "
        f"Columns: {list(df.columns)}. "
        f"`pd`, `np`, `plt` available."
    )
    _ = np.array([]), plt.rcParams
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 01: Resource Planning for the Grid
    ## The Minister's Challenge: Preparing for an EV Future

    **Congratulations!** You've just been hired by the Department of Energy of Stochastia as a senior analyst. Your first task is critical: the government has allocated significant budget to prepare the electricity grid for the growing needs of the nation. The country can currently produce up to 50GWh of electricity. The Minister is particularly concerned about the large-scale adoption of electric vehicles (EVs) that has started in recent years and they expect will continue for the next decade.

    The Minister of Energy sits in their office, surrounded by reports about the climate crisis and the transition to electric mobility. They have one simple question for you:

    > *"How much additional generation capacity do we need to build to ensure we can always meet demand when everyone switches to EVs?"*

    Simple question. High stakes. Your analysis will determine billions of dollars in infrastructure investment. Unfortunately, you only have one day to come up with an answer. The previous analyst was fired after handing a report saying that *the nation will need an additional 150 GWh* on top of the current 50GWh the grid can serve. The Minister says:

    > Our previous Lead Analyst was an alarmist. Their final report claimed that within 10 years, our current 50 GWh capacity would be insufficient and that we need to quadruple our infrastructure to 200 GWh.
    >
    >Frankly, I find these numbers absurd. I look at our daily reports, and we rarely cross 30 GWh even on busy days. We have a 20 GWh surplus right now. The public is already complaining about electricity prices; I will not authorize billions in new power plants based on "ghost numbers".

    Your task in this exercise:
    1. **Analyze** the dataset to find patterns.
    2. Produce a simple **forecast** of electricity demand for the next 10 years. No statistical/ML models are required.
    3. **Present** a capacity plan to the Minister.
    """)
    return


@app.cell
def _(df):
    df.plot()
    return


if __name__ == "__main__":
    app.run()
