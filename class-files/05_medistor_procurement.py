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
    from ortools.linear_solver import pywraplp

    forecast = pd.read_csv(
        Path(__file__).parent / "data" / "05_medistor_procurement.csv",
        parse_dates=["month"],
    )
    forecast = forecast.set_index("month")
    return forecast, mo, np, pd, plt, pywraplp


@app.cell(hide_code=True)
def _(forecast, mo, np, pd, plt, pywraplp):
    _n = len(forecast)
    _start = pd.Timestamp(forecast.index[0]).strftime("%b %Y")
    _end = pd.Timestamp(forecast.index[-1]).strftime("%b %Y")
    _mean_fc = np.mean(forecast["forecast"])
    _ = (plt.rcParams, pywraplp.Solver)
    mo.md(
        f"**Forecast loaded:** {_n} months ({_start}–{_end}), "
        f"mean point forecast {_mean_fc:,.0f} units/month."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 05: Optimal procurement under demand uncertainty

    You have been in the procurement analytics team at **Medistor** for three months.
    The demand forecast you delivered last week — twelve monthly point estimates with
    95% prediction intervals for the private-label antihistamine — is now sitting on
    Elena Voss's desk. She walks over:

    > "Good work on the forecast. Now comes the hard part. I need a procurement
    > schedule for 2024: how many units do we order each month, and when? We have
    > 65,000 units available at the start of January — the last December shipment
    > just arrived. The supplier works on a one-month lead time, so whatever we
    > order this month arrives at the start of next month. They also have a
    > pallet minimum: every order must be a whole number of pallets, and each
    > pallet holds 10,000 units. The warehouse holds at most 120,000. One more
    > thing — the supplier just told us they can fill at most 7 pallets per
    > order due to their production constraints. Give me a plan that minimises cost."

    The data on your desk:

    - `forecast` — twelve-row DataFrame with column `forecast` for
      January–December 2024, from your SARIMAX model in Exercise 4.

    Cost and constraint parameters (defined in the cell below):

    | Parameter | Value |
    |---|---|
    | Unit purchase cost | €4.20 / unit |
    | Holding cost | €1.50 / unit / month |
    | Emergency reorder premium | €3.00 / unit (sourced outside the regular channel when stock runs out) |
    | Warehouse capacity | 120,000 units |
    | Starting inventory | 65,000 units (January 1 on-hand) |
    | Lead time | 1 month (order placed in month $t$ arrives at start of month $t+1$) |
    | Pallet size | 10,000 units; all orders must be a whole number of pallets |
    | Monthly order cap | 7 pallets (supplier production limit) |

    You will:

    1. Formulate the problem — decision variables, objective, constraints.
    2. Solve the MILP and deliver Elena's procurement schedule.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Parameters

    Define the cost and constraint constants here. Downstream cells consume them.
    """)
    return


@app.cell
def _():
    UNIT_COST = 4.20
    HOLDING_COST = 1.50
    EMERGENCY_PREMIUM = 3.00
    CAPACITY = 120_000
    I_START = 65_000
    PALLET_SIZE = 10_000
    MAX_PALLETS = 7
    return (
        CAPACITY,
        EMERGENCY_PREMIUM,
        HOLDING_COST,
        I_START,
        MAX_PALLETS,
        PALLET_SIZE,
        UNIT_COST,
    )


@app.cell(hide_code=True)
def _(
    CAPACITY,
    EMERGENCY_PREMIUM,
    HOLDING_COST,
    I_START,
    MAX_PALLETS,
    PALLET_SIZE,
    UNIT_COST,
    mo,
):
    mo.md(f"""
    **Parameters:** purchase €{UNIT_COST}/unit · "
        f"holding €{HOLDING_COST}/unit/month · emergency premium €{EMERGENCY_PREMIUM}/unit · "
        f"capacity {CAPACITY:,} · start inventory {I_START:,} · "
        f"pallet {PALLET_SIZE:,} · order cap {MAX_PALLETS} pallets/month.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problem formulation

    Before writing any code, write down in plain English (or maths):

    1. **Decision variables** — what are you choosing, and for which time periods?
       There are two sourcing channels: regular orders (pallet-constrained, one-month
       lead time) and emergency orders (no minimum, arrives within the month at a
       premium of €3.00 above the regular purchase price).
    2. **Objective function** — what are you minimising? What is the total cost per
       unit sourced via the emergency channel?
    3. **Constraints** — list every hard limit. Which constraints introduce integrality,
       and which one sets a hard ceiling on how much you can order in a single month?
    4. **Lead time** — how does the one-month lead time appear in the inventory
       balance equation? Write out $I_t$ in terms of $I_{t-1}$, the regular order
       placed in the previous month, any emergency units sourced in month $t$,
       and demand $d_t$.
    """)
    return


@app.cell
def _():
    # Your written answer here (use a comment block or a mo.md cell)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Solve the MILP

    Use `pywraplp` with the `SCIP` solver (it handles integer variables).

    Decision variables:
    - $o_t \in \{0, 1, \ldots, 7\}$: pallets ordered via the regular channel at the
      end of month $t$. Physical quantity: $10{,}000 \cdot o_t$ units, arrives at
      the start of month $t+1$. Bounded above by `MAX_PALLETS`.
    - $e_t \geq 0$: units sourced via the emergency channel in month $t$ (no lead
      time, no pallet minimum, costs €4.20 + €3.00 = €7.20 per unit).

    Inventory balance:

    $$I_t = I_{t-1} + 10{,}000\,o_{t-1} + e_t - d_t, \quad I_0 = 65{,}000, \quad o_{-1} = 0$$

    with $I_t \geq 0$ (emergency sourcing ensures demand is always met).

    Constraints:
    - Warehouse capacity: $0 \leq I_t \leq 120{,}000$
    - Monthly order cap: $o_t \leq \texttt{MAX\_PALLETS}$ (encode as the upper bound of the integer variable)
    - Pallet integrality: $o_t \in \mathbb{Z}_{\geq 0}$

    Print the monthly plan (month, regular pallets ordered, emergency units,
    end-of-month inventory) and the total cost breakdown. Plot sourcing
    (regular vs emergency stacked) and inventory level across the year.

    Once you have the plan, answer:

    - In which months does the optimizer resort to emergency sourcing, and why?
    - How does the holding cost of €1.50/unit/month compare to the emergency premium
      of €3.00/unit? What does the breakeven imply about the optimizer's willingness
      to build buffer stock in advance?
    - Would ordering fewer pallets in January be beneficial? Why or why not?
    """)
    return


@app.cell
def _():
    # Your code here
    return


if __name__ == "__main__":
    app.run()
