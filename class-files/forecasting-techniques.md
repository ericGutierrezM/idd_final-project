# Forecasting Techniques — Course Summary

## Overview of Exercises

| # | Case | Technique(s) | Horizon | Frequency |
|---|------|-------------|---------|-----------|
| 01 | Grid Load (Stochastia) | Trend extrapolation / visual analysis | 10 years | Annual |
| 02 | Financial Product (Bayes Capital) | OLS linear regression with prediction intervals | 12 months | Monthly |
| 03 | World CO2 (IEA) | ARIMA | 25 years | Annual |
| 04 | Allergy Tablets (Medistor) | Seasonal ARIMA (SARIMA) | 12 months | Monthly |
| 05 | Procurement Planning (Medistor) | MILP (optimization under uncertainty) | 12 months | Monthly |

---

## Techniques

### 1. Trend Extrapolation (Visual / Naive)
**Exercise 01 — Grid Load**

Fit a simple trend line (by eye or OLS) to a long historical series and project it forward. No statistical model required.

**When to use:**
- Long-horizon planning where precision matters less than order of magnitude
- Series with a clear, stable directional trend and no complex seasonality
- Communicating to non-technical stakeholders who need a defensible number fast

**Limitations:**
- No uncertainty quantification
- Sensitive to structural breaks (e.g. EV adoption inflection)
- Cannot capture seasonality or cycles

---

### 2. OLS Linear Regression with Prediction Intervals
**Exercise 02 — Bayes Capital financial product**

Fit a linear model with time-based features (trend, seasonal dummies, etc.) using OLS. Use the model's standard errors to construct prediction intervals at each forecast step.

**When to use:**
- Series with a linear trend and/or additive seasonality
- When interpretability of coefficients matters (e.g. "each month adds X units")
- When you need calibrated uncertainty intervals alongside point forecasts
- Relatively short horizon (intervals widen quickly)

**Key implementation:** `statsmodels.api` (OLS with `get_prediction()` for intervals)

**Limitations:**
- Assumes linearity and constant variance (homoskedasticity)
- Intervals assume residuals are normally distributed
- Poor fit for nonlinear trends or multiplicative seasonality

---

### 3. ARIMA (AutoRegressive Integrated Moving Average)
**Exercise 03 — World CO2 emissions**

Model the series as a function of its own past values (AR), past forecast errors (MA), and differences to achieve stationarity (I). Orders are `(p, d, q)`.

**When to use:**
- Univariate series with no strong seasonality (or after seasonal adjustment)
- Stationary or near-stationary series (or made stationary via differencing)
- When you want to capture autocorrelation structure in the residuals
- Medium-term horizons where the series has a clear level/trend but no repeating seasonal cycle

**Key decisions:**
- Use ACF/PACF plots or `auto_arima` to select `p`, `d`, `q`
- Apply log transformation for exponential growth or right-skewed series
- Check residuals for remaining autocorrelation (Ljung-Box test)

**Key implementation:** `pmdarima.ARIMA(order=(p, d, q))`

**Limitations:**
- Does not handle seasonality without seasonal extension (see SARIMA)
- Intervals grow wide over long horizons
- Assumes the data-generating process is stable

---

### 4. Seasonal ARIMA (SARIMA)
**Exercise 04 — OTC allergy tablet demand**

Extends ARIMA with seasonal AR, I, and MA terms: `(p, d, q)(P, D, Q, m)` where `m` is the seasonal period (e.g. 12 for monthly data).

**When to use:**
- Series with a clear, repeating seasonal pattern (monthly, weekly, daily)
- When the seasonality is stable across years (consistent amplitude and timing)
- When you need prediction intervals that reflect seasonal uncertainty
- Common for: retail sales, pharmaceutical demand, utility consumption, food delivery orders

**Key decisions:**
- Apply seasonal differencing `D=1` when the seasonal pattern is non-stationary
- "Airline model" `(0,1,1)(0,1,1,12)` is a strong default for monthly data with trend + seasonality
- Log-transform if seasonality amplitude grows with the level (multiplicative seasonality)

**Key implementation:** `pmdarima.ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,m))`

**Limitations:**
- Requires sufficient history (at least 2 full seasonal cycles)
- Assumes the seasonal pattern is stable — breaks with structural change
- Not designed for multiple seasonal cycles (e.g. hourly data with both daily and weekly patterns)

---

### 5. Mixed-Integer Linear Programming (MILP)
**Exercise 05 — Medistor procurement optimization**

Not a forecasting technique per se — MILP is used **downstream of a forecast** to make optimal decisions under the constraints the forecast implies. Given point forecasts and intervals as demand estimates, solve for the ordering schedule that minimizes total cost subject to hard constraints (capacity, lead times, pallet minimums).

**When to use:**
- After generating a demand forecast, when you need to translate it into an operational plan
- When there are hard constraints: warehouse capacity, supplier minimums, lead times, integer quantities
- When the cost structure is explicit and optimization is more appropriate than heuristics

**Key concepts:**
- Decision variables: order quantities per period, emergency sourcing
- Objective: minimize total cost (purchase + holding + emergency premium)
- Inventory balance equation: `I_t = I_{t-1} + order_{t-1} + emergency_t - demand_t`

**Key implementation:** `ortools.linear_solver.pywraplp` with `SCIP` solver

**Limitations:**
- Requires explicit cost parameters (often hard to estimate)
- Sensitive to forecast accuracy — garbage in, garbage out
- Deterministic formulation ignores demand uncertainty (stochastic programming needed for robustness)

---

## Choosing a Technique — Quick Guide

| Situation | Recommended approach |
|-----------|---------------------|
| No seasonal pattern, clear trend, long horizon | Trend extrapolation or ARIMA |
| Single seasonal cycle (e.g. annual), linear trend | OLS with seasonal dummies or SARIMA |
| Single seasonal cycle, non-linear or multiplicative | SARIMA on log-transformed series |
| Multiple seasonal cycles (e.g. hourly: daily + weekly) | Prophet, or ML models with time features |
| Need uncertainty intervals | OLS, ARIMA, SARIMA (all provide CIs natively) |
| Need to optimize decisions downstream of forecast | MILP (e.g. procurement, staffing) |
| Very long horizon (decades), structural uncertainty | ARIMA + scenario analysis |

---

## Relevance to the Glovo Project

The Glovo data has **two seasonal cycles** (daily + weekly) and an **upward trend** — a combination that neither plain ARIMA nor SARIMA handles well natively. Recommended approaches:

- **Naive baseline:** same hour, same day-of-week from the prior week
- **SARIMA:** viable if you aggregate or focus on a single seasonal period
- **Prophet:** designed for multiple seasonality; handles daily + weekly natively
- **ML with time features:** (e.g. LightGBM) with hour-of-day, day-of-week, lag features — often the strongest performer on multi-seasonal data
