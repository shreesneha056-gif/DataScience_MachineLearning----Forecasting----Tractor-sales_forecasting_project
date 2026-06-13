# Tractor Sales Time Series Forecasting (ARIMA & SARIMA)

This repository contains an advanced time-series forecasting case study designed to analyze and predict historical monthly **Tractor Sales**. The project transitions from baseline classical moving averages to sophisticated parametric statistical formulations, specifically evaluating Autoregressive Integrated Moving Average (**ARIMA**) and Seasonal ARIMA (**SARIMA**) models to project dynamic manufacturing and retail demand patterns.

---

## 📂 Dataset Repository & Features

The baseline models extract, train, and test historical configurations hosted in a single transactional log.

### Dataset Profile (`Tractor-Sales.csv`)
* **Attributes:**
  - `Month-Year`: Date identifier formatted as `Mon-YY` (e.g., `Jan-03`, `Feb-03`), parsed and mapped as a strict Datetime Index.
  - `Number of Tractor Sold`: Target dependent variable tracking monthly unit distribution data.
* **Core Characteristics:** The dataset exhibits a non-linear compounding trend paired with distinct, expanding seasonal variations that typically peak during the peak agricultural seasons each year.

---

## 🛠️ Data Engineering & Mathematical Transformations

To conform to standard parametric assumptions, the data pipeline inside `Tractor_Sales_Forecasting_ARIMA,SARIMA.ipynb` applies several mathematical operations:

1. **Variance Stabilization:** Due to expanding seasonal swings (multiplicative patterns), a natural log transformation (`ln`) is applied to stabilize structural variance over time.
2. **Double Differencing:** To eliminate both local structural trends and repeating seasonal fluctuations, the series goes through regular and seasonal differencing:
   - **`ln_ts_diff2`**: Represents the log-transformed series that has been differenced to achieve complete stationarity.
3. **Statistical Validation:** Every transformation level is strictly validated using the **Augmented Dickey-Fuller (ADF)** unit-root test to ensure a mathematically sound model foundation.

---

## 🤖 Modeling Architecture & Optimization Pipeline

The implementation explores, tunes, and compares multiple layers of time-series complexity:

### 1. Holt-Winters Exponential Smoothing (Baseline)
* Deployed to capture baseline parameters across explicit Error, Trend, and Seasonal (`ETS`) configurations as a non-parametric benchmark.

### 2. Manual Parametric Order Identification
* **ACF and PACF Plots:** Autocorrelation and partial autocorrelation tracking metrics are utilized to determine initial logical limits for Autoregressive ($p$) and Moving Average ($q$) configurations.

### 3. Automated Search Framework (`auto_arima`)
* Leverages AIC (Akaike Information Criterion) optimization algorithms to traverse parameter combinations across the stationary log-differenced matrix:
  ```python
  # Isolate optimal seasonal order weights via step-wise grid execution
  auto_arima(ln_ts_diff2)