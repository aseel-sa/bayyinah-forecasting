#  BAYYINAH
### AI-Powered Demand Forecasting Platform

> End-to-end AI platform for demand forecasting, model comparison, and business decision support in manufacturing.

---

##  Problem

Manufacturing businesses rely on spreadsheets and manual planning — leading to stockouts, overstock, and missed sales. Small and medium manufacturers lack access to enterprise forecasting tools or data science teams.

---

##  Solution

BAYYINAH turns messy sales data into clear demand forecasts and actionable business insights — no data science team required.

---

##  How It Works

1. Upload a CSV or Excel file
2. Intake layer auto-detects date, product, and demand columns
3. Quality layer checks missing values, duplicates, and outliers
4. Feature engineering builds lags, rolling averages, and seasonality signals
5. Forecasting engine compares 10 models per product
6. Best model selected per product using walk-forward validation
7. Platform generates forecasts, insights, and scenario simulations

---

##  Key Features

- ✅ Automatic column detection
- ✅ Data quality checks with human-in-the-loop approval
- ✅ Leakage-safe feature engineering
- ✅ 10 forecasting models compared per product
- ✅ Walk-forward backtesting
- ✅ wMAPE-based model evaluation
- ✅ Business-objective scoring (Accuracy / Balanced / Stockout / Overstock)
- ✅ Scenario simulation (What-if analysis)
- ✅ AI-generated business insights

---

##  Forecasting Models

| # | Model |
|---|-------|
| 1 | Naive |
| 2 | Seasonal Naive |
| 3 | Moving Average |
| 4 | Exponential Smoothing |
| 5 | SARIMA |
| 6 | Prophet |
| 7 | Croston |
| 8 | Random Forest |
| 9 | XGBoost |
| 10 | Hybrid Ensemble |

---

##  Model Evaluation

- **Walk-forward validation** — trains on past, tests on future
- **wMAPE** — primary metric (weighted, works across products)
- **MAE, RMSE, MAPE, Bias** — supporting metrics

---

##  Business Objective Scoring

| Objective | Focus |
|-----------|-------|
| Accuracy Focused | Lowest historical error |
| Balanced Planning | Low error + low bias |
| Stockout Protection | Penalizes under-forecasting |
| Overstock Control | Penalizes over-forecasting |

---

##  Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Statsmodels` `Prophet` `FastAPI` `React`

---

##  Future Work

- Real-time data connectors (ERP / WMS integration)
- Automated alerts for anomalies and
