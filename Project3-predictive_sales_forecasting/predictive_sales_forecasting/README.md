# Predictive Sales Forecasting — Retail E-Commerce

## Project Overview
End-to-end sales forecasting project on a retail e-commerce dataset covering 3 years of daily sales data across 5 product categories and 4 regions. The project compares traditional time series models (ARIMA, Prophet) with machine learning models (Random Forest, XGBoost) to forecast weekly sales and export results for Power BI dashboards.

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn
- **Time Series:** ARIMA (statsmodels), Prophet (Meta)
- **Machine Learning:** Random Forest, XGBoost (Scikit-learn)
- **SQL:** SQLite for aggregation and business queries
- **Visualisation:** 10 charts + Power BI-ready CSV exports

## Dataset
- 21,900 rows of daily retail sales data (2021–2023)
- Features: date, category, region, sales, units sold, discount, profit
- Categories: Electronics, Clothing, Grocery, Furniture, Sports
- Regions: North, South, East, West

## Project Structure
```
predictive_sales_forecasting/
├── generate_data.py          # Synthetic dataset generator
├── sales_forecasting.py      # Main analysis script
├── retail_sales_data.csv     # Generated dataset
├── outputs/
│   ├── 01_monthly_sales_trend.png
│   ├── 02_sales_by_category.png
│   ├── 03_sales_heatmap.png
│   ├── 04_regional_sales_pie.png
│   ├── 05_discount_vs_profit.png
│   ├── 06_arima_forecast.png
│   ├── 07_prophet_forecast.png
│   ├── 08_rf_feature_importance.png
│   ├── 09_model_comparison.png
│   ├── 10_actual_vs_predicted.png
│   ├── powerbi_monthly_summary.csv
│   └── powerbi_forecast_results.csv
└── README.md
```

## Key Steps
1. **Data Generation** — Realistic synthetic retail dataset with seasonality, weekend effects, festival boosts, and YoY growth
2. **SQL Analysis** — YoY revenue growth, regional performance, category breakdown using SQLite
3. **EDA** — 5 visualisations: trend, category, heatmap, regional pie, discount vs profit
4. **ARIMA** — Classical time series forecasting with stationarity testing (ADF)
5. **Prophet** — Meta's forecasting model with multiplicative seasonality
6. **Feature Engineering** — 11 features including lag features (7/14/30 day), rolling mean, date parts, seasonal flags
7. **Random Forest** — Ensemble ML model with feature importance analysis
8. **XGBoost** — Gradient boosting model
9. **Model Comparison** — MAE and RMSE comparison across all 4 models
10. **Power BI Export** — Monthly summary and forecast CSVs ready for dashboard

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| ARIMA | ₹3,49,259 | ₹3,49,839 | — |
| Prophet | ₹42,304 | ₹76,387 | — |
| Random Forest | ₹13,268 | ₹19,382 | 0.8777 |
| XGBoost | ₹14,932 | ₹20,982 | 0.8567 |

**Best Model: Random Forest** with R² of 0.8777 and MAE of ₹13,268

## Business Insights
- **YoY Growth:** Sales grew from ₹8.1Cr (2021) → ₹8.7Cr (2022) → ₹9.4Cr (2023) (~8% annual growth)
- **Top Region:** North region leads with ₹7.16Cr total sales
- **Festive Season:** Oct–Dec shows 40% higher sales across all categories
- **Top Category:** Grocery drives highest volume; Electronics highest per-unit value
- **Discount Impact:** Higher discounts beyond 20% show diminishing profit returns

## How to Run
```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet scikit-learn xgboost
python generate_data.py
python sales_forecasting.py
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg)
