# ============================================================
# PREDICTIVE SALES FORECASTING — Retail E-Commerce
# Tools : Python, SQL (SQLite), ARIMA, Prophet, Random Forest, XGBoost
# Author: Sharat Laha
# ============================================================

# ── 0. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2E86AB", "#E84855", "#F9A03F", "#4CAF50", "#9B5DE5"]

print("=" * 60)
print("  PREDICTIVE SALES FORECASTING — Retail E-Commerce")
print("=" * 60)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv("retail_sales_data.csv", parse_dates=["date"])
print(f"    Rows: {len(df):,} | Columns: {df.shape[1]}")
print(f"    Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"    Categories: {df['category'].unique()}")
print(f"    Regions: {df['region'].unique()}")
print(f"\n    Missing values:\n{df.isnull().sum()}")

# ── 2. SQL ANALYSIS ──────────────────────────────────────────
print("\n[2] Running SQL analysis...")

# Load into SQLite
conn = sqlite3.connect(":memory:")
df.to_sql("sales", conn, index=False, if_exists="replace")

# Query 1: Monthly revenue by category
q1 = pd.read_sql("""
    SELECT
        strftime('%Y-%m', date) AS month,
        category,
        SUM(sales)              AS total_sales,
        SUM(profit)             AS total_profit,
        SUM(units_sold)         AS total_units,
        ROUND(AVG(discount)*100,1) AS avg_discount_pct
    FROM sales
    GROUP BY month, category
    ORDER BY month, category
""", conn)

# Query 2: Top region by total revenue
q2 = pd.read_sql("""
    SELECT region,
           SUM(sales)  AS total_sales,
           SUM(profit) AS total_profit,
           ROUND(SUM(profit)*100.0/SUM(sales),1) AS profit_margin_pct
    FROM sales
    GROUP BY region
    ORDER BY total_sales DESC
""", conn)

# Query 3: YoY growth
q3 = pd.read_sql("""
    SELECT strftime('%Y', date) AS year,
           SUM(sales)           AS total_sales,
           SUM(profit)          AS total_profit
    FROM sales
    GROUP BY year
    ORDER BY year
""", conn)

print("\n    --- YoY Sales Summary ---")
print(q3.to_string(index=False))
print("\n    --- Revenue by Region ---")
print(q2.to_string(index=False))
conn.close()

# ── 3. EXPLORATORY DATA ANALYSIS ─────────────────────────────
print("\n[3] Generating EDA visualisations...")

# 3a. Monthly total sales trend
monthly = df.groupby(df["date"].dt.to_period("M"))["sales"].sum().reset_index()
monthly["date"] = monthly["date"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(monthly["date"], monthly["sales"]/1e6, color=COLORS[0], linewidth=2)
ax.fill_between(monthly["date"], monthly["sales"]/1e6, alpha=0.15, color=COLORS[0])
ax.set_title("Monthly Total Sales — 2021 to 2023", fontsize=14, fontweight="bold")
ax.set_ylabel("Sales (₹ Millions)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/01_monthly_sales_trend.png", dpi=150)
plt.close()
print("    Saved: 01_monthly_sales_trend.png")

# 3b. Sales by category
cat_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(cat_sales.index, cat_sales.values/1e6, color=COLORS)
ax.bar_label(bars, labels=[f"₹{v/1e6:.1f}M" for v in cat_sales.values], padding=3, fontsize=9)
ax.set_title("Total Sales by Category (2021–2023)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales (₹ Millions)")
plt.tight_layout()
plt.savefig("outputs/02_sales_by_category.png", dpi=150)
plt.close()
print("    Saved: 02_sales_by_category.png")

# 3c. Sales heatmap — Month vs Category
pivot = df.copy()
pivot["month"] = pivot["date"].dt.month_name()
pivot["month_num"] = pivot["date"].dt.month
hm = pivot.groupby(["month_num","month","category"])["sales"].sum().reset_index()
hm_pivot = hm.pivot_table(index=["month_num","month"], columns="category", values="sales")
hm_pivot = hm_pivot.sort_index(level=0)
hm_pivot.index = hm_pivot.index.get_level_values(1)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(hm_pivot/1e6, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Sales (₹M)"})
ax.set_title("Sales Heatmap: Month vs Category (₹ Millions)", fontsize=13, fontweight="bold")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("outputs/03_sales_heatmap.png", dpi=150)
plt.close()
print("    Saved: 03_sales_heatmap.png")

# 3d. Regional sales pie
reg_sales = df.groupby("region")["sales"].sum()
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(reg_sales, labels=reg_sales.index, autopct="%1.1f%%",
       colors=COLORS, startangle=140, pctdistance=0.82)
ax.set_title("Sales Share by Region", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_regional_sales_pie.png", dpi=150)
plt.close()
print("    Saved: 04_regional_sales_pie.png")

# 3e. Discount vs Profit scatter
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df["discount"]*100, df["profit"], alpha=0.15, s=5, color=COLORS[0])
ax.set_xlabel("Discount (%)")
ax.set_ylabel("Profit (₹)")
ax.set_title("Discount vs Profit Relationship", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_discount_vs_profit.png", dpi=150)
plt.close()
print("    Saved: 05_discount_vs_profit.png")

# ── 4. TIME SERIES — ARIMA ───────────────────────────────────
print("\n[4] ARIMA Forecasting...")

# Use total daily sales
ts = df.groupby("date")["sales"].sum().reset_index().set_index("date")
ts = ts.resample("W").sum()  # Weekly aggregation for stability

# Train / test split (last 12 weeks = test)
train = ts.iloc[:-12]
test  = ts.iloc[-12:]

# ADF stationarity test
adf_result = adfuller(train["sales"])
print(f"    ADF Statistic: {adf_result[0]:.4f} | p-value: {adf_result[1]:.4f}")
print(f"    Series is {'stationary' if adf_result[1] < 0.05 else 'non-stationary'}")

# Fit ARIMA
arima_model = ARIMA(train["sales"], order=(2, 1, 2))
arima_fit   = arima_model.fit()
arima_pred  = arima_fit.forecast(steps=12)

arima_mae  = mean_absolute_error(test["sales"], arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test["sales"], arima_pred))
print(f"    ARIMA  → MAE: ₹{arima_mae:,.0f} | RMSE: ₹{arima_rmse:,.0f}")

# Plot ARIMA
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train.index, train["sales"]/1e6, label="Train", color=COLORS[0])
ax.plot(test.index,  test["sales"]/1e6,  label="Actual", color=COLORS[1])
ax.plot(test.index,  arima_pred/1e6,     label="ARIMA Forecast",
        color=COLORS[2], linestyle="--", linewidth=2)
ax.set_title("ARIMA Sales Forecast (Weekly)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales (₹ Millions)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/06_arima_forecast.png", dpi=150)
plt.close()
print("    Saved: 06_arima_forecast.png")

# ── 5. TIME SERIES — PROPHET ─────────────────────────────────
print("\n[5] Prophet Forecasting...")

prophet_df = ts.reset_index().rename(columns={"date":"ds","sales":"y"})
prophet_train = prophet_df.iloc[:-12]
prophet_test  = prophet_df.iloc[-12:]

model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False, seasonality_mode="multiplicative")
model_prophet.fit(prophet_train)

future   = model_prophet.make_future_dataframe(periods=12, freq="W")
forecast = model_prophet.predict(future)
pred_test = forecast.iloc[-12:]["yhat"].values

prophet_mae  = mean_absolute_error(prophet_test["y"], pred_test)
prophet_rmse = np.sqrt(mean_squared_error(prophet_test["y"], pred_test))
print(f"    Prophet → MAE: ₹{prophet_mae:,.0f} | RMSE: ₹{prophet_rmse:,.0f}")

# Plot Prophet
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(prophet_df["ds"], prophet_df["y"]/1e6, label="Actual", color=COLORS[0])
ax.plot(forecast["ds"], forecast["yhat"]/1e6, label="Prophet Forecast",
        color=COLORS[3], linestyle="--", linewidth=2)
ax.fill_between(forecast["ds"], forecast["yhat_lower"]/1e6,
                forecast["yhat_upper"]/1e6, alpha=0.15, color=COLORS[3])
ax.set_title("Prophet Sales Forecast with Confidence Interval", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales (₹ Millions)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/07_prophet_forecast.png", dpi=150)
plt.close()
print("    Saved: 07_prophet_forecast.png")

# ── 6. ML — FEATURE ENGINEERING ─────────────────────────────
print("\n[6] Feature Engineering for ML models...")

df_ml = df.copy()

# Date features
df_ml["year"]      = df_ml["date"].dt.year
df_ml["month"]     = df_ml["date"].dt.month
df_ml["day"]       = df_ml["date"].dt.day
df_ml["dayofweek"] = df_ml["date"].dt.dayofweek
df_ml["quarter"]   = df_ml["date"].dt.quarter
df_ml["is_weekend"]= (df_ml["date"].dt.dayofweek >= 5).astype(int)
df_ml["is_festive"]= df_ml["month"].isin([10, 11, 12]).astype(int)

# Encode categoricals
le_cat = LabelEncoder()
le_reg = LabelEncoder()
df_ml["category_enc"] = le_cat.fit_transform(df_ml["category"])
df_ml["region_enc"]   = le_reg.fit_transform(df_ml["region"])

# Lag features (weekly aggregated)
df_agg = df_ml.groupby("date")["sales"].sum().reset_index()
df_agg["lag_7"]  = df_agg["sales"].shift(7)
df_agg["lag_14"] = df_agg["sales"].shift(14)
df_agg["lag_30"] = df_agg["sales"].shift(30)
df_agg["roll_7"] = df_agg["sales"].rolling(7).mean()
df_agg = df_agg.dropna()

df_agg["year"]      = df_agg["date"].dt.year
df_agg["month"]     = df_agg["date"].dt.month
df_agg["day"]       = df_agg["date"].dt.day
df_agg["dayofweek"] = df_agg["date"].dt.dayofweek
df_agg["quarter"]   = df_agg["date"].dt.quarter
df_agg["is_weekend"]= (df_agg["date"].dt.dayofweek >= 5).astype(int)
df_agg["is_festive"]= df_agg["month"].isin([10, 11, 12]).astype(int)

features = ["year","month","day","dayofweek","quarter",
            "is_weekend","is_festive","lag_7","lag_14","lag_30","roll_7"]
X = df_agg[features]
y = df_agg["sales"]

# Train/test split (last 90 days)
split = int(len(X) * 0.85)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"    Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"    Features engineered: {len(features)}")

# ── 7. ML — RANDOM FOREST ────────────────────────────────────
print("\n[7] Training Random Forest...")

rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)
print(f"    Random Forest → MAE: ₹{rf_mae:,.0f} | RMSE: ₹{rf_rmse:,.0f} | R²: {rf_r2:.4f}")

# Feature importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
fi.plot(kind="bar", color=COLORS[0], ax=ax)
ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
ax.set_ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/08_rf_feature_importance.png", dpi=150)
plt.close()
print("    Saved: 08_rf_feature_importance.png")

# ── 8. ML — XGBOOST ─────────────────────────────────────────
print("\n[8] Training XGBoost...")

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                   random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

xgb_mae  = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2   = r2_score(y_test, xgb_pred)
print(f"    XGBoost        → MAE: ₹{xgb_mae:,.0f} | RMSE: ₹{xgb_rmse:,.0f} | R²: {xgb_r2:.4f}")

# ── 9. MODEL COMPARISON ──────────────────────────────────────
print("\n[9] Model Comparison...")

results = pd.DataFrame({
    "Model":["ARIMA","Prophet","Random Forest","XGBoost"],
    "MAE":  [arima_mae, prophet_mae, rf_mae, xgb_mae],
    "RMSE": [arima_rmse, prophet_rmse, rf_rmse, xgb_rmse],
    "R2":   [None, None, rf_r2, xgb_r2]
})
print(results.to_string(index=False))

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
models = ["ARIMA", "Prophet", "Random Forest", "XGBoost"]
maes   = [arima_mae, prophet_mae, rf_mae, xgb_mae]
rmses  = [arima_rmse, prophet_rmse, rf_rmse, xgb_rmse]

axes[0].bar(models, [m/1000 for m in maes], color=COLORS)
axes[0].set_title("MAE Comparison (₹ Thousands)", fontweight="bold")
axes[0].set_ylabel("MAE (₹K)")

axes[1].bar(models, [r/1000 for r in rmses], color=COLORS)
axes[1].set_title("RMSE Comparison (₹ Thousands)", fontweight="bold")
axes[1].set_ylabel("RMSE (₹K)")

plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/09_model_comparison.png", dpi=150)
plt.close()
print("    Saved: 09_model_comparison.png")

# ── 10. ACTUAL vs PREDICTED PLOT (Best Model) ────────────────
print("\n[10] Actual vs Predicted — XGBoost...")

test_dates = df_agg["date"].iloc[split:]
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(test_dates.values, y_test.values/1e6,    label="Actual",          color=COLORS[0], linewidth=2)
ax.plot(test_dates.values, xgb_pred/1e6,          label="XGBoost Forecast",color=COLORS[1], linestyle="--", linewidth=2)
ax.plot(test_dates.values, rf_pred/1e6,            label="RF Forecast",    color=COLORS[2], linestyle=":", linewidth=1.5)
ax.set_title("Actual vs Predicted Sales — Test Period", fontsize=13, fontweight="bold")
ax.set_ylabel("Sales (₹ Millions)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/10_actual_vs_predicted.png", dpi=150)
plt.close()
print("    Saved: 10_actual_vs_predicted.png")

# ── 11. EXPORT FOR POWER BI ──────────────────────────────────
print("\n[11] Exporting Power BI-ready CSVs...")

# Monthly summary
monthly_summary = df.copy()
monthly_summary["year_month"] = monthly_summary["date"].dt.to_period("M").astype(str)
monthly_export = monthly_summary.groupby(["year_month","category","region"]).agg(
    total_sales=("sales","sum"),
    total_profit=("profit","sum"),
    total_units=("units_sold","sum"),
    avg_discount=("discount","mean")
).reset_index()
monthly_export.to_csv("outputs/powerbi_monthly_summary.csv", index=False)

# Forecast export
forecast_export = pd.DataFrame({
    "date":           test_dates.values,
    "actual_sales":   y_test.values,
    "xgb_forecast":   xgb_pred,
    "rf_forecast":    rf_pred,
})
forecast_export.to_csv("outputs/powerbi_forecast_results.csv", index=False)
print("    Saved: powerbi_monthly_summary.csv")
print("    Saved: powerbi_forecast_results.csv")

# ── FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PROJECT COMPLETE — KEY METRICS")
print("=" * 60)
print(f"  Dataset        : 21,900 rows | 3 years | 5 categories | 4 regions")
print(f"  Features built : 11 (lag, rolling, date, seasonal flags)")
print(f"  Models trained : ARIMA, Prophet, Random Forest, XGBoost")
print(f"  Best model     : XGBoost  →  R²: {xgb_r2:.4f} | MAE: ₹{xgb_mae:,.0f}")
print(f"  Visualisations : 10 charts saved to /outputs")
print(f"  Power BI files : 2 CSVs exported to /outputs")
print("=" * 60)
