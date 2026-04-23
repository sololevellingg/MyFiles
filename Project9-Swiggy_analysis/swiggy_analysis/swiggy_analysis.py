# =============================================================================
# swiggy_analysis.py — Swiggy Restaurant Data Analysis
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn), SQL (SQLite)
# Author: Sharat Laha
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
BLUE   = "#1A5276"
RED    = "#C0392B"
GREEN  = "#1D8348"
AMBER  = "#D68910"
PURPLE = "#7D3C98"
ORANGE = "#BA4A00"
COLORS = [BLUE, RED, GREEN, AMBER, PURPLE, ORANGE, "#0E6655", "#1A237E", "#4A235A", "#1B5E20"]

print("=" * 65)
print("  SWIGGY RESTAURANT DATA ANALYSIS")
print("=" * 65)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
rest  = pd.read_csv("swiggy_analysis/restaurants.csv")
orders = pd.read_csv("swiggy_analysis/orders.csv", parse_dates=["order_date"])
merged = orders.merge(rest[["restaurant_id","rating","votes","is_veg","online_order"]], on="restaurant_id", how="left")

print(f"    Restaurants : {len(rest):,} | Cities: {rest['city'].nunique()}")
print(f"    Orders      : {len(orders):,} | GMV: ₹{orders['final_amount'].sum():,.0f}")
print(f"    Avg Rating  : {rest['rating'].mean():.2f} / 5.0")
print(f"    Avg Delivery: {rest['avg_delivery_time'].mean():.0f} mins")
print(f"    Veg %       : {rest['is_veg'].mean()*100:.1f}%")

# ── 2. SQL ────────────────────────────────────────────────────────────────────
print("\n[2] SQL analysis...")
conn = sqlite3.connect(":memory:")
rest.to_sql("restaurants",  conn, index=False, if_exists="replace")
orders.to_sql("orders",     conn, index=False, if_exists="replace")

q_city = pd.read_sql("""
    SELECT r.city,
           COUNT(DISTINCT r.restaurant_id)     AS restaurants,
           ROUND(AVG(r.rating),2)              AS avg_rating,
           ROUND(AVG(r.avg_delivery_time),1)   AS avg_delivery_mins,
           ROUND(AVG(r.avg_cost_two),0)        AS avg_cost,
           SUM(o.final_amount)                 AS total_gmv
    FROM restaurants r
    LEFT JOIN orders o ON r.restaurant_id = o.restaurant_id
    GROUP BY r.city ORDER BY total_gmv DESC
""", conn)

q_cuisine = pd.read_sql("""
    SELECT cuisine,
           COUNT(*) AS orders,
           ROUND(AVG(final_amount),0) AS avg_order_value,
           ROUND(AVG(delivery_time),1) AS avg_delivery_mins
    FROM orders GROUP BY cuisine ORDER BY orders DESC LIMIT 10
""", conn)

q_price = pd.read_sql("""
    SELECT r.price_category,
           COUNT(DISTINCT r.restaurant_id) AS restaurants,
           ROUND(AVG(r.rating),2)          AS avg_rating,
           ROUND(AVG(r.avg_delivery_time),1) AS avg_delivery,
           ROUND(AVG(o.final_amount),0)    AS avg_order_value,
           COUNT(o.order_id)               AS total_orders
    FROM restaurants r
    LEFT JOIN orders o ON r.restaurant_id = o.restaurant_id
    GROUP BY r.price_category ORDER BY avg_order_value DESC
""", conn)

q_type = pd.read_sql("""
    SELECT rest_type,
           COUNT(*) AS restaurants,
           ROUND(AVG(rating),2) AS avg_rating,
           ROUND(AVG(avg_delivery_time),1) AS avg_delivery,
           ROUND(AVG(avg_cost_two),0) AS avg_cost
    FROM restaurants GROUP BY rest_type ORDER BY avg_rating DESC
""", conn)

q_monthly = pd.read_sql("""
    SELECT strftime('%Y-%m', order_date) AS month,
           COUNT(*) AS orders,
           ROUND(SUM(final_amount),0) AS gmv,
           ROUND(AVG(final_amount),0) AS avg_order_value
    FROM orders GROUP BY month ORDER BY month
""", conn)

print("\n    --- City Performance ---")
print(q_city[["city","restaurants","avg_rating","avg_delivery_mins","total_gmv"]].to_string(index=False))
conn.close()

# ── 3. VISUALISATIONS ─────────────────────────────────────────────────────────
print("\n[3] Generating visualisations...")

# Chart 1: Top cities by GMV
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(q_city["city"], q_city["total_gmv"]/1e5,
              color=COLORS[:len(q_city)], edgecolor="white")
ax.bar_label(bars, labels=[f"₹{v:.0f}L" for v in q_city["total_gmv"]/1e5],
             padding=3, fontsize=8)
ax.set_title("Total GMV by City (₹ Lakhs)", fontsize=13, fontweight="bold")
ax.set_ylabel("GMV (₹ Lakhs)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/01_gmv_by_city.png", dpi=150)
plt.close()
print("    Saved: 01_gmv_by_city.png")

# Chart 2: Cuisine popularity
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(q_cuisine["cuisine"][::-1], q_cuisine["orders"][::-1], color=BLUE, edgecolor="white")
ax.bar_label(bars, labels=[f"{v:,}" for v in q_cuisine["orders"][::-1]], padding=4, fontsize=9)
ax.set_title("Top 10 Cuisines by Orders", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Orders")
plt.tight_layout()
plt.savefig("outputs/02_cuisine_popularity.png", dpi=150)
plt.close()
print("    Saved: 02_cuisine_popularity.png")

# Chart 3: Rating distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(rest["rating"], bins=20, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].axvline(rest["rating"].mean(), color=RED, linestyle="--",
                label=f"Avg: {rest['rating'].mean():.2f}")
axes[0].set_title("Restaurant Rating Distribution", fontweight="bold")
axes[0].set_xlabel("Rating")
axes[0].legend()

rating_price = rest.groupby("price_category")["rating"].mean().reindex(["Budget","Mid-Range","Premium"])
bars = axes[1].bar(rating_price.index, rating_price.values, color=[BLUE,AMBER,GREEN], edgecolor="white")
axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in rating_price.values], padding=3, fontsize=10)
axes[1].set_title("Avg Rating by Price Category", fontweight="bold")
axes[1].set_ylabel("Avg Rating")
axes[1].set_ylim(0, 5)
plt.suptitle("Rating Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_rating_analysis.png", dpi=150)
plt.close()
print("    Saved: 03_rating_analysis.png")

# Chart 4: Delivery time analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(rest["avg_delivery_time"], bins=20, color=AMBER, edgecolor="white", alpha=0.85)
axes[0].axvline(rest["avg_delivery_time"].mean(), color=RED, linestyle="--",
                label=f"Avg: {rest['avg_delivery_time'].mean():.0f} mins")
axes[0].set_title("Delivery Time Distribution", fontweight="bold")
axes[0].set_xlabel("Minutes")
axes[0].legend()

city_del = rest.groupby("city")["avg_delivery_time"].mean().sort_values()
axes[1].barh(city_del.index, city_del.values, color=ORANGE, edgecolor="white")
axes[1].axvline(city_del.mean(), color=RED, linestyle="--", linewidth=1.5)
axes[1].set_title("Avg Delivery Time by City (mins)", fontweight="bold")
axes[1].set_xlabel("Minutes")
plt.suptitle("Delivery Time Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_delivery_time.png", dpi=150)
plt.close()
print("    Saved: 04_delivery_time.png")

# Chart 5: Cost vs Rating scatter
fig, ax = plt.subplots(figsize=(9, 5))
for cat, color in zip(["Budget","Mid-Range","Premium"],[BLUE,AMBER,GREEN]):
    sub = rest[rest["price_category"]==cat]
    ax.scatter(sub["avg_cost_two"], sub["rating"], alpha=0.4, s=15,
               color=color, label=cat)
ax.set_xlabel("Avg Cost for Two (₹)")
ax.set_ylabel("Rating")
ax.set_title("Cost vs Rating by Price Category", fontsize=13, fontweight="bold")
ax.legend(title="Price Category")
plt.tight_layout()
plt.savefig("outputs/05_cost_vs_rating.png", dpi=150)
plt.close()
print("    Saved: 05_cost_vs_rating.png")

# Chart 6: Restaurant type analysis
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(q_type["rest_type"][::-1], q_type["avg_rating"][::-1], color=PURPLE, edgecolor="white")
axes[0].set_title("Avg Rating by Restaurant Type", fontweight="bold")
axes[0].set_xlabel("Avg Rating")
axes[1].barh(q_type["rest_type"][::-1], q_type["avg_delivery"][::-1], color=ORANGE, edgecolor="white")
axes[1].set_title("Avg Delivery Time by Type (mins)", fontweight="bold")
axes[1].set_xlabel("Minutes")
plt.suptitle("Restaurant Type Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/06_rest_type_analysis.png", dpi=150)
plt.close()
print("    Saved: 06_rest_type_analysis.png")

# Chart 7: Monthly GMV trend
q_monthly["month_dt"] = pd.to_datetime(q_monthly["month"])
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(q_monthly["month_dt"], q_monthly["gmv"]/1e5, color=BLUE, linewidth=2)
ax.fill_between(q_monthly["month_dt"], q_monthly["gmv"]/1e5, alpha=0.15, color=BLUE)
ax.set_title("Monthly GMV Trend (2023–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("GMV (₹ Lakhs)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/07_monthly_gmv.png", dpi=150)
plt.close()
print("    Saved: 07_monthly_gmv.png")

# Chart 8: Discount impact
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
disc_rating = rest.groupby("discount_pct")["rating"].mean()
axes[0].plot(disc_rating.index, disc_rating.values, "o-", color=BLUE, linewidth=2, markersize=6)
axes[0].set_title("Discount % vs Avg Rating", fontweight="bold")
axes[0].set_xlabel("Discount %")
axes[0].set_ylabel("Avg Rating")

disc_orders = orders.groupby("discount_amt")["final_amount"].mean().reset_index()
axes[1].scatter(orders["discount_amt"], orders["final_amount"],
                alpha=0.1, s=5, color=BLUE)
axes[1].set_title("Discount Amount vs Order Value", fontweight="bold")
axes[1].set_xlabel("Discount Amount (₹)")
axes[1].set_ylabel("Final Order Value (₹)")
plt.suptitle("Discount Impact Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/08_discount_impact.png", dpi=150)
plt.close()
print("    Saved: 08_discount_impact.png")

# ── 4. K-MEANS RESTAURANT SEGMENTATION ───────────────────────────────────────
print("\n[4] K-Means Restaurant Segmentation...")
seg_feat = rest[["rating","avg_cost_two","avg_delivery_time","votes","discount_pct"]].copy()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(seg_feat)

sil = {k: silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled))
       for k in range(2, 7)}
best_k = max(sil, key=sil.get)
print(f"    Optimal k={best_k} | Silhouette: {sil[best_k]:.4f}")

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rest["cluster"] = km.fit_predict(X_scaled)

cluster_summary = rest.groupby("cluster").agg(
    count=("restaurant_id","count"),
    avg_rating=("rating","mean"),
    avg_cost=("avg_cost_two","mean"),
    avg_delivery=("avg_delivery_time","mean")
).round(2)
print(f"\n{cluster_summary.to_string()}")

# Chart 9: Cluster scatter
fig, ax = plt.subplots(figsize=(9, 5))
for c in sorted(rest["cluster"].unique()):
    sub = rest[rest["cluster"]==c]
    ax.scatter(sub["avg_cost_two"], sub["rating"], alpha=0.5, s=20,
               color=COLORS[c], label=f"Cluster {c}")
ax.set_xlabel("Avg Cost for Two (₹)")
ax.set_ylabel("Rating")
ax.set_title(f"Restaurant Segmentation — K-Means (k={best_k})\nSilhouette: {sil[best_k]:.4f}",
             fontsize=12, fontweight="bold")
ax.legend(title="Cluster")
plt.tight_layout()
plt.savefig("outputs/09_restaurant_clusters.png", dpi=150)
plt.close()
print("    Saved: 09_restaurant_clusters.png")

# ── 5. ML — RATING PREDICTION ─────────────────────────────────────────────────
print("\n[5] ML — Restaurant Rating Prediction...")
ml = rest.copy()
le = LabelEncoder()
for col in ["city","price_category","rest_type","area"]:
    ml[col] = le.fit_transform(ml[col].astype(str))

features = ["city","price_category","rest_type","avg_cost_two",
            "votes","avg_delivery_time","discount_pct","is_veg","online_order"]
X = ml[features]
y = ml["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf  = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

gb  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_mae  = mean_absolute_error(y_test, gb_pred)
gb_r2   = r2_score(y_test, gb_pred)

print(f"    Random Forest    → MAE: {rf_mae:.4f} | R²: {rf_r2:.4f}")
print(f"    Gradient Boosting→ MAE: {gb_mae:.4f} | R²: {gb_r2:.4f}")

# Chart 10: Feature importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(fi.index, fi.values, color=COLORS[:len(fi)], edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.3f}" for v in fi.values], padding=3, fontsize=9)
ax.set_title(f"Rating Prediction — Feature Importances\n(RF MAE: {rf_mae:.4f} | R²: {rf_r2:.4f})",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Importance Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/10_ml_feature_importance.png", dpi=150)
plt.close()
print("    Saved: 10_ml_feature_importance.png")

# ── 6. POWER BI EXPORT ────────────────────────────────────────────────────────
print("\n[6] Exporting Power BI CSVs...")
q_city.to_csv("outputs/powerbi_city_performance.csv", index=False)
q_cuisine.to_csv("outputs/powerbi_cuisine_stats.csv", index=False)
q_monthly.to_csv("outputs/powerbi_monthly_gmv.csv", index=False)
q_price.to_csv("outputs/powerbi_price_category.csv", index=False)
print("    Saved: 4 Power BI CSVs")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
top_city    = q_city.iloc[0]["city"]
top_cuisine = q_cuisine.iloc[0]["cuisine"]
print(f"\n{'=' * 65}")
print("  KEY INSIGHTS")
print(f"{'=' * 65}")
print(f"  Restaurants      : {len(rest):,} | Cities: {rest['city'].nunique()}")
print(f"  Total GMV        : ₹{orders['final_amount'].sum():,.0f}")
print(f"  Top City         : {top_city} (₹{q_city.iloc[0]['total_gmv']/1e5:.0f}L GMV)")
print(f"  Top Cuisine      : {top_cuisine} ({q_cuisine.iloc[0]['orders']:,} orders)")
print(f"  Avg Rating       : {rest['rating'].mean():.2f} | Avg Delivery: {rest['avg_delivery_time'].mean():.0f} mins")
print(f"  Optimal Segments : k={best_k} (silhouette: {sil[best_k]:.4f})")
print(f"  Best ML Model    : Random Forest → MAE: {rf_mae:.4f} | R²: {rf_r2:.4f}")
print(f"  Visualisations   : 10 charts + 4 Power BI CSVs")
print(f"{'=' * 65}")
