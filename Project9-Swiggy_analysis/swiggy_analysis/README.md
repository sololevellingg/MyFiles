# Swiggy Restaurant Data Analysis

## Project Overview
End-to-end restaurant analytics project on Swiggy platform data — analysing 2,000 restaurants and 12,000 orders across 10 Indian cities (2023–2024). Covers GMV analysis, cuisine popularity, delivery efficiency, rating prediction, discount impact, and K-Means restaurant segmentation.

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **SQL:** SQLite — city GMV, cuisine stats, price category analysis, monthly trends
- **ML:** Random Forest + Gradient Boosting (rating prediction), K-Means (restaurant segmentation)
- **BI:** 4 Power BI-ready CSV exports

## Dataset
- **2,000 restaurants** | **12,000 orders** | Jan 2023 – Dec 2024
- **10 cities:** Mumbai, Delhi, Bangalore, Hyderabad, Pune, Chennai, Kolkata, Ahmedabad, Jaipur, Lucknow
- **10 cuisines:** North Indian, South Indian, Chinese, Italian, Biryani, Fast Food, Desserts, Continental, Thai, Mexican
- Features: city, cuisine, price category, restaurant type, rating, votes, delivery time, cost, discount, GMV

## Key Insights

| Metric | Value |
|--------|-------|
| Total Restaurants | 2,000 |
| Total Orders | 12,000 |
| Total GMV | ₹92.5L |
| Avg Rating | 3.75 / 5.0 |
| Avg Delivery Time | 45 mins |
| Top City (GMV) | Delhi (₹10.2L) |
| Top Cuisine | Fast Food (1,403 orders) |
| K-Means Segments | k=2 (silhouette: 0.2198) |
| RF Rating Prediction MAE | 0.3428 |

## Analysis Breakdown

### SQL Queries
- City-level GMV, avg rating, avg delivery time
- Top 10 cuisines by orders and avg order value
- Price category performance (rating, delivery, order value)
- Restaurant type analysis (rating, delivery, cost)
- Monthly GMV trend (2023–2024)

### EDA (10 Visualisations)
1. GMV by city (bar chart)
2. Top 10 cuisines by orders
3. Rating distribution + avg rating by price category
4. Delivery time distribution + by city
5. Cost vs rating scatter by price category
6. Restaurant type — rating & delivery comparison
7. Monthly GMV trend (area chart)
8. Discount impact on rating and order value
9. K-Means restaurant segmentation scatter
10. ML feature importances

### K-Means Segmentation (k=2)
- **Cluster 0 (Premium):** 612 restaurants — avg rating 4.19, avg cost ₹1,100
- **Cluster 1 (Budget):** 1,388 restaurants — avg rating 3.56, avg cost ₹403

### ML — Rating Prediction
- **Random Forest:** MAE: 0.3428 | R²: 0.1196
- **Gradient Boosting:** MAE: 0.3370 | R²: 0.1470
- Rating is inherently subjective — low R² reflects natural variance in customer opinions

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py
python swiggy_analysis.py
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg/Files-v1)
