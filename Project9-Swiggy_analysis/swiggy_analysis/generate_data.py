# generate_data.py — Swiggy Restaurant Dataset Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N_RESTAURANTS = 2000
N_ORDERS      = 12000
START         = datetime(2023, 1, 1)
END           = datetime(2024, 12, 31)

CITIES    = ["Mumbai","Delhi","Bangalore","Hyderabad","Pune","Chennai","Kolkata","Ahmedabad","Jaipur","Lucknow"]
CUISINES  = ["North Indian","South Indian","Chinese","Italian","Biryani","Fast Food","Desserts","Continental","Thai","Mexican"]
AREAS     = ["Koramangala","Bandra","Connaught Place","Jubilee Hills","Kothrud","Anna Nagar","Salt Lake","Navrangpura","Malviya Nagar","Gomti Nagar"]
PRICE_CAT = ["Budget","Mid-Range","Premium"]
REST_TYPE = ["Quick Bites","Casual Dining","Fine Dining","Cloud Kitchen","Cafe","Dhaba"]

# ── Restaurants ───────────────────────────────────────────────────────────────
restaurants = []
for i in range(N_RESTAURANTS):
    city      = np.random.choice(CITIES)
    cuisine   = np.random.choice(CUISINES, size=np.random.randint(1,4), replace=False).tolist()
    price_cat = np.random.choice(PRICE_CAT, p=[0.40,0.40,0.20])
    rest_type = np.random.choice(REST_TYPE, p=[0.25,0.30,0.10,0.20,0.10,0.05])

    avg_cost  = {"Budget": np.random.randint(100,400),
                 "Mid-Range": np.random.randint(400,800),
                 "Premium": np.random.randint(800,2000)}[price_cat]

    # Rating influenced by price category and type
    base_rating = {"Budget":3.5,"Mid-Range":3.8,"Premium":4.1}[price_cat]
    rating = round(min(5.0, max(1.0, np.random.normal(base_rating, 0.4))), 1)

    votes        = int(np.random.exponential(500))
    delivery_time = np.random.randint(20, 70)
    is_veg       = np.random.random() < 0.35
    discount     = np.random.choice([0,10,20,30,40,50], p=[0.30,0.20,0.20,0.15,0.10,0.05])
    online_order = np.random.random() < 0.80
    listed_in    = np.random.choice(AREAS)

    restaurants.append({
        "restaurant_id":  f"R{str(i+1).zfill(5)}",
        "name":           f"Restaurant_{i+1}",
        "city":           city,
        "area":           listed_in,
        "cuisine":        ", ".join(cuisine),
        "price_category": price_cat,
        "rest_type":      rest_type,
        "avg_cost_two":   avg_cost,
        "rating":         rating,
        "votes":          votes,
        "avg_delivery_time": delivery_time,
        "is_veg":         is_veg,
        "discount_pct":   discount,
        "online_order":   online_order,
    })

rest_df = pd.DataFrame(restaurants)

# ── Orders ────────────────────────────────────────────────────────────────────
orders = []
for i in range(N_ORDERS):
    rest    = rest_df.sample(1).iloc[0]
    order_dt = START + timedelta(days=int(np.random.uniform(0,(END-START).days)))
    items    = np.random.randint(1, 6)
    order_val = int(rest["avg_cost_two"] / 2 * items * np.random.uniform(0.8, 1.2))
    disc_amt  = int(order_val * rest["discount_pct"] / 100)
    final_amt = order_val - disc_amt
    del_time  = int(rest["avg_delivery_time"] * np.random.uniform(0.8, 1.4))
    rated     = np.random.random() < 0.60
    order_rating = round(np.random.normal(rest["rating"], 0.3), 1) if rated else None

    orders.append({
        "order_id":       f"ORD{str(i+1).zfill(6)}",
        "restaurant_id":  rest["restaurant_id"],
        "city":           rest["city"],
        "cuisine":        rest["cuisine"].split(", ")[0],
        "order_date":     order_dt.strftime("%Y-%m-%d"),
        "items_ordered":  items,
        "order_value":    order_val,
        "discount_amt":   disc_amt,
        "final_amount":   final_amt,
        "delivery_time":  del_time,
        "order_rating":   order_rating,
        "price_category": rest["price_category"],
        "rest_type":      rest["rest_type"],
    })

order_df = pd.DataFrame(orders)

rest_df.to_csv("restaurants.csv", index=False)
order_df.to_csv("orders.csv", index=False)

print(f"Restaurants : {len(rest_df):,}")
print(f"Orders      : {len(order_df):,}")
print(f"Avg Rating  : {rest_df['rating'].mean():.2f}")
print(f"Total GMV   : ₹{order_df['final_amount'].sum():,.0f}")
print(f"Avg Delivery: {rest_df['avg_delivery_time'].mean():.0f} mins")
