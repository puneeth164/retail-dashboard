"""
DSA508 Test 2 - Problem 2
Retail Sales Dashboard (Streamlit) connected live to Azure Cosmos DB for MongoDB.

Run locally with:  streamlit run app.py
Connection string is read from AZURE_URI in the .env file (local) or in Streamlit Secrets (cloud).
"""
import json
import os
from datetime import datetime, time

import time as _time

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure

# ------------------------------------------------------------------ setup
st.set_page_config(page_title="Retail Sales Dashboard", page_icon="🛒", layout="wide")
load_dotenv()


@st.cache_resource
def get_collection():
    uri = os.getenv("AZURE_URI")          # local run: from the .env file
    if not uri:
        uri = st.secrets["AZURE_URI"]     # Streamlit Cloud: from the app's Secrets setting
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    return client["retaildb"]["sales"]


col = get_collection()


def retry(fn, tries=6):
    """Azure free tier is limited to 1000 RU/s: if a query is throttled, wait and retry."""
    for t in range(tries):
        try:
            return fn()
        except OperationFailure:
            if t == tries - 1:
                raise
            _time.sleep(1.5)


@st.cache_data(ttl=600)
def run(pipeline_key, pipeline):
    """Run an aggregation pipeline on Azure and return a DataFrame (cached for 10 min)."""
    return pd.DataFrame(retry(lambda: list(col.aggregate(pipeline))))


@st.cache_data(ttl=600)
def filter_options():
    stores = sorted(retry(lambda: col.distinct("storeLocation")))
    methods = sorted(retry(lambda: col.distinct("purchaseMethod")))
    # Cosmos DB returns $min/$max of dates as raw numbers, so read the dates and compare in Python
    dates = pd.to_datetime([d["saleDate"] for d in retry(lambda: list(col.find({}, {"saleDate": 1, "_id": 0})))])
    return stores, methods, dates.min().date(), dates.max().date()


# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
stores, methods, dmin, dmax = filter_options()
sel_stores = st.sidebar.multiselect("Store location", stores, default=stores)
sel_methods = st.sidebar.multiselect("Purchase method", methods, default=methods)
date_range = st.sidebar.date_input("Sale date range", (dmin, dmax), min_value=dmin, max_value=dmax)
coupon_opt = st.sidebar.radio("Coupon", ["All", "Used", "Not used"], horizontal=True)

start, end = (date_range if isinstance(date_range, (list, tuple)) and len(date_range) == 2
              else (dmin, dmax))
match = {
    "storeLocation": {"$in": sel_stores or stores},
    "purchaseMethod": {"$in": sel_methods or methods},
    "saleDate": {"$gte": datetime.combine(start, time.min), "$lte": datetime.combine(end, time.max)},
}
if coupon_opt != "All":
    match["couponUsed"] = coupon_opt == "Used"
key = str(match)  # cache key for the current filter selection

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Azure Cosmos DB for MongoDB · database `retaildb` · collection `sales`")

# ------------------------------------------------------------------ header + KPIs
st.title("🛒 Retail Sales Dashboard")
st.caption("Live metrics from a document database on Azure Cosmos DB for MongoDB")

kpi = run("kpi" + key, [
    {"$match": match},
    {"$group": {"_id": None,
                "revenue": {"$sum": "$totalAmount"},
                "orders": {"$sum": 1},
                "avg_order": {"$avg": "$totalAmount"},
                "satisfaction": {"$avg": "$customer.satisfaction"},
                "coupons": {"$sum": {"$cond": ["$couponUsed", 1, 0]}}}}])

if kpi.empty:
    st.warning("No sales match the selected filters.")
    st.stop()
k = kpi.iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total revenue", f"${k['revenue']:,.0f}")
c2.metric("Orders", f"{int(k['orders']):,}")
c3.metric("Avg order value", f"${k['avg_order']:,.2f}")
c4.metric("Avg satisfaction", f"{k['satisfaction']:.2f} / 5")
c5.metric("Coupon usage", f"{100 * k['coupons'] / k['orders']:.1f}%")

st.markdown("---")

# ------------------------------------------------------------------ row 1: trend + stores
left, right = st.columns(2)

@st.cache_data(ttl=600)
def monthly_trend(cache_key, match):
    """Monthly revenue: fetch date + amount for the filtered sales, then group by month in pandas."""
    rows = retry(lambda: list(col.find(match, {"saleDate": 1, "totalAmount": 1, "_id": 0})))
    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["saleDate"]).dt.to_period("M").dt.to_timestamp()
    return df.groupby("month").agg(revenue=("totalAmount", "sum"),
                                   orders=("totalAmount", "size")).reset_index()


monthly = monthly_trend(key, match)
monthly = monthly.sort_values("month")
left.subheader("Monthly revenue trend")
left.plotly_chart(px.line(monthly, x="month", y="revenue", markers=True,
                          labels={"month": "", "revenue": "Revenue ($)"}), use_container_width=True)

by_store = run("store" + key, [
    {"$match": match},
    {"$group": {"_id": "$storeLocation", "revenue": {"$sum": "$totalAmount"},
                "orders": {"$sum": 1}, "satisfaction": {"$avg": "$customer.satisfaction"}}},
    {"$sort": {"revenue": -1}}]).rename(columns={"_id": "store"})
right.subheader("Revenue by store")
right.plotly_chart(px.bar(by_store, x="store", y="revenue", color="satisfaction",
                          color_continuous_scale="RdYlGn", range_color=(3, 4.5),
                          labels={"store": "", "revenue": "Revenue ($)", "satisfaction": "Avg satisfaction"}),
                   use_container_width=True)

# ------------------------------------------------------------------ row 2: products + categories
left, right = st.columns(2)

products = run("products" + key, [
    {"$match": match},
    {"$unwind": "$items"},
    {"$group": {"_id": "$items.name",
                "revenue": {"$sum": {"$multiply": ["$items.price", "$items.quantity"]}},
                "units": {"$sum": "$items.quantity"}}},
    {"$sort": {"revenue": -1}}, {"$limit": 10}]).rename(columns={"_id": "product"})
left.subheader("Top 10 products by revenue")
left.plotly_chart(px.bar(products.sort_values("revenue"), x="revenue", y="product", orientation="h",
                         hover_data=["units"], labels={"product": "", "revenue": "Revenue ($)"}),
                  use_container_width=True)

cats = run("cats" + key, [
    {"$match": match},
    {"$unwind": "$items"},
    {"$group": {"_id": "$items.category",
                "revenue": {"$sum": {"$multiply": ["$items.price", "$items.quantity"]}}}}]
           ).rename(columns={"_id": "category"})
right.subheader("Revenue share by product category")
right.plotly_chart(px.pie(cats, names="category", values="revenue", hole=0.45), use_container_width=True)

# ------------------------------------------------------------------ row 3: channels + coupons
left, right = st.columns(2)

channel = run("channel" + key, [
    {"$match": match},
    {"$group": {"_id": "$purchaseMethod", "orders": {"$sum": 1},
                "avg_order": {"$avg": "$totalAmount"}}}]).rename(columns={"_id": "method"})
left.subheader("Orders by purchase method")
left.plotly_chart(px.bar(channel, x="method", y="orders", text="orders",
                         hover_data={"avg_order": ":.2f"}, labels={"method": "", "orders": "Orders"}),
                  use_container_width=True)

coupon = run("coupon" + key, [
    {"$match": match},
    {"$group": {"_id": "$couponUsed", "avg_order": {"$avg": "$totalAmount"},
                "satisfaction": {"$avg": "$customer.satisfaction"}, "orders": {"$sum": 1}}}])
coupon["coupon"] = coupon["_id"].map({True: "Coupon used", False: "No coupon"})
right.subheader("Effect of coupons")
right.plotly_chart(px.bar(coupon.melt(id_vars="coupon", value_vars=["avg_order", "satisfaction"]),
                          x="coupon", y="value", color="coupon", facet_col="variable",
                          labels={"coupon": "", "value": ""}).update_yaxes(matches=None, showticklabels=True)
                   .for_each_annotation(lambda a: a.update(
                       text={"variable=avg_order": "Avg order ($)",
                             "variable=satisfaction": "Avg satisfaction"}[a.text]))
                   .update_layout(showlegend=False),
                   use_container_width=True)

# ------------------------------------------------------------------ row 4: customers
left, right = st.columns(2)

ages = run("ages" + key, [
    {"$match": match},
    {"$group": {"_id": "$customer.age", "orders": {"$sum": 1}, "revenue": {"$sum": "$totalAmount"}}}]
           ).rename(columns={"_id": "age"})
ages["age_group"] = pd.cut(ages["age"], [17, 25, 35, 45, 55, 65, 100],
                           labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
age_groups = ages.groupby("age_group", observed=False)[["orders", "revenue"]].sum().reset_index()
left.subheader("Customers by age group")
left.plotly_chart(px.bar(age_groups, x="age_group", y="revenue", hover_data=["orders"],
                         labels={"age_group": "Age group", "revenue": "Revenue ($)"}),
                  use_container_width=True)

sat = run("sat" + key, [
    {"$match": match},
    {"$group": {"_id": {"store": "$storeLocation", "score": "$customer.satisfaction"}, "n": {"$sum": 1}}}])
sat["store"] = sat["_id"].str["store"]
sat["score"] = sat["_id"].str["score"].astype(int).astype(str)
right.subheader("Satisfaction scores by store")
right.plotly_chart(px.bar(sat.sort_values("score"), x="store", y="n", color="score",
                          color_discrete_sequence=px.colors.diverging.RdYlGn,
                          labels={"store": "", "n": "Orders", "score": "Score (1-5)"}),
                   use_container_width=True)

# ------------------------------------------------------------------ raw documents
with st.expander("View sample documents from Azure"):
    sample = retry(lambda: list(col.find(match, {"_id": 0}).limit(5)))
    st.json(json.loads(json.dumps(sample, default=str)), expanded=False)
