"""Cleaning and integration: raw Olist tables -> one row per delivered order.

The output of :func:`build_order_base` is the *pre-feature* table. It still
carries a few post-delivery columns (actual delivery date, review score) because
they are needed to build the target and the historical supplier statistics, but
those columns are explicitly listed in ``POST_OUTCOME_COLS`` and are dropped
before anything reaches the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_table, zip_geo_centroids

# Columns that only exist after the order is delivered / reviewed. They are used
# to build labels and *lagged* history, never as model inputs.
POST_OUTCOME_COLS = [
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_approved_at",
    "delivery_days",
    "delay_days",
    "review_score",
    "delivery_delay",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorised over numpy arrays."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _order_item_aggregates() -> pd.DataFrame:
    """Collapse the item table to one row per order.

    Multi-seller orders are real in Olist, so instead of assuming 1 order = 1
    seller we keep ``seller_count`` and pick a *primary seller*: the seller
    holding the largest share of the order value (ties broken by first item id).
    """
    items = load_table("order_items")
    products = load_table("products")
    trans = load_table("category_translation")

    products = products.merge(trans, on="product_category_name", how="left")
    products["product_category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )
    items = items.merge(
        products[
            [
                "product_id",
                "product_category",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
        ],
        on="product_id",
        how="left",
    )

    agg = items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        unique_product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        total_order_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        max_item_price=("price", "max"),
        product_weight_g=("product_weight_g", "sum"),
        product_length_cm=("product_length_cm", "max"),
        product_height_cm=("product_height_cm", "max"),
        product_width_cm=("product_width_cm", "max"),
        shipping_limit_date=("shipping_limit_date", "max"),
    )

    # Primary seller / primary category = the line with the highest price.
    ranked = items.sort_values(
        ["order_id", "price", "order_item_id"], ascending=[True, False, True]
    )
    primary = ranked.groupby("order_id").first()[
        ["seller_id", "product_id", "product_category"]
    ]
    primary.columns = ["primary_seller_id", "primary_product_id", "product_category"]

    out = agg.join(primary).reset_index()
    out["product_category"] = out["product_category"].fillna("unknown")
    return out


def build_order_base() -> pd.DataFrame:
    """Delivered orders joined with items, sellers, customers, geo and reviews."""
    orders = load_table("orders")
    customers = load_table("customers")
    sellers = load_table("sellers")
    reviews = load_table("reviews")

    delivered = orders[orders["order_status"] == "delivered"].copy()
    required = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    delivered = delivered.dropna(subset=required)

    # ---- target -----------------------------------------------------------
    delivered["delivery_delay"] = (
        delivered["order_delivered_customer_date"]
        > delivered["order_estimated_delivery_date"]
    ).astype(int)
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivered["delay_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    # Drop physically impossible rows (delivery before purchase).
    delivered = delivered[delivered["delivery_days"] > 0]

    df = delivered.merge(_order_item_aggregates(), on="order_id", how="inner")
    df = df.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_city",
                "customer_state",
                "customer_zip_code_prefix",
            ]
        ],
        on="customer_id",
        how="left",
    )
    df = df.merge(
        sellers.rename(columns={"seller_id": "primary_seller_id"}),
        on="primary_seller_id",
        how="left",
    )

    # Review score: kept for the supplier scorecard only (post-outcome column).
    rev = reviews.groupby("order_id")["review_score"].mean().reset_index()
    df = df.merge(rev, on="order_id", how="left")

    # ---- geo distance proxy ----------------------------------------------
    geo = zip_geo_centroids()
    df = df.merge(
        geo.rename(columns={"zip_prefix": "customer_zip_code_prefix",
                            "lat": "cust_lat", "lng": "cust_lng"}),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        geo.rename(columns={"zip_prefix": "seller_zip_code_prefix",
                            "lat": "sell_lat", "lng": "sell_lng"}),
        on="seller_zip_code_prefix",
        how="left",
    )
    df["distance_km"] = haversine_km(
        df["sell_lat"], df["sell_lng"], df["cust_lat"], df["cust_lng"]
    )
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # ---- cleaning ---------------------------------------------------------
    for col in ["product_weight_g", "product_length_cm", "product_height_cm",
                "product_width_cm"]:
        df[col] = df[col].fillna(df[col].median())
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
    df["seller_state"] = df["seller_state"].fillna("unknown")
    df["seller_city"] = df["seller_city"].fillna("unknown")

    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    return df
