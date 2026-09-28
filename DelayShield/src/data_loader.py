"""Loading layer for the raw Olist CSV files.

Every other module in DelayShield reads the raw data through this module so the
file locations and dtype/parse-date decisions live in exactly one place.
"""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

ORDER_DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def raw_path(key: str) -> str:
    return os.path.join(RAW_DIR, FILES[key])


def load_table(key: str) -> pd.DataFrame:
    """Load one raw Olist table by logical name."""
    path = raw_path(key)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing raw file {path}. Download the Olist dataset into data/raw/."
        )
    if key == "orders":
        return pd.read_csv(path, parse_dates=ORDER_DATE_COLS)
    if key == "order_items":
        return pd.read_csv(path, parse_dates=["shipping_limit_date"])
    if key == "reviews":
        return pd.read_csv(
            path, parse_dates=["review_creation_date", "review_answer_timestamp"]
        )
    return pd.read_csv(path)


@lru_cache(maxsize=None)
def load_all() -> dict:
    """Load every raw table once and cache it for the process lifetime."""
    return {key: load_table(key) for key in FILES}


def zip_geo_centroids() -> pd.DataFrame:
    """Average lat/lng per zip-code prefix.

    The raw geolocation table has ~1M rows with several points per prefix; the
    centroid is enough for a source-to-destination distance proxy.
    """
    geo = load_table("geolocation")
    centroids = (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "geolocation_zip_code_prefix": "zip_prefix",
                "geolocation_lat": "lat",
                "geolocation_lng": "lng",
            }
        )
    )
    return centroids


def dataset_profile() -> pd.DataFrame:
    """Row/column counts and missing-value share per raw table (used by the EDA report)."""
    rows = []
    for key in FILES:
        df = load_table(key)
        rows.append(
            {
                "table": key,
                "file": FILES[key],
                "rows": len(df),
                "columns": df.shape[1],
                "missing_cells_pct": round(
                    100 * df.isna().sum().sum() / (df.shape[0] * df.shape[1]), 3
                ),
            }
        )
    return pd.DataFrame(rows)
