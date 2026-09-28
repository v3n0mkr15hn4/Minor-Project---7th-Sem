"""Prediction engine: score a single order or a batch at confirmation time.

The dashboard form only asks for what an operator would actually know when an
order is confirmed (seller, category, value, freight, items, states, date).
Everything else - the seller's historical delay rate, the category and route
rates, the current workload - is looked up from the latest available history
snapshot, exactly as a production service would.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import data_loader as dl
from feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

MODEL_PATH = os.path.join(dl.MODEL_DIR, "delay_model.pkl")
META_PATH = os.path.join(dl.MODEL_DIR, "model_metadata.json")
DATASET_PATH = os.path.join(dl.PROCESSED_DIR, "model_dataset.csv")
SCORED_PATH = os.path.join(dl.PROCESSED_DIR, "scored_orders.csv")

RISK_ORDER = ["LOW", "MEDIUM", "HIGH"]


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python src/train.py")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    with open(META_PATH) as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH, parse_dates=["order_purchase_timestamp"])


@lru_cache(maxsize=1)
def load_scored() -> pd.DataFrame:
    return pd.read_csv(SCORED_PATH, parse_dates=["order_purchase_timestamp",
                                                 "order_estimated_delivery_date"])


@lru_cache(maxsize=1)
def history_snapshot() -> dict:
    """Latest known history values per seller / category / state / route.

    Uses the most recent row of each group in the engineered dataset, i.e. the
    freshest state of the running historical statistics.
    """
    df = load_dataset().sort_values("order_purchase_timestamp")
    seller_cols = [c for c in df.columns if c.startswith("seller_hist_")] + [
        "seller_orders_last_30_days",
        "seller_orders_last_90_days",
        "seller_workload_surge",
        "seller_state",
        "seller_city",
    ]
    sellers = df.groupby("primary_seller_id")[seller_cols].last()
    return {
        "sellers": sellers,
        "category": df.groupby("product_category")["category_hist_delay_rate"].last(),
        "customer_state": df.groupby("customer_state")["customer_state_hist_delay_rate"].last(),
        "route": df.groupby("route")["route_hist_delay_rate"].last(),
        "distance": df.groupby(["seller_state", "customer_state"])["distance_km"].median(),
        "platform_delay_rate": float(df["platform_hist_delay_rate"].iloc[-1]),
        "platform_load": float(df["platform_orders_last_30_days"].iloc[-1]),
        "medians": df[NUMERIC_FEATURES].median(numeric_only=True),
        "seller_ids": sorted(df["primary_seller_id"].unique().tolist()),
        "categories": sorted(df["product_category"].unique().tolist()),
        "customer_states": sorted(df["customer_state"].unique().tolist()),
        "seller_states": sorted(df["seller_state"].unique().tolist()),
    }


def build_feature_row(
    seller_id: str,
    product_category: str,
    total_order_value: float,
    total_freight_value: float,
    item_count: int,
    customer_state: str,
    seller_state: str,
    order_date: pd.Timestamp,
    estimated_lead_time_days: float,
    distance_km: float | None = None,
    product_weight_g: float | None = None,
) -> pd.DataFrame:
    """Assemble one model-ready row from operator inputs + history lookups."""
    snap = history_snapshot()
    med = snap["medians"]
    ts = pd.Timestamp(order_date)
    row = {c: (float(med[c]) if c in med.index else 0.0) for c in NUMERIC_FEATURES}

    # ---- history lookups ---------------------------------------------------
    sellers = snap["sellers"]
    if seller_id in sellers.index:
        s = sellers.loc[seller_id]
        for col in [
            "seller_hist_orders",
            "seller_hist_delay_rate",
            "seller_hist_on_time_rate",
            "seller_hist_avg_delivery_days",
            "seller_hist_avg_freight",
            "seller_hist_avg_delay_days",
            "seller_orders_last_30_days",
            "seller_orders_last_90_days",
            "seller_workload_surge",
        ]:
            if col in s.index:
                row[col] = float(s[col])
    else:  # unseen seller -> platform fallback (cold start)
        row["seller_hist_orders"] = 0.0
        row["seller_hist_delay_rate"] = snap["platform_delay_rate"]
        row["seller_hist_on_time_rate"] = 1 - snap["platform_delay_rate"]

    row["category_hist_delay_rate"] = float(
        snap["category"].get(product_category, snap["platform_delay_rate"])
    )
    row["customer_state_hist_delay_rate"] = float(
        snap["customer_state"].get(customer_state, snap["platform_delay_rate"])
    )
    route = f"{seller_state}->{customer_state}"
    row["route_hist_delay_rate"] = float(
        snap["route"].get(route, snap["platform_delay_rate"])
    )
    row["platform_hist_delay_rate"] = snap["platform_delay_rate"]
    row["platform_orders_last_30_days"] = snap["platform_load"]

    # ---- operator inputs ---------------------------------------------------
    row["order_purchase_month"] = ts.month
    row["order_purchase_week"] = int(ts.isocalendar().week)
    row["order_purchase_day"] = ts.day
    row["order_purchase_weekday"] = ts.weekday()
    row["order_purchase_hour"] = ts.hour
    row["order_purchase_quarter"] = ts.quarter
    row["is_weekend"] = int(ts.weekday() >= 5)
    row["estimated_lead_time_days"] = float(estimated_lead_time_days)
    row["shipping_limit_days"] = min(float(estimated_lead_time_days) * 0.25, 7.0)
    row["total_order_value"] = float(total_order_value)
    row["total_freight_value"] = float(total_freight_value)
    row["freight_ratio"] = float(total_freight_value) / max(float(total_order_value), 1e-6)
    row["item_count"] = int(item_count)
    row["unique_product_count"] = max(1, int(item_count) - (int(item_count) > 2))
    row["seller_count"] = 1
    row["max_item_price"] = float(total_order_value) / max(int(item_count), 1)
    if product_weight_g is not None:
        row["product_weight_g"] = float(product_weight_g)

    if distance_km is None:
        distance_km = snap["distance"].get(
            (seller_state, customer_state), float(med["distance_km"])
        )
    row["distance_km"] = float(distance_km)
    row["same_state"] = int(seller_state == customer_state)

    # ---- derived interactions ---------------------------------------------
    row["lead_time_slack_days"] = (
        row["estimated_lead_time_days"] - row["seller_hist_avg_delivery_days"]
    )
    row["lead_time_ratio"] = row["estimated_lead_time_days"] / max(
        row["seller_hist_avg_delivery_days"], 1e-6
    )
    row["distance_per_promised_day"] = row["distance_km"] / max(
        row["estimated_lead_time_days"], 1e-6
    )

    row["product_category"] = product_category
    row["customer_state"] = customer_state
    row["seller_state"] = seller_state
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_proba(features: pd.DataFrame) -> np.ndarray:
    """Delay probability for a feature frame."""
    model = load_model()
    return model.predict_proba(features[FEATURE_COLUMNS])[:, 1]


def risk_level(prob: float, thresholds: dict | None = None) -> str:
    """Map a probability to LOW / MEDIUM / HIGH using configurable cut-offs."""
    th = thresholds or load_metadata().get("risk_thresholds", {"medium": 0.3, "high": 0.6})
    if prob >= th["high"]:
        return "HIGH"
    if prob >= th["medium"]:
        return "MEDIUM"
    return "LOW"


def risk_levels(probs, thresholds: dict | None = None) -> pd.Series:
    th = thresholds or load_metadata().get("risk_thresholds", {"medium": 0.3, "high": 0.6})
    probs = pd.Series(np.asarray(probs, dtype=float))
    return pd.cut(
        probs,
        bins=[-0.001, th["medium"], th["high"], 1.001],
        labels=RISK_ORDER,
    ).astype(str)


def predict_order(**kwargs) -> dict:
    """End-to-end single-order prediction used by the dashboard form."""
    thresholds = kwargs.pop("thresholds", None)
    features = build_feature_row(**kwargs)
    prob = float(predict_proba(features)[0])
    return {
        "delay_probability": prob,
        "risk_level": risk_level(prob, thresholds),
        "features": features,
    }
