"""Time-aware feature engineering.

Two rules govern this module:

1. **No target leakage.** Every feature must be knowable at *order confirmation*
   (the purchase timestamp). Actual delivery dates, review scores and any
   post-delivery status are never inputs.
2. **No look-ahead in the history features.** A seller's "historical delay rate"
   for an order placed on 2018-03-01 is computed only from that seller's orders
   that were **already delivered before 2018-03-01**. An order placed two days
   earlier has no known outcome yet, so it does not contribute. This is stricter
   than a plain expanding mean over prior rows and it is what makes the
   historical features reproducible in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Windows (days) used for the recent-workload features.
WORKLOAD_WINDOWS = (30, 90)

CATEGORICAL_FEATURES = [
    "product_category",
    "customer_state",
    "seller_state",
]

NUMERIC_FEATURES = [
    # order timing (known at confirmation)
    "order_purchase_month",
    "order_purchase_week",
    "order_purchase_day",
    "order_purchase_weekday",
    "order_purchase_hour",
    "order_purchase_quarter",
    "is_weekend",
    "estimated_lead_time_days",
    "shipping_limit_days",
    "lead_time_slack_days",
    "lead_time_ratio",
    "distance_per_promised_day",
    # commercial
    "total_order_value",
    "total_freight_value",
    "freight_ratio",
    "item_count",
    "unique_product_count",
    "seller_count",
    "max_item_price",
    # product
    "product_weight_g",
    "product_volume_cm3",
    # location
    "distance_km",
    "same_state",
    "customer_zip_code_prefix",
    # lagged seller history
    "seller_hist_orders",
    "seller_hist_delay_rate",
    "seller_hist_on_time_rate",
    "seller_hist_avg_delivery_days",
    "seller_hist_avg_freight",
    "seller_hist_avg_delay_days",
    "seller_orders_last_30_days",
    "seller_orders_last_90_days",
    "seller_workload_surge",
    # lagged context history
    "category_hist_delay_rate",
    "customer_state_hist_delay_rate",
    "route_hist_delay_rate",
    "platform_orders_last_30_days",
    "platform_hist_delay_rate",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "delivery_delay"


def _lagged_outcome_stats(
    df: pd.DataFrame, key: str, prefix: str, value_cols: dict
) -> pd.DataFrame:
    """Outcome statistics of *already delivered* prior orders, per group.

    For every row we look at the rows of the same group whose
    ``order_delivered_customer_date`` is strictly before this row's
    ``order_purchase_timestamp`` and return the count plus the running mean of
    each requested value column.

    Parameters
    ----------
    key : grouping column (seller, category, state, ...).
    value_cols : mapping ``output_suffix -> source column`` to average.
    """
    purchase = df["order_purchase_timestamp"].values.astype("datetime64[ns]")
    delivered = df["order_delivered_customer_date"].values.astype("datetime64[ns]")

    n = len(df)
    counts = np.zeros(n)
    sums = {name: np.zeros(n) for name in value_cols}

    for _, idx in df.groupby(key, observed=True).indices.items():
        idx = np.asarray(idx)
        # order the group rows by when their outcome became known
        order = np.argsort(delivered[idx], kind="mergesort")
        known_at = delivered[idx][order]
        pos = np.searchsorted(known_at, purchase[idx], side="left")
        counts[idx] = pos
        for name, col in value_cols.items():
            vals = df[col].values[idx][order]
            cum = np.concatenate([[0.0], np.nancumsum(vals)])
            sums[name][idx] = cum[pos]

    out = pd.DataFrame(index=df.index)
    out[prefix + "_hist_orders"] = counts
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in value_cols:
            out[prefix + "_hist_" + name] = np.where(
                counts > 0, sums[name] / np.maximum(counts, 1), np.nan
            )
    return out


def _rolling_purchase_counts(df: pd.DataFrame, key, prefix: str) -> pd.DataFrame:
    """How many orders this group received in the last N days.

    Purchase events (not outcomes) are known immediately, so this is the honest
    "current workload" signal available at confirmation time.
    """
    purchase = df["order_purchase_timestamp"].values.astype("datetime64[ns]")
    out = pd.DataFrame(index=df.index)
    groups = df.groupby(key, observed=True).indices
    for window in WORKLOAD_WINDOWS:
        counts = np.zeros(len(df))
        delta = np.timedelta64(window, "D")
        for _, idx in groups.items():
            idx = np.asarray(idx)
            ts = purchase[idx]
            order = np.argsort(ts, kind="mergesort")
            sorted_ts = ts[order]
            hi = np.searchsorted(sorted_ts, ts, side="left")  # strictly earlier orders
            lo = np.searchsorted(sorted_ts, ts - delta, side="left")
            counts[idx] = hi - lo
        out[prefix + "_orders_last_" + str(window) + "_days"] = counts
    return out


def add_features(base: pd.DataFrame) -> pd.DataFrame:
    """Add every model feature to the cleaned order table."""
    df = base.sort_values("order_purchase_timestamp").reset_index(drop=True).copy()
    ts = df["order_purchase_timestamp"]

    # ---- calendar features -------------------------------------------------
    df["order_purchase_year"] = ts.dt.year
    df["order_purchase_month"] = ts.dt.month
    df["order_purchase_week"] = ts.dt.isocalendar().week.astype(int)
    df["order_purchase_day"] = ts.dt.day
    df["order_purchase_weekday"] = ts.dt.weekday
    df["order_purchase_hour"] = ts.dt.hour
    df["order_purchase_quarter"] = ts.dt.quarter
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)

    # ---- promise / commercial features ------------------------------------
    df["estimated_lead_time_days"] = (
        df["order_estimated_delivery_date"] - ts
    ).dt.total_seconds() / 86400
    df["shipping_limit_days"] = (
        df["shipping_limit_date"] - ts
    ).dt.total_seconds() / 86400
    df["freight_ratio"] = df["total_freight_value"] / df["total_order_value"].replace(
        0, np.nan
    )

    # ---- lagged history ----------------------------------------------------
    df["route"] = df["seller_state"].astype(str) + "->" + df["customer_state"].astype(str)

    seller_hist = _lagged_outcome_stats(
        df,
        "primary_seller_id",
        "seller",
        {
            "delay_rate": "delivery_delay",
            "avg_delivery_days": "delivery_days",
            "avg_freight": "total_freight_value",
            "avg_delay_days": "delay_days",
        },
    )
    df = df.join(seller_hist)

    for key, prefix in [
        ("product_category", "category"),
        ("customer_state", "customer_state"),
        ("route", "route"),
    ]:
        stats = _lagged_outcome_stats(df, key, prefix, {"delay_rate": "delivery_delay"})
        df[prefix + "_hist_delay_rate"] = stats[prefix + "_hist_delay_rate"]

    df["__all__"] = 1
    platform = _lagged_outcome_stats(
        df, "__all__", "platform", {"delay_rate": "delivery_delay"}
    )
    df["platform_hist_delay_rate"] = platform["platform_hist_delay_rate"]

    df = df.join(_rolling_purchase_counts(df, "primary_seller_id", "seller"))
    df = df.join(_rolling_purchase_counts(df, "__all__", "platform"))
    df = df.drop(columns="__all__")

    # ---- cold-start fallbacks ---------------------------------------------
    # New sellers / new categories have no delivered history yet. Falling back
    # to the *running platform average available at that time* keeps the value
    # leakage-free (a global mean over the whole dataset would not).
    global_seed = df["delivery_delay"].iloc[: max(200, int(0.01 * len(df)))].mean()
    running_global = df["platform_hist_delay_rate"].fillna(global_seed)
    df["platform_hist_delay_rate"] = running_global

    for col in [
        "seller_hist_delay_rate",
        "category_hist_delay_rate",
        "customer_state_hist_delay_rate",
        "route_hist_delay_rate",
    ]:
        df[col] = df[col].fillna(running_global)
    df["seller_hist_on_time_rate"] = 1 - df["seller_hist_delay_rate"]

    running_days = df["delivery_days"].expanding().mean().shift(1)
    df["seller_hist_avg_delivery_days"] = (
        df["seller_hist_avg_delivery_days"]
        .fillna(running_days)
        .fillna(df["delivery_days"].iloc[:200].mean())
    )
    df["seller_hist_avg_freight"] = (
        df["seller_hist_avg_freight"]
        .fillna(df["total_freight_value"].expanding().mean().shift(1))
        .fillna(df["total_freight_value"].iloc[:200].mean())
    )
    df["seller_hist_avg_delay_days"] = df["seller_hist_avg_delay_days"].fillna(0.0)
    df["freight_ratio"] = df["freight_ratio"].fillna(df["freight_ratio"].median())
    df["shipping_limit_days"] = df["shipping_limit_days"].fillna(
        df["shipping_limit_days"].median()
    )
    # ---- interaction features (all confirmation-time information) ---------
    # How much slack the promised date leaves versus how long this seller has
    # historically taken. Negative slack = the promise is already optimistic.
    df["lead_time_slack_days"] = (
        df["estimated_lead_time_days"] - df["seller_hist_avg_delivery_days"]
    )
    df["lead_time_ratio"] = df["estimated_lead_time_days"] / df[
        "seller_hist_avg_delivery_days"
    ].replace(0, np.nan)
    df["lead_time_ratio"] = df["lead_time_ratio"].fillna(1.0)
    df["distance_per_promised_day"] = df["distance_km"] / df[
        "estimated_lead_time_days"
    ].replace(0, np.nan)
    df["distance_per_promised_day"] = df["distance_per_promised_day"].fillna(0.0)
    # Recent load versus the seller's own 90-day baseline: a surge above 1 means
    # the seller is busier than usual right now.
    baseline = (df["seller_orders_last_90_days"] / 3).replace(0, np.nan)
    df["seller_workload_surge"] = (
        df["seller_orders_last_30_days"] / baseline
    ).fillna(1.0)

    df["is_cold_start_seller"] = (df["seller_hist_orders"] == 0).astype(int)
    return df


def build_model_dataset(base: pd.DataFrame) -> pd.DataFrame:
    """Feature table + the few identifier/audit columns the dashboard needs."""
    df = add_features(base)
    keep = (
        [
            "order_id",
            "customer_id",
            "primary_seller_id",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_purchase_year",
            "customer_city",
            "seller_city",
            "route",
            "is_cold_start_seller",
        ]
        + FEATURE_COLUMNS
        + ["delivery_delay", "delivery_days", "delay_days", "review_score"]
    )
    return df[keep]
