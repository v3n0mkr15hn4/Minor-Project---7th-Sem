"""Supplier prioritization: turn order history + model risk into a scorecard.

The weights below are an **initial operational weighting**, not a validated
scientific formula. They are exposed in the dashboard so a reviewer can change
them and watch the ranking move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "reliability": 0.35,        # historical on-time rate
    "predicted_risk": 0.25,     # model-predicted delay risk (inverted)
    "delivery_speed": 0.15,     # average delivery days (inverted)
    "freight_performance": 0.10,  # freight cost as share of order value (inverted)
    "customer_feedback": 0.15,  # mean review score
}

MIN_ORDERS_FOR_RANKING = 10


def _normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalise to [0, 1]; constant columns collapse to 0.5."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - lo) / (hi - lo)
    return 1 - out if invert else out


def build_supplier_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seller history. ``scored`` must carry ``delay_probability``.

    Review scores are used here (a supplier scorecard is a *retrospective*
    view), never as a feature of the delay model.
    """
    df = scored.copy()
    df["freight_ratio_calc"] = df["total_freight_value"] / df["total_order_value"].replace(0, np.nan)

    grouped = df.groupby("primary_seller_id")
    metrics = grouped.agg(
        order_volume=("order_id", "count"),
        delay_rate=("delivery_delay", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_delay_days_when_late=("delay_days", lambda s: s[s > 0].mean()),
        avg_order_value=("total_order_value", "mean"),
        avg_freight_value=("total_freight_value", "mean"),
        freight_ratio=("freight_ratio_calc", "mean"),
        review_score=("review_score", "mean"),
        predicted_risk=("delay_probability", "mean"),
        seller_state=("seller_state", "first"),
        seller_city=("seller_city", "first"),
        last_order=("order_purchase_timestamp", "max"),
        top_category=("product_category", lambda s: s.mode().iat[0] if len(s.mode()) else "unknown"),
    ).reset_index()

    metrics["on_time_rate"] = 1 - metrics["delay_rate"]
    metrics["avg_delay_days_when_late"] = metrics["avg_delay_days_when_late"].fillna(0.0)
    metrics["review_score"] = metrics["review_score"].fillna(metrics["review_score"].mean())
    metrics["freight_ratio"] = metrics["freight_ratio"].fillna(metrics["freight_ratio"].median())
    metrics = metrics.rename(columns={"primary_seller_id": "seller_id"})
    return score_suppliers(metrics, DEFAULT_WEIGHTS)


def score_suppliers(metrics: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Apply the configurable weighted score and rank suppliers.

    Each component is min-max normalised so that 1 = best, then combined with
    the supplied weights (renormalised to sum to 1). Score is reported on 0-100.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(weights.values()) or 1.0
    w = {k: v / total for k, v in weights.items()}

    m = metrics.copy()
    comp = pd.DataFrame(index=m.index)
    comp["reliability"] = _normalize(m["on_time_rate"])
    comp["predicted_risk"] = _normalize(m["predicted_risk"], invert=True)
    comp["delivery_speed"] = _normalize(m["avg_delivery_days"], invert=True)
    comp["freight_performance"] = _normalize(m["freight_ratio"], invert=True)
    comp["customer_feedback"] = _normalize(m["review_score"])

    m["supplier_score"] = sum(comp[k] * w[k] for k in w) * 100
    for k in w:
        m["component_" + k] = (comp[k] * 100).round(1)

    m = m.sort_values("supplier_score", ascending=False).reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    m["priority_tier"] = pd.cut(
        m["supplier_score"],
        bins=[-0.01, 40, 60, 100.01],
        labels=["WATCHLIST", "STANDARD", "PREFERRED"],
    ).astype(str)
    # Small-volume sellers are ranked but flagged: their rates are noisy.
    m["low_volume_flag"] = (m["order_volume"] < MIN_ORDERS_FOR_RANKING).astype(int)
    return m


def high_risk_suppliers(metrics: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Worst suppliers by score, restricted to sellers with enough volume."""
    eligible = metrics[metrics["order_volume"] >= MIN_ORDERS_FOR_RANKING]
    return eligible.sort_values("supplier_score").head(top_n)
