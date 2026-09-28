"""Action recommendation engine.

Rule-based on purpose: the recommendation layer must be auditable by operations
staff, so it maps (risk level, supplier standing, order context) to the concrete
actions listed in the proposal rather than learning them.
"""
from __future__ import annotations

import pandas as pd

PLAYBOOK = {
    "LOW": {
        "headline": "Normal processing",
        "actions": [
            "Process the order through the standard flow",
            "No customer pre-notification required",
            "Standard tracking cadence",
        ],
        "sla_buffer_days": 0,
    },
    "MEDIUM": {
        "headline": "Monitor and buffer",
        "actions": [
            "Increase shipment tracking frequency",
            "Add a moderate schedule buffer to the promised date",
            "Flag the supplier for weekly performance monitoring",
        ],
        "sla_buffer_days": 2,
    },
    "HIGH": {
        "headline": "Proactive intervention",
        "actions": [
            "Notify the customer proactively with a realistic delivery window",
            "Review an alternate supplier for the same category",
            "Extend the schedule buffer and re-plan the carrier pickup",
            "Escalate shipment tracking to daily checks",
        ],
        "sla_buffer_days": 5,
    },
}


def recommend(
    risk_level: str,
    delay_probability: float | None = None,
    supplier_row: pd.Series | dict | None = None,
    order_value: float | None = None,
) -> dict:
    """Return the playbook for a risk level, extended by supplier/order context."""
    level = str(risk_level).upper()
    base = PLAYBOOK.get(level, PLAYBOOK["LOW"])
    actions = list(base["actions"])
    notes = []

    if supplier_row is not None:
        get = supplier_row.get if hasattr(supplier_row, "get") else lambda k, d=None: None
        delay_rate = get("delay_rate", None)
        score = get("supplier_score", None)
        volume = get("order_volume", None)
        if delay_rate is not None and delay_rate == delay_rate and delay_rate > 0.15:
            actions.append(
                f"Supplier delay rate is {delay_rate:.1%} - open a corrective-action review"
            )
        if score is not None and score == score and score < 40:
            actions.append("Supplier is on the WATCHLIST tier - restrict new high-value orders")
        if volume is not None and volume == volume and volume < 10:
            notes.append("Supplier history is thin (<10 delivered orders); treat rates as noisy")

    if order_value is not None and order_value > 1000 and level in ("MEDIUM", "HIGH"):
        actions.append("High-value order - assign to priority handling queue")

    if delay_probability is not None and level == "HIGH" and delay_probability >= 0.8:
        actions.insert(0, "Very high risk - consider split shipment or express carrier upgrade")

    return {
        "risk_level": level,
        "headline": base["headline"],
        "actions": actions,
        "sla_buffer_days": base["sla_buffer_days"],
        "notes": notes,
    }


def recommend_batch(df: pd.DataFrame, risk_col: str = "risk_level") -> pd.Series:
    """Short one-line recommendation for a table of orders."""
    return df[risk_col].map(lambda r: PLAYBOOK.get(str(r).upper(), PLAYBOOK["LOW"])["headline"])
