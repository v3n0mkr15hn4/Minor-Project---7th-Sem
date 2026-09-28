"""Policy simulation: reactive firefighting vs DelayShield risk-aware action.

Both policies are evaluated on the *same* set of held-out orders with the same
cost assumptions. Every monetary input is an explicit ASSUMPTION - the point of
the simulation is the comparison and its sensitivity, not the absolute number.

Policy definitions
------------------
**Reactive** - nothing happens at confirmation. The team only reacts once the
order is visibly late, so an intervention is run on every order that ends up
late and only a small share of the damage is recovered
(``reactive_effectiveness``).

**DelayShield (risk-aware)** - every order whose predicted delay probability is
at or above ``threshold`` gets a proactive intervention at confirmation time.
Correct flags recover ``proactive_effectiveness`` of the damage; wrong flags
cost a false alert plus the intervention. Orders that slip through undetected
still get the reactive treatment afterwards, so the comparison is fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ASSUMPTIONS = {
    "cost_of_late_delivery": 1200.0,   # ASSUMPTION
    "cost_of_false_alert": 30.0,       # ASSUMPTION
    "cost_of_intervention": 50.0,      # ASSUMPTION
    "proactive_effectiveness": 0.60,   # ASSUMPTION
    "reactive_effectiveness": 0.15,    # ASSUMPTION
    "threshold": 0.15,
}


def break_even_precision(assumptions: dict | None = None) -> float:
    """Precision the alerts must reach for the risk-aware policy to pay off.

    Each correct early flag saves ``(proactive_eff - reactive_eff) * late_cost``
    and each wrong flag costs ``false_alert + intervention``. Setting the two
    equal gives the minimum precision the alert stream needs.
    """
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    gain = (a["proactive_effectiveness"] - a["reactive_effectiveness"]) * a[
        "cost_of_late_delivery"
    ]
    loss = a["cost_of_false_alert"] + a["cost_of_intervention"]
    return float(loss / (gain + loss)) if gain + loss > 0 else 1.0


def _reactive(y_true: np.ndarray, a: dict) -> dict:
    late = int(y_true.sum())
    residual_late = late * (1 - a["reactive_effectiveness"])
    late_cost = residual_late * a["cost_of_late_delivery"]
    intervention_cost = late * a["cost_of_intervention"]
    return {
        "policy": "Reactive (act after the delay is visible)",
        "orders": int(len(y_true)),
        "actual_late": late,
        "flagged_before_delivery": 0,
        "true_alerts": 0,
        "false_alerts": 0,
        "missed_delays": late,
        "interventions": late,
        "delays_mitigated": late * a["reactive_effectiveness"],
        "residual_late_orders": residual_late,
        "late_cost": late_cost,
        "false_alert_cost": 0.0,
        "intervention_cost": intervention_cost,
        "total_cost": late_cost + intervention_cost,
    }


def _risk_aware(y_true: np.ndarray, y_prob: np.ndarray, a: dict) -> dict:
    flag = y_prob >= a["threshold"]
    tp = int((flag & (y_true == 1)).sum())
    fp = int((flag & (y_true == 0)).sum())
    fn = int((~flag & (y_true == 1)).sum())

    # Flagged and truly late -> proactive recovery; missed -> reactive recovery.
    mitigated = tp * a["proactive_effectiveness"] + fn * a["reactive_effectiveness"]
    residual_late = int(y_true.sum()) - mitigated
    late_cost = residual_late * a["cost_of_late_delivery"]
    false_alert_cost = fp * a["cost_of_false_alert"]
    interventions = tp + fp + fn  # proactive flags + reactive clean-up of misses
    intervention_cost = interventions * a["cost_of_intervention"]
    return {
        "policy": "DelayShield risk-aware (act at order confirmation)",
        "orders": int(len(y_true)),
        "actual_late": int(y_true.sum()),
        "flagged_before_delivery": tp + fp,
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_delays": fn,
        "interventions": interventions,
        "delays_mitigated": mitigated,
        "residual_late_orders": residual_late,
        "late_cost": late_cost,
        "false_alert_cost": false_alert_cost,
        "intervention_cost": intervention_cost,
        "total_cost": late_cost + false_alert_cost + intervention_cost,
    }


def simulate(y_true, y_prob, assumptions: dict | None = None) -> pd.DataFrame:
    """Run both policies on the same orders and return one row per policy."""
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = [_reactive(y_true, a), _risk_aware(y_true, y_prob, a)]
    df = pd.DataFrame(rows)
    df["on_time_rate_after_policy"] = 1 - df["residual_late_orders"] / df["orders"]
    return df


def compare(y_true, y_prob, assumptions: dict | None = None) -> dict:
    """Headline deltas between the two policies (DelayShield minus reactive)."""
    res = simulate(y_true, y_prob, assumptions)
    reactive, proactive = res.iloc[0], res.iloc[1]
    saving = reactive["total_cost"] - proactive["total_cost"]
    return {
        "table": res,
        "cost_saving": float(saving),
        "cost_saving_pct": float(
            100 * saving / reactive["total_cost"] if reactive["total_cost"] else 0.0
        ),
        "delays_caught_early": int(proactive["true_alerts"]),
        "false_alerts": int(proactive["false_alerts"]),
        "extra_interventions": float(
            proactive["interventions"] - reactive["interventions"]
        ),
        "service_gain_pp": float(
            100
            * (
                proactive["on_time_rate_after_policy"]
                - reactive["on_time_rate_after_policy"]
            )
        ),
    }


def threshold_sweep(y_true, y_prob, assumptions: dict | None = None,
                    grid: np.ndarray | None = None) -> pd.DataFrame:
    """Total cost / alerts / recall across decision thresholds."""
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    reactive_cost = _reactive(y_true, a)["total_cost"]
    rows = []
    for t in grid:
        r = _risk_aware(y_true, y_prob, {**a, "threshold": float(t)})
        rows.append(
            {
                "threshold": round(float(t), 3),
                "total_cost": r["total_cost"],
                "alerts": r["flagged_before_delivery"],
                "true_alerts": r["true_alerts"],
                "false_alerts": r["false_alerts"],
                "recall": r["true_alerts"] / max(int(y_true.sum()), 1),
                "precision": r["true_alerts"] / max(r["flagged_before_delivery"], 1),
                "saving_vs_reactive": reactive_cost - r["total_cost"],
            }
        )
    return pd.DataFrame(rows)


def sensitivity_analysis(y_true, y_prob, assumptions: dict | None = None,
                         param: str = "cost_of_late_delivery",
                         values: list | None = None) -> pd.DataFrame:
    """How the saving moves when one assumption is varied."""
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    if values is None:
        base = a[param]
        values = [base * m for m in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
    rows = []
    for v in values:
        res = compare(y_true, y_prob, {**a, param: v})
        rows.append(
            {
                param: v,
                "reactive_cost": float(res["table"].iloc[0]["total_cost"]),
                "delayshield_cost": float(res["table"].iloc[1]["total_cost"]),
                "saving": res["cost_saving"],
                "saving_pct": res["cost_saving_pct"],
            }
        )
    return pd.DataFrame(rows)
