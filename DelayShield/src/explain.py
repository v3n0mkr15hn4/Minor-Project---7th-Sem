"""SHAP explanations for DelayShield predictions.

The persisted artefact is an sklearn ``Pipeline`` (preprocessor + classifier),
so explanations are computed on the *transformed* matrix and mapped back to
readable feature names. Tree models use ``TreeExplainer`` or ``Explainer``; a linear final model
falls back to ``LinearExplainer``.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd
import shap

from feature_engineering import FEATURE_COLUMNS
from predict import load_dataset, load_model

BACKGROUND_SIZE = 200

# Human-readable labels for the dashboard.
FEATURE_LABELS = {
    "estimated_lead_time_days": "Promised lead time (days)",
    "lead_time_slack_days": "Slack vs seller's usual delivery time",
    "lead_time_ratio": "Promised lead time / seller's usual time",
    "shipping_limit_days": "Days until shipping limit",
    "distance_km": "Seller-customer distance (km)",
    "distance_per_promised_day": "Distance per promised day",
    "same_state": "Seller and customer in same state",
    "total_order_value": "Order value",
    "total_freight_value": "Freight value",
    "freight_ratio": "Freight as share of order value",
    "item_count": "Number of items",
    "unique_product_count": "Distinct products",
    "seller_count": "Sellers in the order",
    "max_item_price": "Most expensive item",
    "product_weight_g": "Product weight (g)",
    "product_volume_cm3": "Product volume (cm3)",
    "seller_hist_orders": "Seller's delivered order history",
    "seller_hist_delay_rate": "Seller historical delay rate",
    "seller_hist_on_time_rate": "Seller historical on-time rate",
    "seller_hist_avg_delivery_days": "Seller average delivery days",
    "seller_hist_avg_delay_days": "Seller average lateness when late",
    "seller_hist_avg_freight": "Seller average freight",
    "seller_orders_last_30_days": "Seller workload (last 30 days)",
    "seller_orders_last_90_days": "Seller workload (last 90 days)",
    "seller_workload_surge": "Seller workload surge vs baseline",
    "category_hist_delay_rate": "Category historical delay rate",
    "customer_state_hist_delay_rate": "Destination state delay rate",
    "route_hist_delay_rate": "Route historical delay rate",
    "platform_hist_delay_rate": "Platform-wide delay rate",
    "platform_orders_last_30_days": "Platform workload (last 30 days)",
    "order_purchase_month": "Purchase month",
    "order_purchase_week": "Purchase week of year",
    "order_purchase_day": "Purchase day of month",
    "order_purchase_weekday": "Purchase weekday",
    "order_purchase_hour": "Purchase hour",
    "order_purchase_quarter": "Purchase quarter",
    "is_weekend": "Weekend purchase",
    "customer_zip_code_prefix": "Customer zip prefix",
}


def pretty_name(raw: str) -> str:
    """Map a transformed column name back to something a reviewer can read."""
    name = raw.split("__", 1)[-1]
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix, label in [
        ("product_category_", "Category"),
        ("customer_state_", "Customer state"),
        ("seller_state_", "Seller state"),
    ]:
        if name.startswith(prefix):
            return f"{label} = {name[len(prefix):]}"
    return name.replace("_", " ").capitalize()


@lru_cache(maxsize=1)
def _explainer():
    """Build the SHAP explainer once, using a background sample of real orders."""
    model = load_model()
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    data = load_dataset()

    sample_size = min(BACKGROUND_SIZE, len(data))
    background_raw = data[FEATURE_COLUMNS].sample(sample_size, random_state=42)
    background = prep.transform(background_raw)

    if hasattr(background, "toarray"):
        background = background.toarray()

    names = [pretty_name(c) for c in prep.get_feature_names_out()]

    try:
        explainer = shap.TreeExplainer(clf, data=background)
        kind = "tree"
    except Exception:
        try:
            explainer = shap.LinearExplainer(clf, background)
            kind = "linear"
        except Exception:
            explainer = shap.Explainer(clf, background)
            kind = "general"

    return explainer, kind, prep, names


def _shap_matrix(explainer, X) -> np.ndarray:
    """Extract and normalise SHAP values to a 2-D array (rows x features) for the positive class."""
    shap_output = explainer(X) if callable(explainer) else explainer.shap_values(X)

    if hasattr(shap_output, "values"):
        vals = shap_output.values
    else:
        vals = shap_output

    arr = np.asarray(vals)

    # Handle 3D array outputs: (samples, features, classes) or (classes, samples, features)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1]

    return arr


def explain_features(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature SHAP contributions for one order (positive = pushes to late)."""
    explainer, _kind, prep, names = _explainer()
    X = prep.transform(features[FEATURE_COLUMNS])

    if hasattr(X, "toarray"):
        X = X.toarray()

    contrib = _shap_matrix(explainer, X)[0]

    raw_values = np.asarray(X)[0]

    out = pd.DataFrame(
        {
            "feature": names,
            "value": raw_values,
            "shap_value": contrib,
        }
    )
    out["abs_shap"] = out["shap_value"].abs()
    out["direction"] = np.where(out["shap_value"] >= 0, "increases risk", "reduces risk")
    return out.sort_values("abs_shap", ascending=False).reset_index(drop=True)


def top_factors(features: pd.DataFrame, top_n: int = 5) -> dict:
    """Top risk-increasing and risk-reducing factors for one prediction."""
    contrib = explain_features(features)
    increasing = contrib[contrib["shap_value"] > 0].head(top_n)
    reducing = contrib[contrib["shap_value"] < 0].head(top_n)
    return {"all": contrib, "increasing": increasing, "reducing": reducing}


def global_importance(sample_size: int = 500) -> pd.DataFrame:
    """Mean |SHAP| across a random sample of scored orders."""
    explainer, _kind, prep, names = _explainer()
    data = load_dataset()
    sample = data[FEATURE_COLUMNS].sample(min(sample_size, len(data)), random_state=7)
    X = prep.transform(sample)

    if hasattr(X, "toarray"):
        X = X.toarray()

    contrib = _shap_matrix(explainer, X)
    imp = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(contrib).mean(axis=0),
        }
    )
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
