"""Generate the dataset inspection report (reports/eda_report.md).

Covers the pre-implementation deliverable: schema, row counts, missing values,
joins, target distribution, feature list, leakage risks and the final modeling
table design.

    python src/eda_report.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader as dl
from feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model_dataset
from preprocessing import build_order_base

OUT = os.path.join(dl.REPORT_DIR, "eda_report.md")


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> None:
    os.makedirs(dl.REPORT_DIR, exist_ok=True)
    lines = ["# DelayShield - dataset inspection report", ""]

    lines += ["## 1. Raw tables, row counts and missing values", "",
              md_table(dl.dataset_profile()), ""]

    lines += ["## 2. Schema of the tables that feed the model", ""]
    for key in ["orders", "order_items", "products", "sellers", "customers", "reviews"]:
        df = dl.load_table(key)
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing_%": (100 * df.isna().mean()).round(2).values,
            "n_unique": [df[c].nunique() for c in df.columns],
        })
        lines += [f"### {key} ({len(df):,} rows)", "", md_table(schema), ""]

    lines += [
        "## 3. Joins used to build the modeling table",
        "",
        "```",
        "orders (delivered only, all timestamps present)",
        "  |-- order_items      : order_id  -> aggregated to one row per order",
        "  |     |-- products   : product_id (category, weight, dimensions)",
        "  |     '-- sellers    : seller_id of the PRIMARY seller (largest value line)",
        "  |-- customers        : customer_id (state, city, zip prefix)",
        "  |-- geolocation      : zip prefix centroid, twice (seller and customer)",
        "  '-- reviews          : order_id -> score (supplier scorecard only)",
        "```",
        "",
        "Multi-seller orders: 1,275 of 96,470 orders (1.3%) involve more than one",
        "seller. The order-level row keeps `seller_count` and attributes the order to",
        "the primary seller; supplier prioritisation is handled by a separate",
        "seller-level scorecard.",
        "",
    ]

    base = build_order_base()
    lines += [
        "## 4. Target distribution",
        "",
        "`delivery_delay = order_delivered_customer_date > order_estimated_delivery_date`",
        "",
        f"- Usable delivered orders: **{len(base):,}**",
        f"- Late: **{int(base['delivery_delay'].sum()):,}** "
        f"({base['delivery_delay'].mean():.2%})",
        f"- On time: **{int((1 - base['delivery_delay']).sum()):,}**",
        f"- Purchase window: {base['order_purchase_timestamp'].min():%Y-%m-%d} to "
        f"{base['order_purchase_timestamp'].max():%Y-%m-%d}",
        "",
    ]
    quarterly = (
        base.groupby(base["order_purchase_timestamp"].dt.to_period("Q"))
        .agg(orders=("order_id", "size"), delay_rate=("delivery_delay", "mean"))
        .round(4)
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "quarter"})
    )
    quarterly["quarter"] = quarterly["quarter"].astype(str)
    lines += ["### Delay rate over time", "", md_table(quarterly), "",
              "The delay rate is far from stationary (2018 Q1 peaks well above the",
              "2017 average), which is the reason for the chronological split.", ""]

    data = build_model_dataset(base)
    lines += [
        "## 5. Final modeling table",
        "",
        f"- Rows: **{len(data):,}** (one per delivered order)",
        f"- Model features: **{len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}** "
        f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)",
        "",
        "### Numeric features",
        "",
        ", ".join(f"`{c}`" for c in NUMERIC_FEATURES),
        "",
        "### Categorical features",
        "",
        ", ".join(f"`{c}`" for c in CATEGORICAL_FEATURES),
        "",
        "### Feature summary",
        "",
        md_table(data[NUMERIC_FEATURES].describe().T.round(3).reset_index()
                 .rename(columns={"index": "feature"})),
        "",
    ]

    lines += [
        "## 6. Leakage risks and how each is handled",
        "",
        md_table(pd.DataFrame([
            {"Risk": "Actual delivery timestamp",
             "Handling": "Used only to build the target; never a feature."},
            {"Risk": "Carrier hand-over date / approval date",
             "Handling": "Excluded - both are generated after confirmation."},
            {"Risk": "Review score and comments",
             "Handling": "Post-delivery. Used only in the supplier scorecard."},
            {"Risk": "Order status",
             "Handling": "Only 'delivered' rows are modelled; status is not a feature."},
            {"Risk": "Seller history computed over the whole dataset",
             "Handling": "Replaced by lagged statistics using only orders already "
                         "delivered before the current purchase timestamp."},
            {"Risk": "Category / route / platform delay rates",
             "Handling": "Same lagged construction as seller history."},
            {"Risk": "Global means for cold-start fallbacks",
             "Handling": "Running platform average available at that moment, not a "
                         "full-dataset mean."},
            {"Risk": "Random train/test split",
             "Handling": "Chronological split; validation and test come strictly "
                         "after the training window."},
        ])),
        "",
        "## 7. Design notes",
        "",
        "- Orders with a delivery date before the purchase date are dropped as corrupt.",
        "- Product weight/dimension gaps are median-imputed before aggregation.",
        "- Distance is a great-circle proxy between zip-prefix centroids "
        "(the geolocation table has ~1M points; centroids are enough for a proxy).",
        "- Cold-start sellers (5.5% of orders) are flagged and fall back to the "
        "running platform delay rate.",
        "",
    ]

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
