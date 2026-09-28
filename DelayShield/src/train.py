"""Train, evaluate and persist the DelayShield delay-risk models.

Run from the project root:

    python src/train.py

The script executes the whole offline pipeline: clean -> integrate -> engineer
features -> chronological split -> train three models -> evaluate -> tune the
decision threshold -> persist the artefacts the dashboard loads.
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import data_loader as dl
from feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    build_model_dataset,
)
from preprocessing import build_order_base
from supplier_scoring import build_supplier_metrics

# --- chronological split boundaries -------------------------------------------------
# Chosen from the data distribution: ~2 years of history, the last ~8 months
# held out. Validation is used for model selection and threshold tuning, test is
# touched exactly once for the reported numbers.
VAL_START = pd.Timestamp("2018-01-01")
TEST_START = pd.Timestamp("2018-04-01")

# Business cost assumptions (ASSUMPTION - editable in the dashboard).
DEFAULT_COSTS = {
    "cost_of_late_delivery": 1200.0,  # refund/credit + support + churn share (ASSUMPTION)
    "cost_of_false_alert": 30.0,      # unnecessary customer heads-up (ASSUMPTION)
    "cost_of_intervention": 50.0,     # buffer + tracking + comms per order (ASSUMPTION)
    "intervention_effectiveness": 0.6,  # share of true delays mitigated by acting early
}

# Fallback cut-offs. The persisted thresholds are recomputed from the validation
# probability distribution because a class-imbalanced model rarely emits
# probabilities above 0.6 - fixed 0.30/0.60 cut-offs would leave HIGH empty.
RISK_THRESHOLDS = {"medium": 0.30, "high": 0.60}


def derive_risk_thresholds(val_probs: np.ndarray) -> dict:
    """Operational risk bands from the validation probability distribution.

    MEDIUM starts at the 75th percentile and HIGH at the 92nd percentile of
    predicted risk, i.e. roughly the riskiest quarter and the riskiest ~8% of
    orders. These are operational capacity choices, not statistically validated
    boundaries, and the dashboard lets a reviewer move them.
    """
    return {
        "medium": float(round(np.quantile(val_probs, 0.75), 4)),
        "high": float(round(np.quantile(val_probs, 0.92), 4)),
    }


def chronological_split(df: pd.DataFrame):
    """Split by purchase timestamp - never randomly."""
    ts = df["order_purchase_timestamp"]
    train = df[ts < VAL_START]
    val = df[(ts >= VAL_START) & (ts < TEST_START)]
    test = df[ts >= TEST_START]
    return train, val, test


def build_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=30,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models(pos_weight: float, weighted: bool = True) -> dict:
    """The three model families the proposal asks to compare.

    ``weighted`` toggles the class-imbalance handling: with ``True`` each model
    gets balanced class weights, with ``False`` the imbalance is left alone and
    handled purely by tuning the decision threshold. Both routes are trained and
    compared instead of assuming one is better (SMOTE is deliberately avoided:
    synthesising minority rows inside a temporally ordered dataset mixes
    information across time periods).
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor(scale=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced" if weighted else None, C=0.5
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced_subsample" if weighted else None,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=20,
                        reg_lambda=3.0,
                        scale_pos_weight=pos_weight if weighted else 1.0,
                        eval_metric="aucpr",
                        tree_method="hist",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def policy_cost(y_true, y_prob, threshold: float, costs: dict) -> dict:
    """Cost of acting on every order whose predicted risk exceeds ``threshold``.

    All monetary values are ASSUMPTIONS (see DEFAULT_COSTS); the comparison
    between policies is what carries meaning, not the absolute number.
    """
    flag = (y_prob >= threshold).astype(int)
    y_true = np.asarray(y_true)
    tp = int(((flag == 1) & (y_true == 1)).sum())
    fp = int(((flag == 1) & (y_true == 0)).sum())
    fn = int(((flag == 0) & (y_true == 1)).sum())
    tn = int(((flag == 0) & (y_true == 0)).sum())

    eff = costs["intervention_effectiveness"]
    avoided = tp * eff
    late_cost = (fn + tp * (1 - eff)) * costs["cost_of_late_delivery"]
    false_alert_cost = fp * costs["cost_of_false_alert"]
    intervention_cost = (tp + fp) * costs["cost_of_intervention"]
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts": tp + fp,
        "delays_avoided": float(avoided),
        "late_cost": float(late_cost),
        "false_alert_cost": float(false_alert_cost),
        "intervention_cost": float(intervention_cost),
        "total_cost": float(late_cost + false_alert_cost + intervention_cost),
    }


def evaluate(y_true, y_prob, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": float(np.mean(y_true)),
    }


def tune_threshold(y_true, y_prob, objective: str = "f1", costs: dict | None = None):
    """Pick an operating threshold on the validation set."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 3)
    best, best_score = 0.5, -np.inf
    for t in grid:
        if objective == "f1":
            score = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        else:
            score = -policy_cost(y_true, y_prob, t, costs or DEFAULT_COSTS)["total_cost"]
        if score > best_score:
            best, best_score = float(t), float(score)
    return best


def main() -> None:
    t0 = time.time()
    os.makedirs(dl.PROCESSED_DIR, exist_ok=True)
    os.makedirs(dl.MODEL_DIR, exist_ok=True)
    os.makedirs(dl.REPORT_DIR, exist_ok=True)

    print("[1/7] Cleaning + integrating raw Olist tables ...")
    base = build_order_base()
    print(f"      delivered orders with usable timestamps: {len(base):,}")

    print("[2/7] Building time-aware features ...")
    data = build_model_dataset(base)
    data.to_csv(os.path.join(dl.PROCESSED_DIR, "model_dataset.csv"), index=False)
    print(f"      model table: {data.shape[0]:,} rows x {data.shape[1]} cols")

    print("[3/7] Chronological split ...")
    train, val, test = chronological_split(data)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(
            f"      {name:<5} n={len(part):>6,}  "
            f"{part['order_purchase_timestamp'].min():%Y-%m-%d} -> "
            f"{part['order_purchase_timestamp'].max():%Y-%m-%d}  "
            f"late={part[TARGET].mean():.3%}"
        )

    X_tr, y_tr = train[FEATURE_COLUMNS], train[TARGET]
    X_va, y_va = val[FEATURE_COLUMNS], val[TARGET]
    X_te, y_te = test[FEATURE_COLUMNS], test[TARGET]

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"      class imbalance -> scale_pos_weight={pos_weight:.2f}")

    print("[4/7] Training models ...")
    results, fitted = {}, {}
    candidates = {}
    for weighted in (True, False):
        suffix = "_weighted" if weighted else "_threshold_only"
        for fam, model in build_models(pos_weight, weighted).items():
            candidates[fam + suffix] = model
    for name, model in candidates.items():
        t = time.time()
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        thr_f1 = tune_threshold(y_va, p_va, "f1")
        thr_cost = tune_threshold(y_va, p_va, "cost", DEFAULT_COSTS)
        results[name] = {
            "validation": evaluate(y_va, p_va, thr_f1),
            "threshold_f1": thr_f1,
            "threshold_cost": thr_cost,
            "fit_seconds": round(time.time() - t, 1),
        }
        fitted[name] = model
        v = results[name]["validation"]
        print(
            f"      {name:<32} ROC-AUC={v['roc_auc']:.4f} PR-AUC={v['pr_auc']:.4f} "
            f"F1={v['f1']:.4f} R={v['recall']:.4f} P={v['precision']:.4f} "
            f"({results[name]['fit_seconds']}s)"
        )

    # Model choice: PR-AUC on validation. With ~8% positives PR-AUC reflects the
    # ranking quality that matters operationally far better than accuracy.
    best_name = max(results, key=lambda k: results[k]["validation"]["pr_auc"])
    best_model = fitted[best_name]
    print(f"      selected model: {best_name} (highest validation PR-AUC)")

    print("[5/7] Test-set evaluation (touched once) ...")
    p_te = best_model.predict_proba(X_te)[:, 1]
    thr_f1 = results[best_name]["threshold_f1"]
    thr_cost = results[best_name]["threshold_cost"]
    test_metrics = evaluate(y_te, p_te, thr_f1)
    test_metrics_cost_thr = evaluate(y_te, p_te, thr_cost)
    print(
        f"      TEST ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"PR-AUC={test_metrics['pr_auc']:.4f} F1={test_metrics['f1']:.4f} "
        f"Recall={test_metrics['recall']:.4f} Precision={test_metrics['precision']:.4f}"
    )
    for name in results:
        p = fitted[name].predict_proba(X_te)[:, 1]
        results[name]["test"] = evaluate(y_te, p, results[name]["threshold_f1"])

    risk_thresholds = derive_risk_thresholds(best_model.predict_proba(X_va)[:, 1])
    print(
        f"      risk bands -> MEDIUM >= {risk_thresholds['medium']:.4f}, "
        f"HIGH >= {risk_thresholds['high']:.4f}"
    )

    prob_true, prob_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    calibration = {
        "prob_pred": [float(x) for x in prob_pred],
        "prob_true": [float(x) for x in prob_true],
    }

    print("[6/7] Scoring all orders + supplier scorecard ...")
    data["delay_probability"] = best_model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    data["split"] = np.where(
        data["order_purchase_timestamp"] < VAL_START,
        "train",
        np.where(data["order_purchase_timestamp"] < TEST_START, "validation", "test"),
    )
    scored_cols = [
        "order_id", "primary_seller_id", "order_purchase_timestamp",
        "order_estimated_delivery_date", "product_category", "customer_state",
        "seller_state", "customer_city", "seller_city", "total_order_value",
        "total_freight_value", "item_count", "distance_km",
        "estimated_lead_time_days", "seller_hist_delay_rate",
        "seller_hist_on_time_rate", "seller_hist_avg_delivery_days",
        "seller_orders_last_30_days", "delay_probability", "delivery_delay",
        "delivery_days", "delay_days", "review_score", "split",
    ]
    data[scored_cols].to_csv(
        os.path.join(dl.PROCESSED_DIR, "scored_orders.csv"), index=False
    )

    suppliers = build_supplier_metrics(data)
    suppliers.to_csv(os.path.join(dl.PROCESSED_DIR, "supplier_metrics.csv"), index=False)
    print(f"      suppliers scored: {len(suppliers):,}")

    print("[7/7] Persisting artefacts ...")
    joblib.dump(best_model, os.path.join(dl.MODEL_DIR, "delay_model.pkl"))
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        os.path.join(dl.MODEL_DIR, "feature_columns.pkl"),
    )

    metadata = {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "dataset": "Brazilian E-Commerce Public Dataset by Olist (raw, 9 CSVs)",
        "rows_total": int(len(data)),
        "target_definition": "delivery_delay = order_delivered_customer_date > order_estimated_delivery_date",
        "positive_rate_overall": float(data[TARGET].mean()),
        "split": {
            "type": "chronological",
            "val_start": str(VAL_START.date()),
            "test_start": str(TEST_START.date()),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_range": [str(train["order_purchase_timestamp"].min()),
                            str(train["order_purchase_timestamp"].max())],
            "val_range": [str(val["order_purchase_timestamp"].min()),
                          str(val["order_purchase_timestamp"].max())],
            "test_range": [str(test["order_purchase_timestamp"].min()),
                           str(test["order_purchase_timestamp"].max())],
            "train_positive_rate": float(y_tr.mean()),
            "val_positive_rate": float(y_va.mean()),
            "test_positive_rate": float(y_te.mean()),
        },
        "selected_model": best_name,
        "selection_criterion": "highest validation PR-AUC",
        "threshold_f1": thr_f1,
        "threshold_cost_optimal": thr_cost,
        "test_metrics_at_f1_threshold": test_metrics,
        "test_metrics_at_cost_threshold": test_metrics_cost_thr,
        "model_comparison": results,
        "calibration_test": calibration,
        "risk_thresholds": risk_thresholds,
        "risk_thresholds_note": (
            "Derived from validation-set probability quantiles (75th / 92nd); "
            "configurable in the dashboard."
        ),
        "probability_distribution_test": {
            "min": float(p_te.min()), "p50": float(np.quantile(p_te, 0.5)),
            "p90": float(np.quantile(p_te, 0.9)), "p99": float(np.quantile(p_te, 0.99)),
            "max": float(p_te.max()),
        },
        "cost_assumptions": DEFAULT_COSTS,
        "feature_columns": FEATURE_COLUMNS,
        "leakage_policy": (
            "Only order-confirmation-time information is used. Delivery "
            "timestamps, review scores and post-delivery status are excluded "
            "from features; historical seller/category/route statistics use "
            "only orders already delivered before the current purchase time."
        ),
    }
    with open(os.path.join(dl.MODEL_DIR, "model_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    comparison_rows = []
    for name, res in results.items():
        for split_name in ("validation", "test"):
            row = {"model": name, "split": split_name}
            row.update({k: v for k, v in res[split_name].items()
                        if k != "confusion_matrix"})
            comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(dl.REPORT_DIR, "model_comparison.csv"), index=False
    )

    print(f"Done in {time.time() - t0:.1f}s. Artefacts in models/ and data/processed/.")


if __name__ == "__main__":
    main()
