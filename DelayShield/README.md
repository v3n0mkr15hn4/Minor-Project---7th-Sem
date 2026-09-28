# DelayShield — Delivery Risk Intelligence

Explainable machine learning for delivery-delay risk prediction and supplier
prioritization, built end-to-end on the **Brazilian E-Commerce Public Dataset by
Olist** (99,441 orders, 2016–2018).

DelayShield is a decision-support system, not a classification notebook:

```
PREDICT → EXPLAIN → PRIORITIZE → RECOMMEND → SIMULATE → DECIDE
```

At order confirmation it estimates the probability the order will miss its
promised delivery date, explains the estimate with SHAP, ranks suppliers,
recommends an operational action, and simulates a proactive policy against the
reactive status quo under transparent cost assumptions.

---

## 1. Quick start

```bash
# 1. install dependencies (Python 3.10+; tested on 3.13)
pip install -r requirements.txt

# 2. the 9 raw Olist CSVs must be in data/raw/
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# 3. run the offline pipeline (clean → features → train → score → persist)
python src/train.py           # ~50 s on a laptop

# 4. optional: regenerate the dataset inspection report
python src/eda_report.py

# 5. launch the dashboard
streamlit run dashboard/app.py
```

Windows shortcut: `run.bat` (trains if no model exists, then launches the app).
macOS/Linux: `./run.sh`.

---

## 2. What the pipeline does

| Stage | Module | Output |
|---|---|---|
| Load raw tables | `src/data_loader.py` | 9 Olist CSVs, geolocation centroids |
| Clean + integrate | `src/preprocessing.py` | one row per delivered order, target built |
| Time-aware features | `src/feature_engineering.py` | 41 leakage-free features |
| Train + evaluate | `src/train.py` | 6 model configurations, chronological split |
| Predict | `src/predict.py` | single-order and batch scoring |
| Explain | `src/explain.py` | SHAP local + global explanations |
| Prioritize | `src/supplier_scoring.py` | supplier scorecard, configurable weights |
| Recommend | `src/recommendations.py` | rule-based action playbook |
| Simulate | `src/simulation.py` | reactive vs risk-aware policy, sensitivity |
| Dashboard | `dashboard/app.py` | 7-page Streamlit application |

Artefacts written by `src/train.py`:

```
models/delay_model.pkl          selected sklearn Pipeline (preprocessing + model)
models/feature_columns.pkl      feature contract
models/model_metadata.json      split, metrics, thresholds, assumptions, leakage policy
data/processed/model_dataset.csv    engineered modelling table (96,470 rows)
data/processed/scored_orders.csv    every order with its predicted probability
data/processed/supplier_metrics.csv supplier scorecard (2,959 sellers)
reports/model_comparison.csv        all six model configurations, val + test
reports/eda_report.md               dataset inspection deliverable
```

---

## 3. Target definition

```
delivery_delay = order_delivered_customer_date > order_estimated_delivery_date
```

Only `order_status == 'delivered'` rows with all required timestamps are used;
rows where delivery precedes purchase are dropped as corrupt.

- Usable orders: **96,470**
- Late: **7,826 (8.11 %)** — on time: 88,644
- Purchase window: 2016-09-15 → 2018-08-29

---

## 4. Leakage control (the core technical requirement)

Every feature must be knowable **at order confirmation**. Two separate rules are
enforced:

**Rule 1 — no post-outcome columns as features.** Actual delivery date, carrier
hand-over date, approval timestamp, review score and order status are never
model inputs. They are used only to construct the target, the lagged history,
and the retrospective supplier scorecard.

**Rule 2 — no look-ahead inside the history features.** A seller's historical
delay rate for an order placed on 2018-03-01 uses only that seller's orders that
were **already delivered before 2018-03-01**. An order placed two days earlier
has no known outcome yet, so it contributes nothing. This is stricter than an
expanding mean over prior rows, and it is what makes the feature reproducible in
production. Implementation: `_lagged_outcome_stats()` in
`src/feature_engineering.py` sorts each group by the moment its outcome became
known and uses `searchsorted` on the purchase timestamp.

Workload features (`seller_orders_last_30_days`, `..._90_days`,
`seller_workload_surge`, `platform_orders_last_30_days`) count *purchase events*,
which are known immediately, so they are legitimately available at confirmation.

Cold-start fallback: new sellers/categories (5.5 % of orders) fall back to the
**running platform delay rate at that moment**, never a full-dataset mean.

---

## 5. Feature set (41 features)

| Group | Features |
|---|---|
| Order timing | month, week, day, weekday, hour, quarter, weekend flag |
| Promise | `estimated_lead_time_days`, `shipping_limit_days`, `lead_time_slack_days`, `lead_time_ratio` |
| Commercial | order value, freight value, freight ratio, item count, distinct products, seller count, max item price |
| Product | weight, volume |
| Location | `distance_km` (great-circle between zip-prefix centroids), `same_state`, customer zip prefix, `distance_per_promised_day` |
| Seller history (lagged) | delivered-order count, delay rate, on-time rate, avg delivery days, avg lateness, avg freight |
| Workload | seller orders last 30/90 days, workload surge, platform orders last 30 days |
| Context history (lagged) | category delay rate, destination-state delay rate, route delay rate, platform delay rate |
| Categorical | product category, customer state, seller state (one-hot, rare levels grouped) |

`lead_time_slack_days` (promised lead time minus the seller's usual delivery
time) and `distance_per_promised_day` turned out to be among the strongest
signals — the model mostly learns *when the promise is optimistic for this
seller on this route*.

---

## 6. Multi-seller orders

1,275 of 96,470 orders (1.3 %) contain items from more than one seller. The
project does **not** assume `1 order = 1 seller`:

- the order-level row keeps `seller_count` as a feature;
- the order is attributed to a **primary seller** — the seller holding the
  largest share of the order value (ties broken by first item id) — whose lagged
  history drives the seller features;
- supplier prioritization is a **separate seller-level scorecard**
  (`src/supplier_scoring.py`), so multi-seller orders contribute to each
  relevant seller's aggregate there.

---

## 7. Chronological split

A random split would let the model see future orders and the same seller's later
performance — impossible at prediction time — and would inflate every metric.

| Split | Rows | Window | Late rate |
|---|---|---|---|
| Train | 43,693 | 2016-09-15 → 2017-12-31 | 6.60 % |
| Validation | 20,627 | 2018-01-01 → 2018-03-31 | 14.58 % |
| Test | 32,150 | 2018-04-01 → 2018-08-29 | 6.02 % |

Validation is used for model selection and threshold tuning; the test window is
scored once for the reported numbers. The validation window covers the 2018 Q1
disruption, so its delay rate is more than double the test window's — a real
regime shift that the reported numbers do not hide.

---

## 8. Class imbalance

Positives are 8.1 % overall (6.0 % in the test window). Two treatments are
trained and compared for **every** model family:

- **class weighting** (`class_weight='balanced'` / `scale_pos_weight = 14.16`);
- **threshold tuning only** (no weighting; the decision threshold is tuned on
  validation for F1 and separately for expected cost).

**SMOTE was deliberately not used.** Synthesising minority rows inside a
temporally ordered dataset interpolates between orders from different time
periods and leaks information across the split boundary.

---

## 9. Measured results (actual numbers, no rounding up)

All six configurations, validation and test (`reports/model_comparison.csv`):

| Model | Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| logistic_regression_weighted | val | 0.7009 | 0.2836 | 0.2522 | 0.5778 | 0.3512 | 0.1361 |
| logistic_regression_weighted | test | 0.7552 | 0.1491 | 0.1568 | 0.4778 | 0.2362 | 0.0811 |
| random_forest_weighted | val | 0.7068 | 0.2991 | 0.3002 | 0.4707 | 0.3666 | 0.1489 |
| random_forest_weighted | test | 0.6803 | 0.1140 | 0.0977 | 0.5780 | 0.1672 | 0.1503 |
| xgboost_weighted | val | 0.7107 | 0.2886 | 0.2753 | 0.5472 | 0.3663 | 0.1712 |
| xgboost_weighted | test | 0.6808 | 0.1120 | 0.0988 | 0.6059 | 0.1699 | 0.1924 |
| logistic_regression_threshold_only | val | 0.7190 | 0.3105 | 0.4305 | 0.1935 | 0.2670 | 0.1368 |
| logistic_regression_threshold_only | test | 0.7541 | 0.1442 | 0.1799 | 0.1436 | 0.1597 | 0.0572 |
| **random_forest_threshold_only (selected)** | val | 0.6987 | **0.3138** | 0.2953 | 0.4731 | 0.3636 | 0.1202 |
| **random_forest_threshold_only (selected)** | test | 0.7261 | 0.1347 | 0.1061 | 0.7474 | 0.1859 | 0.0591 |
| xgboost_threshold_only | val | 0.7108 | 0.3076 | 0.2875 | 0.5419 | 0.3757 | 0.1193 |
| xgboost_threshold_only | test | 0.6963 | 0.1216 | 0.0980 | 0.6705 | 0.1710 | 0.0587 |

**Selected model: Random Forest without class weighting**, chosen on the highest
validation PR-AUC (0.3138). Selection used validation only — the test column is
reported, never optimised against.

Test-window headline (threshold 0.13, tuned on validation for F1):

```
ROC-AUC 0.7261   PR-AUC 0.1347   Recall 0.7474   Precision 0.1061   Brier 0.0591
Confusion matrix: TN 18,029  FP 12,185  FN 489  TP 1,447   (base rate 6.02 %)
```

Read honestly: the model **ranks** risk usefully (AUC 0.73, and PR-AUC is 2.2×
the 6.0 % base rate) but a flagged order is still more likely on-time than late.
Alerts are a triage queue, not a verdict. Accuracy is *not* used as a selection
metric — predicting "never late" would score 94 % accuracy and be useless.

---

## 10. Risk classification

Fixed 0.30/0.60 bands would leave HIGH empty: the selected model's test
probabilities span 0.040–0.272 (median 0.123). Bands are therefore derived from
the **validation probability distribution**:

```
LOW     p < 0.1273          (below the 75th percentile)
MEDIUM  0.1273 ≤ p < 0.1671 (75th–92nd percentile)
HIGH    p ≥ 0.1671          (top ~8 % of predicted risk)
```

These are operational capacity choices, not validated boundaries, and both
sliders are live in the dashboard sidebar. On the test window the observed delay
rate rises monotonically across the three bands — the operational sanity check
the Dashboard page displays:

| Band | Orders | Observed delay rate |
|---|---|---|
| LOW | 17,587 | 2.47 % |
| MEDIUM | 10,292 | 8.54 % |
| HIGH | 4,271 | 14.56 % |

The HIGH band carries 2.4× the population delay rate (6.02 %) and the LOW band
0.4×.

---

## 11. Supplier prioritization

Per seller: order volume, on-time rate, delay rate, average delivery days,
average lateness when late, freight ratio, mean review score, mean predicted
risk. Components are min-max normalised (1 = best) and combined:

```
Supplier Score = 0.35 × reliability          (historical on-time rate)
               + 0.25 × predicted delay performance (inverted model risk)
               + 0.15 × delivery speed       (inverted avg delivery days)
               + 0.10 × freight performance  (inverted freight/value ratio)
               + 0.15 × customer feedback    (mean review score)
```

This is an **initial operational weighting**, not a proven formula — every
weight is a slider on the Supplier Ranking page and the ranking re-computes
live. Tiers: WATCHLIST < 40, STANDARD 40–60, PREFERRED > 60 (0–100 scale).
Sellers with fewer than 10 delivered orders are ranked but flagged
`low_volume_flag`, because their rates are statistically noisy.

Review scores appear **only** here — a retrospective supplier view — never as a
model feature.

---

## 12. Action recommendation engine

Rule-based on purpose, so operations staff can audit it:

| Risk | Headline | Actions |
|---|---|---|
| LOW | Normal processing | standard flow, no pre-notification, standard tracking |
| MEDIUM | Monitor and buffer | increased tracking, +2-day schedule buffer, weekly supplier monitoring |
| HIGH | Proactive intervention | proactive customer communication, alternate-supplier review, +5-day buffer and carrier re-plan, daily tracking escalation |

Context rules extend the playbook: supplier delay rate > 15 % adds a
corrective-action review, WATCHLIST tier restricts new high-value orders, order
value > 1,000 adds priority handling, probability ≥ 0.8 suggests split shipment
or carrier upgrade.

---

## 13. Policy simulation

Both policies run on the **same** held-out orders with the same assumptions.

- **Reactive** — nothing at confirmation; an intervention runs on every order
  that ends up visibly late, recovering `reactive_effectiveness` (default 15 %)
  of the damage.
- **DelayShield risk-aware** — every order with `p ≥ threshold` gets a proactive
  intervention at confirmation, recovering `proactive_effectiveness`
  (default 60 %). Wrong flags cost a false alert plus the intervention. Orders
  that slip through are still handled reactively, so the comparison is fair.

Default assumptions (**ASSUMPTION — Olist publishes no cost data**; all editable
in the dashboard): late delivery 1,200 · false alert 30 · intervention 50 ·
proactive effectiveness 0.60 · reactive effectiveness 0.15.

The app derives the **break-even precision** analytically:

```
break-even precision = (false_alert + intervention) / ((proactive_eff − reactive_eff) × late_cost + false_alert + intervention)
                     = 80 / (540 + 80) = 12.9 %  under the defaults
```

Measured on the test window with those defaults, the risk-aware policy is cost
positive only above roughly the 0.16 threshold, where alert precision passes the
break-even line — e.g. at threshold 0.18 it catches 435 of 1,936 late orders
(22.5 % recall) at 16.4 % precision for a saving of ~57.8k versus reactive, and
at threshold 0.10 it *loses* ~839k because precision falls to 7.4 %. The
dashboard's threshold sweep shows this band explicitly rather than hiding it.
Note also that the validation-optimal cost threshold (0.09) does **not** transfer
to the test window — an honest demonstration of the regime shift.

---

## 14. Dashboard pages

1. **Dashboard** — KPI row (orders, high-risk orders, average delay probability,
   actual delay rate, high-risk suppliers, avoidable cost), monthly actual vs
   predicted risk, risk mix, and the band-lift sanity table.
2. **Order Risk Prediction** — live form (seller, category, value, freight,
   items, states, date, promised lead time) → probability, risk band, gauge,
   SHAP explanation, recommended action, supplier context, raw feature vector.
3. **Supplier Ranking** — live weight sliders, volume/state filters, ranked
   scorecard, volume-vs-delay scatter, highest-risk supplier chart.
4. **Risk Alerts** — filterable high-risk order queue with recommended actions,
   alert-quality lift versus the population, CSV export.
5. **Analytics** — late vs on-time, probability distribution, delay by category,
   destination state, seller and month, global SHAP importance.
6. **Policy Simulation** — editable assumptions, reactive vs DelayShield KPI
   cards, cost breakdown, threshold sweep, one-parameter sensitivity analysis.
7. **Model & Method** — split table, all six model results, calibration curve,
   confusion matrix, leakage policy and the limitations list.

Sidebar controls apply globally: data scope (test / validation / full history)
and the two risk-band thresholds.

---

## 15. Reviewer Q&A

**Why Olist?** It is the dataset named in the approved proposal, and the only
large public e-commerce dataset that carries *both* a promised delivery date and
an actual delivery date per order, plus seller, product, freight, location and
review tables — everything the abstract requires.

**Why this target?** `actual > estimated` is the SLA breach the business
actually cares about and the customer actually experiences. It needs no
arbitrary threshold, and both timestamps exist for all delivered orders.

**Why a time-based split?** Delivery risk is non-stationary (train 6.6 %,
validation 14.6 %, test 6.0 % late). A random split would train on future orders
and on the same seller's later performance, inflating metrics and being
impossible to reproduce in production.

**How was leakage avoided?** Two rules, section 4: no post-outcome columns as
features, and history features built only from orders already *delivered* before
the current purchase timestamp — verified by construction in
`_lagged_outcome_stats()`.

**Why Random Forest / XGBoost?** The relationships are non-linear and
interaction-heavy (promised lead time matters *relative to* the seller's usual
speed and the route distance). Logistic regression is the interpretable
baseline; both tree families were trained, and all six configurations are
reported. The Random Forest won on validation PR-AUC — notably, the linear
baseline generalised better to the test window, which the report states plainly
rather than burying.

**Why SHAP?** It gives per-order, signed, additive attributions — the operator
needs "why is *this* order risky", not a global feature ranking. Tree SHAP is
exact for the selected model and fast enough for live use.

**How is the supplier score calculated?** Section 11 — five normalised
components with configurable weights, exposed as sliders.

**How are risk thresholds selected?** From validation probability quantiles
(75th / 92nd), because a class-imbalanced model rarely emits probabilities above
0.6. They are operational capacity choices and are adjustable live; the
cost-optimal threshold is computed separately in `train.py`.

**How does the simulation work?** Section 13 — same orders, two policies,
explicit assumptions, plus an analytic break-even precision and a full
sensitivity panel.

**What are the limitations?**
- Ranking quality, not certainty: test ROC-AUC 0.726, alert precision ~11–16 %.
- Regime shift between windows; thresholds do not transfer perfectly.
- All costs and intervention effectiveness are assumptions, not observations.
- Intervention effect cannot be proven from historical data — it needs an A/B
  experiment on flagged orders.
- The prediction form uses each seller's latest history snapshot; production
  would need a continuously updated feature store.
- Olist is Brazilian 2016–2018 data; conclusions do not transfer unchanged to
  another market or period.

**Next production step?** (1) Serve the model behind an API at order
confirmation with a proper feature store recomputing lagged statistics
continuously; (2) run a controlled A/B test on flagged orders to *measure*
intervention effectiveness instead of assuming it; (3) add monitoring for drift
and for delay-rate regime changes with scheduled retraining; (4) re-tune
thresholds against measured costs once real intervention data exists.

---

## 16. Project structure

```
DelayShield/
├── data/
│   ├── raw/          9 original Olist CSVs
│   ├── processed/    model_dataset.csv, scored_orders.csv, supplier_metrics.csv
│   └── sample/       small samples for inspection
├── models/           delay_model.pkl, feature_columns.pkl, model_metadata.json
├── notebooks/        01_eda.ipynb, 02_model_development.ipynb
├── reports/          eda_report.md, model_comparison.csv
├── src/              data_loader, preprocessing, feature_engineering, train,
│                     predict, explain, supplier_scoring, recommendations,
│                     simulation, eda_report
├── dashboard/app.py  Streamlit application
├── requirements.txt
├── run.bat / run.sh
└── README.md
```
