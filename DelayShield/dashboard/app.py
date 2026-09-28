"""DelayShield - Delivery Risk Intelligence.

Streamlit front end for the delay-risk decision-support system:
predict -> explain -> prioritize -> recommend -> simulate -> decide.

Run from the project root:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import predict as P  # noqa: E402
import recommendations as R  # noqa: E402
import simulation as S  # noqa: E402
import supplier_scoring as SS  # noqa: E402

st.set_page_config(
    page_title="DelayShield | Delivery Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {"LOW": "#1a9d5a", "MEDIUM": "#d99413", "HIGH": "#d1394a"}
ACCENT = "#2563eb"
PLOT_TEMPLATE = "plotly_white"

CSS = """
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1500px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    .ds-brand {display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem;}
    .ds-brand .name {font-size:1.45rem; font-weight:700;}
    .ds-brand .tag {font-size:.78rem; text-transform:uppercase; letter-spacing:.13em;
                    color:#64748b;}
    .kpi {border:1px solid rgba(128,138,157,.28); border-radius:12px; padding:.85rem 1rem;
          background:rgba(148,163,184,.07); height:100%;}
    .kpi .label {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em;
                 color:#64748b; font-weight:600;}
    .kpi .value {font-size:1.65rem; font-weight:700; line-height:1.25; margin-top:.15rem;}
    .kpi .sub {font-size:.75rem; color:#64748b;}
    .pill {display:inline-block; padding:.18rem .6rem; border-radius:999px;
           font-size:.72rem; font-weight:700; letter-spacing:.06em;}
    .section {font-size:.78rem; text-transform:uppercase; letter-spacing:.12em;
              color:#64748b; font-weight:700; margin:1.2rem 0 .4rem;}
    .assumption {border-left:3px solid #d99413; background:rgba(217,148,19,.08);
                 padding:.55rem .8rem; border-radius:6px; font-size:.83rem;}
    .risk-banner {border-radius:12px; padding:1rem 1.2rem; color:#fff;}
    .risk-banner .p {font-size:2.4rem; font-weight:800; line-height:1;}
    div[data-testid="stMetricValue"] {font-size:1.4rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    df = P.load_scored()
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_suppliers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(P.dl.PROCESSED_DIR, "supplier_metrics.csv"))


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return P.load_metadata()


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    path = os.path.join(P.dl.REPORT_DIR, "model_comparison.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data(show_spinner=True)
def global_shap(n: int = 400) -> pd.DataFrame:
    import explain as E

    return E.global_importance(n)


def artefacts_ready() -> bool:
    return all(
        os.path.exists(p)
        for p in [
            P.MODEL_PATH,
            P.META_PATH,
            P.SCORED_PATH,
            os.path.join(P.dl.PROCESSED_DIR, "supplier_metrics.csv"),
        ]
    )


if not artefacts_ready():
    st.error(
        "Model artefacts not found. Run the offline pipeline first:\n\n"
        "```\npython src/train.py\n```"
    )
    st.stop()

scored = load_scored()
suppliers_raw = load_suppliers()
meta = load_meta()


# ----------------------------------------------------------------------------- helpers
def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def risk_pill(level: str) -> str:
    color = RISK_COLORS.get(level, "#64748b")
    return f'<span class="pill" style="background:{color}22;color:{color}">{level} RISK</span>'


def money(x: float) -> str:
    return f"{x:,.0f}"


def apply_risk_levels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    out["risk_level"] = P.risk_levels(out["delay_probability"].values, thresholds).values
    return out


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        '<div class="ds-brand"><span style="font-size:1.6rem">🛡️</span>'
        '<div><div class="name">DelayShield</div>'
        '<div class="tag">Delivery Risk Intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Model: **{meta['selected_model']}** · trained {meta['trained_at'][:10]} · "
        f"{meta['rows_total']:,} Olist orders"
    )
    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Order Risk Prediction",
            "Supplier Ranking",
            "Risk Alerts",
            "Analytics",
            "Policy Simulation",
            "Model & Method",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section">Data scope</div>', unsafe_allow_html=True)
    scope = st.selectbox(
        "Order population",
        ["Holdout test period (2018-04 → 2018-08)", "Validation period (2018-01 → 2018-03)",
         "Full history (2016-09 → 2018-08)"],
        help="Predictions on the holdout period were never seen during training.",
    )
    scope_key = {"H": "test", "V": "validation"}.get(scope[0], None)

    st.markdown('<div class="section">Risk bands</div>', unsafe_allow_html=True)
    default_bands = meta["risk_thresholds"]
    med = st.slider(
        "MEDIUM risk starts at", 0.02, 0.60, float(default_bands["medium"]), 0.005,
        help="Operational cut-off, not a statistically validated boundary.",
    )
    high = st.slider(
        "HIGH risk starts at", 0.02, 0.90, float(default_bands["high"]), 0.005
    )
    if high <= med:
        high = med + 0.005
    thresholds = {"medium": med, "high": high}
    st.caption(
        f"Defaults ({default_bands['medium']:.3f} / {default_bands['high']:.3f}) come from "
        "validation-set probability quantiles (75th / 92nd)."
    )

data = scored if scope_key is None else scored[scored["split"] == scope_key]
data = apply_risk_levels(data, thresholds)
suppliers_view = suppliers_raw


# ----------------------------------------------------------------------------- pages
def page_dashboard() -> None:
    st.markdown("## Delivery risk overview")
    st.caption(
        f"{scope} · {len(data):,} orders · predictions produced at order-confirmation time."
    )

    high_orders = data[data["risk_level"] == "HIGH"]
    hist_delay = data["delivery_delay"].mean()
    risky_suppliers = suppliers_view[
        (suppliers_view["order_volume"] >= SS.MIN_ORDERS_FOR_RANKING)
        & (suppliers_view["delay_rate"] > 0.15)
    ]
    a = dict(S.DEFAULT_ASSUMPTIONS, threshold=high)
    cmp_res = S.compare(data["delivery_delay"], data["delay_probability"], a)

    c = st.columns(6)
    kpi(c[0], "Total orders", f"{len(data):,}", scope.split("(")[0].strip())
    kpi(c[1], "High-risk orders", f"{len(high_orders):,}",
        f"{len(high_orders)/max(len(data),1):.1%} of population")
    kpi(c[2], "Avg delay probability", f"{data['delay_probability'].mean():.1%}",
        "model output, all orders")
    kpi(c[3], "Actual delay rate", f"{hist_delay:.1%}",
        "ground truth in this period")
    kpi(c[4], "High-risk suppliers", f"{len(risky_suppliers):,}",
        ">15% delay rate, ≥10 orders")
    kpi(c[5], "Avoidable cost", money(max(cmp_res["cost_saving"], 0)),
        "vs reactive policy (assumptions)")

    st.markdown(
        '<div class="assumption"><b>ASSUMPTION</b> — the avoidable-cost figure uses the '
        "configurable cost model on the Policy Simulation page "
        f"(late delivery {money(a['cost_of_late_delivery'])}, false alert "
        f"{money(a['cost_of_false_alert'])}, intervention {money(a['cost_of_intervention'])}). "
        "It is not an observed Olist cost.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section">Monthly delay rate vs predicted risk</div>',
                    unsafe_allow_html=True)
        monthly = (
            data.groupby("order_month")
            .agg(actual=("delivery_delay", "mean"),
                 predicted=("delay_probability", "mean"),
                 orders=("order_id", "count"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_bar(x=monthly["order_month"], y=monthly["orders"], name="Orders",
                    marker_color="rgba(148,163,184,.35)", yaxis="y2")
        fig.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["actual"],
                                 name="Actual delay rate", line=dict(color="#d1394a", width=3)))
        fig.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["predicted"],
                                 name="Mean predicted risk",
                                 line=dict(color=ACCENT, width=3, dash="dot")))
        fig.update_layout(
            template=PLOT_TEMPLATE, height=360, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Rate", tickformat=".0%"),
            yaxis2=dict(overlaying="y", side="right", title="Orders", showgrid=False),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section">Risk mix</div>', unsafe_allow_html=True)
        mix = data["risk_level"].value_counts().reindex(P.RISK_ORDER).fillna(0)
        fig = px.pie(values=mix.values, names=mix.index, hole=0.58,
                     color=mix.index, color_discrete_map=RISK_COLORS)
        fig.update_layout(template=PLOT_TEMPLATE, height=360,
                          margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Does the risk band separate real delays?</div>',
                unsafe_allow_html=True)
    band = (
        data.groupby("risk_level")
        .agg(orders=("order_id", "count"), actual_delay_rate=("delivery_delay", "mean"))
        .reindex(P.RISK_ORDER)
        .fillna(0)
        .reset_index()
    )
    c1, c2 = st.columns([2, 3])
    fig = px.bar(band, x="risk_level", y="actual_delay_rate", color="risk_level",
                 color_discrete_map=RISK_COLORS, text=band["actual_delay_rate"].map("{:.1%}".format))
    fig.update_layout(template=PLOT_TEMPLATE, height=300, showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(title="Observed delay rate", tickformat=".0%"),
                      xaxis_title="")
    c1.plotly_chart(fig, use_container_width=True)
    band["share_of_orders"] = band["orders"] / band["orders"].sum()
    c2.dataframe(
        band.rename(columns={"risk_level": "Risk band", "orders": "Orders",
                             "actual_delay_rate": "Observed delay rate",
                             "share_of_orders": "Share of orders"}),
        hide_index=True, use_container_width=True,
        column_config={
            "Observed delay rate": st.column_config.NumberColumn(format="percent"),
            "Share of orders": st.column_config.NumberColumn(format="percent"),
        },
    )
    c2.caption(
        "Lift check: the HIGH band should contain a materially higher share of truly "
        "late orders than the LOW band. This is the operational sanity test of the model."
    )


def page_prediction() -> None:
    st.markdown("## Order risk prediction")
    st.caption("Score a new order using only information available at order confirmation.")

    snap = P.history_snapshot()
    top_sellers = (
        suppliers_raw.sort_values("order_volume", ascending=False)["seller_id"].head(300).tolist()
    )
    cats = snap["categories"]
    states = snap["customer_states"]

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        seller = c1.selectbox("Seller", top_sellers,
                              help="300 highest-volume sellers from the Olist history.")
        category = c2.selectbox("Product category", cats,
                                index=cats.index("bed_bath_table") if "bed_bath_table" in cats else 0)
        order_date = c3.date_input("Order date", value=pd.Timestamp("2018-08-15"))

        c1, c2, c3 = st.columns(3)
        order_value = c1.number_input("Order value", 10.0, 20000.0, 350.0, 10.0)
        freight_value = c2.number_input("Freight value", 0.0, 2000.0, 45.0, 5.0)
        items = c3.number_input("Number of items", 1, 25, 2, 1)

        c1, c2, c3 = st.columns(3)
        seller_row = suppliers_raw[suppliers_raw["seller_id"] == seller]
        default_seller_state = (
            seller_row["seller_state"].iat[0] if len(seller_row) else "SP"
        )
        seller_state = c1.selectbox(
            "Seller state", snap["seller_states"],
            index=snap["seller_states"].index(default_seller_state)
            if default_seller_state in snap["seller_states"] else 0,
        )
        customer_state = c2.selectbox("Customer state", states,
                                      index=states.index("RJ") if "RJ" in states else 0)
        lead_time = c3.number_input("Promised lead time (days)", 2.0, 90.0, 20.0, 1.0,
                                    help="Estimated delivery date minus purchase date.")
        submitted = st.form_submit_button("Predict risk", type="primary",
                                          use_container_width=True)

    if not submitted:
        st.info("Fill the order details and press **Predict risk**.")
        return

    result = P.predict_order(
        seller_id=seller,
        product_category=category,
        total_order_value=order_value,
        total_freight_value=freight_value,
        item_count=int(items),
        customer_state=customer_state,
        seller_state=seller_state,
        order_date=pd.Timestamp(order_date),
        estimated_lead_time_days=float(lead_time),
        thresholds=thresholds,
    )
    prob, level = result["delay_probability"], result["risk_level"]
    color = RISK_COLORS[level]

    c1, c2 = st.columns([1, 2])
    c1.markdown(
        f'<div class="risk-banner" style="background:{color}">'
        f'<div style="font-size:.75rem;letter-spacing:.12em">DELAY PROBABILITY</div>'
        f'<div class="p">{prob:.1%}</div>'
        f'<div style="font-weight:700;margin-top:.35rem">{level} RISK</div></div>',
        unsafe_allow_html=True,
    )
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 30}},
        gauge={
            "axis": {"range": [0, max(40, high * 100 * 1.8)]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, med * 100], "color": "rgba(26,157,90,.18)"},
                {"range": [med * 100, high * 100], "color": "rgba(217,148,19,.22)"},
                {"range": [high * 100, max(40, high * 100 * 1.8)], "color": "rgba(209,57,74,.20)"},
            ],
        },
    ))
    gauge.update_layout(height=210, margin=dict(l=20, r=20, t=10, b=10),
                        template=PLOT_TEMPLATE)
    c2.plotly_chart(gauge, use_container_width=True)

    sup_row = seller_row.iloc[0] if len(seller_row) else None
    rec = R.recommend(level, prob, sup_row, order_value)

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section">Why this prediction (SHAP)</div>',
                    unsafe_allow_html=True)
        try:
            import explain as E

            factors = E.top_factors(result["features"], 6)
            contrib = pd.concat([factors["increasing"], factors["reducing"]])
            contrib = contrib.sort_values("shap_value")
            fig = px.bar(
                contrib, x="shap_value", y="feature", orientation="h",
                color=contrib["shap_value"] > 0,
                color_discrete_map={True: RISK_COLORS["HIGH"], False: RISK_COLORS["LOW"]},
            )
            fig.update_layout(template=PLOT_TEMPLATE, height=420, showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="SHAP contribution to delay risk",
                              yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            inc = factors["increasing"].head(4)
            red = factors["reducing"].head(4)
            cc1, cc2 = st.columns(2)
            cc1.markdown("**Pushing risk up**")
            for _, r_ in inc.iterrows():
                cc1.markdown(f"- ↑ {r_['feature']} *(value {r_['value']:,.2f})*")
            cc2.markdown("**Holding risk down**")
            for _, r_ in red.iterrows():
                cc2.markdown(f"- ↓ {r_['feature']} *(value {r_['value']:,.2f})*")
        except Exception as exc:  # pragma: no cover - explanation is best-effort
            st.warning(f"SHAP explanation unavailable: {exc}")

    with right:
        st.markdown('<div class="section">Recommended action</div>', unsafe_allow_html=True)
        st.markdown(risk_pill(level), unsafe_allow_html=True)
        st.markdown(f"### {rec['headline']}")
        for action in rec["actions"]:
            st.markdown(f"- {action}")
        if rec["sla_buffer_days"]:
            st.info(f"Suggested schedule buffer: **+{rec['sla_buffer_days']} days**")
        for note in rec["notes"]:
            st.caption(f"ℹ️ {note}")

        if sup_row is not None:
            st.markdown('<div class="section">Supplier context</div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": ["Supplier score", "Historical delay rate",
                                   "Avg delivery days", "Delivered orders", "Review score"],
                        "Value": [
                            f"{sup_row['supplier_score']:.1f}/100",
                            f"{sup_row['delay_rate']:.1%}",
                            f"{sup_row['avg_delivery_days']:.1f}",
                            f"{int(sup_row['order_volume']):,}",
                            f"{sup_row['review_score']:.2f}",
                        ],
                    }
                ),
                hide_index=True, use_container_width=True,
            )

    with st.expander("Model input row (feature vector sent to the model)"):
        vec = result["features"].T.rename(columns={0: "value"})
        vec["value"] = vec["value"].astype(str)  # mixed numeric/categorical column
        st.dataframe(vec, use_container_width=True)


def page_suppliers() -> None:
    st.markdown("## Supplier prioritization")
    st.caption(
        "Weighted scorecard combining historical reliability, model-predicted risk, "
        "delivery speed, freight performance and customer feedback."
    )

    st.markdown('<div class="section">Initial operational weighting</div>',
                unsafe_allow_html=True)
    cols = st.columns(5)
    weights = {}
    for col, (name, default) in zip(cols, SS.DEFAULT_WEIGHTS.items()):
        weights[name] = col.slider(name.replace("_", " ").title(), 0.0, 1.0, float(default), 0.05)
    st.caption(
        "These weights are an initial operational weighting, not a validated formula. "
        "Components are min-max normalised (1 = best) and re-normalised to sum to 1."
    )

    ranked = SS.score_suppliers(suppliers_raw, weights)

    f1, f2, f3 = st.columns([1, 1, 2])
    min_vol = f1.number_input("Minimum delivered orders", 1, 500,
                              SS.MIN_ORDERS_FOR_RANKING, 1)
    state_opts = ["All"] + sorted(ranked["seller_state"].dropna().unique().tolist())
    state = f2.selectbox("Seller state", state_opts)
    view_mode = f3.radio("View", ["Top performers", "Highest risk (watchlist)", "All"],
                         horizontal=True)

    view = ranked[ranked["order_volume"] >= min_vol]
    if state != "All":
        view = view[view["seller_state"] == state]
    if view_mode == "Top performers":
        view = view.sort_values("supplier_score", ascending=False).head(50)
    elif view_mode == "Highest risk (watchlist)":
        view = view.sort_values("supplier_score").head(50)

    c = st.columns(4)
    kpi(c[0], "Suppliers ranked", f"{len(ranked):,}", "all sellers in the history")
    kpi(c[1], "Meeting volume filter", f"{(ranked['order_volume'] >= min_vol).sum():,}",
        f"≥ {min_vol} delivered orders")
    kpi(c[2], "Watchlist tier", f"{(ranked['priority_tier'] == 'WATCHLIST').sum():,}",
        "score below 40/100")
    kpi(c[3], "Median delay rate", f"{ranked['delay_rate'].median():.1%}", "across suppliers")

    display = view[[
        "rank", "seller_id", "seller_state", "top_category", "order_volume",
        "on_time_rate", "delay_rate", "avg_delivery_days", "freight_ratio",
        "review_score", "predicted_risk", "supplier_score", "priority_tier",
        "low_volume_flag",
    ]].rename(columns={
        "rank": "Rank", "seller_id": "Seller", "seller_state": "State",
        "top_category": "Main category", "order_volume": "Orders",
        "on_time_rate": "On-time", "delay_rate": "Delay rate",
        "avg_delivery_days": "Avg days", "freight_ratio": "Freight/value",
        "review_score": "Review", "predicted_risk": "Predicted risk",
        "supplier_score": "Score", "priority_tier": "Tier",
        "low_volume_flag": "Thin history",
    })
    st.dataframe(
        display, hide_index=True, use_container_width=True, height=430,
        column_config={
            "On-time": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Delay rate": st.column_config.NumberColumn(format="%.2f"),
            "Predicted risk": st.column_config.NumberColumn(format="%.3f"),
            "Freight/value": st.column_config.NumberColumn(format="%.2f"),
            "Review": st.column_config.NumberColumn(format="%.2f"),
            "Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
        },
    )

    c1, c2 = st.columns(2)
    eligible = ranked[ranked["order_volume"] >= min_vol]
    fig = px.scatter(
        eligible, x="order_volume", y="delay_rate", size="avg_order_value",
        color="supplier_score", color_continuous_scale="RdYlGn",
        hover_data=["seller_id", "seller_state", "review_score"],
        labels={"order_volume": "Delivered orders", "delay_rate": "Historical delay rate"},
    )
    fig.update_layout(template=PLOT_TEMPLATE, height=380,
                      margin=dict(l=10, r=10, t=30, b=10),
                      title="Volume vs delay rate (bubble = avg order value)")
    fig.update_yaxes(tickformat=".0%")
    c1.plotly_chart(fig, use_container_width=True)

    worst = SS.high_risk_suppliers(eligible, 15)
    fig = px.bar(worst.sort_values("supplier_score"), x="supplier_score", y="seller_id",
                 orientation="h", color="delay_rate", color_continuous_scale="Reds",
                 labels={"supplier_score": "Supplier score", "seller_id": ""})
    fig.update_layout(template=PLOT_TEMPLATE, height=380,
                      margin=dict(l=10, r=10, t=30, b=10),
                      title="Highest-risk suppliers (lowest score, volume filtered)")
    c2.plotly_chart(fig, use_container_width=True)


def page_alerts() -> None:
    st.markdown("## Risk alerts")
    st.caption("Orders flagged at confirmation time, with the recommended playbook action.")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    levels = c1.multiselect("Risk level", P.RISK_ORDER, default=["HIGH"])
    cats = ["All"] + sorted(data["product_category"].dropna().unique().tolist())
    cat = c2.selectbox("Category", cats)
    states = ["All"] + sorted(data["customer_state"].dropna().unique().tolist())
    state = c3.selectbox("Customer state", states)
    top_n = c4.number_input("Rows", 20, 2000, 200, 20)

    view = data[data["risk_level"].isin(levels)] if levels else data
    if cat != "All":
        view = view[view["product_category"] == cat]
    if state != "All":
        view = view[view["customer_state"] == state]
    view = view.sort_values("delay_probability", ascending=False).head(int(top_n)).copy()
    view["recommended_action"] = R.recommend_batch(view)

    k = st.columns(4)
    kpi(k[0], "Alerts shown", f"{len(view):,}", "after filters")
    kpi(k[1], "Mean risk", f"{view['delay_probability'].mean():.1%}" if len(view) else "-",
        "predicted probability")
    kpi(k[2], "Actually late", f"{view['delivery_delay'].mean():.1%}" if len(view) else "-",
        "ground truth of the flagged set")
    base = data["delivery_delay"].mean()
    lift = (view["delivery_delay"].mean() / base) if len(view) and base else 0
    kpi(k[3], "Lift vs population", f"{lift:.2f}×", f"population delay rate {base:.1%}")

    st.dataframe(
        view[[
            "order_id", "primary_seller_id", "product_category", "customer_state",
            "order_purchase_timestamp", "estimated_lead_time_days", "total_order_value",
            "delay_probability", "risk_level", "recommended_action", "delivery_delay",
        ]].rename(columns={
            "order_id": "Order ID", "primary_seller_id": "Seller",
            "product_category": "Category", "customer_state": "Dest. state",
            "order_purchase_timestamp": "Purchased",
            "estimated_lead_time_days": "Promised days",
            "total_order_value": "Value", "delay_probability": "Delay probability",
            "risk_level": "Risk", "recommended_action": "Recommended action",
            "delivery_delay": "Was late",
        }),
        hide_index=True, use_container_width=True, height=460,
        column_config={
            "Delay probability": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=float(max(data["delay_probability"].max(), 0.3))
            ),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Promised days": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "‘Was late’ is the historical outcome, shown here only so reviewers can audit "
        "alert quality. It is never available to the model at prediction time."
    )
    st.download_button(
        "Download alert list (CSV)",
        view.to_csv(index=False).encode(),
        file_name="delayshield_alerts.csv",
        mime="text/csv",
    )


def page_analytics() -> None:
    st.markdown("## Delay analytics")
    st.caption(f"{scope} · {len(data):,} orders.")

    c1, c2 = st.columns(2)
    counts = data["delivery_delay"].value_counts().rename({0: "On time", 1: "Late"})
    fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                 color=counts.index,
                 color_discrete_map={"On time": RISK_COLORS["LOW"], "Late": RISK_COLORS["HIGH"]})
    fig.update_layout(template=PLOT_TEMPLATE, height=330, title="Late vs on time",
                      margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data, x="delay_probability", nbins=50, color="risk_level",
                       color_discrete_map=RISK_COLORS)
    fig.update_layout(template=PLOT_TEMPLATE, height=330,
                      title="Predicted delay probability distribution",
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Delay probability")
    c2.plotly_chart(fig, use_container_width=True)

    cat = (
        data.groupby("product_category")
        .agg(orders=("order_id", "count"), delay_rate=("delivery_delay", "mean"),
             predicted=("delay_probability", "mean"))
        .query("orders >= 50")
        .sort_values("delay_rate", ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(cat, x="delay_rate", y="product_category", orientation="h",
                 color="delay_rate", color_continuous_scale="Reds",
                 hover_data=["orders", "predicted"])
    fig.update_layout(template=PLOT_TEMPLATE, height=430,
                      title="Delay rate by product category (≥50 orders)",
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title="",
                      xaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    state = (
        data.groupby("customer_state")
        .agg(orders=("order_id", "count"), delay_rate=("delivery_delay", "mean"))
        .query("orders >= 30")
        .sort_values("delay_rate", ascending=False)
        .reset_index()
    )
    fig = px.bar(state, x="customer_state", y="delay_rate", color="delay_rate",
                 color_continuous_scale="Reds", hover_data=["orders"])
    fig.update_layout(template=PLOT_TEMPLATE, height=360,
                      title="Delay rate by destination state",
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(tickformat=".0%"), xaxis_title="")
    c1.plotly_chart(fig, use_container_width=True)

    seller = (
        data.groupby("primary_seller_id")
        .agg(orders=("order_id", "count"), delay_rate=("delivery_delay", "mean"))
        .query("orders >= 30")
        .sort_values("delay_rate", ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(seller, x="delay_rate", y="primary_seller_id", orientation="h",
                 color="delay_rate", color_continuous_scale="Reds", hover_data=["orders"])
    fig.update_layout(template=PLOT_TEMPLATE, height=360,
                      title="Worst sellers by delay rate (≥30 orders)",
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title="",
                      xaxis=dict(tickformat=".0%"))
    c2.plotly_chart(fig, use_container_width=True)

    monthly = (
        data.groupby("order_month")
        .agg(delay_rate=("delivery_delay", "mean"), orders=("order_id", "count"))
        .reset_index()
    )
    fig = px.line(monthly, x="order_month", y="delay_rate", markers=True,
                  hover_data=["orders"])
    fig.update_layout(template=PLOT_TEMPLATE, height=320, title="Delay rate by month",
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(tickformat=".0%"), xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Global feature importance (mean |SHAP| over sampled orders)"):
        try:
            imp = global_shap(400).head(18)
            fig = px.bar(imp.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                         orientation="h", color_discrete_sequence=[ACCENT])
            fig.update_layout(template=PLOT_TEMPLATE, height=520, yaxis_title="",
                              margin=dict(l=10, r=10, t=20, b=10),
                              xaxis_title="Mean |SHAP value|")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # pragma: no cover
            st.warning(f"SHAP importance unavailable: {exc}")


def page_simulation() -> None:
    st.markdown("## Policy simulation — reactive vs DelayShield")
    st.caption(
        "Same orders, same assumptions, two operating policies. Reactive acts only "
        "once an order is visibly late; DelayShield acts at order confirmation on "
        "everything above the alert threshold."
    )

    with st.expander("Cost and effectiveness assumptions (all editable)", expanded=True):
        c1, c2, c3 = st.columns(3)
        late_cost = c1.number_input("Cost of a late delivery", 0.0, 100000.0,
                                    float(S.DEFAULT_ASSUMPTIONS["cost_of_late_delivery"]), 50.0)
        false_cost = c2.number_input("Cost of a false alert", 0.0, 10000.0,
                                     float(S.DEFAULT_ASSUMPTIONS["cost_of_false_alert"]), 5.0)
        interv_cost = c3.number_input("Cost of an intervention", 0.0, 10000.0,
                                      float(S.DEFAULT_ASSUMPTIONS["cost_of_intervention"]), 5.0)
        c1, c2, c3 = st.columns(3)
        proactive_eff = c1.slider("Proactive intervention effectiveness", 0.0, 1.0,
                                  float(S.DEFAULT_ASSUMPTIONS["proactive_effectiveness"]), 0.05,
                                  help="Share of a true delay's damage removed by acting early.")
        reactive_eff = c2.slider("Reactive intervention effectiveness", 0.0, 1.0,
                                 float(S.DEFAULT_ASSUMPTIONS["reactive_effectiveness"]), 0.05)
        threshold = c3.slider("Alert threshold (predicted probability)", 0.02, 0.60,
                              float(meta["risk_thresholds"]["high"]), 0.005)
        st.markdown(
            '<div class="assumption"><b>ASSUMPTION</b> — Olist publishes no cost data. '
            "Every value above is a transparent operational assumption; the sensitivity "
            "panel below shows how the conclusion moves when they change.</div>",
            unsafe_allow_html=True,
        )

    a = {
        "cost_of_late_delivery": late_cost,
        "cost_of_false_alert": false_cost,
        "cost_of_intervention": interv_cost,
        "proactive_effectiveness": proactive_eff,
        "reactive_effectiveness": reactive_eff,
        "threshold": threshold,
    }
    res = S.compare(data["delivery_delay"], data["delay_probability"], a)
    table = res["table"]
    breakeven = S.break_even_precision(a)
    flagged = data[data["delay_probability"] >= threshold]
    precision = flagged["delivery_delay"].mean() if len(flagged) else 0.0

    c = st.columns(5)
    kpi(c[0], "Delays caught early", f"{res['delays_caught_early']:,}",
        f"of {int(table.iloc[0]['actual_late']):,} actual late orders")
    kpi(c[1], "False alerts", f"{res['false_alerts']:,}",
        f"alert precision {precision:.1%}")
    kpi(c[2], "Cost delta", money(res["cost_saving"]),
        f"{res['cost_saving_pct']:+.1f}% vs reactive")
    kpi(c[3], "Service gain", f"{res['service_gain_pp']:+.2f} pp",
        "on-time rate after policy")
    kpi(c[4], "Break-even precision", f"{breakeven:.1%}",
        "alerts must beat this to pay off")

    if precision >= breakeven:
        st.success(
            f"Under these assumptions the risk-aware policy pays off: alert precision "
            f"{precision:.1%} exceeds the {breakeven:.1%} break-even precision."
        )
    else:
        st.warning(
            f"Under these assumptions the risk-aware policy does **not** pay off at this "
            f"threshold: alert precision {precision:.1%} is below the {breakeven:.1%} "
            "break-even point. Raise the threshold or the late-delivery cost to see the "
            "band where it does."
        )

    st.markdown('<div class="section">Policy comparison</div>', unsafe_allow_html=True)
    show = table[[
        "policy", "orders", "actual_late", "flagged_before_delivery", "true_alerts",
        "false_alerts", "missed_delays", "interventions", "residual_late_orders",
        "late_cost", "false_alert_cost", "intervention_cost", "total_cost",
        "on_time_rate_after_policy",
    ]].rename(columns={
        "policy": "Policy", "orders": "Orders", "actual_late": "Actual late",
        "flagged_before_delivery": "Flagged early", "true_alerts": "True alerts",
        "false_alerts": "False alerts", "missed_delays": "Missed delays",
        "interventions": "Interventions", "residual_late_orders": "Residual late",
        "late_cost": "Late cost", "false_alert_cost": "False-alert cost",
        "intervention_cost": "Intervention cost", "total_cost": "Total cost",
        "on_time_rate_after_policy": "On-time after policy",
    })
    st.dataframe(show.round(2), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    stacked = pd.DataFrame({
        "Policy": ["Reactive", "DelayShield"] * 3,
        "Component": ["Late cost"] * 2 + ["False-alert cost"] * 2 + ["Intervention cost"] * 2,
        "Cost": list(table["late_cost"]) + list(table["false_alert_cost"])
                + list(table["intervention_cost"]),
    })
    fig = px.bar(stacked, x="Policy", y="Cost", color="Component", barmode="stack",
                 color_discrete_sequence=[RISK_COLORS["HIGH"], RISK_COLORS["MEDIUM"], ACCENT])
    fig.update_layout(template=PLOT_TEMPLATE, height=380, title="Total policy cost breakdown",
                      margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig, use_container_width=True)

    sweep = S.threshold_sweep(data["delivery_delay"], data["delay_probability"], a,
                              grid=np.arange(0.04, float(data["delay_probability"].max()), 0.01))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["saving_vs_reactive"],
                             name="Saving vs reactive", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall",
                             yaxis="y2", line=dict(color=RISK_COLORS["HIGH"], dash="dot")))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision",
                             yaxis="y2", line=dict(color=RISK_COLORS["LOW"], dash="dash")))
    fig.add_vline(x=threshold, line_dash="dot", line_color="#64748b")
    fig.update_layout(template=PLOT_TEMPLATE, height=380,
                      title="Where the policy pays off (threshold sweep)",
                      margin=dict(l=10, r=10, t=90, b=10),
                      yaxis=dict(title="Saving"),
                      yaxis2=dict(overlaying="y", side="right", title="Rate",
                                  tickformat=".0%", showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="center", x=0.5))
    c2.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Sensitivity analysis</div>', unsafe_allow_html=True)
    param = st.selectbox("Vary one assumption",
                         ["cost_of_late_delivery", "cost_of_false_alert",
                          "cost_of_intervention", "proactive_effectiveness"])
    sens = S.sensitivity_analysis(data["delivery_delay"], data["delay_probability"], a, param)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(sens.round(2), hide_index=True, use_container_width=True)
    fig = px.line(sens, x=param, y=["reactive_cost", "delayshield_cost"], markers=True,
                  color_discrete_sequence=[RISK_COLORS["HIGH"], ACCENT])
    fig.update_layout(template=PLOT_TEMPLATE, height=340,
                      title=f"Total cost as {param.replace('_', ' ')} varies",
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title="Total cost")
    c2.plotly_chart(fig, use_container_width=True)


def page_model() -> None:
    st.markdown("## Model, method and honest limitations")

    split = meta["split"]
    c = st.columns(4)
    kpi(c[0], "Selected model", meta["selected_model"].replace("_", " "),
        meta["selection_criterion"])
    kpi(c[1], "Test ROC-AUC", f"{meta['test_metrics_at_f1_threshold']['roc_auc']:.4f}",
        f"{split['test_rows']:,} held-out orders")
    kpi(c[2], "Test PR-AUC", f"{meta['test_metrics_at_f1_threshold']['pr_auc']:.4f}",
        f"base rate {split['test_positive_rate']:.2%}")
    kpi(c[3], "Test recall / precision",
        f"{meta['test_metrics_at_f1_threshold']['recall']:.2f}"
        f" / {meta['test_metrics_at_f1_threshold']['precision']:.2f}",
        f"threshold {meta['threshold_f1']}")

    st.markdown('<div class="section">Chronological split</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([
            {"Split": "Train", "Rows": split["train_rows"],
             "From": split["train_range"][0][:10], "To": split["train_range"][1][:10],
             "Late rate": split["train_positive_rate"]},
            {"Split": "Validation", "Rows": split["val_rows"],
             "From": split["val_range"][0][:10], "To": split["val_range"][1][:10],
             "Late rate": split["val_positive_rate"]},
            {"Split": "Test", "Rows": split["test_rows"],
             "From": split["test_range"][0][:10], "To": split["test_range"][1][:10],
             "Late rate": split["test_positive_rate"]},
        ]),
        hide_index=True, use_container_width=True,
        column_config={"Late rate": st.column_config.NumberColumn(format="percent")},
    )
    st.caption(
        "A random split would let the model learn from future orders and from the same "
        "seller's later performance, which is impossible at prediction time. The "
        "validation window (Jan-Mar 2018) has a much higher delay rate than the test "
        "window - that regime shift is real and it is why the test metrics are lower "
        "than the validation metrics."
    )

    st.markdown('<div class="section">Model comparison</div>', unsafe_allow_html=True)
    comp = load_comparison()
    if not comp.empty:
        st.dataframe(
            comp[["model", "split", "roc_auc", "pr_auc", "precision", "recall", "f1",
                  "brier", "threshold"]].round(4),
            hide_index=True, use_container_width=True,
        )
    st.caption(
        "Each family is trained twice: with balanced class weights and without "
        "(imbalance handled by threshold tuning only). SMOTE was deliberately not used - "
        "synthesising minority rows across a temporally ordered dataset mixes information "
        "between time periods."
    )

    c1, c2 = st.columns(2)
    cal = meta.get("calibration_test", {})
    if cal:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["prob_pred"], y=cal["prob_true"], mode="lines+markers",
                                 name="Model", line=dict(color=ACCENT, width=3)))
        fig.add_trace(go.Scatter(x=[0, max(cal["prob_pred"])], y=[0, max(cal["prob_pred"])],
                                 mode="lines", name="Perfect calibration",
                                 line=dict(color="#94a3b8", dash="dash")))
        fig.update_layout(template=PLOT_TEMPLATE, height=360,
                          title="Calibration on the test period (quantile bins)",
                          xaxis_title="Mean predicted probability",
                          yaxis_title="Observed delay rate",
                          margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(fig, use_container_width=True)

    cm = meta["test_metrics_at_f1_threshold"]["confusion_matrix"]
    fig = px.imshow(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        x=["Predicted on-time", "Predicted late"],
        y=["Actually on-time", "Actually late"],
        text_auto=True, color_continuous_scale="Blues",
    )
    fig.update_layout(template=PLOT_TEMPLATE, height=360,
                      title=f"Test confusion matrix @ threshold {meta['threshold_f1']}",
                      margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Leakage policy</div>', unsafe_allow_html=True)
    st.info(meta["leakage_policy"])

    st.markdown('<div class="section">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
- **Ranking, not certainty.** Test ROC-AUC ≈ 0.73 means the model orders risk usefully,
  but individual predictions carry wide uncertainty. Alert precision at the operating
  threshold is far below 50%, so alerts are a triage queue, not a verdict.
- **Regime shift.** The 2018 Brazilian carrier disruption changes the delay base rate
  between windows; thresholds tuned on validation do not transfer perfectly to the test
  window, and the simulation page shows exactly where that hurts.
- **Costs are assumptions.** Olist publishes no cost or intervention-effectiveness data.
  Every currency figure in this app is configurable and labelled as an assumption.
- **Intervention effect is assumed, not measured.** Proving it needs a controlled
  experiment (A/B on flagged orders), which historical data cannot supply.
- **Single-snapshot history.** The prediction form uses each seller's latest historical
  statistics; a production service would recompute them continuously from a feature store.
"""
    )


PAGES = {
    "Dashboard": page_dashboard,
    "Order Risk Prediction": page_prediction,
    "Supplier Ranking": page_suppliers,
    "Risk Alerts": page_alerts,
    "Analytics": page_analytics,
    "Policy Simulation": page_simulation,
    "Model & Method": page_model,
}
PAGES[page]()

st.markdown(
    '<hr style="margin-top:2rem;opacity:.25">'
    '<div style="font-size:.75rem;color:#64748b">DelayShield · Delivery Risk Intelligence · '
    "Built on the Brazilian E-Commerce Public Dataset by Olist · "
    "Predict → Explain → Prioritize → Recommend → Simulate → Decide</div>",
    unsafe_allow_html=True,
)
