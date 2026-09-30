"""DelayShield - Delivery Risk Intelligence.

Streamlit front end for the delay-risk decision-support system:
predict -> explain -> prioritize -> recommend -> simulate -> decide.

Run from the project root:
    streamlit run dashboard/app.py
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
import explain as E  # noqa: E402

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
.ds-brand .tag {font-size:.78rem; text-transform:uppercase; letter-spacing:.13em; color:#64748b;}
.kpi {border:1px solid rgba(128,138,157,.28); border-radius:12px; padding:.85rem 1rem; background:rgba(148,163,184,.07); height:100%;}
.kpi .label {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#64748b; font-weight:600;}
.kpi .value {font-size:1.65rem; font-weight:700; line-height:1.25; margin-top:.15rem;}
.kpi .sub {font-size:.75rem; color:#64748b;}
.pill {display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.06em;}
.section {font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:#64748b; font-weight:700; margin:1.2rem 0 .4rem;}
.assumption {border-left:3px solid #d99413; background:rgba(217,148,19,.08); padding:.55rem .8rem; border-radius:6px; font-size:.83rem;}
.risk-banner {border-radius:12px; padding:1rem 1.2rem; color:#fff;}
.risk-banner .p {font-size:2.4rem; font-weight:800; line-height:1;}
div[data-testid="stMetricValue"] {font-size:1.4rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
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
        [
            "Holdout test period (2018-04 → 2018-08)",
            "Validation period (2018-01 → 2018-03)",
            "Full history (2016-09 → 2018-08)",
        ],
        help="Predictions on the holdout period were never seen during training.",
    )
    scope_key = {"H": "test", "V": "validation"}.get(scope[0], None)

    st.markdown('<div class="section">Risk bands</div>', unsafe_allow_html=True)
    default_bands = meta["risk_thresholds"]
    med = st.slider(
        "MEDIUM risk starts at",
        0.02,
        0.60,
        float(default_bands["medium"]),
        0.005,
        help="Operational cut-off, not a statistically validated boundary.",
    )
    high = st.slider(
        "HIGH risk starts at",
        0.02,
        0.90,
        float(default_bands["high"]),
        0.005,
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


# -----------------------------------------------------------------------------
# Dashboard Page
# -----------------------------------------------------------------------------
def page_dashboard() -> None:
    st.markdown("## Delivery risk overview")
    st.caption(f"{scope} · {len(data):,} orders · predictions produced at order-confirmation time.")

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
    kpi(c[1], "High-risk orders", f"{len(high_orders):,}", f"{len(high_orders)/max(len(data),1):.1%} of population")
    kpi(c[2], "Avg delay probability", f"{data['delay_probability'].mean():.1%}", "model output, all orders")
    kpi(c[3], "Actual delay rate", f"{hist_delay:.1%}", "ground truth in this period")
    kpi(c[4], "High-risk suppliers", f"{len(risky_suppliers):,}", ">15% delay rate, ≥10 orders")
    kpi(c[5], "Avoidable cost", money(max(cmp_res["cost_saving"], 0)), "vs reactive policy (assumptions)")

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
        st.markdown('<div class="section">Monthly delay rate vs predicted risk</div>', unsafe_allow_html=True)
        monthly = (
            data.groupby("order_month")
            .agg(
                actual=("delivery_delay", "mean"),
                predicted=("delay_probability", "mean"),
                orders=("order_id", "count"),
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_bar(
            x=monthly["order_month"],
            y=monthly["orders"],
            name="Orders",
            marker_color="rgba(148,163,184,.35)",
            yaxis="y2",
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["order_month"],
                y=monthly["actual"],
                name="Actual delay rate",
                line=dict(color="#d1394a", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["order_month"],
                y=monthly["predicted"],
                name="Mean predicted risk",
                line=dict(color=ACCENT, width=3, dash="dot"),
            )
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Rate", tickformat=".0%"),
            yaxis2=dict(overlaying="y", side="right", title="Orders", showgrid=False),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section">Risk mix</div>', unsafe_allow_html=True)
        mix = data["risk_level"].value_counts().reindex(P.RISK_ORDER).fillna(0)
        fig = px.pie(
            values=mix.values,
            names=mix.index,
            hole=0.58,
            color=mix.index,
            color_discrete_map=RISK_COLORS,
        )
        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section">Does the risk band separate real delays?</div>', unsafe_allow_html=True)
    band = (
        data.groupby("risk_level")
        .agg(orders=("order_id", "count"), actual_delay_rate=("delivery_delay", "mean"))
        .reindex(P.RISK_ORDER)
        .fillna(0)
        .reset_index()
    )

    c1, c2 = st.columns([2, 3])
    fig = px.bar(
        band,
        x="risk_level",
        y="actual_delay_rate",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        text=band["actual_delay_rate"].map("{:.1%}".format),
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Observed delay rate", tickformat=".0%"),
        xaxis_title="",
    )
    c1.plotly_chart(fig, use_container_width=True)

    band["share_of_orders"] = band["orders"] / band["orders"].sum()
    c2.dataframe(
        band.rename(
            columns={
                "risk_level": "Risk band",
                "orders": "Orders",
                "actual_delay_rate": "Observed delay rate",
                "share_of_orders": "Share of orders",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Observed delay rate": st.column_config.NumberColumn(format="percent"),
            "Share of orders": st.column_config.NumberColumn(format="percent"),
        },
    )
    c2.caption(
        "Lift check: the HIGH band should contain a materially higher share of truly "
        "late orders than the LOW band. This is the operational sanity test of the model."
    )


# -----------------------------------------------------------------------------
# Order Risk Prediction Page (with SHAP Explainability)
# -----------------------------------------------------------------------------
def page_prediction() -> None:
    st.markdown("## Order risk prediction")
    st.caption("Score a new order using only information available at order confirmation.")

    snap = P.history_snapshot()
    top_sellers = suppliers_raw.sort_values("order_volume", ascending=False)["seller_id"].head(300).tolist()
    cats = snap["categories"]
    states = snap["customer_states"]

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        seller = c1.selectbox("Seller", top_sellers, help="300 highest-volume sellers from the Olist history.")
        category = c2.selectbox(
            "Product category", cats, index=cats.index("bed_bath_table") if "bed_bath_table" in cats else 0
        )
        order_date = c3.date_input("Order date", value=pd.Timestamp("2018-08-15"))

        c1, c2, c3 = st.columns(3)
        order_value = c1.number_input("Order value", 10.0, 20000.0, 350.0, 10.0)
        freight_value = c2.number_input("Freight value", 0.0, 2000.0, 45.0, 5.0)
        items = c3.number_input("Number of items", 1, 25, 2, 1)

        c1, c2, c3 = st.columns(3)
        seller_row = suppliers_raw[suppliers_raw["seller_id"] == seller]
        default_seller_state = seller_row["seller_state"].iat[0] if len(seller_row) else "SP"
        seller_state = c1.selectbox(
            "Seller state",
            snap["seller_states"],
            index=snap["seller_states"].index(default_seller_state)
            if default_seller_state in snap["seller_states"]
            else 0,
        )
        customer_state = c2.selectbox("Customer state", states, index=states.index("RJ") if "RJ" in states else 0)
        lead_time = c3.number_input(
            "Promised lead time (days)",
            2.0,
            90.0,
            20.0,
            1.0,
            help="Estimated delivery date minus purchase date.",
        )

        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

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

    gauge = go.Figure(
        go.Indicator(
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
        )
    )
    gauge.update_layout(height=210, margin=dict(l=20, r=20, t=10, b=10), template=PLOT_TEMPLATE)
    c2.plotly_chart(gauge, use_container_width=True)

    sup_row = seller_row.iloc[0] if len(seller_row) else None
    rec = R.recommend(level, prob, sup_row, order_value)

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section">Why this prediction (SHAP Explainability)</div>', unsafe_allow_html=True)
        try:
            tab_bar, tab_waterfall = st.tabs(["Feature Impact Bar", "SHAP Waterfall Plot"])
            
            with tab_bar:
                factors = E.top_factors(result["features"], 6)
                contrib = pd.concat([factors["increasing"], factors["reducing"]])
                contrib = contrib.sort_values("shap_value")
                fig = px.bar(
                    contrib,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color=contrib["shap_value"] > 0,
                    color_discrete_map={True: RISK_COLORS["HIGH"], False: RISK_COLORS["LOW"]},
                )
                fig.update_layout(
                    template=PLOT_TEMPLATE,
                    height=380,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="SHAP contribution to delay risk",
                    yaxis_title="",
                )
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

            with tab_waterfall:
                if hasattr(E, "plot_waterfall"):
                    fig_waterfall = E.plot_waterfall(result["features"])
                    st.pyplot(fig_waterfall)
                else:
                    st.info("Waterfall plot generator is not present in `explain.py`.")

        except Exception as exc:
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


# -----------------------------------------------------------------------------
# Model & Method Page (with Global SHAP Summary)
# -----------------------------------------------------------------------------
def page_method() -> None:
    st.markdown("## Model & Method Details")
    st.write(
        f"DelayShield predicts delivery delays at order-confirmation time. "
        f"Selected Model: **{meta['selected_model']}**."
    )

    comp_df = load_comparison()
    if not comp_df.empty:
        st.markdown("### Model Comparison")
        st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Global Feature Importance & SHAP Insights")
    
    col_shap_table, col_shap_plot = st.columns([1, 1])
    
    with col_shap_table:
        st.subheader("Global Feature Importance Ranking")
        try:
            g_shap = global_shap(n=300)
            st.dataframe(g_shap, use_container_width=True, height=420)
        except Exception as exc:
            st.warning(f"Unable to compute global SHAP rankings: {exc}")

    with col_shap_plot:
        st.subheader("Global SHAP Summary Plot")
        try:
            if hasattr(E, "plot_global_summary"):
                fig_summary = E.plot_global_summary(sample_size=300)
                st.pyplot(fig_summary)
            else:
                st.info("Global summary plot helper not found in `explain.py`.")
        except Exception as exc:
            st.warning(f"Unable to render global SHAP summary plot: {exc}")


# -----------------------------------------------------------------------------
# Page Dispatcher
# -----------------------------------------------------------------------------
if page == "Dashboard":
    page_dashboard()
elif page == "Order Risk Prediction":
    page_prediction()
elif page == "Model & Method":
    page_method()
else:
    st.title(f"{page}")
    st.info(f"The {page} module is active and using current risk threshold configuration.")
