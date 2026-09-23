"""Streamlit Dashboard for Olist Customer Churn Prediction & Risk Scoring.

Professional, internship-ready decision support application built with
Streamlit, Plotly, Pandas, and Scikit-Learn.
"""

import sys
from pathlib import Path
import json

# Ensure project root is on sys.path for direct 'streamlit run app/app.py' execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    SNAPSHOT_DATE,
    TARGET_START_DATE,
    TARGET_END_DATE,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    TIER_HIGH,
    TIER_MEDIUM,
    TIER_LOW,
    METRICS_PATH,
    SCORED_CUSTOMERS_PATH,
    HIGH_RISK_EXPORT_PATH,
    MODEL_PIPELINE_PATH,
)
from src.model import load_artifacts

# Page Setup
st.set_page_config(
    page_title="Olist Customer Churn & Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .info-callout {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .risk-high {
        color: #DC2626;
        font-weight: 700;
    }
    .risk-med {
        color: #D97706;
        font-weight: 700;
    }
    .risk-low {
        color: #16A34A;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_app_data():
    """Load cached model artifacts, metrics, and scored customer dataset."""
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    scored_df = pd.read_parquet(SCORED_CUSTOMERS_PATH)
    artifacts = load_artifacts(MODEL_PIPELINE_PATH)
    return metrics, scored_df, artifacts["importance_df"]


metrics, scored_df, importance_df = load_app_data()

# Sidebar Navigation & Methodology Controls
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/4b/Olist_Logo.png",
    width=140,
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View:",
    [
        "1. Executive Overview & Risk Tiers",
        "2. Model Evaluation & Dual PR-AUC",
        "3. Feature Drivers & Interpretability",
        "4. Customer Explorer & Action Center",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Methodology Specifications")
st.sidebar.markdown(
    f"""
    - **Snapshot Date:** `{SNAPSHOT_DATE.strftime('%Y-%m-%d')}`
    - **Target Horizon:** `180 days`
    - **Window:** `2018-03-01` to `2018-08-27`
    - **Customer Key:** `customer_unique_id`
    - **High-Risk Threshold:** `> {int(HIGH_RISK_THRESHOLD*100)}%`
    - **Medium-Risk:** `{int(LOW_RISK_THRESHOLD*100)}% - {int(HIGH_RISK_THRESHOLD*100)}%`
    - **Low-Risk:** `< {int(LOW_RISK_THRESHOLD*100)}%`
    """
)

st.sidebar.info(
    "**Methodological Architecture:**\n\n"
    "• Operational Risk Score uses balanced logistic weights for contrast.\n\n"
    "• Calibrated Probability models true empirical 98.8% base rate.\n\n"
    "• Zero target lookahead leakage across all features."
)


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW & RISK TIERS
# ==============================================================================
if page == "1. Executive Overview & Risk Tiers":
    st.markdown('<div class="main-header">Olist Churn Prediction & Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Executive retention intelligence, behavioral risk tiers, and population base-rate telemetry.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Banner
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Eligible Cohort", f"{len(scored_df):,}", "As of 2018-02-28")
    with kpi2:
        st.metric("Population Churn Rate", "98.82%", "54,872 Churned")
    with kpi3:
        st.metric("Active Repurchase Rate", "1.18%", "653 Active")
    with kpi4:
        st.metric(
            "Model Discrimination",
            f"{metrics['discrimination']['roc_auc_operational']:.3f} AUC",
            "Stratified Test",
        )
    with kpi5:
        st.metric(
            "Minority Active Lift",
            f"{metrics['dual_pr_auc']['active_class_0']['relative_lift_over_baseline']:.2f}x",
            "Over 1.18% Baseline",
        )

    st.markdown(
        """
        <div class="info-callout">
        <b>Crucial Distinction: Operational Risk Score vs. Calibrated Churn Probability</b><br>
        Because general e-commerce in Olist exhibits a 97% one-time buyer pattern, empirical churn is 98.8%. 
        A raw calibrated probability hovers around ~99% for nearly everyone. 
        To make risk scores operationally useful for CRM retention, the <b>Operational Risk Score</b> applies balanced class weighting to expand dynamic contrast across 0–100%. 
        Fixed risk tiers (High &gt; 70%, Medium 30–70%, Low &lt; 30%) are assigned directly to the Operational Risk Score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk Tier Breakdown Cards
    high_count = (scored_df["business_risk_tier"] == TIER_HIGH).sum()
    med_count = (scored_df["business_risk_tier"] == TIER_MEDIUM).sum()
    low_count = (scored_df["business_risk_tier"] == TIER_LOW).sum()

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid #DC2626;">
                <h4 style="margin:0; color:#DC2626;">🚨 High Risk Tier (> 70%)</h4>
                <h2 style="margin:8px 0 4px 0;">{high_count:,} <span style="font-size:1rem; color:#64748B;">({high_count/len(scored_df):.2%})</span></h2>
                <p style="margin:0; font-size:0.9rem; color:#475569;">
                    <b>Avg Recency:</b> {scored_df[scored_df['business_risk_tier']==TIER_HIGH]['recency_days'].mean():.1f} days<br>
                    <b>Avg Historical Spend:</b> R$ {scored_df[scored_df['business_risk_tier']==TIER_HIGH]['total_spend'].mean():.2f}<br>
                    <b>Target Strategy:</b> Aggressive win-back promotions & VIP outreach.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid #D97706;">
                <h4 style="margin:0; color:#D97706;">⚠️ Medium Risk Tier (30% - 70%)</h4>
                <h2 style="margin:8px 0 4px 0;">{med_count:,} <span style="font-size:1rem; color:#64748B;">({med_count/len(scored_df):.2%})</span></h2>
                <p style="margin:0; font-size:0.9rem; color:#475569;">
                    <b>Avg Recency:</b> {scored_df[scored_df['business_risk_tier']==TIER_MEDIUM]['recency_days'].mean():.1f} days<br>
                    <b>Avg Historical Spend:</b> R$ {scored_df[scored_df['business_risk_tier']==TIER_MEDIUM]['total_spend'].mean():.2f}<br>
                    <b>Target Strategy:</b> Automated cross-sell email flows & seasonal category triggers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t3:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid #16A34A;">
                <h4 style="margin:0; color:#16A34A;">✅ Low Risk / Loyal Tier (< 30%)</h4>
                <h2 style="margin:8px 0 4px 0;">{low_count:,} <span style="font-size:1rem; color:#64748B;">({low_count/len(scored_df):.2%})</span></h2>
                <p style="margin:0; font-size:0.9rem; color:#475569;">
                    <b>Avg Recency:</b> {scored_df[scored_df['business_risk_tier']==TIER_LOW]['recency_days'].mean():.1f} days<br>
                    <b>Avg Historical Spend:</b> R$ {scored_df[scored_df['business_risk_tier']==TIER_LOW]['total_spend'].mean():.2f}<br>
                    <b>Target Strategy:</b> Loyalty rewards, review requests, and zero-discount nurture.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Behavioral Contrasts Across Risk Tiers")
    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        # Tier distribution pie
        tier_pie = px.pie(
            names=[TIER_HIGH, TIER_MEDIUM, TIER_LOW],
            values=[high_count, med_count, low_count],
            color=[TIER_HIGH, TIER_MEDIUM, TIER_LOW],
            color_discrete_map={
                TIER_HIGH: "#DC2626",
                TIER_MEDIUM: "#F59E0B",
                TIER_LOW: "#10B981",
            },
            title="Customer Base Segmentation by Business Risk Tier",
            hole=0.45,
        )
        tier_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(tier_pie, use_container_width=True)

    with chart_c2:
        # Recency vs Spend comparison
        tier_summary = scored_df.groupby("business_risk_tier").agg(
            avg_recency=("recency_days", "mean"),
            avg_spend=("total_spend", "mean"),
            avg_review=("avg_review_score", "mean"),
        ).reindex([TIER_LOW, TIER_MEDIUM, TIER_HIGH]).reset_index()

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=tier_summary["business_risk_tier"],
                y=tier_summary["avg_recency"],
                name="Avg Recency (Days)",
                marker_color="#3B82F6",
            )
        )
        fig_bar.update_layout(
            title="Average Inactivity (Days Since Last Order) by Tier",
            xaxis_title="Risk Tier",
            yaxis_title="Days",
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# PAGE 2: MODEL EVALUATION & DUAL PR-AUC
# ==============================================================================
elif page == "2. Model Evaluation & Dual PR-AUC":
    st.markdown('<div class="main-header">Model Performance & Dual PR-AUC Laboratory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Rigorous statistical evaluation under extreme class imbalance (98.8% Churn vs. 1.18% Active).</div>',
        unsafe_allow_html=True,
    )

    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
    with ev_col1:
        st.metric("Test ROC-AUC", f"{metrics['discrimination']['roc_auc_operational']:.4f}")
    with ev_col2:
        st.metric(
            "PR-AUC (Churn=1)",
            f"{metrics['dual_pr_auc']['churn_class_1']['pr_auc']:.4f}",
            f"Baseline: {metrics['dual_pr_auc']['churn_class_1']['baseline_prevalence']:.4f}",
        )
    with ev_col3:
        st.metric(
            "PR-AUC (Active=0)",
            f"{metrics['dual_pr_auc']['active_class_0']['pr_auc']:.4f}",
            f"Baseline: {metrics['dual_pr_auc']['active_class_0']['baseline_prevalence']:.4f}",
        )
    with ev_col4:
        st.metric("Brier Score", f"{metrics['calibration']['brier_score_calibrated']:.6f}", "Calibrated (Lower=Better)")

    tab_roc, tab_pr, tab_cm, tab_decile = st.tabs(
        ["ROC Curve", "Dual PR-AUC Curves", "Confusion Matrix", "Decile Lift & Cumulative Gains"]
    )

    with tab_roc:
        fpr = metrics["discrimination"]["roc_curve"]["fpr"]
        tpr = metrics["discrimination"]["roc_curve"]["tpr"]
        auc_val = metrics["discrimination"]["roc_auc_operational"]

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Logistic Regression (AUC = {auc_val:.3f})", line=dict(color="#2563EB", width=3))
        )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance", line=dict(dash="dash", color="#94A3B8"))
        )
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            height=450,
            margin=dict(t=40, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab_pr:
        pr_c1, pr_c2 = st.columns(2)
        with pr_c1:
            # Majority Churn
            ch_data = metrics["dual_pr_auc"]["churn_class_1"]
            fig_pr_ch = go.Figure()
            fig_pr_ch.add_trace(
                go.Scatter(
                    x=ch_data["pr_curve"]["recall"],
                    y=ch_data["pr_curve"]["precision"],
                    mode="lines",
                    name=f"Model PR (AUC = {ch_data['pr_auc']:.4f})",
                    line=dict(color="#DC2626", width=3),
                )
            )
            fig_pr_ch.add_hline(
                y=ch_data["baseline_prevalence"],
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text=f"Baseline Prevalence ({ch_data['baseline_prevalence']:.4f})",
            )
            fig_pr_ch.update_layout(
                title="PR-AUC: Majority Churn Class (pos_label=1)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                yaxis_range=[0.95, 1.005],
                height=420,
            )
            st.plotly_chart(fig_pr_ch, use_container_width=True)
            st.caption(
                "Notice that PR-AUC for churn=1 appears very close to 1.0 simply because the baseline prevalence is 98.8%. It must be interpreted against this high prior."
            )

        with pr_c2:
            # Minority Active
            act_data = metrics["dual_pr_auc"]["active_class_0"]
            fig_pr_act = go.Figure()
            fig_pr_act.add_trace(
                go.Scatter(
                    x=act_data["pr_curve"]["recall"],
                    y=act_data["pr_curve"]["precision"],
                    mode="lines",
                    name=f"Model PR (AUC = {act_data['pr_auc']:.4f})",
                    line=dict(color="#10B981", width=3),
                )
            )
            fig_pr_act.add_hline(
                y=act_data["baseline_prevalence"],
                line_dash="dash",
                line_color="#94A3B8",
                annotation_text=f"Baseline Active ({act_data['baseline_prevalence']:.4f})",
            )
            fig_pr_act.update_layout(
                title=f"PR-AUC: Minority Active Class ({act_data['relative_lift_over_baseline']:.2f}x Lift)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=420,
            )
            st.plotly_chart(fig_pr_act, use_container_width=True)
            st.caption(
                f"For the 1.18% active minority, the model achieves a **{act_data['relative_lift_over_baseline']:.2f}x lift** in precision over blind outreach."
            )

    with tab_cm:
        cm = metrics["confusion_matrix"]
        cm_matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

        cm_fig = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=["Active (0)", "Churn (1)"],
            y=["Active (0)", "Churn (1)"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix at Default Decision Threshold (0.50)",
        )
        cm_fig.update_layout(height=400)
        st.plotly_chart(cm_fig, use_container_width=True)

        cr_df = pd.DataFrame(metrics["classification_report"]).transpose()
        st.markdown("#### Full Classification Report")
        st.dataframe(cr_df.style.format(precision=3), use_container_width=True)

    with tab_decile:
        decile_df = pd.DataFrame(metrics["decile_lift"])
        fig_lift = px.bar(
            decile_df,
            x="decile",
            y="lift",
            title="Decile Lift: Active Customer Concentration (Decile 1 = Top Propensity)",
            labels={"decile": "Decile Group", "lift": "Lift Factor vs Baseline (1.18%)"},
            color="lift",
            color_continuous_scale="Teal",
        )
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Baseline (1.0x)")
        st.plotly_chart(fig_lift, use_container_width=True)

        st.markdown("#### Detailed Decile Performance Table")
        st.dataframe(
            decile_df[[
                "decile",
                "customers",
                "active_count",
                "active_rate",
                "cum_active_capture_pct",
                "lift",
                "cum_lift",
            ]].style.format(
                {
                    "customers": "{:,}",
                    "active_count": "{:,}",
                    "active_rate": "{:.3%}",
                    "cum_active_capture_pct": "{:.1f}%",
                    "lift": "{:.2f}x",
                    "cum_lift": "{:.2f}x",
                }
            ),
            use_container_width=True,
        )


# ==============================================================================
# PAGE 3: FEATURE DRIVERS & INTERPRETABILITY
# ==============================================================================
elif page == "3. Feature Drivers & Interpretability":
    st.markdown('<div class="main-header">Feature Drivers & Behavioral Coefficients</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Transparent Logistic Regression Odds Ratios and churn risk sensitivities.</div>',
        unsafe_allow_html=True,
    )

    col_desc, col_plot = st.columns([1, 2])
    with col_desc:
        st.markdown(
            """
            #### How to Interpret Odds Ratios:
            - **Odds Ratio > 1.0 (Red):** Increasing this feature multiplies the odds of customer churn.
            - **Odds Ratio < 1.0 (Green):** Increasing this feature reduces churn odds (promotes customer retention).
            - **Baseline Value (1.0):** Neutral effect on churn probability.
            
            **Top Insights:**
            1. **Recency is King:** Every standard deviation increase in days since last purchase more than doubles churn odds ($\text{OR} \approx 2.08$).
            2. **Tenure & Relationship:** Customers with longer tenures demonstrate established loyalty ($\text{OR} \approx 0.58$).
            3. **Review Experience:** High review scores protect retention ($\text{OR} \approx 0.79$).
            4. **Geography:** Orders from São Paulo (SP) enjoy lower churn odds due to faster delivery transit times.
            """
        )

    with col_plot:
        # Plot Odds Ratios
        sorted_imp = importance_df.sort_values(by="odds_ratio", ascending=True)
        colors = ["#10B981" if or_val < 1.0 else "#DC2626" for or_val in sorted_imp["odds_ratio"]]

        fig_coef = go.Figure()
        fig_coef.add_trace(
            go.Bar(
                y=sorted_imp["feature"],
                x=sorted_imp["odds_ratio"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig_coef.add_vline(x=1.0, line_dash="dash", line_color="#475569", annotation_text="Neutral Odds (1.0)")
        fig_coef.update_layout(
            title="Logistic Regression Odds Ratios (e^β)",
            xaxis_title="Odds Ratio (Multiplier of Churn Odds)",
            height=650,
            margin=dict(t=40, b=40, l=150, r=40),
        )
        st.plotly_chart(fig_coef, use_container_width=True)


# ==============================================================================
# PAGE 4: CUSTOMER EXPLORER & ACTION CENTER
# ==============================================================================
elif page == "4. Customer Explorer & Action Center":
    st.markdown('<div class="main-header">Customer Risk Explorer & Action Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Search individual customer risk profiles, inspect behavioral telemetry, and export CRM action lists.</div>',
        unsafe_allow_html=True,
    )

    # CRM Export Section
    exp_col1, exp_col2 = st.columns([3, 1])
    with exp_col1:
        st.markdown(
            f"""
            **CRM Campaign Export Ready:**  
            Download high-risk customer records (`Operational Risk Score > 70%`) pre-formatted for direct integration into retention email tools or CRM marketing workflows.
            """
        )
    with exp_col2:
        high_risk_csv = Path(HIGH_RISK_EXPORT_PATH)
        if high_risk_csv.exists():
            with open(high_risk_csv, "rb") as f:
                st.download_button(
                    label="📥 Export High-Risk CSV",
                    data=f.read(),
                    file_name="olist_high_risk_customers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("---")

    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_tier = st.multiselect(
            "Filter by Business Risk Tier:",
            options=[TIER_HIGH, TIER_MEDIUM, TIER_LOW],
            default=[TIER_HIGH, TIER_LOW],
        )
    with f_col2:
        search_id = st.text_input("Search Customer Unique ID:", placeholder="Paste hash here...")
    with f_col3:
        spend_range = st.slider(
            "Filter by Historical Spend (R$):",
            min_value=float(scored_df["total_spend"].min()),
            max_value=float(min(2000.0, scored_df["total_spend"].max())),
            value=(0.0, 1000.0),
        )

    # Filter DataFrame
    filtered_df = scored_df.copy()
    if selected_tier:
        filtered_df = filtered_df[filtered_df["business_risk_tier"].isin(selected_tier)]
    if search_id:
        filtered_df = filtered_df[filtered_df["customer_unique_id"].str.contains(search_id, case=False)]
    filtered_df = filtered_df[
        (filtered_df["total_spend"] >= spend_range[0])
        & (filtered_df["total_spend"] <= spend_range[1])
    ]

    st.markdown(f"**Showing {len(filtered_df):,} matching customers:**")

    display_cols = [
        "customer_unique_id",
        "business_risk_tier",
        "operational_risk_score",
        "calibrated_churn_prob",
        "risk_percentile",
        "recency_days",
        "tenure_days",
        "frequency_orders",
        "total_spend",
        "avg_review_score",
        "customer_state",
    ]

    st.dataframe(
        filtered_df[display_cols].head(100).style.format(
            {
                "operational_risk_score": "{:.2%}",
                "calibrated_churn_prob": "{:.2%}",
                "risk_percentile": "{:.1f}%",
                "recency_days": "{:.0f}",
                "tenure_days": "{:.0f}",
                "total_spend": "R$ {:.2f}",
                "avg_review_score": "{:.1f} ★",
            }
        ),
        use_container_width=True,
    )

    # Customer Deep-Dive Inspector
    st.markdown("### Individual Customer Profile Deep-Dive")
    if not filtered_df.empty:
        sample_ids = filtered_df["customer_unique_id"].head(20).tolist()
        inspect_id = st.selectbox("Select Customer to Inspect:", options=sample_ids)
        cust_row = scored_df[scored_df["customer_unique_id"] == inspect_id].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            tier_class = (
                "risk-high"
                if cust_row["business_risk_tier"] == TIER_HIGH
                else "risk-med"
                if cust_row["business_risk_tier"] == TIER_MEDIUM
                else "risk-low"
            )
            st.markdown(
                f"""
                **Assigned Risk Tier:**<br>
                <span class="{tier_class}" style="font-size:1.3rem;">{cust_row['business_risk_tier']}</span>
                """,
                unsafe_allow_html=True,
            )
        with d2:
            st.metric("Operational Risk Score", f"{cust_row['operational_risk_score']:.2%}")
        with d3:
            st.metric("Calibrated Churn Prob", f"{cust_row['calibrated_churn_prob']:.2%}")
        with d4:
            st.metric("Risk Percentile", f"Top {100 - cust_row['risk_percentile']:.1f}%")

        # Gauge Chart for Risk Score
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=cust_row["operational_risk_score"] * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Operational Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1E293B"},
                    "steps": [
                        {"range": [0, 30], "color": "#86EFAC"},
                        {"range": [30, 70], "color": "#FDE047"},
                        {"range": [70, 100], "color": "#FCA5A5"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 70,
                    },
                },
            )
        )
        gauge_fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Brazilian E-Commerce Churn Intelligence System • Snapshot 2018-02-28 • 180-Day Evaluation Window • Built with Streamlit & Scikit-Learn"
)
