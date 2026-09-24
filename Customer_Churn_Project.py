# Customer Churn Prediction & Risk Scoring - Complete Project Code
# Combined submission copy generated from the original src modules.


# ============================================================
# SOURCE MODULE: config.py
# ============================================================
"""Global configuration and project constants for Olist Customer Churn Prediction."""

from pathlib import Path
import pandas as pd

# Directory Structure
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Raw Data Filenames
ORDERS_FILE = DATA_RAW_DIR / "olist_orders_dataset.csv"
CUSTOMERS_FILE = DATA_RAW_DIR / "olist_customers_dataset.csv"
ORDER_ITEMS_FILE = DATA_RAW_DIR / "olist_order_items_dataset.csv"
PAYMENTS_FILE = DATA_RAW_DIR / "olist_order_payments_dataset.csv"
REVIEWS_FILE = DATA_RAW_DIR / "olist_order_reviews_dataset.csv"
PRODUCTS_FILE = DATA_RAW_DIR / "olist_products_dataset.csv"
SELLERS_FILE = DATA_RAW_DIR / "olist_sellers_dataset.csv"
GEOLOCATION_FILE = DATA_RAW_DIR / "olist_geolocation_dataset.csv"
CATEGORY_TRANSLATION_FILE = DATA_RAW_DIR / "product_category_name_translation.csv"

# Methodological Setup (Strictly Locked)
SNAPSHOT_DATE = pd.Timestamp("2018-02-28 23:59:59")
TARGET_WINDOW_DAYS = 180
TARGET_START_DATE = pd.Timestamp("2018-03-01 00:00:00")
TARGET_END_DATE = pd.Timestamp("2018-08-27 23:59:59")

# Fixed Risk Tier Thresholds
HIGH_RISK_THRESHOLD = 0.70
LOW_RISK_THRESHOLD = 0.30

# Risk Tier Labels
TIER_HIGH = "High Risk"
TIER_MEDIUM = "Medium Risk"
TIER_LOW = "Low Risk"

# Model & Split Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Processed & Artifact Paths
PROCESSED_COHORT_PATH = DATA_PROCESSED_DIR / "churn_cohort.parquet"
MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCORED_CUSTOMERS_PATH = ARTIFACTS_DIR / "scored_customers.parquet"
HIGH_RISK_EXPORT_PATH = ARTIFACTS_DIR / "high_risk_customers.csv"


# ============================================================
# SOURCE MODULE: data_loader.py
# ============================================================
"""Data loading and relational preparation module for Olist datasets."""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

from src.config import (
    ORDERS_FILE,
    CUSTOMERS_FILE,
    ORDER_ITEMS_FILE,
    PAYMENTS_FILE,
    REVIEWS_FILE,
    PRODUCTS_FILE,
    SELLERS_FILE,
    CATEGORY_TRANSLATION_FILE,
)


def load_raw_tables() -> Dict[str, pd.DataFrame]:
    """Load all relevant Olist raw datasets with appropriate datetime parsing."""
    orders = pd.read_csv(
        ORDERS_FILE,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    customers = pd.read_csv(CUSTOMERS_FILE)

    order_items = pd.read_csv(
        ORDER_ITEMS_FILE,
        parse_dates=["shipping_limit_date"],
    )

    payments = pd.read_csv(PAYMENTS_FILE)

    reviews = pd.read_csv(
        REVIEWS_FILE,
        parse_dates=["review_creation_date", "review_answer_timestamp"],
    )

    products = pd.read_csv(PRODUCTS_FILE)
    sellers = pd.read_csv(SELLERS_FILE)

    cat_translation = None
    if CATEGORY_TRANSLATION_FILE.exists():
        cat_translation = pd.read_csv(CATEGORY_TRANSLATION_FILE)

    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "cat_translation": cat_translation,
    }


def prepare_validated_orders(
    orders: pd.DataFrame, customers: pd.DataFrame, order_items: pd.DataFrame
) -> pd.DataFrame:
    """Merge orders with customers (customer_unique_id) and mark valid delivered orders with items."""
    # Attach customer_unique_id
    orders_merged = orders.merge(
        customers[["customer_id", "customer_unique_id", "customer_state", "customer_city"]],
        on="customer_id",
        how="left",
    )

    # Check existence of items
    items_order_ids = set(order_items["order_id"].unique())
    orders_merged["has_items"] = orders_merged["order_id"].isin(items_order_ids)

    # Mark valid completed orders: delivered and has items
    orders_merged["is_valid_delivered"] = (
        (orders_merged["order_status"] == "delivered") & orders_merged["has_items"]
    )

    return orders_merged


# ============================================================
# SOURCE MODULE: target_builder.py
# ============================================================
"""Target construction module for Olist Customer Churn Prediction.

Methodology:
- Snapshot Date: 2018-02-28 23:59:59
- Target Window: 180 days (2018-03-01 00:00:00 to 2018-08-27 23:59:59)
- Target Definition: A customer is classified as active (churn = 0) if they placed
  at least one valid delivered/completed order whose order_purchase_timestamp falls
  within the 180-day target window. The purchase timestamp determines target-window
  membership; delivery may occur after the window endpoint.
  If no such order exists, churn = 1.
"""

from typing import Tuple, Dict
import pandas as pd

from src.config import SNAPSHOT_DATE, TARGET_END_DATE


def build_cohort_and_targets(orders_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Construct eligible customer cohort and future churn target vector.

    Parameters
    ----------
    orders_df : pd.DataFrame
        Merged orders DataFrame with 'customer_unique_id', 'is_valid_delivered',
        and 'order_purchase_timestamp'.

    Returns
    -------
    cohort_df : pd.DataFrame
        DataFrame of eligible customers with 'customer_unique_id', 'churn',
        and 'target_orders_count'.
    summary_stats : Dict[str, int]
        Counts of eligible, active, and churned customers.
    """
    # Filter to valid delivered orders only
    valid_orders = orders_df[orders_df["is_valid_delivered"]].copy()

    # Eligible cohort: customers with at least one valid completed purchase <= SNAPSHOT_DATE
    hist_valid = valid_orders[valid_orders["order_purchase_timestamp"] <= SNAPSHOT_DATE]
    eligible_cust_ids = sorted(hist_valid["customer_unique_id"].unique())

    # Target window orders: purchase timestamp strictly within target window
    target_valid = valid_orders[
        (valid_orders["order_purchase_timestamp"] > SNAPSHOT_DATE)
        & (valid_orders["order_purchase_timestamp"] <= TARGET_END_DATE)
    ]

    # Count completed orders placed in the target window per eligible customer
    target_counts = (
        target_valid.groupby("customer_unique_id")["order_id"]
        .nunique()
        .reindex(eligible_cust_ids, fill_value=0)
        .rename("target_orders_count")
        .reset_index()
    )

    # Churn definition: churn = 1 if 0 orders in target window, else 0
    target_counts["churn"] = (target_counts["target_orders_count"] == 0).astype(int)

    summary_stats = {
        "eligible_customers": int(len(target_counts)),
        "churned_customers": int((target_counts["churn"] == 1).sum()),
        "active_customers": int((target_counts["churn"] == 0).sum()),
        "churn_rate": float((target_counts["churn"] == 1).mean()),
        "active_rate": float((target_counts["churn"] == 0).mean()),
    }

    return target_counts, summary_stats


# ============================================================
# SOURCE MODULE: feature_engineering.py
# ============================================================
"""Leakage-safe feature engineering module for Olist Customer Churn Prediction.

All features are engineered strictly as of SNAPSHOT_DATE (2018-02-28 23:59:59).
No future events (post-snapshot orders, reviews, or delivery completions) are used.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.config import SNAPSHOT_DATE


NUMERICAL_FEATURE_COLS = [
    "recency_days",
    "tenure_days",
    "frequency_orders",
    "frequency_items",
    "is_repeat_buyer_hist",
    "total_spend",
    "total_freight",
    "avg_order_value",
    "avg_item_price",
    "freight_ratio",
    "avg_review_score",
    "has_low_review",
    "has_review_at_snapshot",
    "avg_delivery_delay_days",
    "late_delivery_rate",
    "pending_delivery_at_snapshot",
    "has_delivery_history",
    "avg_payment_installments",
    "credit_card_share",
    "boleto_share",
    "distinct_categories",
]

CATEGORICAL_FEATURE_COLS = [
    "customer_state",
]


def extract_snapshot_features(
    tables: Dict[str, pd.DataFrame],
    orders_merged: pd.DataFrame,
) -> pd.DataFrame:
    """Extract strictly historical features as of SNAPSHOT_DATE.

    Parameters
    ----------
    tables : Dict[str, pd.DataFrame]
        Dictionary of raw tables loaded by data_loader.
    orders_merged : pd.DataFrame
        Orders merged with customers and validity flags.

    Returns
    -------
    features_df : pd.DataFrame
        Feature matrix for all eligible customers with zero lookahead leakage.
    """
    # 1. Filter historical valid orders strictly on or before SNAPSHOT_DATE
    hist_orders = orders_merged[
        (orders_merged["is_valid_delivered"])
        & (orders_merged["order_purchase_timestamp"] <= SNAPSHOT_DATE)
    ].copy()

    hist_order_ids = set(hist_orders["order_id"].unique())

    # 2. Base RFM & Tenure
    cust_rfm = (
        hist_orders.groupby("customer_unique_id")
        .agg(
            first_purchase=("order_purchase_timestamp", "min"),
            last_purchase=("order_purchase_timestamp", "max"),
            frequency_orders=("order_id", "nunique"),
            customer_state=("customer_state", "first"),
            customer_city=("customer_city", "first"),
        )
        .reset_index()
    )

    cust_rfm["recency_days"] = (
        SNAPSHOT_DATE - cust_rfm["last_purchase"]
    ).dt.total_seconds() / 86400.0
    cust_rfm["tenure_days"] = (
        SNAPSHOT_DATE - cust_rfm["first_purchase"]
    ).dt.total_seconds() / 86400.0
    cust_rfm["is_repeat_buyer_hist"] = (
        cust_rfm["frequency_orders"] > 1
    ).astype(int)

    # 3. Item-level aggregations
    raw_items = tables["order_items"]
    hist_items = raw_items[raw_items["order_id"].isin(hist_order_ids)].merge(
        hist_orders[["order_id", "customer_unique_id"]], on="order_id", how="left"
    )

    items_agg = (
        hist_items.groupby("customer_unique_id")
        .agg(
            frequency_items=("order_item_id", "count"),
            total_spend=("price", "sum"),
            total_freight=("freight_value", "sum"),
            distinct_categories=("product_id", "nunique"),
        )
        .reset_index()
    )

    # 4. Review features (strictly review_creation_date <= SNAPSHOT_DATE)
    raw_reviews = tables["reviews"]
    reviews_hist = raw_reviews[
        raw_reviews["review_creation_date"] <= SNAPSHOT_DATE
    ].merge(
        hist_orders[["order_id", "customer_unique_id"]], on="order_id", how="inner"
    )
    reviews_hist["is_low_review"] = (reviews_hist["review_score"] <= 2).astype(int)

    reviews_agg = (
        reviews_hist.groupby("customer_unique_id")
        .agg(
            avg_review_score=("review_score", "mean"),
            has_low_review=("is_low_review", "max"),
            review_count=("review_score", "count"),
        )
        .reset_index()
    )

    # 5. Logistics features (strictly orders delivered <= SNAPSHOT_DATE)
    delivered_orders = hist_orders[
        hist_orders["order_delivered_customer_date"] <= SNAPSHOT_DATE
    ].copy()
    delivered_orders["delivery_delay_days"] = (
        delivered_orders["order_delivered_customer_date"]
        - delivered_orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    delivered_orders["is_late"] = (
        delivered_orders["delivery_delay_days"] > 0
    ).astype(int)

    logistics_agg = (
        delivered_orders.groupby("customer_unique_id")
        .agg(
            avg_delivery_delay_days=("delivery_delay_days", "mean"),
            late_delivery_rate=("is_late", "mean"),
            delivered_count=("order_id", "count"),
        )
        .reset_index()
    )

    # Pending delivery flag
    pending_orders = hist_orders[
        hist_orders["order_delivered_customer_date"].isna()
        | (hist_orders["order_delivered_customer_date"] > SNAPSHOT_DATE)
    ]
    pending_custs = set(pending_orders["customer_unique_id"].unique())

    # 6. Payment features (strictly payments for historical orders)
    raw_payments = tables["payments"]
    hist_payments = raw_payments[raw_payments["order_id"].isin(hist_order_ids)].merge(
        hist_orders[["order_id", "customer_unique_id"]], on="order_id", how="left"
    )
    hist_payments["cc_val"] = np.where(
        hist_payments["payment_type"] == "credit_card",
        hist_payments["payment_value"],
        0.0,
    )
    hist_payments["boleto_val"] = np.where(
        hist_payments["payment_type"] == "boleto",
        hist_payments["payment_value"],
        0.0,
    )

    payments_agg = (
        hist_payments.groupby("customer_unique_id")
        .agg(
            avg_payment_installments=("payment_installments", "mean"),
            total_payment_val=("payment_value", "sum"),
            cc_payment_val=("cc_val", "sum"),
            boleto_payment_val=("boleto_val", "sum"),
        )
        .reset_index()
    )
    payments_agg["credit_card_share"] = payments_agg["cc_payment_val"] / (
        payments_agg["total_payment_val"] + 1e-6
    )
    payments_agg["boleto_share"] = payments_agg["boleto_payment_val"] / (
        payments_agg["total_payment_val"] + 1e-6
    )

    # 7. Merge into unified customer table
    features = cust_rfm.merge(items_agg, on="customer_unique_id", how="left")
    features["avg_order_value"] = (
        features["total_spend"] / features["frequency_orders"]
    )
    features["avg_item_price"] = (
        features["total_spend"] / features["frequency_items"]
    )
    features["freight_ratio"] = features["total_freight"] / (
        features["total_spend"] + features["total_freight"] + 1e-6
    )

    # Merge Reviews
    features = features.merge(reviews_agg, on="customer_unique_id", how="left")
    features["has_review_at_snapshot"] = (
        features["review_count"].notna().astype(int)
    )
    # Neutral imputation (median) for customers with pending/missing pre-snapshot reviews
    median_review = (
        reviews_agg["avg_review_score"].median()
        if not reviews_agg.empty
        else 4.0
    )
    features["avg_review_score"] = features["avg_review_score"].fillna(
        median_review
    )
    features["has_low_review"] = features["has_low_review"].fillna(0).astype(int)
    features["review_count"] = features["review_count"].fillna(0).astype(int)

    # Merge Logistics
    features = features.merge(logistics_agg, on="customer_unique_id", how="left")
    features["pending_delivery_at_snapshot"] = (
        features["customer_unique_id"].isin(pending_custs).astype(int)
    )
    features["avg_delivery_delay_days"] = features[
        "avg_delivery_delay_days"
    ].fillna(0.0)
    features["late_delivery_rate"] = features["late_delivery_rate"].fillna(0.0)
    features["has_delivery_history"] = (
        features["delivered_count"].notna().astype(int)
    )
    features["delivered_count"] = features["delivered_count"].fillna(0).astype(int)

    # Merge Payments
    features = features.merge(
        payments_agg[
            [
                "customer_unique_id",
                "avg_payment_installments",
                "credit_card_share",
                "boleto_share",
            ]
        ],
        on="customer_unique_id",
        how="left",
    )
    features["avg_payment_installments"] = features[
        "avg_payment_installments"
    ].fillna(1.0)
    features["credit_card_share"] = features["credit_card_share"].fillna(0.0)
    features["boleto_share"] = features["boleto_share"].fillna(0.0)

    # Categorical normalization: retain top 5 states, bundle remainder as 'Other'
    top_states = ["SP", "RJ", "MG", "RS", "PR"]
    features["customer_state"] = features["customer_state"].apply(
        lambda s: s if s in top_states else "Other"
    )

    return features


# ============================================================
# SOURCE MODULE: model.py
# ============================================================
"""Logistic Regression modeling and calibration module for Olist Customer Churn.

Architecture:
1. Operational Risk Model: Logistic Regression with class_weight='balanced'
   - Generates the Operational Risk Score in [0, 1] used for ranking and tiering.
2. Calibrated Probability Model: Logistic Regression with class_weight=None
   - Directly estimates the empirical base-rate churn probability (~98.8%).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.config import (
    RANDOM_STATE,
    MODEL_PIPELINE_PATH,
    PROCESSED_COHORT_PATH,
)
from src.feature_engineering import NUMERICAL_FEATURE_COLS, CATEGORICAL_FEATURE_COLS


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for standardizing numericals and one-hot encoding categoricals."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURE_COLS,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLS,
            ),
        ]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, Pipeline]:
    """Train both the balanced operational risk model and the calibrated empirical model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target labels (1=churn, 0=active).

    Returns
    -------
    operational_pipeline : Pipeline
        Pipeline with class_weight='balanced' for risk scoring & tiering.
    calibrated_pipeline : Pipeline
        Pipeline with class_weight=None for true empirical base-rate probabilities.
    """
    preprocessor_bal = build_preprocessor()
    operational_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor_bal),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    operational_pipeline.fit(X_train, y_train)

    preprocessor_cal = build_preprocessor()
    calibrated_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor_cal),
            (
                "classifier",
                LogisticRegression(
                    class_weight=None,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    calibrated_pipeline.fit(X_train, y_train)

    return operational_pipeline, calibrated_pipeline


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature names, logistic regression log-odds coefficients, and odds ratios."""
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    cat_feature_names = list(
        pre.named_transformers_["cat"].get_feature_names_out(
            CATEGORICAL_FEATURE_COLS
        )
    )
    all_feature_names = NUMERICAL_FEATURE_COLS + cat_feature_names

    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)

    importance_df = pd.DataFrame(
        {
            "feature": all_feature_names,
            "coefficient": coefs,
            "odds_ratio": odds_ratios,
            "abs_impact": np.abs(coefs),
        }
    ).sort_values(by="abs_impact", ascending=False).reset_index(drop=True)

    return importance_df


def save_artifacts(
    operational_pipeline: Pipeline,
    calibrated_pipeline: Pipeline,
    importance_df: pd.DataFrame,
    filepath: Any = MODEL_PIPELINE_PATH,
) -> None:
    """Save trained model pipelines and feature importances."""
    payload = {
        "operational_pipeline": operational_pipeline,
        "calibrated_pipeline": calibrated_pipeline,
        "importance_df": importance_df,
        "numerical_features": NUMERICAL_FEATURE_COLS,
        "categorical_features": CATEGORICAL_FEATURE_COLS,
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, filepath)


def load_artifacts(filepath: Any = MODEL_PIPELINE_PATH) -> Dict[str, Any]:
    """Load serialized model pipelines and feature importances."""
    return joblib.load(filepath)


# ============================================================
# SOURCE MODULE: evaluation.py
# ============================================================
"""Comprehensive evaluation module for Olist Customer Churn Prediction.

Calculates:
- Discrimination: ROC-AUC, ROC Curve coordinates
- Calibration: Brier Score Loss, Expected Calibration Error (ECE)
- Dual PR-AUC:
  * Majority Churn (pos_label=1) vs 0.9882 baseline
  * Minority Active (pos_label=0) vs 0.0118 baseline & Relative Lift
- Classification Report: Precision, Recall, F1 for Class 0 & Class 1
- Confusion Matrix (counts & normalized)
- Cumulative Decile Lift & Gains Analysis
"""

from typing import Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.config import METRICS_PATH


def calculate_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def calculate_decile_lift(
    y_true: np.ndarray, scores: np.ndarray, n_deciles: int = 10
) -> pd.DataFrame:
    """Compute decile lift table for active customer capture based on active propensity."""
    # Active propensity = 1 - churn_score
    active_propensity = 1.0 - scores
    active_target = 1 - y_true  # 1 for active, 0 for churn

    df = pd.DataFrame(
        {
            "score": active_propensity,
            "actual_active": active_target,
        }
    )

    # Rank into deciles (Decile 1 = highest active propensity / lowest churn score)
    df["decile"] = pd.qcut(
        df["score"].rank(method="first", ascending=False),
        q=n_deciles,
        labels=[f"D{i+1}" for i in range(n_deciles)],
    )

    total_active = df["actual_active"].sum()
    total_customers = len(df)
    baseline_active_rate = total_active / total_customers

    summary = (
        df.groupby("decile", observed=True)
        .agg(
            customers=("actual_active", "count"),
            active_count=("actual_active", "sum"),
        )
        .reset_index()
    )

    summary["active_rate"] = summary["active_count"] / summary["customers"]
    summary["cum_active_count"] = summary["active_count"].cumsum()
    summary["cum_active_capture_pct"] = (
        summary["cum_active_count"] / (total_active + 1e-9)
    ) * 100.0
    summary["cum_customer_pct"] = (
        summary["customers"].cumsum() / total_customers
    ) * 100.0
    summary["lift"] = summary["active_rate"] / (baseline_active_rate + 1e-9)
    summary["cum_lift"] = (
        (summary["cum_active_count"] / summary["customers"].cumsum())
        / (baseline_active_rate + 1e-9)
    )

    return summary


def downsample_curve(
    x: np.ndarray, y: np.ndarray, max_points: int = 150
) -> Tuple[list, list]:
    """Downsample dense curve arrays to lightweight lists for JSON serialization."""
    if len(x) <= max_points:
        return [float(v) for v in x], [float(v) for v in y]
    indices = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return [float(x[i]) for i in indices], [float(y[i]) for i in indices]


def evaluate_models(
    y_true: pd.Series,
    operational_scores: np.ndarray,
    calibrated_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Run full evaluation suite on test dataset."""
    y_arr = y_true.to_numpy()

    # 1. Operational predictions at default operational threshold
    y_pred_op = (operational_scores >= threshold).astype(int)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_arr, y_pred_op)
    cm_dict = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # 3. Classification Report
    cr = classification_report(
        y_arr,
        y_pred_op,
        target_names=["Active (0)", "Churn (1)"],
        output_dict=True,
        zero_division=0,
    )

    # 4. Discrimination: ROC-AUC
    roc_auc_op = float(roc_auc_score(y_arr, operational_scores))
    roc_auc_cal = float(roc_auc_score(y_arr, calibrated_probs))

    # ROC Curve coordinates
    fpr, tpr, _ = roc_curve(y_arr, operational_scores)
    fpr_list, tpr_list = downsample_curve(fpr, tpr)

    # 5. Dual Precision-Recall Metrics
    # A. Majority Churn Class (pos_label=1)
    baseline_churn_rate = float(np.mean(y_arr))
    pr_auc_churn = float(average_precision_score(y_arr, operational_scores))
    p_churn, r_churn, _ = precision_recall_curve(
        y_arr, operational_scores, pos_label=1
    )
    p_churn_list, r_churn_list = downsample_curve(r_churn, p_churn)

    # B. Minority Active Class (pos_label=0)
    baseline_active_rate = 1.0 - baseline_churn_rate
    # To compute PR for class 0, invert target and invert score
    pr_auc_active = float(
        average_precision_score(1 - y_arr, 1.0 - operational_scores)
    )
    active_lift = pr_auc_active / (baseline_active_rate + 1e-9)
    p_active, r_active, _ = precision_recall_curve(
        1 - y_arr, 1.0 - operational_scores, pos_label=1
    )
    p_active_list, r_active_list = downsample_curve(r_active, p_active)

    # 6. Probability Calibration Metrics
    brier_calibrated = float(brier_score_loss(y_arr, calibrated_probs))
    brier_operational = float(brier_score_loss(y_arr, operational_scores))
    ece_calibrated = calculate_ece(y_arr, calibrated_probs)
    ece_operational = calculate_ece(y_arr, operational_scores)

    # 7. Decile Lift
    decile_df = calculate_decile_lift(y_arr, operational_scores)
    decile_dict = decile_df.to_dict(orient="records")

    metrics_payload = {
        "discrimination": {
            "roc_auc_operational": roc_auc_op,
            "roc_auc_calibrated": roc_auc_cal,
            "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
        },
        "calibration": {
            "brier_score_calibrated": brier_calibrated,
            "brier_score_operational": brier_operational,
            "ece_calibrated": ece_calibrated,
            "ece_operational": ece_operational,
            "mean_calibrated_prob": float(np.mean(calibrated_probs)),
            "mean_operational_score": float(np.mean(operational_scores)),
        },
        "dual_pr_auc": {
            "churn_class_1": {
                "pr_auc": pr_auc_churn,
                "baseline_prevalence": baseline_churn_rate,
                "pr_curve": {"recall": p_churn_list, "precision": r_churn_list},
            },
            "active_class_0": {
                "pr_auc": pr_auc_active,
                "baseline_prevalence": baseline_active_rate,
                "relative_lift_over_baseline": active_lift,
                "pr_curve": {
                    "recall": p_active_list,
                    "precision": r_active_list,
                },
            },
        },
        "classification_report": cr,
        "confusion_matrix": cm_dict,
        "decile_lift": decile_dict,
        "test_counts": {
            "total_test": len(y_arr),
            "churn_count": int(np.sum(y_arr == 1)),
            "active_count": int(np.sum(y_arr == 0)),
        },
    }

    return metrics_payload


def save_metrics(
    metrics: Dict[str, Any], filepath: Path = METRICS_PATH
) -> None:
    """Save metrics dictionary as formatted JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: Path = METRICS_PATH) -> Dict[str, Any]:
    """Load evaluation metrics JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================
# SOURCE MODULE: risk_scoring.py
# ============================================================
"""Business Risk Tier assignment and customer export module.

Thresholds (Strictly Fixed):
- High Risk:   Operational Risk Score > 0.70
- Medium Risk: 0.30 <= Operational Risk Score <= 0.70
- Low Risk:    Operational Risk Score < 0.30
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    TIER_HIGH,
    TIER_MEDIUM,
    TIER_LOW,
    HIGH_RISK_EXPORT_PATH,
    SCORED_CUSTOMERS_PATH,
)


def assign_risk_tiers(scores: np.ndarray) -> np.ndarray:
    """Vectorized assignment of business risk tiers based on fixed thresholds."""
    tiers = np.empty(len(scores), dtype=object)
    tiers[scores > HIGH_RISK_THRESHOLD] = TIER_HIGH
    tiers[(scores >= LOW_RISK_THRESHOLD) & (scores <= HIGH_RISK_THRESHOLD)] = TIER_MEDIUM
    tiers[scores < LOW_RISK_THRESHOLD] = TIER_LOW
    return tiers


def build_scored_customer_table(
    customer_ids: pd.Series,
    features_df: pd.DataFrame,
    operational_scores: np.ndarray,
    calibrated_probs: np.ndarray,
    actual_churn: pd.Series = None,
) -> pd.DataFrame:
    """Build full customer scoring table with behavioral features and risk tiers."""
    df = features_df.copy()
    df["customer_unique_id"] = customer_ids.values
    df["operational_risk_score"] = np.round(operational_scores, 4)
    df["calibrated_churn_prob"] = np.round(calibrated_probs, 4)
    df["business_risk_tier"] = assign_risk_tiers(df["operational_risk_score"].values)
    
    # Percentile ranking (0-100%)
    df["risk_percentile"] = np.round(
        df["operational_risk_score"].rank(pct=True) * 100.0, 1
    )

    if actual_churn is not None:
        df["actual_churn"] = actual_churn.values

    # Order columns logically
    primary_cols = [
        "customer_unique_id",
        "business_risk_tier",
        "operational_risk_score",
        "calibrated_churn_prob",
        "risk_percentile",
    ]
    if actual_churn is not None:
        primary_cols.append("actual_churn")

    remaining_cols = [c for c in df.columns if c not in primary_cols]
    return df[primary_cols + remaining_cols]


def export_high_risk_customers(
    scored_df: pd.DataFrame,
    filepath: Path = HIGH_RISK_EXPORT_PATH,
    threshold: float = HIGH_RISK_THRESHOLD,
) -> pd.DataFrame:
    """Filter and export high-risk customers for CRM retention actions."""
    high_risk_df = scored_df[scored_df["operational_risk_score"] > threshold].copy()
    high_risk_df = high_risk_df.sort_values(
        by="operational_risk_score", ascending=False
    ).reset_index(drop=True)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    high_risk_df.to_csv(filepath, index=False)
    return high_risk_df


def get_risk_tier_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Generate risk tier distribution summary counts and percentages."""
    total = len(scored_df)
    summary = (
        scored_df.groupby("business_risk_tier")
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_risk_score=("operational_risk_score", "mean"),
            avg_calibrated_prob=("calibrated_churn_prob", "mean"),
            avg_spend=("total_spend", "mean"),
            avg_recency=("recency_days", "mean"),
        )
        .reindex([TIER_HIGH, TIER_MEDIUM, TIER_LOW])
        .reset_index()
    )
    summary["percentage"] = (summary["customer_count"] / total) * 100.0
    return summary


# ============================================================
# SOURCE MODULE: train_pipeline.py
# ============================================================
"""End-to-end training and artifact generation pipeline for Olist Customer Churn."""

import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    SNAPSHOT_DATE,
    TARGET_END_DATE,
    RANDOM_STATE,
    TEST_SIZE,
    PROCESSED_COHORT_PATH,
    MODEL_PIPELINE_PATH,
    METRICS_PATH,
    SCORED_CUSTOMERS_PATH,
    HIGH_RISK_EXPORT_PATH,
)
from src.data_loader import load_raw_tables, prepare_validated_orders
from src.target_builder import build_cohort_and_targets
from src.feature_engineering import (
    extract_snapshot_features,
    NUMERICAL_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
)
from src.model import train_models, extract_feature_importance, save_artifacts
from src.evaluation import evaluate_models, save_metrics
from src.risk_scoring import (
    build_scored_customer_table,
    export_high_risk_customers,
    get_risk_tier_summary,
)


def run_pipeline() -> None:
    """Execute complete end-to-end data processing, training, evaluation, and scoring."""
    start_time = time.time()
    print("=" * 70)
    print("OLIST CUSTOMER CHURN PREDICTION & RISK SCORING PIPELINE")
    print("=" * 70)
    print(f"Snapshot Cutoff Date: {SNAPSHOT_DATE}")
    print(f"Target Window:        {SNAPSHOT_DATE} -> {TARGET_END_DATE} (180 days)")

    # 1. Ingestion
    print("\n[1/6] Loading raw Olist relational tables...")
    tables = load_raw_tables()
    orders_merged = prepare_validated_orders(
        tables["orders"], tables["customers"], tables["order_items"]
    )

    # 2. Target Construction
    print("[2/6] Building eligible customer cohort and churn target labels...")
    cohort_df, summary_stats = build_cohort_and_targets(orders_merged)
    print(f"      Total Eligible Customers: {summary_stats['eligible_customers']:,}")
    print(f"      Churn Customers (1):      {summary_stats['churned_customers']:,} ({summary_stats['churn_rate']:.4%})")
    print(f"      Active Customers (0):     {summary_stats['active_customers']:,} ({summary_stats['active_rate']:.4%})")

    # 3. Feature Engineering
    print("[3/6] Extracting leakage-safe snapshot features as of 2018-02-28...")
    features_df = extract_snapshot_features(tables, orders_merged)
    dataset = cohort_df.merge(features_df, on="customer_unique_id", how="inner")
    print(f"      Processed Feature Matrix Shape: {dataset.shape}")

    # Save processed cohort
    PROCESSED_COHORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_COHORT_PATH, index=False)
    print(f"      Saved processed dataset to: {PROCESSED_COHORT_PATH}")

    # 4. Modeling & Calibration
    print("\n[4/6] Training Logistic Regression models (Stratified 80/20)...")
    feature_cols = NUMERICAL_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    X = dataset[feature_cols]
    y = dataset["churn"]
    ids = dataset["customer_unique_id"]

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    operational_pipe, calibrated_pipe = train_models(X_train, y_train)
    importance_df = extract_feature_importance(operational_pipe)
    save_artifacts(operational_pipe, calibrated_pipe, importance_df, MODEL_PIPELINE_PATH)
    print(f"      Saved model pipelines and feature importances to: {MODEL_PIPELINE_PATH}")

    # 5. Evaluation
    print("\n[5/6] Evaluating on held-out test split (N = {len(y_test):,})...")
    test_operational_scores = operational_pipe.predict_proba(X_test)[:, 1]
    test_calibrated_probs = calibrated_pipe.predict_proba(X_test)[:, 1]

    metrics = evaluate_models(y_test, test_operational_scores, test_calibrated_probs)
    save_metrics(metrics, METRICS_PATH)
    print(f"      Saved metrics to: {METRICS_PATH}")
    print(f"      ROC-AUC (Operational):        {metrics['discrimination']['roc_auc_operational']:.4f}")
    print(f"      PR-AUC (Churn=1):             {metrics['dual_pr_auc']['churn_class_1']['pr_auc']:.4f} (Baseline: {metrics['dual_pr_auc']['churn_class_1']['baseline_prevalence']:.4f})")
    print(f"      PR-AUC (Active=0):            {metrics['dual_pr_auc']['active_class_0']['pr_auc']:.4f} (Baseline: {metrics['dual_pr_auc']['active_class_0']['baseline_prevalence']:.4f})")
    print(f"      Minority Active Lift:         {metrics['dual_pr_auc']['active_class_0']['relative_lift_over_baseline']:.2f}x")
    print(f"      Brier Score (Calibrated):     {metrics['calibration']['brier_score_calibrated']:.6f}")
    print(f"      ECE (Calibrated):             {metrics['calibration']['ece_calibrated']:.6f}")

    # 6. Scoring & Tier Assignment
    print("\n[6/6] Scoring full customer cohort and assigning Risk Tiers...")
    all_operational_scores = operational_pipe.predict_proba(X)[:, 1]
    all_calibrated_probs = calibrated_pipe.predict_proba(X)[:, 1]

    scored_customers = build_scored_customer_table(
        customer_ids=ids,
        features_df=dataset[feature_cols + ["first_purchase", "last_purchase"]],
        operational_scores=all_operational_scores,
        calibrated_probs=all_calibrated_probs,
        actual_churn=y,
    )
    scored_customers.to_parquet(SCORED_CUSTOMERS_PATH, index=False)
    print(f"      Saved full scored dataset to: {SCORED_CUSTOMERS_PATH}")

    high_risk_df = export_high_risk_customers(scored_customers, HIGH_RISK_EXPORT_PATH)
    print(f"      Exported {len(high_risk_df):,} High-Risk customers to: {HIGH_RISK_EXPORT_PATH}")

    # Tier Summary
    summary = get_risk_tier_summary(scored_customers)
    print("\n" + "=" * 70)
    print("BUSINESS RISK TIER DISTRIBUTION")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nPipeline completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline()
