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
