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
