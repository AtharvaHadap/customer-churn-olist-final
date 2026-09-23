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
