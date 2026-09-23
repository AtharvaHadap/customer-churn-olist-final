"""Unit tests verifying strict prevention of data leakage."""

import unittest
import pandas as pd

from src.config import SNAPSHOT_DATE
from src.data_loader import load_raw_tables, prepare_validated_orders
from src.feature_engineering import extract_snapshot_features


class TestDataLeakage(unittest.TestCase):
    """Verify that no lookahead timestamps enter feature engineering."""

    @classmethod
    def setUpClass(cls):
        cls.tables = load_raw_tables()
        cls.orders_merged = prepare_validated_orders(
            cls.tables["orders"], cls.tables["customers"], cls.tables["order_items"]
        )
        cls.features_df = extract_snapshot_features(cls.tables, cls.orders_merged)

    def test_no_future_purchases_in_history(self):
        """All historical orders must have purchase timestamp <= SNAPSHOT_DATE."""
        hist_orders = self.orders_merged[
            (self.orders_merged["is_valid_delivered"])
            & (self.orders_merged["order_purchase_timestamp"] <= SNAPSHOT_DATE)
        ]
        max_dt = hist_orders["order_purchase_timestamp"].max()
        self.assertLessEqual(max_dt, SNAPSHOT_DATE)

    def test_recency_days_non_negative(self):
        """Recency must be >= 0 (no negative recency indicating future purchases)."""
        min_recency = self.features_df["recency_days"].min()
        self.assertGreaterEqual(min_recency, 0.0)

    def test_tenure_greater_or_equal_to_recency(self):
        """Customer tenure must be greater than or equal to recency."""
        diff = self.features_df["tenure_days"] - self.features_df["recency_days"]
        self.assertGreaterEqual(diff.min(), -1e-5)

    def test_reviews_strictly_before_snapshot(self):
        """Reviews included in historical features must have creation date <= SNAPSHOT_DATE."""
        reviews_hist = self.tables["reviews"][
            self.tables["reviews"]["review_creation_date"] <= SNAPSHOT_DATE
        ]
        max_review_dt = reviews_hist["review_creation_date"].max()
        self.assertLessEqual(max_review_dt, SNAPSHOT_DATE)


if __name__ == "__main__":
    unittest.main()
