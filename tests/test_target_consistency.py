"""Unit tests verifying target variable consistency and cohort population counts."""

import unittest
import pandas as pd

from src.data_loader import load_raw_tables, prepare_validated_orders
from src.target_builder import build_cohort_and_targets


class TestTargetConsistency(unittest.TestCase):
    """Verify target derivation against exact empirical cohort counts."""

    @classmethod
    def setUpClass(cls):
        tables = load_raw_tables()
        orders_merged = prepare_validated_orders(
            tables["orders"], tables["customers"], tables["order_items"]
        )
        cls.cohort_df, cls.summary = build_cohort_and_targets(orders_merged)

    def test_total_eligible_customers_count(self):
        """Eligible customer cohort must equal exactly 55,525."""
        self.assertEqual(len(self.cohort_df), 55525)
        self.assertEqual(self.summary["eligible_customers"], 55525)

    def test_active_customers_count(self):
        """Active customers (churn=0) must equal exactly 653."""
        active_count = (self.cohort_df["churn"] == 0).sum()
        self.assertEqual(active_count, 653)
        self.assertEqual(self.summary["active_customers"], 653)

    def test_churned_customers_count(self):
        """Churned customers (churn=1) must equal exactly 54,872."""
        churned_count = (self.cohort_df["churn"] == 1).sum()
        self.assertEqual(churned_count, 54872)
        self.assertEqual(self.summary["churned_customers"], 54872)

    def test_target_is_strictly_binary(self):
        """Churn target values must contain only {0, 1}."""
        unique_targets = set(self.cohort_df["churn"].unique())
        self.assertEqual(unique_targets, {0, 1})

    def test_no_duplicate_customers(self):
        """Every customer_unique_id in the cohort must be unique."""
        self.assertEqual(self.cohort_df["customer_unique_id"].nunique(), len(self.cohort_df))


if __name__ == "__main__":
    unittest.main()
