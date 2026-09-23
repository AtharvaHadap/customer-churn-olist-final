"""Unit tests verifying model scoring, calibration, and risk tier assignments."""

import unittest
import json
import pandas as pd
import numpy as np

from src.config import (
    MODEL_PIPELINE_PATH,
    METRICS_PATH,
    SCORED_CUSTOMERS_PATH,
    HIGH_RISK_EXPORT_PATH,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    TIER_HIGH,
    TIER_MEDIUM,
    TIER_LOW,
)
from src.model import load_artifacts


class TestModelPipeline(unittest.TestCase):
    """Verify saved models, predictions, metrics, and customer exports."""

    @classmethod
    def setUpClass(cls):
        cls.artifacts = load_artifacts(MODEL_PIPELINE_PATH)
        cls.scored_df = pd.read_parquet(SCORED_CUSTOMERS_PATH)
        with open(METRICS_PATH, "r") as f:
            cls.metrics = json.load(f)

    def test_artifacts_contain_both_pipelines(self):
        """Model artifacts must contain operational and calibrated pipelines."""
        self.assertIn("operational_pipeline", self.artifacts)
        self.assertIn("calibrated_pipeline", self.artifacts)
        self.assertIn("importance_df", self.artifacts)

    def test_scores_bounded_zero_to_one(self):
        """All operational scores and calibrated probabilities must be in [0, 1]."""
        self.assertTrue((self.scored_df["operational_risk_score"] >= 0.0).all())
        self.assertTrue((self.scored_df["operational_risk_score"] <= 1.0).all())
        self.assertTrue((self.scored_df["calibrated_churn_prob"] >= 0.0).all())
        self.assertTrue((self.scored_df["calibrated_churn_prob"] <= 1.0).all())

    def test_risk_tier_threshold_assignments(self):
        """Verify strict adherence to fixed thresholds (>0.70 High, 0.30-0.70 Med, <0.30 Low)."""
        high_mask = self.scored_df["operational_risk_score"] > HIGH_RISK_THRESHOLD
        med_mask = (self.scored_df["operational_risk_score"] >= LOW_RISK_THRESHOLD) & (
            self.scored_df["operational_risk_score"] <= HIGH_RISK_THRESHOLD
        )
        low_mask = self.scored_df["operational_risk_score"] < LOW_RISK_THRESHOLD

        self.assertTrue((self.scored_df.loc[high_mask, "business_risk_tier"] == TIER_HIGH).all())
        self.assertTrue((self.scored_df.loc[med_mask, "business_risk_tier"] == TIER_MEDIUM).all())
        self.assertTrue((self.scored_df.loc[low_mask, "business_risk_tier"] == TIER_LOW).all())

    def test_high_risk_export_consistency(self):
        """Exported CSV must exist and only contain customers with score > 0.70."""
        self.assertTrue(HIGH_RISK_EXPORT_PATH.exists())
        export_df = pd.read_csv(HIGH_RISK_EXPORT_PATH)
        self.assertGreater(len(export_df), 0)
        self.assertTrue((export_df["operational_risk_score"] > HIGH_RISK_THRESHOLD).all())
        self.assertTrue((export_df["business_risk_tier"] == TIER_HIGH).all())

    def test_evaluation_metrics_ranges(self):
        """Check that metrics are logically sound and within standard bounds."""
        roc_auc = self.metrics["discrimination"]["roc_auc_operational"]
        self.assertGreater(roc_auc, 0.50)
        self.assertLessEqual(roc_auc, 1.00)

        brier = self.metrics["calibration"]["brier_score_calibrated"]
        self.assertLess(brier, 0.05)  # Well calibrated to 0.011

        lift = self.metrics["dual_pr_auc"]["active_class_0"]["relative_lift_over_baseline"]
        self.assertGreater(lift, 1.0)  # Positive lift over baseline


if __name__ == "__main__":
    unittest.main()
