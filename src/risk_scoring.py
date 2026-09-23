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
