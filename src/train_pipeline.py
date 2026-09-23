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
