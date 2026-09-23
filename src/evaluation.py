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
