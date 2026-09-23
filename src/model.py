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
