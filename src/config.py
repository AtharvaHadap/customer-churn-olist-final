"""Global configuration and project constants for Olist Customer Churn Prediction."""

from pathlib import Path
import pandas as pd

# Directory Structure
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Raw Data Filenames
ORDERS_FILE = DATA_RAW_DIR / "olist_orders_dataset.csv"
CUSTOMERS_FILE = DATA_RAW_DIR / "olist_customers_dataset.csv"
ORDER_ITEMS_FILE = DATA_RAW_DIR / "olist_order_items_dataset.csv"
PAYMENTS_FILE = DATA_RAW_DIR / "olist_order_payments_dataset.csv"
REVIEWS_FILE = DATA_RAW_DIR / "olist_order_reviews_dataset.csv"
PRODUCTS_FILE = DATA_RAW_DIR / "olist_products_dataset.csv"
SELLERS_FILE = DATA_RAW_DIR / "olist_sellers_dataset.csv"
GEOLOCATION_FILE = DATA_RAW_DIR / "olist_geolocation_dataset.csv"
CATEGORY_TRANSLATION_FILE = DATA_RAW_DIR / "product_category_name_translation.csv"

# Methodological Setup (Strictly Locked)
SNAPSHOT_DATE = pd.Timestamp("2018-02-28 23:59:59")
TARGET_WINDOW_DAYS = 180
TARGET_START_DATE = pd.Timestamp("2018-03-01 00:00:00")
TARGET_END_DATE = pd.Timestamp("2018-08-27 23:59:59")

# Fixed Risk Tier Thresholds
HIGH_RISK_THRESHOLD = 0.70
LOW_RISK_THRESHOLD = 0.30

# Risk Tier Labels
TIER_HIGH = "High Risk"
TIER_MEDIUM = "Medium Risk"
TIER_LOW = "Low Risk"

# Model & Split Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Processed & Artifact Paths
PROCESSED_COHORT_PATH = DATA_PROCESSED_DIR / "churn_cohort.parquet"
MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCORED_CUSTOMERS_PATH = ARTIFACTS_DIR / "scored_customers.parquet"
HIGH_RISK_EXPORT_PATH = ARTIFACTS_DIR / "high_risk_customers.csv"
