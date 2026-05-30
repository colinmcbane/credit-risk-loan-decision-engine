# =============================================================================
# PHASE 3 — FEATURE ENGINEERING
# Credit Risk & Loan Decision Engine
# Author: Colin McBane
# Input: data/processed/sba_clean.db (built in Phase 1)
# Output: data/processed/feature_matrix.csv (input for Phase 4)
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# =============================================================================
# FILE PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "sba_clean.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "eda_charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

print("=== Phase 3: Feature Engineering ===")
print(f"Reading from: {DB_PATH}")
print(f"Charts saved to: {CHARTS_DIR}")

# =============================================================================
# STEP 1: LOAD DATA FROM SQLITE
# =============================================================================

def load_data():
    print("\n[1/5] Loading data from SQLite...")

    conn = sqlite3.connect(DB_PATH)

    # Load the clean loans view
    df = pd.read_sql("""
        SELECT
            loan_amount,
            sba_guarantee_pct,
            interest_rate,
            term_months,
            jobs_supported,
            naics_sector,
            business_age,
            borr_state,
            has_collateral,
            is_revolver,
            loan_size_bucket,
            approval_year,
            approval_month,
            is_default
        FROM loans_clean
    """, conn)

    conn.close()

    print(f"   Rows loaded: {len(df):,}")
    print(f"   Columns loaded: {df.shape[1]}")
    print(f"   Default rate: {df['is_default'].mean():.2%}")

    return df

# =============================================================================
# STEP 2: CLEAN AND VALIDATE FEATURES
# =============================================================================

def clean_features(df):
    print("\n[2/5] Cleaning and validating features...")

    # --- DROP sba_guaranteed_amount ---
    # Perfect 1.0 correlation with loan_amount identified in Phase 2 EDA
    # Keeping both causes multicollinearity in logistic regression
    # sba_guarantee_pct retained as it captures guarantee structure independently

    # --- CONSOLIDATE BUSINESS AGE ---
    # Phase 2 identified overlapping category definitions across fiscal years
    # Mapping all variants to a clean ordinal scale
    business_age_map = {
        "NEW, LESS THAN 1 YEAR OLD":              "new",
        "NEW BUSINESS OR 2 YEARS OR LESS":        "new",
        "STARTUP, LOAN FUNDS WILL OPEN BUSINESS": "startup",
        "EXISTING OR MORE THAN 2 YEARS OLD":      "established",
        "LESS THAN 3 YEARS OLD BUT AT LEAST 2":   "young",
        "LESS THAN 4 YEARS OLD BUT AT LEAST 3":   "young",
        "LESS THAN 5 YEARS OLD BUT AT LEAST 4":   "established",
        "EXISTING, 5 OR MORE YEARS":              "mature",
        "CHANGE OF OWNERSHIP":                    "change_of_ownership",
        "UNANSWERED":                             "unknown",
    }
    df["business_age"] = df["business_age"].map(business_age_map).fillna("unknown")

    # --- CLEAN COLLATERAL AND REVOLVER ---
    # Convert Y/N flags to binary integers
    df["has_collateral"] = (df["has_collateral"] == "Y").astype(int)
    df["is_revolver"] = (df["is_revolver"] == "1").astype(int)

    # --- EXCLUDE RECENT VINTAGES ---
    # Loans approved 2023-2025 are right-censored — insufficient time to default
    # Identified in Phase 2 time series analysis
    # Keep for prediction but flag for training exclusion
    df["is_recent"] = (df["approval_year"] >= 2023).astype(int)

    # --- DROP NULLS ---
    before = len(df)
    df = df.dropna(subset=[
        "loan_amount", "interest_rate", "term_months",
        "sba_guarantee_pct", "naics_sector", "is_default"
    ])
    after = len(df)
    print(f"   Rows dropped (nulls): {before - after:,}")
    print(f"   Rows remaining: {after:,}")

    # --- REPORT BUSINESS AGE DISTRIBUTION ---
    print("\n   Business age distribution after consolidation:")
    print(df["business_age"].value_counts())

    return df

# =============================================================================
# STEP 3: TRAIN/TEST SPLIT
# =============================================================================

from sklearn.model_selection import train_test_split

def split_data(df):
    print("\n[3/5] Creating train/test split...")

    # --- EXCLUDE RIGHT-CENSORED LOANS FROM TRAINING ---
    # Loans approved 2023-2025 have not had sufficient time to default
    # Identified in Phase 2 time series analysis as right-censored
    # Keep in a separate holdout set for prediction only — never train on these
    df_train_pool = df[df["is_recent"] == 0].copy()
    df_recent = df[df["is_recent"] == 1].copy()

    print(f"   Training pool (pre-2023): {len(df_train_pool):,} rows")
    print(f"   Recent loans (2023-2025): {len(df_recent):,} rows")

    # --- DEFINE FEATURES AND TARGET ---
    # Drop columns that are not model features
    DROP_COLS = [
        "is_default",      # target variable
        "is_recent",       # flag column not a feature
        "approval_year",   # leakage risk — year correlates with default rate
        "approval_month",  # too granular, low signal
    ]

    FEATURE_COLS = [col for col in df_train_pool.columns if col not in DROP_COLS]

    X = df_train_pool[FEATURE_COLS]
    y = df_train_pool["is_default"]

    print(f"\n   Features: {len(FEATURE_COLS)}")
    print(f"   Feature list: {FEATURE_COLS}")
    print(f"\n   Class distribution:")
    print(f"   Paid in Full: {(y == 0).sum():,} ({(y == 0).mean():.2%})")
    print(f"   Charged Off:  {(y == 1).sum():,} ({(y == 1).mean():.2%})")

    # --- SPLIT 80/20 ---
    # Stratified split preserves class imbalance ratio in both sets
    # Random state fixed for reproducibility
    # Test set goes into a lockbox — never touched until final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\n   Training set: {len(X_train):,} rows")
    print(f"   Test set:     {len(X_test):,} rows")
    print(f"   Train default rate: {y_train.mean():.2%}")
    print(f"   Test default rate:  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, FEATURE_COLS, df_recent

# =============================================================================
# STEP 4: ENCODE CATEGORICALS AND SCALE NUMERICS
# =============================================================================

def encode_and_scale(X_train, X_test):
    print("\n[4/5] Encoding categoricals and scaling numerics...")

    # --- IDENTIFY COLUMN TYPES ---
    categorical_cols = [
        "naics_sector",
        "business_age",
        "borr_state",
        "loan_size_bucket"
    ]

    numeric_cols = [
        "loan_amount",
        "sba_guarantee_pct",
        "interest_rate",
        "term_months",
        "jobs_supported",
        "has_collateral",
        "is_revolver"
    ]

    # --- ENCODE CATEGORICALS ---
    # Using pandas get_dummies (one-hot encoding)
    # drop_first=True removes one category per feature to avoid dummy variable trap
    # Dummy variable trap: if you have 3 categories A B C and encode all three,
    # C is perfectly predicted by A=0 and B=0 — creates multicollinearity
    print("   Encoding categorical columns...")

    X_train_encoded = pd.get_dummies(
        X_train,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    X_test_encoded = pd.get_dummies(
        X_test,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # --- ALIGN TRAIN AND TEST COLUMNS ---
    # After one-hot encoding train and test may have different columns
    # if a category appears in one but not the other
    # Align forces them to have identical columns — fill missing with 0
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0
    )

    print(f"   Columns after encoding: {X_train_encoded.shape[1]}")

    # --- SCALE NUMERICS ---
    # StandardScaler transforms each numeric column to mean=0, std=1
    # Fit ONLY on training data — never fit on test data
    # Fitting on test data would leak test set statistics into the model
    # Transform both train and test using training set parameters
    print("   Scaling numeric columns...")

    # Identify which columns are numeric after encoding
    numeric_cols_present = [
        col for col in numeric_cols
        if col in X_train_encoded.columns
    ]

    scaler = StandardScaler()

    # Fit on training set only
    X_train_encoded[numeric_cols_present] = scaler.fit_transform(
        X_train_encoded[numeric_cols_present]
    )

    # Transform test set using training set parameters
    X_test_encoded[numeric_cols_present] = scaler.transform(
        X_test_encoded[numeric_cols_present]
    )

    print(f"   Numeric columns scaled: {len(numeric_cols_present)}")
    print(f"   Final training shape: {X_train_encoded.shape}")
    print(f"   Final test shape:     {X_test_encoded.shape}")

    return X_train_encoded, X_test_encoded, scaler

