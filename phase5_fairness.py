# =============================================================================
# PHASE 5 — FAIRNESS ANALYSIS & MODEL GOVERNANCE
# Credit Risk & Loan Decision Engine
# Author: Colin McBane
# Input: models/champion_model.pkl
#        data/processed/test_features.csv
#        data/processed/sba_clean.db (raw data for group definitions)
#        data/processed/scaler.pkl
# Output: outputs/fairness/ (all fairness charts and tables)
#         model_card.md
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix
)

# =============================================================================
# FILE PATHS
# =============================================================================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "champion_model.pkl")
CHAMPION_NAME = os.path.join(BASE_DIR, "models", "champion_name.txt")
TEST_PATH     = os.path.join(BASE_DIR, "data", "processed", "test_features.csv")
SCALER_PATH   = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
DB_PATH       = os.path.join(BASE_DIR, "data", "processed", "sba_clean.db")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "fairness")
MODEL_CARD    = os.path.join(BASE_DIR, "model_card.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fairness thresholds — documented per ECOA and SR 11-7 standards
FOUR_FIFTHS_THRESHOLD  = 0.80   # ECOA disparate impact standard
EQUALIZED_ODDS_LOW     = 0.80   # Minimum acceptable ratio
EQUALIZED_ODDS_HIGH    = 1.25   # Maximum acceptable ratio
MIN_GROUP_PCT          = 0.01   # Minimum 1% of test set for reference group
DECISION_THRESHOLD     = 0.35   # Must match Phase 4 threshold exactly

print("=== Phase 5: Fairness Analysis & Model Governance ===")
print(f"Champion model:  {MODEL_PATH}")
print(f"Output dir:      {OUTPUT_DIR}")
print(f"\nFairness Thresholds:")
print(f"   4/5ths rule:        {FOUR_FIFTHS_THRESHOLD}")
print(f"   Equalized odds:     {EQUALIZED_ODDS_LOW} - {EQUALIZED_ODDS_HIGH}")
print(f"   Min group size:     {MIN_GROUP_PCT:.0%} of test set")
print(f"   Decision threshold: {DECISION_THRESHOLD}")

# =============================================================================
# STEP 1: LOAD MODEL AND DATA
# =============================================================================

def load_model_and_data():
    print("\n[1/9] Loading model and data...")

    # Load champion model
    model = joblib.load(MODEL_PATH)
    with open(CHAMPION_NAME, "r") as f:
        champion_name = f.read().strip()
    print(f"   Champion model: {champion_name}")

    # Load scaled test features
    test = pd.read_csv(TEST_PATH)
    X_test = test.drop("is_default", axis=1)
    y_test = test["is_default"]
    print(f"   Test set: {len(X_test):,} rows | {X_test.shape[1]} features")

    # Score test set using champion model
    # Using DECISION_THRESHOLD from Phase 4 — not 0.50 default
    # SMOTE probability shift documented in model card
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    print(f"   Baseline default rate:     {y_test.mean():.2%}")
    print(f"   Model predicted high risk: {y_pred.mean():.2%}")
    print(f"   Note: predicted rate higher than actual due to SMOTE")
    print(f"   probability shift — documented in model card")

    # Load RAW data from SQLite for group definitions
    # Critical: cannot use scaled features for grouping
    # Raw borr_state, business_age, naics_sector needed
    conn = sqlite3.connect(DB_PATH)
    raw_loans = pd.read_sql("""
        SELECT
            rowid,
            borr_state,
            businessage,
            naicscode,
            grossapproval,
            loan_size_bucket,
            loanstatus,
            is_default
        FROM loans
        WHERE loanstatus IN ('P I F', 'CHGOFF')
        ORDER BY rowid
    """, conn)
    conn.close()

    print(f"   Raw loans loaded: {len(raw_loans):,} rows")

    # Align raw data to test set by position
    # Test set is 20% of training pool — last 20% after stratified split
    # Use same random_state=42 split to reconstruct alignment
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(raw_loans)),
        test_size=0.20,
        random_state=42,
        stratify=raw_loans["is_default"]
    )

    raw_test = raw_loans.iloc[test_idx].reset_index(drop=True)
    print(f"   Raw test subset: {len(raw_test):,} rows")

    # Validate alignment
    assert len(raw_test) == len(X_test), \
        f"Alignment failed: raw={len(raw_test)}, scaled={len(X_test)}"
    assert (raw_test["is_default"].values == y_test.values).mean() > 0.99, \
        "Target variable alignment check failed"

    print("   Alignment validated ✓")

    # Attach predictions to raw data
    raw_test["y_prob"]    = y_prob
    raw_test["y_pred"]    = y_pred
    raw_test["y_true"]    = y_test.values

    return model, champion_name, X_test, y_test, y_prob, y_pred, raw_test