# =============================================================================
# PHASE 4 — CREDIT RISK MODELING
# Credit Risk & Loan Decision Engine
# Author: Colin McBane
# Input: data/processed/train_features.csv
#        data/processed/test_features.csv
#        data/processed/scaler_params.csv
# Output: models/ (saved model files)
#         outputs/shap_plots/ (SHAP charts)
#         data/processed/model_results.csv (metrics comparison)
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import shap

# =============================================================================
# FILE PATHS
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "data", "processed", "train_features.csv")
TEST_PATH   = os.path.join(BASE_DIR, "data", "processed", "test_features.csv")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler_params.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
SHAP_DIR    = os.path.join(BASE_DIR, "outputs", "shap_plots")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fixed random seed for reproducibility — required for SR 11-7 documentation
RANDOM_STATE = 42

# K-Fold cross validation folds
N_FOLDS = 5

# SMOTE sampling strategy — 4:1 ratio as recommended by Gemini review
# Balances class imbalance without generating excessive synthetic points
# Prevents memory issues on 382K row training set
SMOTE_STRATEGY = 0.25

# Decision threshold — probability above this = high risk
# Set lower than 0.5 because cost of false negative (approving bad loan)
# is much higher than false positive (rejecting good loan)
DECISION_THRESHOLD = 0.35

print("=== Phase 4: Credit Risk Modeling ===")
print(f"Training data:  {TRAIN_PATH}")
print(f"Test data:      {TEST_PATH}")
print(f"Models saved to: {MODELS_DIR}")
print(f"SHAP plots:     {SHAP_DIR}")
print(f"\nConfiguration:")
print(f"   Random state:       {RANDOM_STATE}")
print(f"   CV folds:           {N_FOLDS}")
print(f"   SMOTE strategy:     {SMOTE_STRATEGY}")
print(f"   Decision threshold: {DECISION_THRESHOLD}")

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data():
    print("\n[1/8] Loading feature matrices...")

    # Load train and test sets built in Phase 3
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # Separate features from target
    X_train = train.drop("is_default", axis=1)
    y_train = train["is_default"]
    X_test  = test.drop("is_default", axis=1)
    y_test  = test["is_default"]

    # Load scaler parameters for stress testing
    scaler_params = pd.read_csv(SCALER_PATH)

    print(f"   Training set:   {X_train.shape[0]:,} rows | {X_train.shape[1]} features")
    print(f"   Test set:       {X_test.shape[0]:,} rows  | {X_test.shape[1]} features")
    print(f"   Train default:  {y_train.mean():.2%}")
    print(f"   Test default:   {y_test.mean():.2%}")
    print(f"   Scaler params:  {len(scaler_params)} features loaded")

    return X_train, y_train, X_test, y_test, scaler_params

# =============================================================================
# STEP 2: EVALUATION UTILITIES
# =============================================================================
