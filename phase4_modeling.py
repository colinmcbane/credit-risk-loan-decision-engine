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

def calculate_ks_statistic(y_true, y_prob):
    """
    KS Statistic — maximum separation between cumulative
    distribution of defaulters and non-defaulters.
    Industry standard metric for credit scorecard evaluation.
    KS > 40 is good, KS > 50 is strong.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    df["cumulative_default"] = (
        df["y_true"].cumsum() / df["y_true"].sum()
    )
    df["cumulative_non_default"] = (
        (1 - df["y_true"]).cumsum() / (1 - df["y_true"]).sum()
    )
    df["ks"] = abs(
        df["cumulative_default"] - df["cumulative_non_default"]
    )

    return df["ks"].max() * 100


def calculate_gini(auc):
    """
    Gini Coefficient = 2 * AUC - 1
    Used in European banking and Basel III regulatory reporting.
    Gini of 0.60 means model captures 60% of a perfect model's power.
    """
    return (2 * auc - 1) * 100


def evaluate_model(model, X_test, y_test, model_name):
    """
    Full evaluation suite for a trained model.
    Returns dictionary of all metrics.
    """
    # Get probability scores
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply decision threshold
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    # Core metrics
    auc  = roc_auc_score(y_test, y_prob)
    ks   = calculate_ks_statistic(y_test.values, y_prob)
    gini = calculate_gini(auc)
    ap   = average_precision_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    results = {
        "model":      model_name,
        "auc":        round(auc, 4),
        "ks":         round(ks, 2),
        "gini":       round(gini, 2),
        "avg_precision": round(ap, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "tp":         int(tp),
        "fp":         int(fp),
        "tn":         int(tn),
        "fn":         int(fn),
        "y_prob":     y_prob
    }

    print(f"\n   {model_name} Results:")
    print(f"   AUC:       {auc:.4f}")
    print(f"   KS:        {ks:.2f}")
    print(f"   Gini:      {gini:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1:        {f1:.4f}")
    print(f"   TP: {tp:,}  FP: {fp:,}  TN: {tn:,}  FN: {fn:,}")

    return results

# =============================================================================
# STEP 3: LOGISTIC REGRESSION BASELINE
# =============================================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n[3/8] Training Logistic Regression baseline...")

    # --- K-FOLD CROSS VALIDATION ---
    # Stratified ensures each fold maintains 7.31% default rate
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_aucs = []

    print(f"   Running {N_FOLDS}-fold cross validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train), 1
    ):
        # Split into fold train and validation
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        # Apply SMOTE only to training fold — never to validation
        # sampling_strategy=0.25 creates 4:1 majority/minority ratio
        # Prevents memory issues on large training folds
        smote = SMOTE(
            sampling_strategy=SMOTE_STRATEGY,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        X_resampled, y_resampled = smote.fit_resample(
            X_fold_train, y_fold_train
        )

        # Train logistic regression on resampled fold
        lr = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
            C=0.1
        )
        lr.fit(X_resampled, y_resampled)

        # Evaluate on validation fold — no SMOTE applied
        val_prob = lr.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, val_prob)
        cv_aucs.append(fold_auc)

        print(f"   Fold {fold}: AUC = {fold_auc:.4f}")

    print(f"\n   CV AUC: {np.mean(cv_aucs):.4f} "
          f"(+/- {np.std(cv_aucs):.4f})")

    # --- TRAIN FINAL MODEL ON FULL TRAINING SET ---
    # After CV gives us confidence in the model
    # Train final version on all training data with SMOTE
    print("\n   Training final model on full training set...")

    smote_full = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    X_train_resampled, y_train_resampled = smote_full.fit_resample(
        X_train, y_train
    )

    lr_final = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs",
        C=0.1
    )
    lr_final.fit(X_train_resampled, y_train_resampled)

    # --- EVALUATE ON LOCKED TEST SET ---
    print("\n   Evaluating on test set...")
    lr_results = evaluate_model(
        lr_final, X_test, y_test, "Logistic Regression"
    )
    lr_results["cv_auc_mean"] = round(np.mean(cv_aucs), 4)
    lr_results["cv_auc_std"]  = round(np.std(cv_aucs), 4)

    # --- SAVE MODEL ---
    model_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    joblib.dump(lr_final, model_path)
    print(f"\n   Model saved: {model_path}")

    return lr_final, lr_results

# =============================================================================
# STEP 4: XGBOOST MODEL
# =============================================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[4/8] Training XGBoost...")

    # --- CLASS WEIGHT CALCULATION ---
    # XGBoost uses scale_pos_weight instead of class_weight
    # scale_pos_weight = count(negative) / count(positive)
    # Tells XGBoost how much more to penalize missing a default
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

    # --- K-FOLD CROSS VALIDATION ---
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_aucs = []

    print(f"   Running {N_FOLDS}-fold cross validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train), 1
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        # SMOTE on training fold only
        smote = SMOTE(
            sampling_strategy=SMOTE_STRATEGY,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        X_resampled, y_resampled = smote.fit_resample(
            X_fold_train, y_fold_train
        )

        # Train XGBoost on resampled fold
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            early_stopping_rounds=50,
            verbosity=0
        )
        xgb_model.fit(
            X_resampled, y_resampled,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        val_prob = xgb_model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, val_prob)
        cv_aucs.append(fold_auc)

        print(f"   Fold {fold}: AUC = {fold_auc:.4f}")

    print(f"\n   CV AUC: {np.mean(cv_aucs):.4f} "
          f"(+/- {np.std(cv_aucs):.4f})")

    # --- TRAIN FINAL MODEL ---
    print("\n   Training final XGBoost on full training set...")

    smote_full = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    X_train_resampled, y_train_resampled = smote_full.fit_resample(
        X_train, y_train
    )

    xgb_final = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0
    )
    xgb_final.fit(X_train_resampled, y_train_resampled)

    # --- EVALUATE ON TEST SET ---
    print("\n   Evaluating on test set...")
    xgb_results = evaluate_model(
        xgb_final, X_test, y_test, "XGBoost"
    )
    xgb_results["cv_auc_mean"] = round(np.mean(cv_aucs), 4)
    xgb_results["cv_auc_std"]  = round(np.std(cv_aucs), 4)

    # --- SAVE MODEL ---
    model_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    joblib.dump(xgb_final, model_path)
    print(f"\n   Model saved: {model_path}")

    return xgb_final, xgb_results

# =============================================================================
# STEP 5: LIGHTGBM MODEL
# =============================================================================

def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\n[5/8] Training LightGBM...")

    # --- CLASS WEIGHT CALCULATION ---
    # LightGBM uses is_unbalance or scale_pos_weight
    # Using scale_pos_weight for consistency with XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

    # --- K-FOLD CROSS VALIDATION ---
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_aucs = []

    print(f"   Running {N_FOLDS}-fold cross validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train), 1
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        # SMOTE on training fold only
        smote = SMOTE(
            sampling_strategy=SMOTE_STRATEGY,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        X_resampled, y_resampled = smote.fit_resample(
            X_fold_train, y_fold_train
        )

        # Train LightGBM on resampled fold
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        )
        lgb_model.fit(
            X_resampled, y_resampled,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )

        val_prob = lgb_model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, val_prob)
        cv_aucs.append(fold_auc)

        print(f"   Fold {fold}: AUC = {fold_auc:.4f}")

    print(f"\n   CV AUC: {np.mean(cv_aucs):.4f} "
          f"(+/- {np.std(cv_aucs):.4f})")

    # --- TRAIN FINAL MODEL ---
    print("\n   Training final LightGBM on full training set...")

    smote_full = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    X_train_resampled, y_train_resampled = smote_full.fit_resample(
        X_train, y_train
    )

    lgb_final = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_final.fit(X_train_resampled, y_train_resampled)

    # --- EVALUATE ON TEST SET ---
    print("\n   Evaluating on test set...")
    lgb_results = evaluate_model(
        lgb_final, X_test, y_test, "LightGBM"
    )
    lgb_results["cv_auc_mean"] = round(np.mean(cv_aucs), 4)
    lgb_results["cv_auc_std"]  = round(np.std(cv_aucs), 4)

    # --- SAVE MODEL ---
    model_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    joblib.dump(lgb_final, model_path)
    print(f"\n   Model saved: {model_path}")

    return lgb_final, lgb_results