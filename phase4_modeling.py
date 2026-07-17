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

# SMOTE sampling strategy — 4:1 ratio
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
    print("\n[2/8] Training Logistic Regression baseline...")

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
    print("\n[3/8] Training XGBoost...")

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
    print("\n[4/8] Training LightGBM...")

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

# =============================================================================
# STEP 6: MODEL COMPARISON — CHAMPION-CHALLENGER FRAMEWORK
# =============================================================================

def compare_models(lr_results, xgb_results, lgb_results,
                   X_test, y_test):
    print("\n[5/8] Champion-Challenger model comparison...")

    all_results = [lr_results, xgb_results, lgb_results]

    # --- BUILD COMPARISON TABLE ---
    comparison = pd.DataFrame([{
        "Model":        r["model"],
        "AUC":          r["auc"],
        "KS Statistic": r["ks"],
        "Gini":         r["gini"],
        "Precision":    r["precision"],
        "Recall":       r["recall"],
        "F1":           r["f1"],
        "CV AUC Mean":  r["cv_auc_mean"],
        "CV AUC Std":   r["cv_auc_std"],
        "True Pos":     r["tp"],
        "False Pos":    r["fp"],
        "True Neg":     r["tn"],
        "False Neg":    r["fn"],
    } for r in all_results])

    print("\n   --- Champion-Challenger Comparison Table ---")
    print(comparison.to_string(index=False))

    # --- DECLARE CHAMPION ---
    # Champion selected by highest AUC on held-out test set
    # AUC is threshold-independent — most reliable single metric
    # for credit risk model selection under SR 11-7
    champion_idx  = comparison["AUC"].idxmax()
    champion_name = comparison.loc[champion_idx, "Model"]
    champion_auc  = comparison.loc[champion_idx, "AUC"]

    print(f"\n   === CHAMPION MODEL: {champion_name} ===")
    print(f"   AUC:  {champion_auc:.4f}")
    print(f"   KS:   {comparison.loc[champion_idx, 'KS Statistic']:.2f}")
    print(f"   Gini: {comparison.loc[champion_idx, 'Gini']:.2f}%")
    print(f"\n   Champion drives Phase 6 decision engine.")
    print(f"   Logistic Regression retained as interpretable baseline.")

    # --- SAVE COMPARISON TABLE ---
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\n   Comparison table saved: {comparison_path}")

    # --- ROC CURVE CHART ---
    print("\n   Generating ROC curve chart...")

    plt.figure(figsize=(10, 8))

    colors = {
        "Logistic Regression": "#9E9E9E",
        "XGBoost":             "#1565C0",
        "LightGBM":            "#2E7D32"
    }

    for r in all_results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        plt.plot(
            fpr, tpr,
            label=f"{r['model']} (AUC={r['auc']:.4f}, "
                  f"KS={r['ks']:.1f}, Gini={r['gini']:.1f}%)",
            color=colors[r["model"]],
            linewidth=2.5
        )

    # Random baseline
    plt.plot([0, 1], [0, 1], "k--",
             linewidth=1, label="Random (AUC=0.50)")

    # NOTE: Decision threshold line removed per model validator review
    # x-axis of ROC curve is False Positive Rate not probability threshold
    # Drawing axvline at x=0.35 would imply acceptable FPR=35% which is incorrect

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        "ROC Curves — Champion-Challenger Comparison\n"
        "SBA 7(a) Credit Risk Models",
        fontsize=13, fontweight="bold"
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(SHAP_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ROC curves saved: {roc_path}")

    # --- CONFUSION MATRIX CHART ---
    print("   Generating confusion matrix charts...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, r in zip(axes, all_results):
        cm = np.array([
            [r["tn"], r["fp"]],
            [r["fn"], r["tp"]]
        ])
        sns.heatmap(
            cm, annot=True, fmt=",",
            cmap="Blues", ax=ax,
            xticklabels=["Pred: Paid", "Pred: Default"],
            yticklabels=["Act: Paid", "Act: Default"]
        )
        ax.set_title(
            f"{r['model']}\n"
            f"AUC={r['auc']:.4f} | "
            f"KS={r['ks']:.1f} | "
            f"Gini={r['gini']:.1f}%",
            fontweight="bold"
        )

    plt.suptitle(
        "Confusion Matrices — All Models\n"
        f"Decision Threshold = {DECISION_THRESHOLD}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    cm_path = os.path.join(SHAP_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Confusion matrices saved: {cm_path}")

    # --- PRECISION RECALL CURVE ---
    print("   Generating precision-recall curve...")

    plt.figure(figsize=(10, 8))

    for r in all_results:
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test, r["y_prob"]
        )
        plt.plot(
            recall_vals, precision_vals,
            label=f"{r['model']} (AP={r['avg_precision']:.4f})",
            color=colors[r["model"]],
            linewidth=2.5
        )

    # Baseline — random classifier precision = default rate
    baseline = y_test.mean()
    plt.axhline(
        y=baseline, color="k", linestyle="--",
        linewidth=1,
        label=f"Baseline (default rate = {baseline:.2%})"
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(
        "Precision-Recall Curves — All Models\n"
        "SBA 7(a) Credit Risk Models",
        fontsize=13, fontweight="bold"
    )
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pr_path = os.path.join(SHAP_DIR, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Precision-recall curves saved: {pr_path}")

    return comparison, champion_name


# =============================================================================
# STEP 7: SHAP EXPLAINABILITY
# =============================================================================

def generate_shap_explanations(champion_model, champion_name,
                                X_test, y_test):
    print(f"\n[6/8] Generating SHAP explanations for {champion_name}...")

    # --- SHAP EXPLAINER ---
    # TreeExplainer optimized for tree-based models (XGBoost, LightGBM)
    # LinearExplainer used for Logistic Regression
    print("   Building SHAP explainer...")

    if champion_name == "Logistic Regression":
        explainer = shap.LinearExplainer(
            champion_model,
            X_test,
            feature_perturbation="interventional"
        )
    else:
        explainer = shap.TreeExplainer(champion_model)

    # Calculate SHAP values on test set sample
    # 5000 rows is statistically representative of 95,536 test rows
    print("   Calculating SHAP values (sample of 5,000 rows)...")

    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_test), size=5000, replace=False
    )
    X_sample = X_test.iloc[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    # Handle LightGBM returning list of arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"   SHAP values shape: {shap_values.shape}")

    # --- CLEAN FEATURE NAMES FOR DISPLAY ---
    feature_label_map = {
        "term_months":              "Term Length (Months)",
        "interest_rate":            "Interest Rate",
        "loan_amount":              "Loan Amount",
        "sba_guarantee_pct":        "SBA Guarantee %",
        "business_age_mature":      "Business Age: Mature",
        "loan_size_bucket_micro":   "Loan Size: Micro",
        "naics_sector_62":          "Industry: Health Care",
        "loan_size_bucket_large":   "Loan Size: Large",
        "borr_state_FL":            "State: Florida",
        "business_age_startup":     "Business Age: Startup",
        "borr_state_TX":            "State: Texas",
        "jobs_supported":           "Jobs Supported",
        "business_age_established": "Business Age: Established",
        "loan_size_bucket_small":   "Loan Size: Small",
        "borr_state_CA":            "State: California",
        "loan_size_bucket_medium":  "Loan Size: Medium",
        "business_age_new":         "Business Age: New",
        "naics_sector_48":          "Industry: Transportation",
        "borr_state_WI":            "State: Wisconsin",
        "borr_state_NJ":            "State: New Jersey",
        "borr_state_NY":            "State: New York",
        "naics_sector_71":          "Industry: Arts & Entertainment",
        "naics_sector_52":          "Industry: Finance & Insurance",
        "borr_state_WA":            "State: Washington",
        "naics_sector_45":          "Industry: Specialty Retail",
        "borr_state_MN":            "State: Minnesota",
    }

    X_sample_display = X_sample.rename(columns=feature_label_map)

    # --- SAFE BASE VALUE EXTRACTION ---
    # Handles both Python lists and numpy arrays from LightGBM
    # np.atleast_1d() converts scalar or array safely before indexing
    # Prevents shape dimension errors in waterfall plot
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)) and \
            len(np.atleast_1d(base_val)) == 2:
        base_val = np.atleast_1d(base_val)[1]

    # --- CREATE EXPLANATION OBJECT ---
    # Modern SHAP requires Explanation object for summary plots
    # Prevents index mismatch between raw numpy arrays
    # and renamed DataFrames in current SHAP versions
    sample_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_val,
        data=X_sample_display.values,
        feature_names=list(X_sample_display.columns)
    )

    # --- GLOBAL FEATURE IMPORTANCE ---
    print("   Generating global SHAP importance chart...")

    fig = plt.figure(figsize=(14, 11))
    shap.plots.bar(
        sample_explanation,
        max_display=20,
        show=False
    )
    ax = plt.gca()
    ax.set_title(
        f"Global SHAP Feature Importance — {champion_name}\n"
        "Mean absolute SHAP value across 5,000 test loans",
        fontweight="bold"
    )
    ax.set_xlabel(
        "mean(|SHAP value|)\n(average impact on model output magnitude)",
        labelpad=20,
        fontsize=12
    )
    ax.xaxis.set_label_coords(0.5, -0.12)
    fig.subplots_adjust(bottom=0.32, top=0.88, left=0.35, right=0.98)
    fig.savefig(
        os.path.join(SHAP_DIR, "shap_global_importance.png"),
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.35
    )
    plt.close(fig)
    print("   Global SHAP chart saved")

    # --- SHAP BEESWARM PLOT ---
    # Each dot is one loan
    # Red = high feature value, Blue = low feature value
    # X-axis = impact on default probability
    print("   Generating SHAP beeswarm plot...")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        sample_explanation,
        show=False,
        max_display=20
    )
    plt.title(
        f"SHAP Value Distribution — {champion_name}\n"
        "Red = high feature value | Blue = low feature value",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_DIR, "shap_beeswarm.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   SHAP beeswarm chart saved")

    # --- INDIVIDUAL LOAN WATERFALL ---
    # Highest risk loan in sample — fed into Gemini API in Phase 6
    print("   Generating waterfall chart for highest risk loan...")

    y_prob_sample = champion_model.predict_proba(X_sample)[:, 1]
    highest_risk_idx = np.argmax(y_prob_sample)

    shap_explanation = shap.Explanation(
        values=shap_values[highest_risk_idx],
        base_values=base_val,
        data=X_sample_display.iloc[highest_risk_idx],
        feature_names=list(X_sample_display.columns)
    )

    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_explanation, show=False, max_display=15)
    plt.title(
        f"SHAP Waterfall — Highest Risk Loan\n"
        f"Predicted Default Probability: "
        f"{y_prob_sample[highest_risk_idx]:.2%}",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(SHAP_DIR, "shap_waterfall_high_risk.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   Waterfall chart saved")

    # --- SAVE SHAP VALUES FOR PHASE 6 ---
    # Phase 6 reads this to generate Adverse Action letters
    # without re-running SHAP computation
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_sample.columns
    )
    shap_df["predicted_prob"] = y_prob_sample
    shap_df["actual_default"] = y_test.iloc[sample_idx].values
    shap_df.to_csv(
        os.path.join(OUTPUT_DIR, "shap_values_sample.csv"),
        index=False
    )
    print("   SHAP values saved for Phase 6")

    print(f"\n   SHAP outputs saved to: {SHAP_DIR}")

    return explainer, shap_values, X_sample

# =============================================================================
# STEP 8: STRESS TESTING
# =============================================================================

def run_stress_tests(champion_model, X_test, y_test, scaler_params):
    print("\n[7/8] Running portfolio stress tests...")

    # --- SAFE SCALER PARAMETERS EXTRACTION ---
    # Validates scaler_params is a DataFrame before indexing
    # Prevents AttributeError if wrong object type is passed
    if isinstance(scaler_params, pd.DataFrame):
        scaler_dict = scaler_params.set_index("feature")["std"].to_dict()
    else:
        raise TypeError(
            "scaler_params must be a pandas DataFrame with "
            "'feature' and 'std' columns."
        )

    def get_std(feature):
        return scaler_dict.get(feature, 1.0)

    # --- DEFINE STRESS SCENARIOS ---
    # Shocks defined in raw units then divided by feature std
    # to convert to standardized space correctly
    # Example: 2% rate hike = 2.0 / std(interest_rate) in scaled space
    #
    # loan_amount intentionally excluded from all stress scenarios
    # Reducing loan_amount on existing portfolio loans would lower
    # modeled risk — economically backwards for recession stress testing
    # term_months extensions used instead to capture liquidity drag
    # jobs_supported negative shock simulates systemic business downscaling
    #
    # Known limitation: tree-based models handle out-of-distribution
    # feature values by capping at furthest leaf node — severe scenarios
    # may underestimate true tail risk due to this plateauing effect
    # Documented here per SR 11-7 model limitation disclosure requirements
    scenarios = {
        "Baseline": {},
        "Rate Hike +200bps": {
            "interest_rate": 2.0 / get_std("interest_rate")
        },
        "Rate Hike +400bps (2022-2023 analog)": {
            "interest_rate": 4.0 / get_std("interest_rate")
        },
        "Pandemic Operational Distress": {
            "interest_rate": 1.0 / get_std("interest_rate"),
            "term_months":   3.0 / get_std("term_months")
        },
        "Severe Macroeconomic Recession": {
            "interest_rate":  4.0 / get_std("interest_rate"),
            "term_months":    6.0 / get_std("term_months"),
            "jobs_supported": -1.5 / get_std("jobs_supported")
        }
    }

    results = []
    print(f"   Running {len(scenarios)} stress scenarios...")

    for scenario_name, adjustments in scenarios.items():
        X_stressed = X_test.copy()

        # Apply standardized shocks
        for feature, shock in adjustments.items():
            if feature in X_stressed.columns:
                X_stressed[feature] = X_stressed[feature] + shock

        # Score stressed portfolio
        stressed_prob = champion_model.predict_proba(X_stressed)[:, 1]
        stressed_pred = (stressed_prob >= DECISION_THRESHOLD).astype(int)

        mean_pd               = stressed_prob.mean()
        high_risk_pct         = (stressed_prob > 0.5).mean()
        predicted_default_rate = stressed_pred.mean()

        # Fixed at 20% — SBA 7(a) guarantees approximately 80% of loan value
        # Cannot calculate from scaled sba_guarantee_pct — scaler transforms
        # to mean=0 which produces LGD values above 100% (mathematically invalid)
        # 0.20 is conservative and consistent with SBA program documentation
        avg_lgd = 0.20
        expected_loss_rate = mean_pd * avg_lgd

        results.append({
            "Scenario":               scenario_name,
            "Mean PD":                round(mean_pd * 100, 2),
            "High Risk % (PD>50%)":   round(high_risk_pct * 100, 2),
            "Predicted Default Rate": round(predicted_default_rate * 100, 2),
            "Avg LGD":                20.0,
            "Expected Loss Rate":     round(expected_loss_rate * 100, 2),
        })

        print(f"   {scenario_name}:")
        print(f"      Mean PD:            {mean_pd:.2%}")
        print(f"      High Risk (PD>50%): {high_risk_pct:.2%}")
        print(f"      Expected Loss Rate: {expected_loss_rate:.2%}")

    # --- BUILD RESULTS TABLE ---
    stress_df = pd.DataFrame(results)

    print("\n   --- Stress Test Summary ---")
    print(stress_df.to_string(index=False))

    # --- SAVE RESULTS ---
    stress_path = os.path.join(OUTPUT_DIR, "stress_test_results.csv")
    stress_df.to_csv(stress_path, index=False)
    print(f"\n   Stress results saved: {stress_path}")

    # --- VISUALIZE ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["Mean PD", "High Risk % (PD>50%)", "Expected Loss Rate"]
    colors  = ["#1565C0", "#B71C1C", "#E65100"]
    titles  = [
        "Mean Probability of Default",
        "High Risk Loans (PD > 50%)",
        "Expected Loss Rate"
    ]

    for ax, metric, color, title in zip(axes, metrics, colors, titles):
        bars = ax.barh(
            stress_df["Scenario"],
            stress_df[metric],
            color=color,
            alpha=0.8
        )
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
        ax.set_xlabel(f"{metric} (%)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, stress_df[metric].max() * 1.35)

    plt.suptitle(
        "Portfolio Stress Test Results — SBA 7(a) Credit Risk Model\n"
        "Macroeconomic shock scenarios | SR 11-7 model limitation "
        "disclosure: tree model tail risk may be underestimated",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    stress_chart_path = os.path.join(SHAP_DIR, "stress_test_results.png")
    plt.savefig(stress_chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Stress chart saved: {stress_chart_path}")

    return stress_df

# =============================================================================
# STEP 9: SAVE CHAMPION MODEL AND RESULTS
# =============================================================================

def save_champion(champion_model, champion_name, comparison):
    print("\n[8/8] Saving champion model and final results...")

    # --- SAVE CHAMPION MODEL ---
    # Saved separately so Phase 6 loads champion directly
    # without needing to know which algorithm won
    # Phase 6 reads champion_model.pkl regardless of whether
    # XGBoost or LightGBM won the challenger comparison
    champion_path = os.path.join(MODELS_DIR, "champion_model.pkl")
    joblib.dump(champion_model, champion_path)
    print(f"   Champion model saved: {champion_path}")

    # --- SAVE CHAMPION NAME ---
    # Plain text file so Phase 6 can read which model won
    # without loading the full model object into memory
    champion_name_path = os.path.join(MODELS_DIR, "champion_name.txt")
    with open(champion_name_path, "w") as f:
        f.write(champion_name)
    print(f"   Champion name saved: {champion_name_path}")

    # --- FINAL SUMMARY ---
    print("\n=== Phase 4 Complete ===")
    print(f"\n   Champion Model: {champion_name}")
    print(f"\n   --- Final Model Comparison ---")
    print(comparison[["Model", "AUC", "KS Statistic",
                       "Gini", "Recall", "F1"]].to_string(index=False))

    # --- AUDIT TRAIL ---
    # Paths generated from actual directory variables
    # ensures printout matches real file locations
    # SR 11-7 requires audit trail documentation to be accurate
    print("\n   --- Files Saved ---")
    print(f"   {os.path.join(MODELS_DIR, 'logistic_regression.pkl')}")
    print(f"   {os.path.join(MODELS_DIR, 'xgboost.pkl')}")
    print(f"   {os.path.join(MODELS_DIR, 'lightgbm.pkl')}")
    print(f"   {os.path.join(MODELS_DIR, 'champion_model.pkl')}")
    print(f"   {os.path.join(MODELS_DIR, 'champion_name.txt')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'shap_values_sample.csv')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'stress_test_results.csv')}")
    print(f"   {os.path.join(SHAP_DIR, 'roc_curves.png')}")
    print(f"   {os.path.join(SHAP_DIR, 'confusion_matrices.png')}")
    print(f"   {os.path.join(SHAP_DIR, 'precision_recall_curves.png')}")
    print(f"   {os.path.join(SHAP_DIR, 'shap_global_importance.png')}")
    print(f"   {os.path.join(SHAP_DIR, 'shap_beeswarm.png')}")
    print(f"   {os.path.join(SHAP_DIR, 'shap_waterfall_high_risk.png')}")
    print(f"   {os.path.join(SHAP_DIR, 'stress_test_results.png')}")
    print("\nReady for Phase 5 — Fairness Analysis")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Step 1 — Load data
    X_train, y_train, X_test, y_test, scaler_params = load_data()

    # Step 2 — Evaluation utilities already defined above

    # Step 3 — Logistic Regression baseline
    lr_model, lr_results = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )

    # Step 4 — XGBoost
    xgb_model, xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    # Step 5 — LightGBM
    lgb_model, lgb_results = train_lightgbm(
        X_train, y_train, X_test, y_test
    )

    # Step 6 — Champion-Challenger comparison
    comparison, champion_name = compare_models(
        lr_results, xgb_results, lgb_results,
        X_test, y_test
    )

    # Step 7 — Identify champion model object
    # Maps champion name string to actual trained model object
    # compare_models returns name only — this retrieves the object
    champion_map = {
        "Logistic Regression": lr_model,
        "XGBoost":             xgb_model,
        "LightGBM":            lgb_model
    }
    champion_model = champion_map[champion_name]

    # Step 8 — SHAP explainability on champion only
    # TreeExplainer for XGBoost/LightGBM
    # LinearExplainer for Logistic Regression
    explainer, shap_values, X_sample = generate_shap_explanations(
        champion_model, champion_name, X_test, y_test
    )

    # Step 9 — Stress testing on champion portfolio
    # scaler_params passed as DataFrame — validated inside function
    stress_df = run_stress_tests(
        champion_model, X_test, y_test, scaler_params
    )

    # Step 10 — Save champion and final results
    save_champion(champion_model, champion_name, comparison)