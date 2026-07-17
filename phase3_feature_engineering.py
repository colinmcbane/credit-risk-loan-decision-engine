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
import joblib
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

    # Save raw group columns for Phase 5 fairness analysis
    # X_train.index / X_test.index are the exact rows selected from df_train_pool
    # so row order here matches train_features.csv / test_features.csv exactly
    _group_cols = ["borr_state", "business_age", "naics_sector", "loan_amount", "loan_size_bucket"]
    _group_rename = {
        "borr_state":   "borrstate",
        "business_age": "businessage",
        "naics_sector": "naicscode",
        "loan_amount":  "grossapproval",
    }
    df_train_pool.loc[X_train.index, _group_cols].rename(columns=_group_rename).to_csv(
        os.path.join(OUTPUT_DIR, "train_groups.csv"), index=False
    )
    df_train_pool.loc[X_test.index, _group_cols].rename(columns=_group_rename).to_csv(
        os.path.join(OUTPUT_DIR, "test_groups.csv"), index=False
    )
    print(f"   Group columns saved: train_groups.csv ({len(X_train):,} rows), "
          f"test_groups.csv ({len(X_test):,} rows)")

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

    # --- SAVE SCALER PARAMETERS ---
    # Required for Phase 4 stress testing
    # Translates raw economic shocks into standardized units
    # Example: a 2% rate hike becomes 2.0 / std(interest_rate) in scaled space
    scaler_params = pd.DataFrame({
        "feature": numeric_cols_present,
        "mean":    scaler.mean_,
        "std":     scaler.scale_
    })
    scaler_params.to_csv(
        os.path.join(OUTPUT_DIR, "scaler_params.csv"),
        index=False
    )
    print(f"   Scaler parameters saved: {len(numeric_cols_present)} features")

    # --- SAVE SCALER OBJECT ---
    # Full fitted scaler saved for Phase 6 production scoring
    # Phase 6 must apply identical preprocessing to new loan applications
    # before running them through the trained model
    # SR 11-7 requires preprocessing steps to be documented and reproducible
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print("   Scaler object saved: data/processed/scaler.pkl")

    return X_train_encoded, X_test_encoded, scaler

# =============================================================================
# STEP 5: FEATURE IMPORTANCE RANKING
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

def rank_feature_importance(X_train, y_train):
    print("\n[5/5] Ranking feature importance...")

    # --- BASELINE RANDOM FOREST ---
    # We use a shallow Random Forest purely as a feature importance tool
    # Not the final model — just a fast way to rank signal strength
    # max_depth=5 keeps it fast — we don't need a perfect model here
    # n_estimators=100 gives stable importance estimates
    print("   Training baseline Random Forest for feature importance...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # --- EXTRACT IMPORTANCE SCORES ---
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n   Top 20 features by importance:")
    print(importance_df.head(20).to_string(index=False))

    # --- IDENTIFY LOW SIGNAL FEATURES ---
    # Features with importance below threshold contribute noise not signal
    # Threshold of 0.001 removes features that explain less than 0.1% of variance
    IMPORTANCE_THRESHOLD = 0.001
    low_signal = importance_df[
        importance_df["importance"] < IMPORTANCE_THRESHOLD
    ]["feature"].tolist()

    print(f"\n   Low signal features (importance < {IMPORTANCE_THRESHOLD}):")
    print(f"   {low_signal}")
    print(f"   Dropping {len(low_signal)} low signal features")

    # --- DROP LOW SIGNAL FEATURES ---
    # This replaces PCA for dimensionality reduction
    # Raw features are preserved for SHAP explainability in Phase 4
    # PCA components cannot be explained to regulators or underwriters
    keep_features = importance_df[
        importance_df["importance"] >= IMPORTANCE_THRESHOLD
    ]["feature"].tolist()

    print(f"   Keeping {len(keep_features)} features")

    # --- SAVE IMPORTANCE CHART ---
    top_features = importance_df.head(25)

    # Clean up feature names for display
    label_map = {
        "term_months":              "Term Length (Months)",
        "interest_rate":            "Interest Rate",
        "loan_amount":              "Loan Amount",
        "sba_guarantee_pct":        "SBA Guarantee %",
        "business_age_mature":      "Business Age: Mature (5+ Yrs)",
        "loan_size_bucket_micro":   "Loan Size: Micro (<$150K)",
        "naics_sector_62":          "Industry: Health Care",
        "loan_size_bucket_large":   "Loan Size: Large ($1M-$2M)",
        "borr_state_FL":            "State: Florida",
        "business_age_startup":     "Business Age: Startup",
        "borr_state_TX":            "State: Texas",
        "jobs_supported":           "Jobs Supported",
        "business_age_established": "Business Age: Established",
        "loan_size_bucket_small":   "Loan Size: Small ($150K-$500K)",
        "borr_state_CA":            "State: California",
        "loan_size_bucket_medium":  "Loan Size: Medium ($500K-$1M)",
        "business_age_new":         "Business Age: New (<2 Yrs)",
        "naics_sector_48":          "Industry: Transportation",
        "borr_state_WI":            "State: Wisconsin",
        "borr_state_NJ":            "State: New Jersey",
        "borr_state_NY":            "State: New York",
        "naics_sector_71":          "Industry: Arts & Entertainment",
        "naics_sector_52":          "Industry: Finance & Insurance",
        "borr_state_WA":            "State: Washington",
        "naics_sector_45":          "Industry: Specialty Retail",
    }

    top_features["feature_label"] = top_features["feature"].map(
        label_map
    ).fillna(top_features["feature"])

    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(
        [x / top_features["importance"].max()
         for x in top_features["importance"]]
    )
    bars = plt.barh(
        top_features["feature_label"][::-1],
        top_features["importance"][::-1],
        color=colors[::-1]
    )
    plt.xlabel("Feature Importance Score")
    plt.title("Top 25 Features by Random Forest Importance\nSBA 7(a) Credit Risk Model")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHARTS_DIR, "feature_importance.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
    print(f"\n   Chart saved to: outputs/eda_charts/feature_importance.png")

    # --- PCA FOR VISUALIZATION ONLY ---
    # PCA used here purely to understand variance structure
    # NOT used to compress features for modeling
    # Raw features retained for SHAP explainability
    print("\n   Running PCA for variance visualization...")

    pca = PCA(random_state=42)
    pca.fit(X_train)

    explained_variance = pd.DataFrame({
        "component": range(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance": pca.explained_variance_ratio_,
        "cumulative_variance": pca.explained_variance_ratio_.cumsum()
    })

    # Find how many components explain 90% of variance
    n_components_90 = (
        explained_variance["cumulative_variance"] < 0.90
    ).sum() + 1

    print(f"   Components needed for 90% variance: {n_components_90}")
    print(f"   Top 5 component variance explained:")
    print(explained_variance.head(5).to_string(index=False))

    # Save PCA scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        explained_variance["component"][:30],
        explained_variance["cumulative_variance"][:30],
        marker="o", color="#1565C0", linewidth=2
    )
    plt.axhline(y=0.90, color="red", linestyle="--",
                label="90% variance threshold")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot — Cumulative Variance Explained\n(Visualization Only — Raw Features Used for Modeling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHARTS_DIR, "pca_scree_plot.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
    print(f"   Chart saved to: outputs/eda_charts/pca_scree_plot.png")

    return keep_features, importance_df

# =============================================================================
# STEP 6: SAVE FEATURE MATRIX
# =============================================================================

def save_feature_matrix(X_train, X_test, y_train, y_test,
                         keep_features, df_recent):
    print("\n[6/6] Saving feature matrix...")

    # --- APPLY FEATURE SELECTION ---
    # Keep only features that passed the importance threshold
    # Ensures train and test have identical columns
    keep_cols_train = [c for c in keep_features if c in X_train.columns]
    keep_cols_test  = [c for c in keep_features if c in X_test.columns]
    final_cols = [c for c in keep_cols_train if c in keep_cols_test]

    X_train_final = X_train[final_cols].copy()
    X_test_final  = X_test[final_cols].copy()

    print(f"   Final feature count: {len(final_cols)}")

    # --- ADD TARGET BACK ---
    X_train_final["is_default"] = y_train.values
    X_test_final["is_default"]  = y_test.values

    # --- SAVE TO CSV ---
    train_path = os.path.join(OUTPUT_DIR, "train_features.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test_features.csv")

    X_train_final.to_csv(train_path, index=False)
    X_test_final.to_csv(test_path,  index=False)

    print(f"   Training set saved: {X_train_final.shape}")
    print(f"   Test set saved:     {X_test_final.shape}")
    print(f"   Train path: {train_path}")
    print(f"   Test path:  {test_path}")

    # --- SAVE RECENT LOANS SEPARATELY ---
    # Right-censored loans saved for Phase 7 live prediction demo
    # These are genuinely unseen loans — never touched during training
    recent_path = os.path.join(OUTPUT_DIR, "recent_loans.csv")
    df_recent.to_csv(recent_path, index=False)
    print(f"   Recent loans saved: {df_recent.shape}")
    print(f"   Recent path: {recent_path}")

    # --- SAVE FEATURE LIST ---
    # Saved so Phase 4 knows exactly which columns to expect
    feature_list_path = os.path.join(OUTPUT_DIR, "feature_list.txt")
    with open(feature_list_path, "w") as f:
        for col in final_cols:
            f.write(col + "\n")
    print(f"   Feature list saved: {feature_list_path}")

    return X_train_final, X_test_final

# =============================================================================
# STEP 7: VALIDATION SUMMARY
# =============================================================================

def validate_output(X_train_final, X_test_final, importance_df):
    print("\n=== Phase 3 Validation Summary ===")

    print(f"\n   Training set:   {X_train_final.shape[0]:,} rows | "
          f"{X_train_final.shape[1] - 1} features")
    print(f"   Test set:       {X_test_final.shape[0]:,} rows | "
          f"{X_test_final.shape[1] - 1} features")
    print(f"   Train default:  "
          f"{X_train_final['is_default'].mean():.2%}")
    print(f"   Test default:   "
          f"{X_test_final['is_default'].mean():.2%}")

    print("\n   --- Top 10 Features for Phase 4 ---")
    print(importance_df.head(10).to_string(index=False))

    print("\n   --- Files Saved ---")
    print("   data/processed/train_features.csv")
    print("   data/processed/test_features.csv")
    print("   data/processed/recent_loans.csv")
    print("   data/processed/feature_list.txt")
    print("   outputs/eda_charts/feature_importance.png")
    print("   outputs/eda_charts/pca_scree_plot.png")

    print("\n=== Phase 3 Complete ===")
    print("Ready for Phase 4 — Credit Risk Modeling")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Step 1 — Load data
    df = load_data()

    # Step 2 — Clean features
    df = clean_features(df)

    # Step 3 — Train/test split
    X_train, X_test, y_train, y_test, FEATURE_COLS, df_recent = split_data(df)

    # Step 4 — Encode and scale
    X_train, X_test, scaler = encode_and_scale(X_train, X_test)

    # Step 5 — Feature importance and PCA visualization
    keep_features, importance_df = rank_feature_importance(X_train, y_train)

    # Step 6 — Save feature matrix
    X_train_final, X_test_final = save_feature_matrix(
        X_train, X_test, y_train, y_test, keep_features, df_recent
    )

    # Step 7 — Validate output
    validate_output(X_train_final, X_test_final, importance_df)