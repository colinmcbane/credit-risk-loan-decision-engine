"""
utils/scorer.py

Loads Phase 4 scored loan data (shap_values_sample.csv) which contains
predicted default probabilities and SHAP-derived features for 5,000 loans.
Serves as the scoring layer for the Phase 6 AI Decision Engine.

Design note: Rather than re-running the LightGBM champion model, Phase 6
consumes the predictions already computed and validated in Phase 4. This
mirrors real MLOps pipelines where scoring and decisioning are separate layers.
"""

import os
import pandas as pd


# ── Path constants ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SHAP_FILE = os.path.join(DATA_DIR, "shap_values_sample.csv")

# Columns that are outputs, not features — exclude from feature matrix
NON_FEATURE_COLS = ["predicted_prob", "actual_default"]

# All feature columns the Phase 4 champion model was trained on
FEATURE_COLUMNS = [
    "term_months",
    "interest_rate",
    "loan_amount",
    "sba_guarantee_pct",
    "business_age_mature",
    "loan_size_bucket_micro",
    "naics_sector_62",
    "loan_size_bucket_large",
    "borr_state_FL",
    "business_age_startup",
    "borr_state_TX",
    "jobs_supported",
    "business_age_established",
    "loan_size_bucket_small",
    "borr_state_CA",
    "loan_size_bucket_medium",
    "business_age_new",
    "naics_sector_48",
    "borr_state_WI",
    "borr_state_NJ",
    "borr_state_NY",
    "naics_sector_71",
    "naics_sector_52",
    "borr_state_WA",
    "naics_sector_45",
    "borr_state_MN",
]


def load_scored_loans() -> pd.DataFrame:
    """
    Load the Phase 4 scored loan dataset from disk.

    Returns a DataFrame containing all feature columns plus predicted_prob
    and actual_default. Each row represents one loan applicant.

    Returns
    -------
    pd.DataFrame
        Scored loan data with shape (n_loans, n_features + 2).

    Raises
    ------
    FileNotFoundError
        If shap_values_sample.csv is not found in data/processed/.
    ValueError
        If expected columns are missing from the loaded file.
    """
    if not os.path.exists(SHAP_FILE):
        raise FileNotFoundError(
            f"Scored loan file not found at: {SHAP_FILE}\n"
            "Ensure Phase 4 completed successfully and saved shap_values_sample.csv."
        )

    df = pd.read_csv(SHAP_FILE)

    # Validate all expected feature columns are present
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"[scorer] Missing expected feature columns: {missing}\n"
            "The shap_values_sample.csv file may be from a different pipeline version."
        )

    if "predicted_prob" not in df.columns:
        raise ValueError(
            "[scorer] 'predicted_prob' column not found. "
            "Ensure Phase 4 saved predicted probabilities to shap_values_sample.csv."
        )

    print(f"[scorer] Loaded {len(df):,} scored loans from shap_values_sample.csv")
    print(f"[scorer] Mean predicted default prob: {df['predicted_prob'].mean():.4f}")
    print(f"[scorer] Predicted prob range: "
          f"{df['predicted_prob'].min():.4f} – {df['predicted_prob'].max():.4f}")

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the feature matrix from the scored loan DataFrame.

    Drops output columns (predicted_prob, actual_default) and returns
    only the columns the champion model was trained on.

    Parameters
    ----------
    df : pd.DataFrame
        Full scored loan DataFrame returned by load_scored_loans().

    Returns
    -------
    pd.DataFrame
        Feature matrix with shape (n_loans, 26).
    """
    return df[FEATURE_COLUMNS].copy()


def get_predictions(df: pd.DataFrame) -> pd.Series:
    """
    Extract predicted default probabilities from the scored loan DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full scored loan DataFrame returned by load_scored_loans().

    Returns
    -------
    pd.Series
        Predicted default probabilities indexed to match df.
    """
    return df["predicted_prob"].copy()