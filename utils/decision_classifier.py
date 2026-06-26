"""
utils/decision_classifier.py

Classifies loan decisions based on predicted default probabilities and
extracts top SHAP-derived reason codes for adverse action letters.

Decision thresholds (tunable via constants):
    >= DENIAL_THRESHOLD        → Denied
    >= UNFAVORABLE_THRESHOLD   → Approved with Unfavorable Terms
    < UNFAVORABLE_THRESHOLD    → Approved

Adverse action letters are required for Denied and Unfavorable Terms
decisions per ECOA (Equal Credit Opportunity Act) Regulation B.

Note on SHAP columns: shap_values_sample.csv stores SHAP values in columns
named after features (e.g. 'term_months', 'loan_amount'). The values in
these columns are SHAP impact scores, not raw feature values. Positive SHAP
values indicate the feature pushed the prediction toward default.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


# ── Decision thresholds ───────────────────────────────────────────────────────
DENIAL_THRESHOLD      = 0.50   # >= this → Denied
UNFAVORABLE_THRESHOLD = 0.30   # >= this and < DENIAL_THRESHOLD → Unfavorable Terms

# ── Decision labels ───────────────────────────────────────────────────────────
DECISION_DENIED       = "Denied"
DECISION_UNFAVORABLE  = "Approved - Unfavorable Terms"
DECISION_APPROVED     = "Approved"

# ── SHAP feature → ECOA-compliant reason code mapping ────────────────────────
# Maps feature column names to human-readable adverse action reason strings.
# These strings are injected into the Gemini letter generation prompt.
FEATURE_REASON_MAP: Dict[str, str] = {
    "term_months":              "Insufficient Loan Term Length",
    "interest_rate":            "Elevated Interest Rate Risk",
    "loan_amount":              "Loan Amount Outside Acceptable Range",
    "sba_guarantee_pct":        "Insufficient SBA Guarantee Coverage",
    "business_age_mature":      "Business Maturity Profile",
    "business_age_startup":     "Startup Risk Profile",
    "business_age_new":         "Early-Stage Business Risk",
    "business_age_established": "Business Establishment Risk Factor",
    "loan_size_bucket_micro":   "Micro Loan Size Risk",
    "loan_size_bucket_small":   "Small Loan Size Risk",
    "loan_size_bucket_medium":  "Medium Loan Size Risk",
    "loan_size_bucket_large":   "Large Loan Amount Risk",
    "jobs_supported":           "Insufficient Employment Impact",
    "naics_sector_45":          "Industry Sector Risk (Retail)",
    "naics_sector_48":          "Industry Sector Risk (Transportation)",
    "naics_sector_52":          "Industry Sector Risk (Finance)",
    "naics_sector_62":          "Industry Sector Risk (Healthcare)",
    "naics_sector_71":          "Industry Sector Risk (Arts and Entertainment)",
    "borr_state_CA":            "Geographic Risk Factor (California)",
    "borr_state_FL":            "Geographic Risk Factor (Florida)",
    "borr_state_MN":            "Geographic Risk Factor (Minnesota)",
    "borr_state_NJ":            "Geographic Risk Factor (New Jersey)",
    "borr_state_NY":            "Geographic Risk Factor (New York)",
    "borr_state_TX":            "Geographic Risk Factor (Texas)",
    "borr_state_WA":            "Geographic Risk Factor (Washington)",
    "borr_state_WI":            "Geographic Risk Factor (Wisconsin)",
}

# Feature columns present in shap_values_sample.csv
# Values in these columns are SHAP scores, not raw feature values
FEATURE_COLUMNS = [
    "term_months", "interest_rate", "loan_amount", "sba_guarantee_pct",
    "business_age_mature", "loan_size_bucket_micro", "naics_sector_62",
    "loan_size_bucket_large", "borr_state_FL", "business_age_startup",
    "borr_state_TX", "jobs_supported", "business_age_established",
    "loan_size_bucket_small", "borr_state_CA", "loan_size_bucket_medium",
    "business_age_new", "naics_sector_48", "borr_state_WI", "borr_state_NJ",
    "borr_state_NY", "naics_sector_71", "naics_sector_52", "borr_state_WA",
    "naics_sector_45", "borr_state_MN",
]


def classify_decision(predicted_prob: float) -> str:
    """
    Classify a single loan decision based on predicted default probability.

    Parameters
    ----------
    predicted_prob : float
        Predicted probability of default (0–1) from the champion model.

    Returns
    -------
    str
        One of: 'Denied', 'Approved - Unfavorable Terms', 'Approved'.
    """
    if predicted_prob >= DENIAL_THRESHOLD:
        return DECISION_DENIED
    elif predicted_prob >= UNFAVORABLE_THRESHOLD:
        return DECISION_UNFAVORABLE
    else:
        return DECISION_APPROVED


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify decisions for all loans in the scored DataFrame.

    Adds a 'decision' column and a 'requires_adverse_action' boolean
    column to the DataFrame based on predicted_prob values.

    Parameters
    ----------
    df : pd.DataFrame
        Scored loan DataFrame containing a 'predicted_prob' column.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with two new columns added:
            - decision: str decision label per applicant
            - requires_adverse_action: bool, True if letter required
    """
    df = df.copy()
    df["decision"] = df["predicted_prob"].apply(classify_decision)
    df["requires_adverse_action"] = df["decision"].isin(
        [DECISION_DENIED, DECISION_UNFAVORABLE]
    )

    counts = df["decision"].value_counts()
    total  = len(df)
    print(f"[classifier] Decision distribution ({total:,} loans):")
    for label, count in counts.items():
        print(f"  {label}: {count:,} ({count / total * 100:.1f}%)")
    print(f"[classifier] Adverse action letters required: "
          f"{df['requires_adverse_action'].sum():,}")

    return df


def extract_reason_codes(row: pd.Series, n_reasons: int = 3) -> List[str]:
    """
    Extract the top N adverse action reason codes for a single applicant.

    Reads SHAP values directly from the feature columns in
    shap_values_sample.csv. Positive SHAP values indicate the feature
    pushed the model's prediction toward default. Only positive
    contributions are considered as adverse action reason codes,
    consistent with ECOA Regulation B requirements.

    Parameters
    ----------
    row : pd.Series
        A single row from the classified loan DataFrame. Feature columns
        contain SHAP impact scores, not raw feature values.
    n_reasons : int
        Number of reason codes to return (default: 3).

    Returns
    -------
    List[str]
        Human-readable reason strings sorted by SHAP impact, highest first.
    """
    reason_scores: List[Tuple[float, str]] = []

    for feature in FEATURE_COLUMNS:
        if feature not in row.index:
            continue

        shap_value = float(row[feature])

        # Only positive SHAP values contributed to the denial —
        # negative values actually reduced default risk for this applicant
        if shap_value > 0:
            reason = FEATURE_REASON_MAP.get(feature, feature)
            reason_scores.append((shap_value, reason))

    # Sort by magnitude descending — strongest risk drivers first
    reason_scores.sort(key=lambda x: x[0], reverse=True)
    top_reasons = [reason for _, reason in reason_scores[:n_reasons]]

    # Fallback in the rare case no features had positive SHAP values
    if not top_reasons:
        top_reasons = [
            "Overall Credit Risk Profile",
            "Loan Characteristics",
            "Applicant Risk Factors",
        ]

    return top_reasons


def extract_all_reason_codes(df: pd.DataFrame,
                              n_reasons: int = 3) -> pd.Series:
    """
    Extract reason codes for all loans requiring adverse action letters.

    Returns an empty list for loans that do not require adverse action.

    Parameters
    ----------
    df : pd.DataFrame
        Classified loan DataFrame with 'requires_adverse_action' column.
    n_reasons : int
        Number of reason codes per applicant (default: 3).

    Returns
    -------
    pd.Series
        Series of List[str] reason codes indexed to match df.
    """
    def _get_reasons(row: pd.Series) -> List[str]:
        if not row.get("requires_adverse_action", False):
            return []
        return extract_reason_codes(row, n_reasons)

    reason_series = df.apply(_get_reasons, axis=1)
    reason_series.name = "reason_codes"

    print(f"[classifier] Reason codes extracted for "
          f"{df['requires_adverse_action'].sum():,} adverse action loans.")

    return reason_series