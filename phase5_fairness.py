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

# =============================================================================
# STEP 2: DISPARATE IMPACT ANALYSIS
# =============================================================================

def calculate_disparate_impact(raw_test, group_col, label,
                                min_count=None):
    """
    Calculate disparate impact ratio for a categorical group column.
    Uses volume-qualified reference group to prevent small-sample
    reference group trap identified in model validation review.

    Disparate Impact Ratio = approval_rate(group) / approval_rate(reference)
    ECOA threshold: ratio < 0.80 = prima facie disparate impact
    """
    if min_count is None:
        min_count = int(len(raw_test) * MIN_GROUP_PCT)

    # Calculate approval rate per group
    # Approval = model predicts NOT high risk (y_pred == 0)
    group_stats = raw_test.groupby(group_col).agg(
        total           = ("y_true", "count"),
        actual_defaults = ("y_true", "sum"),
        predicted_high_risk = ("y_pred", "sum"),
        mean_prob       = ("y_prob", "mean")
    ).reset_index()

    group_stats["actual_default_rate"] = (
        group_stats["actual_defaults"] / group_stats["total"]
    ).round(4)

    group_stats["approval_rate"] = (
        1 - group_stats["predicted_high_risk"] / group_stats["total"]
    ).round(4)

    group_stats["pct_of_total"] = (
        group_stats["total"] / len(raw_test)
    ).round(4)

    # --- VOLUME-QUALIFIED REFERENCE GROUP ---
    # Reference group must meet minimum volume threshold
    # Prevents small-sample groups from anchoring the comparison
    # and generating false compliance failures
    qualified = group_stats[
        group_stats["total"] >= min_count
    ].copy()

    if len(qualified) == 0:
        print(f"   WARNING: No groups meet minimum count for {label}")
        return group_stats, None, None

    # Reference group = highest approval rate among qualified groups
    ref_idx   = qualified["approval_rate"].idxmax()
    ref_group = qualified.loc[ref_idx, group_col]
    ref_rate  = qualified.loc[ref_idx, "approval_rate"]

    # Calculate disparate impact ratio for all qualified groups
    group_stats["di_ratio"] = (
        group_stats["approval_rate"] / ref_rate
    ).round(4)

    group_stats["di_flag"] = (
        group_stats["di_ratio"] < FOUR_FIFTHS_THRESHOLD
    )

    group_stats["reference_group"] = ref_group
    group_stats["reference_rate"]  = ref_rate

    # Count flags
    flagged = group_stats[
        (group_stats["di_flag"]) &
        (group_stats["total"] >= min_count)
    ]

    print(f"\n   {label}:")
    print(f"   Reference group: {ref_group} "
          f"(approval rate: {ref_rate:.2%})")
    print(f"   Groups tested: {len(qualified)}")
    print(f"   Groups flagged (DI < 0.80): {len(flagged)}")

    if len(flagged) > 0:
        print(f"   ⚠️  Flagged groups:")
        for _, row in flagged.iterrows():
            print(f"      {row[group_col]}: "
                  f"approval={row['approval_rate']:.2%} "
                  f"DI={row['di_ratio']:.3f}")

    return group_stats, ref_group, ref_rate


def run_disparate_impact_analysis(raw_test):
    print("\n[2/9] Running disparate impact analysis...")

    results = {}

    # --- BY STATE ---
    print("\n   --- Geographic Disparate Impact ---")
    state_di, state_ref, state_ref_rate = calculate_disparate_impact(
        raw_test, "borr_state", "State"
    )
    results["state"] = state_di
    state_di.to_csv(
        os.path.join(OUTPUT_DIR, "disparate_impact_by_state.csv"),
        index=False
    )

    # --- BY BUSINESS AGE ---
    print("\n   --- Business Age Disparate Impact ---")

    # Map raw business age values to consolidated categories
    # Same mapping used in Phase 3 feature engineering
    age_map = {
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
    raw_test["business_age_clean"] = (
        raw_test["businessage"].map(age_map).fillna("unknown")
    )

    age_di, age_ref, age_ref_rate = calculate_disparate_impact(
        raw_test, "business_age_clean", "Business Age"
    )
    results["business_age"] = age_di
    age_di.to_csv(
        os.path.join(OUTPUT_DIR, "disparate_impact_by_business_age.csv"),
        index=False
    )

    # --- BY NAICS SECTOR ---
    print("\n   --- Industry Disparate Impact ---")

    raw_test["naics_sector"] = (
        raw_test["naicscode"].astype(str).str[:2]
    )

    sector_di, sector_ref, sector_ref_rate = calculate_disparate_impact(
        raw_test, "naics_sector", "Industry Sector"
    )
    results["industry"] = sector_di
    sector_di.to_csv(
        os.path.join(OUTPUT_DIR, "disparate_impact_by_industry.csv"),
        index=False
    )

    # --- BY LOAN SIZE ---
    print("\n   --- Loan Size Disparate Impact ---")

    # Cast to string immediately after pd.cut to prevent categorical
    # index bleed in downstream groupby operations
    raw_test["loan_size_bucket"] = pd.cut(
        raw_test["grossapproval"],
        bins=[0, 150000, 500000, 1000000, 2000000, float("inf")],
        labels=["micro", "small", "medium", "large", "jumbo"]
    ).astype(str)

    size_di, size_ref, size_ref_rate = calculate_disparate_impact(
        raw_test, "loan_size_bucket", "Loan Size"
    )
    results["loan_size"] = size_di
    size_di.to_csv(
        os.path.join(OUTPUT_DIR, "disparate_impact_by_loan_size.csv"),
        index=False
    )

    # --- VISUALIZATION ---
    print("\n   Generating disparate impact chart...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_data = [
        (state_di,  "borr_state",          "State",        state_ref),
        (age_di,    "business_age_clean",   "Business Age", age_ref),
        (sector_di, "naics_sector",         "Industry",     sector_ref),
        (size_di,   "loan_size_bucket",     "Loan Size",    size_ref),
    ]

    for ax, (df, col, title, ref) in zip(axes, plot_data):
        # If more than 20 groups exist keep top 20 by volume
        # but always include reference group so green line is anchored
        if len(df) > 20:
            top_20  = df.nlargest(20, "total")
            ref_row = df[df[col].astype(str) == str(ref)]
            plot_df = (
                pd.concat([top_20, ref_row])
                .drop_duplicates()
                .sort_values("di_ratio", ascending=True)
            )
        else:
            plot_df = df.sort_values("di_ratio", ascending=True)

        colors = [
            "#B71C1C" if flag else "#1565C0"
            for flag in plot_df["di_flag"]
        ]

        ax.barh(
            plot_df[col].astype(str),
            plot_df["di_ratio"],
            color=colors,
            alpha=0.8
        )
        ax.axvline(
            x=FOUR_FIFTHS_THRESHOLD,
            color="red", linestyle="--",
            linewidth=1.5,
            label="4/5ths threshold (0.80)"
        )
        ax.axvline(
            x=1.0, color="green",
            linestyle="--", linewidth=1,
            label=f"Reference: {ref}"
        )
        ax.set_xlabel("Disparate Impact Ratio")
        ax.set_title(
            f"Disparate Impact — {title}\n"
            f"Red = flagged (DI < 0.80)",
            fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Disparate Impact Analysis — SBA 7(a) Credit Risk Model\n"
        "ECOA 4/5ths Rule | Volume-qualified reference groups",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "disparate_impact_analysis.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   Disparate impact chart saved")

    return results

# =============================================================================
# STEP 3: EQUALIZED ODDS ANALYSIS
# =============================================================================

def calculate_equalized_odds(raw_test, group_col, label,
                              ref_group, min_count=None):
    """
    Equalized Odds measures whether TPR and FPR are consistent
    across groups relative to a volume-qualified reference group.

    Uses ratio-based comparison rather than hard deviation threshold
    to account for natural variance in small groups.

    Equalized Odds Ratio = metric(group) / metric(reference)
    Flag if ratio outside 0.80 - 1.25 range
    """
    if min_count is None:
        min_count = int(len(raw_test) * MIN_GROUP_PCT)

    records = []

    for group_val, group_df in raw_test.groupby(group_col):
        if len(group_df) < min_count:
            continue

        y_true = group_df["y_true"].values
        y_pred = group_df["y_pred"].values

        # Skip groups with no positive cases — cannot calculate TPR
        if y_true.sum() == 0:
            continue

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Handle edge case where confusion matrix is not 2x2
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        records.append({
            "group":     group_val,
            "total":     len(group_df),
            "tp":        int(tp),
            "fp":        int(fp),
            "tn":        int(tn),
            "fn":        int(fn),
            "tpr":       round(tpr, 4),
            "fpr":       round(fpr, 4),
            "precision": round(precision, 4),
        })

    if len(records) == 0:
        print(f"   WARNING: No qualified groups for {label}")
        return pd.DataFrame()

    eq_df = pd.DataFrame(records)

    # Get reference group metrics
    ref_rows = eq_df[eq_df["group"].astype(str) == str(ref_group)]

    if len(ref_rows) == 0:
        # Fall back to group with most samples
        ref_idx  = eq_df["total"].idxmax()
        ref_tpr  = eq_df.loc[ref_idx, "tpr"]
        ref_fpr  = eq_df.loc[ref_idx, "fpr"]
        ref_used = eq_df.loc[ref_idx, "group"]
        print(f"   Reference group {ref_group} not found — "
              f"using {ref_used} as fallback")
    else:
        ref_tpr  = ref_rows.iloc[0]["tpr"]
        ref_fpr  = ref_rows.iloc[0]["fpr"]
        ref_used = ref_group

    # Calculate equalized odds ratios
    # Avoid division by zero with small epsilon
    eq_df["tpr_ratio"] = (
        eq_df["tpr"] / (ref_tpr + 1e-9)
    ).round(4)

    eq_df["fpr_ratio"] = (
        eq_df["fpr"] / (ref_fpr + 1e-9)
    ).round(4)

    # Flag groups outside acceptable ratio range
    eq_df["tpr_flag"] = (
        (eq_df["tpr_ratio"] < EQUALIZED_ODDS_LOW) |
        (eq_df["tpr_ratio"] > EQUALIZED_ODDS_HIGH)
    )

    eq_df["fpr_flag"] = (
        (eq_df["fpr_ratio"] < EQUALIZED_ODDS_LOW) |
        (eq_df["fpr_ratio"] > EQUALIZED_ODDS_HIGH)
    )

    eq_df["any_flag"] = eq_df["tpr_flag"] | eq_df["fpr_flag"]
    eq_df["reference_group"] = ref_used
    eq_df["ref_tpr"] = ref_tpr
    eq_df["ref_fpr"] = ref_fpr

    # Print summary
    flagged = eq_df[eq_df["any_flag"]]
    print(f"\n   {label}:")
    print(f"   Reference group: {ref_used} "
          f"(TPR={ref_tpr:.2%}, FPR={ref_fpr:.2%})")
    print(f"   Groups tested: {len(eq_df)}")
    print(f"   Groups flagged: {len(flagged)}")

    if len(flagged) > 0:
        print(f"   ⚠️  Flagged groups:")
        for _, row in flagged.iterrows():
            flags = []
            if row["tpr_flag"]:
                flags.append(f"TPR ratio={row['tpr_ratio']:.3f}")
            if row["fpr_flag"]:
                flags.append(f"FPR ratio={row['fpr_ratio']:.3f}")
            print(f"      {row['group']}: {' | '.join(flags)}")

    return eq_df


def run_equalized_odds_analysis(raw_test, di_results):
    print("\n[3/9] Running equalized odds analysis...")

    eo_results = {}

    # Use same reference groups as disparate impact analysis
    # for consistency across fairness metrics
    state_ref   = di_results["state"]["reference_group"].iloc[0] \
        if di_results["state"] is not None else None
    age_ref     = di_results["business_age"]["reference_group"].iloc[0] \
        if di_results["business_age"] is not None else None
    sector_ref  = di_results["industry"]["reference_group"].iloc[0] \
        if di_results["industry"] is not None else None
    size_ref    = di_results["loan_size"]["reference_group"].iloc[0] \
        if di_results["loan_size"] is not None else None

    # --- BY STATE ---
    state_eo = calculate_equalized_odds(
        raw_test, "borr_state", "State", state_ref
    )
    eo_results["state"] = state_eo
    state_eo.to_csv(
        os.path.join(OUTPUT_DIR, "equalized_odds_by_state.csv"),
        index=False
    )

    # --- BY BUSINESS AGE ---
    age_eo = calculate_equalized_odds(
        raw_test, "business_age_clean", "Business Age", age_ref
    )
    eo_results["business_age"] = age_eo
    age_eo.to_csv(
        os.path.join(OUTPUT_DIR, "equalized_odds_by_business_age.csv"),
        index=False
    )

    # --- BY INDUSTRY ---
    sector_eo = calculate_equalized_odds(
        raw_test, "naics_sector", "Industry", sector_ref
    )
    eo_results["industry"] = sector_eo
    sector_eo.to_csv(
        os.path.join(OUTPUT_DIR, "equalized_odds_by_industry.csv"),
        index=False
    )

    # --- BY LOAN SIZE ---
    size_eo = calculate_equalized_odds(
        raw_test, "loan_size_bucket", "Loan Size", size_ref
    )
    eo_results["loan_size"] = size_eo
    size_eo.to_csv(
        os.path.join(OUTPUT_DIR, "equalized_odds_by_loan_size.csv"),
        index=False
    )

    # --- VISUALIZATION ---
    print("\n   Generating equalized odds chart...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_configs = [
        (state_eo,  "group", "State"),
        (age_eo,    "group", "Business Age"),
        (sector_eo, "group", "Industry"),
        (size_eo,   "group", "Loan Size"),
    ]

    for ax, (df, col, title) in zip(axes, plot_configs):
        if df.empty:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center")
            ax.set_title(title)
            continue

        plot_df = df.copy().sort_values("tpr", ascending=True)

        # Limit to top 15 by volume for readability
        if len(plot_df) > 15:
            plot_df = plot_df.nlargest(15, "total").sort_values(
                "tpr", ascending=True
            )

        x     = np.arange(len(plot_df))
        width = 0.35

        tpr_colors = [
            "#B71C1C" if f else "#1565C0"
            for f in plot_df["tpr_flag"]
        ]
        fpr_colors = [
            "#E65100" if f else "#2E7D32"
            for f in plot_df["fpr_flag"]
        ]

        ax.barh(x - width/2, plot_df["tpr"],
                width, color=tpr_colors, alpha=0.8,
                label="TPR (red=flagged)")
        ax.barh(x + width/2, plot_df["fpr"],
                width, color=fpr_colors, alpha=0.8,
                label="FPR (orange=flagged)")

        ax.set_yticks(x)
        ax.set_yticklabels(
            plot_df[col].astype(str), fontsize=8
        )
        ax.set_xlabel("Rate")
        ax.set_title(
            f"Equalized Odds — {title}\n"
            f"TPR/FPR by group | Ratio bounds: "
            f"{EQUALIZED_ODDS_LOW}-{EQUALIZED_ODDS_HIGH}",
            fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Equalized Odds Analysis — SBA 7(a) Credit Risk Model\n"
        "TPR and FPR parity across demographic proxy groups",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "equalized_odds_analysis.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   Equalized odds chart saved")

    return eo_results

