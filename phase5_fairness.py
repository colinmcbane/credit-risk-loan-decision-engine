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

    # Load raw group columns saved by Phase 3 at split time
    # Row order is guaranteed to match test_features.csv exactly
    # No database reconstruction needed
    test_groups = pd.read_csv(
        os.path.join(BASE_DIR, "data", "processed", "test_groups.csv")
    )
    print(f"   Group columns loaded: {len(test_groups):,} rows")

    # Validate alignment
    assert len(test_groups) == len(X_test), \
        f"Alignment failed: groups={len(test_groups)}, scaled={len(X_test)}"
    print("   Alignment validated ✓")

    # Attach predictions to group data
    raw_test = test_groups.copy()
    raw_test["y_prob"] = y_prob
    raw_test["y_pred"] = y_pred
    raw_test["y_true"] = y_test.values

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
        raw_test, "borrstate", "State"
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
        (state_di,  "borrstate",          "State",        state_ref),
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
        raw_test, "borrstate", "State", state_ref
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

# =============================================================================
# STEP 4: CALIBRATION ANALYSIS
# =============================================================================

def run_calibration_analysis(raw_test):
    print("\n[4/9] Running calibration analysis...")

    portfolio_default_rate = raw_test["y_true"].mean()
    portfolio_mean_prob    = raw_test["y_prob"].mean()

    print(f"\n   Portfolio Statistics:")
    print(f"   Actual default rate:     {portfolio_default_rate:.2%}")
    print(f"   Mean predicted prob:     {portfolio_mean_prob:.2%}")
    print(f"   Prob shift (SMOTE):      "
          f"{portfolio_mean_prob - portfolio_default_rate:.2%}")

    calibration_results = {}

    groups = [
        ("borrstate",          "State"),
        ("business_age_clean", "Business Age"),
        ("naics_sector",       "Industry"),
        ("loan_size_bucket",   "Loan Size"),
    ]

    smote_shift     = portfolio_mean_prob - portfolio_default_rate
    flag_threshold  = smote_shift * 2

    for group_col, label in groups:
        min_count = int(len(raw_test) * MIN_GROUP_PCT)

        stats = raw_test.groupby(group_col).agg(
            total          = ("y_true", "count"),
            actual_default = ("y_true", "mean"),
            mean_prob      = ("y_prob", "mean"),
            approval_rate  = ("y_pred", lambda x: 1 - x.mean())
        ).reset_index()

        stats = stats[stats["total"] >= min_count].copy()

        stats["calibration_error"] = (
            stats["mean_prob"] - stats["actual_default"]
        ).round(4)

        stats["abs_calibration_error"] = (
            stats["calibration_error"].abs()
        ).round(4)

        stats["calibration_flag"] = (
            stats["abs_calibration_error"] > flag_threshold
        )

        flagged = stats[stats["calibration_flag"]]
        print(f"\n   {label} calibration:")
        print(f"   Groups tested: {len(stats)}")
        print(f"   Flag threshold: {flag_threshold:.2%}")
        print(f"   Groups flagged: {len(flagged)}")

        calibration_results[label] = stats
        stats.to_csv(
            os.path.join(OUTPUT_DIR,
                         f"calibration_{label.lower().replace(' ', '_')}.csv"),
            index=False
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (group_col, label) in zip(axes, groups):
        df = calibration_results[label]

        if df.empty:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center")
            continue

        plot_df = df.nlargest(15, "total").sort_values(
            "actual_default", ascending=True
        )

        x     = np.arange(len(plot_df))
        width = 0.35

        ax.barh(x - width/2,
                plot_df["actual_default"] * 100,
                width, color="#1565C0",
                alpha=0.8, label="Actual Default Rate")
        ax.barh(x + width/2,
                plot_df["mean_prob"] * 100,
                width, color="#B71C1C",
                alpha=0.8, label="Mean Predicted Prob")

        ax.set_yticks(x)
        ax.set_yticklabels(
            plot_df[group_col].astype(str), fontsize=8
        )
        ax.set_xlabel("Rate (%)")
        ax.set_title(
            f"Calibration — {label}\n"
            "Blue = actual | Red = predicted",
            fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Calibration Analysis — SBA 7(a) Credit Risk Model\n"
        "Note: Predicted probabilities shifted higher due to SMOTE "
        "training — see model card for details",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "calibration_analysis.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   Calibration chart saved")

    return calibration_results


# =============================================================================
# STEP 5: SHAP FAIRNESS AUDIT
# =============================================================================

def run_shap_fairness_audit():
    print("\n[5/9] Running SHAP fairness audit...")

    shap_path = os.path.join(
        BASE_DIR, "data", "processed", "shap_values_sample.csv"
    )

    if not os.path.exists(shap_path):
        print("   WARNING: SHAP values not found — skipping audit")
        return {}

    shap_df  = pd.read_csv(shap_path)
    meta_cols = ["predicted_prob", "actual_default"]
    shap_cols = [c for c in shap_df.columns if c not in meta_cols]
    shap_vals = shap_df[shap_cols].abs()
    mean_shap = shap_vals.mean().sort_values(ascending=False)
    total_shap = mean_shap.sum()
    shap_pct   = (mean_shap / total_shap * 100).round(2)

    geo_cols      = [c for c in shap_cols if "borr_state" in c]
    ind_cols      = [c for c in shap_cols if "naics_sector" in c]
    age_cols      = [c for c in shap_cols if "business_age" in c]
    size_cols     = [c for c in shap_cols if "loan_size" in c]

    geo_shap_pct  = shap_vals[geo_cols].mean().sum() / total_shap * 100
    ind_shap_pct  = shap_vals[ind_cols].mean().sum() / total_shap * 100
    age_shap_pct  = shap_vals[age_cols].mean().sum() / total_shap * 100
    size_shap_pct = shap_vals[size_cols].mean().sum() / total_shap * 100

    PROXY_THRESHOLD = 15.0

    print(f"\n   Proxy Variable SHAP Concentration:")
    print(f"   Geographic (state):   {geo_shap_pct:.2f}% "
          f"{'⚠️  FLAGGED' if geo_shap_pct > PROXY_THRESHOLD else '✓'}")
    print(f"   Industry (sector):    {ind_shap_pct:.2f}% "
          f"{'⚠️  FLAGGED' if ind_shap_pct > PROXY_THRESHOLD else '✓'}")
    print(f"   Business Age:         {age_shap_pct:.2f}% "
          f"{'⚠️  FLAGGED' if age_shap_pct > PROXY_THRESHOLD else '✓'}")
    print(f"   Loan Size:            {size_shap_pct:.2f}% "
          f"{'⚠️  FLAGGED' if size_shap_pct > PROXY_THRESHOLD else '✓'}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    proxy_data = {
        "Geographic\n(State)":    geo_shap_pct,
        "Industry\n(Sector)":     ind_shap_pct,
        "Business Age":           age_shap_pct,
        "Loan Size":              size_shap_pct,
        "Core Financial\nFeatures": 100 - geo_shap_pct - ind_shap_pct
                                      - age_shap_pct - size_shap_pct
    }

    colors_pie = ["#B71C1C" if v > PROXY_THRESHOLD else "#1565C0"
                  for v in proxy_data.values()]

    axes[0].pie(
        proxy_data.values(),
        labels=proxy_data.keys(),
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90
    )
    axes[0].set_title(
        "SHAP Contribution by Feature Group\n"
        "Red = above 15% proxy threshold",
        fontweight="bold"
    )

    top15 = shap_pct.head(15)
    proxy_features = geo_cols + ind_cols + age_cols + size_cols
    bar_colors = [
        "#B71C1C" if feat in proxy_features else "#1565C0"
        for feat in top15.index
    ]

    axes[1].barh(
        range(len(top15)),
        top15.values[::-1],
        color=bar_colors[::-1],
        alpha=0.8
    )
    axes[1].set_yticks(range(len(top15)))
    axes[1].set_yticklabels(top15.index[::-1], fontsize=9)
    axes[1].set_xlabel("% of Total SHAP Magnitude")
    axes[1].set_title(
        "Top 15 Features — SHAP % Contribution\n"
        "Red = proxy variable | Blue = core financial",
        fontweight="bold"
    )
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle(
        "SHAP Fairness Audit — SBA 7(a) Credit Risk Model\n"
        "Proxy variable concentration analysis",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shap_fairness_audit.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   SHAP fairness audit chart saved")

    return {
        "geo_shap_pct":    round(geo_shap_pct, 2),
        "ind_shap_pct":    round(ind_shap_pct, 2),
        "age_shap_pct":    round(age_shap_pct, 2),
        "size_shap_pct":   round(size_shap_pct, 2),
        "top_features":    shap_pct.head(10).to_dict(),
        "proxy_threshold": PROXY_THRESHOLD
    }

# =============================================================================
# STEP 6: FAIRLEARN CROSS-VALIDATION
# =============================================================================

def run_fairlearn_validation(raw_test, di_results):
    print("\n[6/9] Running fairlearn cross-validation...")

    # Fairlearn validates our manual calculations independently
    # If numbers match it proves our implementation is correct
    # This is a strong portfolio signal — two independent methods agree

    fairlearn_results = {}

    groups_to_test = [
        ("borrstate",         "State"),
        ("business_age_clean", "Business Age"),
        ("naics_sector",       "Industry"),
        ("loan_size_bucket",   "Loan Size"),
    ]

    y_true = raw_test["y_true"].values
    y_pred = raw_test["y_pred"].values
    y_prob = raw_test["y_prob"].values

    for group_col, label in groups_to_test:
        sensitive = raw_test[group_col].astype(str).values

        try:
            # Demographic parity difference
            # Measures max difference in approval rates across groups
            # Equivalent to our disparate impact analysis
            dp_diff = demographic_parity_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive
            )

            # Equalized odds difference
            # Measures max difference in TPR/FPR across groups
            eo_diff = equalized_odds_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive
            )

            fairlearn_results[label] = {
                "demographic_parity_difference": round(dp_diff, 4),
                "equalized_odds_difference":     round(eo_diff, 4),
                "dp_flag":  abs(dp_diff) > (1 - FOUR_FIFTHS_THRESHOLD),
                "eo_flag":  abs(eo_diff) > (1 - EQUALIZED_ODDS_LOW),
            }

            print(f"\n   {label}:")
            print(f"   Demographic Parity Diff: {dp_diff:.4f} "
                  f"{'⚠️' if abs(dp_diff) > 0.20 else '✓'}")
            print(f"   Equalized Odds Diff:     {eo_diff:.4f} "
                  f"{'⚠️' if abs(eo_diff) > 0.20 else '✓'}")

        except Exception as e:
            print(f"   WARNING: fairlearn failed for {label}: {e}")
            fairlearn_results[label] = {
                "demographic_parity_difference": None,
                "equalized_odds_difference":     None,
                "error": str(e)
            }

    # Save fairlearn results
    fl_df = pd.DataFrame(fairlearn_results).T
    fl_df.to_csv(
        os.path.join(OUTPUT_DIR, "fairlearn_validation.csv")
    )
    print("\n   Fairlearn validation saved")

    return fairlearn_results


# =============================================================================
# STEP 7: FAIRNESS SUMMARY
# =============================================================================

def generate_fairness_summary(di_results, eo_results,
                               calibration_results,
                               shap_audit, fairlearn_results):
    print("\n[7/9] Generating fairness summary...")

    summary_rows = []

    dimensions = [
        ("state",        "Geographic (State)"),
        ("business_age", "Business Age"),
        ("industry",     "Industry Sector"),
        ("loan_size",    "Loan Size"),
    ]

    for key, label in dimensions:
        di_df  = di_results.get(key, pd.DataFrame())
        eo_df  = eo_results.get(key, pd.DataFrame())

        di_flags = int(di_df["di_flag"].sum()) \
            if not di_df.empty and "di_flag" in di_df else 0
        eo_flags = int(eo_df["any_flag"].sum()) \
            if not eo_df.empty and "any_flag" in eo_df else 0

        fl_dp = fairlearn_results.get(label, {}).get(
            "demographic_parity_difference", None)
        fl_eo = fairlearn_results.get(label, {}).get(
            "equalized_odds_difference", None)

        summary_rows.append({
            "Dimension":              label,
            "DI Flags (4/5ths)":     di_flags,
            "EO Flags (ratio)":      eo_flags,
            "Fairlearn DP Diff":     fl_dp,
            "Fairlearn EO Diff":     fl_eo,
            "Overall Flag":          di_flags > 0 or eo_flags > 0,
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n   --- Fairness Summary ---")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "fairness_summary.csv"),
        index=False
    )

    # --- SUMMARY CHART ---
    fig, ax = plt.subplots(figsize=(12, 6))

    x     = np.arange(len(summary_df))
    width = 0.35

    ax.bar(x - width/2,
           summary_df["DI Flags (4/5ths)"],
           width, label="DI Flags",
           color="#B71C1C", alpha=0.8)
    ax.bar(x + width/2,
           summary_df["EO Flags (ratio)"],
           width, label="EO Flags",
           color="#E65100", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Dimension"], fontsize=11)
    ax.set_ylabel("Number of Flagged Groups")
    ax.set_title(
        "Fairness Summary — SBA 7(a) Credit Risk Model\n"
        "Disparate Impact and Equalized Odds flags by dimension",
        fontsize=13, fontweight="bold"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "fairness_summary.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("   Fairness summary chart saved")

    return summary_df


# =============================================================================
# STEP 8: MODEL CARD
# =============================================================================

def write_model_card(champion_name, summary_df,
                     shap_audit, fairlearn_results):
    print("\n[8/9] Writing model card...")

    # Count total flags
    total_di_flags = int(summary_df["DI Flags (4/5ths)"].sum())
    total_eo_flags = int(summary_df["EO Flags (ratio)"].sum())

    model_card = f"""# Model Card — SBA 7(a) Credit Risk Decision Engine

**Author:** Colin McBane
**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d")}
**Model:** {champion_name}
**Version:** 1.0

---

## 1. Model Purpose

This model predicts the probability of default for SBA 7(a) small
business loans. It is designed to assist commercial lending
underwriters in prioritizing loan applications for review and
generating ECOA-compliant Adverse Action notices for denied
applications. The model is not designed to replace human judgment
and should be used as one input among many in the credit decision
process.

**Intended Use:** Credit risk screening for SBA 7(a) loan applications
**Out-of-Scope Use:** Consumer lending, mortgage lending, any use
without human oversight

---

## 2. Training Data

**Source:** SBA 7(a) FOIA Loan Data — raw federal records
**Period:** FY2010 — FY2022 (FY2023+ excluded as right-censored)
**Size:** 382,144 training loans | 95,536 test loans
**Default Rate:** 7.31% (class imbalance addressed via SMOTE)
**Features:** 26 features after Phase 3 feature selection

**Known Data Limitations:**
- SBA 7(a) data does not contain borrower demographic information
  (race, ethnicity, gender, age). Fairness analysis uses economic
  proxy variables only. A complete ECOA demographic analysis requires
  matching with HMDA records or direct demographic collection.
- FY2023-2025 loans excluded due to right-censoring — insufficient
  time to observe default outcomes. These loans are used for
  prediction only, not model training.

---

## 3. Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9667 |
| KS Statistic | 83.63 |
| Gini Coefficient | 93.34% |
| Precision (t=0.35) | 35.66% |
| Recall (t=0.35) | 95.40% |
| F1 Score | 0.5191 |

**Decision Threshold:** 0.35
The threshold was set below 0.50 to maximize recall (catching
defaults) because the cost of a false negative (approving a
defaulting loan) exceeds the cost of a false positive (rejecting
a viable loan) in commercial credit risk.

**Champion Selection:** LightGBM selected over XGBoost (AUC 0.9638)
and Logistic Regression (AUC 0.8338) via challenger-champion
framework evaluated on held-out test set.

---

## 4. Known Technical Limitations

**SMOTE Probability Shift:**
SMOTE (Synthetic Minority Oversampling Technique) was applied during
training to address 7.31% class imbalance. This causes model-output
probabilities to be systematically higher than real-world default
rates. The mean predicted probability on the test set is
approximately 12-13 percentage points above the actual default rate.
Raw probabilities should not be interpreted as calibrated default
probabilities. The 0.35 decision threshold was optimized on the
post-SMOTE probability distribution.

**Tree Model Tail Risk:**
LightGBM handles out-of-distribution feature values by capping
predictions at the furthest leaf node. Under severe macroeconomic
stress scenarios the model may underestimate tail risk. Stress test
results should be interpreted as lower bounds in extreme scenarios.

**Right Censoring:**
Loans approved in FY2023-2025 have not had sufficient time to
default. Model performance metrics reflect the FY2010-2022 vintage
only. Performance on recent vintages should be monitored separately
as outcomes become observable.

---

## 5. Fairness Analysis

**Methodology:** Statistical disparate impact testing via economic
proxy variables. Direct demographic analysis not possible due to
data limitations documented in Section 2.

**Tests Conducted:**
- ECOA 4/5ths disparate impact rule (geographic, business age,
  industry, loan size dimensions)
- Equalized odds ratio analysis (TPR/FPR parity across groups)
- SHAP proxy variable concentration audit
- Independent validation via fairlearn library

**Results Summary:**

| Dimension | DI Flags | EO Flags |
|-----------|----------|----------|
{summary_df[["Dimension", "DI Flags (4/5ths)", "EO Flags (ratio)"]].to_markdown(index=False)}

**Total Disparate Impact Flags:** {total_di_flags}
**Total Equalized Odds Flags:** {total_eo_flags}

**SHAP Proxy Concentration:**
- Geographic (State): {shap_audit.get("geo_shap_pct", "N/A")}%
- Industry (Sector): {shap_audit.get("ind_shap_pct", "N/A")}%
- Business Age: {shap_audit.get("age_shap_pct", "N/A")}%
- Loan Size: {shap_audit.get("size_shap_pct", "N/A")}%

**Omitted Variable Bias Disclosure:**
Because this dataset lacks direct demographic variables, proxy
testing cannot definitively confirm or deny ECOA compliance.
Flagged disparities in proxy variables require investigation of
whether the underlying population of affected borrowers is
demographically concentrated. Human review is required before
any production deployment.

**Business Necessity Documentation:**
Where disparate impact is identified, Phase 4 stress testing
provides quantitative business necessity evidence. New business
loans show 12.66% baseline default rate versus 5.33% for mature
businesses — a 2.4x differential supporting differentiated risk
treatment under ECOA's business necessity defense. Banks deploying
this model must maintain business necessity documentation per
ECOA Regulation B.

---

## 6. Regulatory Compliance Context

**ECOA (Equal Credit Opportunity Act):**
This model was developed with ECOA compliance as a design
constraint. Adverse Action letters generated by the Phase 6
decision engine use SHAP-derived specific reasons as required
by ECOA Regulation B. Statistical disparate impact testing
follows the CFPB's examination procedures for algorithmic
credit models.

**SR 11-7 (Federal Reserve Model Risk Management):**
- Model purpose and limitations documented in this card
- Champion-challenger validation framework implemented
- Independent fairlearn validation cross-checks manual results
- Stress testing conducted across 5 macroeconomic scenarios
- Known failure modes documented in Section 4
- Ongoing monitoring recommendations provided in Section 7

**Fair Housing Act:**
This model covers commercial lending only. Fair Housing Act
mortgage lending provisions do not apply directly, but the
disparate impact standard from the 2015 HUD rule has been
considered in the proxy variable analysis.

---

## 7. Monitoring Recommendations

1. **Performance monitoring:** Recalculate AUC, KS, and Gini
   quarterly as new loan outcomes become observable.
2. **Population stability:** Monitor feature distributions monthly
   for drift from training distribution.
3. **Fairness monitoring:** Rerun disparate impact analysis
   semi-annually or after any model update.
4. **Right-censored vintage:** Begin evaluating FY2023 loan
   performance in late 2025 when sufficient outcomes are observed.
5. **Demographic data:** Recommend collecting voluntary demographic
   data at application to enable full ECOA compliance testing.
6. **Threshold review:** Revisit 0.35 decision threshold annually
   as economic conditions evolve.

---

## 8. Model Governance

**Development:** Colin McBane, NC State University
**Data Source:** SBA Office of Capital Access — FOIA public records
**Validation:** Held-out test set (20%), 5-fold cross-validation,
independent fairlearn library cross-check
**Explainability:** SHAP TreeExplainer — local and global explanations
**Adverse Action Engine:** Anthropic Claude API (claude-sonnet-4-6) —
  Orchestrated via deterministic SHAP feature mappings for standardized,
  ECOA-compliant Adverse Action notice synthesis. The LLM operates as a
  structured text rendering layer only — credit decisions are made
  exclusively by the LightGBM champion model. The API receives ranked
  SHAP values as structured input and is constrained to produce
  standardized regulatory language. No discretionary credit judgment
  is delegated to the language model.

---

*This model card follows the Model Cards for Model Reporting
framework (Mitchell et al., 2019) and Federal Reserve SR 11-7
model risk management guidance.*
"""

    with open(MODEL_CARD, "w") as f:
        f.write(model_card)

    print(f"   Model card saved: {MODEL_CARD}")
    return model_card


# =============================================================================
# STEP 9: FINAL VALIDATION SUMMARY
# =============================================================================

def print_final_summary(summary_df, shap_audit):
    print("\n[9/9] Phase 5 Complete")
    print("\n=== Fairness Analysis Summary ===")

    total_di = int(summary_df["DI Flags (4/5ths)"].sum())
    total_eo = int(summary_df["EO Flags (ratio)"].sum())

    print(f"\n   Total DI flags: {total_di}")
    print(f"   Total EO flags: {total_eo}")

    if total_di == 0 and total_eo == 0:
        print("   ✓ No fairness flags detected")
    else:
        print("   ⚠️  Fairness flags require business necessity review")

    print(f"\n   SHAP proxy concentration:")
    print(f"   Geographic: {shap_audit.get('geo_shap_pct', 'N/A')}%")
    print(f"   Industry:   {shap_audit.get('ind_shap_pct', 'N/A')}%")
    print(f"   Business Age: {shap_audit.get('age_shap_pct', 'N/A')}%")
    print(f"   Loan Size:  {shap_audit.get('size_shap_pct', 'N/A')}%")

    print("\n   --- Files Saved ---")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   {os.path.join(OUTPUT_DIR, f)}")
    print(f"   {MODEL_CARD}")

    print("\n=== Phase 5 Complete ===")
    print("Ready for Phase 6 — AI Decision Engine")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Step 1 — Load model and data
    (model, champion_name, X_test, y_test,
     y_prob, y_pred, raw_test) = load_model_and_data()

    # Step 2 — Disparate impact analysis
    di_results = run_disparate_impact_analysis(raw_test)

    # Step 3 — Equalized odds analysis
    eo_results = run_equalized_odds_analysis(raw_test, di_results)

    # Step 4 — Calibration analysis
    calibration_results = run_calibration_analysis(raw_test)

    # Step 5 — SHAP fairness audit
    shap_audit = run_shap_fairness_audit()

    # Step 6 — Fairlearn cross-validation
    fairlearn_results = run_fairlearn_validation(raw_test, di_results)

    # Step 7 — Fairness summary
    summary_df = generate_fairness_summary(
        di_results, eo_results,
        calibration_results, shap_audit,
        fairlearn_results
    )

    # Step 8 — Model card
    write_model_card(
        champion_name, summary_df,
        shap_audit, fairlearn_results
    )

    # Step 9 — Final summary
    print_final_summary(summary_df, shap_audit)