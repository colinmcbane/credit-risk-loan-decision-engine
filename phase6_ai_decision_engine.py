"""
phase6_ai_decision_engine.py

Phase 6 — AI Decision Engine
Credit Risk & Loan Decision Engine
Author: Colin McBane

Orchestrates the full adverse action pipeline:
    1. Load Phase 4 scored loan data (SHAP values + predicted probabilities)
    2. Classify each loan as Denied, Approved-Unfavorable Terms, or Approved
    3. Extract top 3 SHAP-derived reason codes per adverse action applicant
    4. Generate ECOA-compliant adverse action letters via Gemini API
    5. Send letters via Gmail SMTP
    6. Save decision output CSV and email delivery log

Input:  data/processed/shap_values_sample.csv
Output: outputs/phase6_decisions.csv
        outputs/phase6_letters/letter_<id>.txt  (one per adverse action loan)
        outputs/phase6_email_log.csv

ECOA Reference: 12 CFR Part 1002 (Regulation B)
SR 11-7 Reference: Model risk governance — decisioning layer consumes
                   Phase 4 champion model outputs without re-scoring.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

from utils.scorer import load_scored_loans
from utils.decision_classifier import classify_all, extract_all_reason_codes
from utils.letter_generator import generate_all_letters
from utils.email_sender import send_all_emails

load_dotenv()

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# ── DRY RUN safeguard ─────────────────────────────────────────────────────────
# When True: full pipeline runs, letters written to disk, NO emails sent,
#            NO Gemini API calls made. Safe for testing and demonstration.
# When False: live Gemini API calls generate real letters, Gmail SMTP sends
#             real emails. Only set False when ready for live execution.
DRY_RUN = True

# ── Pipeline scope ────────────────────────────────────────────────────────────
# Maximum number of adverse action loans to process in one run.
# Set to None to process all 1,051 adverse action loans.
# Use a small integer (e.g. 5) for rapid end-to-end testing.
MAX_APPLICANTS = 5

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR    = "outputs"
LETTERS_DIR   = os.path.join(OUTPUT_DIR, "phase6_letters")
DECISIONS_CSV = os.path.join(OUTPUT_DIR, "phase6_decisions.csv")
EMAIL_LOG_CSV = os.path.join(OUTPUT_DIR, "phase6_email_log.csv")

# ── Recipient configuration ───────────────────────────────────────────────────
# In production each applicant would have their own email on file.
# For portfolio demonstration all letters route to your Gmail address.
RECIPIENT_EMAIL = os.getenv("GMAIL_ADDRESS", "")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step1_load_and_classify() -> pd.DataFrame:
    """
    Load Phase 4 scored loans and classify each into a decision category.

    Reads shap_values_sample.csv which contains SHAP impact scores and
    predicted default probabilities for 5,000 test loans. Adds decision
    and requires_adverse_action columns.

    Returns
    -------
    pd.DataFrame
        Classified loan DataFrame with reason codes attached.
    """
    print("\n── Step 1: Load and Classify ─────────────────────────────────")
    df = load_scored_loans()
    df = classify_all(df)
    df["reason_codes"] = extract_all_reason_codes(df)
    return df


def step2_save_decisions(df: pd.DataFrame) -> None:
    """
    Save the full decision output to outputs/phase6_decisions.csv.

    Exports applicant index, predicted probability, decision label,
    adverse action flag, and reason codes for every loan in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Classified loan DataFrame with reason_codes column.
    """
    print("\n── Step 2: Save Decision Output ──────────────────────────────")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_cols = [
        "predicted_prob",
        "actual_default",
        "decision",
        "requires_adverse_action",
        "reason_codes",
    ]

    decisions_df = df[output_cols].copy()

    # Convert reason_codes list to pipe-delimited string for CSV readability
    decisions_df["reason_codes"] = decisions_df["reason_codes"].apply(
        lambda codes: " | ".join(codes) if codes else ""
    )

    decisions_df.to_csv(DECISIONS_CSV, index=True, index_label="applicant_id")
    print(f"[orchestrator] Decision output saved: {DECISIONS_CSV}")
    print(f"[orchestrator] Total loans: {len(decisions_df):,}")
    print(f"[orchestrator] Adverse action required: "
          f"{decisions_df['requires_adverse_action'].sum():,}")


def step3_generate_letters(df: pd.DataFrame) -> dict:
    """
    Generate ECOA-compliant adverse action letters via Gemini API.

    Passes the full DataFrame to generate_all_letters and lets the
    generator apply the MAX_APPLICANTS cap internally. This preserves
    the correct dataset structure and decision distribution throughout
    the pipeline.

    In DRY_RUN mode no Gemini API calls are made.

    Parameters
    ----------
    df : pd.DataFrame
        Full classified loan DataFrame with requires_adverse_action column.

    Returns
    -------
    dict
        Letter generation results keyed by applicant ID.
    """
    print("\n── Step 3: Generate Adverse Action Letters ───────────────────")

    if MAX_APPLICANTS is not None:
        print(f"[orchestrator] Cap active: generating letters for "
              f"first {MAX_APPLICANTS} flagged applicants.")
    else:
        print(f"[orchestrator] Processing all adverse action applicants.")

    # Pass full DataFrame — the generator handles the cap internally
    # to avoid slicing the dataset before passing it downstream
    letter_results = generate_all_letters(
        df.copy(),
        output_dir=LETTERS_DIR,
        dry_run=DRY_RUN,
        max_letters=MAX_APPLICANTS,
    )

    return letter_results


def step4_send_emails(letter_results: dict) -> None:
    """
    Send adverse action letters via Gmail SMTP.

    Routes all letters to RECIPIENT_EMAIL for portfolio demonstration.
    In DRY_RUN mode no SMTP connection is opened and no emails are sent.

    Parameters
    ----------
    letter_results : dict
        Output from step3_generate_letters().
    """
    print("\n── Step 4: Send Emails ───────────────────────────────────────")

    if not RECIPIENT_EMAIL:
        raise EnvironmentError(
            "[orchestrator] GMAIL_ADDRESS not set in .env. "
            "Cannot route adverse action emails."
        )

    send_all_emails(
        letter_results,
        recipient_address=RECIPIENT_EMAIL,
        dry_run=DRY_RUN,
    )


def print_summary(df: pd.DataFrame, letter_results: dict) -> None:
    """
    Print a final pipeline execution summary to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Classified loan DataFrame.
    letter_results : dict
        Letter generation results from step3_generate_letters().
    """
    generated     = sum(1 for r in letter_results.values() if r["status"] == "generated")
    dry_run_count = sum(1 for r in letter_results.values() if r["status"] == "dry_run")
    errors        = sum(1 for r in letter_results.values() if r["status"] == "error")

    print("\n" + "=" * 60)
    print("  PHASE 6 — AI DECISION ENGINE SUMMARY")
    print("=" * 60)
    print(f"  Run timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DRY RUN mode:        {DRY_RUN}")
    print(f"  Total loans scored:  {len(df):,}")
    print(f"  Approved:            {(df['decision'] == 'Approved').sum():,}")
    print(f"  Denied:              {(df['decision'] == 'Denied').sum():,}")
    print(f"  Unfavorable Terms:   {(df['decision'] == 'Approved - Unfavorable Terms').sum():,}")
    print(f"  Letters generated:   {generated}")
    print(f"  Letters (dry run):   {dry_run_count}")
    print(f"  Letter errors:       {errors}")
    print(f"  Decisions CSV:       {DECISIONS_CSV}")
    print(f"  Letters directory:   {LETTERS_DIR}")
    print(f"  Email log:           {EMAIL_LOG_CSV}")
    print("=" * 60)
    print("  Phase 6 complete. Ready for Phase 7 — Dashboards.")
    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 6 — AI DECISION ENGINE")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print(f"  Max applicants: {MAX_APPLICANTS if MAX_APPLICANTS else 'All'}")
    print("=" * 60)

    # Step 1 — Load Phase 4 scored loans and classify decisions
    df = step1_load_and_classify()

    # Step 2 — Save full decision output CSV
    step2_save_decisions(df)

    # Step 3 — Generate adverse action letters via Gemini API
    letter_results = step3_generate_letters(df)

    # Step 4 — Send letters via Gmail SMTP
    step4_send_emails(letter_results)

    # Final summary
    print_summary(df, letter_results)