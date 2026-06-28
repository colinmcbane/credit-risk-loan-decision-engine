"""
validate_phase6.py

Validates Phase 6 output files after a pipeline run.
Checks that all expected files exist, have correct structure,
and contain sensible values. Run after phase6_ai_decision_engine.py.

Usage:
    python3 validate_phase6.py
"""

import os
import sys
import pandas as pd


# ── Expected output paths ─────────────────────────────────────────────────────
DECISIONS_CSV = os.path.join("outputs", "phase6_decisions.csv")
EMAIL_LOG_CSV = os.path.join("outputs", "phase6_email_log.csv")
LETTERS_DIR   = os.path.join("outputs", "phase6_letters")

# ── Validation thresholds ─────────────────────────────────────────────────────
EXPECTED_TOTAL_LOANS  = 5000
EXPECTED_MIN_DENIED   = 100
EXPECTED_MIN_ADVERSE  = 100

PASS     = "✓"
FAIL     = "✗"
failures = []


def check(condition: bool, label: str, detail: str = "") -> None:
    """
    Evaluate a single validation check and print result.

    Parameters
    ----------
    condition : bool
        True if the check passed.
    label : str
        Short description of what is being checked.
    detail : str
        Optional extra context printed alongside the result.
    """
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if not condition:
        failures.append(label)


def validate_decisions_csv() -> None:
    """Validate outputs/phase6_decisions.csv structure and contents."""
    print("\n── Validating phase6_decisions.csv ───────────────────────────")

    check(os.path.exists(DECISIONS_CSV), "File exists")
    if not os.path.exists(DECISIONS_CSV):
        return

    df = pd.read_csv(DECISIONS_CSV)

    check(len(df) == EXPECTED_TOTAL_LOANS,
          "Row count correct",
          f"{len(df):,} rows")

    required_cols = [
        "applicant_id", "predicted_prob", "actual_default",
        "decision", "requires_adverse_action", "reason_codes"
    ]
    for col in required_cols:
        check(col in df.columns, f"Column '{col}' present")

    check(
        df["predicted_prob"].between(0.0, 1.0).all(),
        "All predicted_prob values between 0 and 1"
    )

    valid_decisions = {"Approved", "Denied", "Approved - Unfavorable Terms"}
    check(
        df["decision"].isin(valid_decisions).all(),
        "All decision labels valid"
    )

    denied_count = (df["decision"] == "Denied").sum()
    check(
        denied_count >= EXPECTED_MIN_DENIED,
        "Sufficient denied loans present",
        f"{denied_count:,} denied"
    )

    adverse_count = df["requires_adverse_action"].sum()
    check(
        adverse_count >= EXPECTED_MIN_ADVERSE,
        "Sufficient adverse action flags present",
        f"{adverse_count:,} flagged"
    )

    check(
        df[df["requires_adverse_action"]]["reason_codes"].str.len().gt(0).all(),
        "All adverse action loans have reason codes"
    )

    print(f"\n  Decision distribution:")
    for label, count in df["decision"].value_counts().items():
        print(f"    {label}: {count:,} ({count / len(df) * 100:.1f}%)")


def validate_email_log() -> None:
    """Validate outputs/phase6_email_log.csv structure and contents."""
    print("\n── Validating phase6_email_log.csv ───────────────────────────")

    check(os.path.exists(EMAIL_LOG_CSV), "File exists")
    if not os.path.exists(EMAIL_LOG_CSV):
        return

    df = pd.read_csv(EMAIL_LOG_CSV)

    required_cols = [
        "timestamp", "applicant_id", "recipient",
        "decision", "status", "error"
    ]
    for col in required_cols:
        check(col in df.columns, f"Column '{col}' present")

    valid_statuses = {"sent", "dry_run", "error"}
    check(
        df["status"].isin(valid_statuses).all(),
        "All status values valid"
    )

    error_count = (df["status"] == "error").sum()
    check(error_count == 0, "Zero email errors", f"{error_count} errors")

    print(f"\n  Email log summary:")
    for status, count in df["status"].value_counts().items():
        print(f"    {status}: {count:,}")


def validate_letters_dir() -> None:
    """Validate that letter .txt files were written to disk."""
    print("\n── Validating outputs/phase6_letters/ ────────────────────────")

    check(os.path.exists(LETTERS_DIR), "Directory exists")
    if not os.path.exists(LETTERS_DIR):
        return

    letter_files = [
        f for f in os.listdir(LETTERS_DIR)
        if f.startswith("letter_") and f.endswith(".txt")
    ]

    check(len(letter_files) > 0, "Letter files present",
          f"{len(letter_files)} files found")

    if letter_files:
        sample_path = os.path.join(LETTERS_DIR, sorted(letter_files)[0])
        with open(sample_path, "r") as f:
            content = f.read()

        check(len(content) > 50, "Sample letter has content",
              f"{len(content)} chars")
        check("Applicant ID" in content or "Dear" in content,
              "Sample letter contains expected text")

        print(f"\n  Sample letter: {sorted(letter_files)[0]}")
        print(f"  First 200 chars:\n  {content[:200].strip()}")


def print_final_result() -> None:
    """Print overall validation pass/fail summary."""
    print("\n" + "=" * 60)
    if failures:
        print(f"  VALIDATION FAILED — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"    {FAIL} {f}")
        sys.exit(1)
    else:
        print("  VALIDATION PASSED — all checks clean.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 6 — OUTPUT VALIDATION")
    print("=" * 60)

    validate_decisions_csv()
    validate_email_log()
    validate_letters_dir()
    print_final_result()