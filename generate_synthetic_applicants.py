"""
generate_synthetic_applicants.py

Generates synthetic SBA 7(a) loan applicant records using the Gemini API
for out-of-sample validation of the Phase 4 LightGBM champion model.

Produces 50 applicants with known risk profiles (low, moderate, high)
in the exact 26-column schema used by the Phase 6 decision engine.

Output: data/processed/gemini_test_applicants.csv

Usage:
    python3 generate_synthetic_applicants.py
"""

import os
import io
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.0-flash-lite"
OUTPUT_PATH    = os.path.join("data", "processed", "gemini_test_applicants.csv")

# ── Required columns in exact order ──────────────────────────────────────────
REQUIRED_COLUMNS = [
    "term_months", "interest_rate", "loan_amount", "sba_guarantee_pct",
    "business_age_mature", "loan_size_bucket_micro", "naics_sector_62",
    "loan_size_bucket_large", "borr_state_FL", "business_age_startup",
    "borr_state_TX", "jobs_supported", "business_age_established",
    "loan_size_bucket_small", "borr_state_CA", "loan_size_bucket_medium",
    "business_age_new", "naics_sector_48", "borr_state_WI", "borr_state_NJ",
    "borr_state_NY", "naics_sector_71", "naics_sector_52", "borr_state_WA",
    "naics_sector_45", "borr_state_MN",
]

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = """Generate exactly 50 synthetic SBA 7(a) loan applicant records for
testing a credit risk model. Output ONLY a single markdown csv code block.
The first row must be exactly these 26 column headers in this exact order:

term_months,interest_rate,loan_amount,sba_guarantee_pct,business_age_mature,loan_size_bucket_micro,naics_sector_62,loan_size_bucket_large,borr_state_FL,business_age_startup,borr_state_TX,jobs_supported,business_age_established,loan_size_bucket_small,borr_state_CA,loan_size_bucket_medium,business_age_new,naics_sector_48,borr_state_WI,borr_state_NJ,borr_state_NY,naics_sector_71,naics_sector_52,borr_state_WA,naics_sector_45,borr_state_MN

Column rules:
- term_months: integer, range 60-300
- interest_rate: decimal (e.g. 0.065 for 6.5%), range 0.05-0.12
- loan_amount: integer dollars, range 5000-5000000
- sba_guarantee_pct: decimal fraction, range 0.50-0.85
- business_age_mature: 1 if business >10 years old, else 0
- business_age_startup: 1 if startup (loan funds will open business), else 0
- business_age_new: 1 if business <=2 years old, else 0
- business_age_established: 1 if business 2-10 years old, else 0
  CRITICAL: exactly one of the four business_age columns must equal 1 per row
- loan_size_bucket_micro: 1 if loan_amount < 50000, else 0
- loan_size_bucket_small: 1 if 50000 <= loan_amount < 350000, else 0
- loan_size_bucket_medium: 1 if 350000 <= loan_amount < 2000000, else 0
- loan_size_bucket_large: 1 if loan_amount >= 2000000, else 0
  CRITICAL: exactly one bucket must equal 1 and must match loan_amount
- jobs_supported: integer, range 1-500
- naics_sector_45: 1 if retail trade, else 0
- naics_sector_48: 1 if transportation, else 0
- naics_sector_52: 1 if finance/insurance, else 0
- naics_sector_62: 1 if healthcare, else 0
- naics_sector_71: 1 if arts/entertainment, else 0
  Note: at most one naics column can be 1. All can be 0.
- borr_state_CA/FL/MN/NJ/NY/TX/WA/WI: 1 if borrower in that state, else 0
  Note: at most one state column can be 1. All can be 0.

Generate exactly 50 rows in this order:
- Rows 1-15: LOW RISK (mature businesses, medium loans $350k-$2M,
  strong guarantee >0.75, long terms >180 months, low rates <0.07)
- Rows 16-35: MODERATE RISK (mix of business ages, varied loan sizes,
  average rates 0.07-0.09, mixed guarantee levels)
- Rows 36-50: HIGH RISK (startups or new businesses, micro loans <$50k
  or large loans >$2M, short terms <90 months, low guarantee <0.60,
  high rates >0.09)

Output ONLY the markdown csv code block. No explanation, no greeting,
no text outside the code block."""


def call_gemini(client: genai.Client) -> str:
    """
    Call Gemini API and return raw response text.

    Parameters
    ----------
    client : genai.Client
        Initialized Gemini API client.

    Returns
    -------
    str
        Raw text response from Gemini.
    """
    print("[generator] Calling Gemini API...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=PROMPT,
    )
    if not response or not response.text:
        raise RuntimeError("[generator] Gemini returned empty response.")
    return response.text


def extract_csv_from_response(raw_text: str) -> str:
    """
    Extract CSV content from a markdown code block in Gemini's response.

    Gemini wraps CSV output in ```csv ... ``` or ``` ... ``` fences.
    This function strips those fences and returns clean CSV text.

    Parameters
    ----------
    raw_text : str
        Raw response text from Gemini.

    Returns
    -------
    str
        Clean CSV text ready for pandas to parse.
    """
    lines = raw_text.strip().splitlines()
    csv_lines = []
    inside_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            inside_block = not inside_block
            continue
        if inside_block:
            csv_lines.append(line)

    # Fallback — if no code block found, try parsing the whole response
    if not csv_lines:
        print("[generator] No markdown code block found — "
              "attempting to parse full response as CSV.")
        csv_lines = lines

    return "\n".join(csv_lines)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate the generated DataFrame against schema rules.

    Checks column presence, row count, business_age mutual exclusivity,
    loan_size_bucket consistency with loan_amount, and value ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed applicant DataFrame from Gemini output.

    Returns
    -------
    bool
        True if all checks pass, False if any check fails.
    """
    passed = True

    # Column check
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[validator] FAIL — Missing columns: {missing}")
        passed = False

    # Row count
    if len(df) != 50:
        print(f"[validator] WARN — Expected 50 rows, got {len(df)}")

    # Business age mutual exclusivity
    age_cols = [
        "business_age_mature", "business_age_startup",
        "business_age_new", "business_age_established"
    ]
    if all(c in df.columns for c in age_cols):
        age_sum = df[age_cols].sum(axis=1)
        bad = (age_sum != 1).sum()
        if bad > 0:
            print(f"[validator] WARN — {bad} rows have incorrect "
                  f"business_age encoding (should sum to 1)")
        else:
            print("[validator] ✓ business_age encoding correct")

    # Loan size bucket consistency
    bucket_cols = [
        "loan_size_bucket_micro", "loan_size_bucket_small",
        "loan_size_bucket_medium", "loan_size_bucket_large"
    ]
    if all(c in df.columns for c in bucket_cols):
        bucket_sum = df[bucket_cols].sum(axis=1)
        bad = (bucket_sum != 1).sum()
        if bad > 0:
            print(f"[validator] WARN — {bad} rows have incorrect "
                  f"loan_size_bucket encoding (should sum to 1)")
        else:
            print("[validator] ✓ loan_size_bucket encoding correct")

    # Probability range checks
    if "interest_rate" in df.columns:
        out_of_range = (~df["interest_rate"].between(0.04, 0.15)).sum()
        if out_of_range > 0:
            print(f"[validator] WARN — {out_of_range} rows have "
                  f"interest_rate outside expected range")
        else:
            print("[validator] ✓ interest_rate values in range")

    if "sba_guarantee_pct" in df.columns:
        out_of_range = (~df["sba_guarantee_pct"].between(0.45, 0.90)).sum()
        if out_of_range > 0:
            print(f"[validator] WARN — {out_of_range} rows have "
                  f"sba_guarantee_pct outside expected range")
        else:
            print("[validator] ✓ sba_guarantee_pct values in range")

    return passed


def save_csv(df: pd.DataFrame) -> None:
    """
    Save the validated applicant DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Validated applicant DataFrame.
    """
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[generator] Saved {len(df)} applicants to {OUTPUT_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SYNTHETIC APPLICANT GENERATOR — Gemini")
    print("=" * 60)

    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in .env file."
        )

    # Initialize Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Call Gemini and extract CSV
    raw_text   = call_gemini(client)
    csv_text   = extract_csv_from_response(raw_text)

    # Parse into DataFrame
    print("[generator] Parsing CSV response...")
    df = pd.read_csv(io.StringIO(csv_text))
    print(f"[generator] Parsed {len(df)} rows, {len(df.columns)} columns")

    # Validate
    print("\n[generator] Validating schema...")
    validate_dataframe(df)

    # Keep only required columns in correct order
    df = df[[c for c in REQUIRED_COLUMNS if c in df.columns]]

    # Save
    save_csv(df)

    print("\n" + "=" * 60)
    print("  Generation complete.")
    print(f"  File: {OUTPUT_PATH}")
    print(f"  Rows: {len(df)}")
    print("=" * 60)