# =============================================================================
# PHASE 1 — SQL DATA CLEANING
# Credit Risk & Loan Decision Engine
# Author: Colin McBane
# Data: SBA 7(a) FOIA Loan Data (FY2010-2019, FY2020-Present)
# =============================================================================

import pandas as pd
import sqlite3
import os
from tqdm import tqdm

# =============================================================================
# FILE PATHS
# =============================================================================

# Base directory — wherever this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw data files
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FILE_2010_2019 = os.path.join(RAW_DIR, "sba_7a_2010_2019.csv")
FILE_2020_PRESENT = os.path.join(RAW_DIR, "sba_7a_2020_present.csv")

# Output database
DB_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(DB_DIR, "sba_clean.db")

# Make sure processed folder exists
os.makedirs(DB_DIR, exist_ok=True)

print("=== Phase 1: SBA Data Cleaning ===")
print(f"Loading from: {RAW_DIR}")
print(f"Database will be saved to: {DB_PATH}")

# =============================================================================
# STEP 1: LOAD RAW CSV FILES
# =============================================================================

def load_raw_data():
    print("\n[1/6] Loading raw CSV files...")

    # Define columns we actually want — drop address/PII/leakage columns now
    KEEP_COLS = [
        "program", "borrcity", "borrstate", "borrzip",
        "bankname", "bankstate", "grossapproval",
        "sbaguaranteedapproval", "approvaldate", "approvalfy",
        "firstdisbursementdate", "initialinterestrate", "terminmonths",
        "naicscode", "naicsdescription", "projectstate",
        "businesstype", "businessage", "loanstatus",
        "paidinfulldate", "chargeoffdate", "grosschargeoffamount",
        "revolverstatus", "jobssupported", "collateralind",
        "processingmethod", "sbadistrictoffice",
    ]

    # Load first file
    print("   Reading FY2010-2019...")
    df1 = pd.read_csv(
        FILE_2010_2019,
        low_memory=False,
        encoding="latin-1"
    )
    print(f"   FY2010-2019: {len(df1):,} rows loaded")

    # Load second file
    print("   Reading FY2020-Present...")
    df2 = pd.read_csv(
        FILE_2020_PRESENT,
        low_memory=False,
        encoding="latin-1"
    )
    print(f"   FY2020-Present: {len(df2):,} rows loaded")

    # Stack them into one dataframe
    df = pd.concat([df1, df2], ignore_index=True)

    # Normalize column names to lowercase for cleaner downstream processing.
    df.columns = df.columns.str.strip().str.lower()

    print(f"\n   Combined total: {len(df):,} rows")
    print(f"   Columns: {df.shape[1]}")
   
    return df

# =============================================================================
# STEP 2: PROFILE THE RAW DATA
# =============================================================================

def profile_data(df):
    print("\n[2/6] Profiling raw data...")

    # Normalize all column names — strip spaces, fix casing
    df.columns = df.columns.str.strip().str.lower()
    print("Cleaned column names:")
    print(list(df.columns))

    # Shape
    print(f"\n   Rows: {len(df):,}")
    print(f"   Columns: {df.shape[1]}")

    # Null counts and percentages
    print("\n   --- Null Report ---")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_report = pd.DataFrame({
        "null_count": null_counts,
        "null_pct": null_pct
    }).sort_values("null_pct", ascending=False)
    print(null_report[null_report["null_count"] > 0].to_string())

    # LoanStatus distribution — this is your target variable
    print("\n   --- LoanStatus Distribution ---")
    print(df["loanstatus"].value_counts())

    # Data types
    print("\n   --- Data Types ---")
    print(df.dtypes)

    return null_report

# =============================================================================
# STEP 3: CLEAN THE DATA
# =============================================================================

def clean_data(df):
    print("\n[3/6] Cleaning data...")
    starting_rows = len(df)

    # Lowercase all column names so downstream SQL and pandas logic stay consistent.
    df.columns = df.columns.str.strip().str.lower()

    # --- 3A: FILTER LOAN STATUS ---
    # Keep only loans with a final outcome — drop anything undisbursed/cancelled
    print("   [3A] Filtering LoanStatus...")
    df = df[df["loanstatus"].isin(["PIF", "CHGOFF"])]
    print(f"   Rows after status filter: {len(df):,}")

    # --- 3B: CREATE TARGET VARIABLE ---
    # 1 = defaulted (Charged Off), 0 = paid in full
    print("   [3B] Creating target variable...")
    df["is_default"] = (df["loanstatus"] == "CHGOFF").astype(int)
    print(f"   Default rate: {df['is_default'].mean():.2%}")

    # --- 3C: FIX DATE COLUMNS ---
    print("   [3C] Parsing date columns...")
    date_cols = ["approvaldate", "firstdisbursementdate",
                 "paidinfulldate", "chargeoffdate"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- 3D: FIX NUMERIC COLUMNS ---
    print("   [3D] Fixing numeric columns...")
    numeric_cols = ["grossapproval", "sbaguaranteedapproval",
                    "initialinterestrate", "terminmonths",
                    "grosschargeoffamount", "jobssupported"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # --- 3E: FIX CATEGORICAL COLUMNS ---
    print("   [3E] Standardizing categorical columns...")
    cat_cols = ["borrstate", "bankstate", "projectstate",
                "businesstype", "businessage", "revolverstatus",
                "collateralind", "processingmethod"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace("NAN", None)
    
    # --- 3F: DROP NULL CRITICAL COLUMNS ---
    # These columns must exist for a loan record to be useful
    print("   [3F] Dropping rows with nulls in critical columns...")
    critical_cols = ["grossapproval", "loanstatus", "approvaldate",
                     "borrstate", "naicscode", "terminmonths",
                     "initialinterestrate"]
    df = df.dropna(subset=critical_cols)
    print(f"   Rows after null drop: {len(df):,}")

    # --- 3G: REMOVE OBVIOUS BAD DATA ---
    print("   [3G] Removing bad data...")
    # Loan amount must be positive
    df = df[df["grossapproval"] > 0]
    # Interest rate must be realistic
    df = df[(df["initialinterestrate"] > 0) &
            (df["initialinterestrate"] < 30)]
    # Term must be positive
    df = df[df["terminmonths"] > 0]
    # Approval year must be in range
    df = df[(df["approvalfy"] >= 2010) &
            (df["approvalfy"] <= 2026)]
    print(f"   Rows after bad data removal: {len(df):,}")

    # --- 3H: ENGINEER BASIC FEATURES ---
    print("   [3H] Engineering basic features...")
    # SBA guarantee percentage
    df["sba_guarantee_pct"] = (
        df["sbaguaranteedapproval"] / df["grossapproval"]
    ).round(4)

    # Loan size buckets
    df["loan_size_bucket"] = pd.cut(
        df["grossapproval"],
        bins=[0, 150000, 500000, 1000000, 2000000, float("inf")],
        labels=["micro", "small", "medium", "large", "jumbo"]
    )

    # Approval year and month for time series use in Phase 2
    df["approval_year"] = df["approvaldate"].dt.year
    df["approval_month"] = df["approvaldate"].dt.month

    # NAICS sector — first 2 digits of NAICS code = industry sector
    df["naics_sector"] = df["naicscode"].astype(str).str[:2]

    print(f"\n   Cleaning complete.")
    print(f"   Started with: {starting_rows:,} rows")
    print(f"   Ended with:   {len(df):,} rows")
    print(f"   Removed:      {starting_rows - len(df):,} rows")
    print(f"   Final default rate: {df['is_default'].mean():.2%}")

    return df

# =============================================================================
# STEP 4: LOAD CLEAN DATA INTO SQLITE
# =============================================================================

def load_to_sqlite(df):
    print("\n[4/6] Loading clean data into SQLite...")

     # Connect to database — creates the file if it doesn't exist
    conn = sqlite3.connect(DB_PATH)

    # Write main loans table
    print("   Writing loans table...")
    df.to_sql(
        name="loans",
        con=conn,
        if_exists="replace",
        index=False,
        chunksize=10000
    )
    print(f"   loans table: {len(df):,} rows written")

    # Verify it wrote correctly
    row_count = pd.read_sql("SELECT COUNT(*) as count FROM loans", conn)
    print(f"   Verified row count: {row_count['count'][0]:,}")

    # Show table schema
    print("\n   --- Table Schema ---")
    schema = pd.read_sql(
        "SELECT name, type FROM pragma_table_info('loans')", conn
    )
    print(schema.to_string(index=False))

    conn.close()
    print("\n   SQLite database saved.")
    print(f"   Location: {DB_PATH}")

    return conn

# =============================================================================
# STEP 5: RUN SQL CLEANING QUERIES
# =============================================================================

def run_sql_cleaning():
    print("\n[5/6] Running SQL cleaning queries...")

    conn = sqlite3.connect(DB_PATH)

    # --- 5A: CREATE CLEAN VIEW ---
    print("   [5A] Creating clean loans view...")
    conn.execute("DROP VIEW IF EXISTS loans_clean")
    conn.execute("""
        CREATE VIEW loans_clean AS
        SELECT
            -- Identifiers
            rowid                           AS loan_id,
            program,
            processingmethod,

            -- Borrower geography
            borrcity                        AS borr_city,
            borrstate                       AS borr_state,
            borrzip                         AS borr_zip,

            -- Lender info
            bankname                        AS bank_name,
            bankstate                       AS bank_state,

            -- Loan financials
            grossapproval                   AS loan_amount,
            sbaguaranteedapproval           AS sba_guaranteed_amount,
            sba_guarantee_pct,
            initialinterestrate             AS interest_rate,
            terminmonths                    AS term_months,
            revolverstatus                  AS is_revolver,

            -- Business info
            naicscode                       AS naics_code,
            naicsdescription                AS naics_description,
            naics_sector,
            businesstype                    AS business_type,
            businessage                     AS business_age,
            collateralind                   AS has_collateral,
            jobssupported                   AS jobs_supported,

            -- Approval timing
            approvaldate                    AS approval_date,
            approvalfy                      AS approval_fy,
            approval_year,
            approval_month,
            firstdisbursementdate           AS disbursement_date,

            -- Outcome
            loanstatus                      AS loan_status,
            is_default                      AS is_default,
            grosschargeoffamount            AS chargeoff_amount,
            loan_size_bucket,

            -- District
            sbadistrictoffice               AS sba_district

        FROM loans
    """)
    conn.commit()
    print("   loans_clean view created")

    # --- 5B: DEFAULT RATE BY INDUSTRY ---
    print("   [5B] Creating default rate by industry summary...")
    conn.execute("DROP TABLE IF EXISTS summary_by_industry")
    conn.execute("""
        CREATE TABLE summary_by_industry AS
        SELECT
            naics_sector,
            naics_description               AS industry_description,
            COUNT(*)                        AS total_loans,
            SUM(is_default)                 AS total_defaults,
            ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
            ROUND(AVG(loan_amount), 0)      AS avg_loan_amount,
            ROUND(AVG(interest_rate), 2)    AS avg_interest_rate,
            ROUND(AVG(term_months), 0)      AS avg_term_months
        FROM loans_clean
        GROUP BY naics_sector, naics_description
        ORDER BY default_rate_pct DESC
    """)
    conn.commit()
    print("   summary_by_industry table created")

    # --- 5C: DEFAULT RATE BY STATE ---
    print("   [5C] Creating default rate by state summary...")
    conn.execute("DROP TABLE IF EXISTS summary_by_state")
    conn.execute("""
        CREATE TABLE summary_by_state AS
        SELECT
            borr_state,
            COUNT(*)                        AS total_loans,
            SUM(is_default)                 AS total_defaults,
            ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
            ROUND(AVG(loan_amount), 0)      AS avg_loan_amount
        FROM loans_clean
        GROUP BY borr_state
        ORDER BY default_rate_pct DESC
    """)
    conn.commit()
    print("   summary_by_state table created")

    # --- 5D: DEFAULT RATE BY YEAR ---
    print("   [5D] Creating default rate by approval year...")
    conn.execute("DROP TABLE IF EXISTS summary_by_year")
    conn.execute("""
        CREATE TABLE summary_by_year AS
        SELECT
            approval_year,
            COUNT(*)                        AS total_loans,
            SUM(is_default)                 AS total_defaults,
            ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
            ROUND(AVG(loan_amount), 0)      AS avg_loan_amount,
            ROUND(AVG(interest_rate), 2)    AS avg_interest_rate
        FROM loans_clean
        GROUP BY approval_year
        ORDER BY approval_year
    """)
    conn.commit()
    print("   summary_by_year table created")

    # --- 5E: DEFAULT RATE BY BUSINESS TYPE ---
    print("   [5E] Creating default rate by business type...")
    conn.execute("DROP TABLE IF EXISTS summary_by_business_type")
    conn.execute("""
        CREATE TABLE summary_by_business_type AS
        SELECT
            business_type,
            business_age,
            COUNT(*)                        AS total_loans,
            SUM(is_default)                 AS total_defaults,
            ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
            ROUND(AVG(loan_amount), 0)      AS avg_loan_amount
        FROM loans_clean
        GROUP BY business_type, business_age
        ORDER BY default_rate_pct DESC
    """)
    conn.commit()
    print("   summary_by_business_type table created")

    # --- 5F: WINDOW FUNCTION — LOAN RANK WITHIN INDUSTRY ---
    print("   [5F] Creating loan rank within industry table...")
    conn.execute("DROP TABLE IF EXISTS loans_ranked")
    conn.execute("""
        CREATE TABLE loans_ranked AS
        SELECT
            loan_id,
            naics_sector,
            loan_amount,
            interest_rate,
            is_default,
            RANK() OVER (
                PARTITION BY naics_sector
                ORDER BY loan_amount DESC
            ) AS rank_by_amount_in_sector,
            AVG(is_default) OVER (
                PARTITION BY naics_sector
            ) AS sector_default_rate,
            AVG(loan_amount) OVER (
                PARTITION BY naics_sector
            ) AS sector_avg_loan_amount
        FROM loans_clean
    """)
    conn.commit()
    print("   loans_ranked table created")

     # --- 5G: CTE EXAMPLE — HIGH RISK LOANS ---
    print("   [5G] Creating high risk loan summary...")
    conn.execute("DROP TABLE IF EXISTS high_risk_loans")
    conn.execute("""
        CREATE TABLE high_risk_loans AS
        WITH sector_stats AS (
            SELECT
                naics_sector,
                AVG(is_default)                     AS sector_default_rate,
                AVG(loan_amount)                    AS sector_avg_amount,
                AVG(loan_amount) * 3.0              AS sector_p95_amount,
                AVG(interest_rate)                  AS sector_avg_rate,
                AVG(term_months)                    AS sector_avg_term,
                AVG(sba_guarantee_pct)              AS sector_avg_guarantee_pct
            FROM loans_clean
            GROUP BY naics_sector
        ),
        district_stats AS (
            SELECT
                sba_district,
                AVG(is_default)                     AS district_default_rate
            FROM loans_clean
            GROUP BY sba_district
        ),
        loan_flags AS (
            SELECT
                l.loan_id,
                l.naics_sector,
                l.loan_amount,
                l.interest_rate,
                l.term_months,
                l.business_age,
                l.has_collateral,
                l.sba_guarantee_pct,
                l.jobs_supported,
                l.sba_district,
                l.is_default,
                s.sector_default_rate,
                s.sector_avg_amount,
                d.district_default_rate,

                -- FLAG 1: Above average loan size for sector (weight 1)
                -- Mild signal — loan is bigger than peers but not extreme
                CASE
                    WHEN l.loan_amount > s.sector_avg_amount
                    THEN 1 ELSE 0
                END AS above_avg_loan_size,

                -- FLAG 2: Extreme loan size — 3x sector average (weight 3)
                -- Strong signal — outlier ask relative to industry peers
                CASE
                    WHEN l.loan_amount > s.sector_p95_amount
                    THEN 3 ELSE 0
                END AS extreme_loan_score,

                -- FLAG 3: Above average interest rate for sector (weight 1)
                -- Mild signal — priced higher than peers
                CASE
                    WHEN l.interest_rate > s.sector_avg_rate
                    THEN 1 ELSE 0
                END AS above_avg_rate,

                -- FLAG 4: Extreme interest rate — above 15% (weight 3)
                -- Strong signal — very high cost of capital
                CASE
                    WHEN l.interest_rate > 15.0
                    THEN 3 ELSE 0
                END AS extreme_rate_score,

                -- FLAG 5: New business (weight 2)
                -- Strong signal — new businesses default at nearly 2x rate
                CASE
                    WHEN l.business_age = 'NEW BUSINESS OR 2 YEARS OR LESS'
                    THEN 2 ELSE 0
                END AS new_business_score,

                -- FLAG 6: No collateral (weight 2)
                -- Strong signal — nothing backing the loan
                CASE
                    WHEN l.has_collateral = 'N'
                    THEN 2 ELSE 0
                END AS no_collateral_score,

                -- FLAG 7: Extreme term length — over 240 months (weight 2)
                -- Signal — 20+ year commitment is aggressive for most business types
                CASE
                    WHEN l.term_months > 240
                    THEN 2 ELSE 0
                END AS extreme_term_score,

                -- FLAG 8: Very short term at high rate (weight 2)
                -- Signal — predatory structure, high repayment pressure
                CASE
                    WHEN l.term_months < 24
                    AND l.interest_rate > s.sector_avg_rate
                    THEN 2 ELSE 0
                END AS short_term_high_rate_score,

                -- FLAG 9: SBA guarantee too high — over 90% (weight 1)
                -- Mild signal — bank has almost no skin in the game
                CASE
                    WHEN l.sba_guarantee_pct > 0.90
                    THEN 1 ELSE 0
                END AS high_guarantee_score,

                -- FLAG 10: SBA guarantee unusually low — under 40% (weight 1)
                -- Mild signal — unusual structure worth investigating
                CASE
                    WHEN l.sba_guarantee_pct < 0.40
                    THEN 1 ELSE 0
                END AS low_guarantee_score,

                -- FLAG 11: Extreme jobs supported — over 200 (weight 2)
                -- Signal — self-reported, unaudited, likely misrepresentation
                CASE
                    WHEN l.jobs_supported > 200
                    THEN 2 ELSE 0
                END AS extreme_jobs_score,

                -- FLAG 12: High default rate district (weight 2)
                -- Signal — geographic concentration of bad loans
                CASE
                    WHEN d.district_default_rate > 0.25
                    THEN 2 ELSE 0
                END AS high_risk_district_score,

                -- FLAG 13: High default rate sector (weight 2)
                -- Signal — operating in an industry with structural default risk
                CASE
                    WHEN s.sector_default_rate > 0.25
                    THEN 2 ELSE 0
                END AS high_risk_sector_score

            FROM loans_clean l
            JOIN sector_stats s  ON l.naics_sector = s.naics_sector
            JOIN district_stats d ON l.sba_district = d.sba_district
        )
        SELECT *,
            -- Total weighted risk score
            (above_avg_loan_size +
             extreme_loan_score +
             above_avg_rate +
             extreme_rate_score +
             new_business_score +
             no_collateral_score +
             extreme_term_score +
             short_term_high_rate_score +
             high_guarantee_score +
             low_guarantee_score +
             extreme_jobs_score +
             high_risk_district_score +
             high_risk_sector_score)         AS weighted_risk_score,

            -- Risk tier based on total score
            CASE
                WHEN (above_avg_loan_size +
                      extreme_loan_score +
                      above_avg_rate +
                      extreme_rate_score +
                      new_business_score +
                      no_collateral_score +
                      extreme_term_score +
                      short_term_high_rate_score +
                      high_guarantee_score +
                      low_guarantee_score +
                      extreme_jobs_score +
                      high_risk_district_score +
                      high_risk_sector_score) >= 8 THEN 'HIGH'
                WHEN (above_avg_loan_size +
                      extreme_loan_score +
                      above_avg_rate +
                      extreme_rate_score +
                      new_business_score +
                      no_collateral_score +
                      extreme_term_score +
                      short_term_high_rate_score +
                      high_guarantee_score +
                      low_guarantee_score +
                      extreme_jobs_score +
                      high_risk_district_score +
                      high_risk_sector_score) >= 4 THEN 'MEDIUM'
                ELSE 'LOW'
            END                             AS risk_tier

        FROM loan_flags
        WHERE (above_avg_loan_size +
               extreme_loan_score +
               above_avg_rate +
               extreme_rate_score +
               new_business_score +
               no_collateral_score +
               extreme_term_score +
               short_term_high_rate_score +
               high_guarantee_score +
               low_guarantee_score +
               extreme_jobs_score +
               high_risk_district_score +
               high_risk_sector_score) >= 2
    """)
    conn.commit()
    print("   high_risk_loans table created")

# =============================================================================
# STEP 6: FINAL VALIDATION
# =============================================================================

def validate_output():
    print("\n[6/6] Validating final output...")

    conn = sqlite3.connect(DB_PATH)

    # Check all expected tables exist
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name",
        conn
    )
    print("\n   Tables and views in database:")
    for t in tables["name"]:
        count = pd.read_sql(f"SELECT COUNT(*) as n FROM {t}", conn)
        print(f"   {t:<35} {count['n'][0]:>10,} rows")

    # Sample the clean view
    print("\n   --- Sample from loans_clean ---")
    sample = pd.read_sql("""
        SELECT
            loan_id,
            borr_state,
            loan_amount,
            interest_rate,
            term_months,
            naics_sector,
            business_age,
            is_default,
            loan_size_bucket
        FROM loans_clean
        LIMIT 5
    """, conn)
    print(sample.to_string(index=False))

    # Default rate summary
    print("\n   --- Default Rate Summary ---")
    default_summary = pd.read_sql("""
        SELECT
            loan_status,
            COUNT(*)                        AS count,
            ROUND(COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER(), 2)    AS pct
        FROM loans_clean
        GROUP BY loan_status
    """, conn)
    print(default_summary.to_string(index=False))

    # Top 5 industries by default rate
    print("\n   --- Top 5 Industries by Default Rate ---")
    top_industries = pd.read_sql("""
        SELECT
            naics_sector,
            industry_description,
            total_loans,
            default_rate_pct
        FROM summary_by_industry
        WHERE total_loans > 100
        ORDER BY default_rate_pct DESC
        LIMIT 5
    """, conn)
    print(top_industries.to_string(index=False))

    # Top 5 states by default rate
    print("\n   --- Top 5 States by Default Rate ---")
    top_states = pd.read_sql("""
        SELECT
            borr_state,
            total_loans,
            default_rate_pct
        FROM summary_by_state
        WHERE total_loans > 100
        ORDER BY default_rate_pct DESC
        LIMIT 5
    """, conn)
    print(top_states.to_string(index=False))

    conn.close()
    print("\n=== Phase 1 Complete ===")
    print(f"Clean database saved to: {DB_PATH}")
    print("Ready for Phase 2 — Exploratory Analysis in R")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Step 1 — Load raw CSV files
    df = load_raw_data()

    # Step 2 — Profile raw data
    null_report = profile_data(df)

    # Step 3 — Clean the data
    df_clean = clean_data(df)

    # Step 4 — Load into SQLite
    load_to_sqlite(df_clean)

    # Step 5 — Run SQL cleaning queries
    run_sql_cleaning()

    # Step 6 — Validate output
    validate_output()
