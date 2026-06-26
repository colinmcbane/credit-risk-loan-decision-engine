"""
utils/letter_generator.py

Generates ECOA-compliant adverse action letters using the Gemini API.
Each letter is personalized with the applicant's top 3 SHAP-derived
reason codes extracted by the decision classifier.

ECOA Regulation B requires adverse action notices to include:
    1. A statement of the action taken
    2. The name and address of the creditor
    3. A statement of the provisions of the ECOA
    4. The name and address of the federal agency that administers
       compliance with respect to the creditor
    5. Either a statement of specific reasons for the action or
       a disclosure of the applicant's right to request reasons

Reference: 12 CFR Part 1002 (Regulation B)
"""

import os
import time
from google import genai
from dotenv import load_dotenv
from typing import List

load_dotenv()

# ── Gemini configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-1.5-flash"

# ── Creditor information inserted into every letter ───────────────────────────
CREDITOR_NAME    = "SBA Credit Risk Decision Engine"
CREDITOR_ADDRESS = "123 Financial Plaza, Suite 400, Charlotte, NC 28202"
CFPB_CONTACT     = (
    "Consumer Financial Protection Bureau, "
    "1700 G Street NW, Washington, DC 20552 | "
    "www.consumerfinance.gov | 1-855-411-2372"
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Gemini free tier allows 15 requests per minute.
# 4.5 second delay between calls keeps us safely under the limit.
REQUEST_DELAY_SECONDS = 4.5


def initialize_gemini() -> genai.Client:
    """
    Initialize and return the Gemini API client.

    Reads GEMINI_API_KEY from environment variables loaded via dotenv.

    Returns
    -------
    genai.Client
        Configured Gemini API client instance.

    Raises
    ------
    EnvironmentError
        If GEMINI_API_KEY is not set in the environment.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "[letter_generator] GEMINI_API_KEY not found. "
            "Ensure it is set in your .env file."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[letter_generator] Gemini client initialized: {GEMINI_MODEL}")
    return client


def _build_prompt(
    applicant_id: str,
    decision: str,
    predicted_prob: float,
    reason_codes: List[str],
) -> str:
    """
    Build the Gemini prompt for a single adverse action letter.

    Parameters
    ----------
    applicant_id : str
        Unique identifier for the loan applicant (row index as string).
    decision : str
        Decision label — 'Denied' or 'Approved - Unfavorable Terms'.
    predicted_prob : float
        Predicted default probability from the champion model.
    reason_codes : List[str]
        Top 3 SHAP-derived adverse action reason codes.

    Returns
    -------
    str
        Fully formatted prompt string for the Gemini API.
    """
    reason_list = "\n".join(
        f"  {i + 1}. {reason}" for i, reason in enumerate(reason_codes)
    )

    action_statement = (
        "your loan application has been denied"
        if decision == "Denied"
        else (
            "your loan application has been approved; however, "
            "the terms offered are less favorable than those requested "
            "due to elevated credit risk indicators"
        )
    )

    prompt = f"""You are a compliance officer at an SBA-affiliated lending institution.
Write a formal, professional ECOA-compliant adverse action letter for the following loan application.

APPLICANT REFERENCE ID: {applicant_id}
DECISION: {decision}
PREDICTED DEFAULT PROBABILITY: {predicted_prob:.1%}
PRIMARY ADVERSE FACTORS (from credit risk model):
{reason_list}

The letter must include ALL of the following sections in order:
1. Date and applicant reference
2. Clear statement that {action_statement}
3. The specific reasons for this decision based on the adverse factors listed above —
   explain each reason in plain language a small business owner would understand,
   do not use the raw factor names verbatim
4. A statement of rights under the Equal Credit Opportunity Act (ECOA)
5. The applicant's right to obtain a free copy of their credit report within 60 days
6. Contact information for the Consumer Financial Protection Bureau:
   {CFPB_CONTACT}
7. Creditor name and address:
   {CREDITOR_NAME}
   {CREDITOR_ADDRESS}
8. A professional closing

Tone: formal, respectful, and compliant. Do not mention specific probability scores
or model names in the letter body. Do not use placeholder text — write the complete
letter ready to send. Keep the letter under 500 words."""

    return prompt


def generate_letter(
    applicant_id: str,
    decision: str,
    predicted_prob: float,
    reason_codes: List[str],
    client: genai.Client,
) -> str:
    """
    Generate a single adverse action letter via the Gemini API.

    Parameters
    ----------
    applicant_id : str
        Unique identifier for the loan applicant.
    decision : str
        Decision label — 'Denied' or 'Approved - Unfavorable Terms'.
    predicted_prob : float
        Predicted default probability from the champion model.
    reason_codes : List[str]
        Top 3 SHAP-derived adverse action reason codes.
    client : genai.Client
        Initialized Gemini API client.

    Returns
    -------
    str
        Complete adverse action letter text ready for saving and sending.

    Raises
    ------
    RuntimeError
        If the Gemini API returns an empty or blocked response.
    """
    prompt = _build_prompt(
        applicant_id, decision, predicted_prob, reason_codes
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response or not response.text:
        raise RuntimeError(
            f"[letter_generator] Gemini returned empty response "
            f"for applicant {applicant_id}. "
            "Check API key and content safety filters."
        )

    return response.text.strip()


def generate_all_letters(
    df: object,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """
    Generate adverse action letters for all flagged applicants.

    Iterates over all rows where requires_adverse_action is True,
    calls the Gemini API for each, saves letters to disk as .txt files,
    and returns a summary dictionary for logging.

    In dry_run mode, skips the Gemini API call entirely and writes
    a placeholder letter to disk so the full pipeline can be tested
    without consuming API quota.

    Parameters
    ----------
    df : pd.DataFrame
        Classified loan DataFrame with 'requires_adverse_action',
        'decision', 'predicted_prob', and 'reason_codes' columns.
    output_dir : str
        Directory path where individual letter .txt files are saved.
    dry_run : bool
        If True, skips Gemini API calls and writes placeholder letters.

    Returns
    -------
    dict
        Keys are applicant IDs (str), values are dicts with keys:
            - letter_path: str path to saved .txt file
            - decision: str
            - status: 'generated', 'dry_run', or 'error'
            - error: str error message if status is 'error'
    """
    os.makedirs(output_dir, exist_ok=True)

    adverse_df = df[df["requires_adverse_action"]].copy()
    total      = len(adverse_df)

    print(f"[letter_generator] Generating letters for {total:,} applicants "
          f"({'DRY RUN' if dry_run else 'LIVE'})...")

    if not dry_run:
        client = initialize_gemini()
    else:
        client = None
        print("[letter_generator] DRY RUN — Gemini API calls skipped.")

    results = {}

    for i, (idx, row) in enumerate(adverse_df.iterrows(), 1):
        applicant_id   = str(idx)
        decision       = row["decision"]
        predicted_prob = float(row["predicted_prob"])
        reason_codes   = row["reason_codes"]
        letter_path    = os.path.join(
            output_dir, f"letter_{applicant_id}.txt"
        )

        print(f"[letter_generator] ({i}/{total}) "
              f"Applicant {applicant_id} | {decision} | "
              f"PD={predicted_prob:.1%}")

        if dry_run:
            letter_text = (
                f"[DRY RUN] Adverse Action Letter\n"
                f"Applicant ID: {applicant_id}\n"
                f"Decision: {decision}\n"
                f"Predicted Default Probability: {predicted_prob:.1%}\n"
                f"Reason Codes:\n"
                + "\n".join(f"  - {r}" for r in reason_codes)
                + "\n\n[Gemini API call skipped in dry run mode]"
            )
            with open(letter_path, "w") as f:
                f.write(letter_text)

            results[applicant_id] = {
                "letter_path": letter_path,
                "decision":    decision,
                "status":      "dry_run",
            }

        else:
            try:
                letter_text = generate_letter(
                    applicant_id, decision,
                    predicted_prob, reason_codes, client
                )
                with open(letter_path, "w") as f:
                    f.write(letter_text)

                results[applicant_id] = {
                    "letter_path": letter_path,
                    "decision":    decision,
                    "status":      "generated",
                }

                # Respect Gemini free tier rate limit between calls
                if i < total:
                    time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                print(f"[letter_generator] ERROR for applicant "
                      f"{applicant_id}: {e}")
                results[applicant_id] = {
                    "letter_path": None,
                    "decision":    decision,
                    "status":      "error",
                    "error":       str(e),
                }

    generated     = sum(1 for r in results.values() if r["status"] == "generated")
    dry_run_count = sum(1 for r in results.values() if r["status"] == "dry_run")
    errors        = sum(1 for r in results.values() if r["status"] == "error")

    print(f"[letter_generator] Complete — "
          f"Generated: {generated} | "
          f"Dry run: {dry_run_count} | "
          f"Errors: {errors}")

    return results