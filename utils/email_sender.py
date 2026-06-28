"""
utils/email_sender.py

Sends ECOA-compliant adverse action letters via Gmail SMTP.
Each letter is delivered as a plain-text email with the full
letter body inline — no attachments required for compliance.

Gmail SMTP configuration:
    Host: smtp.gmail.com
    Port: 587 (TLS)
    Auth: Gmail App Password (never your regular Gmail password)

In DRY_RUN mode the SMTP connection is never opened — emails are
logged to disk only. This prevents accidental sends during testing.
"""

import os
import smtplib
import csv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# ── Gmail SMTP configuration ──────────────────────────────────────────────────
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SENDER_NAME   = os.getenv("SENDER_NAME", "Credit Risk Decision Engine")

# ── Email content configuration ───────────────────────────────────────────────
EMAIL_SUBJECT_DENIED      = "Notice of Adverse Action on Your SBA Loan Application"
EMAIL_SUBJECT_UNFAVORABLE = "Notice of Loan Approval with Modified Terms"

# ── Output paths ──────────────────────────────────────────────────────────────
EMAIL_LOG_PATH = os.path.join("outputs", "phase6_email_log.csv")


def _build_email(
    recipient_address: str,
    applicant_id: str,
    decision: str,
    letter_text: str,
) -> MIMEMultipart:
    """
    Build a MIME email message containing the adverse action letter.

    Parameters
    ----------
    recipient_address : str
        Destination email address for this applicant.
    applicant_id : str
        Unique applicant identifier used in the subject line.
    decision : str
        Decision label used to select the appropriate subject line.
    letter_text : str
        Full adverse action letter text to include as email body.

    Returns
    -------
    MIMEMultipart
        Fully constructed email message ready for SMTP delivery.
    """
    subject = (
        EMAIL_SUBJECT_DENIED
        if decision == "Denied"
        else EMAIL_SUBJECT_UNFAVORABLE
    )

    msg = MIMEMultipart()
    msg["From"]    = f"{SENDER_NAME} <{GMAIL_ADDRESS}>"
    msg["To"]      = recipient_address
    msg["Subject"] = f"{subject} — Ref: {applicant_id}"

    msg.attach(MIMEText(letter_text, "plain"))

    return msg


def _log_email(
    log_rows: List[Dict],
    applicant_id: str,
    recipient: str,
    decision: str,
    status: str,
    error: str = "",
) -> None:
    """
    Append a single email delivery record to the in-memory log list.

    Parameters
    ----------
    log_rows : List[Dict]
        Running list of log row dicts accumulated during the send loop.
    applicant_id : str
        Unique applicant identifier.
    recipient : str
        Destination email address.
    decision : str
        Decision label for this applicant.
    status : str
        Delivery outcome — 'sent', 'dry_run', or 'error'.
    error : str
        Error message if status is 'error', empty string otherwise.
    """
    log_rows.append({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "applicant_id": applicant_id,
        "recipient":    recipient,
        "decision":     decision,
        "status":       status,
        "error":        error,
    })


def _write_email_log(log_rows: List[Dict]) -> None:
    """
    Write the accumulated email log to disk as a CSV file.

    Creates or overwrites outputs/phase6_email_log.csv with all
    delivery records from the current pipeline run.

    Parameters
    ----------
    log_rows : List[Dict]
        List of log row dicts built during the send loop.
    """
    os.makedirs("outputs", exist_ok=True)
    fieldnames = [
        "timestamp", "applicant_id", "recipient",
        "decision", "status", "error"
    ]
    with open(EMAIL_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"[email_sender] Email log saved: {EMAIL_LOG_PATH}")


def send_all_emails(
    letter_results: Dict,
    recipient_address: str,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Send adverse action letters via Gmail SMTP to all flagged applicants.

    In a production system each applicant would have their own email
    address on file. For this portfolio project all letters are routed
    to a single recipient address (your Gmail) for safe demonstration.

    Opens a single SMTP connection and reuses it across all sends to
    avoid repeated authentication overhead. Falls back gracefully on
    per-message errors without aborting the full batch.

    In dry_run mode the SMTP connection is never opened — all records
    are logged with status 'dry_run' and no emails are sent.

    Parameters
    ----------
    letter_results : Dict
        Output from generate_all_letters() — keys are applicant IDs,
        values are dicts with 'letter_path', 'decision', and 'status'.
    recipient_address : str
        Email address to deliver all letters to during demonstration.
    dry_run : bool
        If True, skips all SMTP calls and logs dry_run status only.

    Returns
    -------
    List[Dict]
        Email delivery log rows written to phase6_email_log.csv.
    """
    # Only attempt to send letters that were successfully generated
    sendable = {
        aid: r for aid, r in letter_results.items()
        if r["status"] in ("generated", "dry_run")
    }
    total = len(sendable)

    print(f"[email_sender] Sending {total:,} emails "
          f"({'DRY RUN' if dry_run else 'LIVE'})...")

    if not dry_run:
        if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
            raise EnvironmentError(
                "[email_sender] GMAIL_ADDRESS or GMAIL_APP_PASSWORD "
                "not found in environment. Check your .env file."
            )

    log_rows: List[Dict] = []

    if dry_run:
        # Log all records without opening SMTP connection
        for applicant_id, result in sendable.items():
            print(f"[email_sender] DRY RUN — would send to "
                  f"{recipient_address} | "
                  f"Applicant {applicant_id} | {result['decision']}")
            _log_email(
                log_rows, applicant_id, recipient_address,
                result["decision"], "dry_run"
            )

        _write_email_log(log_rows)
        print(f"[email_sender] DRY RUN complete — "
              f"{total:,} records logged, 0 emails sent.")
        return log_rows

    # Open a single persistent SMTP connection for the full batch
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            print(f"[email_sender] SMTP authenticated as {GMAIL_ADDRESS}")

            for i, (applicant_id, result) in enumerate(sendable.items(), 1):
                decision    = result["decision"]
                letter_path = result.get("letter_path")

                print(f"[email_sender] ({i}/{total}) "
                      f"Applicant {applicant_id} | {decision}")

                # Read letter text from disk
                if not letter_path or not os.path.exists(letter_path):
                    error_msg = f"Letter file not found: {letter_path}"
                    print(f"[email_sender] ERROR — {error_msg}")
                    _log_email(
                        log_rows, applicant_id, recipient_address,
                        decision, "error", error_msg
                    )
                    continue

                with open(letter_path, "r") as f:
                    letter_text = f.read()

                try:
                    msg = _build_email(
                        recipient_address, applicant_id,
                        decision, letter_text
                    )
                    server.sendmail(
                        GMAIL_ADDRESS,
                        recipient_address,
                        msg.as_string()
                    )
                    _log_email(
                        log_rows, applicant_id, recipient_address,
                        decision, "sent"
                    )

                except smtplib.SMTPException as e:
                    error_msg = str(e)
                    print(f"[email_sender] SMTP ERROR for applicant "
                          f"{applicant_id}: {error_msg}")
                    _log_email(
                        log_rows, applicant_id, recipient_address,
                        decision, "error", error_msg
                    )

    except smtplib.SMTPAuthenticationError:
        raise RuntimeError(
            "[email_sender] Gmail authentication failed. "
            "Verify GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env. "
            "Ensure you are using a Gmail App Password, not your "
            "regular Gmail password."
        )

    sent   = sum(1 for r in log_rows if r["status"] == "sent")
    errors = sum(1 for r in log_rows if r["status"] == "error")

    _write_email_log(log_rows)
    print(f"[email_sender] Complete — Sent: {sent} | Errors: {errors}")

    return log_rows