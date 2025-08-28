import os
import smtplib
import ssl
from email.message import EmailMessage
from urllib.parse import quote
from pathlib import Path

try:
    # Load .env if not already loaded elsewhere
    from dotenv import load_dotenv
    _env_path = Path(__file__).with_name('.env')
    load_dotenv(dotenv_path=_env_path)
except Exception:
    pass

# SMTP configuration (environment-driven for flexibility)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM") or SMTP_USER or ""
APP_NAME = os.getenv("APP_NAME", "Auth App")
# Frontend URL to build clickable password reset link for users
FRONTEND_RESET_URL = os.getenv("FRONTEND_RESET_URL", "http://localhost:8501")


def _build_reset_message(to_email: str, token: str) -> EmailMessage:
    # Compose a plain-text email with both token and deep link
    link = f"{FRONTEND_RESET_URL}?token={quote(token)}"
    subject = f"{APP_NAME} password reset"
    body = (
        f"You requested a password reset for {APP_NAME}.\n\n"
        f"Use this token in the app: {token}\n\n"
        f"Or click this link: {link}\n\n"
        f"This token expires in 15 minutes and can be used once."
    )
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    return msg


def send_password_reset_email(to_email: str, token: str) -> bool:
    # Fail fast if SMTP isn't configured (useful in dev)
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASSWORD and EMAIL_FROM):
        return False

    msg = _build_reset_message(to_email, token)
    try:
        # Support both SMTPS (465) and STARTTLS (e.g., 587)
        if SMTP_PORT == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        return True
    except Exception:
        # Intentionally swallow errors; caller treats False as non-fatal
        return False
