import smtplib
import resend
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.core.config import is_local_environment, settings

resend.api_key = settings.resend_api_key

_FROM = f"{settings.email_from_name} <{settings.email_from}>"


# ═══════════════════════════════════════════════════════
# CORE SEND HELPER
# ═══════════════════════════════════════════════════════

async def _send_email(to_email: str, subject: str, html: str) -> bool:
    """
    Single dispatch point for all outgoing email.
    - local / development  →  Gmail SMTP (no Resend quota burned)
    - staging / production →  Resend
    """
    try:
        if is_local_environment():
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = _FROM
            msg["To"]      = to_email
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(settings.gmail_user, settings.gmail_app_password)
                server.sendmail(settings.gmail_user, to_email, msg.as_string())

        else:
            resend.Emails.send({
                "from":    _FROM,
                "to":      [to_email],
                "subject": subject,
                "html":    html,
            })

        return True

    except Exception as exc:
        print(f"[Email] Failed → {to_email} | {subject} | {exc}")
        return False


# ═══════════════════════════════════════════════════════
# HTML BUILDERS
# ═══════════════════════════════════════════════════════

def _otp_html(name: str, otp: str, purpose: str) -> str:
    action = {
        "signup":          "verify your email address",
        "login":           "sign in to your account",
        "forgot_password": "reset your password",
    }.get(purpose, "continue")

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/></head>
<body style="margin:0;padding:0;background:#0a0f14;font-family:system-ui,sans-serif;">
<div style="max-width:480px;margin:0 auto;padding:40px 20px;">
  <div style="text-align:center;margin-bottom:32px;">
    <span style="color:#fff;font-size:22px;font-weight:700;">⚡ CryptoAI Insights</span>
  </div>
  <div style="background:#111827;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:40px;">
    <h1 style="color:#fff;font-size:20px;font-weight:700;margin:0 0 8px;">
      Your verification code
    </h1>
    <p style="color:#9ca3af;font-size:15px;margin:0 0 28px;line-height:1.6;">
      Hi {name}, use the code below to {action}. It expires in {settings.otp_expire_minutes} minutes.
    </p>
    <div style="text-align:center;margin:0 0 28px;">
      <div style="display:inline-block;background:#0f172a;text-align:center;border:1px solid rgba(0,212,255,0.3);border-radius:12px;padding:20px 40px;">
        <span style="color:#00D4FF;font-size:38px;font-weight:700;letter-spacing:10px;font-family:monospace;">{otp}</span>
      </div>
      <p style="color:#6b7280;font-size:13px;margin:10px 0 0;">
        Expires in {settings.otp_expire_minutes} minutes
      </p>
    </div>
    <div style="background:#1f2937;border-radius:8px;padding:14px;">
      <p style="color:#9ca3af;font-size:13px;margin:0;">
        🔒 If you didn't request this, you can safely ignore this email.
        Never share this code with anyone.
      </p>
    </div>
  </div>
  <p style="color:#4b5563;font-size:12px;text-align:center;margin-top:24px;">
    © 2024 CryptoAI Insights · Not financial advice
  </p>
</div></body></html>"""


def _welcome_html(name: str, points: int) -> str:
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#0a0f14;font-family:system-ui,sans-serif;">
<div style="max-width:480px;margin:0 auto;padding:40px 20px;">
  <div style="background:#111827;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:40px;">
    <h1 style="color:#fff;font-size:22px;font-weight:700;margin:0 0 16px;">
      Welcome to CryptoAI Insights! 🎉
    </h1>
    <p style="color:#9ca3af;font-size:15px;margin:0 0 24px;line-height:1.6;">
      Hi {name}, your account is ready. You have
      <strong style="color:#00D4FF;">{points} free points</strong> —
      that's {points // 10} AI analyses on us.
    </p>
    <a href="{settings.frontend_url}/dashboard"
       style="display:inline-block;background:#00D4FF;color:#000;font-weight:700;font-size:14px;padding:14px 28px;border-radius:10px;text-decoration:none;">
      Start analyzing →
    </a>
  </div>
</div></body></html>"""


# ═══════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════

SUBJECT_MAP = {
    "signup":          "Verify your CryptoAI account",
    "login":           "Your CryptoAI login code",
    "forgot_password": "Reset your CryptoAI password",
}


async def send_otp_email(to_email: str, name: str, otp: str, purpose: str) -> bool:
    subject = SUBJECT_MAP.get(purpose, "Your CryptoAI verification code")
    return await _send_email(to_email, subject, _otp_html(name, otp, purpose))


async def send_welcome_email(to_email: str, name: str, points: int) -> bool:
    subject = f"Welcome to CryptoAI — {points} free points added!"
    return await _send_email(to_email, subject, _welcome_html(name, points))