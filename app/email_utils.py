import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import List
from flask import current_app
from .analytics import get_weekly_usage_summary


def _get_config(key: str, default=None):
    # Prefer Flask config, then env var, then default
    if current_app and current_app.config.get(key) is not None:
        return current_app.config.get(key)
    return os.environ.get(key, default)


def _get_recipients() -> List[str]:
    # Comma-separated list from config/env
    cfg_val = _get_config('WEEKLY_REPORT_RECIPIENTS', '')
    if isinstance(cfg_val, list):
        return cfg_val
    return [email.strip() for email in cfg_val.split(',') if email.strip()]


def build_weekly_usage_email(subject_prefix: str = 'Site usage summary') -> EmailMessage:
    summary = get_weekly_usage_summary(days=int(_get_config('WEEKLY_REPORT_DAYS', 7)))

    # Plain-text body for simplicity and compatibility
    lines = []
    lines.append(f"Date range: {summary['range']['start']} → {summary['range']['end']}")
    lines.append("")
    lines.append("Totals:")
    lines.append(f"- Page visits: {summary['totals']['page_visits']}")
    lines.append(f"- Unique visitors: {summary['totals']['unique_visitors']}")
    lines.append(f"- Registrations: {summary['totals']['registrations']}")
    lines.append("")

    if summary['top_pages']:
        lines.append("Top pages:")
        for item in summary['top_pages']:
            lines.append(f"- {item['page']}: {item['count']}")
        lines.append("")

    if summary['events']:
        lines.append("Events by type:")
        for item in summary['events']:
            lines.append(f"- {item['event_type']}: {item['count']}")
        lines.append("")

    if summary['pile_types']:
        lines.append("Pile selections:")
        for item in summary['pile_types']:
            lines.append(f"- {item['type']}: {item['count']}")
        lines.append("")

    sender = _get_config('MAIL_DEFAULT_SENDER', _get_config('MAIL_USERNAME', 'noreply@example.com'))
    recipients = _get_recipients()

    msg = EmailMessage()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    subject_suffix = _get_config('SITE_NAME', 'UWA Pile Calculator')
    msg['Subject'] = f"{subject_prefix} — {subject_suffix}"
    msg.set_content('\n'.join(lines))
    return msg


def send_email(msg: EmailMessage) -> None:
    host = _get_config('MAIL_SERVER', 'smtp.gmail.com')
    port = int(_get_config('MAIL_PORT', 587))
    username = _get_config('MAIL_USERNAME')
    password = _get_config('MAIL_PASSWORD')
    use_tls = str(_get_config('MAIL_USE_TLS', 'true')).lower() == 'true'
    use_ssl = str(_get_config('MAIL_USE_SSL', 'false')).lower() == 'true'

    if not username or not password:
        raise RuntimeError('MAIL_USERNAME and MAIL_PASSWORD must be configured to send email')

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as server:
            if use_tls:
                server.starttls(context=ssl.create_default_context())
            server.login(username, password)
            server.send_message(msg)


def send_weekly_usage_email() -> None:
    recipients = _get_recipients()
    if not recipients:
        raise RuntimeError('WEEKLY_REPORT_RECIPIENTS is not configured')
    msg = build_weekly_usage_email()
    send_email(msg)


