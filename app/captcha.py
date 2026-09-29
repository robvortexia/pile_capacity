"""Cloudflare Turnstile check for the signup modal and the suggestions form.

The site already sits behind Cloudflare, so Turnstile adds no new vendor and
most people never see a puzzle. Create a widget in the Cloudflare dashboard
(Turnstile > Add widget, hostname uwa-geotech-cpt-calculator.com) and set
TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY on Render to switch it on.
Without both keys the forms work exactly as before, with no captcha.
"""
import json
import logging
import urllib.parse
import urllib.request

from flask import current_app, request

logger = logging.getLogger(__name__)

VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'
# Cloudflare documents tokens as at most 2048 characters.
_MAX_TOKEN_LEN = 2048


def captcha_site_key():
    """The public site key when the captcha is fully configured, else ''."""
    cfg = current_app.config
    if cfg.get('TURNSTILE_SITE_KEY') and cfg.get('TURNSTILE_SECRET_KEY'):
        return cfg['TURNSTILE_SITE_KEY']
    return ''


def captcha_passed(remote_ip=None):
    """Verify the Turnstile token posted with the current form.

    True when the captcha is off. A missing, oversized or rejected token
    fails. If Cloudflare itself can't be reached the submission is let
    through (and logged): locking real engineers out of a free tool over a
    network blip costs more than the odd spam row."""
    if not captcha_site_key():
        return True

    token = request.form.get('cf-turnstile-response', '')
    if not token or len(token) > _MAX_TOKEN_LEN:
        return False

    fields = {'secret': current_app.config['TURNSTILE_SECRET_KEY'], 'response': token}
    if remote_ip:
        fields['remoteip'] = remote_ip
    try:
        req = urllib.request.Request(
            VERIFY_URL, data=urllib.parse.urlencode(fields).encode(),
            headers={'User-Agent': 'UWA-CPT-Calculator'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Turnstile verify unreachable, allowing submission: %s", e)
        return True

    if not result.get('success'):
        logger.info("Turnstile rejected a submission: %s", result.get('error-codes'))
        return False
    return True
