from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session, Response, send_file, current_app, make_response, has_request_context
from werkzeug.utils import secure_filename
import pandas as pd
import os
import math
import json
import plotly.graph_objects as go
import plotly.utils
import numpy as np
from .utils import (
    save_cpt_data, load_cpt_data, create_cpt_graphs,
    save_graphs_data, load_graphs_data, generate_csv_download,
    save_debug_details, load_debug_details, create_bored_pile_graphs,
    save_calculation_results, load_calculation_results, create_helical_pile_graphs,
    sample_processed_profile
)
from .calculations import calculate_pile_capacity, process_cpt_data, pre_input_calc, get_iterative_values, calculate_bored_pile_results, calculate_helical_pile_results, calculate_driven_pile_results, compute_capacity_envelope_bored, compute_capacity_envelope_driven
from .interpolation import process_uploaded_cpt_data
from datetime import datetime, timedelta
from .models import db, Registration, Visit, Suggestion, AnalyticsData, SavedCalculation, CalcFlow
from functools import wraps
from hmac import compare_digest
from collections import deque
from time import time as _now
import csv
import re
import uuid
import zlib
import hashlib
from io import StringIO
from sqlalchemy.sql import func
import logging
import io
from .helical_calculations import calculate_helical_pile_results
from .shallow_calculations import calculate_shallow_footing_results
from .lateral_calculations import calculate_lateral_monopile_results
from .cantilever_calculations import calculate_cantilever_results
from .analytics import record_page_visit, store_analytics_data, get_or_create_user_id, get_page_visit_stats, get_analytics_data_stats, record_event, get_recent_users, get_user_details

# Set pandas options for full precision 
pd.set_option('display.precision', 15)  # Increase default precision
pd.set_option('display.float_format', lambda x: '%.15g' % x)  # Use full precision in string conversions

# INFO in production: DEBUG floods the Render logs with per-request noise.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)


# ---------------------------------------------------------------------------
# Shallow-foundations demo gating (private "coming soon" module).
# Access is granted to specific people only, via either:
#   SHALLOW_DEMO_EMAILS - comma-separated allowlist matched against the
#                         registered session email, or
#   SHALLOW_DEMO_CODE   - a shared code; visiting any shallow URL with
#                         ?code=<code> remembers access in the session.
# With neither set the demo stays fully private (default deny).
# ---------------------------------------------------------------------------
def _shallow_demo_emails():
    raw = os.environ.get('SHALLOW_DEMO_EMAILS', '') or current_app.config.get('SHALLOW_DEMO_EMAILS', '')
    return {e.strip().lower() for e in raw.split(',') if e.strip()}


_DEMO_COOKIE_NAME = 'uwa_demo_ok'
_DEMO_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year


def _maybe_grant_shallow_demo():
    """Remember demo access in the session if a valid ?code= is supplied.

    Also grants access to the private preview modules (lateral, cantilever)
    via the same code. Sets ``g.set_demo_cookie`` so the response handler
    persists a long-lived cookie backing the session flag — this keeps the
    gate open even after a server-side session is invalidated (e.g. by a
    Render redeploy).
    """
    from flask import g
    code = os.environ.get('SHALLOW_DEMO_CODE') or current_app.config.get('SHALLOW_DEMO_CODE')
    supplied = request.args.get('code') or request.form.get('demo_code')
    if code and supplied and compare_digest(str(supplied), str(code)):
        session['shallow_demo_ok'] = True
        session['private_demo_ok'] = True
        session.modified = True
        g.set_demo_cookie = True


def _shallow_demo_allowed():
    """Access gate for the shallow-foundations module.

    The module is public as of release (Barry signed off after the intro
    page was added). Helpers and the SHALLOW_DEMO_EMAILS/SHALLOW_DEMO_CODE
    env vars are kept in case re-gating is ever needed: just restore the
    earlier body that checked session['shallow_demo_ok'] / email allowlist.
    """
    return True


def _private_module_allowed():
    """Access gate for the lateral monopile and embedded cantilever wall
    modules.

    These are public as of release (Barry signed off on both modules). The
    earlier private-preview logic is kept here, commented, in case re-gating
    is ever needed: the in-session ``shallow_demo_ok`` / ``private_demo_ok``
    flag, the long-lived ``uwa_demo_ok`` cookie set by
    ``?code=<SHALLOW_DEMO_CODE>``, or the registered email being in the
    allowlist. To re-gate, restore the body below instead of ``return True``.
    """
    return True
    # --- private-preview gate (restore to re-gate) ---
    # if session.get('shallow_demo_ok'):
    #     return True
    # if session.get('private_demo_ok'):
    #     return True
    # if request.cookies.get(_DEMO_COOKIE_NAME) == 'ok':
    #     session['private_demo_ok'] = True
    #     session.modified = True
    #     return True
    # email = (session.get('user_email') or session.get('email') or '').strip().lower()
    # if email and email in _shallow_demo_emails():
    #     return True
    # return False


def _demo_access():
    """True only for users who arrived via the ?code= demo link.

    Used to gate experimental/preview features to demo users while the rest
    of the site stays public. Nothing is gated right now (the contractor-file
    importer and the saved-calculations history went public in June 2026);
    kept for the next preview feature. This is the original private-preview
    check (session flag set by the code link, backed by the long-lived
    ``uwa_demo_ok`` cookie, plus the email allowlist).
    """
    if session.get('shallow_demo_ok') or session.get('private_demo_ok'):
        return True
    if request.cookies.get(_DEMO_COOKIE_NAME) == 'ok':
        session['private_demo_ok'] = True
        session.modified = True
        return True
    email = (session.get('user_email') or session.get('email') or '').strip().lower()
    return bool(email and email in _shallow_demo_emails())


@bp.app_context_processor
def _inject_demo_flag():
    """Expose ``demo_access`` to every template so preview-only UI can be
    rendered for demo users. No template uses it right now; kept for the
    next preview feature."""
    try:
        return {'demo_access': _demo_access()}
    except Exception:
        return {'demo_access': False}


def _sanity_check_cpt(data_dict):
    """Return human-readable warnings about likely column-order or units
    errors in the parsed CPT data.

    All checks are advisory; the user can still proceed. Heuristics are
    chosen to fire only on data that is clearly inconsistent with real
    CPT readings.
    """
    warnings = []
    if not data_dict:
        return warnings
    n = len(data_dict)

    # 1. Sleeve friction can never exceed tip resistance physically. qc is
    #    uploaded in MPa and fs in kPa, so fs > qc*1000 means fs (kPa)
    #    exceeds qc converted to kPa. If many rows trip this, the qt and
    #    fs columns are almost certainly swapped.
    swap_rows = sum(1 for r in data_dict
                    if r['qc'] > 0 and r['fs'] > r['qc'] * 1000.0)
    if swap_rows > 0.3 * n:
        warnings.append(
            'Sleeve friction (fs) exceeds tip resistance (qt) in %d of %d '
            'rows. The qt and fs columns may be in the wrong order: qt '
            'should be column 2 in MPa, fs column 3 in kPa.' % (swap_rows, n))

    # 2. Unit weight should sit in roughly [10, 25] kN/m^3 for soils. If
    #    the median is well outside that, column 4 is probably misaligned
    #    or in the wrong units.
    gammas = sorted(r['gtot'] for r in data_dict if r['gtot'] > 0)
    if gammas:
        median_g = gammas[len(gammas) // 2]
        if median_g < 10 or median_g > 25:
            warnings.append(
                'Median unit weight is %.1f kN/m^3, outside the typical '
                '10-25 kN/m^3 range. Check that column 4 is unit weight '
                'in kN/m^3.' % median_g)

    # 3. qc should be in MPa. Values in the hundreds-of-MPa range are
    #    rarely physical (Bedrock CPT refusal aside) and usually mean qc
    #    was uploaded in kPa.
    qcs = [r['qc'] for r in data_dict if r['qc'] > 0]
    if qcs and sum(1 for q in qcs if q > 100) > 0.7 * len(qcs):
        warnings.append(
            'Most qt values exceed 100 MPa, which is unusually high. '
            'Check that qt is uploaded in MPa rather than kPa.')

    # 4. Contiguous spans where fs <= 0 while qc is non-zero: the sleeve
    #    reading has dropped out. Ic cannot be computed there (it collapses
    #    to 0), so the calculator treats those spans as clean sand and
    #    computes no shaft friction from them. Near-surface dropout is
    #    common (sleeve not yet in the ground) but matters just as much:
    #    it also feeds the CPT-derived unit weight. Report every span of
    #    at least 5 consecutive readings.
    spans = []
    start = None
    for i, r in enumerate(data_dict):
        dropout = r['fs'] <= 0 and r['qc'] > 0
        if dropout and start is None:
            start = i
        elif not dropout and start is not None:
            if i - start >= 5:
                spans.append((data_dict[start]['z'], data_dict[i - 1]['z']))
            start = None
    if start is not None and n - start >= 5:
        spans.append((data_dict[start]['z'], data_dict[-1]['z']))
    if spans:
        shown = ', '.join('%.2f-%.2f m' % s for s in spans[:3])
        more = '' if len(spans) <= 3 else ' (and %d more spans)' % (len(spans) - 3)
        warnings.append(
            'Sleeve friction (fs) reads zero over %s%s while qc is non-zero, '
            'so the sleeve reading has likely dropped out there. Soil '
            'classification (Ic) and shaft friction over these depths are '
            'unreliable: the calculator treats them as clean sand.'
            % (shown, more))

    # 4. Friction ratio sanity. Median Fr should sit between ~0.1% and
    #    ~10% for soils. Outside that strongly suggests a units / column
    #    mistake we haven't already flagged.
    frs = []
    for r in data_dict:
        if r['qc'] > 0 and r['fs'] >= 0:
            qt_kpa = r['qc'] * 1000.0
            if qt_kpa > 0:
                frs.append(100.0 * r['fs'] / qt_kpa)
    if frs:
        frs.sort()
        median_fr = frs[len(frs) // 2]
        if median_fr < 0.05 and not warnings:
            warnings.append(
                'Median friction ratio Fr is %.3f%%, which is unusually '
                'low. Check the fs (kPa) and qt (MPa) columns and units.'
                % median_fr)
        elif median_fr > 20 and not any('qt and fs' in w for w in warnings):
            warnings.append(
                'Median friction ratio Fr is %.1f%%, which is unusually '
                'high. Check the fs (kPa) and qt (MPa) columns and units.'
                % median_fr)

    return warnings


@bp.after_request
def _attach_demo_cookie(response):
    """Persist the demo-code grant as a long-lived cookie.

    Set when ``_maybe_grant_shallow_demo`` validates a fresh ?code= and sets
    ``g.set_demo_cookie``. The cookie is signed-by-name only (the value is
    a constant 'ok'); the security boundary is still that the user had to
    supply the correct SHALLOW_DEMO_CODE once to set it. This survives
    Flask-Session resets and Render redeploys, which was the failure mode
    behind the "click Continue and get bounced to home" report.
    """
    try:
        from flask import g
        if getattr(g, 'set_demo_cookie', False):
            response.set_cookie(
                _DEMO_COOKIE_NAME, 'ok',
                max_age=_DEMO_COOKIE_MAX_AGE,
                httponly=True,
                samesite='Lax',
                path='/',
            )
    except Exception:
        # Never fail a response because of cookie bookkeeping.
        pass
    return response


# ---------------------------------------------------------------------------
# Saved calculations ("My calculations") - demo-gated history, no login.
#
# Every completed calculation is snapshotted to the database, keyed to an
# anonymous browser id held in a long-lived cookie (same idea as the demo
# cookie: it must survive Render redeploys, which wipe both the server-side
# sessions and the temp files behind cpt_data_id / debug_id). Opening an
# entry rebuilds the wizard session exactly as the original run left it, so
# the results page, the CSV/PDF downloads and back-navigation to steps 2/3
# all work on a restored calculation.
# ---------------------------------------------------------------------------

_ANON_COOKIE_NAME = 'uwa_anon_id'
_ANON_COOKIE_MAX_AGE = 60 * 60 * 24 * 400  # ~13 months (Chrome caps cookies at 400 days)
_ANON_ID_RE = re.compile(r'^[0-9a-fA-F-]{8,64}$')
_HISTORY_MAX_PER_USER = 50

_HISTORY_TYPE_LABELS = {
    'driven': 'Driven pile',
    'bored': 'Bored pile',
    'helical': 'Helical (screw) pile',
    'shallow': 'Shallow footing',
    'lateral': 'Lateral monopile',
    'cantilever': 'Cantilever wall',
}


def _anon_id(create=True):
    """Stable anonymous id for this browser, used to key saved calculations.

    Lives in its own long-lived cookie so it survives server-side session
    resets. Seeded from the analytics session user_id on first use so saved
    calculations line up with the existing admin analytics.
    """
    from flask import g
    cookie = request.cookies.get(_ANON_COOKIE_NAME, '')
    if cookie and _ANON_ID_RE.match(cookie):
        return cookie
    pending = getattr(g, 'anon_id_new', None)
    if pending:
        return pending
    if not create:
        return None
    # Seed from the analytics session id when one exists so saved
    # calculations line up with the admin analytics; otherwise mint a fresh
    # id without touching the session (a session write here would create a
    # server-side session file for every anonymous page view).
    aid = session.get('user_id') or str(uuid.uuid4())
    if not _ANON_ID_RE.match(aid or ''):
        aid = str(uuid.uuid4())
    g.anon_id_new = aid  # picked up by _attach_anon_cookie on the way out
    return aid


@bp.after_request
def _attach_anon_cookie(response):
    """Persist a freshly minted anonymous id as a long-lived cookie."""
    try:
        from flask import g
        aid = getattr(g, 'anon_id_new', None)
        if aid:
            response.set_cookie(
                _ANON_COOKIE_NAME, aid,
                max_age=_ANON_COOKIE_MAX_AGE,
                httponly=True,
                samesite='Lax',
                path='/',
            )
    except Exception:
        pass
    return response


def _history_json_default(o):
    """JSON fallback for numpy scalars/arrays that may sit in session data."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def _history_pack(obj):
    return zlib.compress(json.dumps(obj, default=_history_json_default).encode('utf-8'))


def _history_unpack(blob):
    return json.loads(zlib.decompress(blob).decode('utf-8'))


# ---------------------------------------------------------------------------
# Wizard flow state - one CalcFlow row per upload-to-results run.
#
# The CPT rows and the pre_input_calc profile are stored once at upload
# (cpt_payload); parameters/results/debug details are stored as the wizard
# advances (state_payload). The flow id rides in the wizard URLs (added
# automatically by _add_flow_to_urls), so each browser tab carries its own
# flow, and a Render redeploy no longer strands users mid-wizard the way
# the old temp-file storage did.
# ---------------------------------------------------------------------------

_FLOW_ID_RE = re.compile(r'^[0-9a-f]{32}$')
_FLOWS_MAX_PER_USER = 20
_FLOWS_TTL_DAYS = 30  # stale-flow reaper; saved history is the long-term store
_UPLOAD_MAX_ROWS = 50000  # far beyond any real sounding; bounds CPU and storage per upload
_FLOW_ENDPOINTS = {
    'main.calculator_step',
    'main.download_debug_params',
    'main.download_results',
    'main.download_pdf_report',
    'main.download_intermediary_calcs',
}


def _request_flow_id():
    """Flow id carried by this request's own URL or form, if any."""
    if not has_request_context():
        return None
    for candidate in (request.args.get('flow', ''),
                      request.form.get('flow', '')):
        if candidate and _FLOW_ID_RE.match(candidate):
            return candidate
    return None


def _current_flow_id():
    """Flow id for this request: URL/form arg first, then session fallback
    (the fallback covers wizard URLs entered without the ?flow= arg)."""
    fid = _request_flow_id()
    if fid:
        return fid
    if not has_request_context():
        return None
    fid = session.get('flow_id', '')
    return fid if fid and _FLOW_ID_RE.match(fid) else None


@bp.url_defaults
def _add_flow_to_urls(endpoint, values):
    """Carry the flow id from this request's URL onto every wizard/download
    URL built while handling it, so templates never need to thread it
    through. Deliberately only propagates ids already present in the
    request (never the session fallback): otherwise links to OTHER modules
    rendered on description/landing pages would pick up the last active
    flow of whatever module the user ran previously."""
    if endpoint in _FLOW_ENDPOINTS and 'flow' not in values:
        fid = _request_flow_id()
        if fid:
            values['flow'] = fid


def _flow_create(calc_type, cpt_rows, processed, water_table, original_filename):
    """Persist a new wizard flow; returns its id."""
    fid = uuid.uuid4().hex
    anon = _anon_id()
    row = CalcFlow(
        id=fid,
        anon_id=anon,
        calc_type=calc_type,
        water_table=water_table,
        original_filename=original_filename,
        cpt_payload=_history_pack({'cpt_data': cpt_rows, 'processed': processed}),
    )
    db.session.add(row)
    db.session.commit()
    # Bound storage: drop this browser's oldest flows beyond the cap, and
    # reap stale flows globally (the per-anon cap alone is bypassable by a
    # client that discards cookies; saved history covers long-term reopen).
    try:
        overflow = (CalcFlow.query.filter_by(anon_id=anon)
                    .order_by(CalcFlow.updated_at.desc())
                    .offset(_FLOWS_MAX_PER_USER).all())
        for old in overflow:
            db.session.delete(old)
        cutoff = datetime.utcnow() - timedelta(days=_FLOWS_TTL_DAYS)
        stale = CalcFlow.query.filter(CalcFlow.updated_at < cutoff).delete(
            synchronize_session=False)
        if overflow or stale:
            db.session.commit()
    except Exception:
        db.session.rollback()
    session['flow_id'] = fid  # fallback for URLs entered without ?flow=
    return fid


def _flow_row(expected_type=None):
    """The CalcFlow row for this request, or None. Rows belong to the
    browser (anon cookie) that created them, and a flow created by one
    module is invisible to another module's wizard (``expected_type``):
    without that guard, editing the module segment of a wizard URL would
    run module B's calculation against module A's flow and corrupt it."""
    fid = _current_flow_id()
    if not fid:
        return None
    row = db.session.get(CalcFlow, fid)
    if row is None:
        return None
    if row.anon_id and row.anon_id != _anon_id():
        return None
    if expected_type is not None and row.calc_type != expected_type:
        return None
    return row


def _flow_has_cpt(row):
    """Cheap existence check: flows are always created with a CPT payload,
    so a resolvable row means the wizard has data (no blob load needed)."""
    return row is not None


def _flow_cpt(row):
    """CPT blob for a flow, shaped like the old load_cpt_data return:
    {'cpt_data': rows, 'water_table': wt, 'processed': profile} or None."""
    if row is None or not row.cpt_payload:
        return None
    try:
        blob = _history_unpack(row.cpt_payload)
    except Exception as e:
        logger.error("Could not unpack flow %s cpt payload: %s", row.id, e)
        return None
    if not blob or not blob.get('cpt_data'):
        return None
    blob['water_table'] = row.water_table
    return blob


def _flow_state(row):
    if row is None or not row.state_payload:
        return {}
    try:
        return _history_unpack(row.state_payload) or {}
    except Exception as e:
        logger.error("Could not unpack flow %s state payload: %s", row.id, e)
        return {}


def _flow_save_state(row, **updates):
    state = _flow_state(row)
    state.update(updates)
    row.state_payload = _history_pack(state)
    row.updated_at = datetime.utcnow()
    db.session.commit()
    return state


def _history_fingerprint(calc_type, pile_params, water_table, original_filename, cpt_rows):
    """Content fingerprint of a calculation, used to avoid duplicate history
    entries on results-page refreshes and when reopening a saved entry.
    Digests every CPT row: sampling only the first/last rows made a
    corrected re-issue of the same sounding (same name, count and depth
    range) silently overwrite the previous history entry."""
    rows_digest = hashlib.sha1(
        json.dumps(cpt_rows, default=_history_json_default).encode('utf-8')
    ).hexdigest()
    basis = [calc_type, pile_params, water_table, original_filename, rows_digest]
    return hashlib.sha1(
        json.dumps(basis, sort_keys=True, default=_history_json_default).encode('utf-8')
    ).hexdigest()


def _history_title(calc_type, pile_params, original_filename):
    p = pile_params or {}
    name = (p.get('site_name') or p.get('pile_name') or p.get('wall_name') or '').strip()
    if not name:
        name = (original_filename or '').strip()
    label = _HISTORY_TYPE_LABELS.get(calc_type, calc_type)
    return f'{label} - {name}' if name else label


def _history_params_brief(calc_type, p):
    """One-line summary of the inputs for the history list."""
    try:
        if calc_type == 'driven':
            tips = ', '.join(f'{float(d):g}' for d in p.get('pile_tip_depths', []))
            return f"D {float(p['pile_diameter']):g} m, tips {tips} m"
        if calc_type == 'bored':
            tips = ', '.join(f'{float(d):g}' for d in p.get('pile_tip_depths', []))
            return (f"shaft {float(p['shaft_diameter']):g} m, "
                    f"base {float(p['base_diameter']):g} m, tips {tips} m")
        if calc_type == 'helical':
            return (f"shaft {float(p['shaft_diameter']):g} m, helix "
                    f"{float(p['helix_diameter']):g} m at {float(p['helix_depth']):g} m")
        if calc_type == 'shallow':
            return (f"{float(p['footing_width']):g} x {float(p['footing_length']):g} m "
                    f"footing at {float(p['founding_depth']):g} m")
        if calc_type == 'lateral':
            return f"D {float(p['diameter']):g} m, L {float(p['embedded_length']):g} m"
        if calc_type == 'cantilever':
            return (f"length {float(p['wall_length']):g} m, excavation "
                    f"{float(p['excavation_depth']):g} m")
    except Exception:
        pass
    return ''


def _history_autosave(flow, state):
    """Snapshot the just-completed calculation into the saved history.

    Called on the step-4 results render. Returns True when a new entry was
    stored, False when skipped (already saved, or the flow's CPT payload is
    gone). Callers wrap this in try/except so a storage hiccup can never
    break the results page.
    """
    calc_type = flow.calc_type if flow is not None else None
    if calc_type not in _HISTORY_TYPE_LABELS or state.get('results') is None:
        return False
    cpt_blob = _flow_cpt(flow)
    cpt_rows = (cpt_blob or {}).get('cpt_data') or []
    if not cpt_rows:
        return False  # without the CPT rows the entry could not be reopened faithfully

    fingerprint = _history_fingerprint(
        calc_type, state.get('pile_params'), flow.water_table,
        flow.original_filename, cpt_rows)
    if flow.history_fp == fingerprint:
        return False

    payload = {
        'v': 1,
        'type': calc_type,
        'original_filename': flow.original_filename,
        'water_table': flow.water_table,
        'cpt_data': cpt_rows,
        'pile_params': state.get('pile_params'),
        'results': state['results'],
        'capacity_envelope': state.get('capacity_envelope'),
        'debug': state.get('debug'),
    }
    summary = {
        'filename': flow.original_filename,
        'water_table': flow.water_table,
        'n_points': len(cpt_rows),
        'depth_max': max(r['z'] for r in cpt_rows),
        'params_brief': _history_params_brief(calc_type, state.get('pile_params') or {}),
    }

    anon = _anon_id()
    row = SavedCalculation.query.filter_by(anon_id=anon, fingerprint=fingerprint).first()
    is_new = row is None
    if is_new:
        row = SavedCalculation(anon_id=anon, fingerprint=fingerprint)
        db.session.add(row)
    row.calc_type = calc_type
    row.title = _history_title(calc_type, state.get('pile_params'), flow.original_filename)
    row.summary_json = json.dumps(summary, default=_history_json_default)
    row.payload = _history_pack(payload)
    row.created_at = datetime.utcnow()
    flow.history_fp = fingerprint
    db.session.commit()

    # Keep only the newest entries per browser so the table stays bounded.
    overflow = (SavedCalculation.query.filter_by(anon_id=anon)
                .order_by(SavedCalculation.created_at.desc())
                .offset(_HISTORY_MAX_PER_USER).all())
    for old in overflow:
        db.session.delete(old)
    if overflow:
        db.session.commit()

    logger.info("History autosave: %s '%s' (new=%s)", calc_type, row.title, is_new)
    return is_new


def _render_step4(type, show_modal):
    """Render the results page for any module from the flow state."""
    flow = _flow_row(type)
    state = _flow_state(flow)
    results = state.get('results')
    if results is None:
        flash('No results available. Please complete the analysis first.')
        return redirect(url_for('main.calculator_step', type=type, step=3))

    # Snapshot the completed calculation into "My calculations" so it can
    # be reopened later. Never allowed to break the results page.
    history_saved_now = False
    try:
        history_saved_now = _history_autosave(flow, state)
    except Exception as hist_e:
        logger.warning(f"History autosave failed: {hist_e}")
        db.session.rollback()

    debug = state.get('debug')
    detailed_results = None
    if type == 'helical':
        if isinstance(debug, list) and debug:
            detailed_results = debug[0]
        else:
            logger.error("No debug details found or invalid format")
            flash('Error loading calculation details', 'error')
            return redirect(url_for('main.calculator_step', type=type, step=3))
    elif type in ('driven', 'bored'):
        tips = []
        if isinstance(debug, dict):
            tips = debug.get('tips', [])
        elif isinstance(debug, list):
            tips = debug
        if tips:
            detailed_results = tips[0]

    return render_template(
        f'{type}/steps.html',
        step=4,
        type=type,
        results=results,
        detailed_results=detailed_results,
        capacity_envelope=state.get('capacity_envelope'),
        pile_params=state.get('pile_params') or {},
        original_filename=flow.original_filename if flow is not None else None,
        history_saved_now=history_saved_now,
        show_modal=show_modal,
    )


@bp.route('/history')
def history():
    """List this browser's saved calculations."""
    anon = _anon_id(create=False)
    entries = []
    if anon:
        rows = (db.session.query(
                    SavedCalculation.id, SavedCalculation.calc_type,
                    SavedCalculation.title, SavedCalculation.summary_json,
                    SavedCalculation.created_at)
                .filter(SavedCalculation.anon_id == anon)
                .order_by(SavedCalculation.created_at.desc())
                .all())
        for r in rows:
            try:
                summary = json.loads(r.summary_json) if r.summary_json else {}
            except Exception:
                summary = {}
            entries.append({
                'id': r.id,
                'calc_type': r.calc_type,
                'type_label': _HISTORY_TYPE_LABELS.get(r.calc_type, r.calc_type),
                'title': r.title,
                'summary': summary,
                'created_at': r.created_at,
            })
    return render_template('history.html', entries=entries)


@bp.route('/history/<int:calc_id>/open')
def history_open(calc_id):
    """Rebuild the wizard session from a saved calculation and show step 4."""
    anon = _anon_id(create=False)
    row = db.session.get(SavedCalculation, calc_id) if anon else None
    if row is None or row.anon_id != anon:
        flash('That saved calculation was not found.')
        return redirect(url_for('main.history'))

    try:
        payload = _history_unpack(row.payload)
    except Exception as e:
        logger.error(f"Could not unpack saved calculation {calc_id}: {e}")
        flash('Could not load that saved calculation.')
        return redirect(url_for('main.history'))

    calc_type = payload.get('type')
    if calc_type not in _HISTORY_TYPE_LABELS or not payload.get('cpt_data') or 'results' not in payload:
        flash('Could not load that saved calculation.')
        return redirect(url_for('main.history'))

    # Rebuild the calculation as a fresh flow (the saved payloads predate
    # the stored-profile format, so the profile is recomputed once here).
    water_table = payload.get('water_table')
    if water_table is None:
        water_table = 0
    cpt_rows = payload['cpt_data']
    profile = pre_input_calc({'cpt_data': cpt_rows}, water_table)
    fid = _flow_create(calc_type, cpt_rows, profile, water_table,
                       payload.get('original_filename'))
    flow = db.session.get(CalcFlow, fid)

    state_updates = {'results': payload['results']}
    if payload.get('pile_params') is not None:
        state_updates['pile_params'] = payload['pile_params']
    if payload.get('capacity_envelope') is not None:
        state_updates['capacity_envelope'] = payload['capacity_envelope']
    if payload.get('debug'):
        state_updates['debug'] = payload['debug']
    _flow_save_state(flow, **state_updates)

    # Recompute the fingerprint from the restored state so the step-4 render
    # recognises this calculation as already saved and does not duplicate it.
    flow.history_fp = _history_fingerprint(
        calc_type, payload.get('pile_params'), water_table,
        payload.get('original_filename'), cpt_rows)
    db.session.commit()
    session['type'] = calc_type

    record_event('history', 'history_open', {'calc_type': calc_type, 'saved_id': row.id})
    return redirect(url_for('main.calculator_step', type=calc_type, step=4, flow=fid))


@bp.route('/history/<int:calc_id>/delete', methods=['POST'])
def history_delete(calc_id):
    """Delete one of this browser's saved calculations."""
    anon = _anon_id(create=False)
    row = db.session.get(SavedCalculation, calc_id) if anon else None
    if row is not None and row.anon_id == anon:
        db.session.delete(row)
        db.session.commit()
        record_event('history', 'history_delete', {'saved_id': calc_id})
        flash('Saved calculation deleted.')
    return redirect(url_for('main.history'))


@bp.route('/preview-cpt', methods=['POST'])
def preview_cpt():
    """AJAX preview for Step 1: run the flexible importer on the chosen file
    and return the first interpreted rows, the import note and any sanity
    warnings, so the user sees what the calculator will read before they
    submit. Read-only: touches no session or database state."""
    f = request.files.get('cpt_file')
    if f is None or not f.filename:
        return jsonify({'ok': False, 'error': 'No file received.'}), 400
    if not allowed_file(f.filename):
        return jsonify({'ok': False, 'error': 'Unsupported file type. Use a .csv, .txt or .ags file.'}), 400
    from .cpt_import import parse_cpt_file
    try:
        rows, note = parse_cpt_file(f.read(), f.filename)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})
    if not rows:
        return jsonify({'ok': False, 'error': 'No CPT data rows could be read from the file.'})
    return jsonify({
        'ok': True,
        'note': note or '',
        'warnings': _sanity_check_cpt(rows),
        'n_rows': len(rows),
        'depth_max': rows[-1]['z'],
        'rows': [{'z': round(r['z'], 3), 'qc': round(r['qc'], 3),
                  'fs': round(r['fs'], 2), 'gtot': round(r['gtot'], 2)}
                 for r in rows[:8]],
    })


# The calculator's original domain. Papers cite its URLs (e.g. the LSU LTRC
# FR_682 report and ISFOG2025-598 print pile-capacity-uwa.com links), so the
# re-registered domain 301s every request to the matching current page.
_LEGACY_HOSTS = {'pile-capacity-uwa.com', 'www.pile-capacity-uwa.com'}
_LEGACY_PATH_MAP = {
    '/calculator': '/driven/calculator/1',
    '/sand.pdf': '/driven/description',
    '/srd.pdf': '/driven/description',
    '/clay.pdf': '/driven/description',
}


@bp.before_app_request
def _redirect_legacy_domain():
    host = request.host.split(':')[0].lower()
    if host in _LEGACY_HOSTS:
        target = _LEGACY_PATH_MAP.get(request.path, '/')
        return redirect('https://uwa-geotech-cpt-calculator.com' + target, code=301)


@bp.route('/googlef2236ffa5d780ee8.html')
def google_site_verification():
    """Serve Google Search Console verification file."""
    return Response('google-site-verification: googlef2236ffa5d780ee8.html', mimetype='text/html')


@bp.route('/robots.txt')
def robots_txt():
    """Serve robots.txt for search engine crawlers."""
    content = """User-agent: *
Allow: /
Disallow: /admin
Disallow: /admin/export
Disallow: /admin/send_weekly_report
Disallow: /download_results
Disallow: /download_debug_params
Disallow: /download_intermediary_calcs
Disallow: /download_helical_calculations
Disallow: /register
Disallow: /track_ad_click
Disallow: /sample/
Disallow: /preview-cpt

Sitemap: https://uwa-geotech-cpt-calculator.com/sitemap.xml
"""
    return Response(content.strip(), mimetype='text/plain')


@bp.route('/sitemap.xml')
def sitemap_xml():
    """Serve XML sitemap for search engine indexing."""
    pages = [
        {'loc': '/', 'priority': '1.0', 'changefreq': 'monthly', 'lastmod': '2026-08-10'},
        {'loc': '/driven/description', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-04-10'},
        {'loc': '/bored/description', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-04-10'},
        {'loc': '/helical/description', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-08-10'},
        {'loc': '/shallow/description', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/lateral/description', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/cantilever/description', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/driven/calculator/1', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/bored/calculator/1', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/helical/calculator/1', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/shallow/calculator/1', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/lateral/calculator/1', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/cantilever/calculator/1', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-06-10'},
        {'loc': '/screw-pile-calculator', 'priority': '0.9', 'changefreq': 'monthly', 'lastmod': '2026-08-10'},
        {'loc': '/guides/pile-length-from-cpt', 'priority': '0.8', 'changefreq': 'monthly', 'lastmod': '2026-08-10'},
        {'loc': '/suggestions', 'priority': '0.5', 'changefreq': 'yearly', 'lastmod': '2026-04-10'},
    ]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for page in pages:
        xml += '  <url>\n'
        xml += f'    <loc>https://uwa-geotech-cpt-calculator.com{page["loc"]}</loc>\n'
        xml += f'    <lastmod>{page["lastmod"]}</lastmod>\n'
        xml += f'    <changefreq>{page["changefreq"]}</changefreq>\n'
        xml += f'    <priority>{page["priority"]}</priority>\n'
        xml += '  </url>\n'
    xml += '</urlset>'
    return Response(xml, mimetype='application/xml')


@bp.route('/screw-pile-calculator')
def screw_pile_landing():
    """SEO landing page for the Australian term; the tool itself is the helical module."""
    return render_template('screw_pile.html')


@bp.route('/guides/pile-length-from-cpt')
def guide_pile_length():
    """Guide targeting pile depth / pile length queries."""
    return render_template('guides/pile_length.html')


@bp.route('/sample/<type>')
def use_sample_data(type):
    """Load sample CPT data for demo purposes."""
    if type not in ['driven', 'bored', 'helical', 'shallow', 'lateral', 'cantilever']:
        return redirect(url_for('main.index'))
    if type == 'shallow':
        _maybe_grant_shallow_demo()
        if not _shallow_demo_allowed():
            return redirect(url_for('main.index'))
    if type in ('lateral', 'cantilever'):
        _maybe_grant_shallow_demo()
        if not _private_module_allowed():
            return redirect(url_for('main.index'))

    # Read sample data file
    sample_path = os.path.join(current_app.static_folder, 'data', 'sample_cpt.csv')
    try:
        data_dict = []
        raw_rows = []
        with open(sample_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = line.split(',')
                if len(values) >= 4:
                    row = {
                        'z': float(values[0]),
                        'qc': float(values[1]),
                        'fs': float(values[2]),
                        'gtot': float(values[3])
                    }
                    data_dict.append(row)
                    raw_rows.append(row)

        if not data_dict:
            flash('Error loading sample data')
            return redirect(url_for('main.calculator_step', type=type, step=1))

        processed_data = process_cpt_data(data_dict)
        water_table = 2.0  # Default water table for sample

        cpt_rows = processed_data['cpt_data']
        profile = pre_input_calc({'cpt_data': cpt_rows}, water_table)
        flow_id = _flow_create(type, cpt_rows, profile, water_table, 'sample_data')
        session['type'] = type

        store_analytics_data('sample_data', 'type', type)

        # Redirect to step 1 with sample flag so user can review the data
        return redirect(url_for('main.calculator_step', type=type, step=1, sample=1, flow=flow_id))
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        flash(f'Error loading sample data: {str(e)}')
        return redirect(url_for('main.calculator_step', type=type, step=1))


# Add analytics middleware to track all page visits
@bp.before_request
def track_page_visit():
    # Skip tracking for static files and favicon
    if request.path.startswith('/static') or request.path == '/favicon.ico':
        return
    
    # Record the page visit in the database
    record_page_visit()

ALLOWED_EXTENSIONS = {'csv', 'txt', 'ags'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_data_dataframe(processed_cpt, calc_dict, cpt_profile_dict=None):
    """Create a DataFrame with CPT data and bored pile calculations.

    Tip-INDEPENDENT per-row fields (qb01_adop, delta_z) come from ``cpt_profile_dict``
    (keyed by depth). Tip-DEPENDENT fields come from ``calc_dict``.
    """
    data = []
    cpt_profile_dict = cpt_profile_dict or {}

    # Build a set of depths that have calculations (rounded to avoid float issues)
    calc_depths_set = set(round(float(d), 6) for d in (calc_dict.keys() if calc_dict else []))

    def _lookup(d_map, depth):
        if not d_map:
            return None
        v = d_map.get(depth)
        if v is None:
            v = next((vv for kk, vv in d_map.items() if round(float(kk), 6) == round(float(depth), 6)), None)
        return v

    for i, depth in enumerate(processed_cpt['depth']):
        # If we have calculation depths, skip rows beyond the tip depth
        if calc_depths_set and round(float(depth), 6) not in calc_depths_set:
            continue

        row = {
            'Depth (m)': depth,
            'qt (MPa)': processed_cpt['qt'][i],
            'qc (MPa)': processed_cpt['qc'][i],
            'fs (kPa)': processed_cpt['fs'][i],
            'Fr (%)': processed_cpt['fr_percent'][i],
            'Ic': processed_cpt['lc'][i],
            'sig_v0_prime (kPa)': processed_cpt['sig_v0_prime'][i],
            'u0 (kPa)': processed_cpt['u0'][i],
            'sig_v0 (kPa)': processed_cpt['sig_v0'][i],
        }

        profile = _lookup(cpt_profile_dict, depth)
        if profile is not None:
            row['qb0.1 (MPa)'] = profile.get('qb01_adop', 'N/A')
            row['Delta z (m)'] = profile.get('delta_z', 'N/A')

        calcs = _lookup(calc_dict, depth)
        if calcs is not None:
            row.update({
                'Casing Coefficient': calcs.get('coe_casing', 'N/A'),
                'tf tension (kPa)': calcs.get('tf_tension', 'N/A'),
                'tf compression (kPa)': calcs.get('tf_compression', 'N/A'),
                'Shaft Tension Segment (kN)': calcs.get('qs_tension_segment', 'N/A'),
                'Shaft Compression Segment (kN)': calcs.get('qs_compression_segment', 'N/A'),
                'Cumulative Shaft Tension (kN)': calcs.get('qs_tension_cumulative', 'N/A'),
                'Cumulative Shaft Compression (kN)': calcs.get('qs_compression_cumulative', 'N/A'),
            })
        data.append(row)

    return pd.DataFrame(data)

def create_driven_data_dataframe(processed_cpt, calc_dict, cpt_profile_dict=None):
    """Create a DataFrame with CPT data and driven pile calculations.

    Tip-INDEPENDENT per-row fields (q1, q10, qp_*, qb1_*, qb_final, delta_z) come
    from ``cpt_profile_dict`` (keyed by depth). Tip-DEPENDENT fields come from
    ``calc_dict``.
    """
    data = []
    cpt_profile_dict = cpt_profile_dict or {}

    calc_depths_set = set(round(float(d), 6) for d in (calc_dict.keys() if calc_dict else []))

    def _lookup(d_map, depth):
        if not d_map:
            return None
        v = d_map.get(depth)
        if v is None:
            v = next((vv for kk, vv in d_map.items() if round(float(kk), 6) == round(float(depth), 6)), None)
        return v

    for i, depth in enumerate(processed_cpt['depth']):
        if calc_depths_set and round(float(depth), 6) not in calc_depths_set:
            continue

        row = {
            'Depth (m)': depth,
            'qt (MPa)': processed_cpt['qt'][i],
            'qc (MPa)': processed_cpt['qc'][i],
            'qtc (MPa)': processed_cpt['qtc'][i],
            'fs (kPa)': processed_cpt['fs'][i],
            'Fr (%)': processed_cpt['fr_percent'][i],
            'qtn': processed_cpt['qtn'][i],
            'n': processed_cpt['n'][i],
            'Ic': processed_cpt['lc'][i],
            'gtot (kN/m³)': processed_cpt['gtot'][i],
            'sig_v0 (kPa)': processed_cpt['sig_v0'][i],
            'sig_v0_prime (kPa)': processed_cpt['sig_v0_prime'][i],
            'u0 (kPa)': processed_cpt['u0_kpa'][i],
            'iz1': processed_cpt['iz1'][i],
        }

        profile = _lookup(cpt_profile_dict, depth)
        if profile is not None:
            row.update({
                'q1 (MPa)': profile.get('q1', 'N/A'),
                'q10 (MPa)': profile.get('q10', 'N/A'),
                'qp_sand (MPa)': profile.get('qp_sand', 'N/A'),
                'qp_clay (MPa)': profile.get('qp_clay', 'N/A'),
                'qp_adopted (MPa)': profile.get('qp_adopted', 'N/A'),
                'qb1_sand (MPa)': profile.get('qb1_sand', 'N/A'),
                'qb1_clay (MPa)': profile.get('qb1_clay', 'N/A'),
                'qb1_adopted (MPa)': profile.get('qb1_adopted', 'N/A'),
                'Base Resistance (kN)': profile.get('qb_final', 'N/A'),
                'Delta z (m)': profile.get('delta_z', 'N/A'),
            })

        calcs = _lookup(calc_dict, depth)
        if calcs is not None:
            row.update({
                'h (m)': calcs.get('h', 'N/A'),
                'Casing Coefficient': calcs.get('coe_casing', 'N/A'),
                'delta_ord (degrees)': calcs.get('delta_ord', 'N/A'),
                'orc_val': calcs.get('orc_val', 'N/A'),
                'tf_sand (kPa)': calcs.get('tf_sand', 'N/A'),
                'tf_clay (kPa)': calcs.get('tf_clay', 'N/A'),
                'tf_adop_tension (kPa)': calcs.get('tf_adop_tension', 'N/A'),
                'tf_adop_compression (kPa)': calcs.get('tf_adop_compression', 'N/A'),
                'Shaft Tension Segment (kN)': calcs.get('qs_tension_segment', 'N/A'),
                'Shaft Compression Segment (kN)': calcs.get('qs_compression_segment', 'N/A'),
                'Cumulative Shaft Tension (kN)': calcs.get('qs_tension_cumulative', 'N/A'),
                'Cumulative Shaft Compression (kN)': calcs.get('qs_compression_cumulative', 'N/A'),
            })
        data.append(row)

    return pd.DataFrame(data)

@bp.route('/index')
def index_alias():
    """Legacy alias. 301 so search engines consolidate signals on the root URL."""
    return redirect(url_for('main.index'), code=301)


@bp.route('/')
def index():
    logger.debug("Session keys: %s, registered=%s", list(session.keys()), session.get('registered'))

    # Don't show registration modal on index - let users explore first
    show_modal = False

    # Still check for registration status for cookie handling
    if 'registered' not in session or not session['registered']:
        if request.cookies.get('user_registered') == 'true':
            session['registered'] = True
            session.modified = True

    session.permanent = True
    session.modified = True

    # Reveal the shallow-foundations demo card only to authorised people.
    # The same ?code= URL also opens the gate for the lateral monopile and
    # embedded cantilever wall preview modules.
    _maybe_grant_shallow_demo()
    response = make_response(render_template(
        'index.html', show_modal=show_modal,
        shallow_demo=_shallow_demo_allowed(),
        private_demo=_private_module_allowed()))
    if 'registered' in session and session['registered']:
        response.set_cookie(
            'user_registered',
            'true',
            max_age=31536000,
            httponly=True,
            samesite='Lax',
            path='/'
        )
    return response

@bp.route('/track_ad_click', methods=['POST'])
def track_ad_click():
    """Track when users click the 3D PIV advertisement"""
    try:
        # Store the click event in analytics
        store_analytics_data('ad_click', '3d_piv_research', 'clicked')
        
        # Return success response
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error tracking ad click: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/<type>/calculator/<int:step>', methods=['GET', 'POST'])
def calculator_step(type, step):
    # Check if we're switching pile types
    if 'type' in session and session['type'] != type:
        # Clear session data except for user info
        session_data = {}
        for key in ['user_id', 'email', 'name', 'institution', 'registered', 'user_email', 'affiliation', 'shallow_demo_ok', 'private_demo_ok']:
            if key in session:
                session_data[key] = session[key]
        session.clear()
        session.update(session_data)
    
    # Store the pile type in session and database
    session['type'] = type
    store_analytics_data('pile_selection', 'type', type)

    # Track step visit with user context
    record_event('step_visit', f'{type}_step_{step}', {
        'pile_type': type,
        'step': step,
        'method': request.method,
        'email': session.get('user_email'),
    })

    if type not in ['driven', 'bored', 'helical', 'shallow', 'lateral', 'cantilever']:
        return redirect(url_for('main.index'))

    # Honour the ?code= demo link on any calculator URL so demo-only features
    # (e.g. the flexible CPT importer) are available wherever the user lands.
    _maybe_grant_shallow_demo()

    # Shallow foundations is a private demo: gate to authorised people only.
    if type == 'shallow':
        _maybe_grant_shallow_demo()
        if not _shallow_demo_allowed():
            return redirect(url_for('main.index'))
    if type in ('lateral', 'cantilever'):
        _maybe_grant_shallow_demo()
        if not _private_module_allowed():
            return redirect(url_for('main.index'))

    # Show registration modal if user hasn't registered yet. "Skip for now"
    # is remembered in a 30-day cookie, so the modal stops re-covering
    # later wizard steps and the results page after one dismissal.
    show_modal = False
    if 'registered' not in session or not session['registered']:
        if request.cookies.get('user_registered') == 'true':
            session['registered'] = True
            session.modified = True
        elif request.cookies.get('uwa_reg_skip') != '1':
            show_modal = True

    # Handle helical pile processing specifically
    if type == 'helical' and step == 3 and request.method == 'POST':
        try:
            flow = _flow_row(type)
            cpt_blob = _flow_cpt(flow)
            if not cpt_blob:
                flash('CPT data not found. Please upload data again.', 'error')
                return redirect(url_for('main.calculator_step', type='helical', step=1))

            # Get parameters from form
            pile_params = {
                'site_name': request.form.get('site_name', ''),
                'shaft_diameter': float(request.form.get('shaft_diameter', 0)),
                'helix_diameter': float(request.form.get('helix_diameter', 0)),
                'helix_depth': float(request.form.get('helix_depth', 0)),
                'borehole_depth': 0.0,  # Always set to zero as requested
                'water_table': float(flow.water_table or 0)
            }
            
            # Validate helical pile parameters
            errors = []
            if pile_params['shaft_diameter'] <= 0:
                errors.append('Shaft diameter must be greater than 0')
            if pile_params['helix_diameter'] <= 0:
                errors.append('Helix diameter must be greater than 0')
            if pile_params['helix_depth'] <= 0:
                errors.append('Helix depth must be greater than 0')
            if pile_params['shaft_diameter'] >= pile_params['helix_diameter']:
                errors.append('Shaft diameter must be smaller than helix diameter')

            if errors:
                for error in errors:
                    flash(error)
                return redirect(url_for('main.calculator_step', type='helical', step=3))

            # Store parameters in database
            store_analytics_data('pile_params', data_dict=pile_params)

            # Helix depth must sit within the CPT profile, same rule as
            # the pile modules (Barry, 10 June 2026: no capacity
            # calculation below the deepest CPT reading)
            max_depth = max(row['z'] for row in cpt_blob['cpt_data'])
            if pile_params['helix_depth'] > max_depth:
                flash(f'Helix depth {pile_params["helix_depth"]}m exceeds the deepest CPT reading ({max_depth:.2f}m). Reduce the helix depth or upload deeper CPT data.')
                return redirect(url_for('main.calculator_step', type='helical', step=3))

            # Profile computed once at upload; recompute only for legacy flows
            processed_cpt = cpt_blob.get('processed') or pre_input_calc(cpt_blob, float(flow.water_table or 0))

            # Calculate results
            try:
                results = calculate_helical_pile_results(processed_cpt, pile_params)

                # Store results in database
                store_analytics_data('calculation_results', 'summary', results['summary'])

                # Create detailed results with all necessary data
                detailed_results = {
                    'calculations': results['detailed'],
                    'helix_calculations': {
                        'perimeter': results['detailed'].get('perimeter'),
                        'helix_area': results['detailed'].get('helix_area'),
                        'q1_helix': results['detailed'].get('q1_helix'),
                        'q10_helix': results['detailed'].get('q10_helix'),
                        'qhelix_tension': results['detailed'].get('qhelix_tension'),
                        'qhelix_compression': results['detailed'].get('qhelix_compression')
                    },
                    'input_parameters': pile_params
                }

                # Persist everything on the flow row (debug wrapped in a list,
                # matching the shape the downloads expect)
                _flow_save_state(flow,
                                 pile_params=pile_params,
                                 results=results['summary'],
                                 debug=[detailed_results])

                logger.info("Helical pile calculations completed for %s (flow=%s)", pile_params.get('site_name'), flow.id)

                return redirect(url_for('main.calculator_step', type=type, step=4))
            except Exception as e:
                logger.error(f"Error in helical pile calculations: {str(e)}")
                flash(f'Error in calculation: {str(e)}', 'error')
                return redirect(url_for('main.calculator_step', type='helical', step=3))

        except Exception as e:
            logger.error(f"Error processing helical pile parameters: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('main.calculator_step', type='helical', step=3))
    
    # Handle step 4 - Results display
    elif step == 4:
        return _render_step4(type, show_modal)

    # Continue with the rest of the route handler
    if request.method == 'POST':
        if step == 1:  # Handle file upload
            if 'cpt_file' not in request.files:
                flash('No file selected')
                return redirect(request.url)
            
            # Get water table value first
            water_table = request.form.get('water_table')
            if not water_table:
                flash('Water table depth is required')
                return redirect(request.url)
            
            try:
                water_table = float(water_table)
            except ValueError:
                flash('Water table depth must be a number')
                return redirect(request.url)
            
            file = request.files['cpt_file']
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                try:
                    logger.debug("Starting file processing")
                    record_event('upload', f'{type}_file_upload', {
                        'filename': file.filename,
                        'water_table': water_table,
                        'pile_type': type,
                    })
                    # Store original filename in session (without extension)
                    original_filename = os.path.splitext(secure_filename(file.filename))[0]
                    session['original_filename'] = original_filename
                    
                    # Initialize data structures
                    data_dict = []
                    delimiter = None  # set by the legacy parser; kept None for the demo importer

                    # Two-option Step 1:
                    #   'contractor' -> flexible importer (AGS4 / vendor
                    #                   exports; unit weight derived from the
                    #                   CPT), or
                    #   'user'       -> clean four-column file they supply.
                    cpt_source = request.form.get('cpt_source', 'contractor')
                    if cpt_source != 'user':
                        # Contractor file: flexible importer that reads AGS4 and
                        # vendor/contractor CPT exports (and clean CSVs too), and
                        # derives the unit weight column from the CPT when absent.
                        from .cpt_import import parse_cpt_file
                        try:
                            data_dict, _import_note = parse_cpt_file(file.read(), file.filename)
                        except Exception as _imp_err:
                            logger.error(f"CPT import failed: {_imp_err}")
                            flash(f'Could not read that CPT file: {_imp_err}')
                            return redirect(request.url)
                        if _import_note:
                            flash(_import_note)
                        record_event('upload', f'{type}_cpt_import',
                                     {'note': (_import_note or '')[:200], 'filename': file.filename})
                    else:
                        # User file: clean four-column file
                        # (depth, qt, fs, unit weight), auto-detected delimiter.
                        delimiter = None
                        for line in file:
                            try:
                                decoded_line = line.decode('utf-8').strip()
                                if not decoded_line:  # Skip empty lines
                                    continue

                                # Try to determine the delimiter from first line
                                if delimiter is None:
                                    if '\t' in decoded_line:
                                        delimiter = '\t'
                                    elif ',' in decoded_line:
                                        delimiter = ','
                                    else:
                                        delimiter = ' '

                                # Split and process each line directly
                                values = decoded_line.split(delimiter)
                                if len(values) >= 4:  # Ensure we have all required columns
                                    try:
                                        data_dict.append({
                                            'z': float(values[0]),
                                            'qc': float(values[1]),
                                            'fs': float(values[2]),
                                            'gtot': float(values[3])
                                        })
                                    except (ValueError, IndexError):
                                        continue
                            except UnicodeDecodeError:
                                continue

                    if not data_dict:
                        flash('No valid data found in file')
                        return redirect(request.url)

                    # Flag likely column-order / units mistakes before we
                    # commit the data. Non-blocking so the user can override.
                    for _w in _sanity_check_cpt(data_dict):
                        flash(_w)
                        record_event('upload', f'{type}_cpt_sanity_warning',
                                     {'warning': _w[:200]})

                    processed_data = process_cpt_data(data_dict)
                    if not processed_data or not processed_data.get('cpt_data'):
                        flash('No valid data found in file')
                        return redirect(request.url)
                    logger.debug("process_cpt_data completed")

                    # Surface any corrections the cleaning pass applied
                    # (re-sorted depths, dropped duplicate/negative rows) so
                    # the user knows their file was adjusted.
                    for _note in processed_data.get('upload_notes', []):
                        flash(_note)
                        record_event('upload', f'{type}_cpt_upload_note', {'note': _note[:200]})

                    if len(processed_data['cpt_data']) < 5:
                        flash('Only %d usable CPT rows could be read from the file. '
                              'At least 5 readings are needed for a meaningful profile; '
                              'check the file format and units.' % len(processed_data['cpt_data']))
                        return redirect(request.url)

                    if len(processed_data['cpt_data']) > _UPLOAD_MAX_ROWS:
                        flash('The file contains %d readings, more than the %d this tool '
                              'supports in one upload. Reduce the file (e.g. one sounding '
                              'per upload) and try again.'
                              % (len(processed_data['cpt_data']), _UPLOAD_MAX_ROWS))
                        return redirect(request.url)


                    # Check if we need interpolation for better accuracy
                    cpt_data = processed_data['cpt_data']
                    depths = [row['z'] for row in cpt_data]
                    
                    if len(depths) > 1:
                        min_spacing = min(abs(depths[i+1] - depths[i]) for i in range(len(depths)-1))
                        logger.debug(f"Minimum spacing between data points: {min_spacing}m")
                        logger.debug(f"Total data points: {len(cpt_data)}")
                        logger.debug(f"Depth range: {min(depths):.2f}m to {max(depths):.2f}m")
                        
                        # Use a more conservative threshold for interpolation to prevent timeouts
                        interpolation_threshold = 0.5  # Only interpolate if spacing > 0.5m
                        if min_spacing > interpolation_threshold:
                            # Convert to interpolation format and interpolate
                            interpolation_data = [[row['z'], row['fs'], row['qc'], row['gtot']] for row in cpt_data]
                            
                            try:
                                interpolated_data, warning_message = process_uploaded_cpt_data(
                                    '\n'.join([f"{row[0]} {row[1]} {row[2]} {row[3]}" for row in interpolation_data])
                                )
                                
                                # Convert back to cpt_data format
                                interpolated_cpt_data = []
                                for row in interpolated_data:
                                    interpolated_cpt_data.append({
                                        'z': row[0],
                                        'qc': row[2], 
                                        'fs': row[1],
                                        'gtot': row[3]
                                    })
                                
                                processed_data['cpt_data'] = interpolated_cpt_data
                                flash(warning_message)
                                logger.debug(f"Data interpolated from {len(cpt_data)} to {len(interpolated_cpt_data)} points")
                                
                            except Exception as interp_error:
                                logger.error(f"Interpolation failed: {str(interp_error)}")
                                flash(f"Warning: Interpolation failed ({str(interp_error)}). Using original data with coarse spacing.")
                                # Continue with original data if interpolation fails
                    
                    # Compute the profile once, at upload; every later step
                    # and download reuses it from the persisted flow row.
                    cpt_rows = processed_data['cpt_data']
                    profile = pre_input_calc({'cpt_data': cpt_rows}, water_table)
                    if not profile:
                        flash('Error processing CPT data. Please check your input data.')
                        return redirect(request.url)
                    if profile.get('fallback_count'):
                        flash('%d of %d readings could not be classified from qt/fs and were '
                              'treated as neutral soil (Ic = 2.5). Check those readings if '
                              'the soil profile around them matters.'
                              % (profile['fallback_count'], len(cpt_rows)))

                    flow_id = _flow_create(type, cpt_rows, profile, water_table, original_filename)

                    logger.debug(f"File name: {file.filename}")
                    logger.debug(f"Content type: {file.content_type}")
                    logger.debug(f"Detected delimiter: {delimiter}")

                    # Track upload details including depth range
                    depth_min = min(r['z'] for r in cpt_rows) if cpt_rows else 0
                    depth_max = max(r['z'] for r in cpt_rows) if cpt_rows else 0
                    record_event('upload_success', f'{type}_upload_complete', {
                        'pile_type': type,
                        'filename': file.filename,
                        'data_points': len(cpt_rows),
                        'depth_range': f'{depth_min:.1f} - {depth_max:.1f} m',
                        'water_table': water_table,
                        'delimiter': delimiter,
                    })

                    return redirect(url_for('main.calculator_step', type=type, step=2, flow=flow_id))
                    
                except pd.errors.EmptyDataError:
                    flash('The uploaded file is empty')
                    return redirect(request.url)
                except Exception as e:
                    flash(f'Error processing file: {str(e)}')
                    return redirect(request.url)
            else:
                flash('Invalid file type. Please upload a CSV or TXT file.')
                return redirect(request.url)
        
        elif step == 2:  # Handle CPT data acceptance
            # The data was validated and processed at upload; just confirm
            # the flow still exists. No throwaway recomputation here.
            if not _flow_has_cpt(_flow_row(type)):
                flash('No CPT data available. Please upload data first.')
                return redirect(url_for('main.calculator_step', type=type, step=1))
            return redirect(url_for('main.calculator_step', type=type, step=3))
        
        elif step == 3:
            if type == 'bored':
                flow = _flow_row(type)
                cpt_blob = _flow_cpt(flow)
                if not cpt_blob:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))

                # Add validation for bored pile parameters
                required_fields = ['shaft_diameter', 'base_diameter', 'cased_depth', 'pile_tip_depths']
                for field in required_fields:
                    if field not in request.form or not request.form[field]:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Debug logging
                logger.info(f"Bored pile form submitted with data: {request.form}")
                logger.info(f"Flow water table: {flow.water_table}")
                
                # Parse pile tip depths from the comma-separated string
                tip_depths_str = request.form.get('pile_tip_depths', '')
                try:
                    pile_tip_depths = [float(d.strip()) for d in tip_depths_str.split(',')]
                except ValueError:
                    flash('Invalid pile tip depths format. Please enter numbers separated by commas.')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Validate bored pile parameters
                errors = []
                if float(request.form.get('shaft_diameter', 0)) <= 0:
                    errors.append('Shaft diameter must be greater than 0')
                if float(request.form.get('base_diameter', 0)) <= 0:
                    errors.append('Base diameter must be greater than 0')
                if float(request.form.get('cased_depth', 0)) < 0:
                    errors.append('Cased depth cannot be negative')

                # Validate pile tip depths against the CPT profile, same rule
                # as the driven module (Barry, 10 June 2026: no capacity
                # calculation when the tip is below the deepest CPT reading)
                if not pile_tip_depths:
                    errors.append('At least one pile tip depth is required')
                else:
                    max_depth = max(row['z'] for row in cpt_blob['cpt_data'])
                    for depth in pile_tip_depths:
                        if depth <= 0:
                            errors.append(f'Pile tip depth {depth}m must be greater than 0')
                        if depth > max_depth:
                            errors.append(f'Pile tip depth {depth}m exceeds the deepest CPT reading ({max_depth:.2f}m). Reduce the tip depth or upload deeper CPT data.')

                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                pile_params = {
                    'shaft_diameter': float(request.form['shaft_diameter']),
                    'base_diameter': float(request.form['base_diameter']),
                    'cased_depth': float(request.form['cased_depth']),
                    'water_table': float(flow.water_table or 0),
                    'site_name': request.form.get('file_name', ''),
                    'pile_tip_depths': pile_tip_depths
                }

                record_event('calculation', 'bored_params', {
                    'shaft_diameter': pile_params['shaft_diameter'],
                    'base_diameter': pile_params['base_diameter'],
                    'cased_depth': pile_params['cased_depth'],
                    'tip_depths': pile_tip_depths,
                })

                # Profile computed once at upload; recompute only for legacy flows
                processed_cpt = cpt_blob.get('processed') or pre_input_calc(cpt_blob, float(flow.water_table or 0))

                try:
                    results = calculate_bored_pile_results(processed_cpt, pile_params)

                    logger.info("Bored pile calculation complete (flow=%s)", flow.id)

                    # Compute continuous capacity envelope for the graph
                    try:
                        envelope = compute_capacity_envelope_bored(processed_cpt, pile_params)
                    except Exception as env_e:
                        logger.warning(f"Could not compute capacity envelope: {env_e}")
                        envelope = None

                    _flow_save_state(flow,
                                     pile_params=pile_params,
                                     results=results['summary'],
                                     capacity_envelope=envelope,
                                     debug=results['detailed'])

                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in bored pile calculation: {str(e)}")
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            elif type == 'shallow':
                flow = _flow_row(type)
                cpt_blob = _flow_cpt(flow)
                if not cpt_blob:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))

                # Shallow foundations: footing geometry + analysis options.
                def _opt(name, default):
                    val = request.form.get(name, '')
                    try:
                        return float(val) if val not in (None, '') else default
                    except (TypeError, ValueError):
                        return default

                footing_params = {
                    'water_table': float(flow.water_table or 0),
                    'footing_width': _opt('footing_width', None),
                    'footing_length': _opt('footing_length', None),
                    'founding_depth': _opt('founding_depth', None),
                    'excavation_depth': _opt('excavation_depth', 0),
                    'design_life_years': _opt('design_life_years', 50),
                    'initial_ocr': _opt('initial_ocr', 1),
                    'k0': _opt('k0', 0.5),
                    'ageing_factor': _opt('ageing_factor', 0.66),
                    'creep': _opt('creep', 0.02),
                    'nkt': _opt('nkt', 15),
                    'unit_weight': _opt('unit_weight', None),
                    'site_name': request.form.get('file_name', ''),
                    'force_soil_model': request.form.get('force_soil_model') or None,
                }

                errors = []
                if not footing_params['footing_width'] or footing_params['footing_width'] <= 0:
                    errors.append('Footing width must be greater than 0')
                if not footing_params['footing_length'] or footing_params['footing_length'] <= 0:
                    errors.append('Footing length must be greater than 0')
                if footing_params['founding_depth'] is None or footing_params['founding_depth'] < 0:
                    errors.append('Founding depth below excavation cannot be negative')
                if footing_params['unit_weight'] is not None and not (5 <= footing_params['unit_weight'] <= 30):
                    errors.append('Unit weight must be between 5 and 30 kN/m3 (leave blank to derive it from the CPT)')
                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                record_event('calculation', 'shallow_params', {
                    k: footing_params[k] for k in
                    ('footing_width', 'footing_length', 'founding_depth',
                     'excavation_depth', 'design_life_years')
                })

                # Profile computed once at upload; recompute only for legacy flows
                processed_cpt = cpt_blob.get('processed') or pre_input_calc(cpt_blob, float(flow.water_table or 0))
                try:
                    results = calculate_shallow_footing_results(processed_cpt, footing_params)
                    _flow_save_state(flow, pile_params=footing_params, results=results)
                    store_analytics_data('calculation_results', 'shallow_summary', results.get('summary'))
                    logger.info("Shallow footing calculation complete (%s model)",
                                results.get('soil_decision', {}).get('soil_model_used'))
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in shallow footing calculation: {str(e)}")
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            elif type == 'cantilever':
                flow = _flow_row(type)
                cpt_blob = _flow_cpt(flow)
                if not cpt_blob:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))

                # Embedded cantilever wall in sand.
                def _optc(name, default):
                    val = request.form.get(name, '')
                    try:
                        return float(val) if val not in (None, '') else default
                    except (TypeError, ValueError):
                        return default

                wall_params = {
                    'water_table': float(flow.water_table or 0),
                    'wall_length': _optc('wall_length', None),
                    'excavation_depth': _optc('excavation_depth', None),
                    'EI_kNm2_per_m': _optc('EI_kNm2_per_m', None),
                    'wall_name': request.form.get('wall_name', ''),
                }
                errors = []
                if not wall_params['wall_length'] or wall_params['wall_length'] <= 0:
                    errors.append('Wall length must be greater than 0')
                if not wall_params['excavation_depth'] or wall_params['excavation_depth'] <= 0:
                    errors.append('Excavation depth must be greater than 0')
                if not wall_params['EI_kNm2_per_m'] or wall_params['EI_kNm2_per_m'] <= 0:
                    errors.append('Wall flexural rigidity EI must be greater than 0')
                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                record_event('calculation', 'cantilever_params', {
                    k: wall_params[k] for k in ('wall_length', 'excavation_depth', 'EI_kNm2_per_m')
                })

                # Profile computed once at upload; recompute only for legacy flows
                processed_cpt = cpt_blob.get('processed') or pre_input_calc(cpt_blob, float(flow.water_table or 0))
                try:
                    results = calculate_cantilever_results(processed_cpt, wall_params)
                    _flow_save_state(flow, pile_params=wall_params, results=results)
                    if results.get('aborted'):
                        store_analytics_data('calculation_results', 'cantilever_aborted',
                                             results.get('checks'))
                    else:
                        store_analytics_data('calculation_results', 'cantilever_summary',
                                             results.get('summary'))
                    logger.info("Cantilever wall calculation complete (aborted=%s)",
                                results.get('aborted'))
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in cantilever wall calculation: {str(e)}")
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            elif type == 'lateral':
                flow = _flow_row(type)
                cpt_blob = _flow_cpt(flow)
                if not cpt_blob:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))

                # Laterally loaded monopiles in sand.
                def _opt(name, default):
                    val = request.form.get(name, '')
                    try:
                        return float(val) if val not in (None, '') else default
                    except (TypeError, ValueError):
                        return default

                lateral_params = {
                    'water_table': float(flow.water_table or 0),
                    'gamma_above': _opt('gamma_above', 17.1),
                    'gamma_below': _opt('gamma_below', 19.9),
                    'diameter': _opt('diameter', None),
                    'embedded_length': _opt('embedded_length', None),
                    'pile_type': request.form.get('pile_type', 'Pipe'),
                    'wall_thickness_mm': _opt('wall_thickness_mm', 10),
                    'load_height_above_ground': _opt('load_height_above_ground', 0),
                    'youngs_modulus_GPa': _opt('youngs_modulus_GPa', 210),
                    'hult_coefficient': _opt('hult_coefficient', 0.3),
                    'hult_exp_qc': _opt('hult_exp_qc', 0.7),
                    'hult_exp_sigv': _opt('hult_exp_sigv', 0.1),
                    'pile_name': request.form.get('pile_name', ''),
                    'force_g0_profile': request.form.get('force_g0_profile') or None,
                    'g0_075L_override': _opt('g0_075L_override', None),
                }

                errors = []
                if not lateral_params['diameter'] or lateral_params['diameter'] <= 0:
                    errors.append('Pile diameter must be greater than 0')
                if not lateral_params['embedded_length'] or lateral_params['embedded_length'] <= 0:
                    errors.append('Embedded length must be greater than 0')
                if lateral_params['load_height_above_ground'] is None or lateral_params['load_height_above_ground'] < 0:
                    errors.append('Load height above ground cannot be negative')
                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                record_event('calculation', 'lateral_params', {
                    k: lateral_params[k] for k in
                    ('diameter', 'embedded_length', 'pile_type',
                     'load_height_above_ground', 'youngs_modulus_GPa')
                })

                # Profile computed once at upload; recompute only for legacy flows
                processed_cpt = cpt_blob.get('processed') or pre_input_calc(cpt_blob, float(flow.water_table or 0))
                try:
                    results = calculate_lateral_monopile_results(processed_cpt, lateral_params)
                    _flow_save_state(flow, pile_params=lateral_params, results=results)
                    if results.get('aborted'):
                        # Show the abort/warning page directly at step 4
                        store_analytics_data('calculation_results', 'lateral_aborted',
                                             results.get('checks'))
                    else:
                        store_analytics_data('calculation_results', 'lateral_summary',
                                             results.get('summary'))
                    logger.info("Lateral monopile calculation complete (aborted=%s)",
                                results.get('aborted'))
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in lateral monopile calculation: {str(e)}")
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            elif type == 'driven':
                flow = _flow_row(type)
                cpt_blob = _flow_cpt(flow)
                if not cpt_blob:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))

                # Add validation for driven pile parameters
                required_fields = ['pile_diameter', 'wall_thickness', 'borehole_depth', 'pile_shape', 'pile_end_condition']
                for field in required_fields:
                    if field not in request.form:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Parse pile tip depths from the comma-separated string
                tip_depths_str = request.form.get('pile_tip_depths', '')
                try:
                    pile_tip_depths = [float(d.strip()) for d in tip_depths_str.split(',')]
                except ValueError:
                    flash('Invalid pile tip depths format. Please enter numbers separated by commas.')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
                
                pile_params = {
                    'pile_diameter': float(request.form['pile_diameter']),
                    'wall_thickness': float(request.form['wall_thickness']),
                    'borehole_depth': float(request.form['borehole_depth']),
                    'pile_shape': request.form['pile_shape'],
                    'pile_end_condition': request.form['pile_end_condition'],
                    'water_table': float(flow.water_table or 0),
                    'site_name': request.form.get('site_name', ''),
                    'pile_tip_depths': pile_tip_depths
                }

                record_event('calculation', 'driven_params', {
                    'pile_diameter': pile_params['pile_diameter'],
                    'wall_thickness': pile_params['wall_thickness'],
                    'pile_shape': pile_params['pile_shape'],
                    'pile_end_condition': pile_params['pile_end_condition'],
                    'borehole_depth': pile_params['borehole_depth'],
                    'tip_depths': pile_tip_depths,
                })

                # Validate pile parameters
                errors = []
                if pile_params['pile_diameter'] <= 0:
                    errors.append('Pile diameter must be greater than 0')
                if pile_params['pile_shape'] == 'circular' and pile_params['wall_thickness'] <= 0:
                    errors.append('Wall thickness must be greater than 0')
                if pile_params['pile_shape'] == 'circular' and pile_params['wall_thickness'] >= pile_params['pile_diameter'] * 1000 / 2:
                    errors.append('Wall thickness must be less than half the pile diameter')
                if pile_params['borehole_depth'] < 0:
                    errors.append('Borehole depth cannot be negative')

                # Validate pile tip depths
                if not pile_params.get('pile_tip_depths'):
                    errors.append('At least one pile tip depth is required')
                else:
                    max_depth = max(row['z'] for row in cpt_blob['cpt_data'])
                    for depth in pile_params['pile_tip_depths']:
                        if depth <= 0:
                            errors.append(f'Pile tip depth {depth}m must be greater than 0')
                        if depth > max_depth:
                            errors.append(f'Pile tip depth {depth}m exceeds the deepest CPT reading ({max_depth:.2f}m). Reduce the tip depth or upload deeper CPT data.')

                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                # Profile computed once at upload; recompute only for legacy flows
                processed_cpt = cpt_blob.get('processed') or pre_input_calc(cpt_blob, float(flow.water_table or 0))

                try:
                    # Calculate results using the driven pile specific function
                    results = calculate_driven_pile_results(processed_cpt, pile_params)

                    # Compute continuous capacity envelope for the graph
                    try:
                        envelope = compute_capacity_envelope_driven(processed_cpt, pile_params)
                    except Exception as env_e:
                        logger.warning(f"Could not compute capacity envelope: {env_e}")
                        envelope = None

                    _flow_save_state(flow,
                                     pile_params=pile_params,
                                     results=results['summary'],
                                     capacity_envelope=envelope,
                                     debug=results['detailed'])

                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            elif type == 'helical':
                # Validate required fields
                required_fields = ['shaft_diameter', 'helix_diameter', 'helix_depth', 'borehole_depth', 'water_table']
                for field in required_fields:
                    if field not in request.form or not request.form[field]:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Store parameters in session
                pile_params = {
                    'shaft_diameter': request.form.get('shaft_diameter'),
                    'helix_diameter': request.form.get('helix_diameter'),
                    'helix_depth': request.form.get('helix_depth'),
                    'borehole_depth': request.form.get('borehole_depth'),
                    'water_table': request.form.get('water_table'),
                    'site_name': request.form.get('site_name', '')
                }
                
                # Convert values to float and validate
                for key in ['shaft_diameter', 'helix_diameter', 'helix_depth', 'borehole_depth', 'water_table']:
                    try:
                        if pile_params[key]:
                            value = float(pile_params[key])
                            if key != 'water_table' and value <= 0:
                                flash(f'{key} must be greater than 0')
                                return redirect(url_for('main.calculator_step', type=type, step=3))
                            pile_params[key] = value
                    except (ValueError, TypeError):
                        flash(f'Invalid value for {key}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Store in session
                session['pile_params'] = pile_params

                record_event('calculation', 'helical_params', {
                    'shaft_diameter': pile_params.get('shaft_diameter'),
                    'helix_diameter': pile_params.get('helix_diameter'),
                    'helix_depth': pile_params.get('helix_depth'),
                })

                # Process the data and calculate results
                if 'cpt_data_id' not in session:
                    flash('No CPT data available. Please upload data first.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.', 'error')
                    return redirect(url_for('main.calculator_step', type='helical', step=1))

                # Process the CPT data
                processed_cpt = pre_input_calc(cpt_data, float(session['pile_params']['water_table']))
                
                try:
                    # Calculate results using the helical pile specific function
                    results = calculate_helical_pile_results(processed_cpt, session['pile_params'])
                    
                    # Store summary results in session
                    session['results'] = results['summary']
                    
                    # Create detailed results with all necessary data
                    detailed_results = {
                        'calculations': results['detailed'],
                        'helix_calculations': {
                            'perimeter': results['detailed'].get('perimeter'),
                            'helix_area': results['detailed'].get('helix_area'),
                            'q1_helix': results['detailed'].get('q1_helix'),
                            'q10_helix': results['detailed'].get('q10_helix'),
                            'qhelix_tension': results['detailed'].get('qhelix_tension'),
                            'qhelix_compression': results['detailed'].get('qhelix_compression')
                        },
                        'input_parameters': session['pile_params']
                    }
                    
                    debug_id = save_debug_details([detailed_results])
                    session['debug_id'] = debug_id

                    logger.info("Helical pile calculations completed for %s (debug_id=%s)", pile_params.get('site_name'), debug_id)

                    session.pop('results_id', None)
                    session.pop('detailed_results', None)
                    
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in helical pile calculations: {str(e)}")
                    flash(f"Error in calculation: {str(e)}")
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            else:
                results = calculate_pile_capacity(cpt_data, session['pile_params'], pile_type=type)
                session['results'] = results
        
        elif step == 4:
            return _render_step4(type, show_modal)

        return render_template(f'{type}/steps.html', step=step, type=type, show_modal=show_modal)

    # Handle GET requests
    if step == 2:
        flow = _flow_row(type)
        data = _flow_cpt(flow)
        if not data:
            flash('No CPT data available. Please upload data first.')
            return redirect(url_for('main.calculator_step', type=type, step=1))

        water_table = float(flow.water_table or 0)
        processed = data.get('processed')
        if type == 'bored':
            graphs = create_bored_pile_graphs(data, water_table=water_table, processed=processed)
        elif type == 'helical':
            graphs = create_helical_pile_graphs(data, water_table=water_table, processed=processed)
        else:
            graphs = create_cpt_graphs(data, water_table, processed=processed)

        # Add info message for large datasets
        cpt_data = data['cpt_data']
        if len(cpt_data) > 1000:
            flash(f'Large dataset detected ({len(cpt_data)} data points). Graphs show sampled data for performance. Full dataset will be used for calculations.', 'info')

        return render_template(f'{type}/steps.html', step=step, graphs=graphs, type=type, show_modal=show_modal)

    elif step == 3:
        flow = _flow_row(type)
        if not _flow_has_cpt(flow):
            flash('No CPT data available. Please complete previous steps first.')
            return redirect(url_for('main.calculator_step', type=type, step=1))
        logger.info(f"Rendering step 3 for {type} piles")
        state = _flow_state(flow)
        return render_template(f'{type}/steps.html', step=step, type=type, show_modal=show_modal,
                               pile_params=state.get('pile_params') or {},
                               water_table=flow.water_table)
        
    elif step == 4:
        return _render_step4(type, show_modal)

    # For step 1 with sample data loaded, pass sample data for preview
    sample_data = None
    sample_water_table = None
    if step == 1 and request.args.get('sample'):
        flow = _flow_row(type)
        data = _flow_cpt(flow)
        if data:
            sample_data = data['cpt_data']
            sample_water_table = flow.water_table if flow.water_table is not None else 2.0

    return render_template(f'{type}/steps.html', step=step, type=type, show_modal=show_modal,
                           sample_data=sample_data, sample_water_table=sample_water_table)

# ---------------------------------------------------------------------------
# Deliverable-output helpers shared by the CSV and PDF downloads.
# ---------------------------------------------------------------------------

# Stamped on every CSV and PDF so a file found in a project folder later can
# be traced to the code that produced it. Render injects the commit SHA.
APP_VERSION = (os.environ.get('RENDER_GIT_COMMIT') or os.environ.get('APP_VERSION') or 'dev')[:9]

# One-line method citations, matching the description pages.
_METHOD_CITATIONS = {
    'driven': 'Unified CPT method - Lehane et al. (2020), ISFOG-4 (sand); Lehane et al. (2022), ASCE JGGE 148(9) (clay)',
    'bored': 'Doan & Lehane (2021), CPT-based design method for axial capacities of bored piles in sand and clay',
    'helical': 'Bittar et al. (2023), CPT-based design method for helical piles in sand, Canadian Geotechnical Journal',
    'shallow': 'CPT-based footing load-settlement method, Lehane (2024)',
    'lateral': 'Wang et al. (2020, 2023), CPT-based lateral response of rigid monopiles in sand',
    'cantilever': 'Lehane, Bagbag, Durham & Pine (2019), embedded cantilever wall deflection in sand',
}

_RESULTS_DISCLAIMER = ('Unfactored results computed from the uploaded CPT data; '
                       'to be reviewed by a qualified geotechnical engineer before use in design.')


def _safe_filename_part(value):
    """Sanitise a user-supplied string for use inside a download filename."""
    cleaned = ''.join(c for c in str(value) if c.isalnum() or c in '._- ')
    return cleaned.strip().replace(' ', '_') or 'output'


def _flow_site_name(state):
    p = state.get('pile_params') or {}
    return (p.get('site_name') or p.get('pile_name') or p.get('wall_name') or '').strip()


def _deliverable_header_lines(flow, state):
    """Traceability header prepended to every results CSV."""

    def _clean(value):
        return str(value).replace('\r', ' ').replace('\n', ' ')

    lines = [
        '# UWA CPT Pile Calculator - https://uwa-geotech-cpt-calculator.com',
        '# Module: %s' % _HISTORY_TYPE_LABELS.get(flow.calc_type, flow.calc_type),
    ]
    site = _flow_site_name(state)
    if site:
        lines.append('# Site: %s' % _clean(site))
    if flow.original_filename:
        lines.append('# CPT file: %s' % _clean(flow.original_filename))
    blob = _flow_cpt(flow)
    if blob:
        rows = blob['cpt_data']
        lines.append('# CPT points: %d (%.2f m to %.2f m)' % (len(rows), rows[0]['z'], rows[-1]['z']))
    if flow.water_table is not None:
        lines.append('# Water table depth: %s m (pore pressure assumed hydrostatic below)' % flow.water_table)
    method = _METHOD_CITATIONS.get(flow.calc_type)
    if method:
        lines.append('# Method: %s' % method)
    lines.append('# Generated: %s | App version: %s' % (datetime.now().strftime('%Y-%m-%d %H:%M'), APP_VERSION))
    lines.append('# %s' % _RESULTS_DISCLAIMER)
    return lines


def _write_shallow_results_csv(writer, results):
    sm = results.get('summary') or {}
    sd = results.get('soil_decision') or {}
    inputs = results.get('inputs') or {}
    writer.writerow(['SUMMARY'])
    writer.writerow(['Soil model used', sd.get('soil_model_used', '')])
    writer.writerow(['Average Ic in zone of influence', sd.get('avg_ic', '')])
    writer.writerow(['Footing width B (m)', inputs.get('footing_width_m', '')])
    writer.writerow(['Footing length L (m)', inputs.get('footing_length_m', '')])
    writer.writerow(['Founding depth below excavation (m)', inputs.get('founding_depth_m', '')])
    writer.writerow(['Site excavation depth (m)', inputs.get('excavation_depth_m', '')])
    writer.writerow(['Unit weight used (kN/m3)', inputs.get('unit_weight_used_knm3', '')])
    writer.writerow(['Zone of influence (m)', '%s to %s' % (sm.get('zone_top_m', ''), sm.get('zone_base_m', ''))])
    for key, label in (
        ('qc_avg_mpa', 'qc,avg in zone after excavation (MPa)'),
        ('kc_silt_correction', 'Silt correction Kc'),
        ('avg_friction_angle_deg', 'Average peak friction angle (deg)'),
        ('qb01_kpa', 'qb0.1 bearing stress at s/B=0.1 (kPa)'),
        ('qt_net_kpa', 'qt,net in zone (kPa)'),
        ('avg_su_kpa', 'Average su (kPa)'),
        ('avg_ocr', 'Average OCR'),
        ('svy_footing_kpa', "sigma'vy at footing level (kPa)"),
        ('ultimate_curve_cap_kpa', 'Ultimate bearing pressure, 0.42 qt,net (kPa)'),
        ('bearing_capacity_cpt_kpa', 'Net bearing capacity, CPT (kPa)'),
        ('bearing_capacity_nq_ng_kpa', 'Net bearing capacity, Nq/Ngamma (kPa)'),
        ('bearing_capacity_nc_kpa', 'Net bearing capacity, Nc (kPa)'),
    ):
        if sm.get(key) is not None:
            writer.writerow([label, sm[key]])
    curve = results.get('curve') or {}
    for series in curve.get('series') or []:
        writer.writerow([])
        writer.writerow(['LOAD-SETTLEMENT CURVE: %s' % series.get('name', '')])
        writer.writerow([curve.get('x_label', 'Settlement (mm)'), curve.get('y_label', 'Bearing pressure (kPa)')])
        for pt in series.get('points') or []:
            writer.writerow([pt[0], pt[1]])


def _write_lateral_results_csv(writer, results):
    sm = results.get('summary') or {}
    inputs = results.get('inputs') or {}
    writer.writerow(['SUMMARY'])
    writer.writerow(['Pile diameter D (m)', inputs.get('D_m', '')])
    writer.writerow(['Embedded length L (m)', inputs.get('L_m', '')])
    writer.writerow(['EI (kNm2)', inputs.get('EI_kNm2', '')])
    writer.writerow(['Load height above ground (m)', inputs.get('h_m', '')])
    writer.writerow(['Hu, geotechnical (kN)', sm.get('Hu_kN', '')])
    curves = [('curve_load_disp', 'LOAD-DISPLACEMENT CURVE')]
    # Moment at the pile head is zero when the load is applied at ground level
    if inputs.get('h_m'):
        curves.append(('curve_moment_rotation', 'MOMENT-ROTATION CURVE'))
    for curve_key, title in curves:
        curve = results.get(curve_key) or {}
        for series in curve.get('series') or []:
            writer.writerow([])
            writer.writerow([title])
            writer.writerow([curve.get('x_label', ''), curve.get('y_label', '')])
            for pt in series.get('points') or []:
                writer.writerow([pt[0], pt[1]])


def _write_cantilever_results_csv(writer, results):
    sm = results.get('summary') or {}
    inputs = results.get('inputs') or {}
    writer.writerow(['SUMMARY'])
    writer.writerow(['Maximum wall deflection dmax (mm)', sm.get('dmax_mm', '')])
    writer.writerow(['Wall length L (m)', inputs.get('wall_length_m', '')])
    writer.writerow(['Excavation depth H (m)', inputs.get('excavation_depth_m', '')])
    writer.writerow(['H/L', sm.get('hl_ratio', '')])
    writer.writerow(['Wall EI (kNm2/m)', inputs.get('EI_kNm2_per_m', '')])
    writer.writerow(['EI / (gamma_w x H^4)', sm.get('ei_normalised_raw', '')])
    writer.writerow(["Average phi'p in averaging band (deg)", sm.get('avg_phi_prime_deg', '')])
    writer.writerow(['dqc/dz (MN/m3)', sm.get('dqc_dz_MN_m3', '')])
    writer.writerow(['Maximum settlement smax (mm)', sm.get('smax_mm', '')])
    writer.writerow(['Settlement at x = H/2 (mm)', sm.get('s_at_x_H_over_2_mm', '')])
    profile = results.get('settlement_profile') or {}
    for series in profile.get('series') or []:
        writer.writerow([])
        writer.writerow(['SETTLEMENT PROFILE BEHIND THE WALL'])
        writer.writerow([profile.get('x_label', 'Distance behind wall (m)'), profile.get('y_label', 'Settlement (mm)')])
        for pt in series.get('points') or []:
            writer.writerow([pt[0], pt[1]])


@bp.route('/download_debug_params')
def download_debug_params():
    """Download debug parameters and calculation data as CSV"""
    record_event('download', 'download_detailed_output', {
        'pile_type': session.get('type'),
        'email': session.get('user_email'),
    })
    flow = _flow_row()
    data = _flow_cpt(flow)
    if not data:
        flash('No CPT data available')
        return redirect(url_for('main.index'))

    try:
        state = _flow_state(flow)
        pile_params = state.get('pile_params') or {}

        # Profile computed once at upload; recompute only for legacy flows
        processed = data.get('processed') or pre_input_calc(data, float(flow.water_table or 0))

        # Create a string buffer
        buffer = io.StringIO()

        pile_type = flow.calc_type

        if pile_type == 'bored':
            debug_details = state.get('debug')
            if isinstance(debug_details, dict):
                tips = debug_details.get('tips', [])
                cpt_profile_list = debug_details.get('cpt_profile', [])
            else:
                tips = debug_details or []
                cpt_profile_list = []
            cpt_profile_dict = {p['depth']: p for p in cpt_profile_list}
            if tips:

                for tip_index, tip_detail in enumerate(tips):
                    if tip_index > 0:
                        # Add separator between different tip depth data
                        buffer.write('\n\n' + '='*50 + '\n\n')
                    
                    # Create constants list with tip depth included
                    constants = [
                        ['Tip Depth (m)', tip_detail['tip_depth']],
                        ['Water table depth (m)', float(pile_params['water_table'])]
                    ]
                    
                    # Add pile type specific parameters
                    constants.extend([
                        ['Pile type', 'Bored'],
                        ['Shaft diameter (m)', pile_params.get('shaft_diameter', 'N/A')],
                        ['Base diameter (m)', pile_params.get('base_diameter', 'N/A')],
                        ['Cased depth (m)', pile_params.get('cased_depth', 'N/A')]
                    ])
                    
                    # Add pile calculation constants if available
                    if 'pile_constants' in tip_detail:
                        pile_consts = tip_detail['pile_constants']
                        constants.extend([
                            ['', ''],  # Empty row for spacing
                            ['PILE CALCULATION CONSTANTS', ''],
                            ['Pile Perimeter (m)', pile_consts.get('pile_perimeter', 'N/A')],
                            ['Base Area (m²)', pile_consts.get('base_area', 'N/A')],
                            ['Minimum qb0.1 (MPa)', pile_consts.get('min_qb01', 'N/A')],
                            ['Total Base Resistance (kN)', pile_consts.get('base_resistance', 'N/A')]
                        ])
                    
                    # Write constants
                    df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                    buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_detail["tip_depth"]}m\n')
                    df_constants.to_csv(buffer, index=False)
                    
                    # Process calculations for this tip depth
                    calcs = tip_detail['calculations']
                    calc_dict = {calc['depth']: calc for calc in calcs}

                    # Create and populate DataFrame
                    df_data = create_data_dataframe(processed, calc_dict, cpt_profile_dict)
                    
                    # Add a blank line between constants and data
                    buffer.write('\nCPT DATA AND CALCULATIONS\n')
                    df_data.to_csv(buffer, index=False)
        elif pile_type == 'driven':
            # For driven piles, use the stored debug details like bored piles
            debug_details = state.get('debug')
            if debug_details:
                if isinstance(debug_details, dict):
                    tips = debug_details.get('tips', [])
                    cpt_profile_list = debug_details.get('cpt_profile', [])
                else:
                    tips = debug_details or []
                    cpt_profile_list = []
                cpt_profile_dict = {p['depth']: p for p in cpt_profile_list}
                if tips:

                    for tip_index, tip_detail in enumerate(tips):
                        if tip_index > 0:
                            # Add separator between different tip depth data
                            buffer.write('\n\n' + '='*50 + '\n\n')
                        
                        # Create constants list with tip depth included
                        constants = [
                            ['Tip Depth (m)', tip_detail['tip_depth']],
                            ['Water table depth (m)', float(pile_params['water_table'])]
                        ]
                        
                        # Calculate area and perimeter like in calculate_driven_pile_results
                        pile_shape = 0 if pile_params.get('pile_shape') == 'circular' else 1
                        nominal_size_don = float(pile_params.get('pile_diameter', 0))
                        
                        # Calculate area
                        if pile_shape == 0:  # circular
                            area_value = (3.14159/4) * (nominal_size_don**2)
                            pile_perimeter = 3.14159 * nominal_size_don
                        else:  # square
                            area_value = nominal_size_don**2
                            pile_perimeter = 4 * nominal_size_don
                        
                        # Add pile type specific parameters
                        constants.extend([
                            ['Pile type', 'Driven'],
                            ['Pile end condition', pile_params.get('pile_end_condition', 'N/A')],
                            ['Pile shape', pile_params.get('pile_shape', 'N/A')],
                            ['Pile diameter/width (m)', pile_params.get('pile_diameter', 'N/A')],
                            ['Wall thickness (mm)', pile_params.get('wall_thickness', 'N/A')],
                            ['Borehole depth (m)', pile_params.get('borehole_depth', 'N/A')],
                            ['Pile Area (m²)', f'{area_value:.4f}'],
                            ['Pile Perimeter (m)', f'{pile_perimeter:.4f}']
                        ])
                        
                        # Add pile calculation constants if available
                        if 'pile_constants' in tip_detail:
                            pile_consts = tip_detail['pile_constants']
                            constants.extend([
                                ['', ''],  # Empty row for spacing
                                ['PILE CALCULATION CONSTANTS', ''],
                                ['Internal Friction Ratio (IFR)', pile_consts.get('ifr_value', 'N/A')],
                                ['Area Ratio (Are)', pile_consts.get('are_value', 'N/A')],
                                ['Effective Diameter (Dstar)', pile_consts.get('dstar_value', 'N/A')],
                                ['Pile Shape Code', pile_consts.get('pile_shape', 'N/A')],
                                ['End Condition Code', pile_consts.get('pile_end_condition', 'N/A')]
                            ])
                        
                        # Write constants
                        df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                        buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_detail["tip_depth"]}m\n')
                        df_constants.to_csv(buffer, index=False)
                        
                        # Process calculations for this tip depth
                        calcs = tip_detail['calculations']
                        calc_dict = {calc['depth']: calc for calc in calcs}
                        
                        # Create and populate DataFrame for driven piles
                        df_data = create_driven_data_dataframe(processed, calc_dict, cpt_profile_dict)
                        
                        # Add a blank line between constants and data
                        buffer.write('\nCPT DATA AND CALCULATIONS\n')
                        df_data.to_csv(buffer, index=False)
            else:
                # Fallback to simplified format if no debug details
                results = state.get('results', [])

                for result_index, result in enumerate(results):
                    if result_index > 0:
                        buffer.write('\n\n' + '='*50 + '\n\n')
                    
                    tip_depth = result['tipdepth']
                    
                    # Create parameters list
                    constants = [
                        ['Tip Depth (m)', tip_depth],
                        ['Water table depth (m)', float(pile_params['water_table'])],
                        ['Pile type', 'Driven'],
                        ['Pile end condition', pile_params.get('pile_end_condition', 'N/A')],
                        ['Pile shape', pile_params.get('pile_shape', 'N/A')],
                        ['Pile diameter/width (m)', pile_params.get('pile_diameter', 'N/A')],
                        ['Wall thickness (mm)', pile_params.get('wall_thickness', 'N/A')],
                        ['Borehole depth (m)', pile_params.get('borehole_depth', 'N/A')]
                    ]
                    
                    # Write parameters
                    df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                    buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_depth}m\n')
                    df_constants.to_csv(buffer, index=False)
                    
                    # Write basic results for this tip depth
                    buffer.write('\nBASIC RESULTS\n')
                    df_result = pd.DataFrame([{
                        'Tip Depth (m)': result['tipdepth'],
                        'Tension Capacity (kN)': result['tension_capacity'],
                        'Compression Capacity (kN)': result['compression_capacity']
                    }])
                    df_result.to_csv(buffer, index=False)
        elif pile_type == 'helical':
            # For helical piles, use the stored debug details like bored piles
            debug_details = state.get('debug')
            if True:
                if debug_details and isinstance(debug_details, list) and len(debug_details) > 0:
                    detail_data = debug_details[0]
                    
                    # Write INPUT PARAMETERS section
                    buffer.write('INPUT PARAMETERS\n')
                    input_params = pd.DataFrame([
                        ['shaft_diameter (m)', pile_params.get('shaft_diameter', '')],
                        ['helix_diameter (m)', pile_params.get('helix_diameter', '')],
                        ['helix_depth (m)', pile_params.get('helix_depth', '')],
                        ['borehole_depth (m)', pile_params.get('borehole_depth', '')],
                        ['water_table (m)', pile_params.get('water_table', '')]
                    ])
                    input_params.to_csv(buffer, index=False, header=False)
                    buffer.write('\n')
                    
                    # Write GEOMETRIC CONSTANTS section
                    buffer.write('GEOMETRIC CONSTANTS\n')
                    if 'helix_calculations' in detail_data:
                        geometric_constants = pd.DataFrame([
                            ['Shaft Perimeter (m)', detail_data['helix_calculations'].get('perimeter', '')],
                            ['Helix Area (m²)', detail_data['helix_calculations'].get('helix_area', '')]
                        ])
                        geometric_constants.to_csv(buffer, index=False, header=False)
                    buffer.write('\n')
                    
                    # Write DETAILED CALCULATION TABLE section
                    buffer.write('DETAILED CALCULATION TABLE\n')
                    if 'calculations' in detail_data:
                        calcs_data = []
                        depths = detail_data['calculations'].get('depth', [])
                        
                        for i, depth in enumerate(depths):
                            row = {
                                'Depth (m)': depth,
                                # CPT Data
                                'qt (MPa)': detail_data['calculations'].get('qt', [])[i] if 'qt' in detail_data['calculations'] else '',
                                'qc (MPa)': detail_data['calculations'].get('qc', [])[i] if 'qc' in detail_data['calculations'] else '',
                                'fs (kPa)': detail_data['calculations'].get('fs', [])[i] if 'fs' in detail_data['calculations'] else '',
                                'Fr (%)': detail_data['calculations'].get('fr_percent', [])[i] if 'fr_percent' in detail_data['calculations'] else '',
                                'Ic': detail_data['calculations'].get('lc', [])[i] if 'lc' in detail_data['calculations'] else '',
                                'Soil Type': detail_data['calculations'].get('soil_type', [])[i] if 'soil_type' in detail_data['calculations'] else '',
                                # Pile Capacity Parameters
                                'q1 (MPa)': detail_data['calculations'].get('q1', [])[i] if 'q1' in detail_data['calculations'] else '',
                                'q10 (MPa)': detail_data['calculations'].get('q10', [])[i] if 'q10' in detail_data['calculations'] else '',
                                # Shaft Calculations
                                'Casing Coefficient': detail_data['calculations'].get('coe_casing', [])[i] if 'coe_casing' in detail_data['calculations'] else '',
                                'Delta Z (m)': detail_data['calculations'].get('delta_z', [])[i] if 'delta_z' in detail_data['calculations'] else '',
                                'Shaft Force (kN)': detail_data['calculations'].get('qshaft_segment', [])[i] if 'qshaft_segment' in detail_data['calculations'] else '',
                                'Cumulative Shaft Force (kN)': detail_data['calculations'].get('qshaft_kn', [])[i] if 'qshaft_kn' in detail_data['calculations'] else ''
                            }
                            calcs_data.append(row)
                        
                        if calcs_data:
                            df_calcs = pd.DataFrame(calcs_data)
                            df_calcs.to_csv(buffer, index=False)
                    
                    # Write HELIX RESULTS section
                    buffer.write('\nHELIX RESULTS\n')
                    helix_results = pd.DataFrame([
                        ['q1 at Helix (MPa)', detail_data['helix_calculations'].get('q1_helix', '')],
                        ['q10 at Helix (MPa)', detail_data['helix_calculations'].get('q10_helix', '')],
                        ['Helix Tension Component (kN)', detail_data['helix_calculations'].get('qhelix_tension', '')],
                        ['Helix Compression Component (kN)', detail_data['helix_calculations'].get('qhelix_compression', '')]
                    ])
                    helix_results.to_csv(buffer, index=False, header=False)
                    buffer.write('\n')
                    
                    # Write FINAL RESULTS section
                    summary_results = state.get('results') or {}
                    buffer.write('FINAL RESULTS\n')
                    final_results = pd.DataFrame([
                        ['Ultimate Tension Capacity (kN)', summary_results.get('qult_tension', '')],
                        ['Ultimate Compression Capacity (kN)', summary_results.get('qult_compression', '')],
                        ['Tension Capacity at 10mm (kN)', summary_results.get('q_delta_10mm_tension', '')],
                        ['Compression Capacity at 10mm (kN)', summary_results.get('q_delta_10mm_compression', '')],
                        ['Installation Torque (kNm)', summary_results.get('installation_torque', '')]
                    ])
                    final_results.to_csv(buffer, index=False, header=False)

        # Get the buffer value
        buffer_value = buffer.getvalue()

        # Create a response with the CSV data
        user_filename = flow.original_filename or 'output'
        download_name = f"detailed_output_{_safe_filename_part(user_filename)}_{datetime.now().strftime('%d%m%Y')}.csv"

        return Response(
            buffer_value,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={download_name}"}
        )

    except Exception as e:
        logger.error(f"Debug download error: {str(e)}")
        flash(f"Error generating download: {str(e)}")
        return redirect(url_for('main.calculator_step', type=flow.calc_type if flow else 'driven', step=4))

@bp.route('/download_results')
def download_results():
    """Download calculation results as CSV, for all six modules."""
    record_event('download', 'download_csv', {
        'pile_type': session.get('type'),
        'email': session.get('user_email'),
    })
    flow = _flow_row()
    state = _flow_state(flow)
    results = state.get('results')
    if flow is None or results is None:
        flash('No results available')
        return redirect(url_for('main.index'))

    pile_type = flow.calc_type

    try:
        output = io.StringIO()
        for line in _deliverable_header_lines(flow, state):
            output.write(line + '\r\n')
        writer = csv.writer(output)

        if pile_type in ('driven', 'bored') and isinstance(results, list):
            writer.writerow(['Tip Depth (m)', 'Compression Capacity (kN)', 'Tension Capacity (kN)'])
            for result in results:
                writer.writerow([
                    result.get('tipdepth', 'N/A'),
                    result.get('compression_capacity', 'N/A'),
                    result.get('tension_capacity', 'N/A')
                ])
        elif pile_type == 'helical' and isinstance(results, dict):
            def _cap(key):
                # A capacity value must come from the calculation or be
                # visibly absent. Never substitute a number.
                if key not in results:
                    logger.warning("download_results: '%s' missing from helical summary; writing N/A", key)
                    return 'N/A'
                return results[key]

            writer.writerow(['CAPACITY', 'Qshaft (kN)', 'Q at delta=10mm (kN)', 'Qult (kN)', 'Installation torque (kNm)'])
            writer.writerow(['Tension', _cap('qshaft'), _cap('q_delta_10mm_tension'),
                             _cap('qult_tension'), _cap('installation_torque')])
            writer.writerow(['Compression', _cap('qshaft'), _cap('q_delta_10mm_compression'),
                             _cap('qult_compression'), '-'])
        elif pile_type == 'shallow' and isinstance(results, dict) and results.get('summary'):
            _write_shallow_results_csv(writer, results)
        elif pile_type == 'lateral' and isinstance(results, dict):
            if results.get('aborted'):
                flash('The analysis stopped before producing results, so there is nothing to download.')
                return redirect(url_for('main.calculator_step', type=pile_type, step=4))
            _write_lateral_results_csv(writer, results)
        elif pile_type == 'cantilever' and isinstance(results, dict):
            if results.get('aborted'):
                flash('The analysis stopped before producing results, so there is nothing to download.')
                return redirect(url_for('main.calculator_step', type=pile_type, step=4))
            _write_cantilever_results_csv(writer, results)
        else:
            # A mismatched results shape must never fall into another
            # module's writer. Refuse clearly instead.
            flash('Results format not recognized; please re-run the calculation.')
            return redirect(url_for('main.calculator_step', type=pile_type, step=4))

        base_names = {
            'driven': 'driven_pile_results',
            'bored': 'bored_pile_results',
            'helical': 'helical_pile_results',
            'shallow': 'shallow_footing_results',
            'lateral': 'lateral_monopile_results',
            'cantilever': 'cantilever_wall_results',
        }
        site_name = _flow_site_name(state)
        name_part = ('_' + _safe_filename_part(site_name)) if site_name else ''
        download_name = f"{base_names.get(pile_type, 'results')}{name_part}_{datetime.now().strftime('%Y%m%d')}.csv"

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={download_name}"}
        )
    except Exception as e:
        current_app.logger.error(f"Error generating results download: {str(e)}")
        flash(f'Error generating results download: {str(e)}')
        return redirect(url_for('main.calculator_step', type=pile_type, step=4))

_METHOD_SHORT = {
    'driven': 'Unified CPT method - Lehane et al. (2020, 2022)',
    'bored': 'Doan & Lehane (2021)',
    'helical': 'Bittar et al. (2023)',
    'shallow': 'Lehane (2024), CPT-based load-settlement',
    'lateral': 'Wang et al. (2020, 2023)',
    'cantilever': 'Lehane et al. (2019)',
}

_PDF_REFERENCES = {
    'driven': [
        'Lehane B.M., Liu Z., Bittar E., et al. (2020). A new CPT-based axial pile capacity '
        'design method for driven piles in sand. Proc 4th Int. Symposium on Frontiers in '
        'Offshore Geotechnics, ISFOG-4, Austin, Texas, 462-477.',
        'Lehane B.M., Liu Z., Bittar E., et al. (2022). CPT-based axial pile capacity design '
        'method for driven piles in clay. J. Geotech. &amp; Geoenv. Engrg., ASCE, 148(9).',
    ],
    'bored': [
        'Doan L.V. &amp; Lehane B.M. (2021). CPT-based design method for axial capacities of '
        'bored piles in sand and clay.',
    ],
    'helical': [
        'Bittar E.J., Lehane B.M., Blake A., et al. (2023). CPT-based design method for '
        'helical piles in sand. Canadian Geotechnical Journal.',
    ],
    'shallow': [
        'Lehane, B.M. (2024). Ongoing development of applications of the Cone Penetration Test '
        'in interpretation and design. 11th James Mitchell Honor Lecture, Proc. 7th Int. Conf. '
        'on Geotechnical and Geophysical Site Characterisation, Barcelona, 1, 87-104.',
    ],
    'lateral': [
        'Wang, H., Lehane B.M., Bransby, M.F., Wang, L.Z. and Hong, Y. (2020). A simple '
        'approach for predicting the ultimate lateral capacity of a rigid pile in sand. '
        'Geotechnique Letters, 10, 429-435.',
        'Wang, H., Lehane B.M., Bransby, M.F., Wang, L.Z., Hong, Y. and Askarinejad, A. (2023). '
        'Lateral behavior of monopiles in sand under monotonic loading. Ocean Engineering, 277, 114334.',
    ],
    'cantilever': [
        'Lehane, B.M., Bagbag, A., Durham, C., &amp; Pine, T. (2019). Design charts for lateral '
        'deflection of embedded cantilever walls in unsaturated Perth sands. Proc. 13th ANZ '
        'Conference on Geomechanics, Perth.',
    ],
}


@bp.route('/download_pdf_report')
def download_pdf_report():
    """Generate and download a deliverable-grade PDF report (all six modules)."""
    record_event('download', 'download_pdf', {
        'pile_type': session.get('type'),
        'email': session.get('user_email'),
    })
    flow = _flow_row()
    state = _flow_state(flow)
    results = state.get('results')
    if flow is None or results is None:
        flash('No results available')
        return redirect(url_for('main.index'))
    if isinstance(results, dict) and results.get('aborted'):
        flash('The analysis stopped before producing results, so there is no report to download.')
        return redirect(url_for('main.calculator_step', type=flow.calc_type, step=4))

    try:
        return _build_pdf_report(flow, state, results)
    except ImportError:
        flash('PDF generation is not available. Please install reportlab.')
        return redirect(url_for('main.calculator_step', type=flow.calc_type, step=4))
    except Exception as e:
        current_app.logger.error(f"Error generating PDF report: {str(e)}")
        flash(f'Error generating PDF: {str(e)}')
        return redirect(url_for('main.calculator_step', type=flow.calc_type, step=4))


def _pdf_chart_image(draw, width_mm=150, height_mm=100):
    """Render a matplotlib chart to a reportlab Image flowable.

    ``draw`` receives the matplotlib figure and adds its own axes. Uses the
    object-oriented Figure API rather than pyplot: pyplot keeps a
    process-global figure registry that is not thread-safe, and this runs
    inside (possibly concurrent) request handlers.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from reportlab.platypus import Image
    from reportlab.lib.units import mm as _mm

    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=150)
    FigureCanvasAgg(fig)
    draw(fig)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image(buf, width=width_mm * _mm, height=height_mm * _mm)


def _build_pdf_report(flow, state, results):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    pile_type = flow.calc_type
    pile_params = state.get('pile_params') or {}
    blob = _flow_cpt(flow) or {}
    cpt_rows = blob.get('cpt_data') or []
    processed = blob.get('processed')
    if processed is None and cpt_rows:
        processed = pre_input_calc(blob, float(flow.water_table or 0))

    buffer = io.BytesIO()

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(20 * mm, 12 * mm,
                          f'UWA CPT Pile Calculator | uwa-geotech-cpt-calculator.com | '
                          f'Developed by Vortexia Solutions | version {APP_VERSION}')
        canvas.drawRightString(190 * mm, 12 * mm, f'Page {doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=20 * mm, bottomMargin=22 * mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                 fontSize=18, spaceAfter=6 * mm,
                                 textColor=colors.HexColor('#003087'))
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                    fontSize=10, textColor=colors.grey,
                                    spaceAfter=4 * mm)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                   fontSize=13, spaceAfter=3 * mm, spaceBefore=6 * mm,
                                   textColor=colors.HexColor('#003087'))
    normal_style = styles['Normal']
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                 fontSize=8, textColor=colors.grey)

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003087')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])
    centered_table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003087')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])

    module_titles = {
        'driven': 'Driven Pile Axial Capacity',
        'bored': 'Bored Pile Axial Capacity',
        'helical': 'Helical (Screw) Pile Capacity',
        'shallow': 'Shallow Footing Load-Settlement',
        'lateral': 'Laterally Loaded Monopile Response',
        'cantilever': 'Embedded Cantilever Wall Deflection',
    }

    elements = []
    elements.append(Paragraph('UWA CPT Pile Calculator Report', title_style))
    elements.append(Paragraph(
        f'{module_titles.get(pile_type, pile_type.title())} &mdash; '
        f'Generated {datetime.now().strftime("%d %B %Y, %H:%M")}',
        subtitle_style))
    elements.append(Spacer(1, 2 * mm))

    # --- Calculation data (CPT traceability) ------------------------------
    elements.append(Paragraph('Calculation Data', heading_style))
    data_rows = []
    site = _flow_site_name(state)
    if site:
        data_rows.append(['Site / project name', site])
    data_rows.append(['CPT data file', flow.original_filename or 'N/A'])
    if cpt_rows:
        data_rows.append(['CPT readings', f"{len(cpt_rows)} points, "
                                          f"{cpt_rows[0]['z']:.2f} m to {cpt_rows[-1]['z']:.2f} m depth"])
    data_rows.append(['Water table depth', f"{flow.water_table} m" if flow.water_table is not None else 'N/A'])
    data_rows.append(['Design method', _METHOD_SHORT.get(pile_type, 'N/A')])
    data_rows.append(['App version', APP_VERSION])
    data_table = Table([['Item', 'Value']] + data_rows, colWidths=[60 * mm, 100 * mm])
    data_table.setStyle(table_style)
    elements.append(data_table)

    # --- Input parameters --------------------------------------------------
    elements.append(Paragraph('Input Parameters', heading_style))
    param_data = []
    if pile_type == 'driven':
        param_data = [
            ['Pile Shape', pile_params.get('pile_shape', 'N/A')],
            ['Pile End Condition', pile_params.get('pile_end_condition', 'N/A')],
            ['Pile Diameter (m)', str(pile_params.get('pile_diameter', 'N/A'))],
            ['Wall Thickness (mm)', str(pile_params.get('wall_thickness', 'N/A'))],
            ['Borehole Depth (m)', str(pile_params.get('borehole_depth', 'N/A'))],
        ]
    elif pile_type == 'bored':
        param_data = [
            ['Shaft Diameter (m)', str(pile_params.get('shaft_diameter', 'N/A'))],
            ['Base Diameter (m)', str(pile_params.get('base_diameter', 'N/A'))],
            ['Cased Depth (m)', str(pile_params.get('cased_depth', 'N/A'))],
        ]
    elif pile_type == 'helical':
        param_data = [
            ['Shaft Diameter (m)', str(pile_params.get('shaft_diameter', 'N/A'))],
            ['Helix Diameter (m)', str(pile_params.get('helix_diameter', 'N/A'))],
            ['Helix Depth (m)', str(pile_params.get('helix_depth', 'N/A'))],
        ]
    elif pile_type == 'shallow':
        inputs = results.get('inputs') or {}
        param_data = [
            ['Footing Width B (m)', str(inputs.get('footing_width_m', 'N/A'))],
            ['Footing Length L (m)', str(inputs.get('footing_length_m', 'N/A'))],
            ['Founding Depth below excavation (m)', str(inputs.get('founding_depth_m', 'N/A'))],
            ['Site Excavation Depth (m)', str(inputs.get('excavation_depth_m', 'N/A'))],
            ['Design Life (years)', str(inputs.get('design_life_years', pile_params.get('design_life_years', 'N/A')))],
            ['Unit Weight Used (kN/m3)', str(inputs.get('unit_weight_used_knm3', 'N/A'))],
        ]
    elif pile_type == 'lateral':
        inputs = results.get('inputs') or {}
        param_data = [
            ['Pile Diameter D (m)', str(inputs.get('D_m', 'N/A'))],
            ['Embedded Length L (m)', str(inputs.get('L_m', 'N/A'))],
            ['EI (kNm2)', str(inputs.get('EI_kNm2', 'N/A'))],
            ['Load Height above Ground (m)', str(inputs.get('h_m', 'N/A'))],
        ]
    elif pile_type == 'cantilever':
        inputs = results.get('inputs') or {}
        param_data = [
            ['Wall Length L (m)', str(inputs.get('wall_length_m', 'N/A'))],
            ['Excavation Depth H (m)', str(inputs.get('excavation_depth_m', 'N/A'))],
            ['Wall EI (kNm2/m)', str(inputs.get('EI_kNm2_per_m', 'N/A'))],
        ]

    if param_data:
        param_table = Table([['Parameter', 'Value']] + param_data, colWidths=[90 * mm, 70 * mm])
        param_table.setStyle(table_style)
        elements.append(param_table)

    # --- Results -----------------------------------------------------------
    elements.append(Paragraph('Results', heading_style))

    chart = None
    if pile_type in ('driven', 'bored'):
        cap_data = [['Tip Depth (m)', 'Tension Capacity (kN)', 'Compression Capacity (kN)']]
        if isinstance(results, list):
            for r in results:
                cap_data.append([
                    f"{r['tipdepth']:.2f}",
                    f"{r['tension_capacity']:.0f}",
                    f"{r['compression_capacity']:.0f}"
                ])
        cap_table = Table(cap_data, colWidths=[50 * mm, 55 * mm, 55 * mm])
        cap_table.setStyle(centered_table_style)
        elements.append(cap_table)

        envelope = state.get('capacity_envelope')

        def _draw_capacity(fig):
            ax = fig.add_subplot(111)
            if envelope and envelope.get('depths'):
                ax.plot(envelope['tension'], envelope['depths'],
                        color='#1a73e8', alpha=0.4, lw=1.2, label='Tension profile (any tip depth)')
                ax.plot(envelope['compression'], envelope['depths'],
                        color='#dc2626', alpha=0.4, lw=1.2, label='Compression profile (any tip depth)')
            if isinstance(results, list) and results:
                tips = [r['tipdepth'] for r in results]
                ax.plot([r['tension_capacity'] for r in results], tips,
                        'o-', color='#1a73e8', lw=1.8, ms=5, label='Tension (entered tip depths)')
                ax.plot([r['compression_capacity'] for r in results], tips,
                        'o-', color='#dc2626', lw=1.8, ms=5, label='Compression (entered tip depths)')
            ax.invert_yaxis()
            ax.set_xlabel('Capacity (kN)')
            ax.set_ylabel('Depth (m)')
            ax.set_xlim(left=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            ax.set_title('Capacity vs Depth', fontsize=10)

        chart = _pdf_chart_image(_draw_capacity, width_mm=145, height_mm=110)

    elif pile_type == 'helical':
        cap_data = [
            ['CAPACITY', 'Qshaft (kN)', 'Q at δ=10mm (kN)', 'Qult (kN)'],
            ['Tension',
             f"{results.get('qshaft', 0):.1f}",
             f"{results.get('q_delta_10mm_tension', 0):.1f}",
             f"{results.get('qult_tension', 0):.1f}"],
            ['Compression',
             f"{results.get('qshaft', 0):.1f}",
             f"{results.get('q_delta_10mm_compression', 0):.1f}",
             f"{results.get('qult_compression', 0):.1f}"],
        ]
        cap_table = Table(cap_data, colWidths=[35 * mm, 40 * mm, 45 * mm, 40 * mm])
        cap_table.setStyle(centered_table_style)
        elements.append(cap_table)
        elements.append(Spacer(1, 3 * mm))
        extra_data = [
            ['Installation Torque (kNm)', f"{results.get('installation_torque', 0):.1f}"],
            ['Tip Depth (m)', f"{results.get('tipdepth', 0):.2f}"],
            ['qb0.1 Compression (MPa)', f"{results.get('qb01_comp', 0):.2f}"],
            ['qb0.1 Tension (MPa)', f"{results.get('qb01_tension', 0):.2f}"],
        ]
        extra_table = Table(extra_data, colWidths=[90 * mm, 70 * mm])
        extra_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(extra_table)

        table_rows = results.get('helical_deflection_table') or []
        if table_rows:
            def _draw_helical(fig):
                ax = fig.add_subplot(111)
                dc = [0] + [r['delta_mm_compression'] for r in table_rows if r.get('q_compression') is not None]
                qc_ = [0] + [r['q_compression'] for r in table_rows if r.get('q_compression') is not None]
                dt = [0] + [r['delta_mm_tension'] for r in table_rows if r.get('q_tension') is not None]
                qt_ = [0] + [r['q_tension'] for r in table_rows if r.get('q_tension') is not None]
                ax.plot(dc, qc_, color='#dc2626', lw=1.8, label='Compression')
                ax.plot(dt, qt_, color='#1a73e8', lw=1.8, label='Tension')
                ax.set_xlabel('Pile head displacement (mm)')
                ax.set_ylabel('Load (kN)')
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
                ax.set_title('Load vs Displacement', fontsize=10)

            chart = _pdf_chart_image(_draw_helical, width_mm=145, height_mm=100)

    elif pile_type == 'shallow':
        sm = results.get('summary') or {}
        sd = results.get('soil_decision') or {}
        res_rows = [['Soil model used', str(sd.get('soil_model_used', 'N/A'))],
                    ['Average Ic in zone of influence', f"{sd.get('avg_ic', 0):.2f}"],
                    ['Zone of influence (m)', f"{sm.get('zone_top_m', 0):.2f} to {sm.get('zone_base_m', 0):.2f}"]]
        for key, label, fmt in (
            ('qc_avg_mpa', 'qc,avg in zone (MPa)', '%.2f'),
            ('avg_friction_angle_deg', "Average peak friction angle (deg)", '%.1f'),
            ('qb01_kpa', 'qb0.1 at s/B = 0.1 (kPa)', '%.0f'),
            ('qt_net_kpa', 'qt,net in zone (kPa)', '%.1f'),
            ('avg_su_kpa', 'Average su (kPa)', '%.1f'),
            ('avg_ocr', 'Average OCR', '%.1f'),
            ('bearing_capacity_cpt_kpa', 'Net bearing capacity, CPT (kPa)', '%.0f'),
            ('bearing_capacity_nq_ng_kpa', 'Net bearing capacity, Nq/Ngamma (kPa)', '%.0f'),
            ('bearing_capacity_nc_kpa', 'Net bearing capacity, Nc (kPa)', '%.0f'),
        ):
            if sm.get(key) is not None:
                res_rows.append([label, fmt % sm[key]])
        res_table = Table([['Quantity', 'Value']] + res_rows, colWidths=[90 * mm, 70 * mm])
        res_table.setStyle(table_style)
        elements.append(res_table)

        curve = results.get('curve') or {}

        def _draw_shallow(fig):
            ax = fig.add_subplot(111)
            palette = ['#1a73e8', '#e8710a', '#188038']
            for i, series in enumerate(curve.get('series') or []):
                pts = series.get('points') or []
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        lw=1.8, color=palette[i % len(palette)], label=series.get('name', ''))
            ax.set_xlabel(curve.get('x_label', 'Settlement (mm)'))
            ax.set_ylabel(curve.get('y_label', 'Bearing pressure (kPa)'))
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            ax.set_title('Load-Settlement Response', fontsize=10)

        if curve.get('series'):
            chart = _pdf_chart_image(_draw_shallow, width_mm=145, height_mm=100)

    elif pile_type == 'lateral':
        sm = results.get('summary') or {}
        inputs = results.get('inputs') or {}
        res_rows = [['Hu, geotechnical (MN)', f"{sm.get('Hu_kN', 0) / 1000:.2f}"]]
        res_table = Table([['Quantity', 'Value']] + res_rows, colWidths=[90 * mm, 70 * mm])
        res_table.setStyle(table_style)
        elements.append(res_table)

        def _draw_lateral(fig):
            has_mr = bool(inputs.get('h_m')) and results.get('curve_moment_rotation')
            n_ax = 2 if has_mr else 1
            curve = results.get('curve_load_disp') or {}
            ax = fig.add_subplot(1, n_ax, 1)
            for series in curve.get('series') or []:
                pts = series.get('points') or []
                ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=2.5, lw=1.6, color='#1a73e8')
            ax.set_xlabel(curve.get('x_label', ''), fontsize=8)
            ax.set_ylabel(curve.get('y_label', ''), fontsize=8)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.set_title('Load vs Displacement', fontsize=9)
            if has_mr:
                curve2 = results.get('curve_moment_rotation') or {}
                ax2 = fig.add_subplot(1, 2, 2)
                for series in curve2.get('series') or []:
                    pts = series.get('points') or []
                    ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=2.5, lw=1.6, color='#e8710a')
                ax2.set_xlabel(curve2.get('x_label', ''), fontsize=8)
                ax2.set_ylabel(curve2.get('y_label', ''), fontsize=8)
                ax2.set_xlim(left=0)
                ax2.set_ylim(bottom=0)
                ax2.grid(alpha=0.25)
                ax2.set_title('Moment vs Rotation', fontsize=9)

        chart = _pdf_chart_image(_draw_lateral, width_mm=160, height_mm=85)

    elif pile_type == 'cantilever':
        sm = results.get('summary') or {}
        res_rows = [
            ['Maximum wall deflection dmax (mm)', f"{sm.get('dmax_mm', 0):.1f}"],
            ['H/L', f"{sm.get('hl_ratio', 0):.2f}"],
            ["Average phi'p in averaging band (deg)", f"{sm.get('avg_phi_prime_deg', 0):.0f}"],
            ['dqc/dz (MN/m3)', f"{sm.get('dqc_dz_MN_m3', 0):.1f}"],
            ['Maximum settlement smax (mm)', f"{sm.get('smax_mm', 0):.1f}"],
            ['Settlement at x = H/2 (mm)', f"{sm.get('s_at_x_H_over_2_mm', 0):.1f}"],
        ]
        res_table = Table([['Quantity', 'Value']] + res_rows, colWidths=[90 * mm, 70 * mm])
        res_table.setStyle(table_style)
        elements.append(res_table)

        profile = results.get('settlement_profile') or {}

        def _draw_cantilever(fig):
            ax = fig.add_subplot(111)
            for series in profile.get('series') or []:
                pts = series.get('points') or []
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.8, color='#1a73e8')
            ax.set_xlabel(profile.get('x_label', 'Distance behind wall, x (m)'))
            ax.set_ylabel(profile.get('y_label', 'Settlement (mm)'))
            ax.invert_yaxis()
            ax.grid(alpha=0.25)
            ax.set_title('Settlement Profile behind the Wall', fontsize=10)

        if profile.get('series'):
            chart = _pdf_chart_image(_draw_cantilever, width_mm=145, height_mm=90)

    if chart is not None:
        elements.append(Spacer(1, 4 * mm))
        elements.append(chart)

    # --- CPT profile (qt and Ic vs depth) ---------------------------------
    if processed and processed.get('depth'):
        sampled = sample_processed_profile(processed, max_points=800)

        def _draw_profile(fig):
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(sampled['qt'], sampled['depth'], color='#1a73e8', lw=1.0)
            ax1.set_xlabel('qt (MPa)', fontsize=8)
            ax1.set_ylabel('Depth (m)', fontsize=8)
            ax1.set_xlim(left=0)
            ax1.invert_yaxis()
            ax1.grid(alpha=0.25)
            ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
            ax2.plot(sampled['lc'], sampled['depth'], color='#7b1fa2', lw=1.0)
            ax2.set_xlabel('Ic', fontsize=8)
            ax2.set_xlim(0, 4)
            for boundary in (1.31, 2.05, 2.60, 2.95):
                ax2.axvline(boundary, color='#bbbbbb', lw=0.6, ls=':')
            ax2.grid(alpha=0.25)

        elements.append(Paragraph('CPT Profile Used', heading_style))
        elements.append(_pdf_chart_image(_draw_profile, width_mm=150, height_mm=95))

    # --- Assumptions -------------------------------------------------------
    elements.append(Paragraph('Assumptions', heading_style))
    assumptions = []
    if flow.water_table is not None:
        assumptions.append(f'Groundwater at {flow.water_table} m depth; pore pressure taken as '
                           'hydrostatic below this level (10 kPa per metre).')
    assumptions.append('Cone resistance qt is taken equal to the uploaded qc; no pore-pressure '
                       'correction is applied by the tool.')
    assumptions.append('Soil behaviour is classified from the CPT via the soil behaviour type '
                       'index Ic (Robertson).')
    assumptions.append('The unit weight profile is as supplied in the uploaded file (derived '
                       'from CPT correlations after Robertson &amp; Cabal (2010) when the '
                       'flexible importer generated it).')
    if pile_type in ('driven', 'bored'):
        assumptions.append('Capacities are ultimate (unfactored) values and do not include pile '
                           'or soil plug weights.')
    elif pile_type == 'helical':
        assumptions.append('Capacities are ultimate (unfactored) values; the method is '
                           'calibrated for sand profiles.')
    elif pile_type == 'lateral':
        assumptions.append('The method applies to rigid monopiles in sand profiles under '
                           'monotonic lateral loading.')
    elif pile_type == 'cantilever':
        assumptions.append('The design charts apply to embedded cantilever walls in sand.')
    elif pile_type == 'shallow':
        assumptions.append('Bearing values are net pressures; the load-settlement response '
                           'follows the CPT-based approach for the classified soil type.')
    for a in assumptions:
        elements.append(Paragraph(f'&bull;&nbsp;{a}', normal_style))
        elements.append(Spacer(1, 1 * mm))

    # --- Method references -------------------------------------------------
    elements.append(Paragraph('Method Reference', heading_style))
    for ref in _PDF_REFERENCES.get(pile_type, []):
        elements.append(Paragraph(ref, normal_style))
        elements.append(Spacer(1, 2 * mm))

    # --- Disclaimer --------------------------------------------------------
    elements.append(Paragraph('Important Notice', heading_style))
    elements.append(Paragraph(
        'This report was generated automatically by the UWA CPT Pile Calculator from '
        'user-uploaded CPT data. The results are unfactored, reflect only the information '
        'contained in that data, and are provided as an aid to design. They must be reviewed '
        'and approved by a qualified geotechnical engineer before being relied on in any '
        'design.', normal_style))

    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)

    site_name = _flow_site_name(state)
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{pile_type}_report"
    if site_name:
        filename += f"_{_safe_filename_part(site_name)}"
    filename += f"_{date_str}.pdf"

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


@bp.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    affiliation = request.form.get('affiliation')
    
    if not email or not affiliation:
        flash('Please fill in all fields', 'error')
        return redirect(url_for('main.index'))
        
    ip_addr = request.remote_addr
    country = None
    try:
        import urllib.request, json as _json
        geo = _json.loads(urllib.request.urlopen(
            f'http://ip-api.com/json/{ip_addr}?fields=status,country,city', timeout=3
        ).read())
        if geo.get('status') == 'success':
            parts = [geo.get('city'), geo.get('country')]
            country = ', '.join(p for p in parts if p)
    except Exception:
        pass

    registration = Registration(
        email=email,
        affiliation=affiliation,
        ip_address=ip_addr,
        country=country
    )
    
    db.session.add(registration)
    db.session.commit()
    
    # Set session as permanent and add registration info
    session.permanent = True
    session['registered'] = True
    session['user_email'] = email
    session['affiliation'] = affiliation
    session.modified = True
    
    # Store registration in analytics
    store_analytics_data('registration', 'email', email)
    store_analytics_data('registration', 'affiliation', affiliation)
    try:
        record_event('registration', 'user_registered', {'email': email, 'affiliation': affiliation})
    except Exception:
        pass
    
    # Set a more persistent cookie
    next_url = request.form.get('next') or request.args.get('next') or url_for('main.index')
    response = make_response(redirect(next_url))
    response.set_cookie(
        'user_registered', 
        'true',
        max_age=31536000,  # 365 days in seconds
        httponly=True,
        samesite='Lax',
        path='/'
    )
    return response


@bp.route('/suggestions', methods=['GET', 'POST'])
def suggestions():
    submitted = False
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        category = request.form.get('category', 'general')
        message = request.form.get('message', '').strip()

        if not message:
            flash('Please enter a suggestion.', 'error')
            return redirect(url_for('main.suggestions'))

        suggestion = Suggestion(
            name=name or None,
            email=email or None,
            category=category,
            message=message,
            ip_address=request.remote_addr
        )
        db.session.add(suggestion)
        db.session.commit()
        record_event('suggestion', 'suggestion_submitted', {'category': category})
        submitted = True

    return render_template('suggestions.html', submitted=submitted)


_ADMIN_RATE_LIMIT = 5
_ADMIN_RATE_WINDOW_SECONDS = 60 * 60
_admin_failed_attempts = {}


def _client_ip():
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_password = os.environ.get('ADMIN_PASSWORD')
        if not admin_password:
            logger.error("ADMIN_PASSWORD env var is not set; refusing admin access")
            return Response('Server misconfigured: admin disabled.', 500)

        ip = _client_ip()
        attempts = _admin_failed_attempts.setdefault(ip, deque())
        cutoff = _now() - _ADMIN_RATE_WINDOW_SECONDS
        while attempts and attempts[0] < cutoff:
            attempts.popleft()

        if len(attempts) >= _ADMIN_RATE_LIMIT:
            logger.warning("Admin rate limit hit for ip=%s", ip)
            return Response(
                'Too many failed attempts. Try again later.', 429,
                {'Retry-After': str(_ADMIN_RATE_WINDOW_SECONDS)})

        auth = request.authorization
        supplied = (auth.password or '') if auth else ''
        if not compare_digest(supplied, admin_password):
            attempts.append(_now())
            logger.warning("Failed admin auth from ip=%s (failures in window=%d)", ip, len(attempts))
            return Response(
                'Could not verify your access level for that URL.\n'
                'You have to login with proper credentials', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})

        _admin_failed_attempts.pop(ip, None)
        logger.info("Admin auth succeeded for ip=%s", ip)
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/admin')
@admin_required
def admin():
    # Exclude demo/test registrations
    demo_emails = ['demo@uwa.edu.au']
    real_filter = ~Registration.email.in_(demo_emails)

    # Get all registrations
    registrations = Registration.query.filter(real_filter).order_by(Registration.timestamp.desc()).all()

    # Calculate some basic analytics
    total_users = len(registrations)
    unique_users = db.session.query(func.count(func.distinct(Registration.email))).filter(real_filter).scalar() or 0

    # Get registrations by day - Modified this query
    daily_stats = db.session.query(
        db.func.date(Registration.timestamp).label('date'),
        db.func.count(Registration.id).label('count')
    ).filter(real_filter).group_by(
        db.func.date(Registration.timestamp)
    ).order_by(
        db.func.date(Registration.timestamp).desc()
    ).limit(30).all()

    # Get top affiliations
    top_affiliations = db.session.query(
        Registration.affiliation,
        func.count(Registration.id).label('count')
    ).filter(real_filter).group_by(Registration.affiliation)\
     .order_by(func.count(Registration.id).desc())\
     .limit(10).all()

    # Get visit counts for last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    visit_stats = db.session.query(
        Visit.email,
        func.count(Visit.id).label('visit_count')
    ).filter(
        Visit.timestamp >= thirty_days_ago
    ).group_by(Visit.email)\
     .order_by(func.count(Visit.id).desc())\
     .all()
    
    # Get enhanced analytics data from our new tables
    from .analytics import get_page_visit_stats, get_analytics_data_stats
    
    # Get page visit statistics
    page_visit_stats = get_page_visit_stats(days=30)
    
    # Get analytics data statistics for pile types
    pile_type_stats = get_analytics_data_stats('pile_selection', days=30)
    
    # Get analytics data statistics for pile parameters
    param_stats = get_analytics_data_stats('pile_params', days=30)
    
    # Get advertisement click statistics
    ad_click_stats = get_analytics_data_stats('ad_click', days=30)

    # Build 14-day rolling new-registration series
    # Need 44 days of data so we have a full 14-day window for the last 30 days
    rolling_start = datetime.utcnow() - timedelta(days=44)
    reg_by_day = db.session.query(
        db.func.date(Registration.timestamp).label('date'),
        db.func.count(Registration.id).label('count')
    ).filter(
        Registration.timestamp >= rolling_start,
        real_filter
    ).group_by(
        db.func.date(Registration.timestamp)
    ).all()
    # Convert to dict for quick lookup
    reg_lookup = {str(r.date): r.count for r in reg_by_day}
    # Zero-fill every day in the range
    today = datetime.utcnow().date()
    all_days = [(today - timedelta(days=i)) for i in range(44, -1, -1)]
    daily_counts = [reg_lookup.get(str(d), 0) for d in all_days]
    # Compute 14-day rolling sum
    rolling_14d = []
    for i in range(len(all_days)):
        window = daily_counts[max(0, i - 13):i + 1]
        rolling_14d.append({'date': str(all_days[i]), 'count': sum(window)})
    # Only send the last 30 days of rolling data to the template
    rolling_14d_json = rolling_14d[-30:]

    # Recent user activity feed (last 100 events)
    recent_activity = db.session.query(AnalyticsData).filter(
        AnalyticsData.data_type == 'event'
    ).order_by(AnalyticsData.timestamp.desc()).limit(100).all()

    # Recent users with geo, session duration, browser info
    recent_users = get_recent_users(days=30, limit=30)

    # User journey stats: how far users get in the funnel
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    funnel_steps = {}
    for step_name in ['step_1', 'step_2', 'step_3', 'step_4']:
        count = db.session.query(func.count(func.distinct(AnalyticsData.user_id))).filter(
            AnalyticsData.data_type == 'event',
            AnalyticsData.data_key == 'step_visit',
            AnalyticsData.data_value.contains(step_name),
            AnalyticsData.timestamp >= seven_days_ago
        ).scalar() or 0
        funnel_steps[step_name] = count

    # Download stats (last 30 days)
    download_stats = db.session.query(
        AnalyticsData.data_key,
        func.count(AnalyticsData.id).label('count')
    ).filter(
        AnalyticsData.data_type == 'event',
        AnalyticsData.data_key == 'download',
        AnalyticsData.timestamp >= thirty_days_ago
    ).group_by(AnalyticsData.data_key).all()

    # Download breakdown by type
    download_breakdown = db.session.query(
        AnalyticsData.data_value,
        func.count(AnalyticsData.id).label('count')
    ).filter(
        AnalyticsData.data_type == 'event',
        AnalyticsData.data_key == 'download',
        AnalyticsData.timestamp >= thirty_days_ago
    ).group_by(AnalyticsData.data_value).order_by(func.count(AnalyticsData.id).desc()).all()

    # Recent suggestions
    try:
        recent_suggestions = Suggestion.query.order_by(Suggestion.timestamp.desc()).limit(20).all()
    except Exception:
        db.session.rollback()
        recent_suggestions = []

    return render_template('admin.html',
                         registrations=registrations,
                         total_users=total_users,
                         unique_users=unique_users,
                         daily_stats=daily_stats,
                         top_affiliations=top_affiliations,
                         visit_stats=visit_stats,
                         page_visit_stats=page_visit_stats,
                         pile_type_stats=pile_type_stats,
                         param_stats=param_stats,
                         ad_click_stats=ad_click_stats,
                         rolling_14d_json=rolling_14d_json,
                         recent_activity=recent_activity,
                         recent_users=recent_users,
                         funnel_steps=funnel_steps,
                         download_breakdown=download_breakdown,
                         recent_suggestions=recent_suggestions)

@bp.route('/admin/user/<user_id>')
@admin_required
def admin_user_detail(user_id):
    """Show detailed info for a specific user session."""
    details = get_user_details(user_id)
    if not details:
        flash('User not found')
        return redirect(url_for('main.admin'))
    return render_template('admin_user.html', user=details)


@bp.route('/admin/send_weekly_report')
@admin_required
def send_weekly_report():
    from .email_utils import send_weekly_usage_email
    try:
        send_weekly_usage_email()
        flash('Weekly usage report sent')
    except Exception as e:
        current_app.logger.error(f"Weekly report send failed: {e}")
        flash(f'Failed to send: {e}', 'error')
    return redirect(url_for('main.admin'))

@bp.route('/admin/export')
@admin_required
def export_registrations():
    demo_emails = ['demo@uwa.edu.au']
    registrations = Registration.query.filter(~Registration.email.in_(demo_emails)).order_by(Registration.timestamp.desc()).all()
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Email', 'Affiliation', 'IP Address', 'Timestamp'])
    
    for reg in registrations:
        cw.writerow([reg.email, reg.affiliation, reg.ip_address, 
                    reg.timestamp.strftime('%Y-%m-%d %H:%M:%S')])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=registrations.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@bp.route('/<type>/description')
def pile_description(type):
    if type not in ['driven', 'bored', 'helical', 'shallow', 'lateral', 'cantilever']:
        return redirect(url_for('main.index'))
    if type == 'shallow':
        _maybe_grant_shallow_demo()
        if not _shallow_demo_allowed():
            return redirect(url_for('main.index'))
    if type in ('lateral', 'cantilever'):
        _maybe_grant_shallow_demo()
        if not _private_module_allowed():
            return redirect(url_for('main.index'))
    return render_template(f'{type}/description.html', type=type)

@bp.route('/download_intermediary_calcs')
def download_intermediary_calcs():
    record_event('download', 'download_intermediary', {
        'pile_type': session.get('type'),
        'email': session.get('user_email'),
    })
    """Download intermediary calculations used for graphs as CSV"""
    flow = _flow_row()
    data = _flow_cpt(flow)
    if not data:
        flash('No CPT data available')
        return redirect(url_for('main.index'))

    try:
        # Profile computed once at upload; recompute only for legacy flows
        processed = data.get('processed') or pre_input_calc(data, float(flow.water_table or 0))

        if not processed:
            flash('Error processing data')
            return redirect(url_for('main.index'))
        
        # Create DataFrame with all intermediate calculations
        df = pd.DataFrame({
            'Depth (m)': processed['depth'],
            'qc (MPa)': processed['qc'],
            'qt (MPa)': processed['qt'],
            'fs (kPa)': processed['fs'],
            'Unit Weight (kN/m³)': processed['gtot'],
            'Water Pressure u0 (kPa)': processed['u0_kpa'],
            'Total Vertical Stress σv0 (kPa)': processed['sig_v0'],
            'Effective Vertical Stress σv0\' (kPa)': processed['sig_v0_prime'],
            'Fr (%)': processed['fr_percent'],
            'Normalized Tip Resistance Qtn': processed['qtn'],
            'Stress Exponent n': processed['n'],
            'Soil Behavior Type Index Ic': processed['lc'],
            'Pore Pressure Ratio Bq': processed['bq'],
            'Correction Factor Kc': processed['kc'],
            'Corrected Tip Resistance qtc (MPa)': processed['qtc'],
            'Soil Behavior Index Iz': processed['iz1']
        })
        
        # Get the current date in DDMMYYYY format
        current_date = datetime.now().strftime('%d%m%Y')

        # Use original filename if available
        filename = flow.original_filename or ''
        if filename:
            base_name = _safe_filename_part(os.path.splitext(filename)[0])
            download_name = f"{base_name}_intermediary_calcs_{current_date}.csv"
        else:
            download_name = f"intermediary_calcs_{current_date}.csv"

        return send_file(
            io.BytesIO(df.to_csv(index=False).encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        logger.error(f"Error generating intermediary calculations: {str(e)}")
        flash('Error generating calculations')
        return redirect(url_for('main.calculator_step', type=flow.calc_type if flow else 'driven', step=2))

@bp.route('/download_helical_calculations')
def download_helical_calculations():
    record_event('download', 'download_helical_calcs', {
        'email': session.get('user_email'),
    })
    """Download all helical pile calculation data in CSV format"""
    try:
        # Get the calculation results from the session
        if 'results_id' not in session:
            flash('No calculation results found', 'error')
            return redirect(url_for('main.index'))
        
        # Load the calculation results
        results = load_calculation_results(session['results_id'])
        if not results:
            flash('Calculation results not found', 'error')
            return redirect(url_for('main.index'))
        
        # Debug logging - print the keys in the results dictionary
        current_app.logger.info(f"Results keys: {list(results.keys())}")
        
        # Save summary back to session to ensure it's available for download_results
        if 'summary' in results:
            session['results'] = results['summary']
            current_app.logger.info("Stored summary results back in session")
        
        # Get pile parameters
        pile_params = {}
        if 'pile_params' in results:
            pile_params = results['pile_params']
        elif 'detailed' in results and 'input_parameters' in results['detailed']:
            pile_params = results['detailed']['input_parameters']
        elif 'summary' in results and isinstance(results['summary'], dict) and 'tipdepth' in results['summary']:
            # Try to extract from session if not in results
            pile_params = session.get('pile_params', {})
        
        # Debug logging
        current_app.logger.info(f"Downloading helical pile calculations")
        current_app.logger.info(f"Pile params: {pile_params}")
        
        # Get the detailed results
        detailed_results = {}
        if 'detailed' in results:
            detailed_results = results['detailed']
        elif 'detailed_results' in results:
            detailed_results = results['detailed_results']
        
        # Debug logging - print the keys in the detailed_results dictionary
        if detailed_results:
            current_app.logger.info(f"Detailed results keys: {list(detailed_results.keys())}")
        else:
            current_app.logger.warning("No detailed results found")
        
        # Get the user's filename or use a default
        user_filename = pile_params.get('site_name', '')
        if not user_filename:
            user_filename = "helical_pile_calculations"
        current_app.logger.info(f"Using original filename: {user_filename}")
        
        # Clean the filename
        user_filename = ''.join(c for c in user_filename if c.isalnum() or c in '._- ')
        user_filename = user_filename.strip()
        if not user_filename:
            user_filename = "helical_pile_calculations"
        current_app.logger.info(f"Final user_filename: {user_filename}")
        
        # Create a timestamp for the filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%d%m%Y")
        download_name = f"{user_filename}_detailed_{timestamp}.csv"
        current_app.logger.info(f"Final download_name: {download_name}")
        
        # Create a CSV from the data
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # If we have pre-formatted download data, use it
        if 'download_data' in results and results['download_data']:
            current_app.logger.info("Using pre-formatted download data")
            for row in results['download_data']:
                writer.writerow(row)
        else:
            # Otherwise, build the CSV from detailed results
            current_app.logger.info("Building CSV from detailed results")
            
            # Create header row
            header = [
                "Depth (m)",
                "qt (MPa)",
                "qc (MPa)",
                "fs (kPa)",
                "Fr (%)",
                "Ic",
                "Soil Type",
                "q1 (MPa)",
                "q10 (MPa)",
                "Casing Coefficient",
                "Delta Z (m)",
                "Shaft Segment (kN)",
                "Cumulative Shaft (kN)",
                "Tension Capacity (kN)",
                "Compression Capacity (kN)"
            ]
            writer.writerow(header)
            
            # Check if we have depth data
            if not detailed_results or 'depth' not in detailed_results or not detailed_results['depth']:
                current_app.logger.error("No depth data found in detailed results")
                writer.writerow(["No detailed calculation data available"])
            else:
                # Add data rows - only up to helix depth for helical piles
                helix_depth = pile_params.get('helix_depth', None)
                for i in range(len(detailed_results['depth'])):
                    # For helical piles, stop adding rows once we reach or exceed the helix depth
                    if helix_depth and detailed_results['depth'][i] > float(helix_depth):
                        break
                        
                    try:
                        row = [
                            detailed_results['depth'][i],
                            detailed_results.get('qt', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('qc', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('fs', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('fr_percent', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('lc', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('soil_type', ['Unknown'] * len(detailed_results['depth']))[i],
                            detailed_results.get('q1', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('q10', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('coe_casing', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('delta_z', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('qshaft_segment', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('qshaft_kn', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('tension_capacity', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('compression_capacity', [0] * len(detailed_results['depth']))[i]
                        ]
                        writer.writerow(row)
                    except (IndexError, KeyError) as e:
                        current_app.logger.error(f"Error writing row {i}: {str(e)}")
                        continue
            
            # Add empty row for spacing
            writer.writerow([])
            
            # Add summary information
            writer.writerow(["SUMMARY INFORMATION"])
            writer.writerow(["Input Parameters"])
            for key, value in pile_params.items():
                writer.writerow([key, value])
            
            # Add geometric constants if available
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Geometric Constants"])
            writer.writerow(["Perimeter (m)", detailed_results.get('perimeter', 'N/A')])
            writer.writerow(["Helix Area (m²)", detailed_results.get('helix_area', 'N/A')])
            
            # Add helix information if available
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Helix Information"])
            writer.writerow(["Helix Depth (m)", pile_params.get('helix_depth', 'N/A')])
            writer.writerow(["q1 at Helix", detailed_results.get('q1_helix', 'N/A')])
            writer.writerow(["q10 at Helix", detailed_results.get('q10_helix', 'N/A')])
            writer.writerow(["Helix Tension Component (kN)", detailed_results.get('qhelix_tension', 'N/A')])
            writer.writerow(["Helix Compression Component (kN)", detailed_results.get('qhelix_compression', 'N/A')])
            
            # Add effective depth calculations if available
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Effective Depth Calculations"])
            writer.writerow(["Tension Effective Depth (m)", detailed_results.get('tension_effective_depth', 'N/A')])
            writer.writerow(["Tension Min q10", detailed_results.get('tension_min_q10', 'N/A')])
            writer.writerow(["q(10mm) Tension", detailed_results.get('q_10mm_tens', 'N/A')])
            writer.writerow(["Compression Effective Depth (m)", detailed_results.get('compression_effective_depth', 'N/A')])
            writer.writerow(["Compression Min q10", detailed_results.get('compression_min_q10', 'N/A')])
            writer.writerow(["q(10mm) Compression", detailed_results.get('q_10mm_comp', 'N/A')])
            
            # Add final results
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Final Results"])
            
            # Get summary data from different possible locations
            summary_data = {}
            if 'summary' in results:
                summary_data = results['summary']
            elif 'summary_data' in results:
                summary_data = results['summary_data']
            elif 'summary_data' in session:
                summary_data = session['summary_data']
            
            # Write final results
            writer.writerow(["Ultimate Tension Capacity (kN)", summary_data.get('qult_tension', detailed_results.get('qult_tension', 'N/A'))])
            writer.writerow(["Ultimate Compression Capacity (kN)", summary_data.get('qult_compression', detailed_results.get('qult_compression', 'N/A'))])
            writer.writerow(["Tension Capacity at 10mm (kN)", detailed_results.get('q_delta_10mm_tension', 'N/A')])
            writer.writerow(["Compression Capacity at 10mm (kN)", detailed_results.get('q_delta_10mm_compression', 'N/A')])
            writer.writerow(["Installation Torque (kNm)", detailed_results.get('installation_torque', 'N/A')])
        
        # Prepare the file for download
        output.seek(0)
        
        # Add more debug logging right before sending the file
        current_app.logger.info(f"Sending file with size: {len(output.getvalue())} bytes")
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_name
        )
    
    except Exception as e:
        # Log the full exception with traceback
        import traceback
        current_app.logger.error(f"Error in download_helical_calculations: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        
        # Flash a more helpful error message
        flash(f"Error generating download: {str(e)}", 'error')
        return redirect(url_for('main.index'))