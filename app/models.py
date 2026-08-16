from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Registration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    affiliation = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    country = db.Column(db.String(100), nullable=True)

class Visit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45)) 

# New models for enhanced analytics
class PageVisit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=True)  # Can be null for anonymous users
    user_id = db.Column(db.String(120), nullable=True)  # For tracking anonymous users with a session ID
    page_url = db.Column(db.String(255), nullable=False)
    referrer = db.Column(db.String(255), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    session_id = db.Column(db.String(120), nullable=True)
    # Classified at write time so the dashboard can segment people from
    # crawlers cheaply; older rows are backfilled from the stored user agent.
    is_bot = db.Column(db.Boolean, nullable=True)
    # ISO country code from Cloudflare's CF-IPCountry header (null before
    # Aug 2026 and for requests that bypass Cloudflare).
    country = db.Column(db.String(8), nullable=True)
    
class AnalyticsData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(120), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    data_type = db.Column(db.String(50), nullable=False)  # E.g., 'calc_params', 'pile_type', etc.
    data_key = db.Column(db.String(100), nullable=True)
    data_value = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    session_id = db.Column(db.String(120), nullable=True)

class Suggestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    category = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45), nullable=True)

class CalcFlow(db.Model):
    """One wizard run (upload through results), persisted server-side.

    Replaces the OS-temp-file storage behind the old cpt_data_id/debug_id:
    rows survive Render redeploys (the instance filesystem is wiped on every
    deploy), and the flow id travels in the wizard URLs so two browser tabs
    can run independent calculations without clobbering each other.

    ``cpt_payload`` is written once at upload and holds the raw CPT rows
    plus the processed profile from pre_input_calc (zlib-compressed JSON),
    so the profile is computed exactly once per upload. ``state_payload``
    holds the smaller step-3/4 state (parameters, results, capacity
    envelope, debug details) and is rewritten as the wizard advances.
    """
    id = db.Column(db.String(32), primary_key=True)
    anon_id = db.Column(db.String(64), nullable=False, index=True)
    calc_type = db.Column(db.String(20), nullable=False)
    water_table = db.Column(db.Float, nullable=True)
    original_filename = db.Column(db.String(255), nullable=True)
    # Deferred so listing/existence checks don't drag the blobs across the wire
    cpt_payload = db.deferred(db.Column(db.LargeBinary, nullable=True))
    state_payload = db.deferred(db.Column(db.LargeBinary, nullable=True))
    history_fp = db.Column(db.String(40), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)


class SavedCalculation(db.Model):
    """A completed calculation saved for the no-login history feature.

    Keyed to the anonymous browser id (uwa_anon_id cookie). ``payload`` is a
    zlib-compressed JSON snapshot of everything needed to rebuild the wizard
    session (CPT rows, parameters, results, debug details); ``summary_json``
    is a small uncompressed extract for rendering the history list without
    touching the payload.
    """
    id = db.Column(db.Integer, primary_key=True)
    anon_id = db.Column(db.String(64), nullable=False, index=True)
    fingerprint = db.Column(db.String(40), nullable=False, index=True)
    calc_type = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(200), nullable=True)
    summary_json = db.Column(db.Text, nullable=True)
    payload = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)