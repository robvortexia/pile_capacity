import json
import re
import uuid
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from flask import request, session
from datetime import datetime, timedelta
from sqlalchemy import func
from .models import db, PageVisit, AnalyticsData, Visit, Registration
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Bot / client classification, shared by the visit recorder and the dashboard.
# ---------------------------------------------------------------------------

BOT_UA_RE = re.compile(
    r'bot|crawl|spider|slurp|curl|wget|python|httpx|aiohttp|requests|headless|'
    r'phantom|scrapy|go-http|java/|libwww|okhttp|facebookexternalhit|semrush|'
    r'ahrefs|mj12|dotbot|petalbot|yandex|baidu|gptbot|oai-searchbot|claudebot|'
    r'claude-web|perplexity|bytespider|ccbot|amazonbot|applebot|duckduckbot|'
    r'bingpreview|dataforseo|serpstat|screaming|zoominfo|censys|expanse|'
    r'monitor|uptime|feedfetcher|preview',
    re.I)

_AGENT_FAMILIES = [
    ('Googlebot', 'googlebot'), ('Google (other)', 'google-'),
    ('Bingbot', 'bingbot'), ('Bing preview', 'bingpreview'),
    ('GPTBot (OpenAI)', 'gptbot'), ('OAI-SearchBot', 'oai-searchbot'),
    ('ChatGPT-User', 'chatgpt'), ('ClaudeBot (Anthropic)', 'claude'),
    ('PerplexityBot', 'perplexity'), ('AhrefsBot', 'ahrefs'),
    ('SemrushBot', 'semrush'), ('Bytespider (TikTok)', 'bytespider'),
    ('Amazonbot', 'amazonbot'), ('Applebot', 'applebot'),
    ('YandexBot', 'yandex'), ('Baidu', 'baidu'), ('PetalBot', 'petalbot'),
    ('MJ12bot', 'mj12'), ('DotBot', 'dotbot'), ('CCBot', 'ccbot'),
    ('DuckDuckBot', 'duckduckbot'), ('Facebook', 'facebook'),
    ('python scripts', 'python'), ('curl / wget', 'curl'), ('curl / wget', 'wget'),
    ('Go clients', 'go-http'), ('Scrapy', 'scrapy'),
    ('Headless browsers', 'headless'),
]


def classify_user_agent(ua):
    """Return (is_bot, family_label) for a raw user-agent string.

    An empty user agent counts as a bot: no real browser sends one, and the
    empty-UA bucket was the single biggest source of the August 2026 traffic
    spike."""
    if not ua:
        return True, '(no user agent)'
    low = ua.lower()
    for label, needle in _AGENT_FAMILIES:
        if needle in low:
            return True, label
    if BOT_UA_RE.search(ua):
        return True, 'Other bots'
    return False, _parse_ua_short(ua)


def _valid_ip(value):
    """Parse a candidate IP; None for anything that isn't one. Header values
    are client-forgeable junk on requests that bypass Cloudflare, and the
    ip_address columns are VARCHAR(45), so never store unvalidated text."""
    import ipaddress
    try:
        return str(ipaddress.ip_address((value or '').strip()))
    except ValueError:
        return None


def real_client_ip():
    """Best-effort real client IP, for analytics only.

    Cloudflare fronts the site, so its CF-Connecting-IP header is the
    authoritative client address; request.remote_addr is just a Cloudflare
    edge. Fall back to the first X-Forwarded-For hop, then the socket."""
    if not request:
        return None
    for candidate in (request.headers.get('CF-Connecting-IP'),
                      request.headers.get('X-Forwarded-For', '').split(',')[0]):
        ip = _valid_ip(candidate)
        if ip:
            return ip
    return request.remote_addr


def request_country():
    """ISO country code for this request, from Cloudflare's geolocation
    header. None when the request bypassed Cloudflare (direct Render URL,
    local dev) or the location is unknown."""
    if not request:
        return None
    code = (request.headers.get('CF-IPCountry') or '').strip().upper()
    if not code or code in ('XX', 'T1'):
        return None
    return code[:8]

# In-memory cache so we don't re-lookup the same IP during one app lifecycle
_geo_cache: Dict[str, Dict[str, str]] = {}

_ANON_COOKIE_RE = re.compile(r'^[0-9a-fA-F-]{8,64}$')


def get_or_create_user_id():
    """Stable id for analytics without forcing a server-side session.

    Prefers an existing session id (registered users / legacy sessions),
    then the long-lived uwa_anon_id cookie. Only when neither exists is a
    fresh id minted, and it is handed to the anon-cookie machinery via
    flask.g rather than written to the session: a session write here would
    create a server-side session file for every anonymous page view.
    """
    if 'user_id' in session:
        return session['user_id']
    cookie = request.cookies.get('uwa_anon_id', '')
    if _ANON_COOKIE_RE.match(cookie):
        return cookie
    from flask import g
    pending = getattr(g, 'anon_id_new', None)
    if pending:
        return pending
    aid = str(uuid.uuid4())
    g.anon_id_new = aid  # persisted as the uwa_anon_id cookie by routes._attach_anon_cookie
    return aid

def record_page_visit(page_url=None, referrer=None):
    """Record a page visit to the database"""
    try:
        email = session.get('user_email')
        user_id = get_or_create_user_id()
        
        # If page_url is not provided, use the current request path
        if page_url is None:
            page_url = request.path
            
        # If referrer is not provided, use the request referrer
        if referrer is None:
            referrer = request.referrer

        ua = request.user_agent.string
        is_bot, _family = classify_user_agent(ua)
        page_visit = PageVisit(
            email=email,
            user_id=user_id,
            page_url=page_url,
            referrer=referrer,
            user_agent=ua,
            ip_address=real_client_ip(),
            is_bot=is_bot,
            country=request_country(),
            session_id=session.get('_id')  # Flask-Session ID if available
        )

        db.session.add(page_visit)

        # Also record in the original Visit model for backward compatibility
        if email:
            visit = Visit(
                email=email,
                ip_address=real_client_ip()
            )
            db.session.add(visit)
        db.session.commit()
            
        return True
    except Exception as e:
        print(f"Error recording page visit: {str(e)}")
        db.session.rollback()
        return False

def store_analytics_data(data_type, data_key=None, data_value=None, data_dict=None):
    """Store analytics data in the database
    
    Args:
        data_type (str): Type of data (e.g., 'calc_params', 'pile_type')
        data_key (str, optional): Key for the data
        data_value (any, optional): Value for the data, will be converted to string/JSON
        data_dict (dict, optional): Dictionary of key-value pairs to store (multiple entries)
    """
    try:
        email = session.get('user_email')
        user_id = get_or_create_user_id()
        
        if data_dict:
            # Store multiple key-value pairs
            for key, value in data_dict.items():
                # Convert non-string values to JSON
                if not isinstance(value, str):
                    value = json.dumps(value)
                    
                analytics_data = AnalyticsData(
                    email=email,
                    user_id=user_id,
                    data_type=data_type,
                    data_key=key,
                    data_value=value,
                    session_id=session.get('_id')
                )
                db.session.add(analytics_data)
        else:
            # Store single key-value pair
            if not isinstance(data_value, str):
                data_value = json.dumps(data_value)
                
            analytics_data = AnalyticsData(
                email=email,
                user_id=user_id,
                data_type=data_type,
                data_key=data_key,
                data_value=data_value,
                session_id=session.get('_id')
            )
            db.session.add(analytics_data)
            
        db.session.commit()
        return True
    except Exception as e:
        print(f"Error storing analytics data: {str(e)}")
        db.session.rollback()
        return False

def get_page_visit_stats(days=30):
    """Get page visit statistics for the last N days"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Total visits by page
    page_stats = db.session.query(
        PageVisit.page_url,
        func.count(PageVisit.id).label('visit_count')
    ).filter(
        PageVisit.timestamp >= start_date
    ).group_by(
        PageVisit.page_url
    ).order_by(
        func.count(PageVisit.id).desc()
    ).all()
    
    # Visits by day
    daily_stats = db.session.query(
        func.date(PageVisit.timestamp).label('date'),
        func.count(PageVisit.id).label('count')
    ).filter(
        PageVisit.timestamp >= start_date
    ).group_by(
        func.date(PageVisit.timestamp)
    ).order_by(
        func.date(PageVisit.timestamp)
    ).all()
    
    # Unique visitors
    unique_visitors = db.session.query(
        func.count(func.distinct(PageVisit.user_id))
    ).filter(
        PageVisit.timestamp >= start_date
    ).scalar()
    
    return {
        'page_stats': page_stats,
        'daily_stats': daily_stats,
        'daily_stats_json': [{'date': str(row[0]), 'count': int(row[1])} for row in daily_stats],
        'unique_visitors': unique_visitors
    }

def get_analytics_data_stats(data_type=None, days=30):
    """Get statistics for analytics data"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    query = db.session.query(
        AnalyticsData.data_type,
        AnalyticsData.data_key,
        func.count(AnalyticsData.id).label('count')
    ).filter(
        AnalyticsData.timestamp >= start_date
    )
    
    if data_type:
        query = query.filter(AnalyticsData.data_type == data_type)
        
    stats = query.group_by(
        AnalyticsData.data_type,
        AnalyticsData.data_key
    ).order_by(
        AnalyticsData.data_type,
        func.count(AnalyticsData.id).desc()
    ).all()
    
    return stats 


def record_event(event_type: str, event_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> bool:
    """Record a fine-grained user event to `AnalyticsData`.

    Args:
        event_type: High-level category, e.g., 'upload', 'calculation', 'download'.
        event_name: Specific name, e.g., 'upload_cpt', 'calc_bored', 'download_results_csv'.
        details: Optional payload; stored as JSON string.
    """
    payload = {
        'name': event_name,
        'path': request.path if request else None,
        'method': request.method if request else None,
        'ip': real_client_ip(),
        'ua': request.user_agent.string if request else None,
        'ref': request.referrer if request else None,
        'details': details or {}
    }
    return store_analytics_data('event', data_key=event_type, data_value=payload)


def get_weekly_usage_summary(days: int = 7) -> Dict[str, Any]:
    """Build a weekly usage summary across key tables for the last `days`.

    Returns a dict of simple primitives and small lists safe to render/email.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Totals
    total_page_visits = db.session.query(func.count(PageVisit.id)).filter(PageVisit.timestamp >= start_date).scalar() or 0
    unique_visitors = db.session.query(func.count(func.distinct(PageVisit.user_id))).filter(PageVisit.timestamp >= start_date).scalar() or 0
    total_registrations = db.session.query(func.count(Registration.id)).filter(Registration.timestamp >= start_date).scalar() or 0

    # Top pages
    top_pages_rows = db.session.query(
        PageVisit.page_url,
        func.count(PageVisit.id).label('count')
    ).filter(
        PageVisit.timestamp >= start_date
    ).group_by(PageVisit.page_url).order_by(func.count(PageVisit.id).desc()).limit(10).all()
    top_pages = [{'page': r[0], 'count': int(r[1])} for r in top_pages_rows]

    # Event breakdown
    event_rows = db.session.query(
        AnalyticsData.data_key,  # event_type
        func.count(AnalyticsData.id).label('count')
    ).filter(
        AnalyticsData.timestamp >= start_date,
        AnalyticsData.data_type == 'event'
    ).group_by(AnalyticsData.data_key).order_by(func.count(AnalyticsData.id).desc()).all()
    events = [{'event_type': r[0], 'count': int(r[1])} for r in event_rows]

    # Pile type selections
    pile_rows = get_analytics_data_stats('pile_selection', days=days)
    pile_types = [{'type': r[1], 'count': int(r[2])} for r in pile_rows]

    return {
        'range': {'start': start_date.isoformat() + 'Z', 'end': end_date.isoformat() + 'Z'},
        'totals': {
            'page_visits': int(total_page_visits),
            'unique_visitors': int(unique_visitors),
            'registrations': int(total_registrations)
        },
        'top_pages': top_pages,
        'events': events,
        'pile_types': pile_types,
    }


def lookup_ip_geo(ip: str) -> Dict[str, str]:
    """Look up country/city for an IP using ip-api.com (free, no key needed).
    Results are cached in memory so we only call the API once per IP per app restart."""
    if not ip or ip in ('127.0.0.1', '::1', 'unknown'):
        return {'country': 'Local', 'city': '', 'isp': ''}

    if ip in _geo_cache:
        return _geo_cache[ip]

    try:
        url = f'http://ip-api.com/json/{ip}?fields=status,country,city,isp'
        req = urllib.request.Request(url, headers={'User-Agent': 'UWA-CPT-Calculator'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get('status') == 'success':
                result = {
                    'country': data.get('country', ''),
                    'city': data.get('city', ''),
                    'isp': data.get('isp', ''),
                }
            else:
                result = {'country': '', 'city': '', 'isp': ''}
    except Exception:
        result = {'country': '', 'city': '', 'isp': ''}

    _geo_cache[ip] = result
    return result


def get_user_details(user_id: str) -> Dict[str, Any]:
    """Build a detailed profile for a given user_id from existing analytics data."""
    # All page visits for this user
    visits = PageVisit.query.filter_by(user_id=user_id).order_by(PageVisit.timestamp).all()

    if not visits:
        return None

    first_visit = visits[0]
    last_visit = visits[-1]
    duration_seconds = (last_visit.timestamp - first_visit.timestamp).total_seconds()

    # Get their IP and geo info
    ip = first_visit.ip_address or ''
    geo = lookup_ip_geo(ip)

    # Parse user agent for a readable browser/OS string
    ua = first_visit.user_agent or ''
    browser_os = _parse_ua_short(ua)

    # Get their events (parameters, uploads, downloads)
    events = AnalyticsData.query.filter_by(
        user_id=user_id, data_type='event'
    ).order_by(AnalyticsData.timestamp).all()

    # Check if they registered
    reg = Registration.query.filter_by(ip_address=ip).first() if ip else None

    return {
        'user_id': user_id,
        'ip': ip,
        'geo': geo,
        'browser_os': browser_os,
        'referrer': first_visit.referrer or 'Direct',
        'email': first_visit.email or (reg.email if reg else None),
        'affiliation': reg.affiliation if reg else None,
        'first_seen': first_visit.timestamp,
        'last_seen': last_visit.timestamp,
        'duration_minutes': round(duration_seconds / 60, 1),
        'page_count': len(visits),
        'pages': [{'url': v.page_url, 'time': v.timestamp} for v in visits],
        'events': events,
    }


def get_recent_users(days: int = 7, limit: int = 30) -> list:
    """Get a summary list of recent unique users with geo and session info."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Get distinct user_ids from recent page visits
    user_rows = db.session.query(
        PageVisit.user_id,
        func.min(PageVisit.timestamp).label('first_seen'),
        func.max(PageVisit.timestamp).label('last_seen'),
        func.count(PageVisit.id).label('page_count'),
        func.min(PageVisit.ip_address).label('ip'),
        func.min(PageVisit.referrer).label('referrer'),
        func.min(PageVisit.email).label('email'),
        func.min(PageVisit.user_agent).label('ua'),
        func.max(PageVisit.country).label('country_code'),
    ).filter(
        PageVisit.timestamp >= cutoff,
        PageVisit.user_id.isnot(None)
    ).group_by(
        PageVisit.user_id
    ).order_by(
        func.max(PageVisit.timestamp).desc()
    ).limit(limit).all()

    users = []
    for row in user_rows:
        ip = row.ip or ''
        geo = lookup_ip_geo(ip)
        duration = (row.last_seen - row.first_seen).total_seconds()
        users.append({
            'user_id': row.user_id,
            'first_seen': row.first_seen,
            'last_seen': row.last_seen,
            'duration_minutes': round(duration / 60, 1),
            'page_count': row.page_count,
            'ip': ip,
            # Prefer the Cloudflare-recorded country; older rows fall back to
            # a geo lookup of the stored IP.
            'country': country_name(row.country_code) if row.country_code else geo.get('country', ''),
            'city': geo.get('city', ''),
            'referrer': row.referrer or 'Direct',
            'email': row.email,
            'browser_os': _parse_ua_short(row.ua or ''),
        })

    return users


_COUNTRY_NAMES = {
    'AU': 'Australia', 'US': 'United States', 'GB': 'United Kingdom',
    'NZ': 'New Zealand', 'CA': 'Canada', 'IE': 'Ireland', 'IN': 'India',
    'DE': 'Germany', 'FR': 'France', 'NL': 'Netherlands', 'BE': 'Belgium',
    'CH': 'Switzerland', 'AT': 'Austria', 'IT': 'Italy', 'ES': 'Spain',
    'PT': 'Portugal', 'SE': 'Sweden', 'NO': 'Norway', 'DK': 'Denmark',
    'FI': 'Finland', 'PL': 'Poland', 'GR': 'Greece', 'TR': 'Turkey',
    'SG': 'Singapore', 'MY': 'Malaysia', 'ID': 'Indonesia', 'TH': 'Thailand',
    'VN': 'Vietnam', 'PH': 'Philippines', 'CN': 'China', 'HK': 'Hong Kong',
    'TW': 'Taiwan', 'JP': 'Japan', 'KR': 'South Korea', 'PK': 'Pakistan',
    'BD': 'Bangladesh', 'LK': 'Sri Lanka', 'AE': 'UAE', 'SA': 'Saudi Arabia',
    'QA': 'Qatar', 'ZA': 'South Africa', 'NG': 'Nigeria', 'KE': 'Kenya',
    'EG': 'Egypt', 'BR': 'Brazil', 'MX': 'Mexico', 'AR': 'Argentina',
    'CL': 'Chile', 'PE': 'Peru', 'CO': 'Colombia',
}


def country_name(code):
    if not code:
        return 'Unknown'
    return _COUNTRY_NAMES.get(code, code)


# Events that mean someone genuinely used the calculator (a successful CPT
# upload, a completed calculation, a results download) rather than just
# loading pages. Crawler-triggered events are filtered by user agent.
_ENGAGED_EVENT_KEYS = ('upload_success', 'calculation', 'download')


def _as_date(value):
    """func.date() returns a date on Postgres but a string on SQLite."""
    if isinstance(value, str):
        return datetime.strptime(value[:10], '%Y-%m-%d').date()
    return value


def get_audience_stats(days: int = 30) -> Dict[str, Any]:
    """People vs bots vs genuinely-engaged users: daily series, headline
    numbers, country split and bot families for the admin dashboard."""
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    day_list = []
    d = start.date()
    while d <= now.date():
        day_list.append(d)
        d += timedelta(days=1)

    daily = {d: {'date': str(d), 'human_views': 0, 'bot_views': 0,
                 'dau': 0, 'engaged': 0} for d in day_list}

    # Views per day, split human/bot. Legacy rows with is_bot NULL
    # (pre-backfill) count as human, matching the old behaviour.
    rows = (db.session.query(
                func.date(PageVisit.timestamp),
                PageVisit.is_bot,
                func.count(PageVisit.id))
            .filter(PageVisit.timestamp >= start)
            .group_by(func.date(PageVisit.timestamp), PageVisit.is_bot)
            .all())
    for day, bot_flag, views in rows:
        key = _as_date(day)
        if key not in daily:
            continue
        if bot_flag:
            daily[key]['bot_views'] += int(views)
        else:
            daily[key]['human_views'] += int(views)

    # Distinct human visitors per day. Grouped by date only: grouping by
    # (date, is_bot) would count a user twice on a day where they have both
    # a legacy NULL row and a freshly classified False row.
    dau_rows = (db.session.query(
                    func.date(PageVisit.timestamp),
                    func.count(func.distinct(PageVisit.user_id)))
                .filter(PageVisit.timestamp >= start,
                        PageVisit.is_bot.isnot(True))
                .group_by(func.date(PageVisit.timestamp))
                .all())
    for day, users in dau_rows:
        key = _as_date(day)
        if key in daily:
            daily[key]['dau'] = int(users)

    # Engaged users per day, from the event payloads (excluding crawler hits
    # on download links).
    ev_rows = (db.session.query(AnalyticsData.timestamp, AnalyticsData.user_id,
                                AnalyticsData.data_value)
               .filter(AnalyticsData.timestamp >= start,
                       AnalyticsData.data_type == 'event',
                       AnalyticsData.data_key.in_(_ENGAGED_EVENT_KEYS))
               .all())
    engaged_by_day = defaultdict(set)
    engaged_uids = set()
    for ts, uid, val in ev_rows:
        try:
            ua = (json.loads(val) or {}).get('ua') or ''
        except Exception:
            ua = ''
        if ua and classify_user_agent(ua)[0]:
            continue
        key = uid or 'anon'
        engaged_by_day[ts.date()].add(key)
        engaged_uids.add(key)
    for d in day_list:
        if d in daily:
            daily[d]['engaged'] = len(engaged_by_day.get(d, ()))

    daily_series = [daily[d] for d in day_list]

    # Headline numbers. "Yesterday" is the last full UTC day.
    last7 = [daily[d] for d in day_list if d >= (now.date() - timedelta(days=7)) and d < now.date()]
    yesterday = daily.get(now.date() - timedelta(days=1), {})
    # Same window as the 7-day view tiles: full UTC days ending yesterday.
    engaged_7d = set()
    for d, uids in engaged_by_day.items():
        if d >= now.date() - timedelta(days=7) and d < now.date():
            engaged_7d |= uids
    totals = {
        'human_views_7d': sum(x['human_views'] for x in last7),
        'bot_views_7d': sum(x['bot_views'] for x in last7),
        'dau_yesterday': yesterday.get('dau', 0),
        'engaged_7d': len(engaged_7d),
        'engaged_window': len(engaged_uids),
        'human_views_window': sum(x['human_views'] for x in daily_series),
        'bot_views_window': sum(x['bot_views'] for x in daily_series),
        'registrations_window': db.session.query(func.count(Registration.id))
                                  .filter(Registration.timestamp >= start).scalar() or 0,
    }

    # Country split for human traffic. Country is recorded per visit from
    # Cloudflare from Aug 2026; older rows show as Unknown.
    eng_country = Counter()
    if engaged_uids:
        uid_rows = (db.session.query(PageVisit.user_id, PageVisit.country,
                                     func.count(PageVisit.id))
                    .filter(PageVisit.user_id.in_(list(engaged_uids)),
                            PageVisit.country.isnot(None))
                    .group_by(PageVisit.user_id, PageVisit.country).all())
        best = {}
        for uid, code, n in uid_rows:
            if uid not in best or n > best[uid][1]:
                best[uid] = (code, n)
        for code, _n in best.values():
            eng_country[code] += 1

    c_rows = (db.session.query(PageVisit.country,
                               func.count(PageVisit.id),
                               func.count(func.distinct(PageVisit.user_id)))
              .filter(PageVisit.timestamp >= start, PageVisit.is_bot.isnot(True))
              .group_by(PageVisit.country)
              .order_by(func.count(PageVisit.id).desc())
              .all())
    countries = [{
        'code': code or '',
        'name': country_name(code),
        'views': int(views),
        'visitors': int(visitors),
        'engaged': int(eng_country.get(code, 0)) if code else 0,
    } for code, views, visitors in c_rows]

    # Bot families: classify each distinct UA string once.
    ua_rows = (db.session.query(PageVisit.user_agent, func.count(PageVisit.id))
               .filter(PageVisit.timestamp >= start, PageVisit.is_bot.is_(True))
               .group_by(PageVisit.user_agent).all())
    fam = Counter()
    for ua, n in ua_rows:
        fam[classify_user_agent(ua or '')[1]] += int(n)
    bot_families = [{'family': f, 'views': c} for f, c in fam.most_common(12)]

    return {
        'daily': daily_series,
        'totals': totals,
        'countries': countries,
        'bot_families': bot_families,
        'window_days': days,
    }


def _parse_ua_short(ua: str) -> str:
    """Extract a short browser/OS label from a user agent string."""
    ua_lower = ua.lower()
    # Browser
    browser = 'Other'
    if 'edg/' in ua_lower:
        browser = 'Edge'
    elif 'chrome/' in ua_lower and 'chromium' not in ua_lower:
        browser = 'Chrome'
    elif 'firefox/' in ua_lower:
        browser = 'Firefox'
    elif 'safari/' in ua_lower and 'chrome' not in ua_lower:
        browser = 'Safari'
    # OS
    os_name = ''
    if 'windows' in ua_lower:
        os_name = 'Windows'
    elif 'mac os' in ua_lower or 'macintosh' in ua_lower:
        os_name = 'Mac'
    elif 'linux' in ua_lower:
        os_name = 'Linux'
    elif 'android' in ua_lower:
        os_name = 'Android'
    elif 'iphone' in ua_lower or 'ipad' in ua_lower:
        os_name = 'iOS'

    if os_name:
        return f'{browser} / {os_name}'
    return browser
