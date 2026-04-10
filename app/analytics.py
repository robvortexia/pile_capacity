import json
import uuid
import urllib.request
import urllib.error
from flask import request, session
from datetime import datetime, timedelta
from sqlalchemy import func
from .models import db, PageVisit, AnalyticsData, Visit, Registration
from typing import List, Dict, Any, Optional

# In-memory cache so we don't re-lookup the same IP during one app lifecycle
_geo_cache: Dict[str, Dict[str, str]] = {}

def get_or_create_user_id():
    """Get user ID from session or create a new one if it doesn't exist"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

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
            
        page_visit = PageVisit(
            email=email,
            user_id=user_id,
            page_url=page_url,
            referrer=referrer,
            user_agent=request.user_agent.string,
            ip_address=request.remote_addr,
            session_id=session.get('_id')  # Flask-Session ID if available
        )
        
        db.session.add(page_visit)
        db.session.commit()
        
        # Also record in the original Visit model for backward compatibility
        if email:
            visit = Visit(
                email=email,
                ip_address=request.remote_addr
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
        'ip': request.remote_addr if request else None,
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
            'country': geo.get('country', ''),
            'city': geo.get('city', ''),
            'referrer': row.referrer or 'Direct',
            'email': row.email,
            'browser_os': _parse_ua_short(row.ua or ''),
        })

    return users


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
