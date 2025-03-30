import json
import uuid
from flask import request, session
from datetime import datetime, timedelta
from sqlalchemy import func
from .models import db, PageVisit, AnalyticsData, Visit, Registration

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