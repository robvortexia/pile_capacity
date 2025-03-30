from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Registration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    affiliation = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45)) 

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
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_id = db.Column(db.String(120), nullable=True)
    
class AnalyticsData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(120), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    data_type = db.Column(db.String(50), nullable=False)  # E.g., 'calc_params', 'pile_type', etc.
    data_key = db.Column(db.String(100), nullable=True)
    data_value = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    session_id = db.Column(db.String(120), nullable=True) 