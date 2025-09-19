from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from .models import db
from datetime import timedelta
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def _init_scheduler(app):
    """Initialize APScheduler for weekly email reports if enabled."""
    if not app.config.get('ENABLE_SCHEDULER', False):
        return None
    scheduler = BackgroundScheduler(timezone=app.config.get('SCHEDULER_TIMEZONE', 'UTC'))
    # Every Monday at 07:00 UTC by default
    cron = app.config.get('WEEKLY_REPORT_CRON', {'day_of_week': 'mon', 'hour': 7, 'minute': 0})

    def _send_weekly():
        from .email_utils import send_weekly_usage_email
        with app.app_context():
            try:
                send_weekly_usage_email()
            except Exception as e:
                app.logger.error(f"Weekly report failed: {e}")

    scheduler.add_job(_send_weekly, CronTrigger(**cron), id='weekly_usage_email', replace_existing=True)
    scheduler.start()
    return scheduler

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Use PostgreSQL database if DATABASE_URL is provided, otherwise use SQLite
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///registrations.db')
    
    # Handle deprecated postgres:// URI format (Render uses postgresql://)
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)
    
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Session configuration - updated for better persistence
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
    app.config['SESSION_PERMANENT'] = True
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../flask_session')
    app.config['SESSION_USE_SIGNER'] = True
    app.config['SESSION_KEY_PREFIX'] = 'uwa_pile_calc_'
    
    # Initialize Flask-Session
    Session(app)
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        from .routes import bp as main_blueprint
        app.register_blueprint(main_blueprint)

        # Scheduler (optional)
        _init_scheduler(app)
    
    return app 