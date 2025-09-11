import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size 
    SESSION_TYPE = 'filesystem'
    
    # Performance settings for large file processing
    GRAPH_SAMPLE_SIZE = 1000  # Maximum points to use for graph visualization
    REQUEST_TIMEOUT = 300  # 5 minutes for large file processing
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year cache for static files

    # Email (SMTP) configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USE_SSL = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')

    # Weekly report recipients (comma-separated)
    WEEKLY_REPORT_RECIPIENTS = os.environ.get('WEEKLY_REPORT_RECIPIENTS', '')
    WEEKLY_REPORT_DAYS = int(os.environ.get('WEEKLY_REPORT_DAYS', 7))
    SITE_NAME = os.environ.get('SITE_NAME', 'UWA Pile Calculator')

    # Scheduler toggle and timing
    ENABLE_SCHEDULER = os.environ.get('ENABLE_SCHEDULER', 'false').lower() == 'true'
    SCHEDULER_TIMEZONE = os.environ.get('SCHEDULER_TIMEZONE', 'UTC')
    # Cron can be overridden via env pieces; default mon 07:00
    WEEKLY_REPORT_CRON = {
        'day_of_week': os.environ.get('WEEKLY_REPORT_CRON_DOW', 'mon'),
        'hour': int(os.environ.get('WEEKLY_REPORT_CRON_HOUR', 7)),
        'minute': int(os.environ.get('WEEKLY_REPORT_CRON_MIN', 0)),
    }