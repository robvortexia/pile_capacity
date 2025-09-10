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