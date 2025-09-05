import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size (increased for large CPT files)
    SESSION_TYPE = 'filesystem'
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year
    # Extended timeout for large file processing
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    # Request timeout settings for large dataset processing
    REQUEST_TIMEOUT = 300  # 5 minutes for large dataset processing
    # Large dataset processing settings
    LARGE_DATASET_THRESHOLD = 1000  # Points above which to apply optimizations
    GRAPH_SUBSAMPLE_SIZE = 500  # Target number of points for graph display