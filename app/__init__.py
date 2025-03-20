from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from .models import db
from datetime import timedelta
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///registrations.db'
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
    
    return app 