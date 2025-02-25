from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .models import db
from datetime import timedelta

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///registrations.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Make sure session is permanent and lasts for 1 year
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
    app.config['SESSION_PERMANENT'] = True
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        from .routes import bp as main_blueprint
        app.register_blueprint(main_blueprint)
    
    return app 