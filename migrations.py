"""
Database migration script for setting up the PostgreSQL database on Render.

Run this script after setting up your PostgreSQL database on Render and configuring
the DATABASE_URL environment variable in your web service.

Usage: python migrations.py
"""

import os
from flask import Flask
from app.models import db, Registration, Visit, PageVisit, AnalyticsData

def create_app():
    app = Flask(__name__)
    
    # Configure database
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    
    # Handle deprecated postgres:// URI format (Render uses postgresql://)
    if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)
    
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    return app

def create_tables():
    """Create all the database tables"""
    app = create_app()
    
    with app.app_context():
        # Create tables
        db.create_all()
        print("Database tables created successfully!")

def migrate_from_sqlite():
    """Migrate data from SQLite to PostgreSQL if needed"""
    # This function would contain code to migrate existing data
    # from a SQLite database to PostgreSQL if needed
    # For simplicity, we're not implementing that now
    pass

if __name__ == '__main__':
    create_tables()
    # Uncomment if you need to migrate data from SQLite
    # migrate_from_sqlite() 