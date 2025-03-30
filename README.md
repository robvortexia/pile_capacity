# UWA Pile Capacity Calculator

A Flask-based web application for calculating pile capacity using CPT data.

## Setup Instructions

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the development server: `python run.py`

## Database Configuration

The application now supports PostgreSQL for persistent analytics data across deployments.

### Local Development

By default, the application uses SQLite for local development. No additional configuration is needed.

### Deployment on Render

To set up the PostgreSQL database on Render:

1. Create a new PostgreSQL database in your Render dashboard
2. Add the `DATABASE_URL` environment variable to your web service:
   - Go to your web service's dashboard
   - Click on "Environment" in the left sidebar
   - Add a new environment variable with key `DATABASE_URL` and the value of your database's "Internal Database URL"
3. After deploying your application, run the database migrations:
   - Connect to your web service shell via the Render dashboard
   - Run: `python migrations.py`

## Analytics Features

The application now stores analytics data in the database, including:

- Page visits
- User registrations
- Pile type selections
- Calculation parameters and results

This data persists across deployments and can be viewed in the admin dashboard.