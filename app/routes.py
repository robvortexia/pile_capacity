from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session, Response
from werkzeug.utils import secure_filename
import pandas as pd
import os
import math
import json
import plotly.graph_objects as go
import plotly.utils
import numpy as np
from .utils import save_cpt_data, load_cpt_data, create_cpt_graphs, save_graphs_data, load_graphs_data, generate_csv_download, save_debug_details, load_debug_details, create_bored_pile_graphs
from .calculations import calculate_pile_capacity, process_cpt_data, pre_input_calc, get_iterative_values
from datetime import datetime, timedelta
from .models import db, Registration, Visit
from functools import wraps
import csv
from io import StringIO
from sqlalchemy.sql import func

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/')
@bp.route('/index')
def index():
    show_modal = True
    if 'registered' in session:
        show_modal = False
        # Log visit if user is registered
        email = session.get('user_email')
        if email:
            visit = Visit(
                email=email,
                ip_address=request.remote_addr
            )
            db.session.add(visit)
            db.session.commit()
    return render_template('index.html', show_modal=show_modal)

@bp.route('/calculator/<type>/step/<int:step>', methods=['GET', 'POST'])
def calculator_step(type, step):
    if type not in ['driven', 'bored']:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        if step == 1:  # Handle file upload
            if 'cpt_file' not in request.files:
                flash('No file selected')
                return redirect(request.url)
            
            water_table = request.form.get('water_table')
            if not water_table:
                flash('Water table depth is required')
                return redirect(request.url)
            
            try:
                water_table = float(water_table)
            except ValueError:
                flash('Water table depth must be a number')
                return redirect(request.url)
            
            file = request.files['cpt_file']
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                try:
                    content = file.read().decode('utf-8')
                    
                    first_line = content.split('\n')[0]
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif ',' in first_line:
                        delimiter = ','
                    else:
                        delimiter = ' '
                    
                    from io import StringIO
                    file_obj = StringIO(content)
                    
                    df = pd.read_csv(file_obj, delimiter=delimiter, header=None, 
                                   names=['z', 'qc', 'fs', 'gtot'],
                                   skipinitialspace=True)
                    
                    for col in ['z', 'qc', 'fs', 'gtot']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if df.isnull().values.any():
                        flash('File contains invalid numeric data')
                        return redirect(request.url)
                    
                    data_dict = df.to_dict('records')
                    processed_data = process_cpt_data(data_dict)
                    
                    file_id = save_cpt_data(processed_data['cpt_data'], water_table)
                    session['cpt_data_id'] = file_id
                    session['water_table'] = water_table
                    
                    return redirect(url_for('main.calculator_step', type=type, step=2))
                    
                except pd.errors.EmptyDataError:
                    flash('The uploaded file is empty')
                    return redirect(request.url)
                except Exception as e:
                    flash(f'Error processing file: {str(e)}')
                    return redirect(request.url)
            else:
                flash('Invalid file type. Please upload a CSV or TXT file.')
                return redirect(request.url)
        
        elif step == 2:  # Handle CPT data acceptance
            if 'cpt_data_id' not in session:
                flash('No CPT data available. Please upload data first.')
                return redirect(url_for('main.calculator_step', type=type, step=1))
            
            data = load_cpt_data(session['cpt_data_id'])
            if not data:
                flash('CPT data not found. Please upload data again.')
                return redirect(url_for('main.calculator_step', type=type, step=1))
                
            processed_data = pre_input_calc(data, data['water_table'])
            if not processed_data:
                flash('Error processing CPT data. Please check your input data.')
                return redirect(url_for('main.calculator_step', type=type, step=1))
                
            return redirect(url_for('main.calculator_step', type=type, step=3))
        
        elif step == 3:  # Handle pile parameters
            try:
                print("Form data:", request.form)
                
                # Different parameter handling based on pile type
                if type == 'bored':
                    params = {
                        'site_name': request.form.get('site_name', ''),
                        'shaft_diameter': float(request.form['shaft_diameter']),
                        'base_diameter': float(request.form['base_diameter']),
                        'borehole_depth': float(request.form['borehole_depth']),
                        'pile_tip_depths': [float(d.strip()) for d in request.form['pile_tip_depths'].split(',')]
                    }
                else:  # driven pile parameters
                    params = {
                        'pile_shape': request.form['pile_shape'],
                        'pile_end_condition': request.form['pile_end_condition'],
                        'pile_diameter': float(request.form['pile_diameter']),
                        'borehole_depth': float(request.form['borehole_depth']),
                        'pile_tip_depths': [float(d.strip()) for d in request.form['pile_tip_depths'].split(',')]
                    }
                    
                    if request.form['pile_end_condition'] == 'open':
                        params['wall_thickness'] = float(request.form['wall_thickness'])
                
                print("Processed parameters:", params)
                
                if 'cpt_data_id' not in session:
                    flash('CPT data not found in session. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                data = load_cpt_data(session['cpt_data_id'])
                if not data:
                    flash('CPT data could not be loaded. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                print("Calculating pile capacity with data:", data)
                
                # Pass the pile type to calculate_pile_capacity
                calc_result = calculate_pile_capacity(data, params, pile_type=type)
                
                if not calc_result:
                    flash('Error in calculation. Please check your inputs.')
                    return redirect(request.url)
                
                session['results'] = calc_result
                return redirect(url_for('main.calculator_step', type=type, step=4))
                
            except ValueError as e:
                flash(f'Invalid value entered: {str(e)}')
                return redirect(request.url)
            except Exception as e:
                flash(f'Error calculating pile capacity: {str(e)}')
                print(f"Error in step 3: {str(e)}")
                return redirect(request.url)
    
    # Handle GET requests
    if step == 2:
        if 'cpt_data_id' not in session:
            flash('No CPT data available. Please upload data first.')
            return redirect(url_for('main.calculator_step', type=type, step=1))
        
        data = load_cpt_data(session['cpt_data_id'])
        if not data:
            flash('CPT data not found. Please upload data again.')
            return redirect(url_for('main.calculator_step', type=type, step=1))
        
        if type == 'bored':
            graphs = create_bored_pile_graphs(data)
        else:
            graphs = create_cpt_graphs(data)
        return render_template(f'{type}/steps.html', step=step, graphs=graphs, type=type)
    
    elif step == 3:
        if 'cpt_data_id' not in session:
            flash('No CPT data available. Please complete previous steps first.')
            return redirect(url_for('main.calculator_step', type=type, step=1))
        return render_template(f'{type}/steps.html', step=step, type=type)
        
    elif step == 4:
        if 'results' not in session:
            flash('No results available. Please complete the analysis first.')
            return redirect(url_for('main.calculator_step', type=type, step=3))
        
        debug_id = session.get('debug_id')
        if debug_id:
            debug_details = load_debug_details(debug_id)
        else:
            debug_details = {}
        
        return render_template(f'{type}/steps.html', step=step, type=type, results=session['results'], debug_details=debug_details)
    
    return render_template(f'{type}/steps.html', step=step, type=type)

@bp.route('/download_debug_params')
def download_debug_params():
    """Download debug parameters and calculation data as CSV"""
    if 'cpt_data_id' not in session:
        flash('No CPT data available')
        return redirect(url_for('main.index'))
    
    try:
        data = load_cpt_data(session['cpt_data_id'])
        if not data:
            flash('CPT data not found')
            return redirect(url_for('main.index'))
        
        processed = pre_input_calc(data)
        if not processed:
            flash('No processed data available')
            return redirect(url_for('main.index'))
        
        debug_id = session.get('debug_id')
        if debug_id:
            debug_details = load_debug_details(debug_id)
        else:
            debug_details = {}

        # If we have debug details, pick one scenario (e.g., the first tip) to include
        if debug_details:
            first_tip = list(debug_details.keys())[0]
            dbg = debug_details[first_tip]
        else:
            dbg = {}

        df = pd.DataFrame({
            'depth': processed['depth'],
            'h': processed['h'],
            'qc': processed['qc'],
            'qt': processed['qt'],
            'gtot': processed['gtot'],
            'u0_kpa': processed['u0_kpa'],
            'sig_v0': processed['sig_v0'],
            'sig_v0_prime': processed['sig_v0_prime'],
            'fs': processed['fs'],
            'fr_percent': processed['fr_percent'],
            'qtn': processed['qtn'],
            'n': processed['n'],
            'lc': processed['lc'],
            'bq': processed['bq'],
            'kc': processed['kc'],
            'iz1': processed['iz1'],
            'qtc': processed['qtc']
        })

        for key in ['tf_sand', 'tf_clay', 'tf_adop_tension', 'tf_adop_compression', 
                    'qs_tension', 'qs_compression', 'qb_sand', 'qb_clay', 'qb_final']:
            if key in dbg and len(dbg[key]) == len(df):
                df[key] = dbg[key]
        
        return generate_csv_download(
            df,
            f"debug_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
    except Exception as e:
        flash(f'Error generating debug data: {str(e)}')
        return redirect(url_for('main.calculator_step', type='driven', step=4))

@bp.route('/download_results')
def download_results():
    """Download calculation results as CSV"""
    if 'results' not in session:
        flash('No results available')
        return redirect(url_for('main.index'))
    
    try:
        res_list = session['results']
        df = pd.DataFrame(res_list)
        return generate_csv_download(
            df,
            f"calculation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    except Exception as e:
        flash(f'Error generating results: {str(e)}')
        return redirect(url_for('main.calculator_step', type='driven', step=4))

@bp.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    affiliation = request.form.get('affiliation')
    
    if not email or not affiliation:
        flash('Please fill in all fields', 'error')
        return redirect(url_for('main.index'))
        
    registration = Registration(
        email=email,
        affiliation=affiliation,
        ip_address=request.remote_addr
    )
    
    db.session.add(registration)
    db.session.commit()
    
    session['registered'] = True
    session['user_email'] = email  # Store email in session
    session.permanent = True
    return redirect(url_for('main.index'))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_password = 'barry2002'  # Changed password
        auth = request.authorization
        if not auth or auth.password != admin_password:
            return Response(
                'Could not verify your access level for that URL.\n'
                'You have to login with proper credentials', 401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated_function

@bp.route('/admin')
@admin_required
def admin():
    # Get all registrations
    registrations = Registration.query.order_by(Registration.timestamp.desc()).all()
    
    # Calculate some basic analytics
    total_users = len(registrations)
    
    # Get registrations by day - Modified this query
    daily_stats = db.session.query(
        db.func.date(Registration.timestamp).label('date'),
        db.func.count(Registration.id).label('count')
    ).group_by(
        db.func.date(Registration.timestamp)
    ).order_by(
        db.func.date(Registration.timestamp).desc()
    ).limit(30).all()

    # Get top affiliations
    top_affiliations = db.session.query(
        Registration.affiliation,
        func.count(Registration.id).label('count')
    ).group_by(Registration.affiliation)\
     .order_by(func.count(Registration.id).desc())\
     .limit(10).all()

    # Get visit counts for last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    visit_stats = db.session.query(
        Visit.email,
        func.count(Visit.id).label('visit_count')
    ).filter(
        Visit.timestamp >= thirty_days_ago
    ).group_by(Visit.email)\
     .order_by(func.count(Visit.id).desc())\
     .all()

    return render_template('admin.html', 
                         registrations=registrations,
                         total_users=total_users,
                         daily_stats=daily_stats,
                         top_affiliations=top_affiliations,
                         visit_stats=visit_stats)

@bp.route('/admin/export')
@admin_required
def export_registrations():
    registrations = Registration.query.order_by(Registration.timestamp.desc()).all()
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Email', 'Affiliation', 'IP Address', 'Timestamp'])
    
    for reg in registrations:
        cw.writerow([reg.email, reg.affiliation, reg.ip_address, 
                    reg.timestamp.strftime('%Y-%m-%d %H:%M:%S')])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=registrations.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@bp.route('/<type>/description')
def pile_description(type):
    if type not in ['driven', 'bored']:
        return redirect(url_for('main.index'))
    return render_template(f'{type}/description.html', type=type)
