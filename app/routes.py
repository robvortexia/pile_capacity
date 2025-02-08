from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session, Response, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import os
import math
import json
import plotly.graph_objects as go
import plotly.utils
import numpy as np
from .utils import save_cpt_data, load_cpt_data, create_cpt_graphs, save_graphs_data, load_graphs_data, generate_csv_download, save_debug_details, load_debug_details, create_bored_pile_graphs
from .calculations import calculate_pile_capacity, process_cpt_data, pre_input_calc, get_iterative_values, calculate_bored_pile_results
from datetime import datetime, timedelta
from .models import db, Registration, Visit
from functools import wraps
import csv
from io import StringIO
from sqlalchemy.sql import func
import logging
import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
            
            # Get water table value first
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
                    logger.debug("Starting file processing")
                    # Store original filename in session (without extension)
                    original_filename = os.path.splitext(secure_filename(file.filename))[0]
                    session['original_filename'] = original_filename
                    
                    # Initialize data structures
                    data_dict = []
                    delimiter = None
                    
                    # Process file line by line
                    for line in file:
                        try:
                            decoded_line = line.decode('utf-8').strip()
                            if not decoded_line:  # Skip empty lines
                                continue
                            
                            # Try to determine the delimiter from first line
                            if delimiter is None:
                                if '\t' in decoded_line:
                                    delimiter = '\t'
                                elif ',' in decoded_line:
                                    delimiter = ','
                                else:
                                    delimiter = ' '
                            
                            # Split and process each line directly
                            values = decoded_line.split(delimiter)
                            if len(values) >= 4:  # Ensure we have all required columns
                                try:
                                    data_dict.append({
                                        'z': float(values[0]),
                                        'qc': float(values[1]),
                                        'fs': float(values[2]),
                                        'gtot': float(values[3])
                                    })
                                except (ValueError, IndexError):
                                    continue
                        except UnicodeDecodeError:
                            continue
                    
                    if not data_dict:
                        flash('No valid data found in file')
                        return redirect(request.url)
                    
                    processed_data = process_cpt_data(data_dict)
                    logger.debug("process_cpt_data completed")
                    
                    # Save processed data
                    file_id = save_cpt_data(processed_data['cpt_data'], water_table)
                    session['cpt_data_id'] = file_id
                    session['water_table'] = water_table
                    
                    logger.debug(f"File name: {file.filename}")
                    logger.debug(f"Content type: {file.content_type}")
                    logger.debug(f"Detected delimiter: {delimiter}")
                    
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
            
            # Verify the data exists and is valid
            data = load_cpt_data(session['cpt_data_id'])
            if not data:
                flash('CPT data not found. Please upload data again.')
                return redirect(url_for('main.calculator_step', type=type, step=1))
                
            # Process the data to ensure it's valid
            processed_data = pre_input_calc(data, data['water_table'])
            if not processed_data:
                flash('Error processing CPT data. Please check your input data.')
                return redirect(url_for('main.calculator_step', type=type, step=1))
                
            return redirect(url_for('main.calculator_step', type=type, step=3))
        
        elif step == 3:
            try:
                # Debug logging
                print("Form data received:", request.form)
                
                # Store the pile parameters in session based on pile type
                if type == 'bored':
                    session['type'] = 'bored'
                    session['pile_params'] = {
                        'file_name': request.form.get('file_name', ''),
                        'shaft_diameter': float(request.form.get('shaft_diameter')),
                        'base_diameter': float(request.form.get('base_diameter')),
                        'cased_depth': float(request.form.get('cased_depth')),
                        'water_table': float(request.form.get('water_table', 0)),
                        'tip_depths': [float(d.strip()) for d in request.form.get('tip_depths', '').split(',')],
                        'borehole_depth': float(request.form.get('cased_depth')),
                        'pile_shape': 0,  # Always circular for bored piles
                        'pile_end_condition': 0,  # Always open-ended for bored piles
                        'pile_tip_depths': [float(d.strip()) for d in request.form.get('tip_depths', '').split(',')]
                    }
                else:  # driven pile
                    session['type'] = 'driven'
                    session['pile_params'] = {
                        'site_name': request.form.get('site_name', ''),
                        'pile_end_condition': request.form.get('pile_end_condition'),
                        'pile_shape': request.form.get('pile_shape'),
                        'pile_diameter': float(request.form.get('pile_diameter')),
                        'wall_thickness': float(request.form.get('wall_thickness', 0)),
                        'borehole_depth': float(request.form.get('borehole_depth')),
                        'pile_tip_depths': [float(d.strip()) for d in request.form.get('pile_tip_depths', '').split(',')],
                        'water_table': float(session.get('water_table', 0))
                    }
                
                print("Pile params stored in session:", session['pile_params'])
                
                # Calculate results using the parameters and CPT data
                if 'cpt_data_id' not in session:
                    flash('No CPT data found. Please go back to step 1.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please go back to step 1.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                print("About to calculate pile capacity")
                if type == 'bored':
                    water_table = float(session['pile_params']['water_table'])
                    processed_cpt = pre_input_calc(cpt_data, water_table)
                    results = calculate_bored_pile_results(processed_cpt, session['pile_params'])
                    
                    # Store only summary in session, save detailed results to storage
                    session['results'] = results['summary']
                    debug_id = save_debug_details(results['detailed'])
                    session['debug_id'] = debug_id
                    results_id = save_graphs_data(results['detailed'])
                    session['detailed_results_id'] = results_id
                else:
                    results = calculate_pile_capacity(cpt_data, session['pile_params'], pile_type=type)
                    session['results'] = results
                
                return redirect(url_for('main.calculator_step', type=type, step=4))
                
            except Exception as e:
                print("Error occurred:", str(e))
                flash(f'Error calculating results: {str(e)}')
                return redirect(url_for('main.calculator_step', type=type, step=3))
        
        elif step == 4:
            if 'results' not in session:
                flash('No results available. Please complete the analysis first.')
                return redirect(url_for('main.calculator_step', type=type, step=3))
            
            detailed_results = None
            if type == 'bored' and 'detailed_results_id' in session:
                detailed_results = load_graphs_data(session['detailed_results_id'])
            
            return render_template(
                f'{type}/steps.html', 
                step=step, 
                type=type, 
                results=session['results'],
                detailed_results=detailed_results
            )
        
        return render_template(f'{type}/steps.html', step=step, type=type)
    
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
            water_table = float(session.get('water_table', 0))
            graphs = create_cpt_graphs(data, water_table)
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
        
        detailed_results = None
        if type == 'bored' and 'detailed_results_id' in session:
            detailed_results = load_graphs_data(session['detailed_results_id'])
        
        return render_template(
            f'{type}/steps.html', 
            step=step, 
            type=type, 
            results=session['results'],
            detailed_results=detailed_results
        )
    
    return render_template(f'{type}/steps.html', step=step, type=type)

@bp.route('/download_debug_params')
def download_debug_params():
    """Download debug parameters and calculation data as CSV"""
    if 'cpt_data_id' not in session:
        flash('No CPT data available')
        return redirect(url_for('main.index'))
    
    try:
        # Add debug logging
        print("Session contents:", dict(session))
        print("Pile type:", session.get('type'))
        print("Pile params:", session.get('pile_params'))
        
        data = load_cpt_data(session['cpt_data_id'])
        if not data:
            flash('CPT data not found')
            return redirect(url_for('main.index'))
        
        # Get pile parameters from session
        pile_params = session.get('pile_params', {})
        print("Retrieved pile_params:", pile_params)
        
        # Process the CPT data
        water_table = float(session['pile_params']['water_table'])
        processed = pre_input_calc(data, water_table)
        
        # Create a string buffer
        buffer = io.StringIO()
        
        debug_id = session.get('debug_id')
        if debug_id:
            debug_details = load_debug_details(debug_id)
            if debug_details and len(debug_details) > 0:
                
                for tip_index, tip_detail in enumerate(debug_details):
                    if tip_index > 0:
                        # Add separator between different tip depth data
                        buffer.write('\n\n' + '='*50 + '\n\n')
                    
                    # Create constants list with tip depth included
                    constants = [
                        ['Tip Depth (m)', tip_detail['tip_depth']],
                        ['Water table depth (m)', float(session['pile_params']['water_table'])]
                    ]
                    
                    # Add pile type specific parameters
                    if session.get('type') == 'driven':
                        constants.extend([
                            ['Pile type', 'Driven'],
                            ['Pile end condition', pile_params.get('pile_end_condition', 'N/A')],
                            ['Pile shape', pile_params.get('pile_shape', 'N/A')],
                            ['Pile diameter/width (m)', pile_params.get('pile_diameter', 'N/A')],
                            ['Wall thickness (mm)', pile_params.get('wall_thickness', 'N/A')],
                            ['Borehole depth (m)', pile_params.get('borehole_depth', 'N/A')]
                        ])
                    else:
                        constants.extend([
                            ['Pile type', 'Bored'],
                            ['Shaft diameter (m)', pile_params.get('shaft_diameter', 'N/A')],
                            ['Base diameter (m)', pile_params.get('base_diameter', 'N/A')],
                            ['Cased depth (m)', pile_params.get('cased_depth', 'N/A')]
                        ])
                    
                    # Write constants
                    df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                    buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_detail["tip_depth"]}m\n')
                    df_constants.to_csv(buffer, index=False)
                    
                    # Process calculations for this tip depth
                    calcs = tip_detail['calculations']
                    calc_dict = {calc['depth']: calc for calc in calcs}
                    
                    # Create and populate DataFrame
                    df_data = create_data_dataframe(processed, calc_dict)
                    
                    # Add a blank line between constants and data
                    buffer.write('\nCPT DATA AND CALCULATIONS\n')
                    df_data.to_csv(buffer, index=False)
        
        # Prepare the file for download
        buffer.seek(0)
        
        # Get the current date in DDMMYYYY format
        current_date = datetime.now().strftime('%d%m%Y')
        
        # Get the user's filename from session, default to original filename if not found
        pile_params = session.get('pile_params', {})
        if session.get('type') == 'driven':
            user_filename = pile_params.get('site_name', '')
        else:
            user_filename = pile_params.get('file_name', '')

        # If no user filename, try to use the original uploaded filename
        if not user_filename:
            user_filename = session.get('original_filename', '')
            print("Using original filename:", user_filename)

        print("Final user_filename:", user_filename)
        
        # Create the filename, including the user's input if it exists
        if user_filename:
            download_name = f"detailed_output_{user_filename}_{current_date}.csv"
        else:
            download_name = f"detailed_output_{current_date}.csv"
            
        print("Final download_name:", download_name)
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_name
        )
        
    except Exception as e:
        print(f"Debug download error: {str(e)}")
        flash(f'Error generating debug data: {str(e)}')
        return redirect(url_for('main.calculator_step', type='bored', step=4))

def create_data_dataframe(processed, calc_dict):
    """Helper function to create the data DataFrame with calculations"""
    df_data = pd.DataFrame({
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
    
    # Add calculation columns
    df_data['coe_casing'] = float('nan')
    df_data['qb01_adop'] = float('nan')
    df_data['tf_tension'] = float('nan')
    df_data['tf_compression'] = float('nan')
    df_data['qs_tension_segment'] = float('nan')
    df_data['qs_compression_segment'] = float('nan')
    df_data['qs_tension_cumulative'] = float('nan')
    df_data['qs_compression_cumulative'] = float('nan')
    
    # Fill in calculations where they exist
    for idx, row in df_data.iterrows():
        depth = row['depth']
        if depth in calc_dict:
            calc = calc_dict[depth]
            for col in ['coe_casing', 'qb01_adop', 'tf_tension', 'tf_compression', 
                       'qs_tension_segment', 'qs_compression_segment', 
                       'qs_tension_cumulative', 'qs_compression_cumulative']:
                df_data.at[idx, col] = calc[col]
    
    return df_data

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
