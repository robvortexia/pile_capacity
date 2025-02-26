from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session, Response, send_file, current_app
from werkzeug.utils import secure_filename
import pandas as pd
import os
import math
import json
import plotly.graph_objects as go
import plotly.utils
import numpy as np
from .utils import (
    save_cpt_data, load_cpt_data, create_cpt_graphs, 
    save_graphs_data, load_graphs_data, generate_csv_download, 
    save_debug_details, load_debug_details, create_bored_pile_graphs,
    save_calculation_results, load_calculation_results
)
from .calculations import calculate_pile_capacity, process_cpt_data, pre_input_calc, get_iterative_values, calculate_bored_pile_results, calculate_helical_pile_results
from datetime import datetime, timedelta
from .models import db, Registration, Visit
from functools import wraps
import csv
from io import StringIO
from sqlalchemy.sql import func
import logging
import io
from .helical_calculations import calculate_helical_pile_results

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

@bp.route('/<type>/calculator/<int:step>', methods=['GET', 'POST'])
def calculator_step(type, step):
    # Check if we're switching pile types
    if 'type' in session and session['type'] != type:
        # Store registration data
        registered = session.get('registered')
        user_email = session.get('user_email')
        
        # Store CPT data
        cpt_data_id = session.get('cpt_data_id')
        original_filename = session.get('original_filename')
        water_table = session.get('water_table')
        
        # Clear the session
        session.clear()
        
        # Restore registration data
        if registered:
            session['registered'] = registered
            session['user_email'] = user_email
        
        # Restore CPT data
        if cpt_data_id:
            session['cpt_data_id'] = cpt_data_id
        if original_filename:
            session['original_filename'] = original_filename
        if water_table:
            session['water_table'] = water_table
    
    # Set the current pile type
    session['type'] = type
    
    if type not in ['driven', 'bored', 'helical']:
        return redirect(url_for('main.index'))
    
    # Handle helical pile processing specifically
    if type == 'helical' and step == 3 and request.method == 'POST':
        try:
            # Get parameters from form
            pile_params = {
                'site_name': request.form.get('site_name', ''),
                'shaft_diameter': float(request.form.get('shaft_diameter', 0)),
                'helix_diameter': float(request.form.get('helix_diameter', 0)),
                'helix_depth': float(request.form.get('helix_depth', 0)),
                'borehole_depth': float(request.form.get('borehole_depth', 0)),
                'water_table': float(request.form.get('water_table', 0))
            }
            
            # Store parameters in session
            session['pile_params'] = pile_params
            
            # Load CPT data
            if 'cpt_data_id' in session:
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.', 'error')
                    return redirect(url_for('main.calculator_step', type='helical', step=1))
                
                # Process the CPT data
                water_table = float(pile_params['water_table'])
                processed_cpt = pre_input_calc(cpt_data, water_table)
                
                # Calculate results
                try:
                    results = calculate_helical_pile_results(processed_cpt, pile_params)
                    
                    # DON'T store entire results in session - just store a reference ID
                    results_id = save_calculation_results(results)
                    session['results_id'] = results_id
                    
                    # Only store basic summary data in session
                    session['summary_data'] = {
                        'tipdepth': results['summary']['tipdepth'],
                        'qshaft': results['summary']['qshaft'],
                        'qult_tension': results['summary']['qult_tension'],
                        'qult_compression': results['summary']['qult_compression']
                    }
                    
                    # Redirect to step 4
                    return redirect(url_for('main.calculator_step', type='helical', step=4))
                except Exception as e:
                    logger.error(f"Error in helical pile calculations: {str(e)}")
                    flash(f'Error in calculation: {str(e)}', 'error')
                    return redirect(url_for('main.calculator_step', type='helical', step=3))
            else:
                flash('No CPT data available', 'error')
                return redirect(url_for('main.calculator_step', type='helical', step=1))
                
        except Exception as e:
            logger.error(f"Error processing helical pile parameters: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('main.calculator_step', type='helical', step=3))
    
    # Handle step 4 - Results display
    elif type == 'helical' and step == 4:
        # Check if we have results ID
        if 'results_id' not in session:
            flash('No calculation results available. Please complete the analysis first.', 'warning')
            return redirect(url_for('main.calculator_step', type='helical', step=3))
        
        # Load results from storage
        results = load_calculation_results(session['results_id'])
        if not results:
            flash('Results not found. Please recalculate.', 'error')
            return redirect(url_for('main.calculator_step', type='helical', step=3))
        
        # Render the results page
        return render_template(
            'helical/steps.html',
            step=step,
            type=type,
            results=results['summary'],
            detailed_results=results['detailed']
        )
    
    # Continue with the rest of the route handler
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
            if type == 'bored':
                # Add validation for bored pile parameters
                required_fields = ['shaft_diameter', 'base_diameter', 'cased_depth', 'water_table', 'pile_tip_depths']
                for field in required_fields:
                    if field not in request.form or not request.form[field]:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Parse pile tip depths from the comma-separated string
                tip_depths_str = request.form.get('pile_tip_depths', '')
                try:
                    pile_tip_depths = [float(d.strip()) for d in tip_depths_str.split(',')]
                except ValueError:
                    flash('Invalid pile tip depths format. Please enter numbers separated by commas.')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Store parameters in session
                session['pile_params'] = {
                    'shaft_diameter': float(request.form['shaft_diameter']),
                    'base_diameter': float(request.form['base_diameter']),
                    'cased_depth': float(request.form['cased_depth']),
                    'water_table': float(request.form['water_table']),
                    'site_name': request.form.get('file_name', ''),
                    'pile_tip_depths': pile_tip_depths
                }
                
                # Process the data and calculate results
                if 'cpt_data_id' not in session:
                    flash('No CPT data available. Please upload data first.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                # Calculate results
                results = calculate_pile_capacity(cpt_data, session['pile_params'], pile_type=type)
                session['results'] = results
                
                return redirect(url_for('main.calculator_step', type=type, step=4))
            elif type == 'driven':
                # Add validation for driven pile parameters
                required_fields = ['pile_diameter', 'wall_thickness', 'borehole_depth', 'pile_shape', 'pile_end_condition']
                for field in required_fields:
                    if field not in request.form:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Parse pile tip depths from the comma-separated string
                tip_depths_str = request.form.get('pile_tip_depths', '')
                try:
                    pile_tip_depths = [float(d.strip()) for d in tip_depths_str.split(',')]
                except ValueError:
                    flash('Invalid pile tip depths format. Please enter numbers separated by commas.')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
                
                session['pile_params'] = {
                    'pile_diameter': float(request.form['pile_diameter']),
                    'wall_thickness': float(request.form['wall_thickness']),
                    'borehole_depth': float(request.form['borehole_depth']),
                    'pile_shape': request.form['pile_shape'],
                    'pile_end_condition': request.form['pile_end_condition'],
                    'water_table': float(session['water_table']),  # Get from previous step
                    'site_name': request.form.get('site_name', ''),
                    'pile_tip_depths': pile_tip_depths
                }
                
                # Process the data and calculate results
                if 'cpt_data_id' not in session:
                    flash('No CPT data available. Please upload data first.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                # Calculate results
                results = calculate_pile_capacity(cpt_data, session['pile_params'], pile_type=type)
                session['results'] = results
                
                return redirect(url_for('main.calculator_step', type=type, step=4))
            elif type == 'helical':
                # Validate required fields
                required_fields = ['shaft_diameter', 'helix_diameter', 'helix_depth', 'borehole_depth', 'water_table']
                for field in required_fields:
                    if field not in request.form or not request.form[field]:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Store parameters in session
                pile_params = {
                    'shaft_diameter': request.form.get('shaft_diameter'),
                    'helix_diameter': request.form.get('helix_diameter'),
                    'helix_depth': request.form.get('helix_depth'),
                    'borehole_depth': request.form.get('borehole_depth'),
                    'water_table': request.form.get('water_table'),
                    'site_name': request.form.get('site_name', '')
                }
                
                # Convert values to float and validate
                for key in ['shaft_diameter', 'helix_diameter', 'helix_depth', 'borehole_depth', 'water_table']:
                    try:
                        if pile_params[key]:
                            value = float(pile_params[key])
                            if key != 'water_table' and value <= 0:
                                flash(f'{key} must be greater than 0')
                                return redirect(url_for('main.calculator_step', type=type, step=3))
                            pile_params[key] = value
                    except (ValueError, TypeError):
                        flash(f'Invalid value for {key}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Store in session
                session['pile_params'] = pile_params
                
                # Process the data and calculate results
                if 'cpt_data_id' not in session:
                    flash('No CPT data available. Please upload data first.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.', 'error')
                    return redirect(url_for('main.calculator_step', type='helical', step=1))
                
                # Process the CPT data
                processed_cpt = pre_input_calc(cpt_data, float(session['pile_params']['water_table']))
                
                try:
                    # Use our new isolated module function
                    results = calculate_helical_pile_results(processed_cpt, session['pile_params'])
                    
                    # Store results in session
                    session['results'] = results['summary']
                    session['detailed_results'] = results['detailed']
                    
                    # Log successful calculation
                    logger.info(f"Helical pile calculations completed successfully for {pile_params['site_name']}")
                    
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in helical pile calculations: {str(e)}")
                    flash(f"Error in calculation: {str(e)}")
                    return redirect(url_for('main.calculator_step', type=type, step=3))
            else:
                results = calculate_pile_capacity(cpt_data, session['pile_params'], pile_type=type)
                session['results'] = results
        
        elif step == 4:
            if 'results' not in session:
                flash('No results available. Please complete the analysis first.')
                return redirect(url_for('main.calculator_step', type=type, step=3))
            
            detailed_results = None
            if type == 'bored' and 'detailed_results' in session:
                detailed_results = session['detailed_results']
            
            # Debug print to see what's in the results
            print("RESULTS IN SESSION:", session['results'])
            
            # For GET requests, add debug output
            if request.method == 'GET':
                print("Session data on results page:", session.get('pile_params'))
            
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
        elif type == 'helical':
            graphs = create_cpt_graphs(data, data['water_table'])
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
        if type == 'bored' and 'detailed_results' in session:
            detailed_results = session['detailed_results']
        
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
        water_table = float(pile_params.get('water_table', 0))
        processed = pre_input_calc(data, water_table)
        
        # Create a string buffer
        buffer = io.StringIO()
        
        pile_type = session.get('type')
        
        if pile_type == 'bored' and 'debug_id' in session:
            debug_id = session['debug_id']
            debug_details = load_debug_details(debug_id)
            if debug_details and len(debug_details) > 0:
                
                for tip_index, tip_detail in enumerate(debug_details):
                    if tip_index > 0:
                        # Add separator between different tip depth data
                        buffer.write('\n\n' + '='*50 + '\n\n')
                    
                    # Create constants list with tip depth included
                    constants = [
                        ['Tip Depth (m)', tip_detail['tip_depth']],
                        ['Water table depth (m)', float(pile_params['water_table'])]
                    ]
                    
                    # Add pile type specific parameters
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
        elif pile_type == 'driven':
            # For driven piles, create a simplified output format
            results = session.get('results', [])
            
            for result_index, result in enumerate(results):
                if result_index > 0:
                    buffer.write('\n\n' + '='*50 + '\n\n')
                
                tip_depth = result['tipdepth']
                
                # Create parameters list
                constants = [
                    ['Tip Depth (m)', tip_depth],
                    ['Water table depth (m)', float(pile_params['water_table'])],
                    ['Pile type', 'Driven'],
                    ['Pile end condition', pile_params.get('pile_end_condition', 'N/A')],
                    ['Pile shape', pile_params.get('pile_shape', 'N/A')],
                    ['Pile diameter/width (m)', pile_params.get('pile_diameter', 'N/A')],
                    ['Wall thickness (mm)', pile_params.get('wall_thickness', 'N/A')],
                    ['Borehole depth (m)', pile_params.get('borehole_depth', 'N/A')]
                ]
                
                # Write parameters
                df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_depth}m\n')
                df_constants.to_csv(buffer, index=False)
                
                # Write results for this tip depth
                buffer.write('\nRESULTS\n')
                df_result = pd.DataFrame([{
                    'Tip Depth (m)': result['tipdepth'],
                    'q1 (MPa)': result.get('q1', 'N/A'),  # Use get() to handle missing keys
                    'q10 (MPa)': result.get('q10', 'N/A'),  # Use get() to handle missing keys
                    'Tension Capacity (kN)': result['tension_capacity'],
                    'Compression Capacity (kN)': result['compression_capacity']
                }])
                df_result.to_csv(buffer, index=False)
                
                # Write CPT data
                buffer.write('\nCPT DATA\n')
                df_cpt = pd.DataFrame({
                    'depth': processed['depth'],
                    'qc (MPa)': processed['qc'],
                    'qt (MPa)': processed['qt'],
                    'fs (kPa)': processed['fs'],
                    'Unit Weight (kN/m³)': processed['gtot'],
                    'Water Pressure u0 (kPa)': processed['u0_kpa'],
                    'Total Vertical Stress σv0 (kPa)': processed['sig_v0'],
                    'Effective Vertical Stress σv0\' (kPa)': processed['sig_v0_prime'],
                    'Fr (%)': processed['fr_percent'],
                    'Normalized Tip Resistance Qtn': processed['qtn'],
                    'Stress Exponent n': processed['n'],
                    'Soil Behavior Type Index Ic': processed['lc'],
                    'Pore Pressure Ratio Bq': processed['bq'],
                    'Correction Factor Kc': processed['kc'],
                    'Corrected Tip Resistance qtc (MPa)': processed['qtc'],
                    'Soil Behavior Index Iz': processed['iz1']
                })
                df_cpt = pd.DataFrame({
                    'depth': processed['depth'],
                    'qc (MPa)': processed['qc'],
                    'qt (MPa)': processed['qt'],
                    'fs (kPa)': processed['fs'],
                    'Unit Weight (kN/m³)': processed['gtot'],
                    'Water Pressure u0 (kPa)': processed['u0_kpa'],
                    'Total Vertical Stress σv0 (kPa)': processed['sig_v0'],
                    'Effective Vertical Stress σv0\' (kPa)': processed['sig_v0_prime'],
                    'Fr (%)': processed['fr_percent'],
                    'Normalized Tip Resistance Qtn': processed['qtn'],
                    'Stress Exponent n': processed['n'],
                    'Soil Behavior Type Index Ic': processed['lc'],
                    'Pore Pressure Ratio Bq': processed['bq'],
                    'Correction Factor Kc': processed['kc'],
                    'Corrected Tip Resistance qtc (MPa)': processed['qtc'],
                    'Soil Behavior Index Iz': processed['iz1']
                })
                df_cpt.to_csv(buffer, index=False)
        elif pile_type == 'helical':
            # For helical piles, create a detailed output format
            results = session.get('results', [])
            detailed_results = session.get('detailed_results', {})
            
            # Get pile parameters from session
            session_pile_params = session.get('pile_params', {})
            
            for result_index, result in enumerate(results):
                if result_index > 0:
                    buffer.write('\n\n' + '='*50 + '\n\n')
                
                tip_depth = result['tipdepth']
                
                # Create parameters table
                constants = [
                    ['Tip Depth (m)', tip_depth],
                    ['Water table depth (m)', session_pile_params.get('water_table', 0)],
                    ['Pile type', 'Helical'],
                    ['Shaft diameter (m)', session_pile_params.get('shaft_diameter', 'N/A')],
                    ['Helix diameter (m)', session_pile_params.get('helix_diameter', 'N/A')],
                    ['Helix depth (m)', session_pile_params.get('helix_depth', 'N/A')],
                    ['Borehole depth (m)', session_pile_params.get('borehole_depth', 'N/A')],
                    ['Site name', session_pile_params.get('site_name', 'N/A')]
                ]
                
                # Write constants
                df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                buffer.write(f'INPUT PARAMETERS\n')
                df_constants.to_csv(buffer, index=False)
                
                # Write results
                buffer.write('\nRESULTS\n')
                df_result = pd.DataFrame([{
                    'Depth (m)': result['tipdepth'],
                    'Tension Capacity (kN)': result['tension_capacity'],
                    'Compression Capacity (kN)': result['compression_capacity']
                }])
                df_result.to_csv(buffer, index=False)
                
                # Add intermediate calculations if available
                if detailed_results and 'depth' in detailed_results:
                    buffer.write('\nINTERMEDIATE CALCULATIONS\n')
                    
                    # Create DataFrame from detailed results
                    calc_data = {
                        'Depth (m)': detailed_results['depth'],
                        'qt (MPa)': detailed_results['qt'],
                        'qc (MPa)': detailed_results['qc'],
                        'fs (kPa)': detailed_results['fs'],
                        'Fr (%)': detailed_results['fr_percent'],
                        'Soil Behavior Type Index Ic': detailed_results['lc'],
                        'Soil Type': detailed_results['soil_type'],
                        'q1 (MPa)': detailed_results['q1'],
                        'q10 (MPa)': detailed_results['q10'],
                        'Casing Coefficient': detailed_results['coe_casing'],
                        'Delta Z (m)': detailed_results['delta_z'],
                        'Shaft Capacity Segment (kN)': detailed_results['qshaft_segment'],
                        'Shaft Capacity Cumulative (kN)': detailed_results['qshaft_kn']
                    }
                    
                    # Add capacity columns if available
                    if 'tension_capacity' in detailed_results:
                        calc_data['Tension Capacity (kN)'] = detailed_results['tension_capacity']
                    if 'compression_capacity' in detailed_results:
                        calc_data['Compression Capacity (kN)'] = detailed_results['compression_capacity']
                        
                    df_calculations = pd.DataFrame(calc_data)
                    df_calculations.to_csv(buffer, index=False)
                    
                    # Add helix-specific calculations
                    buffer.write('\nHELIX CALCULATIONS\n')
                    helix_data = [
                        ['Perimeter (m)', detailed_results.get('perimeter', 'N/A')],
                        ['Helix Area (m²)', detailed_results.get('helix_area', 'N/A')],
                        ['Helix Index (depth point)', detailed_results.get('helix_index', 'N/A')],
                        ['q1 at Helix (MPa)', detailed_results.get('q1_helix', 'N/A')],
                        ['q10 at Helix (MPa)', detailed_results.get('q10_helix', 'N/A')],
                        ['Helix Tension Component (kN)', detailed_results.get('qhelix_tension', 'N/A')],
                        ['Helix Compression Component (kN)', detailed_results.get('qhelix_compression', 'N/A')]
                    ]
                    df_helix = pd.DataFrame(helix_data, columns=['Parameter', 'Value'])
                    df_helix.to_csv(buffer, index=False)
        
        # Get the buffer value
        buffer_value = buffer.getvalue()
        
        # Create a response with the CSV data
        user_filename = session.get('original_filename', 'output')
        download_name = f"detailed_output_{user_filename}_{datetime.now().strftime('%d%m%Y')}.csv"
        print("Using original filename:", user_filename)
        print("Final user_filename:", user_filename)
        print("Final download_name:", download_name)
        
        return Response(
            buffer_value,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={download_name}"}
        )
    
    except Exception as e:
        print(f"Debug download error: {str(e)}")
        flash(f"Error generating download: {str(e)}")
        return redirect(url_for('main.calculator_step', type=session.get('type', 'helical'), step=4))

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
    
    # Set session as permanent and add registration info
    session.permanent = True
    session['registered'] = True
    session['user_email'] = email
    
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
    if type not in ['driven', 'bored', 'helical']:
        return redirect(url_for('main.index'))
    return render_template(f'{type}/description.html', type=type)

@bp.route('/download_intermediary_calcs')
def download_intermediary_calcs():
    """Download intermediary calculations used for graphs as CSV"""
    if 'cpt_data_id' not in session:
        flash('No CPT data available')
        return redirect(url_for('main.index'))
    
    try:
        data = load_cpt_data(session['cpt_data_id'])
        if not data:
            flash('CPT data not found')
            return redirect(url_for('main.index'))
        
        # Get water table from the data itself since it's stored with the CPT data
        water_table = data['water_table']
        processed = pre_input_calc(data, water_table)
        
        if not processed:
            flash('Error processing data')
            return redirect(url_for('main.index'))
        
        # Create DataFrame with all intermediate calculations
        df = pd.DataFrame({
            'Depth (m)': processed['depth'],
            'qc (MPa)': processed['qc'],
            'qt (MPa)': processed['qt'],
            'fs (kPa)': processed['fs'],
            'Unit Weight (kN/m³)': processed['gtot'],
            'Water Pressure u0 (kPa)': processed['u0_kpa'],
            'Total Vertical Stress σv0 (kPa)': processed['sig_v0'],
            'Effective Vertical Stress σv0\' (kPa)': processed['sig_v0_prime'],
            'Fr (%)': processed['fr_percent'],
            'Normalized Tip Resistance Qtn': processed['qtn'],
            'Stress Exponent n': processed['n'],
            'Soil Behavior Type Index Ic': processed['lc'],
            'Pore Pressure Ratio Bq': processed['bq'],
            'Correction Factor Kc': processed['kc'],
            'Corrected Tip Resistance qtc (MPa)': processed['qtc'],
            'Soil Behavior Index Iz': processed['iz1']
        })
        
        # Get the current date in DDMMYYYY format
        current_date = datetime.now().strftime('%d%m%Y')
        
        # Use original filename if available
        filename = session.get('original_filename', '')
        if filename:
            base_name = os.path.splitext(filename)[0]
            download_name = f"{base_name}_intermediary_calcs_{current_date}.csv"
        else:
            download_name = f"intermediary_calcs_{current_date}.csv"
        
        return send_file(
            io.BytesIO(df.to_csv(index=False).encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_name
        )
        
    except Exception as e:
        print(f"Error generating intermediary calculations: {str(e)}")
        flash('Error generating calculations')
        return redirect(url_for('main.calculator_step', type=session.get('type', 'driven'), step=2))

@bp.route('/download_helical_calculations')
def download_helical_calculations():
    """Download all helical pile calculation data in CSV format"""
    try:
        # Get the calculation results from the session
        if 'results_id' not in session:
            flash('No calculation results found', 'error')
            return redirect(url_for('main.index'))
        
        # Load the calculation results
        results = load_calculation_results(session['results_id'])
        if not results:
            flash('Calculation results not found', 'error')
            return redirect(url_for('main.index'))
        
        # Debug logging - print the keys in the results dictionary
        current_app.logger.info(f"Results keys: {list(results.keys())}")
        
        # Get pile parameters
        pile_params = {}
        if 'pile_params' in results:
            pile_params = results['pile_params']
        elif 'detailed' in results and 'input_parameters' in results['detailed']:
            pile_params = results['detailed']['input_parameters']
        elif 'summary' in results and isinstance(results['summary'], dict) and 'tipdepth' in results['summary']:
            # Try to extract from session if not in results
            pile_params = session.get('pile_params', {})
        
        # Debug logging
        current_app.logger.info(f"Downloading helical pile calculations")
        current_app.logger.info(f"Pile params: {pile_params}")
        
        # Get the detailed results
        detailed_results = {}
        if 'detailed' in results:
            detailed_results = results['detailed']
        elif 'detailed_results' in results:
            detailed_results = results['detailed_results']
        
        # Debug logging - print the keys in the detailed_results dictionary
        if detailed_results:
            current_app.logger.info(f"Detailed results keys: {list(detailed_results.keys())}")
        else:
            current_app.logger.warning("No detailed results found")
        
        # Get the user's filename or use a default
        user_filename = pile_params.get('site_name', '')
        if not user_filename:
            user_filename = "helical_pile_calculations"
        current_app.logger.info(f"Using original filename: {user_filename}")
        
        # Clean the filename
        user_filename = ''.join(c for c in user_filename if c.isalnum() or c in '._- ')
        user_filename = user_filename.strip()
        if not user_filename:
            user_filename = "helical_pile_calculations"
        current_app.logger.info(f"Final user_filename: {user_filename}")
        
        # Create a timestamp for the filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%d%m%Y")
        download_name = f"{user_filename}_detailed_{timestamp}.csv"
        current_app.logger.info(f"Final download_name: {download_name}")
        
        # Create a CSV from the data
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # If we have pre-formatted download data, use it
        if 'download_data' in results and results['download_data']:
            current_app.logger.info("Using pre-formatted download data")
            for row in results['download_data']:
                writer.writerow(row)
        else:
            # Otherwise, build the CSV from detailed results
            current_app.logger.info("Building CSV from detailed results")
            
            # Create header row
            header = [
                "Depth (m)",
                "qt (MPa)",
                "qc (MPa)",
                "fs (kPa)",
                "Fr (%)",
                "Ic",
                "Soil Type",
                "q1 (MPa)",
                "q10 (MPa)",
                "Casing Coefficient",
                "Delta Z (m)",
                "Shaft Segment (kN)",
                "Cumulative Shaft (kN)",
                "Tension Capacity (kN)",
                "Compression Capacity (kN)"
            ]
            writer.writerow(header)
            
            # Check if we have depth data
            if not detailed_results or 'depth' not in detailed_results or not detailed_results['depth']:
                current_app.logger.error("No depth data found in detailed results")
                writer.writerow(["No detailed calculation data available"])
            else:
                # Add data rows
                for i in range(len(detailed_results['depth'])):
                    try:
                        row = [
                            detailed_results['depth'][i],
                            detailed_results.get('qt', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('qc', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('fs', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('fr_percent', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('lc', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('soil_type', ['Unknown'] * len(detailed_results['depth']))[i],
                            detailed_results.get('q1', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('q10', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('coe_casing', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('delta_z', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('qshaft_segment', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('qshaft_kn', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('tension_capacity', [0] * len(detailed_results['depth']))[i],
                            detailed_results.get('compression_capacity', [0] * len(detailed_results['depth']))[i]
                        ]
                        writer.writerow(row)
                    except (IndexError, KeyError) as e:
                        current_app.logger.error(f"Error writing row {i}: {str(e)}")
                        continue
            
            # Add empty row for spacing
            writer.writerow([])
            
            # Add summary information
            writer.writerow(["SUMMARY INFORMATION"])
            writer.writerow(["Input Parameters"])
            for key, value in pile_params.items():
                writer.writerow([key, value])
            
            # Add geometric constants if available
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Geometric Constants"])
            writer.writerow(["Perimeter (m)", detailed_results.get('perimeter', 'N/A')])
            writer.writerow(["Helix Area (m²)", detailed_results.get('helix_area', 'N/A')])
            
            # Add helix information if available
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Helix Information"])
            writer.writerow(["Helix Depth (m)", pile_params.get('helix_depth', 'N/A')])
            writer.writerow(["q1 at Helix", detailed_results.get('q1_helix', 'N/A')])
            writer.writerow(["q10 at Helix", detailed_results.get('q10_helix', 'N/A')])
            writer.writerow(["Helix Tension Capacity (kN)", detailed_results.get('qhelix_tension', 'N/A')])
            writer.writerow(["Helix Compression Capacity (kN)", detailed_results.get('qhelix_compression', 'N/A')])
            
            # Add effective depth calculations if available
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Effective Depth Calculations"])
            writer.writerow(["Tension Effective Depth (m)", detailed_results.get('tension_effective_depth', 'N/A')])
            writer.writerow(["Tension Min q10", detailed_results.get('tension_min_q10', 'N/A')])
            writer.writerow(["q(10mm) Tension", detailed_results.get('q_10mm_tens', 'N/A')])
            writer.writerow(["Compression Effective Depth (m)", detailed_results.get('compression_effective_depth', 'N/A')])
            writer.writerow(["Compression Min q10", detailed_results.get('compression_min_q10', 'N/A')])
            writer.writerow(["q(10mm) Compression", detailed_results.get('q_10mm_comp', 'N/A')])
            
            # Add final results
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Final Results"])
            
            # Get summary data from different possible locations
            summary_data = {}
            if 'summary' in results:
                summary_data = results['summary']
            elif 'summary_data' in results:
                summary_data = results['summary_data']
            elif 'summary_data' in session:
                summary_data = session['summary_data']
            
            # Write final results
            writer.writerow(["Ultimate Tension Capacity (kN)", summary_data.get('qult_tension', detailed_results.get('qult_tension', 'N/A'))])
            writer.writerow(["Ultimate Compression Capacity (kN)", summary_data.get('qult_compression', detailed_results.get('qult_compression', 'N/A'))])
            writer.writerow(["Tension Capacity at 10mm (kN)", detailed_results.get('q_delta_10mm_tension', 'N/A')])
            writer.writerow(["Compression Capacity at 10mm (kN)", detailed_results.get('q_delta_10mm_compression', 'N/A')])
            writer.writerow(["Installation Torque (kNm)", detailed_results.get('installation_torque', 'N/A')])
        
        # Prepare the file for download
        output.seek(0)
        
        # Add more debug logging right before sending the file
        current_app.logger.info(f"Sending file with size: {len(output.getvalue())} bytes")
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_name
        )
    
    except Exception as e:
        # Log the full exception with traceback
        import traceback
        current_app.logger.error(f"Error in download_helical_calculations: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        
        # Flash a more helpful error message
        flash(f"Error generating download: {str(e)}", 'error')
        return redirect(url_for('main.index'))