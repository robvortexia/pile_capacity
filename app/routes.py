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
                    # Convert all numeric arrays in processed_cpt to lists
                    for key in processed_cpt:
                        if isinstance(processed_cpt[key], np.ndarray):
                            processed_cpt[key] = processed_cpt[key].tolist()
                    
                    results = calculate_helical_pile_results(processed_cpt, pile_params)
                    
                    # Store results in file and keep ID in session
                    results_id = save_calculation_results(results)
                    session['results_id'] = results_id
                    
                    # Store basic summary data in session for quick access
                    session['summary_data'] = {
                        'tipdepth': results['summary'][0]['tipdepth'],
                        'qshaft': results['summary'][0]['qshaft'],
                        'qult_tension': results['summary'][0]['qult_tension'],
                        'qult_compression': results['summary'][0]['qult_compression']
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
        
        # Render the results page with the first summary result
        return render_template(
            'helical/steps.html',
            step=step,
            type=type,
            results=results['summary'][0],  # Pass the first summary result directly
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
    try:
        if 'results_id' not in session:
            flash('No calculation data available')
            return redirect(url_for('main.index'))
        
        # Load results from file storage
        results = load_calculation_results(session['results_id'])
        if not results:
            flash('Results not found')
            return redirect(url_for('main.index'))
        
        # For helical piles, use the pre-formatted download data
        if session.get('type') == 'helical':
            if 'download_data' in results and results['download_data']:
                # Create a buffer to store CSV data
                buffer = StringIO()
                writer = csv.writer(buffer)
                
                # Write the pre-formatted data
                for row in results['download_data']:
                    writer.writerow(row)
                
                # Set buffer position to start
                buffer.seek(0)
                
                # Return the CSV file
                return Response(
                    buffer.getvalue(),
                    mimetype='text/csv',
                    headers={
                        'Content-Disposition': f'attachment; filename=helical_pile_detailed_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    }
                )
            else:
                flash('Detailed output data not available')
                return redirect(url_for('main.calculator_step', type='helical', step=4))
        
        # For other pile types, use the existing logic
        else:
            # Create a buffer to store CSV data
            buffer = StringIO()
            writer = csv.writer(buffer)
            
            # Write pile type
            writer.writerow([f"PILE TYPE: {session.get('type', 'driven').upper()}"])
            writer.writerow([])  # Empty row for spacing
            
            # Write input parameters
            writer.writerow(["INPUT PARAMETERS"])
            params = session.get('pile_params', {})
            for key, value in params.items():
                writer.writerow([key, value])
            
            writer.writerow([])  # Empty row for spacing
            
            # Write CPT data if available
            if 'cpt_data_id' in session:
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if cpt_data:
                    writer.writerow(["CPT DATA"])
                    writer.writerow(["Depth (m)", "qc (MPa)", "fs (kPa)", "u2 (kPa)"])
                    for i in range(len(cpt_data['depth'])):
                        writer.writerow([
                            cpt_data['depth'][i],
                            cpt_data['qc'][i],
                            cpt_data['fs'][i],
                            cpt_data.get('u2', [0]*len(cpt_data['depth']))[i]
                        ])
                    writer.writerow([])  # Empty row for spacing
            
            # Write calculation results
            if 'detailed' in results:
                for tip_result in results['detailed']:
                    writer.writerow([f"CALCULATIONS FOR TIP DEPTH: {tip_result['tip_depth']} m"])
                    writer.writerow([
                        "Depth (m)",
                        "qt (MPa)",
                        "Ic",
                        "Fr (%)",
                        "Casing Coefficient",
                        "qb0.1 (MPa)",
                        "tf Tension (kPa)",
                        "tf Compression (kPa)",
                        "Delta Z (m)",
                        "Tension Segment (kN)",
                        "Compression Segment (kN)",
                        "Tension Cumulative (kN)",
                        "Compression Cumulative (kN)"
                    ])
                    
                    for calc in tip_result['calculations']:
                        writer.writerow([
                            calc['depth'],
                            calc['qt'],
                            calc['lc'],
                            calc['fr'],
                            calc['coe_casing'],
                            calc['qb01_adop'],
                            calc['tf_tension'],
                            calc['tf_compression'],
                            calc['delta_z'],
                            calc['qs_tension_segment'],
                            calc['qs_compression_segment'],
                            calc['qs_tension_cumulative'],
                            calc['qs_compression_cumulative']
                        ])
                    writer.writerow([])  # Empty row for spacing
            
            # Set buffer position to start
            buffer.seek(0)
            
            # Return the CSV file
            return Response(
                buffer.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=calculation_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                }
            )
    except Exception as e:
        logger.error(f"Error in download_debug_params: {str(e)}")
        flash(f'Error generating detailed output: {str(e)}')
        return redirect(url_for('main.calculator_step', type=session.get('type', 'driven'), step=4))

@bp.route('/download_results')
def download_results():
    """Download calculation results as CSV"""
    if 'results_id' not in session:
        flash('No results available')
        return redirect(url_for('main.index'))
    
    try:
        # Load results from file storage
        results = load_calculation_results(session['results_id'])
        if not results:
            flash('Results not found')
            return redirect(url_for('main.index'))
        
        # Create a buffer for CSV data
        buffer = StringIO()
        writer = csv.writer(buffer)
        
        # For helical piles, create a simplified summary
        if session.get('type') == 'helical':
            # Write header
            writer.writerow(['HELICAL PILE CALCULATION RESULTS'])
            writer.writerow([])
            
            # Get the summary data
            summary = results['summary'][0]
            
            # Write pile parameters
            writer.writerow(['PILE PARAMETERS'])
            pile_params = results['detailed']['input_parameters']
            for key, value in pile_params.items():
                writer.writerow([key, value])
            writer.writerow([])
            
            # Write capacity results
            writer.writerow(['CAPACITY RESULTS'])
            writer.writerow(['Parameter', 'Tension', 'Compression'])
            writer.writerow(['Shaft Capacity (kN)', summary['qshaft'], summary['qshaft']])
            writer.writerow(['Capacity at 10mm (kN)', summary['q_delta_10mm_tension'], summary['q_delta_10mm_compression']])
            writer.writerow(['Ultimate Capacity (kN)', summary['qult_tension'], summary['qult_compression']])
            writer.writerow([])
            
            # Write additional parameters
            writer.writerow(['ADDITIONAL PARAMETERS'])
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Tip Depth (m)', summary['tipdepth']])
            writer.writerow(['Effective Depth (m)', summary['effective_depth']])
            writer.writerow(['qb0.1 Compression (MPa)', summary['qb01_comp']])
            writer.writerow(['qb0.1 Tension (MPa)', summary['qb01_tension']])
            writer.writerow(['Installation Torque (kNm)', summary['installation_torque']])
        else:
            # For other pile types, use existing logic
            df = pd.DataFrame(results)
            buffer.write(df.to_csv(index=False))
        
        # Set buffer position to start
        buffer.seek(0)
        
        # Return the CSV file
        return Response(
            buffer.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=calculation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
    except Exception as e:
        logger.error(f"Error in download_results: {str(e)}")
        flash(f'Error generating results: {str(e)}')
        return redirect(url_for('main.calculator_step', type=session.get('type', 'driven'), step=4))

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
        
        # Create a buffer for CSV data
        buffer = StringIO()
        writer = csv.writer(buffer)
        
        # If we have pre-formatted download data, use it
        if 'download_data' in results and results['download_data']:
            for row in results['download_data']:
                writer.writerow(row)
        else:
            # Get detailed results
            detailed = results.get('detailed', {})
            
            # Write header
            writer.writerow(["HELICAL PILE CALCULATION RESULTS"])
            writer.writerow([])
            
            # Write input parameters
            writer.writerow(["INPUT PARAMETERS"])
            if 'input_parameters' in detailed:
                for key, value in detailed['input_parameters'].items():
                    writer.writerow([key, value])
            
            writer.writerow([])
            writer.writerow(["GEOMETRIC CONSTANTS"])
            writer.writerow(["Perimeter (m)", detailed.get('perimeter', '')])
            writer.writerow(["Helix Area (m²)", detailed.get('helix_area', '')])
            
            writer.writerow([])
            writer.writerow(["DETAILED CALCULATIONS"])
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
            
            for i in range(len(detailed['depth'])):
                row = [
                    detailed['depth'][i],
                    detailed['qt'][i],
                    detailed['qc'][i],
                    detailed['fs'][i],
                    detailed['fr_percent'][i],
                    detailed['lc'][i],
                    detailed['soil_type'][i],
                    detailed['q1'][i],
                    detailed['q10'][i],
                    detailed['coe_casing'][i],
                    detailed['delta_z'][i],
                    detailed['qshaft_segment'][i],
                    detailed['qshaft_kn'][i],
                    detailed['tension_capacity'][i],
                    detailed['compression_capacity'][i]
                ]
                writer.writerow(row)
        
        # Set buffer position to start
        buffer.seek(0)
        
        # Return the CSV file
        return Response(
            buffer.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=helical_calculations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
    except Exception as e:
        logger.error(f"Error in download_helical_calculations: {str(e)}")
        flash(f'Error generating helical calculations: {str(e)}')
        return redirect(url_for('main.calculator_step', type='helical', step=4))