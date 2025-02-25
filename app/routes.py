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
from .calculations import calculate_pile_capacity, process_cpt_data, pre_input_calc, get_iterative_values, calculate_bored_pile_results, calculate_helical_pile_results
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
    if type not in ['driven', 'bored', 'helical']:
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
            # Validate form data
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
                # Add validation for helical pile parameters
                required_fields = ['shaft_diameter', 'helix_diameter', 'helix_depth', 'borehole_depth', 'water_table']
                for field in required_fields:
                    if field not in request.form or not request.form[field]:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Store parameters in session
                session['pile_params'] = {
                    'shaft_diameter': float(request.form['shaft_diameter']),
                    'helix_diameter': float(request.form['helix_diameter']),
                    'helix_depth': float(request.form['helix_depth']),
                    'borehole_depth': float(request.form['borehole_depth']),
                    'water_table': float(request.form['water_table']),
                    'site_name': request.form.get('site_name', '')
                }
                
                # Process the data and calculate results
                if 'cpt_data_id' not in session:
                    flash('No CPT data available. Please upload data first.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                # Process the CPT data
                processed_cpt = pre_input_calc(cpt_data, float(session['pile_params']['water_table']))
                
                # Calculate results based on pile type
                results = calculate_helical_pile_results(processed_cpt, session['pile_params'])
                
                # Store results in session
                session['results'] = results['summary']
                session['detailed_results'] = results['detailed']
                
                return redirect(url_for('main.calculator_step', type=type, step=4))
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
                df_cpt.to_csv(buffer, index=False)
        elif pile_type == 'helical':
            # For helical piles, create a detailed output format
            results = session.get('results', [])
            
            for result_index, result in enumerate(results):
                if result_index > 0:
                    buffer.write('\n\n' + '='*50 + '\n\n')
                
                tip_depth = result['tipdepth']
                
                # Create parameters list - use get() to handle missing keys
                constants = [
                    ['Tip Depth (m)', tip_depth],
                    ['Water table depth (m)', float(pile_params.get('water_table', 0))],
                    ['Pile type', 'Helical'],
                    ['Shaft diameter (m)', pile_params.get('shaft_diameter', 'N/A')]
                ]
                
                # Add helical-specific parameters if they exist
                if 'helix_diameter' in pile_params:
                    constants.append(['Helix diameter (m)', pile_params.get('helix_diameter')])
                if 'helix_depth' in pile_params:
                    constants.append(['Helix depth (m)', pile_params.get('helix_depth')])
                if 'borehole_depth' in pile_params:
                    constants.append(['Borehole depth (m)', pile_params.get('borehole_depth')])
                
                # Write parameters
                df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_depth}m\n')
                df_constants.to_csv(buffer, index=False)
                
                # Write results for this tip depth
                buffer.write('\nRESULTS\n')
                df_result = pd.DataFrame([{
                    'Tip Depth (m)': result['tipdepth'],
                    'q1 (MPa)': result.get('q1', 'N/A'),
                    'q10 (MPa)': result.get('q10', 'N/A'),
                    'Tension Capacity (kN)': result['tension_capacity'],
                    'Compression Capacity (kN)': result['compression_capacity']
                }])
                df_result.to_csv(buffer, index=False)
                
                # Add intermediate calculations
                if 'cpt_data_id' in session:
                    # Calculate helical pile specific intermediate values
                    shaft_diameter = float(pile_params.get('shaft_diameter', 0))
                    helix_diameter = float(pile_params.get('helix_diameter', 0))
                    helix_depth = float(pile_params.get('helix_depth', 0))
                    borehole_depth = float(pile_params.get('borehole_depth', 0))
                    
                    # Calculate constants
                    perimeter = math.pi * shaft_diameter
                    helix_area = math.pi * (helix_diameter ** 2) * 0.25
                    
                    # Calculate q1 and q10 values for all depths
                    q1_values = []
                    q10_values = []
                    for qt in processed['qt']:
                        q1 = qt * (0.1 ** 0.6)
                        q10 = qt * ((0.01/helix_diameter) ** 0.6) if helix_diameter > 0 else 0
                        q1_values.append(q1)
                        q10_values.append(q10)
                    
                    # Calculate casing coefficient and soil type
                    coe_casing = []
                    soil_type = []
                    
                    for i, depth_val in enumerate(processed['depth']):
                        # Casing coefficient
                        if depth_val < borehole_depth:
                            casing = 0
                        elif depth_val < helix_depth:
                            casing = 1
                        else:
                            casing = 0
                        coe_casing.append(casing)
                        
                        # Soil type based on Ic value - simplified to Sand/Clay-Silt
                        ic = processed['lc'][i] if i < len(processed['lc']) else 0
                        if ic < 2.2:
                            soil = "Sand"
                        else:
                            soil = "Clay/Silt"
                        soil_type.append(soil)
                    
                    # Calculate delta_z and shaft capacity
                    delta_z = []
                    qshaft_segment = []
                    qshaft_kn = []
                    cumulative_qshaft = 0
                    
                    for i, depth_val in enumerate(processed['depth']):
                        # Calculate delta z
                        if i == 0:
                            current_delta_z = depth_val
                        else:
                            current_delta_z = depth_val - processed['depth'][i-1]
                        
                        delta_z.append(current_delta_z)
                        
                        # Calculate shaft capacity increment
                        qshaft_increment = (coe_casing[i] * current_delta_z * 1000 * processed['qt'][i] * perimeter) / 175
                        qshaft_segment.append(qshaft_increment)
                        cumulative_qshaft += qshaft_increment
                        qshaft_kn.append(cumulative_qshaft)
                    
                    # Calculate helix capacities
                    helix_index = min(range(len(processed['depth'])), key=lambda i: abs(processed['depth'][i] - helix_depth))
                    
                    # Get q1 and q10 at helix depth
                    q1_helix = q1_values[helix_index]
                    q10_helix = q10_values[helix_index]
                    
                    # Calculate helix capacities
                    qhelix_tension = q10_helix * helix_area * 1000
                    qhelix_compression = q1_helix * helix_area * 1000
                    
                    # Calculate total capacities
                    tension_capacity_array = []
                    compression_capacity_array = []
                    
                    for i in range(len(processed['depth'])):
                        if processed['depth'][i] <= helix_depth:
                            tension_capacity = qshaft_kn[i] + qhelix_tension
                            compression_capacity = qshaft_kn[i] + qhelix_compression
                        else:
                            tension_capacity = qshaft_kn[i]
                            compression_capacity = qshaft_kn[i]
                        
                        tension_capacity_array.append(tension_capacity)
                        compression_capacity_array.append(compression_capacity)
                    
                    # Create a DataFrame with all intermediate calculations
                    buffer.write('\nINTERMEDIATE CALCULATIONS\n')
                    df_intermediate = pd.DataFrame({
                        'Depth (m)': processed['depth'],
                        'qt (MPa)': processed['qt'],
                        'qc (MPa)': processed['qc'],
                        'fs (kPa)': processed['fs'],
                        'Unit Weight (kN/m³)': processed['gtot'],
                        'Fr (%)': processed['fr_percent'],
                        'Soil Behavior Type Index Ic': processed['lc'],
                        'Soil Type': soil_type,
                        'Normalized Tip Resistance Qtn': processed.get('qtn', [0] * len(processed['depth'])),
                        'Stress Exponent n': processed.get('n', [0] * len(processed['depth'])),
                        'q1 (MPa)': q1_values,
                        'q10 (MPa)': q10_values,
                        'Casing Coefficient': coe_casing,
                        'Delta Z (m)': delta_z,
                        'Shaft Capacity Segment (kN)': qshaft_segment,
                        'Shaft Capacity Cumulative (kN)': qshaft_kn,
                        'Tension Capacity (kN)': tension_capacity_array,
                        'Compression Capacity (kN)': compression_capacity_array,
                        'Pore Pressure Ratio Bq': processed.get('bq', [0] * len(processed['depth'])),
                        'Effective Vertical Stress σv0\' (kPa)': processed.get('sig_v0_prime', [0] * len(processed['depth']))
                    })
                    df_intermediate.to_csv(buffer, index=False)
                    
                    # Add helix-specific calculations
                    buffer.write('\nHELIX CALCULATIONS\n')
                    df_helix = pd.DataFrame([{
                        'Shaft Perimeter (m)': perimeter,
                        'Helix Area (m²)': helix_area,
                        'q1 at Helix Depth (MPa)': q1_helix,
                        'q10 at Helix Depth (MPa)': q10_helix,
                        'Helix Tension Capacity (kN)': qhelix_tension,
                        'Helix Compression Capacity (kN)': qhelix_compression,
                        'Total Tension Capacity (kN)': tension_capacity_array[helix_index],
                        'Total Compression Capacity (kN)': compression_capacity_array[helix_index]
                    }])
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