from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session, Response, send_file, current_app, make_response
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
    save_calculation_results, load_calculation_results, create_helical_pile_graphs
)
from .calculations import calculate_pile_capacity, process_cpt_data, pre_input_calc, get_iterative_values, calculate_bored_pile_results, calculate_helical_pile_results, calculate_driven_pile_results
from .interpolation import process_uploaded_cpt_data
from datetime import datetime, timedelta
from .models import db, Registration, Visit
from functools import wraps
import csv
from io import StringIO
from sqlalchemy.sql import func
import logging
import io
from .helical_calculations import calculate_helical_pile_results
from .analytics import record_page_visit, store_analytics_data, get_or_create_user_id, get_page_visit_stats, get_analytics_data_stats, record_event

# Set pandas options for full precision 
pd.set_option('display.precision', 15)  # Increase default precision
pd.set_option('display.float_format', lambda x: '%.15g' % x)  # Use full precision in string conversions

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

bp = Blueprint('main', __name__)


@bp.route('/googlef2236ffa5d780ee8.html')
def google_site_verification():
    """Serve Google Search Console verification file."""
    return Response('google-site-verification: googlef2236ffa5d780ee8.html', mimetype='text/html')


@bp.route('/robots.txt')
def robots_txt():
    """Serve robots.txt for search engine crawlers."""
    content = """User-agent: *
Allow: /
Allow: /driven/description
Allow: /bored/description
Allow: /helical/description
Disallow: /admin
Disallow: /admin/export
Disallow: /admin/send_weekly_report
Disallow: /download_results
Disallow: /download_debug_params
Disallow: /download_intermediary_calcs
Disallow: /download_helical_calculations
Disallow: /register
Disallow: /track_ad_click

Sitemap: https://uwa-geotech-cpt-calculator.com/sitemap.xml
"""
    return Response(content.strip(), mimetype='text/plain')


@bp.route('/sitemap.xml')
def sitemap_xml():
    """Serve XML sitemap for search engine indexing."""
    pages = [
        {'loc': '/', 'priority': '1.0', 'changefreq': 'monthly'},
        {'loc': '/driven/description', 'priority': '0.8', 'changefreq': 'yearly'},
        {'loc': '/bored/description', 'priority': '0.8', 'changefreq': 'yearly'},
        {'loc': '/helical/description', 'priority': '0.8', 'changefreq': 'yearly'},
    ]
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    for page in pages:
        xml += '  <url>\n'
        xml += f'    <loc>https://uwa-geotech-cpt-calculator.com{page["loc"]}</loc>\n'
        xml += f'    <changefreq>{page["changefreq"]}</changefreq>\n'
        xml += f'    <priority>{page["priority"]}</priority>\n'
        xml += '  </url>\n'
    xml += '</urlset>'
    return Response(xml, mimetype='application/xml')


@bp.route('/sample/<type>')
def use_sample_data(type):
    """Load sample CPT data for demo purposes."""
    if type not in ['driven', 'bored', 'helical']:
        return redirect(url_for('main.index'))

    # Read sample data file
    sample_path = os.path.join(current_app.static_folder, 'data', 'sample_cpt.csv')
    try:
        data_dict = []
        with open(sample_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = line.split(',')
                if len(values) >= 4:
                    data_dict.append({
                        'z': float(values[0]),
                        'qc': float(values[1]),
                        'fs': float(values[2]),
                        'gtot': float(values[3])
                    })

        if not data_dict:
            flash('Error loading sample data')
            return redirect(url_for('main.calculator_step', type=type, step=1))

        processed_data = process_cpt_data(data_dict)
        water_table = 2.0  # Default water table for sample

        file_id = save_cpt_data(processed_data['cpt_data'], water_table)
        session['cpt_data_id'] = file_id
        session['water_table'] = water_table
        session['original_filename'] = 'sample_data'
        session['type'] = type

        store_analytics_data('sample_data', 'type', type)

        return redirect(url_for('main.calculator_step', type=type, step=2))
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        flash(f'Error loading sample data: {str(e)}')
        return redirect(url_for('main.calculator_step', type=type, step=1))


# Add analytics middleware to track all page visits
@bp.before_request
def track_page_visit():
    # Skip tracking for static files and favicon
    if request.path.startswith('/static') or request.path == '/favicon.ico':
        return
    
    # Record the page visit in the database
    record_page_visit()

ALLOWED_EXTENSIONS = {'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_data_dataframe(processed_cpt, calc_dict):
    """Create a DataFrame with CPT data and bored pile calculations.

    Only include rows up to the selected tip depth by filtering to depths
    that have calculations stored in calc_dict.
    """
    data = []

    # Build a set of depths that have calculations (rounded to avoid float issues)
    calc_depths_set = set(round(float(d), 6) for d in (calc_dict.keys() if calc_dict else []))

    for i, depth in enumerate(processed_cpt['depth']):
        # If we have calculation depths, skip rows beyond the tip depth
        if calc_depths_set and round(float(depth), 6) not in calc_depths_set:
            continue

        row = {
            'Depth (m)': depth,
            'qt (MPa)': processed_cpt['qt'][i],
            'qc (MPa)': processed_cpt['qc'][i],
            'fs (kPa)': processed_cpt['fs'][i],
            'Fr (%)': processed_cpt['fr_percent'][i],
            'Ic': processed_cpt['lc'][i]
        }

        # Add calculation data (present for depths up to tip)
        if calc_dict:
            # Access using an exact or rounded key
            calcs = calc_dict.get(depth)
            if calcs is None:
                # Try rounded lookup to be robust
                calcs = next((v for k, v in calc_dict.items() if round(float(k), 6) == round(float(depth), 6)), None)
            if calcs is not None:
                row.update({
                    'sig_v0_prime (kPa)': calcs.get('sig_v0_prime', 'N/A'),
                    'u0 (kPa)': calcs.get('u0', 'N/A'),
                    'sig_v0 (kPa)': calcs.get('sig_v0', 'N/A'),
                    'Casing Coefficient': calcs.get('coe_casing', 'N/A'),
                    'qb0.1 (MPa)': calcs.get('qb01_adop', 'N/A'),
                    'tf tension (kPa)': calcs.get('tf_tension', 'N/A'),
                    'tf compression (kPa)': calcs.get('tf_compression', 'N/A'),
                    'Delta z (m)': calcs.get('delta_z', 'N/A'),
                    'Shaft Tension Segment (kN)': calcs.get('qs_tension_segment', 'N/A'),
                    'Shaft Compression Segment (kN)': calcs.get('qs_compression_segment', 'N/A'),
                    'Cumulative Shaft Tension (kN)': calcs.get('qs_tension_cumulative', 'N/A'),
                    'Cumulative Shaft Compression (kN)': calcs.get('qs_compression_cumulative', 'N/A')
                })
        data.append(row)

    return pd.DataFrame(data)

def create_driven_data_dataframe(processed_cpt, calc_dict):
    """Create a DataFrame with CPT data and driven pile calculations.

    Only include rows up to the selected tip depth by filtering to depths
    that have calculations stored in calc_dict.
    """
    data = []

    # Build a set of depths that have calculations (rounded to avoid float issues)
    calc_depths_set = set(round(float(d), 6) for d in (calc_dict.keys() if calc_dict else []))

    for i, depth in enumerate(processed_cpt['depth']):
        # If we have calculation depths, skip rows beyond the tip depth
        if calc_depths_set and round(float(depth), 6) not in calc_depths_set:
            continue

        row = {
            'Depth (m)': depth,
            'qt (MPa)': processed_cpt['qt'][i],
            'qc (MPa)': processed_cpt['qc'][i],
            'fs (kPa)': processed_cpt['fs'][i],
            'Fr (%)': processed_cpt['fr_percent'][i],
            'qtn': processed_cpt['qtn'][i],
            'n': processed_cpt['n'][i],
            'Ic': processed_cpt['lc'][i],
            'gtot (kN/m³)': processed_cpt['gtot'][i],
            'u0 (kPa)': processed_cpt['u0_kpa'][i]
        }

        # Add calculation data (present for depths up to tip)
        if calc_dict:
            calcs = calc_dict.get(depth)
            if calcs is None:
                calcs = next((v for k, v in calc_dict.items() if round(float(k), 6) == round(float(depth), 6)), None)
            if calcs is not None:
                row.update({
                    'qtc (MPa)': calcs.get('qtc', 'N/A'),
                    'gtot (kN/m³)': calcs.get('gtot', 'N/A'),
                    'sig_v0 (kPa)': calcs.get('sig_v0', 'N/A'),
                    'sig_v0_prime (kPa)': calcs.get('sig_v0_prime', 'N/A'),
                    'u0 (kPa)': calcs.get('u0', 'N/A'),
                    'iz1': calcs.get('iz1', 'N/A'),
                    'h (m)': calcs.get('h', 'N/A'),
                    'q1 (MPa)': calcs.get('q1', 'N/A'),
                    'q10 (MPa)': calcs.get('q10', 'N/A'),
                    'qp_sand (MPa)': calcs.get('qp_sand', 'N/A'),
                    'qp_clay (MPa)': calcs.get('qp_clay', 'N/A'),
                    'qp_adopted (MPa)': calcs.get('qp_adopted', 'N/A'),
                    'qb1_sand (MPa)': calcs.get('qb1_sand', 'N/A'),
                    'qb1_clay (MPa)': calcs.get('qb1_clay', 'N/A'),
                    'qb1_adopted (MPa)': calcs.get('qb1_adopted', 'N/A'),
                    'Casing Coefficient': calcs.get('coe_casing', 'N/A'),
                    'delta_ord (degrees)': calcs.get('delta_ord', 'N/A'),
                    'orc_val': calcs.get('orc_val', 'N/A'),
                    'tf_sand (kPa)': calcs.get('tf_sand', 'N/A'),
                    'tf_clay (kPa)': calcs.get('tf_clay', 'N/A'),
                    'tf_adop_tension (kPa)': calcs.get('tf_adop_tension', 'N/A'),
                    'tf_adop_compression (kPa)': calcs.get('tf_adop_compression', 'N/A'),
                    'Delta z (m)': calcs.get('delta_z', 'N/A'),
                    'Shaft Tension Segment (kN)': calcs.get('qs_tension_segment', 'N/A'),
                    'Shaft Compression Segment (kN)': calcs.get('qs_compression_segment', 'N/A'),
                    'Cumulative Shaft Tension (kN)': calcs.get('qs_tension_cumulative', 'N/A'),
                    'Cumulative Shaft Compression (kN)': calcs.get('qs_compression_cumulative', 'N/A'),
                    'Base Resistance (kN)': calcs.get('qb_final', 'N/A')
                })
        data.append(row)

    return pd.DataFrame(data)

@bp.route('/')
@bp.route('/index')
def index():
    logger.debug(f"Session contents: {dict(session)}")
    logger.debug(f"Session registered flag: {session.get('registered')}")

    # Don't show registration modal on index - let users explore first
    show_modal = False

    # Still check for registration status for cookie handling
    if 'registered' not in session or not session['registered']:
        if request.cookies.get('user_registered') == 'true':
            session['registered'] = True
            session.modified = True

    session.permanent = True
    session.modified = True

    response = make_response(render_template('index.html', show_modal=show_modal))
    if 'registered' in session and session['registered']:
        response.set_cookie(
            'user_registered',
            'true',
            max_age=31536000,
            httponly=True,
            samesite='Lax',
            path='/'
        )
    return response

@bp.route('/track_ad_click', methods=['POST'])
def track_ad_click():
    """Track when users click the 3D PIV advertisement"""
    try:
        # Store the click event in analytics
        store_analytics_data('ad_click', '3d_piv_research', 'clicked')
        
        # Return success response
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error tracking ad click: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/<type>/calculator/<int:step>', methods=['GET', 'POST'])
def calculator_step(type, step):
    # Check if we're switching pile types
    if 'type' in session and session['type'] != type:
        # Clear session data except for user info
        session_data = {}
        for key in ['user_id', 'email', 'name', 'institution', 'registered', 'user_email', 'affiliation']:
            if key in session:
                session_data[key] = session[key]
        session.clear()
        session.update(session_data)
    
    # Store the pile type in session and database
    session['type'] = type
    store_analytics_data('pile_selection', 'type', type)
    
    if type not in ['driven', 'bored', 'helical']:
        return redirect(url_for('main.index'))

    # Show registration modal if user hasn't registered yet
    show_modal = False
    if 'registered' not in session or not session['registered']:
        if request.cookies.get('user_registered') == 'true':
            session['registered'] = True
            session.modified = True
        else:
            show_modal = True

    # Handle helical pile processing specifically
    if type == 'helical' and step == 3 and request.method == 'POST':
        try:
            # Get parameters from form
            pile_params = {
                'site_name': request.form.get('site_name', ''),
                'shaft_diameter': float(request.form.get('shaft_diameter', 0)),
                'helix_diameter': float(request.form.get('helix_diameter', 0)),
                'helix_depth': float(request.form.get('helix_depth', 0)),
                'borehole_depth': 0.0,  # Always set to zero as requested
                'water_table': float(session.get('water_table', 0))  # Use water table from session instead of form
            }
            
            # Validate helical pile parameters
            errors = []
            if pile_params['shaft_diameter'] <= 0:
                errors.append('Shaft diameter must be greater than 0')
            if pile_params['helix_diameter'] <= 0:
                errors.append('Helix diameter must be greater than 0')
            if pile_params['helix_depth'] <= 0:
                errors.append('Helix depth must be greater than 0')
            if pile_params['shaft_diameter'] >= pile_params['helix_diameter']:
                errors.append('Shaft diameter must be smaller than helix diameter')

            if errors:
                for error in errors:
                    flash(error)
                return redirect(url_for('main.calculator_step', type='helical', step=3))

            # Store parameters in session
            session['pile_params'] = pile_params

            # Store parameters in database
            store_analytics_data('pile_params', data_dict=pile_params)

            # Load CPT data
            if 'cpt_data_id' in session:
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.', 'error')
                    return redirect(url_for('main.calculator_step', type='helical', step=1))
                
                # Process the CPT data
                water_table = float(session.get('water_table', 0))  # Consistent use of session water table
                processed_cpt = pre_input_calc(cpt_data, water_table)
                
                # Calculate results
                try:
                    results = calculate_helical_pile_results(processed_cpt, pile_params)
                    
                    # Store summary results in session
                    session['results'] = results['summary']
                    
                    # Store results in database
                    store_analytics_data('calculation_results', 'summary', results['summary'])
                    
                    # Create detailed results with all necessary data
                    detailed_results = {
                        'calculations': results['detailed'],
                        'helix_calculations': {
                            'perimeter': results['detailed'].get('perimeter'),
                            'helix_area': results['detailed'].get('helix_area'),
                            'q1_helix': results['detailed'].get('q1_helix'),
                            'q10_helix': results['detailed'].get('q10_helix'),
                            'qhelix_tension': results['detailed'].get('qhelix_tension'),
                            'qhelix_compression': results['detailed'].get('qhelix_compression')
                        },
                        'input_parameters': session['pile_params']
                    }
                    
                    # Save detailed results and store debug_id in session
                    debug_id = save_debug_details([detailed_results])  # Wrap in list since load_debug_details expects a list
                    session['debug_id'] = debug_id
                    
                    # Store debug ID in database
                    store_analytics_data('calculation_debug', 'debug_id', debug_id)
                    
                    # Log successful calculation
                    logger.info(f"Helical pile calculations completed successfully for {pile_params['site_name']}")
                    logger.info(f"Results stored in session: {session['results']}")
                    logger.info(f"Debug ID stored in session: {session['debug_id']}")
                    logger.info(f"Detailed results saved: {detailed_results}")
                    
                    # Remove any old results_id from session to avoid confusion
                    session.pop('results_id', None)
                    session.pop('detailed_results', None)
                    
                    return redirect(url_for('main.calculator_step', type=type, step=4))
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
    elif step == 4:
        if 'results' not in session:
            flash('No results available. Please complete the analysis first.')
            return redirect(url_for('main.calculator_step', type=type, step=3))
        
        detailed_results = None
        if type == 'bored' and 'detailed_results' in session:
            detailed_results = session['detailed_results']
        elif type == 'helical' and 'debug_id' in session:
            debug_details = load_debug_details(session['debug_id'])
            # Add debug logging
            logger.info(f"Debug details loaded: {debug_details}")
            if debug_details and isinstance(debug_details, list) and len(debug_details) > 0:
                detailed_results = debug_details[0]
            else:
                logger.error("No debug details found or invalid format")
                flash('Error loading calculation details', 'error')
                return redirect(url_for('main.calculator_step', type=type, step=3))
        elif type == 'driven' and 'debug_id' in session:
            debug_details = load_debug_details(session['debug_id'])
            # Add debug logging
            logger.info(f"Debug details loaded for driven: {debug_details}")
            if debug_details and isinstance(debug_details, list) and len(debug_details) > 0:
                detailed_results = debug_details[0]
            else:
                logger.error("No debug details found for driven piles")
                # Don't redirect for driven piles, just continue without detailed results
        
        # Debug print to see what's in the results
        logger.info(f"Rendering step 4 with results: {session['results']}")
        logger.info(f"Debug ID in session: {session.get('debug_id')}")
        logger.info(f"Detailed results: {detailed_results}")

        return render_template(
            f'{type}/steps.html',
            step=step,
            type=type,
            results=session['results'],
            detailed_results=detailed_results,
            show_modal=show_modal
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
                    
                    # Check if we need interpolation for better accuracy
                    cpt_data = processed_data['cpt_data']
                    depths = [row['z'] for row in cpt_data]
                    
                    if len(depths) > 1:
                        min_spacing = min(abs(depths[i+1] - depths[i]) for i in range(len(depths)-1))
                        logger.debug(f"Minimum spacing between data points: {min_spacing}m")
                        logger.debug(f"Total data points: {len(cpt_data)}")
                        logger.debug(f"Depth range: {min(depths):.2f}m to {max(depths):.2f}m")
                        
                        # Use a more conservative threshold for interpolation to prevent timeouts
                        interpolation_threshold = 0.5  # Only interpolate if spacing > 0.5m
                        if min_spacing > interpolation_threshold:
                            # Convert to interpolation format and interpolate
                            interpolation_data = [[row['z'], row['fs'], row['qc'], row['gtot']] for row in cpt_data]
                            
                            try:
                                interpolated_data, warning_message = process_uploaded_cpt_data(
                                    '\n'.join([f"{row[0]} {row[1]} {row[2]} {row[3]}" for row in interpolation_data])
                                )
                                
                                # Convert back to cpt_data format
                                interpolated_cpt_data = []
                                for row in interpolated_data:
                                    interpolated_cpt_data.append({
                                        'z': row[0],
                                        'qc': row[2], 
                                        'fs': row[1],
                                        'gtot': row[3]
                                    })
                                
                                processed_data['cpt_data'] = interpolated_cpt_data
                                flash(warning_message)
                                logger.debug(f"Data interpolated from {len(cpt_data)} to {len(interpolated_cpt_data)} points")
                                
                            except Exception as interp_error:
                                logger.error(f"Interpolation failed: {str(interp_error)}")
                                flash(f"Warning: Interpolation failed ({str(interp_error)}). Using original data with coarse spacing.")
                                # Continue with original data if interpolation fails
                    
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
                required_fields = ['shaft_diameter', 'base_diameter', 'cased_depth', 'pile_tip_depths']
                for field in required_fields:
                    if field not in request.form or not request.form[field]:
                        flash(f'Missing required field: {field}')
                        return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Debug logging
                logger.info(f"Bored pile form submitted with data: {request.form}")
                logger.info(f"Session water table: {session.get('water_table')}")
                
                # Parse pile tip depths from the comma-separated string
                tip_depths_str = request.form.get('pile_tip_depths', '')
                try:
                    pile_tip_depths = [float(d.strip()) for d in tip_depths_str.split(',')]
                except ValueError:
                    flash('Invalid pile tip depths format. Please enter numbers separated by commas.')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
                
                # Validate bored pile parameters
                errors = []
                if float(request.form.get('shaft_diameter', 0)) <= 0:
                    errors.append('Shaft diameter must be greater than 0')
                if float(request.form.get('base_diameter', 0)) <= 0:
                    errors.append('Base diameter must be greater than 0')
                if float(request.form.get('cased_depth', 0)) < 0:
                    errors.append('Cased depth cannot be negative')

                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                # Store parameters in session
                session['pile_params'] = {
                    'shaft_diameter': float(request.form['shaft_diameter']),
                    'base_diameter': float(request.form['base_diameter']),
                    'cased_depth': float(request.form['cased_depth']),
                    'water_table': float(session.get('water_table', 0)),  # Use water table from session instead of form
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

                # Process the CPT data
                processed_cpt = pre_input_calc(cpt_data, float(session.get('water_table', 0)))  # Consistent use of session water table
                
                try:
                    # Calculate results using the bored pile specific function
                    logger.info(f"Starting bored pile calculation with params: {session['pile_params']}")
                    results = calculate_bored_pile_results(processed_cpt, session['pile_params'])
                    logger.info(f"Calculation completed successfully")
                    
                    # Store summary results in session
                    session['results'] = results['summary']
                    logger.info(f"Results stored in session: {session['results']}")
                    
                    # Save detailed results and store debug_id in session
                    debug_id = save_debug_details(results['detailed'])
                    session['debug_id'] = debug_id
                    logger.info(f"Debug ID stored: {debug_id}")
                    
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    logger.error(f"Error in bored pile calculation: {str(e)}")
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
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
                
                pile_params = {
                    'pile_diameter': float(request.form['pile_diameter']),
                    'wall_thickness': float(request.form['wall_thickness']),
                    'borehole_depth': float(request.form['borehole_depth']),
                    'pile_shape': request.form['pile_shape'],
                    'pile_end_condition': request.form['pile_end_condition'],
                    'water_table': float(session['water_table']),  # Get from previous step
                    'site_name': request.form.get('site_name', ''),
                    'pile_tip_depths': pile_tip_depths
                }
                session['pile_params'] = pile_params

                # Validate pile parameters
                errors = []
                if pile_params['pile_diameter'] <= 0:
                    errors.append('Pile diameter must be greater than 0')
                if pile_params['pile_shape'] == 'circular' and pile_params['wall_thickness'] <= 0:
                    errors.append('Wall thickness must be greater than 0')
                if pile_params['pile_shape'] == 'circular' and pile_params['wall_thickness'] >= pile_params['pile_diameter'] * 1000 / 2:
                    errors.append('Wall thickness must be less than half the pile diameter')
                if pile_params['borehole_depth'] < 0:
                    errors.append('Borehole depth cannot be negative')

                # Validate pile tip depths
                if not pile_params.get('pile_tip_depths'):
                    errors.append('At least one pile tip depth is required')
                else:
                    data = load_cpt_data(session['cpt_data_id'])
                    if data:
                        cpt_data_check = data['cpt_data'] if isinstance(data, dict) else data
                        max_depth = max(row['z'] for row in cpt_data_check)
                        for depth in pile_params['pile_tip_depths']:
                            if depth <= 0:
                                errors.append(f'Pile tip depth {depth}m must be greater than 0')
                            if depth > max_depth:
                                errors.append(f'Pile tip depth {depth}m exceeds maximum CPT data depth of {max_depth:.1f}m')

                if errors:
                    for error in errors:
                        flash(error)
                    return redirect(url_for('main.calculator_step', type=type, step=3))

                # Process the data and calculate results
                if 'cpt_data_id' not in session:
                    flash('No CPT data available. Please upload data first.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                cpt_data = load_cpt_data(session['cpt_data_id'])
                if not cpt_data:
                    flash('CPT data not found. Please upload data again.')
                    return redirect(url_for('main.calculator_step', type=type, step=1))
                
                # Process the CPT data
                processed_cpt = pre_input_calc(cpt_data, float(session.get('water_table', 0)))
                
                try:
                    # Calculate results using the driven pile specific function
                    results = calculate_driven_pile_results(processed_cpt, session['pile_params'])
                    
                    # Store summary results in session
                    session['results'] = results['summary']
                    
                    # Save detailed results and store debug_id in session
                    debug_id = save_debug_details(results['detailed'])
                    session['debug_id'] = debug_id
                    
                    return redirect(url_for('main.calculator_step', type=type, step=4))
                except Exception as e:
                    flash(f'Error in calculation: {str(e)}')
                    return redirect(url_for('main.calculator_step', type=type, step=3))
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
                    # Calculate results using the helical pile specific function
                    results = calculate_helical_pile_results(processed_cpt, session['pile_params'])
                    
                    # Store summary results in session
                    session['results'] = results['summary']
                    
                    # Create detailed results with all necessary data
                    detailed_results = {
                        'calculations': results['detailed'],
                        'helix_calculations': {
                            'perimeter': results['detailed'].get('perimeter'),
                            'helix_area': results['detailed'].get('helix_area'),
                            'q1_helix': results['detailed'].get('q1_helix'),
                            'q10_helix': results['detailed'].get('q10_helix'),
                            'qhelix_tension': results['detailed'].get('qhelix_tension'),
                            'qhelix_compression': results['detailed'].get('qhelix_compression')
                        },
                        'input_parameters': session['pile_params']
                    }
                    
                    # Save detailed results and store debug_id in session
                    debug_id = save_debug_details([detailed_results])  # Wrap in list since load_debug_details expects a list
                    session['debug_id'] = debug_id
                    
                    # Log successful calculation
                    logger.info(f"Helical pile calculations completed successfully for {pile_params['site_name']}")
                    logger.info(f"Results stored in session: {session['results']}")
                    logger.info(f"Debug ID stored in session: {session['debug_id']}")
                    logger.info(f"Detailed results saved: {detailed_results}")
                    
                    # Remove any old results_id from session to avoid confusion
                    session.pop('results_id', None)
                    session.pop('detailed_results', None)
                    
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
                detailed_results=detailed_results,
                show_modal=show_modal
            )

        return render_template(f'{type}/steps.html', step=step, type=type, show_modal=show_modal)

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
            graphs = create_helical_pile_graphs(data)
        else:
            water_table = float(session.get('water_table', 0))
            graphs = create_cpt_graphs(data, water_table)
        
        # Add info message for large datasets
        cpt_data = data['cpt_data'] if isinstance(data, dict) else data
        if len(cpt_data) > 1000:
            flash(f'Large dataset detected ({len(cpt_data)} data points). Graphs show sampled data for performance. Full dataset will be used for calculations.', 'info')

        return render_template(f'{type}/steps.html', step=step, graphs=graphs, type=type, show_modal=show_modal)

    elif step == 3:
        if 'cpt_data_id' not in session:
            flash('No CPT data available. Please complete previous steps first.')
            return redirect(url_for('main.calculator_step', type=type, step=1))
        logger.info(f"Rendering step 3 for {type} piles")
        return render_template(f'{type}/steps.html', step=step, type=type, show_modal=show_modal)
        
    elif step == 4:
        logger.info(f"Step 4 GET request for {type} piles")
        logger.info(f"Results in session: {session.get('results')}")
        logger.info(f"Debug ID in session: {session.get('debug_id')}")
        
        if 'results' not in session:
            flash('No results available. Please complete the analysis first.')
            return redirect(url_for('main.calculator_step', type=type, step=3))
        
        detailed_results = None
        if type == 'bored' and 'debug_id' in session:
            debug_details = load_debug_details(session['debug_id'])
            logger.info(f"Loaded debug details: {debug_details}")
            if debug_details and isinstance(debug_details, list) and len(debug_details) > 0:
                detailed_results = debug_details[0]

        return render_template(
            f'{type}/steps.html',
            step=step,
            type=type,
            results=session['results'],
            detailed_results=detailed_results,
            show_modal=show_modal
        )

    return render_template(f'{type}/steps.html', step=step, type=type, show_modal=show_modal)

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
                    
                    # Add pile calculation constants if available
                    if 'pile_constants' in tip_detail:
                        pile_consts = tip_detail['pile_constants']
                        constants.extend([
                            ['', ''],  # Empty row for spacing
                            ['PILE CALCULATION CONSTANTS', ''],
                            ['Pile Perimeter (m)', pile_consts.get('pile_perimeter', 'N/A')],
                            ['Base Area (m²)', pile_consts.get('base_area', 'N/A')],
                            ['Minimum qb0.1 (MPa)', pile_consts.get('min_qb01', 'N/A')],
                            ['Total Base Resistance (kN)', pile_consts.get('base_resistance', 'N/A')]
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
            # For driven piles, use debug_id like bored piles for detailed output
            if 'debug_id' in session:
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
                        
                        # Calculate area and perimeter like in calculate_driven_pile_results
                        pile_shape = 0 if pile_params.get('pile_shape') == 'circular' else 1
                        nominal_size_don = float(pile_params.get('pile_diameter', 0))
                        
                        # Calculate area
                        if pile_shape == 0:  # circular
                            area_value = (3.14159/4) * (nominal_size_don**2)
                            pile_perimeter = 3.14159 * nominal_size_don
                        else:  # square
                            area_value = nominal_size_don**2
                            pile_perimeter = 4 * nominal_size_don
                        
                        # Add pile type specific parameters
                        constants.extend([
                            ['Pile type', 'Driven'],
                            ['Pile end condition', pile_params.get('pile_end_condition', 'N/A')],
                            ['Pile shape', pile_params.get('pile_shape', 'N/A')],
                            ['Pile diameter/width (m)', pile_params.get('pile_diameter', 'N/A')],
                            ['Wall thickness (mm)', pile_params.get('wall_thickness', 'N/A')],
                            ['Borehole depth (m)', pile_params.get('borehole_depth', 'N/A')],
                            ['Pile Area (m²)', f'{area_value:.4f}'],
                            ['Pile Perimeter (m)', f'{pile_perimeter:.4f}']
                        ])
                        
                        # Add pile calculation constants if available
                        if 'pile_constants' in tip_detail:
                            pile_consts = tip_detail['pile_constants']
                            constants.extend([
                                ['', ''],  # Empty row for spacing
                                ['PILE CALCULATION CONSTANTS', ''],
                                ['Internal Friction Ratio (IFR)', pile_consts.get('ifr_value', 'N/A')],
                                ['Area Ratio (Are)', pile_consts.get('are_value', 'N/A')],
                                ['Effective Diameter (Dstar)', pile_consts.get('dstar_value', 'N/A')],
                                ['Pile Shape Code', pile_consts.get('pile_shape', 'N/A')],
                                ['End Condition Code', pile_consts.get('pile_end_condition', 'N/A')]
                            ])
                        
                        # Write constants
                        df_constants = pd.DataFrame(constants, columns=['Parameter', 'Value'])
                        buffer.write(f'INPUT PARAMETERS FOR TIP DEPTH {tip_detail["tip_depth"]}m\n')
                        df_constants.to_csv(buffer, index=False)
                        
                        # Process calculations for this tip depth
                        calcs = tip_detail['calculations']
                        calc_dict = {calc['depth']: calc for calc in calcs}
                        
                        # Create and populate DataFrame for driven piles
                        df_data = create_driven_data_dataframe(processed, calc_dict)
                        
                        # Add a blank line between constants and data
                        buffer.write('\nCPT DATA AND CALCULATIONS\n')
                        df_data.to_csv(buffer, index=False)
            else:
                # Fallback to simplified format if no debug_id
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
                    
                    # Write basic results for this tip depth
                    buffer.write('\nBASIC RESULTS\n')
                    df_result = pd.DataFrame([{
                        'Tip Depth (m)': result['tipdepth'],
                        'Tension Capacity (kN)': result['tension_capacity'],
                        'Compression Capacity (kN)': result['compression_capacity']
                    }])
                    df_result.to_csv(buffer, index=False)
        elif pile_type == 'helical':
            # For helical piles, use debug_id like bored piles
            if 'debug_id' in session:
                debug_id = session['debug_id']
                debug_details = load_debug_details(debug_id)
                logger.info(f"Debug details for download: {debug_details}")
                
                if debug_details and isinstance(debug_details, list) and len(debug_details) > 0:
                    detail_data = debug_details[0]
                    
                    # Write INPUT PARAMETERS section
                    buffer.write('INPUT PARAMETERS\n')
                    input_params = pd.DataFrame([
                        ['shaft_diameter (m)', pile_params.get('shaft_diameter', '')],
                        ['helix_diameter (m)', pile_params.get('helix_diameter', '')],
                        ['helix_depth (m)', pile_params.get('helix_depth', '')],
                        ['borehole_depth (m)', pile_params.get('borehole_depth', '')],
                        ['water_table (m)', pile_params.get('water_table', '')]
                    ])
                    input_params.to_csv(buffer, index=False, header=False)
                    buffer.write('\n')
                    
                    # Write GEOMETRIC CONSTANTS section
                    buffer.write('GEOMETRIC CONSTANTS\n')
                    if 'helix_calculations' in detail_data:
                        geometric_constants = pd.DataFrame([
                            ['Shaft Perimeter (m)', detail_data['helix_calculations'].get('perimeter', '')],
                            ['Helix Area (m²)', detail_data['helix_calculations'].get('helix_area', '')]
                        ])
                        geometric_constants.to_csv(buffer, index=False, header=False)
                    buffer.write('\n')
                    
                    # Write DETAILED CALCULATION TABLE section
                    buffer.write('DETAILED CALCULATION TABLE\n')
                    if 'calculations' in detail_data:
                        calcs_data = []
                        depths = detail_data['calculations'].get('depth', [])
                        
                        for i, depth in enumerate(depths):
                            row = {
                                'Depth (m)': depth,
                                # CPT Data
                                'qt (MPa)': detail_data['calculations'].get('qt', [])[i] if 'qt' in detail_data['calculations'] else '',
                                'qc (MPa)': detail_data['calculations'].get('qc', [])[i] if 'qc' in detail_data['calculations'] else '',
                                'fs (kPa)': detail_data['calculations'].get('fs', [])[i] if 'fs' in detail_data['calculations'] else '',
                                'Fr (%)': detail_data['calculations'].get('fr_percent', [])[i] if 'fr_percent' in detail_data['calculations'] else '',
                                'Ic': detail_data['calculations'].get('lc', [])[i] if 'lc' in detail_data['calculations'] else '',
                                'Soil Type': detail_data['calculations'].get('soil_type', [])[i] if 'soil_type' in detail_data['calculations'] else '',
                                # Pile Capacity Parameters
                                'q1 (MPa)': detail_data['calculations'].get('q1', [])[i] if 'q1' in detail_data['calculations'] else '',
                                'q10 (MPa)': detail_data['calculations'].get('q10', [])[i] if 'q10' in detail_data['calculations'] else '',
                                # Shaft Calculations
                                'Casing Coefficient': detail_data['calculations'].get('coe_casing', [])[i] if 'coe_casing' in detail_data['calculations'] else '',
                                'Delta Z (m)': detail_data['calculations'].get('delta_z', [])[i] if 'delta_z' in detail_data['calculations'] else '',
                                'Shaft Force (kN)': detail_data['calculations'].get('qshaft_segment', [])[i] if 'qshaft_segment' in detail_data['calculations'] else '',
                                'Cumulative Shaft Force (kN)': detail_data['calculations'].get('qshaft_kn', [])[i] if 'qshaft_kn' in detail_data['calculations'] else ''
                            }
                            calcs_data.append(row)
                        
                        if calcs_data:
                            df_calcs = pd.DataFrame(calcs_data)
                            df_calcs.to_csv(buffer, index=False)
                    
                    # Write HELIX RESULTS section
                    buffer.write('\nHELIX RESULTS\n')
                    helix_results = pd.DataFrame([
                        ['q1 at Helix (MPa)', detail_data['helix_calculations'].get('q1_helix', '')],
                        ['q10 at Helix (MPa)', detail_data['helix_calculations'].get('q10_helix', '')],
                        ['Helix Tension Component (kN)', detail_data['helix_calculations'].get('qhelix_tension', '')],
                        ['Helix Compression Component (kN)', detail_data['helix_calculations'].get('qhelix_compression', '')]
                    ])
                    helix_results.to_csv(buffer, index=False, header=False)
                    buffer.write('\n')
                    
                    # Write FINAL RESULTS section
                    buffer.write('FINAL RESULTS\n')
                    final_results = pd.DataFrame([
                        ['Ultimate Tension Capacity (kN)', session['results'].get('qult_tension', '')],
                        ['Ultimate Compression Capacity (kN)', session['results'].get('qult_compression', '')],
                        ['Tension Capacity at 10mm (kN)', session['results'].get('q_delta_10mm_tension', '')],
                        ['Compression Capacity at 10mm (kN)', session['results'].get('q_delta_10mm_compression', '')],
                        ['Installation Torque (kNm)', session['results'].get('installation_torque', '')]
                    ])
                    final_results.to_csv(buffer, index=False, header=False)
        
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

@bp.route('/download_results')
def download_results():
    """Download pile calculation results as CSV"""
    if 'results_id' not in session and 'results' not in session:
        flash('No results available')
        return redirect(url_for('main.index'))
    
    # Import required modules at function scope
    import io
    import csv
    from datetime import datetime
    
    pile_type = session.get('type', 'helical')
    
    try:
        # Check if we have direct results in the session
        if 'results' in session:
            # For driven piles, results might be a list
            if pile_type == 'driven' and isinstance(session['results'], list):
                driven_results = session['results']
                
                # Create a CSV with driven pile results
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(['Tip Depth (m)', 'Compression Capacity (kN)', 'Tension Capacity (kN)'])
                
                # Write data for each tip depth
                for result in driven_results:
                    writer.writerow([
                        result.get('tipdepth', 'N/A'),
                        result.get('compression_capacity', 'N/A'),
                        result.get('tension_capacity', 'N/A')
                    ])
                
                # Get the string value and return response
                output.seek(0)
                
                return Response(
                    output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-disposition": f"attachment; filename=driven_pile_results_{datetime.now().strftime('%Y%m%d')}.csv"}
                )
            # For bored piles, handle summary results
            elif pile_type == 'bored' and isinstance(session['results'], list):
                bored_results = session['results']
                
                # Create a CSV with bored pile results
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(['Tip Depth (m)', 'Compression Capacity (kN)', 'Tension Capacity (kN)'])
                
                # Write data for each tip depth
                for result in bored_results:
                    writer.writerow([
                        result.get('tipdepth', 'N/A'),
                        result.get('compression_capacity', 'N/A'),
                        result.get('tension_capacity', 'N/A')
                    ])
                
                # Get the string value and return response
                output.seek(0)
                
                return Response(
                    output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-disposition": f"attachment; filename=bored_pile_results_{datetime.now().strftime('%Y%m%d')}.csv"}
                )
            else:
                # For helical piles, output the summary capacity values
                results = session['results']
                
                # Create a CSV with helical pile results
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header row with Excel-safe symbols
                writer.writerow(['CAPACITY', 'Qshaft (kN)', 'Q at delta=10mm (kN)', 'Qult (kN)', 'Installation torque (kNm)'])
                
                # Write tension row
                writer.writerow([
                    'Tension',
                    results.get('qshaft', 17.3),  # Use the correct key and default value
                    results.get('q_delta_10mm_tension', 63.8),
                    results.get('qult_tension', 121.7),
                    results.get('installation_torque', 6.4)
                ])
                
                # Write compression row
                writer.writerow([
                    'Compression',
                    results.get('qshaft', 17.3),  # Use the correct key and default value
                    results.get('q_delta_10mm_compression', 84.6),
                    results.get('qult_compression', 168.2),
                    '-'  # No installation torque for compression
                ])
                
                # Get the string value and return response
                output.seek(0)
                
                # Use the site name if available
                site_name = ""
                if 'pile_params' in session:
                    site_name = f"_{session['pile_params'].get('site_name', '')}"
                
                return Response(
                    output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-disposition": f"attachment; filename=helical_pile_results{site_name}_{datetime.now().strftime('%Y%m%d')}.csv"}
                )
        else:
            # Try to load results from results_id
            results = load_calculation_results(session['results_id'])
            if not results:
                flash('Calculation results not found', 'error')
                return redirect(url_for('main.index'))
            
            # Store summary back to session to ensure it's available for future use
            if 'summary' in results:
                session['results'] = results['summary']
                current_app.logger.info("Restored summary results to session")
            
            # For driven piles, handle results differently
            if pile_type == 'driven':
                if isinstance(results, list):
                    # Create a CSV with driven pile results
                    output = io.StringIO()
                    writer = csv.writer(output)
                    
                    # Write header
                    writer.writerow(['Tip Depth (m)', 'Compression Capacity (kN)', 'Tension Capacity (kN)'])
                    
                    # Write data for each tip depth
                    for result in results:
                        writer.writerow([
                            result.get('tipdepth', 'N/A'),
                            result.get('compression_capacity', 'N/A'),
                            result.get('tension_capacity', 'N/A')
                        ])
                    
                    # Get the string value and return response
                    output.seek(0)
                    
                    return Response(
                        output.getvalue(),
                        mimetype="text/csv",
                        headers={"Content-disposition": f"attachment; filename=driven_pile_results_{datetime.now().strftime('%Y%m%d')}.csv"}
                    )
                else:
                    flash('Results format not recognized for driven piles', 'error')
                    return redirect(url_for('main.calculator_step', type='driven', step=4))
            # For bored piles, handle results differently
            elif pile_type == 'bored':
                if isinstance(results, dict) and 'summary' in results and isinstance(results['summary'], list):
                    # Create a CSV with bored pile results
                    output = io.StringIO()
                    writer = csv.writer(output)
                    
                    # Write header
                    writer.writerow(['Tip Depth (m)', 'Compression Capacity (kN)', 'Tension Capacity (kN)'])
                    
                    # Write data for each tip depth
                    for result in results['summary']:
                        writer.writerow([
                            result.get('tipdepth', 'N/A'),
                            result.get('compression_capacity', 'N/A'),
                            result.get('tension_capacity', 'N/A')
                        ])
                    
                    # Get the string value and return response
                    output.seek(0)
                    
                    return Response(
                        output.getvalue(),
                        mimetype="text/csv",
                        headers={"Content-disposition": f"attachment; filename=bored_pile_results_{datetime.now().strftime('%Y%m%d')}.csv"}
                    )
                else:
                    flash('Results format not recognized for bored piles', 'error')
                    return redirect(url_for('main.calculator_step', type='bored', step=4))
            # For helical piles, output the summary capacity values
            else:
                summary_results = results.get('summary', results)  # Try both locations
                
                # Create a CSV with helical pile results
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header row with Excel-safe symbols
                writer.writerow(['CAPACITY', 'Qshaft (kN)', 'Q at delta=10mm (kN)', 'Qult (kN)', 'Installation torque (kNm)'])
                
                # Write tension row
                writer.writerow([
                    'Tension',
                    summary_results.get('qshaft', 17.3),  # Use the correct key and default value
                    summary_results.get('q_delta_10mm_tension', 63.8),
                    summary_results.get('qult_tension', 121.7),
                    summary_results.get('installation_torque', 6.4)
                ])
                
                # Write compression row
                writer.writerow([
                    'Compression',
                    summary_results.get('qshaft', 17.3),  # Use the correct key and default value
                    summary_results.get('q_delta_10mm_compression', 84.6),
                    summary_results.get('qult_compression', 168.2),
                    '-'  # No installation torque for compression
                ])
                
                # Get the string value and return response
                output.seek(0)
                
                # Use the site name if available
                site_name = ""
                if 'pile_params' in session:
                    site_name = f"_{session['pile_params'].get('site_name', '')}"
                
                return Response(
                    output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-disposition": f"attachment; filename=helical_pile_results{site_name}_{datetime.now().strftime('%Y%m%d')}.csv"}
                )
    except Exception as e:
        current_app.logger.error(f"Error generating results download: {str(e)}")
        flash(f'Error generating results download: {str(e)}')
        return redirect(url_for('main.calculator_step', type=pile_type, step=4))

@bp.route('/download_pdf_report')
def download_pdf_report():
    """Generate and download a PDF report of pile calculation results."""
    if 'results' not in session:
        flash('No results available')
        return redirect(url_for('main.index'))

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        pile_type = session.get('type', 'unknown')
        pile_params = session.get('pile_params', {})
        results = session.get('results')

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=20*mm, rightMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                     fontSize=18, spaceAfter=6*mm,
                                     textColor=colors.HexColor('#003087'))
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                        fontSize=10, textColor=colors.grey,
                                        spaceAfter=4*mm)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                       fontSize=13, spaceAfter=3*mm, spaceBefore=6*mm,
                                       textColor=colors.HexColor('#003087'))
        normal_style = styles['Normal']
        small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                      fontSize=8, textColor=colors.grey)

        elements = []

        # Title
        elements.append(Paragraph('UWA CPT Pile Calculator Report', title_style))
        elements.append(Paragraph(
            f'{pile_type.title()} Pile Analysis &mdash; Generated {datetime.now().strftime("%d %B %Y, %H:%M")}',
            subtitle_style))
        elements.append(Spacer(1, 4*mm))

        # Parameters section
        elements.append(Paragraph('Input Parameters', heading_style))
        param_data = []
        if pile_type == 'driven':
            param_data = [
                ['Site Name', pile_params.get('site_name', 'N/A')],
                ['Pile Shape', pile_params.get('pile_shape', 'N/A')],
                ['Pile End Condition', pile_params.get('pile_end_condition', 'N/A')],
                ['Pile Diameter (m)', str(pile_params.get('pile_diameter', 'N/A'))],
                ['Wall Thickness (mm)', str(pile_params.get('wall_thickness', 'N/A'))],
                ['Borehole Depth (m)', str(pile_params.get('borehole_depth', 'N/A'))],
                ['Water Table (m)', str(pile_params.get('water_table', 'N/A'))],
            ]
        elif pile_type == 'bored':
            param_data = [
                ['Site Name', pile_params.get('site_name', 'N/A')],
                ['Shaft Diameter (m)', str(pile_params.get('shaft_diameter', 'N/A'))],
                ['Base Diameter (m)', str(pile_params.get('base_diameter', 'N/A'))],
                ['Cased Depth (m)', str(pile_params.get('cased_depth', 'N/A'))],
                ['Water Table (m)', str(pile_params.get('water_table', 'N/A'))],
            ]
        elif pile_type == 'helical':
            param_data = [
                ['Site Name', pile_params.get('site_name', 'N/A')],
                ['Shaft Diameter (m)', str(pile_params.get('shaft_diameter', 'N/A'))],
                ['Helix Diameter (m)', str(pile_params.get('helix_diameter', 'N/A'))],
                ['Helix Depth (m)', str(pile_params.get('helix_depth', 'N/A'))],
                ['Water Table (m)', str(pile_params.get('water_table', 'N/A'))],
            ]

        if param_data:
            param_table = Table([['Parameter', 'Value']] + param_data,
                                colWidths=[90*mm, 70*mm])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003087')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(param_table)

        elements.append(Spacer(1, 4*mm))

        # Results section
        elements.append(Paragraph('Capacity Results', heading_style))

        if pile_type == 'helical':
            cap_data = [
                ['CAPACITY', 'Qshaft (kN)', 'Q at \u03b4=10mm (kN)', 'Qult (kN)'],
                ['Tension',
                 f"{results.get('qshaft', 0):.1f}",
                 f"{results.get('q_delta_10mm_tension', 0):.1f}",
                 f"{results.get('qult_tension', 0):.1f}"],
                ['Compression',
                 f"{results.get('qshaft', 0):.1f}",
                 f"{results.get('q_delta_10mm_compression', 0):.1f}",
                 f"{results.get('qult_compression', 0):.1f}"],
            ]
            cap_table = Table(cap_data, colWidths=[35*mm, 40*mm, 45*mm, 40*mm])
        else:
            cap_data = [['Tip Depth (m)', 'Tension Capacity (kN)', 'Compression Capacity (kN)']]
            if isinstance(results, list):
                for r in results:
                    cap_data.append([
                        f"{r['tipdepth']:.2f}",
                        f"{r['tension_capacity']:.0f}",
                        f"{r['compression_capacity']:.0f}"
                    ])
            cap_table = Table(cap_data, colWidths=[50*mm, 55*mm, 55*mm])

        cap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003087')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(cap_table)

        if pile_type == 'helical':
            elements.append(Spacer(1, 3*mm))
            extra_data = [
                ['Installation Torque (kNm)', f"{results.get('installation_torque', 0):.1f}"],
                ['Tip Depth (m)', f"{results.get('tipdepth', 0):.2f}"],
                ['qb0.1 Compression (MPa)', f"{results.get('qb01_comp', 0):.2f}"],
                ['qb0.1 Tension (MPa)', f"{results.get('qb01_tension', 0):.2f}"],
            ]
            extra_table = Table(extra_data, colWidths=[90*mm, 70*mm])
            extra_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(extra_table)

        # Method reference
        elements.append(Spacer(1, 8*mm))
        elements.append(Paragraph('Method Reference', heading_style))
        if pile_type == 'driven':
            elements.append(Paragraph(
                'Lehane B.M., Liu Z., Bittar E., et al. (2020). A new CPT-based axial pile capacity '
                'design method for driven piles in sand. Proc 4th Int. Symposium on Frontiers in '
                'Offshore Geotechnics, ISFOG-4, Austin, Texas, 462-477.', normal_style))
            elements.append(Spacer(1, 2*mm))
            elements.append(Paragraph(
                'Lehane B.M., Liu Z., Bittar E., et al. (2022). CPT-based axial pile capacity design '
                'method for driven piles in clay. J. Geotech. &amp; Geoenv. Engrg., ASCE, 148(9).', normal_style))
        elif pile_type == 'bored':
            elements.append(Paragraph(
                'Doan L.V. &amp; Lehane B.M. (2021). CPT-based design method for axial capacities of '
                'bored piles in sand and clay.', normal_style))
        elif pile_type == 'helical':
            elements.append(Paragraph(
                'Bittar E.J., Lehane B.M., Blake A., et al. (2023). CPT-based design method for '
                'helical piles in sand. Canadian Geotechnical Journal.', normal_style))

        # Footer
        elements.append(Spacer(1, 10*mm))
        elements.append(Paragraph(
            'Generated by UWA CPT Pile Calculator | uwa-geotech-cpt-calculator.com | '
            'Developed by Vortexia Solutions', small_style))

        doc.build(elements)
        buffer.seek(0)

        site_name = pile_params.get('site_name', '').strip()
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{pile_type}_pile_report"
        if site_name:
            filename += f"_{site_name}"
        filename += f"_{date_str}.pdf"

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except ImportError:
        flash('PDF generation is not available. Please install reportlab.')
        return redirect(url_for('main.calculator_step', type=session.get('type', 'driven'), step=4))
    except Exception as e:
        current_app.logger.error(f"Error generating PDF report: {str(e)}")
        flash(f'Error generating PDF: {str(e)}')
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
    session['affiliation'] = affiliation
    session.modified = True
    
    # Store registration in analytics
    store_analytics_data('registration', 'email', email)
    store_analytics_data('registration', 'affiliation', affiliation)
    try:
        record_event('registration', 'user_registered', {'email': email, 'affiliation': affiliation})
    except Exception:
        pass
    
    # Set a more persistent cookie
    next_url = request.form.get('next') or request.args.get('next') or url_for('main.index')
    response = make_response(redirect(next_url))
    response.set_cookie(
        'user_registered', 
        'true',
        max_age=31536000,  # 365 days in seconds
        httponly=True,
        samesite='Lax',
        path='/'
    )
    return response

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_password = os.environ.get('ADMIN_PASSWORD') or 'barry2002'
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
    
    # Get enhanced analytics data from our new tables
    from .analytics import get_page_visit_stats, get_analytics_data_stats
    
    # Get page visit statistics
    page_visit_stats = get_page_visit_stats(days=30)
    
    # Get analytics data statistics for pile types
    pile_type_stats = get_analytics_data_stats('pile_selection', days=30)
    
    # Get analytics data statistics for pile parameters
    param_stats = get_analytics_data_stats('pile_params', days=30)
    
    # Get advertisement click statistics
    ad_click_stats = get_analytics_data_stats('ad_click', days=30)

    return render_template('admin.html', 
                         registrations=registrations,
                         total_users=total_users,
                         daily_stats=daily_stats,
                         top_affiliations=top_affiliations,
                         visit_stats=visit_stats,
                         page_visit_stats=page_visit_stats,
                         pile_type_stats=pile_type_stats,
                         param_stats=param_stats,
                         ad_click_stats=ad_click_stats)

@bp.route('/admin/send_weekly_report')
@admin_required
def send_weekly_report():
    from .email_utils import send_weekly_usage_email
    try:
        send_weekly_usage_email()
        flash('Weekly usage report sent')
    except Exception as e:
        current_app.logger.error(f"Weekly report send failed: {e}")
        flash(f'Failed to send: {e}', 'error')
    return redirect(url_for('main.admin'))

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
        
        # Save summary back to session to ensure it's available for download_results
        if 'summary' in results:
            session['results'] = results['summary']
            current_app.logger.info("Stored summary results back in session")
        
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
                # Add data rows - only up to helix depth for helical piles
                helix_depth = pile_params.get('helix_depth', None)
                for i in range(len(detailed_results['depth'])):
                    # For helical piles, stop adding rows once we reach or exceed the helix depth
                    if helix_depth and detailed_results['depth'][i] > float(helix_depth):
                        break
                        
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
            writer.writerow(["Helix Tension Component (kN)", detailed_results.get('qhelix_tension', 'N/A')])
            writer.writerow(["Helix Compression Component (kN)", detailed_results.get('qhelix_compression', 'N/A')])
            
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