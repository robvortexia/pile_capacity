import pandas as pd
import io
from flask import send_file
import plotly.graph_objects as go
import plotly.utils
import json
import numpy as np
import tempfile
import os
import math
from .calculations import pre_input_calc

def generate_csv_download(data, filename="results.csv"):
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    binary_buffer = io.BytesIO(buffer.getvalue().encode())
    binary_buffer.seek(0)
    return send_file(
        binary_buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    ) 

def save_cpt_data(data, water_table):
    temp_dir = tempfile.gettempdir()
    file_id = os.urandom(16).hex()
    
    with open(os.path.join(temp_dir, f'cpt_data_{file_id}.json'), 'w') as f:
        json.dump({
            'cpt_data': data,
            'water_table': water_table
        }, f)
    
    return file_id

def load_cpt_data(file_id):
    temp_dir = tempfile.gettempdir()
    try:
        with open(os.path.join(temp_dir, f'cpt_data_{file_id}.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_debug_details(debug_details):
    temp_dir = tempfile.gettempdir()
    file_id = os.urandom(16).hex()
    with open(os.path.join(temp_dir, f'debug_details_{file_id}.json'), 'w') as f:
        json.dump(debug_details, f)
    return file_id

def load_debug_details(file_id):
    temp_dir = tempfile.gettempdir()
    try:
        with open(os.path.join(temp_dir, f'debug_details_{file_id}.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_graphs_data(graphs):
    temp_dir = tempfile.gettempdir()
    file_id = os.urandom(16).hex()
    with open(os.path.join(temp_dir, f'graphs_{file_id}.json'), 'w') as f:
        json.dump(graphs, f)
    return file_id

def load_graphs_data(file_id):
    temp_dir = tempfile.gettempdir()
    try:
        with open(os.path.join(temp_dir, f'graphs_{file_id}.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def create_cpt_graphs(data):
    processed = pre_input_calc({'cpt_data': data['cpt_data']})
    if not processed:
        return None
    
    qt_graph = {
        'data': [
            go.Scatter(
                x=processed['qt'],
                y=processed['depth'],
                mode='lines',
                name='qt'
            )
        ],
        'layout': {
            'title': 'Qt vs Depth',
            'xaxis': {'title': 'Qt'},
            'yaxis': {'title': 'Depth (m)', 'autorange': 'reversed'},
            'height': 800
        }
    }
    
    fr_graph = {
        'data': [
            go.Scatter(
                x=processed['fr_percent'],
                y=processed['depth'],
                mode='lines',
                name='Fr%'
            )
        ],
        'layout': {
            'title': 'Fr% vs Depth',
            'xaxis': {'title': 'Fr%'},
            'yaxis': {'title': 'Depth (m)', 'autorange': 'reversed'},
            'height': 800
        }
    }
    
    ic_graph = {
        'data': [
            go.Scatter(
                x=processed['lc'],
                y=processed['depth'],
                mode='lines',
                name='Ic'
            )
        ],
        'layout': {
            'title': 'Ic vs Depth',
            'xaxis': {'title': 'Ic'},
            'yaxis': {'title': 'Depth (m)', 'autorange': 'reversed'},
            'height': 800
        }
    }
    
    iz_graph = {
        'data': [
            go.Scatter(
                x=processed['iz1'],
                y=processed['depth'],
                mode='lines',
                name='Iz'
            )
        ],
        'layout': {
            'title': 'Iz (when Iz < 10)',
            'xaxis': {
                'title': 'Iz',
                'range': [-5, 10],
                'dtick': 3,
                'gridcolor': 'lightgrey'
            },
            'yaxis': {
                'title': 'Depth (m)',
                'autorange': 'reversed',
                'gridcolor': 'lightgrey'
            },
            'plot_bgcolor': 'white',
            'showgrid': True,
            'height': 800
        }
    }
    
    return {
        'qt': json.dumps(qt_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'fr': json.dumps(fr_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'ic': json.dumps(ic_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'iz': json.dumps(iz_graph, cls=plotly.utils.PlotlyJSONEncoder)
    }

def create_bored_pile_graphs(data):
    processed = pre_input_calc(data)
    if not processed:
        return None
    
    qt_graph = {
        'data': [
            go.Scatter(
                x=processed['qt'],
                y=processed['depth'],
                mode='lines',
                name='qt'
            )
        ],
        'layout': {
            'title': 'Qt vs Depth',
            'xaxis': {'title': 'Qt (MPa)'},
            'yaxis': {'title': 'Depth (m)', 'autorange': 'reversed'},
            'plot_bgcolor': 'white',
            'showgrid': True
        }
    }
    
    fr_graph = {
        'data': [
            go.Scatter(
                x=processed['fr_percent'],
                y=processed['depth'],
                mode='lines',
                name='Fr%'
            )
        ],
        'layout': {
            'title': 'Fr% vs Depth',
            'xaxis': {'title': 'Fr%'},
            'yaxis': {'title': 'Depth (m)', 'autorange': 'reversed'},
            'plot_bgcolor': 'white',
            'showgrid': True
        }
    }
    
    ic_graph = {
        'data': [
            go.Scatter(
                x=processed['lc'],
                y=processed['depth'],
                mode='lines',
                name='Ic'
            )
        ],
        'layout': {
            'title': 'Ic vs Depth',
            'xaxis': {'title': 'Ic'},
            'yaxis': {'title': 'Depth (m)', 'autorange': 'reversed'},
            'plot_bgcolor': 'white',
            'showgrid': True
        }
    }
    
    return {
        'qt': json.dumps(qt_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'fr': json.dumps(fr_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'ic': json.dumps(ic_graph, cls=plotly.utils.PlotlyJSONEncoder)
    }

def validate_and_process_data(df):
    expected_columns = {
        'Depth (m)': 'z',
        'Cone resistance qt (MPa)': 'qc',
        'Cone sleeve friction, fs (kPa)': 'fs',
        'Unit weight (kN/m²)': 'gtot'
    }
    
    missing_cols = [col for col in expected_columns.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    processed_data = []
    for _, row in df.iterrows():
        processed_data.append({
            'z': float(row['Depth (m)']),
            'qc': float(row['Cone resistance qt (MPa)']),
            'fs': float(row['Cone sleeve friction, fs (kPa)']),
            'gtot': float(row['Unit weight (kN/m²)'])
        })
    
    return processed_data
