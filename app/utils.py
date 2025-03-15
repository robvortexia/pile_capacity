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
from flask import session
import uuid
from flask import current_app

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

def create_cpt_graphs(data, water_table=None):
    if water_table is None:
        water_table = data.get('water_table', 0)
    processed_data = pre_input_calc(data, water_table)
    if not processed_data:
        return None
    
    # Common layout settings
    base_layout = {
        'showline': True,
        'linewidth': 1,
        'linecolor': 'lightgrey',
        'mirror': True,
        'showgrid': True,
        'gridcolor': 'lightgrey',
        'gridwidth': 1,
        'side': 'top',
        'zeroline': False,
        'automargin': True
    }
    
    # Common graph settings
    common_layout = {
        'plot_bgcolor': 'white',
        'margin': {'l': 50, 'r': 20, 't': 30, 'b': 30},
        'font': {'size': 10},
        'autosize': True,
        'showlegend': False,
        'yaxis': {
            'title': {'text': 'Depth (m)', 'standoff': 5},
            'autorange': 'reversed',
            'range': [max(processed_data['depth']), 0],
            'dtick': 10,
            'tickfont': {'size': 10},
            **base_layout
        }
    }
    
    qt_graph = {
        'data': [
            go.Scatter(
                x=processed_data['qt'],
                y=processed_data['depth'],
                mode='lines',
                name='qt',
                line={'color': 'blue', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'qt (MPa)', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 12}, 'y': 0.95},
            'xaxis': {
                'title': None,
                'dtick': 5,
                'tickfont': {'size': 10},
                **base_layout
            },
            **common_layout
        }
    }
    
    fr_graph = {
        'data': [
            go.Scatter(
                x=processed_data['fr_percent'],
                y=processed_data['depth'],
                mode='lines',
                name='Fr%',
                line={'color': 'green', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'Fr (%)', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 12}, 'y': 0.95},
            'xaxis': {
                'title': None,
                'dtick': 2,
                'range': [0, 10],
                'tickfont': {'size': 10},
                **base_layout
            },
            **common_layout
        }
    }
    
    ic_graph = {
        'data': [
            go.Scatter(
                x=processed_data['lc'],
                y=processed_data['depth'],
                mode='lines',
                name='Ic',
                line={'color': 'purple', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'Ic', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 12}, 'y': 0.95},
            'xaxis': {
                'title': None,
                'dtick': 1,
                'range': [1, 4],
                'tickfont': {'size': 10},
                **base_layout
            },
            **common_layout
        }
    }
    
    iz_graph = {
        'data': [
            go.Scatter(
                x=processed_data['iz1'],
                y=processed_data['depth'],
                mode='lines',
                name='Iz',
                line={'color': 'red', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'Iz', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 12}, 'y': 0.95},
            'xaxis': {
                'title': None,
                'range': [0, 4],
                'dtick': 1,
                'tickfont': {'size': 10},
                **base_layout
            },
            **common_layout
        }
    }

    return {
        'qt': json.dumps(qt_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'fr': json.dumps(fr_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'ic': json.dumps(ic_graph, cls=plotly.utils.PlotlyJSONEncoder),
        'iz': json.dumps(iz_graph, cls=plotly.utils.PlotlyJSONEncoder)
    }

def create_bored_pile_graphs(data):
    # Get water table from session
    water_table = float(session['water_table'])
    processed_data = pre_input_calc(data, water_table)
    
    if not processed_data:
        return None

    # Common layout settings
    base_layout = {
        'showline': True,
        'linewidth': 1,
        'linecolor': 'lightgrey',
        'mirror': True,
        'showgrid': True,
        'gridcolor': 'lightgrey',
        'gridwidth': 1,
        'side': 'top'
    }

    qt_graph = {
        'data': [
            go.Scatter(
                x=processed_data['qt'],
                y=processed_data['depth'],
                mode='lines',
                name='qt',
                line={'color': 'blue', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'qt', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 14}},
            'xaxis': {
                'title': None,  # Removed x-axis title
                'dtick': 7,
                **base_layout
            },
            'yaxis': {
                'title': 'Depth (m)',
                'autorange': 'reversed',
                'range': [0, 100],
                'dtick': 25,
                **base_layout
            },
            'plot_bgcolor': 'white',
            'margin': {'l': 80, 'r': 30, 't': 40, 'b': 50},
            'font': {'size': 12}
        }
    }
    
    fr_graph = {
        'data': [
            go.Scatter(
                x=processed_data['fr_percent'],
                y=processed_data['depth'],
                mode='lines',
                name='Fr%',
                line={'color': 'green', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'Fr (%)', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 14}},
            'xaxis': {
                'title': None,  # Removed x-axis title
                'dtick': 3,
                **base_layout
            },
            'yaxis': {
                'title': 'Depth (m)',
                'autorange': 'reversed',
                'range': [0, 100],
                'dtick': 25,
                **base_layout
            },
            'plot_bgcolor': 'white',
            'margin': {'l': 80, 'r': 30, 't': 40, 'b': 50},
            'font': {'size': 12}
        }
    }
    
    ic_graph = {
        'data': [
            go.Scatter(
                x=processed_data['lc'],
                y=processed_data['depth'],
                mode='lines',
                name='Ic',
                line={'color': 'purple', 'width': 1.5}
            )
        ],
        'layout': {
            'title': {'text': 'Ic', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 14}},
            'xaxis': {
                'title': None,  # Removed x-axis title
                'dtick': 1,
                **base_layout
            },
            'yaxis': {
                'title': 'Depth (m)',
                'autorange': 'reversed',
                'range': [0, 100],
                'dtick': 25,
                **base_layout
            },
            'plot_bgcolor': 'white',
            'margin': {'l': 80, 'r': 30, 't': 40, 'b': 50},
            'font': {'size': 12}
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

def save_calculation_results(results):
    """Save calculation results to a file and return the file ID."""
    results_id = str(uuid.uuid4())
    results_dir = os.path.join(current_app.instance_path, 'results')
    
    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save results to file
    results_path = os.path.join(results_dir, f'{results_id}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    return results_id

def load_calculation_results(results_id):
    """Load calculation results from a file."""
    results_path = os.path.join(current_app.instance_path, 'results', f'{results_id}.json')
    
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results