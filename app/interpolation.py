import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def validate_depth_spacing(depths, min_spacing=0.1):
    """
    Validate that all depths are at least min_spacing apart.
    
    Args:
        depths (list): List of depths
        min_spacing (float): Minimum allowed spacing between depths
    
    Returns:
        bool: True if valid, False if any points are too close
    """
    sorted_depths = sorted(depths)
    differences = np.diff(sorted_depths)
    return np.all(differences >= min_spacing)

def process_uploaded_cpt_data(file_content, min_spacing=0.1, target_interval=0.1):
    """
    Process uploaded CPT data and ensure minimum spacing between points.
    If data is not granular enough, automatically interpolate to target_interval.
    
    Args:
        file_content (str): Content of the uploaded file
        min_spacing (float): Minimum spacing between points in meters (default 0.1)
        target_interval (float): Target spacing for interpolated data (default 0.1)
    
    Returns:
        tuple: (processed_data, warning_message)
            - processed_data: List of lists containing the processed data
            - warning_message: String with any warnings about the data
    """
    # Split the content into lines and remove empty lines
    lines = [line.strip() for line in file_content.split('\n') if line.strip()]
    
    # Skip header if present (assuming first line might be header)
    if any(header in lines[0].lower() for header in ['depth', 'fs', 'qc', 'ic']):
        lines = lines[1:]
    
    # Parse the data
    data = []
    for line in lines:
        # Split by whitespace or tab
        values = line.split()
        if len(values) >= 4:  # Ensure we have at least 4 values
            try:
                depth = float(values[0])
                fs = float(values[1])
                qc = float(values[2])
                ic = float(values[3])
                data.append([depth, fs, qc, ic])
            except ValueError:
                continue
    
    if not data:
        raise ValueError("No valid data found in the uploaded file")
    
    # Sort data by depth
    data.sort(key=lambda x: x[0])
    
    # Check spacing between points
    depths = [row[0] for row in data]
    if not validate_depth_spacing(depths, min_spacing):
        warning = f"Warning: Some data points are closer than {min_spacing}m apart. These points may be automatically adjusted."
        
        # Remove points that are too close together
        filtered_data = [data[0]]  # Keep first point
        last_depth = data[0][0]
        
        for point in data[1:]:
            current_depth = point[0]
            if current_depth - last_depth >= min_spacing:
                filtered_data.append(point)
                last_depth = current_depth
        
        data = filtered_data
    
    # Check if data needs interpolation for granularity
    depths = [row[0] for row in data]
    min_spacing_actual = min(np.diff(depths)) if len(depths) > 1 else float('inf')
    
    if min_spacing_actual > target_interval:
        # Data is not granular enough - interpolate to target_interval
        original_count = len(data)
        data = interpolate_cpt_data(data, target_interval)
        interpolated_count = len(data)
        
        warning = f"Data interpolated from {original_count} points to {interpolated_count} points at {target_interval}m intervals for improved calculation accuracy."
        return data, warning
    
    return data, "Data processed successfully."

def interpolate_cpt_data(data, target_interval=0.1):
    """
    Interpolate CPT data to create a dense dataset at regular intervals.
    
    Args:
        data (list of lists): Input data in format [[depth, fs, qc, ic], ...]
        target_interval (float): Target spacing between interpolated points in meters (default 0.1)
    
    Returns:
        list of lists: Interpolated data at regular intervals
    """
    if not data:
        return []
    
    # Extract columns
    depths = [row[0] for row in data]
    fs_values = [row[1] for row in data]
    qc_values = [row[2] for row in data]
    ic_values = [row[3] for row in data]
    
    # Create interpolation functions for each parameter
    fs_interp = interp1d(depths, fs_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    qc_interp = interp1d(depths, qc_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    ic_interp = interp1d(depths, ic_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Create target depth array at regular intervals
    min_depth = min(depths)
    max_depth = max(depths)
    
    # Generate depths from min_depth to max_depth at target_interval spacing
    target_depths = np.arange(min_depth, max_depth + target_interval, target_interval)
    
    # Round to avoid floating point precision issues
    target_depths = np.round(target_depths, decimals=3)
    
    # Interpolate values at target depths
    interpolated_data = []
    for depth in target_depths:
        fs_interpolated = float(fs_interp(depth))
        qc_interpolated = float(qc_interp(depth))
        ic_interpolated = float(ic_interp(depth))
        
        interpolated_data.append([depth, fs_interpolated, qc_interpolated, ic_interpolated])
    
    return interpolated_data

def format_cpt_data_for_download(data):
    """
    Format CPT data for download.
    
    Args:
        data (list of lists): Data points
    
    Returns:
        str: Formatted string for download
    """
    # Create header
    header = "Depth (m)\tfs (kPa)\tqc (MPa)\tIc\n"
    
    # Format data rows
    rows = []
    for row in data:
        # Format with full precision instead of 2 significant figures
        formatted_row = "\t".join(str(float(x)) for x in row)
        rows.append(formatted_row)
    
    return header + "\n".join(rows) 