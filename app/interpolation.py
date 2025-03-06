import numpy as np
import pandas as pd

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

def process_uploaded_cpt_data(file_content, min_spacing=0.1):
    """
    Process uploaded CPT data and ensure minimum spacing between points.
    
    Args:
        file_content (str): Content of the uploaded file
        min_spacing (float): Minimum spacing between points in meters (default 0.1)
    
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
        
        return filtered_data, warning
    
    return data, "Data processed successfully."

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
        # Format each number to 3 decimal places
        formatted_row = "\t".join(f"{x:.3f}" for x in row)
        rows.append(formatted_row)
    
    return header + "\n".join(rows) 