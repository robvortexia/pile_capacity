# app/helical_calculations.py

import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_helical_constants(pile_params):
    """
    Calculate basic geometric constants for helical pile analysis
    
    Args:
        pile_params (dict): Dictionary containing pile parameters including:
            - shaft_diameter (float): Diameter of pile shaft in meters
            - helix_diameter (float): Diameter of helical plate in meters
    
    Returns:
        dict: Dictionary containing:
            - perimeter (float): Shaft perimeter in meters
            - helix_area (float): Area of helical plate in square meters
    """
    
    # Extract parameters
    shaft_diameter = float(pile_params['shaft_diameter'])
    helix_diameter = float(pile_params['helix_diameter'])
    
    # Calculate perimeter of shaft
    perimeter = math.pi * shaft_diameter
    
    # Calculate area of helical plate
    helix_area = math.pi * (helix_diameter ** 2) * 0.25
    
    return {
        'perimeter': perimeter,
        'helix_area': helix_area
    }

def calculate_q1_q10(processed_cpt, helix_diameter):
    """
    Calculate q1 and q10 values iteratively for each depth
    
    Args:
        processed_cpt (dict): Dictionary containing CPT data including:
            - qt (list): Cone tip resistance values
        helix_diameter (float): Diameter of helical plate in meters
    
    Returns:
        dict: Dictionary containing:
            - q1 (list): q1 values for each depth
            - q10 (list): q10 values for each depth
    """
    
    q1_values = []
    q10_values = []
    
    for qt in processed_cpt['qt']:
        # Calculate q1 using qt * (0.1)^0.6
        q1 = qt * (0.1 ** 0.6)
        
        # Calculate q10 using qt * (0.01/D)^0.6
        q10 = qt * ((0.01/helix_diameter) ** 0.6) if helix_diameter > 0 else 0
        
        q1_values.append(q1)
        q10_values.append(q10)
    
    return {
        'q1': q1_values,
        'q10': q10_values
    }

def calculate_coe_casing_and_soil_type(processed_cpt, borehole_depth, tip_depth):
    """
    Calculate coefficient of casing and soil type for each depth
    
    Args:
        processed_cpt (dict): Dictionary containing CPT data including:
            - depth (list): Depth values
            - lc (list): Soil behavior type index values
        borehole_depth (float): Depth of borehole
        tip_depth (float): Depth to pile tip
    
    Returns:
        dict: Dictionary containing:
            - coe_casing (list): Casing coefficient values for each depth
            - soil_type (list): Soil type indicators for each depth
    """
    
    coe_casing = []
    soil_type = []
    
    for i, depth in enumerate(processed_cpt['depth']):
        # Calculate casing coefficient
        if depth < borehole_depth:
            casing = 0
        elif depth < tip_depth:
            casing = 1
        else:
            casing = 0
            
        coe_casing.append(casing)
        
        # Calculate soil type based on Ic value
        ic_value = processed_cpt['lc'][i]
        if ic_value > 2.2:
            soil = "Clay/Silt"
        else:
            soil = "Sand"
            
        soil_type.append(soil)
    
    return {
        'coe_casing': coe_casing,
        'soil_type': soil_type
    }

def calculate_delta_z_and_qshaft(processed_cpt, coe_casing, perimeter):
    """
    Calculate delta z and shaft capacity for each depth
    
    Args:
        processed_cpt (dict): Dictionary containing CPT data
        coe_casing (list): Casing coefficient values for each depth
        perimeter (float): Pile shaft perimeter in meters
    
    Returns:
        dict: Dictionary containing:
            - delta_z (list): Change in depth between consecutive points
            - qshaft_kn (list): Cumulative shaft capacity in kN
    """
    
    depths = processed_cpt['depth']
    qt_values = processed_cpt['qt']
    
    delta_z = []
    qshaft_segment = []
    qshaft_kn = []
    
    # Initialize cumulative shaft capacity
    cumulative_qshaft = 0
    
    for i in range(len(depths)):
        # Calculate delta z
        if i == 0:
            # For the first point, set delta_z to None or empty string
            current_delta_z = None
            delta_z.append(current_delta_z)
            
            # First segment should be 0
            qshaft_segment_value = 0
            qshaft_segment.append(qshaft_segment_value)
            qshaft_kn.append(0)
        else:
            # For subsequent points, calculate difference from previous depth
            current_delta_z = depths[i] - depths[i-1]
            delta_z.append(current_delta_z)
            
            # Calculate shaft capacity increment
            qshaft_segment_value = (coe_casing[i] * current_delta_z * 1000 * qt_values[i] * perimeter) / 175
            qshaft_segment.append(qshaft_segment_value)
            
            # Update cumulative shaft capacity
            cumulative_qshaft += qshaft_segment_value
            qshaft_kn.append(cumulative_qshaft)
    
    return {
        'delta_z': delta_z,
        'qshaft_segment': qshaft_segment,
        'qshaft_kn': qshaft_kn
    }

def calculate_helical_pile_capacity(processed_cpt, pile_params, qshaft_kn):
    """
    Calculate final tension and compression capacities for helical pile
    """
    
    depths = processed_cpt['depth']
    helix_depth = float(pile_params['helix_depth'])
    helix_diameter = float(pile_params['helix_diameter'])
    helix_area = math.pi * (helix_diameter ** 2) * 0.25
    
    # Find the index for the helix depth
    helix_index = min(range(len(depths)), key=lambda i: abs(depths[i] - helix_depth))
    
    # Calculate q1 and q10 values for all depths first
    q1_values = []
    q10_values = []
    for qt in processed_cpt['qt']:
        q1 = qt * (0.1 ** 0.6)
        q10 = qt * ((0.01/helix_diameter) ** 0.6) if helix_diameter > 0 else 0
        q1_values.append(q1)
        q10_values.append(q10)
    
    # Calculate q1 and q10 at helix depth
    qt_helix = processed_cpt['qt'][helix_index]
    q1_helix = q1_values[helix_index]
    q10_helix = q10_values[helix_index]
    
    # Calculate helix capacities
    qhelix_tension = q10_helix * helix_area * 1000  # kN
    qhelix_compression = q1_helix * helix_area * 1000  # kN
    
    # Get shaft capacity at the helix depth
    shaft_capacity = qshaft_kn[helix_index]
    
    # Calculate total capacities
    tension_capacity = shaft_capacity + qhelix_tension
    compression_capacity = shaft_capacity + qhelix_compression
    
    # Create summary results
    summary = [{
        'tipdepth': helix_depth,
        'tension_capacity': tension_capacity,
        'compression_capacity': compression_capacity
    }]
    
    # Create detailed results for plotting - with modified logic
    tension_capacity_array = []
    compression_capacity_array = []
    
    # Both tension and compression capacities are the same - they use the Qshaft value
    for i, depth_val in enumerate(depths):
        # Use the shaft capacity directly for both tension and compression
        capacity = qshaft_kn[i]
        
        tension_capacity_array.append(capacity)
        compression_capacity_array.append(capacity)
    
    # Calculate the ultimate tension capacity (Qult)
    # Find the q1 value at (helix depth - helix diameter)
    effective_depth = helix_depth - helix_diameter
    effective_index = min(range(len(depths)), key=lambda i: abs(depths[i] - effective_depth))
    q1_at_effective_depth = q1_values[effective_index]

    # Take the minimum q1 value between these two points
    min_q1_tension = min(q1_helix, q1_at_effective_depth)

    # Calculate qb0.1 for tension (60% of the minimum q1)
    qb01_tension = 0.6 * min_q1_tension

    # Get the shaft capacity at the effective depth
    qshaft_tension = qshaft_kn[effective_index]

    # Calculate ultimate tension capacity
    # Qult = Qshaft + qb0.1 * 1000 * Ab
    qult_tension = qshaft_tension + (qb01_tension * 1000 * helix_area)

    # Calculate the ultimate compression capacity (Qult)
    # Find the q1 value at (helix depth + helix diameter)
    extended_depth = helix_depth + helix_diameter
    # Make sure we don't go beyond the available data
    if extended_depth > depths[-1]:
        extended_depth = depths[-1]
    extended_index = min(range(len(depths)), key=lambda i: abs(depths[i] - extended_depth))
    q1_at_extended_depth = q1_values[extended_index]

    # Take the minimum q1 value between these two points
    min_q1_compression = min(q1_helix, q1_at_extended_depth)

    # Calculate qb0.1 for compression (80% of the minimum q1)
    qb01_compression = 0.8 * min_q1_compression

    # Get the shaft capacity at the effective depth
    qshaft_compression = qshaft_kn[effective_index]

    # Calculate ultimate compression capacity
    # Qult = Qshaft + qb0.1 * 1000 * Ab
    qult_compression = qshaft_compression + (qb01_compression * 1000 * helix_area)

    # Store in results
    results = {
        'summary': summary,
        'tension_capacity_array': tension_capacity_array,
        'compression_capacity_array': compression_capacity_array,
        'qult_tension': qult_tension,
        'qult_compression': qult_compression,
        'qhelix_tension': qhelix_tension,
        'qhelix_compression': qhelix_compression
    }
    
    return results

def calculate_shaft_capacity(processed_cpt, params, perimeter):
    """
    Calculate shaft capacity for helical piles
    """
    
    # Get required parameters
    borehole_depth = float(params['borehole_depth'])
    helix_depth = float(params['helix_depth'])
    
    # Calculate casing coefficients and soil types
    soil_params = calculate_coe_casing_and_soil_type(
        processed_cpt, 
        borehole_depth, 
        helix_depth
    )
    coe_casing = soil_params['coe_casing']
    
    # Calculate delta z and shaft capacity
    shaft_results = calculate_delta_z_and_qshaft(
        processed_cpt,
        coe_casing,
        perimeter
    )
    
    return shaft_results['qshaft_kn']

def calculate_helical_intermediate_values(processed_cpt, params):
    """
    Calculate intermediate values for helical pile analysis
    
    Args:
        processed_cpt (dict): Processed CPT data
        params (dict): Pile parameters
        
    Returns:
        dict: Dictionary containing intermediate calculation results
    """
    # Extract parameters
    shaft_diameter = float(params['shaft_diameter'])
    helix_diameter = float(params['helix_diameter'])
    helix_depth = float(params['helix_depth'])
    borehole_depth = float(params['borehole_depth'])
    
    # Calculate constants
    perimeter = math.pi * shaft_diameter
    helix_area = math.pi * (helix_diameter ** 2) * 0.25
    
    # Get depths array
    depths = processed_cpt['depth']
    
    # Calculate q1 and q10 values
    q1_values = []
    q10_values = []
    for qt in processed_cpt['qt']:
        # Prevent division by zero
        safe_helix_diameter = max(helix_diameter, 0.001)  # Use at least 1mm to prevent division by zero
        
        q1 = qt * (0.1 ** 0.6)
        q10 = qt * ((0.01/safe_helix_diameter) ** 0.6)
        q1_values.append(q1)
        q10_values.append(q10)
    
    # Calculate casing coefficient and soil type
    coe_casing = []
    soil_type = []
    
    for i, depth_val in enumerate(depths):
        # Casing coefficient
        if depth_val < borehole_depth:
            casing = 0
        elif depth_val < helix_depth:
            casing = 1
        else:
            casing = 0
        coe_casing.append(casing)
        
        # Soil type based on Ic value
        ic = processed_cpt['lc'][i] if i < len(processed_cpt['lc']) else 0
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
    
    for i, depth_val in enumerate(depths):
        # Calculate delta z
        if i == 0:
            # For the first point, set delta_z to 0
            current_delta_z = 0.0
            delta_z.append(current_delta_z)
            
            # First segment should be 0
            qshaft_segment.append(0)
            qshaft_kn.append(0)
        else:
            # For subsequent points, calculate difference from previous depth
            current_delta_z = depth_val - depths[i-1]
            delta_z.append(current_delta_z)
            
            # Calculate shaft capacity increment
            qshaft_increment = (coe_casing[i] * current_delta_z * 1000 * processed_cpt['qt'][i] * perimeter) / 175
            qshaft_segment.append(qshaft_increment)
            
            # Update cumulative shaft capacity
            cumulative_qshaft += qshaft_increment
            qshaft_kn.append(cumulative_qshaft)
    
    # Calculate helix capacities
    helix_index = min(range(len(depths)), key=lambda i: abs(depths[i] - helix_depth))
    
    # Get values at the helix depth for the template
    q1_helix = q1_values[helix_index]
    q10_helix = q10_values[helix_index]
    
    # Keep these variables for template compatibility
    qhelix_tension = q10_helix * helix_area * 1000
    qhelix_compression = q1_helix * helix_area * 1000
    
    # Calculate effective depth for tension capacity (tipdepth-Dh)
    effective_depth = helix_depth - helix_diameter
    effective_index = min(range(len(depths)), key=lambda i: abs(depths[i] - effective_depth))
    q1_at_effective_depth = q1_values[effective_index]
    q10_at_effective_depth = q10_values[effective_index]
    
    # Calculate extended depth for compression capacity (tipdepth+Dh)
    extended_depth = helix_depth + helix_diameter
    # Make sure we don't go beyond the available data
    if extended_depth > depths[-1]:
        extended_depth = depths[-1]
    extended_index = min(range(len(depths)), key=lambda i: abs(depths[i] - extended_depth))
    q1_at_extended_depth = q1_values[extended_index]
    q10_at_extended_depth = q10_values[extended_index]
    
    
    # Take the minimum q1 value between these two points for tension
    min_q1_tension = min(q1_helix, q1_at_effective_depth)
    
    # Take the minimum q1 value between these two points for compression
    min_q1_compression = min(q1_helix, q1_at_extended_depth)

    
    # Take the minimum q10 values between appropriate points
    min_q10_tension = min(q10_helix, q10_at_effective_depth)  # Between tip and tip-Dh
    min_q10_compression = min(q10_helix, q10_at_extended_depth)  # Between tip and tip+Dh
    
    # Calculate qb0.1 values for compression and tension at tipdepth
    qb01_comp = 0.8 * min_q1_compression  # 80% of q1 for compression
    qb01_tension = 0.6 * min_q1_tension  # 60% of minimum q1 for tension, not q10_helix
    
    # Calculate helix capacities with qb0.1 values at tipdepth
    qb01_comp_capacity = qb01_comp * helix_area * 1000  # kN
    qb01_tension_capacity = qb01_tension * helix_area * 1000  # kN
    
    # Calculate total capacities at tipdepth (original helix depth)
    tipdepth_shaft = qshaft_kn[helix_index]
    qult_comp_tipdepth = tipdepth_shaft + qb01_comp_capacity
    qult_tension_tipdepth = tipdepth_shaft + qb01_tension_capacity
    
    # Calculate 10mm settlement capacities using min q10 values as per new formulas
    q_10mm_comp_tipdepth = 0.8 * min_q10_compression * helix_area * 1000 + tipdepth_shaft
    q_10mm_tension_tipdepth = 0.6 * min_q10_tension * helix_area * 1000 + tipdepth_shaft
    
    # Calculate tension and compression capacities
    tension_capacity = []
    compression_capacity = []
    
    # Calculate effective depth for capacity (tipdepth-Dh)
    effective_depth = helix_depth - helix_diameter
    effective_index = min(range(len(depths)), key=lambda i: abs(depths[i] - effective_depth))
    
    for i in range(len(depths)):
        if i <= effective_index:
            # For depths above or at the effective depth (tipdepth-Dh)
            # Use the correct factors and minimum q1 values
            tension_capacity.append(qshaft_kn[i] + (0.6 * min_q1_tension * helix_area * 1000))
            compression_capacity.append(qshaft_kn[i] + (0.8 * min_q1_compression * helix_area * 1000))
        else:
            # For depths below the effective depth
            tension_capacity.append(qshaft_kn[i])
            compression_capacity.append(qshaft_kn[i])
    
    # Calculate ultimate capacities at effective depth (tipdepth-Dh)
    qult_tension = tension_capacity[effective_index]
    qult_compression = compression_capacity[effective_index]
    
    # Calculate settlements using min q10 values - correct formulas
    q_delta_10mm_compression = 0.8 * min_q10_compression * helix_area * 1000 + qshaft_kn[effective_index]
    q_delta_10mm_tension = 0.6 * min_q10_tension * helix_area * 1000 + qshaft_kn[effective_index]
    
    # Calculate installation torque using new formula
    installation_torque = 0.4 * qult_tension * (shaft_diameter ** 0.92)
    
    # Create output dictionary
    result = {
        'depth': depths,
        'qt': processed_cpt['qt'],
        'qc': processed_cpt['qc'],
        'fs': processed_cpt['fs'],
        'fr_percent': processed_cpt['fr_percent'],
        'lc': processed_cpt['lc'],
        'q1': q1_values,
        'q10': q10_values,
        'q1_helix': q1_helix,
        'q10_helix': q10_helix,
        'qhelix_tension': qhelix_tension,
        'qhelix_compression': qhelix_compression,
        'soil_type': soil_type,
        'coe_casing': coe_casing,
        'delta_z': delta_z,
        'qshaft_segment': qshaft_segment,
        'qshaft_kn': qshaft_kn,
        'tension_capacity': tension_capacity,
        'compression_capacity': compression_capacity,
        'qult_tension': qult_tension,
        'qult_compression': qult_compression,
        'q_delta_10mm_tension': q_delta_10mm_tension,
        'q_delta_10mm_compression': q_delta_10mm_compression,
        'installation_torque': installation_torque,
        'perimeter': perimeter,
        'helix_area': helix_area,
        'shaft_diameter': shaft_diameter,
        'helix_diameter': helix_diameter,
        'helix_depth': helix_depth,
        'borehole_depth': borehole_depth,
        'qb01_comp': qb01_comp,
        'qb01_tension': qb01_tension,
        'qb01_comp_capacity': qb01_comp_capacity,
        'qb01_tension_capacity': qb01_tension_capacity,
        'qult_comp_tipdepth': qult_comp_tipdepth,
        'qult_tension_tipdepth': qult_tension_tipdepth,
        'q_10mm_comp_tipdepth': q_10mm_comp_tipdepth,
        'q_10mm_tension_tipdepth': q_10mm_tension_tipdepth
    }
    
    return result

def calculate_helical_pile_results(processed_cpt, params):
    """
    Main entry point for helical pile calculations
    
    Args:
        processed_cpt (dict): Processed CPT data
        params (dict): Pile parameters
        
    Returns:
        dict: Results including summary and detailed calculations
    """
    logger.info("Starting helical pile calculations")
    logger.debug(f"Received params: {params}")
    
    try:
        # Ensure all required parameters are available and are floats
        required_params = ['shaft_diameter', 'helix_diameter', 'helix_depth', 'borehole_depth', 'water_table']
        for param in required_params:
            if param not in params:
                logger.error(f"Missing required parameter: {param}")
                raise ValueError(f"Missing required parameter: {param}")
            params[param] = float(params[param])
        
        # Get geometric constants
        constants = calculate_helical_constants(params)
        logger.debug(f"Calculated constants: {constants}")
        perimeter = constants['perimeter']
        helix_area = constants['helix_area']
        
        # Calculate all intermediate values
        detailed_results = calculate_helical_intermediate_values(processed_cpt, params)
        
        # Get the helix depth and calculate effective depth
        helix_depth = params['helix_depth']
        helix_diameter = params['helix_diameter']
        effective_depth = helix_depth - helix_diameter
        
        # Find the nearest depth index to our effective depth (tipdepth-Dh)
        effective_index = min(range(len(detailed_results['depth'])), key=lambda i: abs(detailed_results['depth'][i] - effective_depth))
        
        # Get ultimate capacities (already calculated in intermediate values)
        qult_tension = detailed_results.get('qult_tension', 0)
        qult_compression = detailed_results.get('qult_compression', 0)
        q_delta_10mm_tension = detailed_results.get('q_delta_10mm_tension', 0)
        q_delta_10mm_compression = detailed_results.get('q_delta_10mm_compression', 0)
        installation_torque = detailed_results.get('installation_torque', 0)
        
        # Get capacities at original tipdepth
        qb01_comp = detailed_results.get('qb01_comp', 0)
        qb01_tension = detailed_results.get('qb01_tension', 0)
        qult_comp_tipdepth = detailed_results.get('qult_comp_tipdepth', 0)
        qult_tension_tipdepth = detailed_results.get('qult_tension_tipdepth', 0)
        q_10mm_comp_tipdepth = detailed_results.get('q_10mm_comp_tipdepth', 0)
        q_10mm_tension_tipdepth = detailed_results.get('q_10mm_tension_tipdepth', 0)
        
        # Summary results
        summary = {
            'tipdepth': helix_depth,
            'effective_depth': effective_depth,  # Add effective depth to summary
            'qshaft': detailed_results['qshaft_kn'][effective_index],  # Use effective_index instead of helix_index
            'qult_tension': qult_tension,
            'qult_compression': qult_compression,
            'q_delta_10mm_tension': q_delta_10mm_tension,
            'q_delta_10mm_compression': q_delta_10mm_compression,
            'installation_torque': installation_torque,
            # Add values at original tipdepth to summary
            'qb01_comp': qb01_comp,
            'qb01_tension': qb01_tension,
            'qult_comp_tipdepth': qult_comp_tipdepth,
            'qult_tension_tipdepth': qult_tension_tipdepth,
            'q_10mm_comp_tipdepth': q_10mm_comp_tipdepth,
            'q_10mm_tension_tipdepth': q_10mm_tension_tipdepth
        }
        
        # Add input parameters to the detailed results for completeness
        detailed_results['input_parameters'] = params
        
        # Create a tabular format for download that can be easily converted to CSV
        download_rows = []
        
        # Add header row with title
        download_rows.append(["HELICAL PILE CALCULATION RESULTS"])
        download_rows.append([])  # Empty row for spacing
        
        # Add input parameters first
        download_rows.append(["INPUT PARAMETERS"])
        for key, value in params.items():
            download_rows.append([key, value])
            
        download_rows.append([])  # Empty row for spacing
        download_rows.append(["GEOMETRIC CONSTANTS"])
        download_rows.append(["Perimeter (m)", perimeter])
        download_rows.append(["Helix Area (m²)", helix_area])
        
        download_rows.append([])  # Empty row for spacing
        download_rows.append(["HELIX PROPERTIES"])
        download_rows.append(["Helix Depth (m)", helix_depth])
        download_rows.append(["q1 at Helix", detailed_results['q1_helix']])
        download_rows.append(["q10 at Helix", detailed_results['q10_helix']])
        download_rows.append(["qb0.1 Compression (MPa)", detailed_results['qb01_comp']])
        download_rows.append(["qb0.1 Tension (MPa)", detailed_results['qb01_tension']])
        
        download_rows.append([])  # Empty row for spacing
        download_rows.append(["FINAL RESULTS"])
        download_rows.append(["Ultimate Tension Capacity (kN)", qult_tension])
        download_rows.append(["Ultimate Compression Capacity (kN)", qult_compression])
        download_rows.append(["Tension Capacity at 10mm (kN)", q_delta_10mm_tension])
        download_rows.append(["Compression Capacity at 10mm (kN)", q_delta_10mm_compression])
        download_rows.append(["Installation Torque (kNm)", installation_torque])
        
        download_rows.append([])  # Empty row for spacing
        download_rows.append(["RESULTS AT ORIGINAL TIPDEPTH"])
        download_rows.append(["qb0.1 Compression (MPa)", qb01_comp])
        download_rows.append(["qb0.1 Tension (MPa)", qb01_tension])
        download_rows.append(["Ultimate Compression Capacity at Tipdepth (kN)", qult_comp_tipdepth])
        download_rows.append(["Ultimate Tension Capacity at Tipdepth (kN)", qult_tension_tipdepth])
        download_rows.append(["Compression Capacity at 10mm at Tipdepth (kN)", q_10mm_comp_tipdepth])
        download_rows.append(["Tension Capacity at 10mm at Tipdepth (kN)", q_10mm_tension_tipdepth])

        download_rows.append([])  # Empty row for spacing
        download_rows.append(["DETAILED CALCULATION TABLE"])
        download_rows.append([])  # Empty row for spacing
        
        # Add header row for detailed calculations
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
        download_rows.append(header)
        
        # Add data rows
        for i in range(len(detailed_results['depth'])):
            row = [
                detailed_results['depth'][i],
                detailed_results['qt'][i],
                detailed_results['qc'][i],
                detailed_results['fs'][i],
                detailed_results['fr_percent'][i],
                detailed_results['lc'][i],
                detailed_results['soil_type'][i],
                detailed_results['q1'][i],
                detailed_results['q10'][i],
                detailed_results['coe_casing'][i],
                detailed_results['delta_z'][i],
                detailed_results['qshaft_segment'][i],
                detailed_results['qshaft_kn'][i],
                detailed_results['tension_capacity'][i],
                detailed_results['compression_capacity'][i]
            ]
            download_rows.append(row)
        
        # Return both summary and detailed results
        return {
            'summary': summary,
            'detailed': detailed_results,
            'download_data': download_rows
        }
    
    except Exception as e:
        logger.error(f"Error in helical pile calculations: {str(e)}")
        raise