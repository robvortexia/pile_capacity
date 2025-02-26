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
    qshaft_kn = []
    
    # Initialize cumulative shaft capacity
    cumulative_qshaft = 0
    
    for i in range(len(depths)):
        # Calculate delta z
        if i == 0:
            # For the first point, use the depth itself as delta_z
            current_delta_z = depths[i]
        else:
            # For subsequent points, calculate difference from previous depth
            current_delta_z = depths[i] - depths[i-1]
        
        delta_z.append(current_delta_z)
        
        # Calculate shaft capacity increment
        qshaft_increment = (coe_casing[i] * current_delta_z * 1000 * qt_values[i] * perimeter) / 175
        
        # Update cumulative shaft capacity
        cumulative_qshaft += qshaft_increment
        qshaft_kn.append(cumulative_qshaft)
    
    return {
        'delta_z': delta_z,
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
        'qult_compression': qult_compression
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
    
    # Calculate q1 and q10 values for all depths
    q1_values = []
    q10_values = []
    for qt in processed_cpt['qt']:
        q1 = qt * (0.1 ** 0.6)
        q10 = qt * ((0.01/helix_diameter) ** 0.6) if helix_diameter > 0 else 0
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
            current_delta_z = depth_val
        else:
            current_delta_z = depth_val - depths[i-1]
        
        delta_z.append(current_delta_z)
        
        # Calculate shaft capacity increment
        qshaft_increment = (coe_casing[i] * current_delta_z * 1000 * processed_cpt['qt'][i] * perimeter) / 175
        qshaft_segment.append(qshaft_increment)
        cumulative_qshaft += qshaft_increment
        qshaft_kn.append(cumulative_qshaft)
    
    # Calculate helix capacities
    helix_index = min(range(len(depths)), key=lambda i: abs(depths[i] - helix_depth))
    
    # Get q1 and q10 at helix depth
    q1_helix = q1_values[helix_index]
    q10_helix = q10_values[helix_index]
    
    # Calculate helix capacities
    qhelix_tension = q10_helix * helix_area * 1000
    qhelix_compression = q1_helix * helix_area * 1000
    
    # Prepare results dictionary
    results = {
        'depth': depths,
        'qt': processed_cpt['qt'],
        'qc': processed_cpt['qc'],
        'fs': processed_cpt['fs'],
        'fr_percent': processed_cpt['fr_percent'],
        'lc': processed_cpt['lc'],
        'soil_type': soil_type,
        'q1': q1_values,
        'q10': q10_values,
        'coe_casing': coe_casing,
        'delta_z': delta_z,
        'qshaft_segment': qshaft_segment,
        'qshaft_kn': qshaft_kn,
        'helix_index': helix_index,
        'q1_helix': q1_helix,
        'q10_helix': q10_helix,
        'qhelix_tension': qhelix_tension,
        'qhelix_compression': qhelix_compression,
        'perimeter': perimeter,
        'helix_area': helix_area
    }
    
    return results

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
        
        # Calculate shaft capacity
        qshaft_kn = calculate_shaft_capacity(processed_cpt, params, perimeter)
        
        # Calculate helical pile intermediate values
        intermediate_results = calculate_helical_intermediate_values(processed_cpt, params)
        
        # Calculate final results
        results = calculate_helical_pile_capacity(processed_cpt, params, qshaft_kn)
        logger.info("Helical pile calculations completed successfully")
        
        # Calculate intermediate values
        detailed_results = calculate_helical_intermediate_values(processed_cpt, params)
        
        # Get the helix depth
        helix_depth = params['helix_depth']
        
        # Find the nearest depth index to our helix depth
        helix_index = min(range(len(detailed_results['depth'])), key=lambda i: abs(detailed_results['depth'][i] - helix_depth))
        
        # Calculate effective depth (helix depth - helix diameter)
        effective_depth = helix_depth - params['helix_diameter']
        effective_index = min(range(len(detailed_results['depth'])), key=lambda i: abs(detailed_results['depth'][i] - effective_depth))
        
        # Get shaft capacity
        qshaft = detailed_results['qshaft_kn'][effective_index]
        
        # Get ultimate capacities (already calculated in intermediate values)
        qult_tension = detailed_results.get('qult_tension', 0)
        qult_compression = detailed_results.get('qult_compression', 0)
        
        # Summary results
        summary = {
            'tipdepth': helix_depth,
            'qshaft': qshaft,
            'qult_tension': qult_tension,
            'qult_compression': qult_compression
        }
        
        # Return both summary and detailed results
        return {
            'summary': summary,
            'detailed': detailed_results
        }
    
    except Exception as e:
        logger.error(f"Error in helical pile calculations: {str(e)}")
        raise