import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from scipy.interpolate import interp1d

def get_ifr(diameter):
    return math.tanh(0.3 * ((diameter * 1000 / 35.68)**0.5))

def get_pile_perimeter(pile_shape, nominal_size_don):
    if pile_shape == 0:  # circular
        return math.pi * nominal_size_don
    else:  # square
        return 4 * nominal_size_don

def get_ar(pile_end_condition, nominal_size_don, ifr_value, diameter):
    # Matches JS logic: if pile_end_condition != 1 (i.e., open-ended), reduce the area ratio
    if pile_end_condition != 1:
        return 1 - ifr_value * ((diameter/nominal_size_don)**2)
    return 1

def get_area_b(pile_shape, nominal_size_don):
    if pile_shape == 0:  # circular
        return (math.pi/4) * (nominal_size_don**2)
    else:
        return nominal_size_don**2

def get_dstar(pile_end_condition, nominal_size_don, diameter):
    # Matches JS logic
    # closed-ended: dstar = nominalSizeDoN
    # open-ended: dstar = sqrt(nominalSizeDoN^2 - diameter^2)
    if pile_end_condition == 1:  # closed
        return nominal_size_don
    else:
        val = (nominal_size_don**2) - (diameter**2)
        if val < 0:
            # If this occurs due to rounding, ensure non-negative
            val = 0
        return math.sqrt(val)

def get_coe_casing(depth, borehole_depth, tip_depth):
    """Calculate casing coefficient with smoother transitions"""
    if depth <= borehole_depth:
        return 0
    elif depth <= tip_depth:
        return 1
    else:
        # Gradual reduction beyond tip depth
        transition_zone = 0.5  # 0.5m transition zone
        if depth <= (tip_depth + transition_zone):
            return max(0, 1 - (depth - tip_depth)/transition_zone)
        return 0

def get_kc(lc_value):
    if lc_value < 2.05:
        return 1
    elif lc_value > 2.5:
        return 1
    else:
        return 3.93*(lc_value**2) - 14.78*lc_value + 14.78

def get_qtc(kc, qt):
    return kc * qt

def get_delta_ord(qtc, sig_v0_prime, nominal_size_don):
    delta_ord_value = (1000*qtc*0.1)*((1000*qtc/sig_v0_prime)**(-0.33))*(35.7/(1000*nominal_size_don))
    if math.isnan(delta_ord_value):
        return 0
    return delta_ord_value

def get_orc(qtc, are_value, nominal_size_don, h):
    return ((qtc*1000)/44)*(are_value**0.3)*((max(1, h/nominal_size_don))**(-0.4))

def get_tf_sand(coe_casing, delta_ord, orc):
    return coe_casing*(delta_ord+orc)*0.554

def get_tf_clay(qt_value, coe_casing_value, h_value, dstar_value):
    # If dstar_value is 0, to avoid division by zero, set it to nominalSizeDoN later
    return 0.07*qt_value*1000*coe_casing_value*(max(1,h_value/dstar_value)**(-0.25))

def get_iz1(qtn_value, fr_percent_value):
    return qtn_value - 12*math.exp(-1.4*fr_percent_value)

def get_tf_adop_tension(iz1_value, tf_clay_value, tf_sand_value, lc_value):
    if iz1_value < 0:
        return 0.5*tf_clay_value
    else:
        if lc_value > 2.5:
            return tf_clay_value
        else:
            return 0.75*tf_sand_value

def get_tf_adop_compression(iz1_value, tf_clay_value, tf_sand_value, lc_value):
    if iz1_value < 0:
        return 0.5*tf_clay_value
    else:
        if lc_value > 2.5:
            return tf_clay_value
        else:
            return tf_sand_value

def get_qs(i, depth, tf_adop, pile_perimeter, previous_qs):
    if i == 0:
        delta_depth = depth[i]-0
        return tf_adop*pile_perimeter*delta_depth
    else:
        delta_depth = depth[i]-depth[i-1]
        return previous_qs + tf_adop*pile_perimeter*delta_depth

def get_lc(qtn_value, fr_percent_value):
    # Ensure positive values for logarithm calculations
    safe_qtn = max(qtn_value, 0.001)  # Minimum value to prevent log(0)
    safe_fr = max(fr_percent_value, 0.001)  # Minimum value to prevent log(0)
    
    try:
        return math.sqrt((3.47 - math.log10(safe_qtn))**2 + (math.log10(safe_fr)+1.22)**2)
    except (ValueError, ZeroDivisionError):
        # Return a reasonable default value if calculation fails
        return 2.5

def get_qtn(qt_value, sig_v0_value, sig_v0_prime_value, n_estimate):
    # Prevent division by zero in effective stress
    safe_sig_v0_prime = max(sig_v0_prime_value, 0.1)  # Minimum 0.1 kPa to prevent division by zero
    
    try:
        return ((1000*qt_value - sig_v0_value)/100)*((100/safe_sig_v0_prime)**n_estimate)
    except (ValueError, ZeroDivisionError, OverflowError):
        # Return a reasonable default value if calculation fails
        return 1.0

def get_iterative_values(qt_value, sig_v0_value, sig_v0_prime_value, fr_percent_value):
    n = 0.0
    lc = 0.0 if fr_percent_value == 0 else get_lc(qt_value, fr_percent_value)
    ntrial = 0.381*lc + 0.05*(sig_v0_prime_value/100)-0.15
    err = abs(ntrial - n)
    qtn_val = 0
    while err > 0.001:
        qtn_val = get_qtn(qt_value, sig_v0_value, sig_v0_prime_value, ntrial)
        lc = 0.0 if fr_percent_value == 0 else get_lc(qtn_val, fr_percent_value)
        n = min(1, 0.381*lc + 0.05*(sig_v0_prime_value/100)-0.15)
        err = abs(ntrial - n)
        ntrial = n
    return {'qtn': qtn_val, 'lc': lc, 'n': n}

def get_fr_percent(fs_value, qc_value, sig_v0_value):
    denominator = qc_value * 1000 - sig_v0_value
    if denominator <= 0:
        # Handle edge case where qc*1000 is less than or equal to sig_v0
        # This can happen with very small qc values or high stresses
        # Return a small positive value to avoid division by zero
        return 0.1
    return (fs_value / denominator) * 100

def get_qp_clay_array(depthArray, qtArray, nominalSizeDoN, nominalSizeT):
    """Calculate qp values for clay using vectorized operations"""
    depthArray = np.array(depthArray)
    qtArray = np.array(qtArray)
    
    # Points less than 8*diameter return 0
    qpArray = np.zeros_like(qtArray)
    valid_points = depthArray > (8*nominalSizeDoN)
    
    if not np.any(valid_points):
        return qpArray.tolist()
    
    # For valid points, calculate the moving average
    window_size = int(np.ceil(20 * nominalSizeT / np.mean(np.diff(depthArray))))
    window_size = max(1, min(window_size, len(qtArray) // 2))  # Reasonable limits
    
    # Use numpy's convolve for efficient moving average
    weights = np.ones(2*window_size + 1) / (2*window_size + 1)
    qpArray[valid_points] = np.convolve(qtArray, weights, mode='same')[valid_points]
    
    return qpArray.tolist()

def get_qp_sand_array(depthArray, qtArray, nominalSizeDoN):
    """Calculate qp values for sand using vectorized operations"""
    depthArray = np.array(depthArray)
    qtArray = np.array(qtArray)
    
    # Points less than 8*diameter return 0
    qpArray = np.zeros_like(qtArray)
    valid_points = depthArray > (8*nominalSizeDoN)
    
    if not np.any(valid_points):
        return qpArray.tolist()
    
    # For valid points, calculate the moving average
    window_size = int(np.ceil(1.5 * nominalSizeDoN / np.mean(np.diff(depthArray))))
    window_size = max(1, min(window_size, len(qtArray) // 2))  # Reasonable limits
    
    # Use numpy's convolve for efficient moving average
    weights = np.ones(2*window_size + 1) / (2*window_size + 1)
    qpArray[valid_points] = np.convolve(qtArray, weights, mode='same')[valid_points]
    
    return qpArray.tolist()

def get_qp_mix_array(qp_sand_val, qp_clay_val, lc):
    if lc > 2.5:
        return qp_clay_val
    return qp_sand_val

def get_qb1_sand(are_value, qp_sand):
    return (0.12+0.38*are_value)*qp_sand

def get_qb1_clay(qp_clay, d_star, nominalSizeDoN):
    return qp_clay*(0.2+0.6*((d_star/nominalSizeDoN)**2))

def get_qb1_adop(lc_value, qb1_sand, qb1_clay):
    if lc_value > 2.5:
        return qb1_clay
    else:
        return qb1_sand

def get_qb_final(qb1_adop, area_b):
    return qb1_adop*1000*area_b

def get_qb_sand(qp_value, are_value, area_value):
    return qp_value*1000*(0.15+0.45*are_value)*area_value

def get_qb_clay(pile_end_condition, qt_value, area_value):
    if pile_end_condition == 1:
        return 0.8*qt_value*1000*area_value
    else:
        return 0.4*qt_value*1000*area_value

def pre_input_calc(data, water_table):
    print(f"Water table value in pre_input_calc: {water_table}")
    try:
        cpt_data = data['cpt_data'] if isinstance(data, dict) else data
        # Convert to numpy arrays for calculations but keep as lists for output
        depth = np.array([row['z'] for row in cpt_data])
        qc = np.array([row['qc'] for row in cpt_data])
        gtot = np.array([row['gtot'] for row in cpt_data])
        fs = np.array([row['fs'] for row in cpt_data])
        
        # Pre-allocate lists for better performance
        n_points = len(depth)
        qt = list(qc)  # Direct copy since qt = qc
        u0_kpa = [0] * n_points
        sig_v0 = [0] * n_points
        sig_v0_prime = [0] * n_points
        fr_percent = [0] * n_points
        qtn = [0] * n_points
        n = [0] * n_points
        lc = [0] * n_points
        bq = [0] * n_points
        kc = [0] * n_points
        iz1 = [0] * n_points
        qtc = [0] * n_points

        # Use numpy for the simple calculations, then convert back to lists
        for i in range(n_points):
            try:
                # Water pressure calculation
                # Using 10 kN/m³ for unit weight of water (simplified from 9.81 kN/m³ for engineering convenience)
                u0_kpa[i] = 0 if depth[i] <= water_table else (depth[i]-water_table)*10
                
                # Stress calculations - incremental method for varying soil properties
                if i == 0:
                    sig_v0[i] = depth[i] * gtot[i]  # First layer: stress = depth × unit weight
                else:
                    sig_v0[i] = sig_v0[i-1] + (gtot[i] * (depth[i] - depth[i-1]))  # Incremental: previous stress + layer stress
                sig_v0_prime[i] = sig_v0[i]-u0_kpa[i]
                
                # Ensure effective stress is positive
                if sig_v0_prime[i] <= 0:
                    sig_v0_prime[i] = 0.1  # Minimum value to prevent calculation errors
                
                # Friction ratio calculation
                fr_percent[i] = get_fr_percent(fs[i], qc[i], sig_v0[i])
                
                # Get iterative values (keeping this as is since it's complex)
                iterative_values = get_iterative_values(qt[i], sig_v0[i], sig_v0_prime[i], fr_percent[i])
                qtn[i] = iterative_values['qtn']
                lc[i] = iterative_values['lc']
                n[i] = iterative_values['n']
                
                # Calculate remaining parameters
                kc[i] = get_kc(lc[i])
                qtc[i] = get_qtc(kc[i], qt[i])
                iz1[i] = get_iz1(qtn[i], fr_percent[i])
                
            except Exception as e:
                print(f"Error processing data point {i} at depth {depth[i]}: {str(e)}")
                # Use default values for this point
                fr_percent[i] = 1.0
                qtn[i] = 1.0
                lc[i] = 2.5
                n[i] = 0.5
                kc[i] = 1.0
                qtc[i] = qt[i]
                iz1[i] = 1.0

        return {
            'depth': depth.tolist(),
            'h': [0]*n_points,
            'qc': qc.tolist(),
            'qt': qt,
            'gtot': gtot.tolist(),
            'u0_kpa': u0_kpa,
            'sig_v0': sig_v0,
            'sig_v0_prime': sig_v0_prime,
            'fs': fs.tolist(),
            'fr_percent': fr_percent,
            'qtn': qtn,
            'n': n,
            'lc': lc,
            'bq': bq,
            'kc': kc,
            'iz1': iz1,
            'qtc': qtc
        }
    except Exception as e:
        print(f"Error in pre_input_calc: {str(e)}")
        return None

def interpolate_at_depth(depths, values, target_depth):
    """
    Interpolate a value at a specific depth
    
    Args:
        depths (list): List of depths
        values (list): List of values corresponding to depths
        target_depth (float): Depth at which to interpolate
        
    Returns:
        float: Interpolated value at target depth
    """
    # Create interpolation function
    f = interp1d(depths, values, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Get the interpolated value with full precision
    # Using np.float64 to maintain maximum precision
    return np.float64(f(target_depth))

def calculate_pile_capacity(cpt_data, params, pile_type='driven'):
    print(f"Params received in calculate_pile_capacity: {params}")
    print(f"Water table from params: {params['water_table']}")
    processed_cpt = pre_input_calc(cpt_data, float(params['water_table']))
    
    # Check data point spacing
    depths = processed_cpt['depth']
    min_spacing = min(np.diff(depths))
    print(f"Minimum spacing between data points: {min_spacing}m")
    
    # If spacing is already smaller than 0.1m, skip interpolation
    needs_interpolation = min_spacing > 0.1
    
    if pile_type == 'bored':
        # Bored pile specific calculations
        shaft_diameter = float(params['shaft_diameter'])
        base_diameter = float(params['base_diameter'])
        borehole_depth = float(params['cased_depth'])
        pile_shape = 0  # circular
        pile_perimeter = get_pile_perimeter(pile_shape, shaft_diameter)
        Ab = math.pi * base_diameter * base_diameter * 0.25  # Base area
        
        results = []
        
        for tip in params['pile_tip_depths']:
            tip_depth = float(tip)  # Ensure tip is float
            
            # If we don't need interpolation, find nearest point
            if not needs_interpolation:
                tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip_depth))
                tip_depth = depths[tip_index]  # Use actual depth
            
            # Calculate zone for qb0.1 calculation (tip to tip + base_diameter)
            zone_end = tip_depth + base_diameter
            qb01_values = []
            
            qs_tension_cumulative = 0
            qs_compression_cumulative = 0
            
            # Calculate shaft resistance up to the tip depth
            for i in range(len(depths)):
                if depths[i] > tip_depth:
                    continue
                
                # Get values at current depth
                qt_val = processed_cpt['qt'][i]
                lc_val = processed_cpt['lc'][i]
                
                # Calculate casing coefficient
                coe_casing = get_coe_casing(depths[i], borehole_depth, tip_depth)
                
                # Calculate compression first since tension depends on it
                tf_compression = get_tf_compression_bored(coe_casing, lc_val, qt_val)
                tf_tension = get_tf_tension_bored(coe_casing, lc_val, tf_compression)
                
                # Calculate qb01_adop for base resistance
                qb01_adop = get_qb01_adop(lc_val, qt_val)
                if depths[i] >= tip_depth and depths[i] <= zone_end:
                    qb01_values.append(qb01_adop)
                
                # Calculate shaft resistance for this segment
                prev_depth = depths[i-1] if i > 0 else 0
                delta_z = depths[i] - prev_depth
                qs_tension_segment = tf_tension * pile_perimeter * delta_z
                qs_compression_segment = tf_compression * pile_perimeter * delta_z
                
                qs_tension_cumulative += qs_tension_segment
                qs_compression_cumulative += qs_compression_segment
            
            # Calculate base resistance using minimum qb01_adop in the zone
            min_qb01 = min(qb01_values) if qb01_values else 0
            base_resistance = np.float64(min_qb01 * Ab * 1000)  # Convert to kN with full precision
            
            results.append({
                'tipdepth': tip_depth,
                'tension_capacity': qs_tension_cumulative,
                'compression_capacity': qs_compression_cumulative + base_resistance
            })
        
        return results
    else:
        # Driven pile logic - optimized version
        pile_shape = 0 if params['pile_shape'] == 'circular' else 1
        pile_end_condition = 0 if params['pile_end_condition'] == 'open' else 1
        nominal_size_don = float(params['pile_diameter'])
        nominal_size_t = float(params.get('wall_thickness', 0))/1000
    
        borehole = float(params['borehole_depth'])
        tip_depths = [float(tip) for tip in params['pile_tip_depths']]

        if pile_end_condition == 0:
            diameter = nominal_size_don - 2*nominal_size_t
        else:
            diameter = nominal_size_don

        # Calculate constants once
        ifr_value = get_ifr(diameter)
        are_value = get_ar(pile_end_condition, nominal_size_don, ifr_value, diameter)
        area_value = get_area_b(pile_shape, nominal_size_don)
        dstar_value = get_dstar(pile_end_condition, nominal_size_don, diameter)
        if dstar_value == 0:
            dstar_value = nominal_size_don
        pile_perimeter = get_pile_perimeter(pile_shape, nominal_size_don)

        # Pre-calculate arrays outside the loop
        qp_sand_array = np.array(get_qp_sand_array(depths, processed_cpt['qtc'], nominal_size_don))
        qp_clay_array = np.array(get_qp_clay_array(depths, processed_cpt['qt'], nominal_size_don, nominal_size_t))
        
        results = []
        
        for tip_depth in tip_depths:
            # If we don't need interpolation, find nearest point
            if not needs_interpolation:
                tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip_depth))
                tip_depth = depths[tip_index]
            
            # Calculate h values using vectorized operations
            h = np.maximum(0, tip_depth - np.array(depths))
            
            # Initialize arrays
            qs_tension = np.zeros(len(depths))
            qs_compression = np.zeros(len(depths))
            
            # Vectorized calculations where possible
            coe_casing = np.array([get_coe_casing(d, borehole, tip_depth) for d in depths])
            lc_values = np.array(processed_cpt['lc'])
            qt_values = np.array(processed_cpt['qt'])
            qtc_values = np.array(processed_cpt['qtc'])
            
            # Calculate qp values using vectorized operations
            qp_values = np.where(lc_values > 2.5, qp_clay_array, qp_sand_array)
            
            # Calculate base resistance components
            qb1_sand = get_qb1_sand(are_value, qp_sand_array)
            qb1_clay = get_qb1_clay(qp_clay_array, dstar_value, nominal_size_don)
            qb1_adop = np.where(lc_values > 2.5, qb1_clay, qb1_sand)
            qb_final = get_qb_final(qb1_adop, area_value)
            
            # Calculate shaft resistance
            for i in range(len(depths)):
                if depths[i] > tip_depth:
                    continue
                
                if lc_values[i] < 2.5:
                    delta_ord = get_delta_ord(qtc_values[i], processed_cpt['sig_v0_prime'][i], nominal_size_don)
                    orc_val = get_orc(qtc_values[i], are_value, nominal_size_don, h[i])
                    tf_sand = get_tf_sand(coe_casing[i], delta_ord, orc_val)
                else:
                    delta_ord = 0
                    orc_val = 0
                    tf_sand = 0
                
                tf_clay = get_tf_clay(qt_values[i], coe_casing[i], h[i], dstar_value)
                tf_adop_tension = get_tf_adop_tension(processed_cpt['iz1'][i], tf_clay, tf_sand, lc_values[i])
                tf_adop_compression = get_tf_adop_compression(processed_cpt['iz1'][i], tf_clay, tf_sand, lc_values[i])
                
                # Calculate cumulative shaft resistance
                if i == 0:
                    delta_z = depths[i]
                else:
                    delta_z = depths[i] - depths[i-1]
                
                qs_tension[i] = (0 if i == 0 else qs_tension[i-1]) + tf_adop_tension * pile_perimeter * delta_z
                qs_compression[i] = (0 if i == 0 else qs_compression[i-1]) + tf_adop_compression * pile_perimeter * delta_z
            
            # Get final capacities
            if needs_interpolation:
                tension_capacity = interpolate_at_depth(depths, qs_tension, tip_depth)
                compression_base = interpolate_at_depth(depths, qb_final, tip_depth)
                compression_shaft = interpolate_at_depth(depths, qs_compression, tip_depth)
                compression_capacity = compression_shaft + compression_base
            else:
                tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip_depth))
                tension_capacity = qs_tension[tip_index]
                compression_capacity = qs_compression[tip_index] + qb_final[tip_index]
            
            results.append({
                'tipdepth': tip_depth,
                'tension_capacity': np.float64(tension_capacity),  # Ensure full precision
                'compression_capacity': np.float64(compression_capacity)  # Ensure full precision
            })
        
        return results

def process_cpt_data(data):
    try:
        # Data is already in the correct format, no need for additional processing
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if all(key in data[0] for key in ['z', 'fs', 'qc', 'gtot']):
                return {'cpt_data': data}
        
        # If we somehow get here with other data types, handle them
        cpt_data = []
        if isinstance(data, pd.DataFrame):
            cpt_data = [
                {
                    'z': float(row[0]),
                    'qc': float(row[1]),
                    'fs': float(row[2]),
                    'gtot': float(row[3])
                }
                for _, row in data.iterrows()
            ]
        elif isinstance(data, list):
            cpt_data = data  # Data should already be in correct format from route handler
            
        if not cpt_data:
            raise ValueError("No data processed")
            
        return {'cpt_data': cpt_data}
        
    except Exception as e:
        logger.error(f"Error processing CPT data: {str(e)}")
        return None

def create_cpt_graphs(data, water_table=0):
    processed_data = pre_input_calc({'cpt_data': data}, water_table)

    ic_fig = go.Figure()
    ic_fig.add_trace(go.Scatter(
        x=processed_data['lc'],
        y=processed_data['depth'],
        mode='lines',
        name='Ic'
    ))
    ic_fig.update_layout(
        title='Ic vs Depth',
        xaxis_title='Ic',
        yaxis_title='Depth (m)',
        yaxis_autorange='reversed',
        xaxis_range=[1, 5],
        height=800,
        xaxis={'side': 'top'},  # Place x-axis at the top
        yaxis={'side': 'left'},  # Keep y-axis on the left
        margin={'l': 60, 'r': 30, 't': 40, 'b': 30}  # Adjust margins for top x-axis
    )

    graphs = {}
    graphs['ic'] = json.dumps(ic_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphs

def get_qb01_adop(lc, qt):
    return 0.11 * lc * qt

def get_tf_adop_tension_bored(qb01_adop, lc):
    if qb01_adop == 0:
        return 0
    if lc < 2.05:
        return 0.8 * qb01_adop
    return qb01_adop

def get_tf_adop_compression_bored(qt, lc):
    if qt == 0:
        return 0
    return 0.008 * 101 * (lc ** 1.6) * ((1000 * qt) / 101) ** 0.8

def get_qs_bored(tf_adop, pile_perimeter, depth_i, depth_prev):
    delta_z = depth_i - depth_prev
    return tf_adop * pile_perimeter * delta_z

def calculate_bored_pile_results(processed_cpt, params):
    water_table = float(params['water_table'])
    print(f"Water table value from params: {water_table}")
    shaft_diameter = float(params['shaft_diameter'])
    base_diameter = float(params['base_diameter'])
    pile_shape = 0  # circular
    pile_perimeter = get_pile_perimeter(pile_shape, shaft_diameter)
    borehole_depth = float(params['cased_depth'])
    
    # Calculate base area
    Ab = math.pi * base_diameter * base_diameter * 0.25
    
    depths = processed_cpt['depth']
    detailed_results = []
    summary_results = []
    
    for tip in params['pile_tip_depths']:
        tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip))
        chosen_tip = depths[tip_index]
        
        depth_calculations = []
        qs_tension_cumulative = 0
        qs_compression_cumulative = 0
        
        # Find the zone for qb0.1 calculation (tip to tip + base_diameter)
        zone_end = chosen_tip + base_diameter
        zone_end_index = min(range(len(depths)), key=lambda i: abs(depths[i]-zone_end))
        qb01_values = []
        
        for i in range(len(depths)):
            if depths[i] > chosen_tip:
                continue
                
            qt_val = processed_cpt['qt'][i]
            lc_val = processed_cpt['lc'][i]
            fr_val = processed_cpt['fr_percent'][i]
            
            # Calculate casing coefficient
            coe_casing = get_coe_casing(depths[i], borehole_depth, chosen_tip)
            
            # Calculate qb01_adop
            qb01_adop = get_qb01_adop(lc_val, qt_val)
            
            # Store qb01_adop if in the relevant zone for base resistance
            if tip_index <= i <= zone_end_index:
                qb01_values.append(qb01_adop)
            
            # Calculate compression first since tension depends on it
            tf_compression = get_tf_compression_bored(coe_casing, lc_val, qt_val)
            tf_tension = get_tf_tension_bored(coe_casing, lc_val, tf_compression)
            
            # Calculate shaft resistance for this segment
            prev_depth = depths[i-1] if i > 0 else 0
            delta_z = depths[i] - prev_depth
            qs_tension_segment = tf_tension * pile_perimeter * delta_z
            qs_compression_segment = tf_compression * pile_perimeter * delta_z
            
            # Update cumulative values
            qs_tension_cumulative += qs_tension_segment
            qs_compression_cumulative += qs_compression_segment
            
            # Store all calculations for this depth
            depth_calculations.append({
                'depth': np.float64(depths[i]),
                'qt': np.float64(qt_val),
                'qc': np.float64(processed_cpt['qc'][i]),
                'fs': np.float64(processed_cpt['fs'][i]),
                'lc': np.float64(lc_val),
                'fr': np.float64(fr_val),
                'sig_v0_prime': np.float64(processed_cpt['sig_v0_prime'][i]),
                'u0': np.float64(processed_cpt['u0'][i]),
                'sig_v0': np.float64(processed_cpt['sig_v0'][i]),
                'qtn': np.float64(processed_cpt['qtn'][i]),
                'n': np.float64(processed_cpt['n'][i]),
                'coe_casing': np.float64(coe_casing),
                'qb01_adop': np.float64(qb01_adop),
                'tf_tension': np.float64(tf_tension),
                'tf_compression': np.float64(tf_compression),
                'delta_z': np.float64(delta_z),
                'qs_tension_segment': np.float64(qs_tension_segment),
                'qs_compression_segment': np.float64(qs_compression_segment),
                'qs_tension_cumulative': np.float64(qs_tension_cumulative),
                'qs_compression_cumulative': np.float64(qs_compression_cumulative)
            })
        
        # Calculate base resistance using minimum qb01_adop in the zone
        min_qb01 = min(qb01_values) if qb01_values else 0
        base_resistance = np.float64(min_qb01 * Ab * 1000)  # Convert to kN with full precision
        
        detailed_results.append({
            'tip_depth': chosen_tip,
            'calculations': depth_calculations,
            'pile_constants': {
                'shaft_diameter': np.float64(shaft_diameter),
                'base_diameter': np.float64(base_diameter),
                'borehole_depth': np.float64(borehole_depth),
                'pile_perimeter': np.float64(pile_perimeter),
                'base_area': np.float64(Ab),
                'min_qb01': np.float64(min_qb01),
                'base_resistance': np.float64(base_resistance)
            }
        })
        
        summary_results.append({
            'tipdepth': chosen_tip,
            'tension_capacity': np.float64(qs_tension_cumulative),  # Use full precision
            'compression_capacity': np.float64(qs_compression_cumulative + base_resistance)  # Use full precision
        })
    
    return {
        'summary': summary_results,
        'detailed': detailed_results
    }

def get_tf_tension_bored(coe_casing, lc, tf_compression):
    """Calculate tension transfer function for bored piles"""
    if coe_casing == 0:
        return 0
    if lc < 2.05:
        return 0.8 * tf_compression
    return tf_compression

def get_tf_compression_bored(coe_casing, lc, qt):
    """Calculate compression transfer function for bored piles"""
    if coe_casing == 0:
        return 0
    return 0.008 * 101 * (lc ** 1.6) * ((1000 * qt / 101) ** 0.8)

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
    shaft_diameter = pile_params['shaft_diameter']
    helix_diameter = pile_params['helix_diameter']
    
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
        q10 = qt * ((0.01/helix_diameter) ** 0.6)
        
        q1_values.append(q1)
        q10_values.append(q10)
    
    return {
        'q1': q1_values,
        'q10': q10_values
    }

def calculate_helical_pile_capacity(processed_cpt, pile_params, qshaft_kn):
    """
    Calculate final tension and compression capacities for helical pile
    
    Args:
        processed_cpt (dict): Dictionary containing CPT data including:
            - depth (list): Depth values
        pile_params (dict): Dictionary containing pile parameters including:
            - helix_depth (float): Depth to helical plate
            - helix_diameter (float): Diameter of helical plate
        qshaft_kn (list): Shaft capacity values for each depth
    
    Returns:
        dict: Dictionary containing summary and detailed results
    """
    
    depths = processed_cpt['depth']
    helix_depth = float(pile_params['helix_depth'])
    
    # Find the index for the helix depth
    helix_index = min(range(len(depths)), key=lambda i: abs(depths[i] - helix_depth))
    
    # Get shaft capacity at the helix depth
    shaft_capacity = qshaft_kn[helix_index]
    
    # Calculate total capacities
    tension_capacity = shaft_capacity
    compression_capacity = shaft_capacity * 1.2  # Example factor, adjust as needed
    
    # Create summary results
    summary = [{
        'tipdepth': helix_depth,
        'tension_capacity': tension_capacity,
        'compression_capacity': compression_capacity
    }]
    
    # Create detailed results for plotting
    detailed = {
        'depth': depths,
        'tension_capacity': qshaft_kn,
        'compression_capacity': [q * 1.2 for q in qshaft_kn]  # Example factor
    }
    
    return {
        'summary': summary,
        'detailed': detailed
    }

def calculate_delta_z_and_qshaft(processed_cpt, coe_casing, perimeter):
    """
    Calculate delta z and shaft capacity for each depth
    
    Args:
        processed_cpt (dict): Dictionary containing CPT data including:
            - depth (list): Depth values
            - qt (list): Cone tip resistance values
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
            # For the first point, set delta_z to None (will display as blank)
            current_delta_z = None
            delta_z.append(current_delta_z)
            
            # First segment should be 0
            qshaft_kn.append(0)
        else:
            # For subsequent points, calculate difference from previous depth
            current_delta_z = depths[i] - depths[i-1]
            delta_z.append(current_delta_z)
            
            # Calculate shaft capacity increment
            qshaft_increment = (coe_casing[i] * current_delta_z * 1000 * qt_values[i] * perimeter) / 175
            qshaft_kn.append(cumulative_qshaft + qshaft_increment)
        
        delta_z.append(current_delta_z)
        
        # Calculate shaft capacity increment
        # Qshaft_kN[i] = Qshaft_kN[i-1] + (Coe.Casing_i*deltaZ_i*1000*qt_i*perimeter)/175
        qshaft_increment = (coe_casing[i] * current_delta_z * 1000 * qt_values[i] * perimeter) / 175
        
        # Update cumulative shaft capacity
        cumulative_qshaft += qshaft_increment
        qshaft_kn.append(cumulative_qshaft)
    
    return {
        'delta_z': delta_z,
        'qshaft_kn': qshaft_kn
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
        # If depth < borehole_depth: 0
        # If depth < tip_depth: 1
        # Otherwise: 0
        if depth < borehole_depth:
            casing = 0
        elif depth < tip_depth:
            casing = 1
        else:
            casing = 0
            
        coe_casing.append(casing)
        
        # Calculate soil type based on Ic value
        # If Ic > 2.2: "Clay/Silt"
        # If Ic <= 2.2: "Sand"
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

def calculate_helical_pile_results(processed_cpt, params):
    """Calculate helical pile results"""
    print("Received params in calculate_helical_pile_results:", params)
    
    # Get geometric constants
    constants = calculate_helical_constants(params)
    print("Calculated constants:", constants)
    perimeter = constants['perimeter']
    helix_area = constants['helix_area']
    
    # Calculate q1 and q10 values
    q_values = calculate_q1_q10(processed_cpt, params['helix_diameter'])
    print("Calculated q values:", q_values)
    
    # Get helix depth index
    helix_depth = float(params['helix_depth'])
    helix_index = min(range(len(processed_cpt['depth'])), 
                     key=lambda i: abs(processed_cpt['depth'][i] - helix_depth))
    
    # Calculate shaft capacity
    qshaft_kn = calculate_shaft_capacity(processed_cpt, params, perimeter)
    
    # Calculate total capacities
    results = calculate_helical_pile_capacity(processed_cpt, params, qshaft_kn)
    print("Final results:", results)
    
    return {
        'summary': results['summary'],
        'detailed': results['detailed']
    }

def calculate_shaft_capacity(processed_cpt, params, perimeter):
    """
    Calculate shaft capacity for helical piles
    
    Args:
        processed_cpt (dict): Processed CPT data
        params (dict): Pile parameters
        perimeter (float): Shaft perimeter
    
    Returns:
        list: Shaft capacity values for each depth
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
    
    # Calculate total capacities
    tension_capacity_array = []
    compression_capacity_array = []
    
    for i in range(len(depths)):
        if depths[i] <= helix_depth:
            tension_capacity = qshaft_kn[i]
            compression_capacity = qshaft_kn[i]
        else:
            tension_capacity = qshaft_kn[i] + qhelix_tension
            compression_capacity = qshaft_kn[i] + qhelix_compression
        
        tension_capacity_array.append(tension_capacity)
        compression_capacity_array.append(compression_capacity)
    
    return {
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
        'tension_capacity': tension_capacity_array,
        'compression_capacity': compression_capacity_array,
        'helix_index': helix_index,
        'q1_helix': q1_helix,
        'q10_helix': q10_helix,
        'qhelix_tension': qhelix_tension,
        'qhelix_compression': qhelix_compression,
        'perimeter': perimeter,
        'helix_area': helix_area
    }

def calculate_driven_pile_results(processed_cpt, params):
    """
    Calculate driven pile results with detailed calculations for each depth
    Similar to calculate_bored_pile_results but for driven piles
    """
    print(f"Params received in calculate_driven_pile_results: {params}")
    print(f"Water table from params: {params['water_table']}")
    
    # Extract parameters
    pile_shape = 0 if params['pile_shape'] == 'circular' else 1
    pile_end_condition = 0 if params['pile_end_condition'] == 'open' else 1
    nominal_size_don = float(params['pile_diameter'])
    nominal_size_t = float(params.get('wall_thickness', 0))/1000
    borehole = float(params['borehole_depth'])
    tip_depths = [float(tip) for tip in params['pile_tip_depths']]

    if pile_end_condition == 0:
        diameter = nominal_size_don - 2*nominal_size_t
    else:
        diameter = nominal_size_don

    # Calculate constants once
    ifr_value = get_ifr(diameter)
    are_value = get_ar(pile_end_condition, nominal_size_don, ifr_value, diameter)
    area_value = get_area_b(pile_shape, nominal_size_don)
    dstar_value = get_dstar(pile_end_condition, nominal_size_don, diameter)
    if dstar_value == 0:
        dstar_value = nominal_size_don
    pile_perimeter = get_pile_perimeter(pile_shape, nominal_size_don)

    depths = processed_cpt['depth']
    
    # Pre-calculate arrays outside the loop
    qp_sand_array = np.array(get_qp_sand_array(depths, processed_cpt['qtc'], nominal_size_don))
    qp_clay_array = np.array(get_qp_clay_array(depths, processed_cpt['qt'], nominal_size_don, nominal_size_t))
    
    detailed_results = []
    summary_results = []
    
    # Check data point spacing
    min_spacing = min(np.diff(depths))
    print(f"Minimum spacing between data points: {min_spacing}m")
    needs_interpolation = min_spacing > 0.1
    
    for tip_depth in tip_depths:
        # If we don't need interpolation, find nearest point
        if not needs_interpolation:
            tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip_depth))
            tip_depth = depths[tip_index]
        
        depth_calculations = []
        
        # Calculate h values using vectorized operations
        h = np.maximum(0, tip_depth - np.array(depths))
        
        # Initialize arrays
        qs_tension = np.zeros(len(depths))
        qs_compression = np.zeros(len(depths))
        
        # Vectorized calculations where possible
        coe_casing = np.array([get_coe_casing(d, borehole, tip_depth) for d in depths])
        lc_values = np.array(processed_cpt['lc'])
        qt_values = np.array(processed_cpt['qt'])
        qtc_values = np.array(processed_cpt['qtc'])
        fs_values = np.array(processed_cpt['fs'])
        fr_values = np.array(processed_cpt['fr_percent'])
        
        # Calculate qp values using vectorized operations
        qp_values = np.where(lc_values > 2.5, qp_clay_array, qp_sand_array)
        
        # Calculate base resistance components
        qb1_sand = get_qb1_sand(are_value, qp_sand_array)
        qb1_clay = get_qb1_clay(qp_clay_array, dstar_value, nominal_size_don)
        qb1_adop = np.where(lc_values > 2.5, qb1_clay, qb1_sand)
        qb_final = get_qb_final(qb1_adop, area_value)
        
        # Calculate q1 and q10 values for all depths
        q1_values = []
        q10_values = []
        for qt in qt_values:
            q1 = qt * (0.1 ** 0.6)
            q10 = qt * ((0.01/nominal_size_don) ** 0.6) if nominal_size_don > 0 else 0
            q1_values.append(q1)
            q10_values.append(q10)
        
        # Calculate shaft resistance and store detailed data
        qs_tension_cumulative = 0
        qs_compression_cumulative = 0
        
        for i in range(len(depths)):
            if depths[i] > tip_depth:
                continue
            
            # Calculate tf values
            if lc_values[i] < 2.5:
                delta_ord = get_delta_ord(qtc_values[i], processed_cpt['sig_v0_prime'][i], nominal_size_don)
                orc_val = get_orc(qtc_values[i], are_value, nominal_size_don, h[i])
                tf_sand = get_tf_sand(coe_casing[i], delta_ord, orc_val)
            else:
                delta_ord = 0
                orc_val = 0
                tf_sand = 0
            
            tf_clay = get_tf_clay(qt_values[i], coe_casing[i], h[i], dstar_value)
            tf_adop_tension = get_tf_adop_tension(processed_cpt['iz1'][i], tf_clay, tf_sand, lc_values[i])
            tf_adop_compression = get_tf_adop_compression(processed_cpt['iz1'][i], tf_clay, tf_sand, lc_values[i])
            
            # Calculate delta_z for this segment
            if i == 0:
                delta_z = depths[i]
            else:
                delta_z = depths[i] - depths[i-1]
            
            # Calculate shaft resistance for this segment
            qs_tension_segment = tf_adop_tension * pile_perimeter * delta_z
            qs_compression_segment = tf_adop_compression * pile_perimeter * delta_z
            
            # Update cumulative values
            qs_tension_cumulative += qs_tension_segment
            qs_compression_cumulative += qs_compression_segment
            
            # Store all calculations for this depth
            depth_calculations.append({
                'depth': np.float64(depths[i]),
                'qt': np.float64(qt_values[i]),
                'qc': np.float64(processed_cpt['qc'][i]),
                'qtc': np.float64(qtc_values[i]),
                'fs': np.float64(fs_values[i]),
                'fr_percent': np.float64(fr_values[i]),
                'lc': np.float64(lc_values[i]),
                'gtot': np.float64(processed_cpt['gtot'][i]),
                'sig_v0': np.float64(processed_cpt['sig_v0'][i]),
                'sig_v0_prime': np.float64(processed_cpt['sig_v0_prime'][i]),
                'u0': np.float64(processed_cpt['u0_kpa'][i]),
                'qtn': np.float64(processed_cpt['qtn'][i]),
                'n': np.float64(processed_cpt['n'][i]),
                'iz1': np.float64(processed_cpt['iz1'][i]),
                'h': np.float64(h[i]),
                'q1': np.float64(q1_values[i]),
                'q10': np.float64(q10_values[i]),
                'qp_sand': np.float64(qp_sand_array[i]),
                'qp_clay': np.float64(qp_clay_array[i]),
                'qp_adopted': np.float64(qp_values[i]),
                'qb1_sand': np.float64(qb1_sand[i]),
                'qb1_clay': np.float64(qb1_clay[i]),
                'qb1_adopted': np.float64(qb1_adop[i]),
                'coe_casing': np.float64(coe_casing[i]),
                'delta_ord': np.float64(delta_ord),
                'orc_val': np.float64(orc_val),
                'tf_sand': np.float64(tf_sand),
                'tf_clay': np.float64(tf_clay),
                'tf_adop_tension': np.float64(tf_adop_tension),
                'tf_adop_compression': np.float64(tf_adop_compression),
                'delta_z': np.float64(delta_z),
                'qs_tension_segment': np.float64(qs_tension_segment),
                'qs_compression_segment': np.float64(qs_compression_segment),
                'qs_tension_cumulative': np.float64(qs_tension_cumulative),
                'qs_compression_cumulative': np.float64(qs_compression_cumulative),
                'qb_final': np.float64(qb_final[i])
            })
        
        # Get final capacities
        if needs_interpolation:
            tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip_depth))
            tension_capacity = qs_tension_cumulative
            compression_base = qb_final[tip_index]
            compression_capacity = qs_compression_cumulative + compression_base
        else:
            tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip_depth))
            tension_capacity = qs_tension_cumulative
            compression_capacity = qs_compression_cumulative + qb_final[tip_index]
        
        detailed_results.append({
            'tip_depth': tip_depth,
            'calculations': depth_calculations,
            'pile_constants': {
                'nominal_size_don': np.float64(nominal_size_don),
                'nominal_size_t': np.float64(nominal_size_t),
                'diameter': np.float64(diameter),
                'borehole_depth': np.float64(borehole),
                'pile_perimeter': np.float64(pile_perimeter),
                'ifr_value': np.float64(ifr_value),
                'are_value': np.float64(are_value),
                'area_value': np.float64(area_value),
                'dstar_value': np.float64(dstar_value),
                'pile_shape': pile_shape,
                'pile_end_condition': pile_end_condition
            }
        })
        
        summary_results.append({
            'tipdepth': tip_depth,
            'tension_capacity': np.float64(tension_capacity),
            'compression_capacity': np.float64(compression_capacity)
        })
    
    return {
        'summary': summary_results,
        'detailed': detailed_results
    }
