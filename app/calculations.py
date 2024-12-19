import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.utils

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

def get_coe_casing(depth, tip, borehole):
    if depth < borehole:
        return 0
    elif borehole <= depth <= tip:
        return 1
    else:
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
    return math.sqrt((3.47 - math.log10(qtn_value))**2 + (math.log10(fr_percent_value)+1.22)**2)

def get_qtn(qt_value, sig_v0_value, sig_v0_prime_value, n_estimate):
    return ((1000*qt_value - sig_v0_value)/100)*((100/sig_v0_prime_value)**n_estimate)

def get_iterative_values(qt_value, sig_v0_value, sig_v0_prime_value, fr_percent_value):
    n = 0.0
    lc = 0.0 if fr_percent_value == 0 else get_lc(qt_value, fr_percent_value)
    ntrial = 0.381*lc + 0.05*(sig_v0_prime_value/100)-0.15
    err = abs(ntrial - n)
    qtn_val = 0
    while err > 0.001:
        qtn_val = get_qtn(qt_value, sig_v0_value, sig_v0_prime_value, ntrial)
        lc = 0.0 if fr_percent_value == 0 else get_lc(qtn_val, fr_percent_value)
        n = min(1, 0.381*lc + 0.05*(fr_percent_value/100)-0.15)
        err = abs(ntrial - n)
        ntrial = n
    return {'qtn': qtn_val, 'lc': lc, 'n': n}

def get_fr_percent(fs_value, qc_value, sig_v0_value):
    return (fs_value/(qc_value*1000 - sig_v0_value))*100

def get_qp_clay_array(depthArray, qtArray, nominalSizeDoN, nominalSizeT):
    diameter = nominalSizeDoN
    t = nominalSizeT
    qpArray = []
    for i in range(len(depthArray)):
        if i == len(depthArray)-1:
            deltaz = 0
        else:
            deltaz = depthArray[i+1]-depthArray[i]
        
        if deltaz != 0:
            avgcells = 20*t / deltaz
        else:
            avgcells = 0
        qpc = 0
        depth = depthArray[i]
        if depth > (8*diameter) and avgcells > 0:
            qtsum = 0
            n = 0
            start_j = max(i - round(avgcells), 0)
            end_j = min(i + round(avgcells), len(qtArray)-1)
            for idx in range(start_j, end_j+1):
                val = qtArray[idx]
                if val > 0:
                    qtsum += val
                    n += 1
            if n > 0:
                qpc = qtsum/n
            else:
                qpc = 0
        else:
            qpc = 0
        qpArray.append(qpc)
    return qpArray

def get_qp_sand_array(depthArray, qtArray, nominalSizeDoN):
    diameter = nominalSizeDoN
    qpArray = []
    for i in range(len(depthArray)):
        if i == len(depthArray)-1:
            deltaz = 0
        else:
            deltaz = depthArray[i+1]-depthArray[i]

        if deltaz == 0:
            qpArray.append(0)
            continue
        avgcells = round(1.5*diameter / deltaz)
        qpc = 0
        depth = depthArray[i]
        if depth > (8*diameter) and avgcells > 0:
            qtsum = 0
            n = 0
            start_j = max(i-avgcells, 0)
            end_j = min(i+avgcells, len(qtArray)-1)
            for idx in range(start_j, end_j+1):
                val = qtArray[idx]
                if val > 0:
                    qtsum += val
                    n += 1
            if n > 0:
                qpc = qtsum/n
            else:
                qpc = 0
        else:
            qpc = 0
        qpArray.append(qpc)
    return qpArray

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

def pre_input_calc(data, rl_water_table=0):
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
            # Water pressure calculation
            u0_kpa[i] = 0 if depth[i] <= rl_water_table else (depth[i]-rl_water_table)*9.81
            
            # Stress calculations
            sig_v0[i] = depth[i]*gtot[i]
            sig_v0_prime[i] = sig_v0[i]-u0_kpa[i]
            
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

def calculate_pile_capacity(cpt_data, params, pile_type='driven'):
    processed_cpt = pre_input_calc(cpt_data)
    
    # Match JS logic for bored/CFA:
    # Bored/CFA piles are considered open-ended (pileEndCondition=0) in the JS code.
    if pile_type == 'bored':
        nominal_size_don = float(params['shaft_diameter'])
        base_diameter = float(params['base_diameter'])
        pile_shape = 0  # assuming circular (matches JS)
        pile_end_condition = 0  # open-ended for bored/CFA, as per JS logic
        nominal_size_t = 0  # no wall thickness for bored piles
    else:
        # Driven pile logic
        pile_shape = 0 if params['pile_shape'] == 'circular' else 1
        pile_end_condition = 0 if params['pile_end_condition'] == 'open' else 1
        nominal_size_don = float(params['pile_diameter'])
        nominal_size_t = float(params.get('wall_thickness', 0))/1000
    
    borehole = float(params['borehole_depth'])
    tip_depths = params['pile_tip_depths']

    if pile_end_condition == 0:
        diameter = nominal_size_don - 2*nominal_size_t
    else:
        diameter = nominal_size_don

    ifr_value = get_ifr(diameter)
    are_value = get_ar(pile_end_condition, nominal_size_don, ifr_value, diameter)
    area_value = get_area_b(pile_shape, nominal_size_don)
    dstar_value = get_dstar(pile_end_condition, nominal_size_don, diameter)
    
    # If dstar_value=0 (can happen if diameter=nominalSizeDon), fallback to nominal_size_don 
    # to avoid division by zero errors.
    if dstar_value == 0:
        dstar_value = nominal_size_don

    pile_perimeter = get_pile_perimeter(pile_shape, nominal_size_don)

    qp_clay = get_qp_clay_array(processed_cpt['depth'], processed_cpt['qt'], nominal_size_don, nominal_size_t)
    qp_sand = get_qp_sand_array(processed_cpt['depth'], processed_cpt['qtc'], nominal_size_don)

    depths = processed_cpt['depth']
    results = []
    for tip in tip_depths:
        # Find closest depth index to given tip to match JS logic which uses exact indices
        tip_index = min(range(len(depths)), key=lambda i: abs(depths[i]-tip))
        chosen_tip = depths[tip_index]  # actual CPT depth used

        h = [max(0, chosen_tip - d) for d in processed_cpt['depth']]

        qs_tension = []
        qs_compression = []
        qb_sand_list = []
        qb_clay_list = []
        qb_final_list = []

        for i in range(len(processed_cpt['depth'])):
            depth = processed_cpt['depth'][i]
            qt_val = processed_cpt['qt'][i]
            qtc_val = processed_cpt['qtc'][i]
            lc_val = processed_cpt['lc'][i]
            coe_casing = get_coe_casing(depth, chosen_tip, borehole)

            qp_value = get_qp_mix_array(qp_sand[i], qp_clay[i], lc_val)
            qb1_sand_val = get_qb1_sand(are_value, qp_sand[i])
            qb1_clay_val = get_qb1_clay(qp_clay[i], dstar_value, nominal_size_don)
            qb1_adop_val = get_qb1_adop(lc_val, qb1_sand_val, qb1_clay_val)

            qb_final_val = get_qb_final(qb1_adop_val, area_value)
            qb_sand_val = get_qb_sand(qp_value, are_value, area_value)
            qb_clay_val = get_qb_clay(pile_end_condition, qt_val, area_value)

            if lc_val < 2.5:
                delta_ord = get_delta_ord(qtc_val, processed_cpt['sig_v0_prime'][i], nominal_size_don)
                orc_val = get_orc(qtc_val, are_value, nominal_size_don, h[i])
                tf_sand = get_tf_sand(coe_casing, delta_ord, orc_val)
            else:
                tf_sand = 0

            tf_clay = get_tf_clay(qt_val, coe_casing, h[i], dstar_value)
            tf_adop_tension = get_tf_adop_tension(processed_cpt['iz1'][i], tf_clay, tf_sand, lc_val)
            tf_adop_compression = get_tf_adop_compression(processed_cpt['iz1'][i], tf_clay, tf_sand, lc_val)

            prev_qs_tension = qs_tension[-1] if qs_tension else 0
            prev_qs_compression = qs_compression[-1] if qs_compression else 0
            qs_tension.append(get_qs(i, processed_cpt['depth'], tf_adop_tension, pile_perimeter, prev_qs_tension))
            qs_compression.append(get_qs(i, processed_cpt['depth'], tf_adop_compression, pile_perimeter, prev_qs_compression))
            qb_sand_list.append(qb_sand_val)
            qb_clay_list.append(qb_clay_val)
            qb_final_list.append(qb_final_val)

        tension_capacity = qs_tension[tip_index]
        compression_capacity = qs_compression[tip_index] + qb_final_list[tip_index]

        results.append({
            'tipdepth': chosen_tip,
            'tension_capacity': tension_capacity,
            'compression_capacity': compression_capacity
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

def create_cpt_graphs(data):
    processed_data = pre_input_calc({'cpt_data': data})

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
        height=800
    )

    graphs = {}
    graphs['ic'] = json.dumps(ic_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphs
