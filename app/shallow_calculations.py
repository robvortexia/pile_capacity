"""
Shallow foundations (Step 4) - load-settlement of footings on sand and clay
from CPT data, following the UWA/Lehane method captured in the workbook
"Footing-settlement-forcoding.xlsx".

Step 1-3 are shared with the pile modules. This module consumes the step-3
output (``processed_cpt`` from ``pre_input_calc``) plus footing geometry and
returns the bearing-pressure vs settlement curve(s) and a summary box.

Methodology verified to machine precision against every computed cell in the
workbook for both the sand sample (From pileapp-1) and clay sample
(From pileapp-2): all summary cells, the per-row post-excavation qc, and the
full settlement curves agree to ~1e-13 or better.

Excel cell references are given in comments for traceability:
  Intro+sand-calc  -> sand branch
  Clay-calc        -> clay branch
  Sand-output / Clay-output -> the plotted curves
  BC-calc          -> traditional bearing-capacity box
"""
import math

# Soil-behaviour-type Ic thresholds for the founding material (workbook B6).
IC_SAND_MAX = 2.25     # below -> sand/silty sand
IC_CLAY_MIN = 2.59     # at or above -> clay; between the two -> silt (ask user)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _zone_indices(depth, dex, founding_depth, zone_base_below_exc):
    """Rows of the zone of influence, footing base down to base of influence.

    Mirrors the workbook's MATCH(value, depth-below-excavation, 1): the largest
    row whose depth-below-excavation is <= the threshold, inclusive both ends.
    Depths here are below ORIGINAL ground level, so we add ``dex``.
    """
    foot_orig = dex + founding_depth
    zone_orig = dex + zone_base_below_exc
    foot_idx = None
    zone_idx = None
    for i, z in enumerate(depth):
        if z <= foot_orig:
            foot_idx = i
        if z <= zone_orig:
            zone_idx = i
    return foot_idx, zone_idx


def _geometric_sb(start=1e-6, ratio=1.1, max_settlement_mm=600.0, beq=1.0,
                  max_points=240):
    """s/B series: 0 then geometric (start * ratio**k), as in the workbook."""
    sb = [0.0, start]
    while len(sb) < max_points and (sb[-1] * beq * 1000.0) < max_settlement_mm:
        sb.append(sb[-1] * ratio)
    return sb


def _downsample(idx_lists, max_points=400):
    """Stride-sample parallel lists down to <= max_points (display only)."""
    n = len(idx_lists[0])
    if n <= max_points:
        return [list(s) for s in idx_lists]
    stride = (n // max_points) + 1
    out = [list(s[::stride]) for s in idx_lists]
    for k, s in enumerate(idx_lists):       # always keep the last point
        if (n - 1) % stride:
            out[k].append(s[-1])
    return out


def _representative_unit_weight(processed_cpt, params, foot_idx):
    """Single unit weight for the post-excavation stress recompute.

    Per Barry's spec, the program uses the FIRST unit weight entered in the
    CSV (workbook D24 = pileapp!E2). Honours an explicit override if provided.
    A future upgrade may switch to per-layer integration.
    """
    if params.get('unit_weight') not in (None, '', 0):
        return float(params['unit_weight'])
    gtot = processed_cpt['gtot']
    return float(gtot[0]) if gtot else 0.0


# ---------------------------------------------------------------------------
# sand branch  (Intro+sand-calc + Sand-output + BC-calc)
# ---------------------------------------------------------------------------
def _sand_branch(pc, p, gamma_const):
    depth = pc['depth']
    n = len(depth)
    B, L = p['B'], p['L']
    dex, D = p['dex'], p['founding_depth']
    K0, OCR0, ageing, creep = p['k0'], p['initial_ocr'], p['ageing_factor'], p['creep']
    beq = (B * L) ** 0.5

    zone_base = D + B ** 0.7                                  # D30 (below exc)
    foot_idx, zone_idx = _zone_indices(depth, dex, D, zone_base)

    fp = [0.0] * n           # AC col: peak friction angle (pre-exc Dr)
    dr_clamped = [0.0] * n   # N col
    svy = [0.0] * n          # L col: preconsolidation = OCR0 * sigma'v0 (pre-exc)
    for i in range(n):
        svp = pc['sig_v0_prime'][i]
        k_pre = (svp + 2 * K0 * svp) / 3.0                   # K col mean stress
        m = (1 / 2.93) * math.log((ageing * pc['qt'][i] * 1000) / (205 * k_pre ** 0.5))
        dr_clamped[i] = min(max(m, 0.1), 0.99)
        fp[i] = 32 + 3 * (5 * m - 1)                         # = 29 + 15*Dr
        svy[i] = OCR0 * svp

    fp_avg = sum(fp[foot_idx:zone_idx + 1]) / (zone_idx - foot_idx + 1)   # AC34
    kp = (1 + math.sin(math.radians(fp_avg))) / (1 - math.sin(math.radians(fp_avg)))  # AC37

    qc_post = [0.0] * n      # W col: qc after excavation unloading
    for i in range(n):
        z = depth[i]
        if z < dex:
            continue
        r = gamma_const * (z - dex) - pc['u0_kpa'][i]        # R: post-exc sigma'v0
        s_ocr = (svy[i] / r) if r else 0.0                   # S: OCR after excavation
        u = min(kp, K0 * s_ocr ** math.sin(math.radians(fp[i])))   # U: Ko after exc
        v = r * (1 + 2 * u) / 3.0                            # V: post-exc mean stress
        qc_from_dr = 0.001 * (1 / ageing) * 205 * (v ** 0.5) * math.exp(2.93 * dr_clamped[i])
        qc_post[i] = min(qc_from_dr, pc['qt'][i])

    span = range(foot_idx, zone_idx + 1)
    qc_avg = sum(qc_post[i] for i in span) / (zone_idx - foot_idx + 1)    # AC31 (MPa)
    ic_avg = sum(pc['lc'][i] for i in span) / (zone_idx - foot_idx + 1)   # AC36
    svy_foot = svy[foot_idx]                                              # AC33
    qb01 = 1000 * 0.16 * qc_avg                                          # AC32 (kPa)

    # ---- settlement curve (Sand-output) ----
    t_init = float(p.get('t_initial_days', 0.05))           # C8
    t_final = 365.0 * p['design_life_years']                # C9
    x = 3 + 70 * creep * math.log10(t_final / t_init)       # C23 time factor
    trans_sb = (svy_foot / (1000 * qc_avg)) ** 2 * x        # C24
    s_ocr_mm = 0.666 * trans_sb * 1000 * beq                # C25 settlement offset

    series = []
    for sb in _geometric_sb(beq=beq):
        q = 1000 * qc_avg * (sb / x) ** 0.5                 # O col (kPa)
        s_mm = 1000 * beq * sb - s_ocr_mm                   # P col (mm)
        if s_mm >= 0:                                       # Q col: plot s>0 only
            series.append([round(s_mm, 4), round(q, 4)])
    if not series or series[0][0] > 0:
        series.insert(0, [0.0, 0.0])

    bc = _traditional_bc_sand(pc, p, gamma_const, fp_avg, foot_idx)

    qd, qb, qa = _downsample([list(depth), list(pc['qt']), qc_post])

    return {
        'summary': {
            'qc_avg_mpa': qc_avg,
            'qb01_kpa': qb01,
            'svy_footing_kpa': svy_foot,
            'avg_friction_angle_deg': fp_avg,
            'avg_ic': ic_avg,
            'kp': kp,
            'time_factor_x': x,
            'settlement_offset_ocr_mm': s_ocr_mm,
            'bearing_capacity_cpt_kpa': qb01,
            'bearing_capacity_nq_ng_kpa': bc,
            'zone_top_m': depth[foot_idx],
            'zone_base_m': depth[zone_idx],
        },
        'curve': {
            'x_label': 'Settlement (mm)',
            'y_label': 'Bearing pressure (kPa)',
            'default_scale': {'x_max': 100, 'y_max': 500},
            'series': [{'name': 'Sand (drained)', 'points': series}],
        },
        'qc_before_after': {                                # for the qc plot
            'depth': qd,
            'qc_before': qb,
            'qc_after': qa,
        },
    }


def _traditional_bc_sand(pc, p, gamma_const, fp_avg, foot_idx):
    """BC-calc sand: Vesic Nq/Ng with shape & depth factors.

    Surcharge uses the post-excavation effective vertical stress at the
    footing row (R column in the workbook): gamma * D - u0_at_footing.
    Unit weight for the N_gamma term is the profile value (workbook C16
    fix per Barry: 'the value should be the unit weight specified in the
    CPT data').
    """
    B, L, D = p['B'], p['L'], p['founding_depth']
    phi = math.radians(fp_avg)
    b_over_l, d_over_b = B / L, D / B
    sq = 1 + b_over_l * math.tan(phi)
    sg = 1 - 0.2 * b_over_l
    dq = (1 + 2 * math.tan(phi) * (1 - math.sin(phi)) ** 2 *
          (d_over_b if d_over_b < 1 else math.atan(d_over_b)))
    nq = math.exp(math.pi * math.tan(phi)) * math.tan(math.pi / 4 + phi / 2) ** 2
    ng = 2 * (nq + 1) * math.tan(phi)
    # effective surcharge at founding level after excavation (R col @ foot)
    u0_foot = pc['u0_kpa'][foot_idx] if foot_idx is not None else 0.0
    sigv0_base = max(gamma_const * D - u0_foot, 0.0)
    dw1 = p['water_table'] - p['dex'] - D                   # water depth below footing
    gamma_eff = gamma_const if dw1 > B else gamma_const - 10
    return sq * dq * nq * sigv0_base + 0.5 * ng * gamma_eff * B * sg * 1.0


# ---------------------------------------------------------------------------
# clay branch  (Clay-calc + Clay-output + BC-calc)
# ---------------------------------------------------------------------------
def _clay_branch(pc, p, gamma_const):
    depth = pc['depth']
    n = len(depth)
    B, L = p['B'], p['L']
    dex, D, nkt = p['dex'], p['founding_depth'], p['nkt']
    beq = (B * L) ** 0.5

    zone_base = D + 0.5 * B                                   # D30 clay (below exc)
    foot_idx, zone_idx = _zone_indices(depth, dex, D, zone_base)

    su = [0.0] * n; svy = [0.0] * n; ocr = [0.0] * n
    for i in range(n):
        qt_kpa = pc['qt'][i] * 1000
        net = qt_kpa - pc['sig_v0'][i]
        su[i] = net / nkt                                    # P col
        svy[i] = 0.33 * net                                  # O col preconsolidation
        z = depth[i]
        r = gamma_const * (z - dex) - pc['u0_kpa'][i] if z > dex else 0.0   # R col
        ocr[i] = (svy[i] / r) if (z > dex and r) else 0.0    # T col

    span = range(foot_idx, zone_idx + 1)
    cnt = zone_idx - foot_idx + 1
    su_avg = sum(su[i] for i in span) / cnt                  # AC31
    qt_avg_mpa = sum(pc['qt'][i] for i in span) / cnt
    qt_net = 1000 * qt_avg_mpa - D * gamma_const             # AC33
    qf = 0.45 * qt_net                                       # AC34
    ocr_avg = sum(ocr[i] for i in span) / cnt                # AC35
    svy_foot = svy[foot_idx]                                 # AC32
    a_imm = 2.8 if ocr_avg >= 3 else 2.2                     # G4
    a_lt = 2.0 if ocr_avg >= 3 else 1.4                      # G5

    cap = 0.42 * qt_net
    imm, lt = [], []
    for sb in _geometric_sb(beq=beq):
        s_mm = sb * beq * 1000
        imm.append([round(s_mm, 4), round(min(a_imm * qt_net * sb ** 0.5, cap), 4)])
        lt.append([round(s_mm, 4), round(min(a_lt * qt_net * sb ** 0.5, cap), 4)])

    bc = _traditional_bc_clay(p, su_avg, ocr_avg)

    return {
        'summary': {
            'qt_net_kpa': qt_net,
            'avg_su_kpa': su_avg,
            'avg_ocr': ocr_avg,
            'svy_footing_kpa': svy_foot,
            'a_immediate': a_imm,
            'a_longterm': a_lt,
            'bearing_capacity_cpt_kpa': qf,
            'bearing_capacity_nc_kpa': bc,
            'ultimate_curve_cap_kpa': cap,
            'zone_top_m': depth[foot_idx],
            'zone_base_m': depth[zone_idx],
        },
        'curve': {
            'x_label': 'Settlement (mm)',
            'y_label': 'Bearing pressure (kPa)',
            'default_scale': {'x_max': 100, 'y_max': 500},
            'series': [
                {'name': 'Immediate', 'points': imm},
                {'name': 'Long term', 'points': lt},
            ],
        },
    }


def _traditional_bc_clay(p, su_avg, ocr_avg):
    """BC-calc clay: qnet = 5.14 * sc * dc * correction * su (matches workbook)."""
    B, L, D = p['B'], p['L'], p['founding_depth']
    sc = 1 + (B / L) * 0.2
    dc = 1 + 0.4 * (D / B)
    corr = 1.0 if ocr_avg > 3 else 0.66
    return 5.14 * sc * dc * corr * su_avg


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _read_params(params):
    """Coerce + default the footing params (workbook defaults in parentheses)."""
    return {
        'water_table': float(params.get('water_table', 0)),
        'B': float(params['footing_width']),
        'L': float(params['footing_length']),
        'founding_depth': float(params['founding_depth']),       # D below excavation
        'dex': float(params.get('excavation_depth', 0)),
        'design_life_years': float(params.get('design_life_years', 50)),
        'initial_ocr': float(params.get('initial_ocr', 1)),      # D20
        'k0': float(params.get('k0', 0.5)),                      # D21
        'ageing_factor': float(params.get('ageing_factor', 0.66)),  # D22
        'creep': float(params.get('creep', 0.02)),               # D32
        'nkt': float(params.get('nkt', 15)),                     # D35
        't_initial_days': float(params.get('t_initial_days', 0.05)),
        'unit_weight': params.get('unit_weight'),
        'force_soil_model': params.get('force_soil_model'),      # 'sand'|'clay' override
    }


def calculate_shallow_footing_results(processed_cpt, params):
    """Step 4 for shallow foundations.

    Args:
        processed_cpt: dict from ``pre_input_calc`` (per-depth arrays incl.
            depth, qt, lc=Ic, sig_v0, sig_v0_prime, u0_kpa, gtot).
        params: footing geometry + analysis options (see ``_read_params``).

    Returns:
        dict with ``soil_decision``, ``warnings``, ``summary``, ``curve`` and,
        for sand, ``qc_before_after``.
    """
    print(f"Params received in calculate_shallow_footing_results: {params}")
    p = _read_params(params)
    depth = processed_cpt['depth']
    if not depth:
        raise ValueError('No CPT data supplied to shallow footing calculation.')

    B, dex, D = p['B'], p['dex'], p['founding_depth']
    warnings = []

    # ---- soil decision: average Ic over the (sand) zone of influence ----
    sand_zone_base = D + B ** 0.7
    foot_idx, zone_idx = _zone_indices(depth, dex, D, sand_zone_base)
    if foot_idx is None:
        raise ValueError('Founding depth is below the bottom of the CPT data.')
    if zone_idx is None or zone_idx <= foot_idx:
        zone_idx = min(foot_idx + 1, len(depth) - 1)
        warnings.append('CPT data barely reach the founding level; zone of '
                        'influence averaging is truncated.')

    avg_ic = sum(processed_cpt['lc'][foot_idx:zone_idx + 1]) / (zone_idx - foot_idx + 1)

    if avg_ic < IC_SAND_MAX:
        classification, default_model, message = (
            'sand', 'sand',
            'The material in the zone of influence of the footing is a sand or silty sand.')
    elif avg_ic < IC_CLAY_MIN:
        classification, default_model, message = (
            'silt', 'sand',
            'The material in the zone of influence beneath the footing is silt. '
            'Continue the analysis assuming the sand formulation?')
    else:
        classification, default_model, message = (
            'clay', 'clay',
            'The material in the zone of influence is clay; the clay formulation will be used.')

    soil_model = p['force_soil_model'] or default_model
    if soil_model not in ('sand', 'clay'):
        soil_model = default_model

    # ---- warnings (workbook B9/B10) ----
    dw1 = p['water_table'] - dex - D
    if dw1 < 1:
        warnings.append('Warning: water level is above footing level.')
    zone_base_below_exc = (D + B ** 0.7) if soil_model == 'sand' else (D + 0.5 * B)
    dt = dex + zone_base_below_exc
    if max(depth) < dt:
        warnings.append('The CPT data do not extend as far as the base of the '
                        'zone of influence of the footing.')

    gamma_const = _representative_unit_weight(processed_cpt, p, foot_idx)

    if soil_model == 'sand':
        result = _sand_branch(processed_cpt, p, gamma_const)
    else:
        result = _clay_branch(processed_cpt, p, gamma_const)

    result['soil_decision'] = {
        'avg_ic': avg_ic,
        'classification': classification,
        'soil_model_used': soil_model,
        'requires_user_confirmation': classification == 'silt' and not p['force_soil_model'],
        'message': message,
    }
    result['warnings'] = warnings
    result['inputs'] = {
        'footing_width_m': B, 'footing_length_m': p['L'],
        'founding_depth_m': D, 'excavation_depth_m': dex,
        'design_life_years': p['design_life_years'],
        'unit_weight_used_knm3': gamma_const,
    }
    return result
