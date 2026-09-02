"""
Shallow foundations (Step 4) — load-settlement of footings on sand, sandy
silt, silt / clayey silt, and clay from CPT data.

Follows the spec captured in ``Footing-settlement-forcoding-rev2.xlsx``
(Barry Lehane, 1-Sep-2026).

rev2 changed three cells against rev1, all on Clay-form. G4 and G5 gained
ceilings of 2.79 and 1.97 on the immediate and long term A factors; that is
the correction Barry asked for. G6, the Ic that drives CF1 and CF2, was also
replaced by the literal 2.37, which looks like a test value left in place:
at Ic 2.37 the workbook's own G4 and G5 come out at 1.46 and 1.03, so the
delivered file never exercises the MIN it adds. We keep taking Ic from the
zone of influence as rev1 did, so G6 is not mirrored here. A cell-by-cell
diff of all eight sheets shows nothing else moved.

rev1 (28-May-2026) made the settlement app applicable across the full SBT
range. Its changes over the revision before it:

* Four Ic bands replace the previous three:
    - Ic <= 2.05          -> sand (workbook sand formulation)
    - 2.05 < Ic < 2.35    -> sandy silt (sand formulation with Kc on qc,avg)
    - 2.35 <= Ic < 2.60   -> silt or clayey silt (modified clay)
    - Ic >= 2.60          -> clay (workbook clay formulation)
* Sand-form C8: Kc = 3.93*Ic^2 - 14.78*Ic + 14.78 (clamped to 1 for Ic<=2.05),
  applied to the zone-averaged qc.
* Clay-form A factors gain two Ic-dependent multipliers:
    CF1 = 3.93*Ic^2 - 14.78*Ic + 14.78
    CF2 = 1 + 1.18*(Ic - 2.05)
    A_immediate = (0.46 if OCR < 3 else 0.58) * CF1 * CF2   (uncapped in rev1)
    A_longterm  = (0.30 if OCR < 3 else 0.41) * CF1 * CF2   (uncapped in rev1)
* BC-calc sand: average phi' is reduced by 20*(Ic - 2.05) degrees when
  2.05 < Ic <= 2.35 (the Kp used in the qc-after-excavation column still
  uses the unreduced phi', per the workbook).
* BC-calc clay: su correction = 0.66 (OCR <= 3) or 1 (OCR > 3).

Excel cell references are kept inline for traceability:
  Intro+sand-calc -> sand branch and Ic decision
  Clay-calc       -> clay branch
  Sand-form       -> sand load-settlement curve
  Clay-form       -> clay load-settlement curves
  BC-calc         -> traditional bearing-capacity comparison
"""
import math

# Ic band boundaries (workbook B6).
IC_SAND_MAX = 2.05         # <= this  -> sand
IC_SANDY_SILT_MAX = 2.35   # <= this  -> sandy silt (modified sand)
IC_CLAY_MIN = 2.60         # >= this  -> clay; (2.35..2.60) is silt/clayey silt

# Ceilings on the clay A factors (rev2 Clay-form G4/G5). CF1 x CF2 keeps
# growing with Ic, so without a ceiling the A factors run past anything a clay
# can mobilise. Barry supplied 2.79 and 1.97 as literals. They happen to equal
# 0.58 and 0.41 times CF1 x CF2 evaluated at Ic = 2.60, the bottom of the clay
# band, to 2 dp; that equivalence holds only on the OCR >= 3 branch, where the
# ceilings bind from Ic 2.60. On the OCR < 3 branch they bind from Ic 2.69
# (immediate) and 2.72 (long term).
A_IMMEDIATE_MAX = 2.79     # G4
A_LONGTERM_MAX = 1.97      # G5


# ---------------------------------------------------------------------------
# Ic-dependent correction factors
# ---------------------------------------------------------------------------
def _kc_silt_correction(ic_avg):
    """Sand-form C8 silt correction factor on qc,avg.

    Ic <= 2.05         -> Kc = 1 (clean sand)
    2.05 < Ic <= 2.35  -> Kc = 3.93*Ic^2 - 14.78*Ic + 14.78 (~1 at 2.05, ~1.75 at 2.35)
    """
    if ic_avg <= IC_SAND_MAX:
        return 1.0
    return 3.93 * ic_avg ** 2 - 14.78 * ic_avg + 14.78


def _cf_clay(ic_avg):
    """Clay-form G8 (CF1) and G9 (CF2). Applied for all Ic > 2.35."""
    cf1 = 3.93 * ic_avg ** 2 - 14.78 * ic_avg + 14.78
    cf2 = 1 + 1.18 * (ic_avg - IC_SAND_MAX)
    return cf1, cf2


def _phi_reduction_for_silt(ic_avg):
    """BC-calc J3: phi reduction for sandy silt (degrees).

    Ic <= 2.05         -> 0 (no reduction; clean sand)
    2.05 < Ic <= 2.35  -> 20 * (Ic - 2.05)
    """
    if ic_avg <= IC_SAND_MAX:
        return 0.0
    return 20.0 * (ic_avg - IC_SAND_MAX)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _zone_indices(depth, dex, founding_depth, zone_base_below_exc):
    """Rows of the zone of influence, footing base down to base of influence.

    Mirrors the workbook's MATCH(value, depth-below-excavation, 1): the
    largest row whose depth-below-excavation is <= the threshold, inclusive
    both ends. Depths here are below ORIGINAL ground level, so we add dex.
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


def _representative_unit_weight(processed_cpt, params, foot_idx, zone_idx):
    """Single unit weight for the post-excavation stress recompute.

    An explicit user override is honoured first. Otherwise the average unit
    weight over the zone of influence (footing base to zone base) is used.
    Barry's original spec took the FIRST value in the file (workbook D24 =
    pileapp!E2), written for hand-built files with one constant unit-weight
    column, where the average gives the identical number. For contractor
    files the importer derives a per-depth unit weight and row 0 is often an
    fs=0 surface artifact (about 12 kN/m3), so the zone average is the
    faithful generalisation rather than a behaviour change.
    """
    if params.get('unit_weight') not in (None, '', 0):
        return float(params['unit_weight'])
    gtot = processed_cpt['gtot']
    if not len(gtot):
        return 0.0
    if foot_idx is None:
        return float(gtot[0])
    end = zone_idx if (zone_idx is not None and zone_idx >= foot_idx) else foot_idx
    span = gtot[foot_idx:end + 1]
    return float(sum(span) / len(span))


def _classify(avg_ic):
    """Return (classification_slug, branch, message) for the four-band model."""
    if avg_ic <= IC_SAND_MAX:
        return (
            'sand', 'sand',
            'The material in the zone of influence of the footing is assessed '
            'to be a sand and the analysis will use the sand formulation.'
        )
    if avg_ic < IC_SANDY_SILT_MAX:
        return (
            'sandy_silt', 'sand',
            'The material in the zone of influence beneath the footing is '
            'assessed to be a sandy silt and the analysis will use a modified '
            'form of the sand formulation.'
        )
    if avg_ic < IC_CLAY_MIN:
        return (
            'silt_clayey_silt', 'clay',
            'The material in the zone of influence is assessed to be silt or '
            'clayey silt and a modified form of clay formulations will be used.'
        )
    return (
        'clay', 'clay',
        'The material in the zone of influence of the footing is assessed to '
        'be a clay and the analysis will use the clay formulation.'
    )


# ---------------------------------------------------------------------------
# sand branch  (Intro+sand-calc + Sand-form + BC-calc)
# ---------------------------------------------------------------------------
def _sand_branch(pc, p, gamma_const, ic_avg):
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
        fp[i] = 32 + 3 * (4.7 * m - 1)                       # AC col (workbook uses 4.7*Dr)
        svy[i] = OCR0 * svp

    fp_avg = sum(fp[foot_idx:zone_idx + 1]) / (zone_idx - foot_idx + 1)   # AC34
    kp = (1 + math.sin(math.radians(fp_avg))) / (1 - math.sin(math.radians(fp_avg)))  # AC37 — uses raw fp_avg

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
    cnt = zone_idx - foot_idx + 1
    qc_avg_raw = sum(qc_post[i] for i in span) / cnt                      # AC31 (MPa)
    kc = _kc_silt_correction(ic_avg)                                       # Sand-form C8
    qc_avg = kc * qc_avg_raw                                              # Sand-form C20
    svy_foot = svy[foot_idx]                                              # AC33
    qb01 = 1000 * 0.16 * qc_avg                                           # Sand-form C21

    # ---- settlement curve (Sand-form N..Q) ----
    t_init = float(p.get('t_initial_days', 0.05))           # Sand-form C9
    t_final = 365.0 * p['design_life_years']                # Sand-form C10
    x = 3 + 70 * creep * math.log10(t_final / t_init)       # Sand-form C24 time factor
    trans_sb = (svy_foot / (1000 * qc_avg)) ** 2 * x        # Sand-form C25
    s_ocr_mm = 0.666 * trans_sb * 1000 * beq                # Sand-form C26 settlement offset

    series = []
    for sb in _geometric_sb(beq=beq):
        q = 1000 * qc_avg * (sb / x) ** 0.5                 # O col (kPa)
        s_mm = 1000 * beq * sb - s_ocr_mm                   # P col (mm)
        if s_mm >= 0:                                       # Q col: plot s>0 only
            series.append([round(s_mm, 4), round(q, 4)])
    if not series or series[0][0] > 0:
        series.insert(0, [0.0, 0.0])

    phi_red = _phi_reduction_for_silt(ic_avg)               # BC-calc J3
    fp_bc = fp_avg - phi_red                                # BC-calc J4
    bc = _traditional_bc_sand(pc, p, gamma_const, fp_bc, foot_idx)

    qd, qb, qa = _downsample([list(depth), list(pc['qt']), qc_post])

    return {
        'summary': {
            'qc_avg_mpa': qc_avg,
            'qc_avg_raw_mpa': qc_avg_raw,
            'kc_silt_correction': kc,
            'qb01_kpa': qb01,
            'svy_footing_kpa': svy_foot,
            'avg_friction_angle_deg': fp_avg,
            'phi_reduction_deg': phi_red,
            'phi_used_in_bc_deg': fp_bc,
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


def _traditional_bc_sand(pc, p, gamma_const, fp_used_deg, foot_idx):
    """BC-calc sand: net Vesic Nq / Ng with shape and depth factors (C24).

    ``fp_used_deg`` is already reduced for sandy silt per BC-calc J4.
    Surcharge uses the post-excavation effective vertical stress at the
    matched footing-base row (R column = gamma*(z-dex) - u0), so it lines up
    with intro AC32 to numerical precision rather than relying on the nominal
    founding depth D. The workbook subtracts C15 (= sigma'v0 at footing) at
    the end to give a NET capacity.

    Note: the workbook's BC-calc C16 references 'From pileapp-2'!E2 for the
    sand unit weight, which is a wiring bug — the sand sample's gamma should
    come from pileapp-1. We use the analysed sample's own gamma here, so the
    BC sand number can differ from the workbook's printed C24 for the sand
    test case while still using the same formula structure.
    """
    B, L, D = p['B'], p['L'], p['founding_depth']
    phi = math.radians(fp_used_deg)
    b_over_l, d_over_b = B / L, D / B
    sq = 1 + b_over_l * math.tan(phi)                         # C10
    sg = 1 - 0.2 * b_over_l                                   # C18
    dq = (1 + 2 * math.tan(phi) * (1 - math.sin(phi)) ** 2 *  # C14
          (d_over_b if d_over_b < 1 else math.atan(d_over_b)))
    nq = math.exp(math.pi * math.tan(phi)) * math.tan(math.pi / 4 + phi / 2) ** 2  # C20
    ng = 2 * (nq + 1) * math.tan(phi)                         # C21
    # Surcharge at founding level after excavation (R col @ matched row).
    z_foot = pc['depth'][foot_idx] if foot_idx is not None else (p['dex'] + D)
    u0_foot = pc['u0_kpa'][foot_idx] if foot_idx is not None else 0.0
    sigv0_base = max(gamma_const * (z_foot - p['dex']) - u0_foot, 0.0)  # C15 = AC32
    dw1 = p['water_table'] - p['dex'] - D                     # water depth below footing
    gamma_eff = gamma_const if dw1 > B else gamma_const - 10  # C17 (buoyant if WT within B)
    return (sq * dq * nq * sigv0_base                         # term1: Nq*sigv0 etc.
            + 0.5 * ng * gamma_eff * B * sg * 1.0             # term2: Ng*gamma'*B/2
            - sigv0_base)                                     # NET (workbook -C15)


# ---------------------------------------------------------------------------
# clay branch  (Clay-calc + Clay-form + BC-calc)
# ---------------------------------------------------------------------------
def _clay_branch(pc, p, gamma_const, ic_avg):
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

    # Clay-form A factors. C1, C2 are OCR-dependent base coefficients;
    # CF1, CF2 are Ic-dependent multipliers (apply for all Ic > 2.35).
    c1 = 0.58 if ocr_avg >= 3 else 0.46                      # G10
    c2 = 0.41 if ocr_avg >= 3 else 0.30                      # G11
    cf1, cf2 = _cf_clay(ic_avg)                              # G8, G9
    a_imm = min(c1 * cf1 * cf2, A_IMMEDIATE_MAX)             # G4
    a_lt = min(c2 * cf1 * cf2, A_LONGTERM_MAX)               # G5

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
            'avg_ic': ic_avg,
            'svy_footing_kpa': svy_foot,
            'cf1': cf1, 'cf2': cf2,
            'c1_base': c1, 'c2_base': c2,
            'a_immediate': a_imm,
            'a_longterm': a_lt,
            'a_immediate_max': A_IMMEDIATE_MAX,
            'a_longterm_max': A_LONGTERM_MAX,
            # The two ceilings bind at different Ic, so track them separately.
            # bool() keeps these plain Python bools: the template branches on
            # them and a numpy scalar would survive JSON round-tripping as a
            # truthy string.
            'a_immediate_at_ceiling': bool((c1 * cf1 * cf2) > A_IMMEDIATE_MAX),
            'a_longterm_at_ceiling': bool((c2 * cf1 * cf2) > A_LONGTERM_MAX),
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
    """BC-calc clay: qnet = 5.14 * sc * dc * correction * su.

    correction is 1 for OCR > 3 and 0.66 otherwise (BC-calc C28).
    """
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
    """Step 4 for shallow foundations (rev2 — four Ic bands, capped clay A).

    Args:
        processed_cpt: dict from ``pre_input_calc`` (per-depth arrays incl.
            depth, qt, lc=Ic, sig_v0, sig_v0_prime, u0_kpa, gtot).
        params: footing geometry + analysis options (see ``_read_params``).

    Returns:
        dict with ``soil_decision``, ``warnings``, ``summary``, ``curve`` and,
        for the sand branch, ``qc_before_after``.
    """
    print(f"Params received in calculate_shallow_footing_results: {params}")
    p = _read_params(params)
    depth = processed_cpt['depth']
    if not depth:
        raise ValueError('No CPT data supplied to shallow footing calculation.')

    B, dex, D = p['B'], p['dex'], p['founding_depth']
    warnings = []

    # ---- soil decision: average Ic over the sand zone of influence ----
    # Per the workbook, Ic for classification is averaged over the SAND zone
    # (D + B^0.7) below the footing. The clay branch then re-zones to D + 0.5B
    # for its own averages.
    sand_zone_base = D + B ** 0.7
    foot_idx, zone_idx = _zone_indices(depth, dex, D, sand_zone_base)
    if foot_idx is None:
        raise ValueError('Founding depth is below the bottom of the CPT data.')
    if zone_idx is None or zone_idx <= foot_idx:
        zone_idx = min(foot_idx + 1, len(depth) - 1)
        warnings.append('CPT data barely reach the founding level; zone of '
                        'influence averaging is truncated.')

    avg_ic = sum(processed_cpt['lc'][foot_idx:zone_idx + 1]) / (zone_idx - foot_idx + 1)
    classification, default_branch, message = _classify(avg_ic)

    # Advanced override: user can still force 'sand' or 'clay' regardless of Ic.
    forced = p['force_soil_model']
    if forced in ('sand', 'clay'):
        branch = forced
    else:
        branch = default_branch

    # ---- warnings (workbook B9 / B10) ----
    dw1 = p['water_table'] - dex - D
    if dw1 < 1:
        warnings.append('Warning: water level is above footing level.')
    zone_base_below_exc = (D + B ** 0.7) if branch == 'sand' else (D + 0.5 * B)
    dt = dex + zone_base_below_exc
    if max(depth) < dt:
        warnings.append('The CPT data do not extend as far as the base of the '
                        'zone of influence of the footing.')

    gamma_const = _representative_unit_weight(processed_cpt, p, foot_idx, zone_idx)
    if gamma_const and gamma_const < 14.0:
        warnings.append(
            'Unit weight used (%.1f kN/m3) is unusually low for soil. Check '
            'the unit-weight column in the CPT file, or enter a site unit '
            'weight in the parameters form.' % gamma_const)

    if branch == 'sand':
        result = _sand_branch(processed_cpt, p, gamma_const, avg_ic)
    else:
        result = _clay_branch(processed_cpt, p, gamma_const, avg_ic)

    result['soil_decision'] = {
        'avg_ic': avg_ic,
        'classification': classification,
        'soil_model_used': branch,
        'requires_user_confirmation': False,
        'message': message,
    }
    result['warnings'] = warnings
    result['inputs'] = {
        'footing_width_m': B, 'footing_length_m': p['L'],
        'founding_depth_m': D, 'excavation_depth_m': dex,
        'design_life_years': p['design_life_years'],
        'unit_weight_used_knm3': gamma_const,
        'unit_weight_source': ('entered in form'
                               if p.get('unit_weight') not in (None, '', 0)
                               else 'CPT average over zone of influence'),
    }
    return result
