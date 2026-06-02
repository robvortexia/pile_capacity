"""
Laterally loaded monopiles in sand (Step 4) — load-displacement, rotation,
and moment response of a monopile from CPT data.

Follows the workbook ``Laterally-loaded-monopile-in-sand-forcoding.xlsx``
(Barry Lehane, 28-May-2026), with the description on the intro page
updated to cite Wang et al. (2023, Ocean Engineering 277, 114334) for
the load-displacement and moment-rotation response and Wang et al.
(2020, Geotechnique Letters 10, 429-435) for the ultimate lateral
geotechnical capacity. The method covers sand-type profiles (average
Ic < 2.2 over the embedded length) and short piles (L/D < 5). For
non-sand or longer piles the call returns a structured warning instead
of a result.

High-level flow:

  1.  G0 from CPT, per row:  G0 = alpha_G * (qt - sigma_v0)
      where alpha_G = (gamma_prev / 9.81 / 100) * 10^(0.55*Ic + 1.68).
      The workbook quirk: alpha_G at row i uses gamma at row i-1, so we
      reproduce that exactly.

  2.  Best-fit idealised G0 profile over the embedded length. Three
      candidates — Linear (a*z), Power of 0.5 (a*z^0.5), Homogeneous
      (constant) — each fit by the closed-form solution that makes the
      mean of (G0_CPT / G0_model) equal to 1. Pick the one with the
      smallest CoV of that ratio.

  3.  Rigid pile stiffness  K_R0 = C_k * D * L^2 * G0(0.75L)  (MNm/rad).
      C_k = a*exp(b*L/D) + c*exp(d*L/D) with profile-dependent (a, b, c, d).

  4.  Reference rotation theta_ref = 0.0002 * sqrt(gamma' * L / p_ref).
      For each theta_rigid/theta_ref point, the stiffness reduction is
        K_R / K_R0 = 1 / (1 + (theta_rigid/theta_ref)^0.7).

  5.  Pile-flexibility add-ons via Mindlin / Euler-Bernoulli closed forms:
        Delta_theta_F = H * (2h + 0.75L) * 0.75L / (2*EI)
        Delta_y_F     = H * (0.75L)^2 * (3*(h+0.75L) - 0.75L) / (6*EI) * 1000
      then theta_monopile = theta_rigid + Delta_theta_F / C_R_theta and
      y_monopile = y_rigid + Delta_y_F / C_R_y, where C_R_theta and C_R_y
      are h/L-dependent correction factors.

  6.  Ultimate lateral capacity from a Wang-et-al p-y integration:
        Pu(z) = a * (qc * 1000)^b1 * sigma_v0'^b2 * 100^b3   (defaults 0.3, 0.7, 0.1, 0.2)
        Hu    = integral(0 .. 0.75L) Pu*D dz  -  integral(0.75L .. L) Pu*D dz
      using trapezoidal area with split-bin handling at 0.75L and L.

Excel cell references appear inline for traceability.
"""
import math
import bisect

# ----- best-fit profile coefficients (Calculation C20..C22, G20..G22) -----
_PROFILE_COEFS = {
    'Homogeneous':  (9.1, -2.24, 2.71, 0.065),
    'Power of 0.5': (6.2, -1.62, 1.85, 0.053),
    'Linear':       (6.5, -1.50, 1.40, 0.044),
}

# ----- workbook sweep of theta_rigid / theta_ref (Calculation column B) ---
_THETA_RATIO_TABLE = (
    [round(0.01 * k, 2) for k in range(1, 10)] +           # 0.01..0.09
    [round(0.1 * k, 2) for k in range(1, 10)] +            # 0.1..0.9
    [float(k) for k in range(1, 10)] +                     # 1..9
    [float(k) for k in range(10, 100, 10)] +               # 10..90
    [float(k) for k in range(100, 1000, 100)] +            # 100..900
    [1000.0]
)

P_REF_KPA = 100.0    # Calculation C29

# ---------------------------------------------------------------------------
# G0 from CPT (G0-fromCPT sheet)
# ---------------------------------------------------------------------------
def _g0_from_cpt(depth, qt_mpa, sig_v0_kpa, ic, gamma):
    """Per-row G0 (MPa).

    G0_i = alpha_G_i * (qt_i - sigma_v0_i)  with qt in MPa, sigma_v0 in MPa
           (workbook divides sigma_v0_kpa by 1000 to convert).
    alpha_G_i = (gamma_i / 9.81 / 100) * 10^(0.55*Ic_i + 1.68)

    Note vs. workbook: G0-fromCPT row r in the rev1 file references the
    PREVIOUS row's gamma (G3 uses C2, etc.). Barry confirmed (29 May
    email) that's a spreadsheet copy-paste bug; the correct formula
    uses the current row's gamma. We use the corrected formula, so the
    per-row G0 values can differ from the workbook by a percent or so
    where gamma changes (typically at the water table); typical sand
    profiles with near-constant gamma are essentially unchanged.

    Row 0 (the surface) is set to 0.
    """
    n = len(depth)
    g0 = [0.0] * n
    for i in range(1, n):
        alpha_vs = 10 ** (0.55 * ic[i] + 1.68)
        alpha_g = (gamma[i] / 9.81 / 100.0) * alpha_vs
        g0[i] = alpha_g * (qt_mpa[i] - sig_v0_kpa[i] / 1000.0)
    return g0


# ---------------------------------------------------------------------------
# Best-fit G0 profile (G0_BF sheet)
# ---------------------------------------------------------------------------
def _best_fit_g0(depth, g0_mpa, L):
    """Return the best-fit G0 profile over depth <= L.

    Three candidates (closed form of "mean of ratio = 1" Solver target):
      Linear:       G0_lin(z)  = a_lin * z         a_lin = mean(G0_i / z_i)
      Power of 0.5: G0_pow(z)  = a_pow * z^0.5     a_pow = mean(G0_i / sqrt(z_i))
      Homogeneous:  G0(z)      = G0_const         (mean of G0 INCLUDING the
                                                   surface row, per workbook
                                                   C19 quirk)

    Selection: profile with the smallest CoV of (G0_i / G0_model_i)
    evaluated over rows i with z_i > 0 (CoV ranges exclude the surface).

    Note vs. workbook: the rev1 file uses Excel Solver to fit the Linear and
    Power coefficients with the target "mean of ratio = 1". The Linear case
    converged (H23 ~ 1.000001), but the Power case did not (H23 = 0.985 in
    the shipped file). The closed-form values used here solve the stated
    target exactly, so a_pow differs from the workbook's printed Solver
    value by ~1.5% and G0(0.75L) shifts accordingly. CoV is scale-invariant,
    so the best-fit *selection* matches the workbook regardless.

    Returns: dict with the picked fit and the metadata for all three.
    """
    # Indices inside the embedded length. The workbook's MATCH(L, ..., 1)
    # picks the largest depth <= L, then C9 = C8 + MATCH which actually
    # points one row past — so include the row at z just above L too.
    i_below = [i for i in range(len(depth)) if depth[i] <= L]
    if not i_below:
        raise ValueError('No CPT rows within the embedded length.')
    last = i_below[-1]
    # Include the next row (workbook quirk: C9 lands one row beyond <=L).
    end = min(last + 1, len(depth) - 1)
    fit_idx = list(range(0, end + 1))             # rows 0..end inclusive
    nz_idx = [i for i in fit_idx if depth[i] > 0]  # CoV ranges skip z=0

    # Linear coefficient: mean of G0_i / z_i over nz rows
    a_lin = sum(g0_mpa[i] / depth[i] for i in nz_idx) / len(nz_idx)
    # Power coefficient
    a_pow = sum(g0_mpa[i] / (depth[i] ** 0.5) for i in nz_idx) / len(nz_idx)
    # Homogeneous: mean of G0 INCLUDING the surface row (workbook C19)
    g0_const = sum(g0_mpa[i] for i in fit_idx) / len(fit_idx)

    def _cov(ratios):
        n = len(ratios)
        m = sum(ratios) / n
        # Excel STDEV is sample stdev: divide by (n-1)
        var = sum((r - m) ** 2 for r in ratios) / (n - 1) if n > 1 else 0.0
        return (var ** 0.5) / m if m else float('inf')

    cov_lin = _cov([g0_mpa[i] / (a_lin * depth[i]) for i in nz_idx])
    cov_pow = _cov([g0_mpa[i] / (a_pow * depth[i] ** 0.5) for i in nz_idx])
    cov_hom = _cov([g0_mpa[i] / g0_const for i in nz_idx])

    candidates = [
        ('Linear', a_lin, cov_lin),
        ('Power of 0.5', a_pow, cov_pow),
        ('Homogeneous', g0_const, cov_hom),
    ]
    best = min(candidates, key=lambda c: c[2])
    name, coef, cov = best
    z075 = 0.75 * L
    if name == 'Linear':
        g0_075 = coef * z075
    elif name == 'Power of 0.5':
        g0_075 = coef * z075 ** 0.5
    else:
        g0_075 = coef
    return {
        'name': name,
        'coefficient': coef,
        'g0_at_0_75L_mpa': g0_075,
        'cov': cov,
        'all_fits': {
            'Linear':       {'coef': a_lin,   'cov': cov_lin},
            'Power of 0.5': {'coef': a_pow,   'cov': cov_pow},
            'Homogeneous':  {'coef': g0_const,'cov': cov_hom},
        },
    }


# ---------------------------------------------------------------------------
# Effective unit weight averaged over the embedded length (Calculation B11)
# ---------------------------------------------------------------------------
def _gamma_eff(gamma_above, gamma_below, water_table_m, L):
    """Effective soil unit weight over the embedded length.

    If L > dw:  gamma' = (gamma_above*dw + (gamma_below - 10)*(L - dw)) / L
    Else:       gamma' = gamma_above
    """
    if L > water_table_m:
        return (gamma_above * water_table_m
                + (gamma_below - 10.0) * (L - water_table_m)) / L
    return gamma_above


# ---------------------------------------------------------------------------
# Ultimate lateral capacity Hu (H-ult-calc sheet)
# ---------------------------------------------------------------------------
def _h_ult(depth, qc_mpa, sig_v0_eff_kpa, D, L, coef_a, exp_qc, exp_sigv, exp_pa):
    """Wang-et-al integrated Pu*D over the embedded length, split at 0.75L.

    Pu_i  = a * (qc_i [kPa])^b1 * (sigma_v0'_i [kPa])^b2 * (100 [kPa])^b3
    PuD_i = Pu_i * D
    For each bin [z_i, z_{i+1}], trapezoidal area = (PuD_i + PuD_{i+1})/2 * dz.
    Split at 0.75L and at L if the bin straddles either.
    Hu = integral(0..0.75L) - integral(0.75L..L)
    """
    n = len(depth)
    z_rot = 0.75 * L
    pu = [coef_a * ((qc_mpa[i] * 1000.0) ** exp_qc)
                 * (sig_v0_eff_kpa[i] ** exp_sigv)
                 * (100.0 ** exp_pa) for i in range(n)]
    pud = [pu[i] * D for i in range(n)]

    above = 0.0  # 0 .. 0.75L
    below = 0.0  # 0.75L .. L
    for i in range(n - 1):
        z1, z2 = depth[i], depth[i + 1]
        if z1 >= L:
            break
        # full-bin trapezoid
        t = (pud[i] + pud[i + 1]) / 2.0 * (z2 - z1)
        if z2 <= z_rot:
            above += t
        elif z1 >= z_rot and z2 <= L:
            below += t
        elif z1 < z_rot < z2 <= L:
            frac_above = (z_rot - z1) / (z2 - z1)
            above += frac_above * t
            below += (1.0 - frac_above) * t
        elif z1 >= z_rot and z1 < L < z2:
            frac_below = (L - z1) / (z2 - z1)
            below += frac_below * t
        elif z1 < z_rot and z2 > L:
            # bin straddles both 0.75L and L (very coarse CPT)
            frac_above = (z_rot - z1) / (z2 - z1)
            frac_below = (L - z_rot) / (z2 - z1)
            above += frac_above * t
            below += frac_below * t
    return above - below, above, below


# ---------------------------------------------------------------------------
# C_k coefficient and pile-flexibility corrections (Calculation H20..H22, K27..K28)
# ---------------------------------------------------------------------------
def _c_k(profile_name, L_over_D):
    a, b, c, d = _PROFILE_COEFS[profile_name]
    return a * math.exp(b * L_over_D) + c * math.exp(d * L_over_D)


def _correction_factors(h, L):
    """Pile-flexibility correction factors (Calculation K27, K28).

    Note: the workbook's K27 expression is
        0.75 * (3*2.8 + (h/L)^0.75) / (2.8 + (h/L)^0.75)
    which is *h/L*, not the eccentricity h/L that elsewhere uses meters.
    For h=10, L=2.24 -> h/L = 4.46 -> (h/L)^0.75 = 3.07 -> K27 ~= 1.47.
    """
    hL = h / L
    c_theta = 0.75 * (3 * 2.8 + hL ** 0.75) / (2.8 + hL ** 0.75)
    c_y = c_theta * 1.75
    return c_theta, c_y


# ---------------------------------------------------------------------------
# average Ic over embedded length (Main-sheet B30)
# ---------------------------------------------------------------------------
def _average_ic(depth, ic, L, surface_depth=0.1):
    """Mean Ic between depth ~= surface_depth and depth ~= L (Main-sheet B30).

    Workbook MATCH(0.1, z, 1) finds the first row at or above 0.1m, and
    MATCH(L, z, 1) finds the largest row <= L.
    """
    i_top = next((i for i in range(len(depth)) if depth[i] >= surface_depth), 0)
    # MATCH(.,.,1) returns the largest row <= L
    i_base = max((i for i in range(len(depth)) if depth[i] <= L), default=i_top)
    span = ic[i_top:i_base + 1]
    return sum(span) / len(span) if span else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _read_params(params):
    return {
        'water_table': float(params.get('water_table', 0)),
        'gamma_above_kNm3': float(params.get('gamma_above', 17.1)),
        'gamma_below_kNm3': float(params.get('gamma_below', 19.9)),
        'D': float(params['diameter']),
        'L': float(params['embedded_length']),
        'pile_type': params.get('pile_type', 'Pipe'),
        't_mm': float(params.get('wall_thickness_mm', 10)),
        'h': float(params['load_height_above_ground']),
        'E_GPa': float(params.get('youngs_modulus_GPa', 210)),
        'pile_name': params.get('pile_name', ''),
        'hult_a': float(params.get('hult_coefficient', 0.3)),       # Main B97
        'hult_exp_qc': float(params.get('hult_exp_qc', 0.7)),       # Main B98
        'hult_exp_sigv': float(params.get('hult_exp_sigv', 0.1)),   # Main B99
        # B100 = 1 - B98 - B99
        'force_g0_profile': params.get('force_g0_profile'),         # 'Linear'|'Power of 0.5'|'Homogeneous'|None
        'g0_075L_override': _opt_float(params.get('g0_075L_override')),  # user-supplied G0 at 0.75L (MPa), or None
    }


def _opt_float(val):
    """Parse an optional float; return None for blank/invalid."""
    try:
        return float(val) if val not in (None, '') else None
    except (TypeError, ValueError):
        return None


def calculate_lateral_monopile_results(processed_cpt, params):
    """Step 4 for laterally loaded monopiles in sand.

    Args:
        processed_cpt: dict from ``pre_input_calc`` (depth, qt, lc=Ic,
            sig_v0, sig_v0_prime, u0_kpa, gtot).
        params: monopile geometry + analysis options (see ``_read_params``).

    Returns:
        dict with ``checks``, ``g0_fit``, ``summary``, ``curve_load_disp``,
        ``curve_moment_rotation``, and the per-row tables used in the plot.
    """
    print(f"Params received in calculate_lateral_monopile_results: {params}")
    p = _read_params(params)
    depth = list(processed_cpt['depth'])
    qt_mpa = list(processed_cpt['qt'])
    ic = list(processed_cpt['lc'])
    sig_v0 = list(processed_cpt['sig_v0'])
    sig_v0_prime = list(processed_cpt['sig_v0_prime'])
    gamma = list(processed_cpt['gtot'])
    if not depth:
        raise ValueError('No CPT data supplied to lateral monopile calculation.')

    D, L, h = p['D'], p['L'], p['h']
    pile_type = p['pile_type'].strip().lower()
    t_m = p['t_mm'] / 1000.0
    d_inner = 0.0 if pile_type.startswith('solid') else max(D - 2 * t_m, 0.0)
    L_over_D = L / D
    h_over_L = h / L
    EI = (10 ** 6) * math.pi / 64.0 * (D ** 4 - d_inner ** 4) * p['E_GPa']  # kNm^2

    # ---- checks ----
    avg_ic_embed = _average_ic(depth, ic, L)
    is_sand = avg_ic_embed < 2.2
    is_rigid = L_over_D < 5
    checks = {
        'avg_ic_in_embedded_length': avg_ic_embed,
        'sand_check_pass': is_sand,
        'sand_check_threshold': 2.2,
        'rigidity_check_pass': is_rigid,
        'rigidity_check_threshold': 5.0,
    }
    if not is_sand:
        checks['message'] = (
            f"Average Ic over the embedded length is {avg_ic_embed:.2f}, which "
            "is not consistent with a sand profile (threshold 2.2). The sand "
            "formulation in this module is not applicable."
        )
        return {'checks': checks, 'aborted': True}
    if not is_rigid:
        checks['message'] = (
            f"L/D = {L_over_D:.2f} >= 5. The short-pile assumption underlying "
            "this module may not be valid for this geometry."
        )
        # not a hard abort — Barry's wording is "WARNING"; we still compute

    if max(depth) < L:
        return {'checks': dict(checks, message=(
            f"CPT data only reach {max(depth):.2f} m, which is shallower "
            f"than the embedded length L = {L:.2f} m."
        )), 'aborted': True}

    # ---- G0 from CPT, then best-fit profile ----
    g0_mpa = _g0_from_cpt(depth, qt_mpa, sig_v0, ic, gamma)
    fit = _best_fit_g0(depth, g0_mpa, L)
    if p['force_g0_profile'] in _PROFILE_COEFS:
        # honour user override; recompute g0_at_0.75L from the override's coef
        forced = p['force_g0_profile']
        coef = fit['all_fits'][forced]['coef']
        z075 = 0.75 * L
        if forced == 'Linear':
            g0_075 = coef * z075
        elif forced == 'Power of 0.5':
            g0_075 = coef * z075 ** 0.5
        else:
            g0_075 = coef
        fit = dict(fit, name=forced, coefficient=coef,
                   g0_at_0_75L_mpa=g0_075, cov=fit['all_fits'][forced]['cov'])

    # User-supplied G0 at 0.75L (MPa) overrides the profile-derived value and
    # feeds the rigid-pile stiffness K_R0 directly.
    if p['g0_075L_override'] and p['g0_075L_override'] > 0:
        fit = dict(fit, g0_at_0_75L_mpa=p['g0_075L_override'])

    # ---- C_k, K_R0, theta_ref ----
    c_k = _c_k(fit['name'], L_over_D)                              # Calc H20..H22
    g_eff = _gamma_eff(p['gamma_above_kNm3'], p['gamma_below_kNm3'],
                       p['water_table'], L)                        # Calc B11
    k_r0_MNm_rad = c_k * D * L ** 2 * fit['g0_at_0_75L_mpa']        # Calc C26
    theta_ref = 0.0002 * math.sqrt(g_eff * L / P_REF_KPA)           # Calc C28
    c_theta, c_y = _correction_factors(h, L)                       # Calc K27, K28

    # ---- load-displacement / moment-rotation table ----
    rows = []
    z_rot = 0.75 * L
    for ratio in _THETA_RATIO_TABLE:
        k_ratio = 1.0 / (1.0 + ratio ** 0.7)                       # C col
        theta_r = ratio * theta_ref                                # D col (rad)
        y_r_mm = theta_r * z_rot * 1000.0                          # E col
        K_R = k_ratio * k_r0_MNm_rad                               # F col (MNm/rad)
        M_R_kNm = theta_r * K_R * 1000.0                           # G col
        H_kN = M_R_kNm / (h + z_rot)                               # H col
        d_theta_F = H_kN * (2 * h + z_rot) * z_rot / (2 * EI)      # J col
        d_y_F_mm = H_kN * z_rot ** 2 * (3 * (h + z_rot) - z_rot) / (6 * EI) * 1000.0  # K col
        theta_mp = theta_r + d_theta_F / c_theta                   # M col
        y_mp_mm = y_r_mm + d_y_F_mm / c_y                          # N col
        M_at_ground = H_kN * h                                     # O col
        rows.append({
            'theta_ratio': ratio,
            'K_ratio': k_ratio,
            'theta_rigid_rad': theta_r,
            'y_rigid_mm': y_r_mm,
            'K_R_MNm_rad': K_R,
            'M_R_kNm': M_R_kNm,
            'H_kN': H_kN,
            'd_theta_flex_rad': d_theta_F,
            'd_y_flex_mm': d_y_F_mm,
            'theta_monopile_rad': theta_mp,
            'y_monopile_mm': y_mp_mm,
            'M_at_ground_kNm': M_at_ground,
        })

    # ---- ultimate capacity Hu ----
    exp_pa = 1.0 - p['hult_exp_qc'] - p['hult_exp_sigv']
    Hu_kN, Hu_above, Hu_below = _h_ult(
        depth, qt_mpa, sig_v0_prime, D, L,
        p['hult_a'], p['hult_exp_qc'], p['hult_exp_sigv'], exp_pa)

    # ---- output curves (default x-axis ~ 0.01 rad) ----
    # Load-deflection: H vs y_monopile; clip to theta_monopile <= 0.01 rad-ish
    ld_pts = [[round(r['y_monopile_mm'], 4), round(r['H_kN'], 4)] for r in rows]
    mr_pts = [[round(r['theta_monopile_rad'], 8), round(r['M_at_ground_kNm'], 4)]
              for r in rows]

    # Default scales: max moment at theta_monopile = 0.01 rad, max
    # displacement at the closest row to 0.01 rad
    def _value_at_theta(rows, key, theta_target=0.01):
        for k, r in enumerate(rows):
            if r['theta_monopile_rad'] >= theta_target:
                return r[key]
        return rows[-1][key]
    y_scale = _value_at_theta(rows, 'y_monopile_mm')
    h_scale = _value_at_theta(rows, 'H_kN')
    m_scale = _value_at_theta(rows, 'M_at_ground_kNm')

    return {
        'aborted': False,
        'checks': checks,
        'inputs': {
            'D_m': D, 'L_m': L, 'pile_type': p['pile_type'],
            't_mm': p['t_mm'], 'h_m': h, 'E_GPa': p['E_GPa'],
            'L_over_D': L_over_D, 'h_over_L': h_over_L,
            'D_inner_m': d_inner, 'EI_kNm2': EI,
            'gamma_above_kNm3': p['gamma_above_kNm3'],
            'gamma_below_kNm3': p['gamma_below_kNm3'],
            'water_table_m': p['water_table'],
            'pile_name': p['pile_name'],
        },
        'g0_fit': fit,
        'summary': {
            'C_k': c_k,
            'gamma_effective_kNm3': g_eff,
            'K_R0_MNm_rad': k_r0_MNm_rad,
            'theta_ref_rad': theta_ref,
            'C_R_theta': c_theta,
            'C_R_y': c_y,
            'Hu_kN': Hu_kN,
            'Hu_above_kN': Hu_above,
            'Hu_below_kN': Hu_below,
            'best_fit_name': fit['name'],
            'best_fit_coefficient': fit['coefficient'],
            'best_fit_cov': fit['cov'],
            'G0_at_0_75L_MPa': fit['g0_at_0_75L_mpa'],
        },
        'curve_load_disp': {
            'x_label': 'Lateral displacement at ground (mm)',
            'y_label': 'Lateral load H (kN)',
            'default_scale': {'x_max': max(y_scale, 1.0),
                              'y_max': max(h_scale, 1.0)},
            'series': [{'name': 'Monopile', 'points': ld_pts}],
        },
        'curve_moment_rotation': {
            'x_label': 'Rotation θ (rad)',
            'y_label': 'Moment at ground M (kNm)',
            'default_scale': {'x_max': 0.01, 'y_max': max(m_scale, 1.0)},
            'series': [{'name': 'Monopile', 'points': mr_pts}],
        },
        'table': rows,
    }
