"""
Embedded cantilever-wall deflection (Step 4) — predicts the maximum wall
deflection and ground-settlement profile behind a rigid sheet-piled
cantilever wall in sand, from CPT data.

Follows the workbook ``Cantilever-wall deflection-forcoding.xlsx``
(Barry Lehane, 28-May-2026). The method:

  1. Average peak friction angle phi'_p and the gradient dqc/dz are
     computed over a depth band 1.5*H below ground (clamped by H/L).
     phi'_p uses the workbook's 32 + 3*(5*Dr - 1) relation with Dr from
     the standard ageing/qc/p' formula.
     dqc/dz is the slope of a least-squares fit through the origin of
     qc against depth in the same band (the workbook used Excel Solver
     here; the closed-form least-squares-through-origin solution is
     used here for parity).

  2. Hard checks:
       - Average Ic in the depth band must be below 2.2 (sand).
       - Water-table depth must be at least 2*H (programme not suitable
         otherwise).
       - dqc/dz must be > 0 (qc must not reduce with depth).
       - EI/(gamma_w * H^4) must fall in [5, 4500].
       - H/L must fall in [0.2, 0.671].
       - CPT data must reach the base of the wall.

  3. Maximum normalised wall deflection d/H is read from a stored 4D
     lookup table (sheet "Interpolation" columns A..DG, rows 1..46)
     indexed by:
         phi'_p (deg)          : table at 35, 43 (clamp + linear interp/extrap)
         dqc/dz (MN/m^3)       : table at 1.0, 2.5, 6.8 (clamp + linear interp)
         H/L                   : 9 stored values per (phi', dqc/dz) chunk
         EI/(gamma_w * H^4)    : 20 stored normalised values per curve
     Quadrilinear interpolation, replicating the workbook's nested
     four-step lookup (DI..DT columns in the Interpolation sheet).

  4. Maximum settlement behind the wall:
         smax = 0.6 * dmax if H/L > 0.5 else 0.5 * dmax
         s(x = H/2) = smax * exp(-0.5)   (round to 2 dp)

The workbook fixes K0 = 0.7, sand ageing factor 0.666, and
phi'_cv = 32 deg. These are exposed as constants here.
"""
import math
from ._cantilever_lookup import LOOKUP


# ---- workbook constants (Test Calculations C4, C5, C7) -------------------
K0_FIXED = 0.7
AGEING_FACTOR = 0.666
PHI_CV_DEG = 32.0
GAMMA_W_KNM3 = 10.0      # used in EI/(gamma_w * H^4)
P_REF_KPA = 100.0

# Table support bounds (Test Calculations K3, K4, K5; Interpolation DP/DM)
PHI_MIN, PHI_MAX = 35.0, 50.0          # workbook clamp on phi'_p
DQC_MIN, DQC_MAX = 1.0, 6.8
HL_MIN, HL_MAX = 0.2, 0.67
EI_MIN, EI_MAX = 5.0, 4500.0
IC_SAND_MAX = 2.2

# Stored axis values (sorted)
PHI_VALUES_STORED = sorted(LOOKUP.keys())                  # [35, 43]
DQC_VALUES_STORED = sorted({d for f in LOOKUP for d in LOOKUP[f]})   # [1.0, 2.5, 6.8]


# ---------------------------------------------------------------------------
# Per-row table (Test Calculations rows 28+) — CPT-derived columns
# ---------------------------------------------------------------------------
def _per_row(depth, qt_mpa, sig_v0_prime_kpa, ic):
    """For each CPT row, compute Dr and the peak friction angle phi'_p.

    Mirrors Test Calculations columns J (Dr) and O (phi'_p). Uses the
    fixed K0 and ageing factor.

    Note vs. workbook: Test Calculations O28 in the rev1 file uses
    32 + 3*(5*Dr - 1). Barry confirmed (29 May email) that the multiplier
    should be 4.7 not 5 (matching AC45 in the shallow / monopile
    workbooks). The corrected value is used here, so the printed phi'_p
    values can differ from Test Calculations by ~0.03 deg on the
    average for typical sands.
    """
    n = len(depth)
    dr = [0.0] * n
    phi_p = [0.0] * n
    for i in range(n):
        svp = sig_v0_prime_kpa[i]
        sigh0 = K0_FIXED * svp                     # H col
        p_kpa = (svp + 2 * sigh0) / 3.0            # I col
        if p_kpa > 0 and qt_mpa[i] > 0:
            m = (1.0 / 2.93) * math.log(
                (AGEING_FACTOR * qt_mpa[i] * 1000.0) / (205.0 * p_kpa ** 0.5))
        else:
            m = 0.1
        dr[i] = max(0.1, m)
        phi_p[i] = PHI_CV_DEG + 3.0 * (4.7 * dr[i] - 1.0)   # O col (4.7 per Barry's correction)
    return dr, phi_p


# ---------------------------------------------------------------------------
# Depth band for averaging (Test Calculations C15)
# ---------------------------------------------------------------------------
def _averaging_depth(L, H):
    """Depth limit for averaging phi' and fitting dqc/dz (workbook C15).

    1.5*H, but clamped so that H/L stays in [0.2, 0.67].
    """
    hl = H / L
    if hl > 0.67:
        return 1.5 * L * 0.67
    if hl < 0.2:
        return 1.5 * L * 0.2
    return 1.5 * H


def _band_indices(depth, top_z, bottom_z):
    """Indices of rows whose depth is within [top_z, bottom_z]."""
    return [i for i, z in enumerate(depth) if top_z <= z <= bottom_z]


# ---------------------------------------------------------------------------
# dqc/dz from a least-squares fit through the origin (Test Calculations C20)
# ---------------------------------------------------------------------------
def _dqc_dz_through_origin(depth, qc_mpa, idx):
    """Best dqc/dz minimising sum(qc - dqc/dz * z)^2 with no intercept.

    The workbook used Excel Solver on cell C20 to minimise C21 (sum of
    squared residuals) with the fit qc = (dqc/dz) * z. The closed form
    is:  dqc/dz = sum(qc_i * z_i) / sum(z_i^2)
    over the rows i in the averaging band.
    """
    num = sum(qc_mpa[i] * depth[i] for i in idx)
    den = sum(depth[i] ** 2 for i in idx)
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Quadrilinear interpolation against the stored table
# ---------------------------------------------------------------------------
def _interp_on_ei(curve, ei_q):
    """Linear interpolation on the EI axis for one (f', dqc/dz, H/L) curve.

    ``curve`` is a list of (EI, d/H) sorted by EI ascending. ``ei_q`` is
    expected to be in the supported range (clamp before calling).
    """
    pts = curve
    if ei_q <= pts[0][0]:
        return pts[0][1]
    if ei_q >= pts[-1][0]:
        return pts[-1][1]
    for k in range(len(pts) - 1):
        x0, y0 = pts[k]
        x1, y1 = pts[k + 1]
        if x0 <= ei_q <= x1:
            return y0 + (ei_q - x0) * (y1 - y0) / (x1 - x0)
    return pts[-1][1]


def _interp_one_chunk(chunk, hl_q, ei_q):
    """Interpolate within a single (f', dqc/dz) chunk to user's (H/L, EI).

    ``chunk`` is a list of {hl, curve=[(EI, d/H), ...]} sorted by H/L
    DESCENDING. The workbook's Step 2 (DS column) brackets the user's
    H/L between two consecutive stored H/L values and linearly
    interpolates the Step-1 d/H values.
    """
    # H/L is descending in the chunk; clamp at the edges
    hls = [c['hl'] for c in chunk]
    if hl_q >= hls[0]:
        return _interp_on_ei(chunk[0]['curve'], ei_q)
    if hl_q <= hls[-1]:
        return _interp_on_ei(chunk[-1]['curve'], ei_q)
    for k in range(len(chunk) - 1):
        hl_hi = chunk[k]['hl']      # higher (descending order)
        hl_lo = chunk[k + 1]['hl']
        if hl_lo <= hl_q <= hl_hi:
            dh_hi = _interp_on_ei(chunk[k]['curve'], ei_q)
            dh_lo = _interp_on_ei(chunk[k + 1]['curve'], ei_q)
            # Linear in H/L between hl_hi and hl_lo:
            #   dH = dh_hi - (dh_hi - dh_lo) * (hl_hi - hl_q) / (hl_hi - hl_lo)
            return dh_hi - (dh_hi - dh_lo) * (hl_hi - hl_q) / (hl_hi - hl_lo)
    return _interp_on_ei(chunk[-1]['curve'], ei_q)


def _interp_on_dqc(per_dqc, dqc_q, hl_q, ei_q):
    """For one f', interpolate across dqc/dz given the per-dqc chunks."""
    dqcs = sorted(per_dqc.keys())
    if dqc_q <= dqcs[0]:
        return _interp_one_chunk(per_dqc[dqcs[0]], hl_q, ei_q)
    if dqc_q >= dqcs[-1]:
        return _interp_one_chunk(per_dqc[dqcs[-1]], hl_q, ei_q)
    for k in range(len(dqcs) - 1):
        a, b = dqcs[k], dqcs[k + 1]
        if a <= dqc_q <= b:
            ya = _interp_one_chunk(per_dqc[a], hl_q, ei_q)
            yb = _interp_one_chunk(per_dqc[b], hl_q, ei_q)
            return ya + (dqc_q - a) * (yb - ya) / (b - a)
    return _interp_one_chunk(per_dqc[dqcs[-1]], hl_q, ei_q)


def _interp_d_over_h(phi_q, dqc_q, hl_q, ei_q):
    """Top-level quadrilinear interpolation against the stored table.

    All four inputs MUST be clamped to the table's supported ranges
    before being passed in. f' is interpolated/extrapolated linearly
    between the two stored values (35 and 43); the workbook does this
    in column DT112+ with the same closed form.
    """
    phis = PHI_VALUES_STORED
    if phi_q <= phis[0]:
        return _interp_on_dqc(LOOKUP[phis[0]], dqc_q, hl_q, ei_q)
    # Note: workbook extrapolates linearly above 43 if the clamped phi
    # exceeds the stored values. We mirror that by always interpolating
    # / extrapolating between the two stored values.
    a, b = phis[0], phis[-1]
    ya = _interp_on_dqc(LOOKUP[a], dqc_q, hl_q, ei_q)
    yb = _interp_on_dqc(LOOKUP[b], dqc_q, hl_q, ei_q)
    return ya + (phi_q - a) * (yb - ya) / (b - a)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _read_params(params):
    return {
        'water_table': float(params.get('water_table', 0)),
        'L': float(params['wall_length']),                  # C9
        'H': float(params['excavation_depth']),             # C10
        'EI': float(params['EI_kNm2_per_m']),               # C11
        'wall_name': params.get('wall_name', ''),
    }


def calculate_cantilever_results(processed_cpt, params):
    """Step 4 for embedded cantilever wall.

    Args:
        processed_cpt: dict from ``pre_input_calc`` (depth, qt, lc=Ic,
            sig_v0, sig_v0_prime, u0_kpa, gtot).
        params: wall geometry + EI (see ``_read_params``).

    Returns:
        dict with ``checks``, ``summary``, and (if not aborted) the
        derived d/H, dmax (mm), smax (mm), s at x=H/2 (mm). If the
        analysis is aborted, ``aborted: True`` and a structured message.
    """
    print(f"Params received in calculate_cantilever_results: {params}")
    p = _read_params(params)
    depth = list(processed_cpt['depth'])
    qt_mpa = list(processed_cpt['qt'])
    ic = list(processed_cpt['lc'])
    sig_v0_prime = list(processed_cpt['sig_v0_prime'])
    if not depth:
        raise ValueError('No CPT data supplied to cantilever wall calculation.')

    L, H, EI = p['L'], p['H'], p['EI']
    dw = p['water_table']
    max_cpt = max(depth)
    hl = H / L

    # ---- hard checks gathered up front so we can return ALL applicable msgs
    checks = {
        'wall_length_m': L, 'excavation_depth_m': H, 'EI_kNm2_per_m': EI,
        'water_table_m': dw, 'max_cpt_depth_m': max_cpt, 'hl_ratio': hl,
    }
    fatal = []
    if dw < 2 * H:
        fatal.append('Program not suitable if the depth to the water table is '
                     'less than double the excavation depth.')
    if L > max_cpt:
        fatal.append('Base of wall is below the maximum CPT data depth.')
    if hl < 0.2:
        fatal.append('H/L is very small (< 0.2) and outside the range of '
                     'applicability of the program.')
    if hl > 0.671:
        fatal.append('H/L is excessive (> 0.671) and outside the range of '
                     'applicability of the program.')
    ei_norm_raw = EI / (GAMMA_W_KNM3 * H ** 4)
    if not (EI_MIN <= ei_norm_raw <= EI_MAX):
        fatal.append('EI / (gamma_w * H^4) = %.2f is outside the supported '
                     'range [%.0f, %.0f].' % (ei_norm_raw, EI_MIN, EI_MAX))

    # ---- averaging band + Ic sand check + dqc/dz fit ----
    # Workbook quirk: phi'p / Ic averages skip the very first CPT row
    # (Test Calculations C16 = 29, hard-coded; row 28 is the first data
    # row, row 29 is the second). The dqc/dz sum-of-squares starts at
    # the first row (Q28 = (B28-P28)^2). We match both bands here.
    z_top = depth[0]
    z_band = _averaging_depth(L, H)
    band_full = _band_indices(depth, z_top, z_band)
    # Skip the very first depth (the workbook starts averaging one row in)
    band_phi_ic = band_full[1:] if len(band_full) > 1 else band_full
    if not band_full:
        fatal.append('No CPT rows fall in the averaging band 0..%.2f m.' % z_band)

    if band_full:
        ic_avg = (sum(ic[i] for i in band_phi_ic) / len(band_phi_ic)
                  if band_phi_ic else 0.0)
        if ic_avg > IC_SAND_MAX:
            fatal.append('Average Ic in the averaging band is %.2f (> %.1f). '
                         'Soil profile is not sand; program is not applicable.'
                         % (ic_avg, IC_SAND_MAX))
        dr, phi_p = _per_row(depth, qt_mpa, sig_v0_prime, ic)
        phi_avg = (sum(phi_p[i] for i in band_phi_ic) / len(band_phi_ic)
                   if band_phi_ic else 0.0)
        dqc_dz = _dqc_dz_through_origin(depth, qt_mpa, band_full)
        if dqc_dz <= 0:
            fatal.append('Program is not suitable for cases where qc reduces '
                         'with depth (dqc/dz <= 0).')
    else:
        ic_avg = phi_avg = dqc_dz = 0.0

    checks['avg_ic_in_band'] = ic_avg
    checks['avg_phi_prime_deg'] = phi_avg
    checks['dqc_dz_MN_m3'] = dqc_dz

    if fatal:
        return {
            'aborted': True,
            'checks': checks,
            'message': ' '.join(fatal),
            'fatal': fatal,
        }

    # ---- clamped inputs for the lookup ----
    # The lookup (Interpolation DJ51) uses the *unrounded* phi'p; the K3
    # column on the Test Calculations sheet rounds to 1 dp for display only.
    phi_clamped = max(PHI_MIN, min(phi_avg, PHI_MAX))
    phi_for_display = max(PHI_MIN, min(round(phi_avg, 1), PHI_MAX))
    dqc_clamped = max(DQC_MIN, min(dqc_dz, DQC_MAX))
    hl_clamped = max(HL_MIN, min(hl, HL_MAX))
    ei_norm = EI / (GAMMA_W_KNM3 * ((L * hl_clamped) ** 4))   # uses CLAMPED H
    ei_clamped = max(EI_MIN, min(ei_norm, EI_MAX))

    # ---- table lookup ----
    d_over_h = _interp_d_over_h(phi_clamped, dqc_clamped, hl_clamped, ei_clamped)
    # Effective H used in dimensionalisation is the clamped H
    H_effective = hl_clamped * L
    dmax_mm = d_over_h * H_effective * 1000.0

    # ---- settlement model ----
    smax_mm = 0.6 * dmax_mm if hl_clamped > 0.5 else 0.5 * dmax_mm
    s_h_half_mm = round(smax_mm * math.exp(-0.5), 2)

    # ---- settlement profile vs distance behind the wall (x = 0..3H) ----
    # Workbook's text mentions "smax * exp(-x/H * decay)" implicitly via the
    # x=H/2 spec. The same exponential decay is plotted vs x/H.
    decay_const = -math.log(s_h_half_mm / smax_mm) / 0.5 if smax_mm > 0 else 1.0
    n_pts = 60
    profile = []
    for k in range(n_pts + 1):
        x = (3.0 * H) * (k / n_pts)
        s = smax_mm * math.exp(-decay_const * (x / H))
        profile.append([round(x, 4), round(s, 4)])

    return {
        'aborted': False,
        'checks': checks,
        'inputs': {
            'wall_length_m': L,
            'excavation_depth_m': H,
            'EI_kNm2_per_m': EI,
            'water_table_m': dw,
            'wall_name': p['wall_name'],
            'hl_ratio_clamped': hl_clamped,
            'phi_avg_deg_clamped': phi_clamped,
            'phi_avg_deg_display': phi_for_display,
            'dqc_dz_MN_m3_clamped': dqc_clamped,
            'ei_normalised_clamped': ei_clamped,
        },
        'summary': {
            'd_over_H': d_over_h,
            'dmax_mm': dmax_mm,
            'smax_mm': smax_mm,
            's_at_x_H_over_2_mm': s_h_half_mm,
            'avg_phi_prime_deg': phi_avg,
            'avg_phi_prime_deg_clamped': phi_clamped,
            'dqc_dz_MN_m3': dqc_dz,
            'dqc_dz_MN_m3_clamped': dqc_clamped,
            'hl_ratio': hl,
            'ei_normalised_raw': ei_norm_raw,
            'ei_normalised_used': ei_clamped,
            'H_effective_m': H_effective,
        },
        'settlement_profile': {
            'x_label': 'Distance behind wall, x (m)',
            'y_label': 'Settlement (mm)',
            'default_scale': {'x_max': 2.0 * H, 'y_max': max(smax_mm * 1.05, 1.0)},
            'series': [{'name': 'Settlement', 'points': profile}],
        },
    }
