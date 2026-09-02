"""Regression test for the Clay-form G4/G5 ceilings on the clay A factors.

rev2 of Footing-settlement-forcoding.xlsx (Barry Lehane, 1-Sep-2026) changed
Clay-form G4/G5 from a bare product to

    G4 = MIN($G$10*$G$8*$G$9, 2.79)     immediate
    G5 = MIN($G$11*$G$8*$G$9, 1.97)     long term

CF1 * CF2 grows without limit as Ic rises, so before the ceiling a soft clay
could be handed an A of 6 or more and the footing looked far stiffer than it
is. Run with:  python test_shallow_clay_cap.py
"""
import os

os.environ.setdefault('SECRET_KEY', 't')

from app.calculations import pre_input_calc
from app.shallow_calculations import (
    calculate_shallow_footing_results,
    _cf_clay,
    _geometric_sb,
    A_IMMEDIATE_MAX,
    A_LONGTERM_MAX,
)

PARAMS = {
    'footing_width': 2.0, 'footing_length': 4.0, 'founding_depth': 1.0,
    'excavation_depth': 0.0, 'water_table': 2.0, 'design_life_years': 50,
    'initial_ocr': 1, 'k0': 0.5, 'ageing_factor': 0.66, 'creep': 0.02,
    'nkt': 15, 't_initial_days': 0.05,
}


def _uniform_cpt(qc_mpa, fs_kpa, gtot=18.0, zmax=12.0, dz=0.1):
    n = int(round(zmax / dz))
    return [{'z': round((i + 1) * dz, 3), 'qc': qc_mpa, 'fs': fs_kpa, 'gtot': gtot}
            for i in range(n)]


def _run(qc_mpa, fs_kpa, **overrides):
    pc = pre_input_calc(_uniform_cpt(qc_mpa, fs_kpa), PARAMS['water_table'])
    p = dict(PARAMS)
    p.update(overrides)
    return calculate_shallow_footing_results(pc, p)


def test_constants_match_the_workbook():
    assert A_IMMEDIATE_MAX == 2.79, A_IMMEDIATE_MAX
    assert A_LONGTERM_MAX == 1.97, A_LONGTERM_MAX


def test_ceilings_are_the_ic_260_ocr3_values():
    """The workbook ceilings are the A values the formulation reaches at the
    top of the silt band (Ic = 2.60) with OCR >= 3. Guards against a typo."""
    cf1, cf2 = _cf_clay(2.60)
    assert abs(0.58 * cf1 * cf2 - A_IMMEDIATE_MAX) < 0.005, 0.58 * cf1 * cf2
    assert abs(0.41 * cf1 * cf2 - A_LONGTERM_MAX) < 0.005, 0.41 * cf1 * cf2


def test_soft_clay_is_held_at_the_ceiling():
    """Ic ~ 2.94: uncapped A would be ~6.2 / ~4.4."""
    r = _run(0.6, 30.0)
    sm = r['summary']
    assert r['soil_decision']['soil_model_used'] == 'clay'
    assert sm['avg_ic'] > 2.60, sm['avg_ic']
    raw_imm = sm['c1_base'] * sm['cf1'] * sm['cf2']
    assert raw_imm > A_IMMEDIATE_MAX, raw_imm
    assert sm['a_immediate'] == A_IMMEDIATE_MAX, sm['a_immediate']
    assert sm['a_longterm'] == A_LONGTERM_MAX, sm['a_longterm']
    assert sm['a_immediate_at_ceiling'] is True
    assert sm['a_longterm_at_ceiling'] is True


def test_stiff_clay_below_the_ceiling_is_untouched():
    """Ic ~ 2.50: still inside the silt band ramp, so no clipping."""
    r = _run(2.5, 110.0)
    sm = r['summary']
    assert r['soil_decision']['soil_model_used'] == 'clay'
    raw_imm = sm['c1_base'] * sm['cf1'] * sm['cf2']
    raw_lt = sm['c2_base'] * sm['cf1'] * sm['cf2']
    assert raw_imm < A_IMMEDIATE_MAX, raw_imm
    assert abs(sm['a_immediate'] - raw_imm) < 1e-9
    assert abs(sm['a_longterm'] - raw_lt) < 1e-9
    assert sm['a_immediate_at_ceiling'] is False
    assert sm['a_longterm_at_ceiling'] is False


def test_only_one_ceiling_can_bind():
    """The two ceilings bind at different Ic, so the flags must be independent.
    On the OCR < 3 branch the immediate one binds first (Ic 2.69 vs 2.72)."""
    r = _run(0.70, 2.0, founding_depth=6.0)
    sm = r['summary']
    assert r['soil_decision']['soil_model_used'] == 'clay'
    assert sm['avg_ocr'] < 3, sm['avg_ocr']
    raw_imm = sm['c1_base'] * sm['cf1'] * sm['cf2']
    raw_lt = sm['c2_base'] * sm['cf1'] * sm['cf2']
    assert raw_imm > A_IMMEDIATE_MAX, raw_imm
    assert raw_lt < A_LONGTERM_MAX, raw_lt
    assert sm['a_immediate'] == A_IMMEDIATE_MAX
    assert abs(sm['a_longterm'] - raw_lt) < 1e-9
    assert sm['a_immediate_at_ceiling'] is True
    assert sm['a_longterm_at_ceiling'] is False


def test_curve_respects_the_capped_a():
    """The load-settlement curve is built from the capped A, not the raw one."""
    r = _run(0.6, 30.0)
    sm = r['summary']
    qt_net, cap = sm['qt_net_kpa'], sm['ultimate_curve_cap_kpa']
    beq = (PARAMS['footing_width'] * PARAMS['footing_length']) ** 0.5
    sb_series = _geometric_sb(beq=beq)
    for name, a in (('Immediate', sm['a_immediate']), ('Long term', sm['a_longterm'])):
        series = next(s for s in r['curve']['series'] if s['name'] == name)
        assert len(series['points']) == len(sb_series), name
        for sb, (s_mm, q) in zip(sb_series, series['points']):
            assert abs(s_mm - sb * beq * 1000) < 1e-3, (name, s_mm)
            expected = round(min(a * qt_net * sb ** 0.5, cap), 4)
            assert abs(q - expected) < 1e-9, (name, s_mm, q, expected)


def test_sand_branch_is_unaffected():
    """The ceilings live in the clay branch only."""
    r = _run(12.0, 60.0)
    assert r['soil_decision']['soil_model_used'] == 'sand'
    assert 'a_immediate' not in r['summary']


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
            print('PASS  %s' % name)
        except AssertionError as exc:
            failures += 1
            print('FAIL  %s: %s' % (name, exc))
    print('\n%s' % ('all clay-cap tests passed' if not failures
                    else '%d test(s) failed' % failures))
    raise SystemExit(1 if failures else 0)
