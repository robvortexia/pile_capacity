"""Flexible CPT file importer (demo feature, gated behind the ?code= demo link).

Turns several real-world CPT file formats into the standard row list
``[{'z', 'qc', 'fs', 'gtot'}]`` that ``process_cpt_data`` / ``pre_input_calc``
consume:

  * clean numeric tables: the existing depth, qt(MPa), fs(kPa), unit-weight
    (kN/m3) four-column layout;
  * vendor exports with a metadata header and a column-heading row (e.g. the
    Probedrill .txt files), where columns are mapped by heading keyword rather
    than fixed position;
  * AGS4 .ags files (the SCPT "Static Cone Penetration Tests - Data" group).

Downstream unit convention (what the calculator expects): z in m, qc(=qt) in
MPa, fs in kPa, gtot in kN/m3. When a file has no unit-weight column, gtot is
derived per depth from the CPT via Robertson & Cabal (2010).
"""
import csv
import io
import math

# Robertson & Cabal (2010) unit-weight correlation constants.
PA_KPA = 100.0       # atmospheric pressure (kPa)
GAMMA_W = 9.81       # unit weight of water (kN/m3)
GAMMA_MIN = 11.0     # clamp: practical lower bound for any soil (per Barry Lehane;
                     # the correlation can return lower values at shallow depths)
GAMMA_MAX = 22.5     # clamp: practical upper bound for soils


def unit_weight_from_cpt(qt_mpa, fs_kpa):
    """Total unit weight (kN/m3) from CPT, Robertson & Cabal (2010) Eq. 2.

        gamma = gamma_w * (0.27*log10(Rf) + 0.36*log10(qt/pa) + 1.236)

    Rf = fs/qt * 100 (percent, fs and qt in the same units); pa = 100 kPa.
    Rf and qt/pa are floored to stay inside the log domain, and the result is
    clamped to [9.81, 22.5] kN/m3 (the correlation's sensible range).
    """
    qt_kpa = (qt_mpa or 0.0) * 1000.0
    if qt_kpa <= 0 or fs_kpa is None or fs_kpa < 0:
        return 18.0  # neutral fallback near the ground surface / bad reading
    rf = max((fs_kpa / qt_kpa) * 100.0, 0.1)
    qn = max(qt_kpa / PA_KPA, 0.5)
    g = GAMMA_W * (0.27 * math.log10(rf) + 0.36 * math.log10(qn) + 1.236)
    return round(min(max(g, GAMMA_MIN), GAMMA_MAX), 2)


def parse_cpt_file(content, filename=''):
    """Parse a CPT file into (rows, note).

    rows = list of {'z', 'qc', 'fs', 'gtot'}; note = human-readable summary of
    what was detected (flashed to the user). Raises ValueError on a file with
    no usable CPT data.
    """
    text = _decode(content)
    if _looks_like_ags4(text, filename):
        return _parse_ags4(text)
    return _parse_tabular(text)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _decode(content):
    if isinstance(content, bytes):
        for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
            try:
                content = content.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = content.decode('utf-8', 'ignore')
    # Normalise line endings (files seen with CRLF and bare/doubled CR).
    return content.replace('\r\n', '\n').replace('\r', '\n')


def _to_float(tok):
    """Parse a token to a finite float, or None (also rejects NaN/inf)."""
    try:
        f = float(str(tok).strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# AGS4 (.ags) — SCPT group
# ---------------------------------------------------------------------------
def _looks_like_ags4(text, filename=''):
    if filename.lower().endswith('.ags'):
        return True
    head = text[:4000]
    return '"GROUP"' in head and '"HEADING"' in head


def _parse_ags4(text):
    group, headings, units = None, [], []
    soundings, order = {}, []     # group rows by (LOCA_ID, SCPG_TESN)
    unit_of = {}
    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        tag = rec[0].strip().upper()
        if tag == 'GROUP':
            group = rec[1].strip().upper() if len(rec) > 1 else None
            headings, units = [], []
        elif group == 'SCPT' and tag == 'HEADING':
            headings = [h.strip().upper() for h in rec[1:]]
        elif group == 'SCPT' and tag == 'UNIT':
            units = [u.strip() for u in rec[1:]]
            unit_of = {headings[i]: (units[i] if i < len(units) else '')
                       for i in range(len(headings))}
        elif group == 'SCPT' and tag == 'DATA':
            vals = rec[1:]
            d = {headings[i]: (vals[i] if i < len(vals) else '')
                 for i in range(len(headings))}
            key = (d.get('LOCA_ID', ''), d.get('SCPG_TESN', ''))
            if key not in soundings:
                soundings[key] = []
                order.append(key)
            soundings[key].append(d)
    if not order:
        raise ValueError('No SCPT (static CPT) data group found in the AGS4 file.')

    def conv(code, raw, target):
        """Convert an AGS value to the target unit (MPa or kPa), reading the
        UNIT row so we never hardcode (AGS dictionary default for qc/qt/fs is
        MPa, but some files export kPa)."""
        f = _to_float(raw)
        if f is None:
            return None
        u = (unit_of.get(code, '') or '').lower()
        if target == 'MPa':
            return f / 1000.0 if u == 'kpa' else f
        # target kPa: AGS fs/qc default is MPa, so scale up unless already kPa
        return f if u == 'kpa' else f * 1000.0

    first = soundings[order[0]]
    rows, used_qt = [], False
    for d in first:
        z = _to_float(d.get('SCPT_DPTH'))
        qt = conv('SCPT_QT', d.get('SCPT_QT'), 'MPa')
        qc = conv('SCPT_RES', d.get('SCPT_RES'), 'MPa')
        q = qt if (qt is not None and qt > 0) else qc
        fs = conv('SCPT_FRES', d.get('SCPT_FRES'), 'kPa')
        if z is None or q is None or q <= 0 or fs is None or fs < 0:
            continue
        if qt is not None and qt > 0:
            used_qt = True
        rows.append({'z': z, 'qc': q, 'fs': fs, 'gtot': unit_weight_from_cpt(q, fs)})
    if not rows:
        raise ValueError('AGS4 SCPT group found but no usable depth/qc/fs rows.')

    note = 'Read AGS4 SCPT group (%d readings' % len(rows)
    if len(order) > 1:
        note += '; file had %d soundings, used the first (%s)' % (len(order), order[0][0] or 'first')
    note += '). '
    if not used_qt:
        note += 'Used cone resistance qc as qt (no corrected qt in file). '
    note += 'Unit weight derived from the CPT via Robertson & Cabal (2010).'
    return rows, note


# ---------------------------------------------------------------------------
# Tabular: clean CSV + vendor exports with a heading row
# ---------------------------------------------------------------------------
_GAMMA_KW = ('unit weight', 'unit wt', 'gamma', 'density', 'gtot')


def _detect_delim(lines):
    for ch in ('\t', ',', ';'):
        if any(ch in ln for ln in lines):
            return ch
    return None  # whitespace


def _split(ln, delim):
    return ln.split() if delim is None else [c.strip() for c in ln.split(delim)]


def _map_headings(heading):
    """Map a lower-cased heading row to column indices for z/qt/qc/fs/gamma."""
    col = {'z': None, 'qt': None, 'qc': None, 'fs': None, 'gamma': None}
    for idx, h in enumerate(heading):
        is_ratio = 'ratio' in h
        if col['z'] is None and 'depth' in h:
            col['z'] = idx
        elif col['gamma'] is None and any(k in h for k in _GAMMA_KW):
            col['gamma'] = idx
        elif col['qt'] is None and 'corrected' in h and ('q' in h or 'cone' in h):
            col['qt'] = idx
        elif (col['fs'] is None and not is_ratio and
              ('sleeve' in h or ('friction' in h and 'res' in h) or
               ('local' in h and 'friction' in h) or h.strip() in ('fs', 'f_s'))):
            col['fs'] = idx
        elif col['qc'] is None and (('cone' in h and 'res' in h) or
                                    ('tip' in h and 'res' in h) or h.strip() in ('qc', 'q_c')):
            col['qc'] = idx
    return col


def _parse_tabular(text):
    nonempty = [ln for ln in text.split('\n') if ln.strip()]
    if not nonempty:
        raise ValueError('The file is empty.')
    delim = _detect_delim(nonempty)

    def is_data(ln):
        toks = _split(ln, delim)
        return len(toks) >= 3 and _to_float(toks[0]) is not None

    data_idx = [i for i, ln in enumerate(nonempty) if is_data(ln)]
    if not data_idx:
        raise ValueError('No numeric CPT data rows found in the file.')
    first = data_idx[0]

    def clean_toks(ln):
        toks = _split(ln, delim)
        while toks and toks[-1] == '':   # drop trailing empties (trailing delimiter)
            toks.pop()
        return toks

    ncol = len(clean_toks(nonempty[first]))

    # Heading row: the nearest preceding line that names the columns. Prefer a
    # line containing a "depth" token (robust to trailing delimiters / column
    # count drift); else a same-width, mostly-text line (e.g. the Probedrill
    # "depth log / cone resistance / ..." row).
    heading = None
    for j in range(first - 1, -1, -1):
        toks = clean_toks(nonempty[j])
        if not toks:
            continue
        n_alpha = sum(1 for t in toks if _to_float(t) is None and any(c.isalpha() for c in t))
        has_depth = any('depth' in t.lower() for t in toks)
        if (has_depth and n_alpha >= 2) or (len(toks) == ncol and n_alpha >= max(2, ncol // 2)):
            heading = [t.lower() for t in toks]
            break

    col = _map_headings(heading) if heading else {k: None for k in ('z', 'qt', 'qc', 'fs', 'gamma')}
    mapped_by_heading = bool(heading) and col['z'] is not None and col['fs'] is not None and (
        col['qc'] is not None or col['qt'] is not None)
    if not mapped_by_heading:
        # Positional fallback: clean layout depth, qt, fs, [unit weight].
        col = {'z': 0, 'qt': None, 'qc': 1, 'fs': 2, 'gamma': (3 if ncol >= 4 else None)}

    rows, used_qt, derived_gamma = [], False, False
    for i in data_idx:
        toks = _split(nonempty[i], delim)

        def get(c):
            return _to_float(toks[c]) if (c is not None and c < len(toks)) else None

        z = get(col['z'])
        qt = get(col['qt'])
        qc = get(col['qc'])
        q = qt if (qt is not None and qt > 0) else qc
        fs = get(col['fs'])
        if z is None or q is None or q <= 0 or fs is None or fs < 0:
            continue
        if qt is not None and qt > 0:
            used_qt = True
        gam = get(col['gamma'])
        if gam is None or gam <= 0:
            gam = unit_weight_from_cpt(q, fs)
            derived_gamma = True
        rows.append({'z': z, 'qc': q, 'fs': fs, 'gtot': gam})
    if not rows:
        raise ValueError('Found a data table but could not read depth/qc/fs values from it.')

    if mapped_by_heading:
        note = 'Read a contractor CPT export and mapped columns by heading (depth, cone resistance, friction). '
        if col['qt'] is not None and not used_qt:
            note += 'Used cone resistance qc as qt (corrected qt column was empty). '
    else:
        note = 'Read a plain numeric table (depth, qt, fs%s). ' % (', unit weight' if col['gamma'] is not None else '')
    if derived_gamma:
        note += 'Unit weight not in the file; derived per depth from the CPT via Robertson & Cabal (2010).'
    return rows, note.strip()
