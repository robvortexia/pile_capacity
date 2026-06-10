"""Flexible CPT file importer (demo feature, gated behind the ?code= demo link).

Turns several real-world CPT file formats into the standard row list
``[{'z', 'qc', 'fs', 'gtot'}]`` that ``process_cpt_data`` / ``pre_input_calc``
consume:

  * clean numeric tables: the existing depth, qt(MPa), fs(kPa), unit-weight
    (kN/m3) four-column layout;
  * vendor exports with a metadata header and a column-heading row (e.g. the
    Probedrill .txt files), where columns are mapped by heading keyword rather
    than fixed position;
  * vendor exports where the header is split over two rows, names above units
    (e.g. the CPT South Australia .txt files: "Depth / Tip / Sleeve / ..."
    over "(m) / qc (MPa) / fs (kPa) / ..."): the two rows are merged cell-wise
    before mapping;
  * AGS4 .ags files (the SCPT "Static Cone Penetration Tests - Data" group).

Downstream unit convention (what the calculator expects): z in m, qc(=qt) in
MPa, fs in kPa, gtot in kN/m3. When a file has no unit-weight column, gtot is
derived per depth from the CPT via Robertson & Cabal (2010).
"""
import csv
import io
import math
import re

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
    """Map a lower-cased heading row to column indices for z/qt/qc/fs/gamma.

    Symbols (qc, qt, fs) are matched as whole words within the cell, so
    "qc (MPa)" or a merged "tip qc (mpa)" map, while "inclination" never
    trips the "fs" test. Ratio-like columns are excluded from qt/qc/fs/gamma:
    "friction ratio", Rf, Bq, percentages, and formula headings such as
    "fs/qc (%)" or "Bq = u2/qt (-)" (a slash between two channel symbols;
    slashes in unit text like "kN/m2" or "kN/m3" do not count). A cell that
    only hints at depth via a standalone "z" word can be overridden by a
    later cell that says "depth" outright (so "Zone (Z)" cannot steal the
    depth column).
    """
    col = {'z': None, 'qt': None, 'qc': None, 'fs': None, 'gamma': None}
    z_weak = False
    for idx, h in enumerate(heading):
        words = set(re.findall(r'[a-z][a-z0-9_]*', h))
        is_ratio = ('ratio' in h or '%' in h or 'rf' in words or 'bq' in words or
                    re.search(r'\b(?:fs|qs|qc|qt|u[012]?)\s*/\s*(?:fs|qs|qc|qt|u[012]?)\b', h)
                    is not None)
        if 'depth' in h and (col['z'] is None or z_weak):
            col['z'], z_weak = idx, False
        elif col['z'] is None and 'z' in words:
            col['z'], z_weak = idx, True
        elif col['gamma'] is None and not is_ratio and any(k in h for k in _GAMMA_KW):
            col['gamma'] = idx
        elif col['qt'] is None and not is_ratio and (
                ('corrected' in h and ('q' in h or 'cone' in h)) or
                'qt' in words or 'q_t' in words):
            col['qt'] = idx
        elif (col['fs'] is None and not is_ratio and
              ('sleeve' in h or ('friction' in h and 'res' in h) or
               ('local' in h and 'friction' in h) or 'fs' in words or 'f_s' in words)):
            col['fs'] = idx
        elif col['qc'] is None and not is_ratio and (('cone' in h and 'res' in h) or
                                                     'tip' in h or 'qc' in words or 'q_c' in words):
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
    # "depth log / cone resistance / ..." row). Some vendors split the header
    # over two rows, names above units (CPT South Australia: "Depth / Tip /
    # Sleeve / ..." over "(m) / qc (MPa) / fs (kPa) / ..."); when the row
    # above the nearest candidate is also heading-like, try the cell-wise
    # merge of the two ("depth (m)", "tip qc (mpa)", ...) first.
    def heading_toks(j):
        if j < 0:
            return None
        toks = clean_toks(nonempty[j])
        if not toks:
            return None
        n_alpha = sum(1 for t in toks if _to_float(t) is None and any(c.isalpha() for c in t))
        has_depth = any('depth' in t.lower() for t in toks)
        if (has_depth and n_alpha >= 2) or (len(toks) == ncol and n_alpha >= max(2, ncol // 2)):
            return [t.lower() for t in toks]
        return None

    def maps_fully(c):
        return c['z'] is not None and c['fs'] is not None and (
            c['qc'] is not None or c['qt'] is not None)

    heading, col, z_assumed = None, None, False
    for j in range(first - 1, -1, -1):
        near = heading_toks(j)
        if near is None:
            continue
        candidates, merged = [near], None
        above = heading_toks(j - 1)
        if above is not None:
            candidates = [near, above]
            if delim is not None or len(above) == len(near):
                # Two-row name+unit header: merge cell-wise, padding the
                # shorter row (delimited cells are explicit, so width drift
                # is just trailing extras; whitespace-split rows must match
                # exactly or multi-word names misalign).
                width = max(len(above), len(near))
                merged = [((above[k] if k < len(above) else '') + ' ' +
                           (near[k] if k < len(near) else '')).strip()
                          for k in range(width)]
                candidates = [near, merged, above]

        def defaulted_z(c, allow):
            # Depth defaulting to column 0 (the universal CPT layout) when
            # qc/fs mapped and column 0 is unclaimed: this is how a
            # units-only header row like "(m) / qc (MPa) / fs (kPa)" keeps
            # its column positions and unit info.
            if (c['z'] is None and allow and
                    0 not in (c['qc'], c['qt'], c['fs'], c['gamma'])):
                c['z'] = 0
                return True
            return False

        # Pass 1 wants the depth column named explicitly; pass 2 allows the
        # column-0 default. The nearest row goes first so a stray texty line
        # above a complete header cannot poison the mapping via the merge.
        for allow_default_z in (False, True):
            for cand in candidates:
                if delim is None and len(cand) != ncol:
                    # Whitespace-split headings only align with the data
                    # when token counts match (multi-word names break up).
                    continue
                c = _map_headings(cand)
                z_assumed = defaulted_z(c, allow_default_z)
                if maps_fully(c):
                    heading, col = cand, c
                    break
            if heading is not None:
                break
        # If the nearest row won on its own but the merged header maps to
        # the identical columns, prefer the merge: same mapping, plus the
        # name+unit text for unit scaling.
        if heading is near and merged is not None:
            cm = _map_headings(merged)
            defaulted_z(cm, z_assumed)
            if cm == col:
                heading = merged
        if heading is None:   # nearest texty row, as before -> positional fallback
            heading, col, z_assumed = near, _map_headings(near), False
        break

    if col is None:
        col = {k: None for k in ('z', 'qt', 'qc', 'fs', 'gamma')}
    mapped_by_heading = heading is not None and maps_fully(col)
    if not mapped_by_heading:
        # Positional fallback: clean layout depth, qt, fs, [unit weight].
        col = {'z': 0, 'qt': None, 'qc': 1, 'fs': 2, 'gamma': (3 if ncol >= 4 else None)}
        if col['gamma'] is not None:
            # Trust column 4 as unit weight only if it reads like one: vendor
            # exports often carry friction ratio or pore pressure there, and
            # mistaking those for gamma wrecks the stress profile.
            vals = []
            for i in data_idx:
                toks = _split(nonempty[i], delim)
                v = _to_float(toks[3]) if len(toks) > 3 else None
                if v is not None and v > 0:
                    vals.append(v)
            vals.sort()
            if not vals or not (8.0 <= vals[len(vals) // 2] <= 26.0):
                col['gamma'] = None

    # Unit conversion where the heading states units: downstream expects q in
    # MPa and fs in kPa, but some exports put fs in MPa or q in kPa.
    def heading_scale(key, expect):
        c = col.get(key)
        if not mapped_by_heading or c is None or c >= len(heading):
            return 1.0
        h = heading[c]
        if expect == 'MPa' and 'kpa' in h and 'mpa' not in h:
            return 1e-3
        if expect == 'kPa' and 'mpa' in h and 'kpa' not in h:
            return 1e3
        return 1.0

    qt_scale = heading_scale('qt', 'MPa')
    qc_scale = heading_scale('qc', 'MPa')
    fs_scale = heading_scale('fs', 'kPa')
    converted_units = any(s != 1.0 for s in (qt_scale, qc_scale, fs_scale))

    rows, used_qt, derived_gamma = [], False, False
    for i in data_idx:
        toks = _split(nonempty[i], delim)

        def get(c):
            return _to_float(toks[c]) if (c is not None and c < len(toks)) else None

        z = get(col['z'])
        qt = get(col['qt'])
        qc = get(col['qc'])
        if qt is not None:
            qt *= qt_scale
        if qc is not None:
            qc *= qc_scale
        q = qt if (qt is not None and qt > 0) else qc
        fs = get(col['fs'])
        if fs is not None:
            fs *= fs_scale
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
        if z_assumed:
            note += 'The header did not name a depth column; assumed the first column is depth. '
        if col['qt'] is not None and not used_qt:
            note += 'Used cone resistance qc as qt (corrected qt column was empty). '
        if converted_units:
            note += 'Converted units to MPa/kPa per the file\'s column headings. '
    else:
        note = 'Read a plain numeric table (depth, qt, fs%s). ' % (', unit weight' if col['gamma'] is not None else '')
    if derived_gamma:
        note += 'Unit weight not in the file; derived per depth from the CPT via Robertson & Cabal (2010).'
    return rows, note.strip()
