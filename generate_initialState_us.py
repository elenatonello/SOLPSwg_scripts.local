#!/usr/bin/env python3
"""
generate_initialState_us.py

Minimal self-contained tool to generate an unstructured `b2fstate` by
mapping selected fields (`na`, `ne`, `te`, `ti`, `po`) from a source
`b2fstate` onto an unstructured geometry. Other fields are preserved
from a provided unstructured state template.

The source is a structured b2fgmtry/b2fstate pair by default. With
`--src-unstructured` the source pair is already unstructured
(b2fgmtry_us / b2fstate_us), i.e. the plasma solution is transferred
from one unstructured mesh onto a different unstructured mesh.

Usage:
  generate_initialState_us.py --gmtry-st GMST --gmtry-us GMUS \
    --state-st SSTAT --state-us TEMPLATE_US --out OUT [--src-unstructured]

This script is deliberately small and includes only the readers/writers
needed to parse the text files and perform nearest-neighbour mapping.
"""
from __future__ import annotations
import re
import argparse
import sys
from pathlib import Path
import numpy as np


_thisdir = Path(__file__).resolve().parent
if str(_thisdir) not in sys.path:
    sys.path.insert(0, str(_thisdir))

from b2state_io import B2StateIO


def parse_species(spec, ns):
    """Parse '0,2,4-6' into a sorted list of 0-based species indices."""
    out = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            if '-' in tok:
                a, b = tok.split('-')
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(tok))
        except ValueError:
            sys.exit(f"ERROR: cannot parse species '{tok}' in --species {spec!r}")
    bad = sorted(i for i in out if not 0 <= i < ns)
    if bad or not out:
        sys.exit(f"ERROR: --species {spec!r} gives invalid indices {bad}; valid range is 0..{ns - 1}")
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gmtry-st', required=True, help='source b2fgmtry (unstructured if --src-unstructured)')
    p.add_argument('--gmtry-us', required=True, help='target unstructured b2fgmtry')
    p.add_argument('--state-st', required=True, help='source b2fstate (unstructured if --src-unstructured)')
    p.add_argument('--state-us', required=True, help='unstructured b2fstate template to preserve fields from')
    p.add_argument('--out', required=True)
    p.add_argument('--species', default=None,
                   help="0-based species indices whose 'na' is mapped, e.g. '0,2,4-6' "
                        "(default: all). Unselected species keep the template values.")
    p.add_argument('--smooth', type=float, default=0.0, metavar='SIGMA',
                   help='Gaussian-smooth the mapped fields along each flux tube of the target mesh; '
                        'SIGMA is the kernel width in cells (default 0 = off). na, ne, te, ti are '
                        'smoothed in log space, po linearly. Ghost cells are not smoothed.')
    p.add_argument('--src-unstructured', action='store_true',
                   help='source geometry/state are already unstructured (b2fgmtry_us / b2fstate_us); '
                        'map between two unstructured meshes')
    p.add_argument('--debug', action='store_true', help='print debug info about mapping')
    args = p.parse_args()
    if args.smooth < 0:
        sys.exit('ERROR: --smooth SIGMA must be >= 0')

    io = B2StateIO()
    if args.src_unstructured:
        gmtry_st = io.read_b2fgmtry_us(Path(args.gmtry_st))
        state_st = io.read_b2fstate_us(Path(args.state_st))
        if gmtry_st['nCv'] != state_st['nCv']:
            sys.exit(f"ERROR: source geometry has nCv={gmtry_st['nCv']} but source state has nCv={state_st['nCv']}")
        ns = int(state_st['ns'])
    else:
        gmtry_st = io.read_b2fgmtry_st(Path(args.gmtry_st))
        state_st = io.read_b2fstate_st(Path(args.state_st))
        ns = int(state_st['na'].shape[2])
    gmtry_us = io.read_b2fgmtry_us(Path(args.gmtry_us))
    version_line, blocks = io.parse_b2fstate_template(Path(args.state_us))

    nCv_us = int(gmtry_us['nCv'])
    for b in blocks:
        if b[0].strip().split()[-1] == 'nCv,nFc,ns' and len(b[1]) >= 3 and int(b[1][2]) != ns:
            sys.exit(f"ERROR: template {args.state_us} has ns={int(b[1][2])} but source state has ns={ns}")

    # prepare state_us from template, but will replace selected fields
    def extract_block(blocks, key):
        for b in blocks:
            lab = b[0].strip().split()[-1]
            if lab == key:
                return b[1]
        return None

    state_us = {}
    state_us['version'] = version_line or state_st.get('version', '')
    state_us['nFc'] = int(gmtry_us.get('nFc', 0))
    for meta in ('zamin', 'zamax', 'zn', 'am'):
        v = extract_block(blocks, meta)
        state_us[meta] = np.asarray(v) if v is not None and v.size > 0 else state_st.get(meta, np.zeros(ns))

    def arr_or_zeros(name, shape):
        v = extract_block(blocks, name)
        if v is not None and v.size > 0:
            return np.asarray(v).reshape(shape, order='F')
        return np.zeros(shape)

    state_us['na'] = arr_or_zeros('na', (nCv_us, ns))
    state_us['ua'] = arr_or_zeros('ua', (nCv_us, ns))
    state_us['ne'] = arr_or_zeros('ne', (nCv_us,))
    state_us['te'] = arr_or_zeros('te', (nCv_us,))
    state_us['ti'] = arr_or_zeros('ti', (nCv_us,))
    state_us['tn'] = arr_or_zeros('tn', (nCv_us,))
    state_us['po'] = arr_or_zeros('po', (nCv_us,))

    src_xy = np.column_stack([gmtry_st['cvX'], gmtry_st['cvY']])
    dst_xy = np.column_stack([gmtry_us['cvX'], gmtry_us['cvY']])
    idx = io.nearest_map(src_xy, dst_xy)

    if args.debug:
        ne_src = state_st['ne'].ravel(order='F')
        print('DEBUG: structured ne src: min', ne_src.min(), 'max', ne_src.max(), 'mean', ne_src.mean())
        print('DEBUG: mapping indices sample', idx[:10])
        print('DEBUG: src coords sample', src_xy[idx[:5]])
        print('DEBUG: dst coords sample', dst_xy[:5])

    # flat per-cell source arrays (structured: F-order flattening of (nx+2, ny+2);
    # unstructured: already flat over nCv)
    na_src = state_st['na'].reshape((-1, ns), order='F')
    ne_src = state_st['ne'].ravel(order='F')
    te_src = state_st['te'].ravel(order='F')
    ti_src = state_st['ti'].ravel(order='F')
    po_src = state_st['po'].ravel(order='F')

    species = parse_species(args.species, ns) if args.species else list(range(ns))
    print('Mapping na for species:', ', '.join(
        f"{i} (zn={state_st['zn'][i]:g}, z={state_st['zamin'][i]:g}..{state_st['zamax'][i]:g})" for i in species))
    state_us['na'][:, species] = na_src[idx][:, species]
    state_us['ne'] = ne_src[idx]
    state_us['te'] = te_src[idx]
    state_us['ti'] = ti_src[idx]
    state_us['po'] = po_src[idx]

    if args.smooth > 0:
        try:
            tubes, closed = io.read_flux_tubes(Path(args.gmtry_us))
        except ValueError as e:
            sys.exit(f"ERROR: {e}")
        print(f"Smoothing along {len(tubes)} flux tubes ({sum(closed)} closed), sigma = {args.smooth:g} cells")
        sm = io.smooth_along_tubes
        state_us['na'][:, species] = sm(state_us['na'][:, species], tubes, closed, args.smooth, log=True)
        for name in ('ne', 'te', 'ti'):
            state_us[name] = sm(state_us[name], tubes, closed, args.smooth, log=True)
        state_us['po'] = sm(state_us['po'], tubes, closed, args.smooth)

    state_us['tn'] = np.copy(state_us['ti'])

    if args.debug:
        print('DEBUG: mapped ne dest: min', state_us['ne'].min(), 'max', state_us['ne'].max(), 'mean', state_us['ne'].mean())
        print('DEBUG: sample mapped ne', state_us['ne'][:10])

    io.write_from_template(Path(args.out), version_line, blocks, state_us, gmtry_us)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
