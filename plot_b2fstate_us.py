#!/usr/bin/env python3
"""Plot scalar fields from an unstructured `b2fstate` on the CV centers.

Usage example:
  plot_b2fstate_us.py --gmtry-us GMUS --state-us B2FSTATE_US --field ne --save ne.png --tri

Features:
- reads geometry with `B2StateIO.read_b2fgmtry_us`
- reads unstructured state via `B2StateIO.parse_b2fstate_template`
- plots a scatter colored by the chosen field
- optional Delaunay triangulation (requires SciPy) to use `tripcolor`
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
_thisdir = Path(__file__).resolve().parent
if str(_thisdir) not in sys.path:
    sys.path.insert(0, str(_thisdir))

from b2state_io import B2StateIO
from plotter import Plotter


def extract_block(blocks, key: str):
    for b in blocks:
        lab = b[0].strip().split()[-1]
        if lab == key:
            return b[1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gmtry-us', required=True)
    p.add_argument('--state-us', required=True)
    p.add_argument('--field', default='ne', help='field to plot (ne, na, te, ti, po, ua)')
    p.add_argument('--slice', type=int, default=0, help='for multi-z fields select slice index (na, ua)')
    p.add_argument('--tri', action='store_true', help='use Delaunay triangulation for tripcolor (requires scipy)')
    p.add_argument('--cmap', default='viridis')
    p.add_argument('--size', type=int, default=12, help='marker size for scatter')
    p.add_argument('--save', help='output image file (png, pdf, ...). If omitted, shows interactively')
    args = p.parse_args()

    io = B2StateIO()
    gmtry = io.read_b2fgmtry_us(Path(args.gmtry_us))
    version_line, blocks = io.parse_b2fstate_template(Path(args.state_us))

    nCv = int(gmtry['nCv'])
    x = gmtry['cvX']
    y = gmtry['cvY']

    vals = extract_block(blocks, args.field)
    if vals is None or vals.size == 0:
        raise SystemExit(f"Field '{args.field}' not found in {args.state_us}")

    # determine shape: scalar per cell, or per-cell per-slice
    if vals.size == nCv:
        field = np.asarray(vals)
    elif vals.size % nCv == 0:
        ns = vals.size // nCv
        arr = np.asarray(vals).reshape((nCv, ns), order='F')
        idx = args.slice if args.slice < ns else 0
        field = arr[:, idx]
    else:
        raise SystemExit('Unrecognized field size vs nCv')

    pl = Plotter(figsize=(8, 6))

    if args.tri:
        tris = pl.delaunay(x, y)
        if tris is not None:
            m = pl.tripcolor(x, y, tris, field, cmap=args.cmap)
            pl.colorbar(m)
        else:
            m = pl.scatter(x, y, c=field, s=args.size, cmap=args.cmap)
            pl.set_title(f"{args.field} (scatter; triangulation unavailable)")
            pl.colorbar(m)
    else:
        m = pl.scatter(x, y, c=field, s=args.size, cmap=args.cmap)
        pl.colorbar(m)

    pl.set_aspect_equal()
    pl.set_labels('X', 'Y')
    pl.set_title(f"{args.field} on unstructured mesh")

    if args.save:
        pl.save(args.save)
        print(f"Wrote {args.save}")
    else:
        pl.show()


if __name__ == '__main__':
    main()
