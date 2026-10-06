#!/usr/bin/env python3
"""Maximum and/or minimum of plasmastate quantities as a function of time.

Reads every plasmastate.XXXX snapshot, takes the max/min of the chosen quantities over the
mesh cells and reports value and location (R, Z) versus the simulation time stored in the file.
Quantity syntax is the one of plot_plasmastate.py (NAME, or NAME:IDX for species-resolved
fields; see `plot_plasmastate.py --list`).

Run from the run directory (plasmastate files in ./b2mn.exe.dir, geometry from ./b2fgmtry or
../baserun/b2fgmtry). Prints a table, and opens a plot (use --save to write a PNG instead).

Usage:
    plasmastate_extrema.py                          # max of te,ti
    plasmastate_extrema.py -q te,ti --stat both     # max and min
    plasmastate_extrema.py -q na:1 --stat min
    plasmastate_extrema.py -q ti --box 0.64 1.2 -0.8 0.8      # only search inside R/Z box (e.g. skip inner wall)
    plasmastate_extrema.py -q te --csv te_max.csv   # also write the table
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from plot_plasmastate import (FIELDS, get_quantity, read_cf, read_plasmastate, setup_display,
                              show_figures, valid_cells)


def main():
    here = os.getcwd()
    gm = os.path.join(here, 'b2fgmtry')
    if not os.path.exists(gm):
        gm = os.path.join(here, '..', 'baserun', 'b2fgmtry')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-q', '--quantity', default='te,ti',
                    help='comma-separated fields, NAME or NAME:IDX (default: te,ti)')
    ap.add_argument('--stat', choices=('max', 'min', 'both'), default='max')
    ap.add_argument('--box', nargs=4, type=float, metavar=('RMIN', 'RMAX', 'ZMIN', 'ZMAX'),
                    help='only consider cells whose centre lies inside this R,Z box')
    ap.add_argument('--csv', help='write the table to this file')
    ap.add_argument('--log', action='store_true', help='log y axis')
    ap.add_argument('-d', '--dir', default=os.path.join(here, 'b2mn.exe.dir'))
    ap.add_argument('-g', '--geometry', default=gm)
    ap.add_argument('-o', '--out', default=os.path.join(here, 'plasmastate_2d'))
    ap.add_argument('--save', action='store_true', help='write the PNG to --out instead of showing it')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, 'plasmastate.*')))
    if not files:
        sys.exit(f'no plasmastate files found in {a.dir}')
    specs = [s.strip() for s in a.quantity.split(',') if s.strip()]
    stats = ('max', 'min') if a.stat == 'both' else (a.stat,)

    geo = read_cf(a.geometry, {'cvX', 'cvY', 'cvVxP'})
    nCv = len(geo['cvX'])
    sel = valid_cells(geo)
    if a.box:
        r0, r1, z0, z1 = a.box
        sel &= (geo['cvX'] >= r0) & (geo['cvX'] <= r1) & (geo['cvY'] >= z0) & (geo['cvY'] <= z1)
    sel = np.flatnonzero(sel)
    if not sel.size:
        sys.exit('no cells selected')
    zamin = read_cf('b2fstate', {'zamin'}).get('zamin') if os.path.exists('b2fstate') else None

    rows = {}  # (label, stat) -> list of (tag, t, value, R, Z)
    units = {}
    for fn in files:
        tag = fn.rsplit('.', 1)[1]
        ps = read_plasmastate(fn)
        t = float(ps['time'][0]) if 'time' in ps else np.nan
        for spec in specs:
            label, unit, v = get_quantity(ps, spec, nCv, zamin)
            units[label] = unit
            vs = v[sel]
            for st in stats:
                i = sel[np.argmax(vs) if st == 'max' else np.argmin(vs)]
                rows.setdefault((label, st), []).append((tag, t, v[i], geo['cvX'][i], geo['cvY'][i]))

    lines = []
    for (label, st), r in rows.items():
        lines.append(f'# {st} of {label} [{units[label]}]' + (f'  (box R {a.box[0]}..{a.box[1]}, Z {a.box[2]}..{a.box[3]})' if a.box else ''))
        lines.append(f'{"snapshot":>9} {"t [s]":>12} {"value":>14} {"R [m]":>9} {"Z [m]":>9}')
        lines += [f'{tag:>9} {t:12.6g} {val:14.6g} {R:9.4f} {Z:9.4f}' for tag, t, val, R, Z in r]
        lines.append('')
    print('\n'.join(lines))
    if a.csv:
        with open(a.csv, 'w') as f:
            f.write('quantity,stat,snapshot,time_s,value,R_m,Z_m\n')
            for (label, st), r in rows.items():
                for tag, t, val, R, Z in r:
                    f.write(f'{label},{st},{tag},{t:.8g},{val:.8g},{R:.5f},{Z:.5f}\n')
        print('table written to', a.csv)

    tmpdir = setup_display(a.save, a.out)
    # one panel per unit family so Te/Ti share axes but na/ua/po do not
    groups = {}
    for (label, st), r in rows.items():
        groups.setdefault(units[label], []).append((label, st, r))
    fig, axs = plt.subplots(len(groups), 1, figsize=(9, 3.6 * len(groups) + 1), sharex=True,
                            constrained_layout=True, squeeze=False)
    for ax, (unit, items) in zip(axs[:, 0], groups.items()):
        colors = {}
        for label, st, r in items:
            t = [x[1] for x in r]
            c = colors.setdefault(label, f'C{len(colors)}')
            ax.plot(t, [x[2] for x in r], marker='o', ms=4, ls='--' if st == 'min' else '-',
                    color=c, label=f'{st} {label}')
        ax.set_ylabel(f'[{unit}]')
        if a.log:
            ax.set_yscale('log')
        ax.grid(alpha=0.3)
        ax.legend()
    axs[-1, 0].set_xlabel('time [s]')
    axs[0, 0].set_title('plasmastate extrema' + (f' (box R {a.box[0]}..{a.box[1]}, Z {a.box[2]}..{a.box[3]})' if a.box else ''))
    shown = []
    if a.save or tmpdir:
        png = os.path.join(a.out if a.save else tmpdir,
                           f'extrema_{"_".join(s.replace(":", "") for s in specs)}_{a.stat}.png')
        fig.savefig(png, dpi=130)
        plt.close(fig)
        shown.append(png)
        if a.save:
            print('figure written to', png)
    show_figures(a.save, tmpdir, shown)


if __name__ == '__main__':
    main()
