#!/usr/bin/env python3
"""Plot face fluxes (fhi, fhe, fch, fna:IDX, ...) from plasmastate.XXXX files on the 2D B2.5 mesh.

Face fluxes are integrated over the face (fhi/fhe in W). Layout, as used by b2ursd:
    fdf(nFc, 0:1)  -> file order [all faces comp 0][all faces comp 1]; total face flux = comp0 + comp1,
    positive from cell fcCv(f,1) to fcCv(f,2); fcCv is (nFc, 2) in the b2fgmtry file.
Species-resolved ones (fna, fhm, ...) are (nFc, 0:1, 0:ns-1) and selected as NAME:IDX.

Two views:
    default : every face is drawn as a coloured line segment (colour = signed flux)
    --div   : net flux OUT of each cell (sum over its faces), drawn on the cell polygons

Run from the run directory (same defaults as plot_plasmastate.py).

Usage:
    plot_plasmastate_flux.py 0011                       # fhi on faces
    plot_plasmastate_flux.py -q fhi,fhe 0011            # side by side
    plot_plasmastate_flux.py -q fhi --div 0011          # divergence of fhi per cell
    plot_plasmastate_flux.py -q fna:1 --comp 1 0011     # only component 1 of D+ particle flux
    plot_plasmastate_flux.py -q fhi --zoom 0.60 0.66 -0.30 -0.05 0011
"""
import argparse
import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, SymLogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_plasmastate import build_polygons, read_cf, read_plasmastate  # noqa: E402


def face_flux(ps, spec, nFc, comp):
    name, _, idx = spec.partition(':')
    if name not in ps:
        sys.exit(f"'{name}' is not in the plasmastate file")
    v = ps[name]
    if len(v) == 2 * nFc:
        arr = v.reshape(2, nFc)
    elif len(v) % (2 * nFc) == 0:
        ns = len(v) // (2 * nFc)
        arr = v.reshape(ns, 2, nFc)[int(idx or 0)]
    else:
        sys.exit(f"'{name}' (size {len(v)}) is not a face flux")
    return name + (f'[{idx}]' if idx else ''), (arr.sum(0) if comp is None else arr[comp])


def main():
    here = os.getcwd()
    gm = os.path.join(here, 'b2fgmtry')
    if not os.path.exists(gm):
        gm = os.path.join(here, '..', 'baserun', 'b2fgmtry')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ids', nargs='*', help="plasmastate suffixes, e.g. 0012 (default: all)")
    ap.add_argument('-q', '--quantity', default='fhi')
    ap.add_argument('--div', action='store_true', help='net outflow per cell instead of face lines')
    ap.add_argument('--comp', type=int, choices=(0, 1), default=None,
                    help='use only this flux component (default: comp0 + comp1)')
    ap.add_argument('-d', '--dir', default=os.path.join(here, 'b2mn.exe.dir'))
    ap.add_argument('-g', '--geometry', default=gm)
    ap.add_argument('-o', '--out', default=os.path.join(here, 'plasmastate_2d'))
    ap.add_argument('--save', action='store_true', help='write PNGs to --out instead of showing figures')
    ap.add_argument('--vmax', type=float, default=None, help='symmetric colour limit [file units]')
    ap.add_argument('--zoom', nargs=4, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    a = ap.parse_args()

    files = (sorted(glob.glob(os.path.join(a.dir, 'plasmastate.*'))) if not a.ids else
             [os.path.join(a.dir, 'plasmastate.' + i) for i in a.ids])
    if not files:
        sys.exit(f'no plasmastate files found in {a.dir}')
    geo = read_cf(a.geometry, {'cvX', 'cvY', 'vxX', 'vxY', 'cvVxP', 'cvVx', 'fcCv', 'fcVx'})
    nCv, nFc = len(geo['cvX']), len(geo['fcCv']) // 2
    fcCv = geo['fcCv'].reshape(2, nFc) - 1
    fcVx = geo['fcVx'].reshape(2, nFc) - 1
    vx = np.column_stack([geo['vxX'], geo['vxY']])
    segs = np.stack([vx[fcVx[0]], vx[fcVx[1]]], axis=1)
    polys, ok = build_polygons(geo)
    sel = np.flatnonzero(ok)
    polys = [polys[i] for i in sel]

    if a.save:
        matplotlib.use('Agg')
        os.makedirs(a.out, exist_ok=True)
    else:
        try:
            matplotlib.use('TkAgg', force=True)
            import tkinter
            tkinter.Tk().destroy()
        except Exception:
            matplotlib.use('Agg', force=True)
            a.save = True
            os.makedirs(a.out, exist_ok=True)
    import matplotlib.pyplot as plt  # noqa: F811  (after backend choice)

    specs = [s.strip() for s in a.quantity.split(',') if s.strip()]
    for fn in files:
        tag = fn.rsplit('.', 1)[1]
        ps = read_plasmastate(fn)
        t = float(ps['time'][0]) if 'time' in ps else np.nan
        fig, axs = plt.subplots(1, len(specs), figsize=(5.5 * len(specs) + 1, 8), sharey=True,
                                constrained_layout=True, squeeze=False)
        for ax, spec in zip(axs[0], specs):
            label, f = face_flux(ps, spec, nFc, a.comp)
            if a.div:
                v = np.zeros(nCv)
                np.add.at(v, fcCv[0], f)    # leaves cell 1
                np.add.at(v, fcCv[1], -f)   # enters cell 2
                vs = v[sel]
                title = f'net outflow of {label}'
            else:
                vs, title = f, f'{label} (+: cell1 -> cell2)'
            m = a.vmax if a.vmax else np.nanmax(np.abs(vs))
            norm = SymLogNorm(linthresh=max(m * 1e-6, 1e-30), vmin=-m, vmax=m, base=10)
            if a.div:
                col = PolyCollection(polys, array=vs, norm=norm, cmap='RdBu_r', edgecolors='none')
            else:
                col = LineCollection(segs, array=vs, norm=norm, cmap='RdBu_r', linewidths=1.5)
            ax.add_collection(col)
            ax.set_aspect('equal')
            ax.set_xlim(*(a.zoom[:2] if a.zoom else (vx[:, 0].min(), vx[:, 0].max())))
            ax.set_ylim(*(a.zoom[2:] if a.zoom else (vx[:, 1].min(), vx[:, 1].max())))
            ax.set_title(f'{title}\n(plasmastate.{tag}, t={t:.4g} s)')
            ax.set_xlabel('R [m]')
            fig.colorbar(col, ax=ax, shrink=0.8)
            i = int(np.argmax(np.abs(vs)))
            print(f'{fn} {title}: |max| = {vs[i]:.4g} at index {i}')
        axs[0][0].set_ylabel('Z [m]')
        if a.save:
            png = os.path.join(a.out, f'{"_".join(s.replace(":", "") for s in specs)}'
                                      f'{"_div" if a.div else "_faces"}_{tag}.png')
            fig.savefig(png, dpi=130)
            plt.close(fig)
            print('wrote', png)
    if not a.save:
        plt.show()


if __name__ == '__main__':
    main()
