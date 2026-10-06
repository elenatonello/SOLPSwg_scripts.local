#!/usr/bin/env python3
"""Show where the wall faces are grazing, from the b2fgmtry of an unstructured B2.5 grid.

For every wall face (a face between an interior cell and a guard cell) it computes
    wrk = |fcQalf(0) * fcBb(0) / fcBb(3)|        (= Pbs/S, the magnetic contact fraction)
and classifies it with Qalfmin / Qalfmax (read from the file, or given on the command line):
    wrk == 0              closed-surface (core-type) faces, drawn grey
    wrk <  Qalfmin        grazing: boundary conditions use the fixed floor sink S*Qalfmax, the
                          guard velocity and potential are pinned to the interior values
    Qalfmin <= wrk < Qalfmax   raised to Pbs = Qalfmax*S by the geometry prep (if it was run)
    wrk >= Qalfmax        normal wall face
Wall faces are drawn as thick coloured segments on top of the (light) cell outlines. With
--compare the faces whose class differs between two Qalfmin values are circled.

Run from a run directory. Geometry is taken from ./b2fgmtry, ./b2mn.exe.dir/b2fgmtry or
../baserun/b2fgmtry (or give -g). The figure is opened in a window (TkAgg, like the other scripts in
scripts.local); use --save to write a PNG to ./qalf_wall/ instead. Without a display the figure is
written to a temporary directory and opened in VS Code.

Usage:
    plot_qalf_wall.py                              # use Qalfmin/Qalfmax stored in the file
    plot_qalf_wall.py --qmin 1e-3 --qmax 1e-3      # what-if values
    plot_qalf_wall.py --qmin 2e-3 --compare 1e-3   # circle the faces that change class vs Qalfmin=1e-3
    plot_qalf_wall.py --zoom 0.6 0.7 0.55 0.8      # zoom on R 0.6-0.7, Z 0.55-0.8
    plot_qalf_wall.py -g b2fgmtry_prep --ref b2fgmtry_orig   # faces at their ORIGINAL class, coloured by whether
                                                             # the prepared file was actually reduced / raised there
    plot_qalf_wall.py --list 30                    # print the 30 most grazing faces
    plot_qalf_wall.py --show pbs                   # colour every wall face by log10 of its stored |Pbs|/S
    plot_qalf_wall.py -g b2fgmtry_prep --ref b2fgmtry_orig --show dpbs   # colour by log10(Pbs_new/Pbs_old):
                                                   # blue = reduced, red = raised, grey = unchanged
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_plasmastate import build_polygons, read_cf  # noqa: E402


def find_geometry(here):
    for p in (os.path.join(here, 'b2fgmtry'), os.path.join(here, 'b2mn.exe.dir', 'b2fgmtry'),
              os.path.join(here, '..', 'baserun', 'b2fgmtry')):
        if os.path.exists(p):
            return p
    sys.exit('b2fgmtry not found; give it with -g')


def main():
    here = os.getcwd()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('-g', '--geometry', default=None)
    ap.add_argument('--qmin', type=float, default=None, help='Qalfmin (default: value in the file)')
    ap.add_argument('--qmax', type=float, default=None, help='Qalfmax (default: value in the file)')
    ap.add_argument('--compare', type=float, default=None, metavar='QMIN2',
                    help='circle faces whose class changes between --qmin and QMIN2')
    ap.add_argument('--ref', default=None, metavar='B2FGMTRY_ORIG',
                    help='unmodified geometry: classify the faces by their wrk in THIS file (where the grazing '
                         'and band faces were), and show for each whether the file given with -g was actually '
                         'reduced / raised there (Pbs/S)')
    ap.add_argument('--show', choices=('class', 'pbs', 'dpbs'), default='class',
                    help="'class' (default): faces coloured by grazing class; 'pbs': by log10 of the stored "
                         "|Pbs|/S of the -g file; 'dpbs': by log10(Pbs/S of -g file / Pbs/S of --ref file) "
                         "(needs --ref)")
    ap.add_argument('--zoom', nargs=4, type=float, metavar=('RMIN', 'RMAX', 'ZMIN', 'ZMAX'))
    ap.add_argument('--list', type=int, default=0, metavar='N', help='print the N lowest non-zero wrk faces')
    ap.add_argument('-o', '--out', default=os.path.join(here, 'qalf_wall'))
    ap.add_argument('--save', action='store_true', help='write a PNG to --out instead of showing a window')
    a = ap.parse_args()
    if a.show == 'dpbs' and not a.ref:
        ap.error('--show dpbs needs --ref (the unmodified geometry to compare with)')

    gm = a.geometry or find_geometry(here)
    g = read_cf(gm, {'cvX', 'cvY', 'vxX', 'vxY', 'cvVxP', 'cvVx', 'fcCv', 'fcVx', 'fcQalf', 'fcBb', 'fcS',
                     'fcPbs', 'Qalfmin', 'Qalfmax', 'nCi,nCg,nCv,nFc,nVx,nFs,nFt'})
    hk = g['nCi,nCg,nCv,nFc,nVx,nFs,nFt']
    nCi, nFc = int(hk[0]), int(hk[3])
    qmin = a.qmin if a.qmin is not None else float(g['Qalfmin'][0]) if 'Qalfmin' in g else 1e-3
    qmax = a.qmax if a.qmax is not None else float(g['Qalfmax'][0]) if 'Qalfmax' in g else 1e-3
    print(f'geometry: {gm}\nQalfmin = {qmin:g}, Qalfmax = {qmax:g}'
          + ('' if a.qmin is not None or a.qmax is not None else '  (as stored in the file)'))

    fcCv = g['fcCv'].reshape(2, nFc) - 1
    fcVx = g['fcVx'].reshape(2, nFc) - 1
    qalf = g['fcQalf'].reshape(2, nFc)
    bb = g['fcBb'].reshape(4, nFc)
    wrk = np.abs(qalf[0] * bb[0] / bb[3])
    wall = np.array([f for f in range(nFc) if (fcCv[0, f] >= nCi) != (fcCv[1, f] >= nCi)])
    inner = np.where(fcCv[0, wall] < nCi, fcCv[0, wall], fcCv[1, wall])
    guard = np.where(fcCv[0, wall] < nCi, fcCv[1, wall], fcCv[0, wall])
    w = wrk[wall]
    x, y = g['cvX'], g['cvY']
    vx = np.column_stack([g['vxX'], g['vxY']])

    def classify(w, qmin):
        c = np.full(len(w), 3)              # 3 normal
        c[(w < qmax * (1 - 1e-9)) & (w >= qmin)] = 2   # band (tolerance: faces raised to Qalfmax read as Qalfmax)
        c[w < qmin] = 1                     # grazing
        c[w == 0] = 0                       # core-type
        return c
    cls = classify(w, qmin)
    names = {0: 'wrk = 0 (closed-surface type)', 1: f'grazing: wrk < Qalfmin ({qmin:g})',
             2: f'Qalfmin <= wrk < Qalfmax ({qmax:g})', 3: 'normal wall face'}
    if a.ref:
        # classes by the wrk of the reference (unmodified) file; refined by what the -g file actually holds
        r = read_cf(a.ref, {'fcCv', 'fcQalf', 'fcBb', 'fcPbs'})
        if not np.array_equal(r['fcCv'], g['fcCv']):
            sys.exit('--ref geometry has a different face-cell connectivity')
        rq, rb = r['fcQalf'].reshape(2, nFc), r['fcBb'].reshape(4, nFc)
        w_ref = np.abs(rq[0] * rb[0] / rb[3])[wall]
        pbs_ref_ratio = np.abs(r['fcPbs'][wall] / g['fcS'][wall])
        pbs_ratio = np.abs(g['fcPbs'][wall] / g['fcS'][wall])
        cls = classify(w_ref, qmin)
        raised = np.abs(pbs_ratio - qmax) <= 1e-9 * qmax + 1e-15
        reduced = pbs_ratio < 0.5 * w_ref
        cls = np.where((cls == 2) & raised, 2, np.where(cls == 2, 4, cls))        # 4: band, Pbs not raised
        cls = np.where((cls == 1) & ~reduced, 5, cls)                              # 5: grazing, Pbs not reduced
        names = {0: 'wrk = 0 (closed-surface type)',
                 1: f'grazing in ref (wrk < {qmin:g}), Pbs reduced in this file',
                 5: f'grazing in ref (wrk < {qmin:g}), Pbs NOT reduced in this file',
                 2: f'band in ref ({qmin:g} <= wrk < {qmax:g}), Pbs/S raised to {qmax:g}',
                 4: f'band in ref ({qmin:g} <= wrk < {qmax:g}), Pbs NOT raised in this file',
                 3: 'normal wall face (unchanged)'}
        print(f'reference geometry: {a.ref}')
        clip = (cls == 2) & (w < qmax * (1 - 1e-9))
        for k in (0, 1, 5, 2, 4, 3):
            print(f'  {names[k]:<62s} {int((cls == k).sum()):4d} faces')
        print(f'  (of the raised faces, {int(clip.sum())} are clipped: Qalf-based wrk = Bx/B < Qalfmax; '
              'Pbs/S is Qalfmax but Qalf0*Bx/B is not)')
    else:
        for k in range(4):
            print(f'  {names[k]:<44s} {int((cls == k).sum()):4d} faces')

    if a.list and a.show == 'class':
        nz = np.where(w > 0)[0]
        o = nz[np.argsort(w[nz])][:a.list]
        print(f'\n{a.list} lowest non-zero wrk (interior cell, guard cell, R, Z, wrk, class):')
        for i in o:
            print(f'  {inner[i] + 1:6d} {guard[i] + 1:6d}  R={x[inner[i]]:.3f} Z={y[inner[i]]:.3f}  '
                  f'wrk={w[i]:.2e}  {"core" if cls[i] == 0 else "GRAZING" if cls[i] in (1, 5) else "band" if cls[i] in (2, 4) else "normal"}')

    changed = None
    if a.compare is not None:
        cls2 = classify(w, a.compare)
        changed = np.where((cls == 1) != (cls2 == 1))[0]
        print(f'\nfaces whose grazing status differs between Qalfmin={qmin:g} and {a.compare:g}: {len(changed)}')
        for i in changed[np.argsort(y[inner[changed]])]:
            print(f'  {inner[i] + 1:6d} {guard[i] + 1:6d}  R={x[inner[i]]:.3f} Z={y[inner[i]]:.3f}  wrk={w[i]:.2e}')

    # same display logic as plot_plasmastate.py: --save writes a PNG; otherwise try a TkAgg window;
    # without a display, write a PNG to a temporary directory and open it in VS Code
    interactive = False
    if a.save:
        matplotlib.use('Agg')
        os.makedirs(a.out, exist_ok=True)
    else:
        try:
            matplotlib.use('TkAgg', force=True)
            import tkinter
            tkinter.Tk().destroy()
            interactive = True
        except Exception:
            matplotlib.use('Agg', force=True)
    tmpdir = None if a.save or interactive else tempfile.mkdtemp(prefix='qalf_wall_')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    polys, ok = build_polygons(g)
    sel = [i for i in np.flatnonzero(ok) if i < nCi]
    fig, ax = plt.subplots(figsize=(7, 11), constrained_layout=True)
    ax.add_collection(PolyCollection([polys[i] for i in sel], facecolors='none', edgecolors='#cccccc',
                                     linewidths=0.2))
    S_w = g['fcS'][wall]
    pbs_now = np.abs(g['fcPbs'][wall]) / S_w
    cbar_lc = None
    if a.show == 'class':
        colors = {0: '#999999', 1: '#d62728', 2: '#ff9f1c', 3: '#1f77b4', 4: '#ffd9a0', 5: '#f4a6a6'}
        widths = {0: 1.5, 1: 4.0, 2: 3.0, 3: 1.5, 4: 3.0, 5: 4.0}
        for k in (0, 3, 4, 2, 5, 1):
            m = cls == k
            if m.any():
                segs = np.stack([vx[fcVx[0, wall[m]]], vx[fcVx[1, wall[m]]]], axis=1)
                ax.add_collection(LineCollection(segs, colors=colors[k], linewidths=widths[k],
                                                 label=f'{names[k]}  [{int(m.sum())}]'))
        if changed is not None and len(changed):
            ax.plot(x[inner[changed]], y[inner[changed]], 'o', mfc='none', mec='k', ms=14, mew=1.5,
                    label=f'grazing status differs vs Qalfmin={a.compare:g}  [{len(changed)}]')
    else:
        from matplotlib.colors import Normalize, TwoSlopeNorm
        if a.show == 'pbs':
            show = pbs_now > 0
            val = np.log10(np.where(show, pbs_now, 1.0))
            norm, cmap, clabel = Normalize(val[show].min(), val[show].max()), 'viridis', 'log10(|Pbs| / S)   (stored in the file)'
            thin_label = 'Pbs = 0 (closed-surface type)'
        else:
            okr = pbs_ref_ratio > 0
            val = np.zeros(len(wall))
            val[okr] = np.log10(pbs_now[okr] / pbs_ref_ratio[okr])
            show = okr & (np.abs(val) > 1e-9)
            lo, hi = min(val.min(), -1e-3), max(val.max(), 1e-3)
            norm, cmap, clabel = TwoSlopeNorm(0.0, vmin=lo, vmax=hi), 'RdBu_r', 'log10( Pbs/S in file / Pbs/S in --ref )'
            thin_label = 'unchanged'
            print(f'\nPbs/S compared with --ref: {int(show.sum())} wall faces changed '
                  f'(reduced {int((val < -1e-9).sum())}, raised {int((val > 1e-9).sum())}); '
                  f'log10 ratio range {val.min():.2f} .. {val.max():.2f}')
        thin = ~show
        if thin.any():
            segs = np.stack([vx[fcVx[0, wall[thin]]], vx[fcVx[1, wall[thin]]]], axis=1)
            ax.add_collection(LineCollection(segs, colors='#aaaaaa', linewidths=1.5,
                                             label=f'{thin_label}  [{int(thin.sum())}]'))
        if show.any():
            segs = np.stack([vx[fcVx[0, wall[show]]], vx[fcVx[1, wall[show]]]], axis=1)
            lc = LineCollection(segs, array=val[show], cmap=cmap, norm=norm, linewidths=4.5)
            ax.add_collection(lc)
            cbar_lc = (lc, clabel)
        if a.list:
            idx = np.where(show)[0]
            idx = idx[np.argsort(-np.abs(val[idx]))] if a.show == 'dpbs' else idx[np.argsort(val[idx])]
            print(f'\n{min(a.list, len(idx))} faces ({"largest |change|" if a.show == "dpbs" else "smallest Pbs/S"}):')
            print('  interior guard    R      Z      Pbs/S now' + ('   Pbs/S ref    ratio' if a.show == 'dpbs' else ''))
            for i in idx[:a.list]:
                line = f'  {inner[i] + 1:6d} {guard[i] + 1:6d}  {x[inner[i]]:.3f} {y[inner[i]]:+.3f}  {pbs_now[i]:.3e}'
                if a.show == 'dpbs':
                    line += f'  {pbs_ref_ratio[i]:.3e}  {pbs_now[i] / pbs_ref_ratio[i]:.3g}'
                print(line)
    if a.show != 'class' and changed is not None and len(changed):
        ax.plot(x[inner[changed]], y[inner[changed]], 'o', mfc='none', mec='k', ms=14, mew=1.5,
                label=f'grazing status differs vs Qalfmin={a.compare:g}  [{len(changed)}]')
    ax.set_aspect('equal')
    ax.set_xlim(*(a.zoom[:2] if a.zoom else (vx[:, 0].min() - 0.01, vx[:, 0].max() + 0.01)))
    ax.set_ylim(*(a.zoom[2:] if a.zoom else (vx[:, 1].min() - 0.01, vx[:, 1].max() + 0.01)))
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(f'wall faces by wrk = Pbs/S   (Qalfmin={qmin:g}, Qalfmax={qmax:g})'
                 + ('\nclasses from --ref, state from -g' if a.ref else ''))
    ax.legend(loc='lower center', fontsize=8, framealpha=0.9)
    if cbar_lc is not None:
        fig.colorbar(cbar_lc[0], ax=ax, shrink=0.55, pad=0.02, label=cbar_lc[1])
        ax.set_title({'pbs': 'stored Pbs/S of the wall faces', 'dpbs': 'change of Pbs/S (file vs --ref)'}[a.show])
    name = (f'qalf_wall_qmin{qmin:g}_qmax{qmax:g}' + ('' if a.show == 'class' else f'_{a.show}')
            + ('_zoom' if a.zoom else '') + '.png')
    if a.save or tmpdir:
        png = os.path.join(a.out if a.save else tmpdir, name)
        fig.savefig(png, dpi=140)
        print('\nwrote', png)
        if tmpdir:
            code = shutil.which('code')
            if code:
                subprocess.run([code, '-r', png], check=False)
            print('No display available; figure written to', tmpdir)
    else:
        plt.show()


if __name__ == '__main__':
    main()
