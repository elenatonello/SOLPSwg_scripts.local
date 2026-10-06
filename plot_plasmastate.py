#!/usr/bin/env python3
"""Plot cell-centred quantities from plasmastate.XXXX files on the 2D B2.5 mesh.

plasmastate files are Fortran unformatted sequential records:
    record 0 : version string
    then per field: header record ('*cf: ', type, count, name), data record (little-endian)
Cell-centred fields (te, ti, ne, po, ...) have nCv values; species-resolved ones (na, ua,
kinrgy, ...) have ns*nCv values and are selected as NAME:IDX (species index, from 0).
Face/other fields cannot be drawn on the cell map and are only shown by --list.
The cell polygons come from the b2fgmtry file (cvVxP/cvVx/vxX/vxY).

Run from the run directory. Defaults: plasmastate files in ./b2mn.exe.dir, geometry from
./b2fgmtry or ../baserun/b2fgmtry. Figures are opened in windows (TkAgg, like the other
scripts in scripts.local); use --save to write PNG + npz files to ./plasmastate_2d/ instead.
Without a display the figures are written to a temporary directory and opened in VS Code.

Usage:
    plot_plasmastate.py --list                  # fields available in the plasmastate file
    plot_plasmastate.py                         # Te and Ti, all snapshots (one window each)
    plot_plasmastate.py 0011                    # a single snapshot
    plot_plasmastate.py -l                      # the most recent snapshot (or: plot_plasmastate.py latest)
    plot_plasmastate.py -q te,ti,po 0011        # chosen quantities side by side
    plot_plasmastate.py -q na:1,ua:1 0011       # species-resolved: density and parallel velocity of species 1
    plot_plasmastate.py -q ne --lin --vmax 1e19 --zoom 0.62 0.8 -0.45 -0.2 0011
    plot_plasmastate.py --save                  # write PNG + npz instead of showing
"""
import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

EV = 1.602176634e-19

# name -> (label, unit, factor applied to the file value); anything else: file units
FIELDS = {
    'te': ('Te', 'eV', 1 / EV),
    'ti': ('Ti', 'eV', 1 / EV),
    'tn': ('Tn', 'eV', 1 / EV),
    'ne': ('ne', 'm^-3', 1.0),
    'na': ('na', 'm^-3', 1.0),
    'ua': ('ua', 'm/s', 1.0),
    'po': ('po', 'V', 1.0),
}


def read_plasmastate(fn):
    out = {}
    with open(fn, 'rb') as f:
        def rec():
            h = f.read(4)
            if len(h) < 4:
                return None
            n = struct.unpack('<i', h)[0]
            d = f.read(n)
            f.read(4)
            return d
        rec()  # version string
        while True:
            h = rec()
            if h is None:
                break
            typ = h[8:16].decode().strip()
            cnt = struct.unpack('<i', h[16:20])[0]
            name = h[20:].decode().strip()
            d = rec() if cnt > 0 else b''
            if typ == 'real':
                out[name] = np.frombuffer(d, '<f8')
            elif typ == 'int':
                out[name] = np.frombuffer(d, '<i4')
    return out


def read_cf(fn, want):
    """Minimal reader for formatted B2.5 '*cf:' files, only the arrays in `want`."""
    d = {}
    with open(fn) as f:
        lines = f.read().split('\n')
    i = 1
    while i < len(lines):
        if not lines[i].startswith('*cf:'):
            i += 1
            continue
        p = lines[i].split()
        typ, n, name = p[1], int(p[2]), p[3]
        i += 1
        if typ == 'char':
            i += 1
            continue
        vals = []
        while len(vals) < n:
            vals += lines[i].split()
            i += 1
        if name in want:
            d[name] = np.array(vals, dtype=float if typ == 'real' else int)
    return d


def build_polygons(geo):
    nCv = len(geo['cvX'])
    start = geo['cvVxP'][:nCv] - 1
    count = geo['cvVxP'][nCv:]
    vx = np.column_stack([geo['vxX'], geo['vxY']])
    polys, ok = [], np.zeros(nCv, bool)
    for c in range(nCv):
        if count[c] < 3:
            polys.append(np.zeros((3, 2)))
            continue
        p = vx[geo['cvVx'][start[c]:start[c] + count[c]] - 1]
        ang = np.arctan2(p[:, 1] - p[:, 1].mean(), p[:, 0] - p[:, 0].mean())
        polys.append(p[np.argsort(ang)])
        ok[c] = True
    return polys, ok


def kind(name, v, nCv):
    """'cell', ('species', ns) or None for fields that are not cell data."""
    if v.dtype.kind != 'f' or len(v) == 0:
        return None
    if len(v) == nCv:
        return 'cell'
    if len(v) % nCv == 0:
        return ('species', len(v) // nCv)
    return None


def list_fields(ps, nCv, zamin):
    print(f'{"field":<14}{"size":>9}  kind')
    for name, v in ps.items():
        k = kind(name, v, nCv)
        if k == 'cell':
            desc = 'cell        -q ' + name
        elif k:
            desc = f'cell x {k[1]} species   -q {name}:0..{k[1] - 1}'
        elif len(v) == 1:
            desc = f'scalar = {v[0]:.6g}'
        else:
            desc = 'face/other (not plottable on the cell map)'
        print(f'{name:<14}{len(v):>9}  {desc}')
    if zamin is not None:
        print('\nspecies index -> charge zamin: ' + ', '.join(f'{i}:{z:g}' for i, z in enumerate(zamin)))


def get_quantity(ps, spec, nCv, zamin):
    name, _, idx = spec.partition(':')
    if name not in ps:
        sys.exit(f"'{name}' is not in the plasmastate file (see --list)")
    v = ps[name]
    k = kind(name, v, nCv)
    if k is None:
        sys.exit(f"'{name}' is not a cell-centred field and cannot be drawn on the map (see --list)")
    label, unit, fac = FIELDS.get(name, (name, 'file units', 1.0))
    if k == 'cell':
        if idx:
            sys.exit(f"'{name}' is not species-resolved; use -q {name}")
        return label, unit, v * fac
    ns = k[1]
    i = int(idx) if idx else 0
    if not 0 <= i < ns:
        sys.exit(f"species index for '{name}' must be 0..{ns - 1}")
    if not idx:
        print(f"note: '{name}' is species-resolved, showing species 0 (use -q {name}:IDX)")
    z = f', zamin={zamin[i]:g}' if zamin is not None and len(zamin) == ns else ''
    return f'{label}[{i}{z}]', unit, v.reshape(ns, nCv)[i] * fac


def setup_display(save, out):
    """Choose the matplotlib backend; return a temp dir if figures must go through files for VS Code."""
    if save:
    
        matplotlib.use('Agg')
        os.makedirs(out, exist_ok=True)
        return None
    # same backend as the other scripts here; fall back if Tk can't open a window
    try:
        matplotlib.use('TkAgg', force=True)
        import tkinter
        tkinter.Tk().destroy()
        return None
    except Exception:
        matplotlib.use('Agg', force=True)
        return tempfile.mkdtemp(prefix='plasmastate_')


def show_figures(save, tmpdir, shown):
    if tmpdir:
        code = shutil.which('code')
        if code:
            subprocess.run([code, '-r', *shown], check=False)
        print('No display available; figures written to', tmpdir)
    elif not save:
        plt.show()


def valid_cells(geo):
    """Mask of cells with at least 3 vertices (the rest are degenerate/boundary placeholders)."""
    return geo['cvVxP'][len(geo['cvX']):] >= 3


def main():
    here = os.getcwd()
    gm = os.path.join(here, 'b2fgmtry')
    if not os.path.exists(gm):
        gm = os.path.join(here, '..', 'baserun', 'b2fgmtry')
    ap = argparse.ArgumentParser(description='Plot plasmastate fields on the 2D B2.5 mesh.')
    ap.add_argument('ids', nargs='*', help="plasmastate suffixes, e.g. 0012, or 'latest' (default: all)")
    ap.add_argument('-l', '--latest', action='store_true', help='only the most recent plasmastate file')
    ap.add_argument('-q', '--quantity', default='te,ti',
                    help="comma-separated fields, NAME or NAME:IDX for species-resolved (default: te,ti)")
    ap.add_argument('--list', action='store_true', help='list the fields in the plasmastate file and exit')
    ap.add_argument('-d', '--dir', default=os.path.join(here, 'b2mn.exe.dir'))
    ap.add_argument('-g', '--geometry', default=gm)
    ap.add_argument('-o', '--out', default=os.path.join(here, 'plasmastate_2d'))
    ap.add_argument('--save', action='store_true', help='write PNG/npz to --out instead of showing figures')
    ap.add_argument('--lin', action='store_true', help='linear colour scale (default: log for positive fields)')
    ap.add_argument('--vmin', type=float, default=None)
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--zoom', nargs=4, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    a = ap.parse_args()

    allfiles = sorted(glob.glob(os.path.join(a.dir, 'plasmastate.*')),
                      key=lambda p: (len(p.rsplit('.', 1)[1]), p.rsplit('.', 1)[1]))
    if a.latest or 'latest' in a.ids:
        files = allfiles[-1:]
    else:
        files = allfiles if not a.ids else [os.path.join(a.dir, 'plasmastate.' + i) for i in a.ids]
    if not files:
        sys.exit(f'no plasmastate files found in {a.dir}')

    geo = read_cf(a.geometry, {'cvX', 'cvY', 'vxX', 'vxY', 'cvVxP', 'cvVx'})
    nCv = len(geo['cvX'])
    zamin = None
    if os.path.exists('b2fstate'):
        zamin = read_cf('b2fstate', {'zamin'}).get('zamin')

    if a.list:
        list_fields(read_plasmastate(files[0]), nCv, zamin)
        return
    specs = [s.strip() for s in a.quantity.split(',') if s.strip()]

    polys, ok = build_polygons(geo)
    sel = np.flatnonzero(ok)
    polys = [polys[i] for i in sel]

    tmpdir = setup_display(a.save, a.out)
    shown = []

    for fn in files:
        tag = fn.rsplit('.', 1)[1]
        ps = read_plasmastate(fn)
        t = float(ps['time'][0]) if 'time' in ps else np.nan
        qs = [get_quantity(ps, s, nCv, zamin) for s in specs]
        if a.save:
            np.savez(os.path.join(a.out, f'{"_".join(s.replace(":", "") for s in specs)}_{tag}.npz'),
                     **{s.replace(':', '_'): q[2] for s, q in zip(specs, qs)},
                     cvX=geo['cvX'], cvY=geo['cvY'], time=t)

        fig, axs = plt.subplots(1, len(qs), figsize=(5.5 * len(qs) + 1, 8), sharey=True,
                                constrained_layout=True, squeeze=False)
        axs = axs[0]
        print(f'{fn}: t={t:.4g} s')
        for ax, (label, unit, v) in zip(axs, qs):
            vs = v[sel]
            hi, lo = np.nanmax(vs), np.nanmin(vs)
            signed = lo < 0 < hi
            if signed:
                m = max(abs(lo), abs(hi))
                vmin = a.vmin if a.vmin is not None else -m
                vmax = a.vmax if a.vmax is not None else m
                norm = Normalize(vmin, vmax) if a.lin or vmin >= 0 else TwoSlopeNorm(0, vmin, vmax)
                cmap = 'RdBu_r'
            else:
                vmax = a.vmax if a.vmax is not None else hi
                pos = vs[vs > 0]
                vmin = a.vmin if a.vmin is not None else (
                    lo if a.lin or not pos.size else max(pos.min(), vmax * 1e-4))
                use_log = not a.lin and pos.size and vmin > 0 and vmax > vmin
                norm = LogNorm(vmin, vmax) if use_log else Normalize(vmin, vmax)
                cmap = 'inferno'
            pc = PolyCollection(polys, array=vs, norm=norm, cmap=cmap, edgecolors='none')
            ax.add_collection(pc)
            ax.set_aspect('equal')
            ax.set_xlim(*(a.zoom[:2] if a.zoom else (np.min(geo['vxX']), np.max(geo['vxX']))))
            ax.set_ylim(*(a.zoom[2:] if a.zoom else (np.min(geo['vxY']), np.max(geo['vxY']))))
            ax.set_title(f'{label} [{unit}]\n(plasmastate.{tag}, t={t:.4g} s)')
            xl = 'R [m]'
            for what, fnc, mk in (('max', np.argmax, '*'), ('min', np.argmin, 'v')):
                if what == 'min' and not signed:
                    continue
                i = sel[fnc(vs)]
                ax.plot(geo['cvX'][i], geo['cvY'][i], marker=mk, ms=14 if mk == '*' else 10,
                        mfc='cyan', mec='k', ls='none')
                xl += (f'\n{what} = {v[i]:.4g} at R = {geo["cvX"][i]:.3f} m, '
                       f'Z = {geo["cvY"][i]:.3f} m ({mk})')
                print(f'    {label:<22} {what} {v[i]:12.5g} {unit} at R = {geo["cvX"][i]:.4f} m, '
                      f'Z = {geo["cvY"][i]:.4f} m')
            ax.set_xlabel(xl)
            fig.colorbar(pc, ax=ax, shrink=0.8)
        axs[0].set_ylabel('Z [m]')
        if a.save or tmpdir:
            png = os.path.join(a.out if a.save else tmpdir,
                               f'{"_".join(s.replace(":", "") for s in specs)}_{tag}.png')
            fig.savefig(png, dpi=130)
            plt.close(fig)
            shown.append(png)
    show_figures(a.save, tmpdir, shown)


if __name__ == '__main__':
    main()
