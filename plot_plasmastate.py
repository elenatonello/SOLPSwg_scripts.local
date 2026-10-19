#!/usr/bin/env python3
"""Plot cell-centred quantities from plasmastate.XXXX files on the 2D B2.5 mesh.

plasmastate files are Fortran unformatted sequential records:
    record 0 : version string
    then per field: header record ('*cf: ', type, count, name), data record (little-endian)
Cell-centred fields (te, ti, ne, po, ...) have nCv values; species-resolved ones (na, ua,
kinrgy, ...) have ns*nCv values and are selected as NAME:IDX (species index, from 0).
Face/other fields cannot be drawn on the cell map and are only shown by --list.
The cell polygons come from the b2fgmtry file (cvVxP/cvVx/vxX/vxY).

Run from the run directory. Defaults: plasmastate files in ./b2mn.exe.dir (-d), geometry from
./b2fgmtry or ../baserun/b2fgmtry (-g). Figures are opened in windows (TkAgg, like the other
scripts in scripts.local); use --save to write PNG + npz files (the npz holds the plotted values and
the cell centres; PNG only with --limits) to ./plasmastate_2d/ (-o) instead.
Without a display the figures are written to a temporary directory and opened in VS Code (`code`).
Species charges (zamin, shown next to the species index) are read from ./b2fstate when it exists;
--limits also looks in b2mn.exe.dir/b2fstate and uses the masses am from there. Without a b2fstate
species are only shown by index.
Plotting details: temperatures are converted from J to eV; the colour scale is logarithmic for
positive fields (down to 1e-4 of the maximum unless --vmin is given; --lin for linear) and
symmetric about 0 for fields of both signs; a star marks the maximum and, for signed fields, a
triangle the minimum, with value and position printed under the plot and in the terminal. Only cells
with a polygon (>= 3 vertices) in b2fgmtry are drawn and searched: the interior cells (on the grids
checked, guard cells have none).

Usage:
    plot_plasmastate.py --list                  # fields available in the plasmastate file
    plot_plasmastate.py                         # Te and Ti, all snapshots (one window each)
    plot_plasmastate.py 0011                    # a single snapshot
    plot_plasmastate.py -l                      # the most recent snapshot (or: plot_plasmastate.py latest)
    plot_plasmastate.py -q te,ti,po 0011        # chosen quantities side by side
    plot_plasmastate.py -q na:1,ua:1 0011       # species-resolved: density and parallel velocity of species 1
    plot_plasmastate.py -q ne --lin --vmax 1e19 --zoom 0.62 0.8 -0.45 -0.2 0011
    plot_plasmastate.py --save                  # write PNG + npz instead of showing
    plot_plasmastate.py --limits -q te,ti,na:1,ua:1,po -l    # where each quantity sits on a limit of the code
    plot_plasmastate.py --limits -q te --tol 1e-6 --b2mn ../b2mn.dat 0011

--limits draws, for every -q quantity, a map of the cells whose value is within --tol (relative, default
1e-3) of a min (blue) or max (red) limit that B2.5 enforces. The limits are read from b2mn.dat
(./b2mn.dat, or --b2mn), with the code defaults for anything not set there. If the quantity has a limit
that is never reached the map is grey and says so in the title (with the closest value reached below the
map); a quantity without any limit is flagged "no limit". The limits known:
    te, ti, tn   b2upht_{te,ti,tn}_min / _max [eV]  (te also b2upht_rte_min*Ti; tn only if
                 b2mn_tn_style = 2, otherwise tn = ti and is not solved)
    na:IDX       b2mndr_na_min / _max [m^-3]
    ua:IDX       b2mndr_ua_max, and b2npmo_ion_vlct_restrict_M * sqrt(pz/rz) when
                 b2npmo_ion_vlct_restrict is on (charged species only; needs b2fstate for am)
    po           b2nppo_restr_po: |po| <= restr * Te[eV]   (only po < 0 if restr < 0)
Any other field (ne, kinrgy, ...) has no limit. The region-dependent min_na, the _core and _abs/_ua
variants of the velocity restriction and the closed-surface exemption of the po limit are not handled.
A cell sitting on a limit is not necessarily wrong, but it means the equation is not being solved there.
"""
import argparse
import glob
import os
import re
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


MP = 1.67262192369e-27   # proton mass [kg]

# parameter -> code default (modules/B2.5/src/modules/b2mod_switches.F), for what b2mn.dat does not set
LIMIT_DEFAULTS = {
    'b2upht_te_min': 0.1, 'b2upht_te_max': 1e30, 'b2upht_rte_min': 0.0,
    'b2upht_ti_min': 0.1, 'b2upht_ti_max': 1e30,
    'b2upht_tn_min': 0.1, 'b2upht_tn_max': 1e30,
    'b2mndr_na_min': 1e4, 'b2mndr_na_max': 1e30, 'b2mndr_ua_max': 1e6,
    'b2npmo_ion_vlct_restrict': 0, 'b2npmo_ion_vlct_restrict_M': 3.0,
    'b2nppo_restr_po': 0.0, 'b2mn_tn_style': 0,
}


def read_b2mn(fn):
    """Numeric parameters of a b2mn.dat ('name' 'value' # comment lines) that are in LIMIT_DEFAULTS."""
    par = dict(LIMIT_DEFAULTS)
    if not os.path.exists(fn):
        print(f'note: {fn} not found, using the code defaults for all limits')
        return par
    with open(fn) as f:
        for line in f:
            m = re.match(r"\s*'([^']+)'\s+'([^']*)'", line)
            if m and m.group(1) in par:
                try:
                    par[m.group(1)] = float(m.group(2).replace('d', 'e').replace('D', 'e'))
                except ValueError:
                    pass
    return par


def find_b2fstate(here, dirname):
    for p in (os.path.join(here, 'b2fstate'), os.path.join(dirname, 'b2fstate')):
        if os.path.exists(p):
            return p
    return None


def quantity_limits(spec, ps, nCv, par, am, zamin, tol, valid):
    """Limits that B2.5 enforces on the quantity `spec` (-q syntax) and the cells sitting on them.

    Returns (label, unit, limits, note). limits is a list of (kind, text, mask, extreme), kind 'min'
    or 'max', mask the cells within tol (relative) of the limit, extreme a text with the value closest
    to it. limits is empty when the quantity has none, and note then says why.
    """
    name, _, idx = spec.partition(':')
    label, unit, v = get_quantity(ps, spec, nCv, zamin)
    short = label.split(',')[0] + ']' if ',' in label else label     # na[1, zamin=1] -> na[1]
    out = []

    def add(kind, text, val, lim):
        """One limit; lim is a number or an array. Not set (<= 0 for a min, >= 1e29 for a max) is skipped."""
        if np.ndim(lim) == 0 and (lim >= 1e29 if kind == 'max' else lim <= 0):
            return
        mask = valid & (np.abs(val - lim) <= tol * np.abs(lim))
        ext = np.nanmin(val[valid]) if kind == 'min' else np.nanmax(val[valid])
        out.append((kind, text, mask, f'{"min" if kind == "min" else "max"} reached {ext:.4g}'))

    if name in ('te', 'ti') or (name == 'tn' and par['b2mn_tn_style'] == 2):
        lo, hi = par[f'b2upht_{name}_min'], par[f'b2upht_{name}_max']
        if name == 'te' and par['b2upht_rte_min'] > 0:       # rte_min*Ti raises the floor
            lo = np.maximum(lo, par['b2upht_rte_min'] * ps['ti'][:nCv] / EV)
        add('min', f'{short} >= {lo if np.ndim(lo) == 0 else "rte_min*Ti"} {unit}', v, lo)
        add('max', f'{short} <= {hi:g} {unit}', v, hi)
    elif name == 'tn':
        return label, unit, out, f'tn is not solved (b2mn_tn_style={par["b2mn_tn_style"]}: tn = ti)'
    elif name == 'na':
        add('min', f'{short} >= {par["b2mndr_na_min"]:g} {unit}', v, par['b2mndr_na_min'])
        add('max', f'{short} <= {par["b2mndr_na_max"]:g} {unit}', v, par['b2mndr_na_max'])
    elif name == 'ua':
        i = int(idx) if idx else 0
        add('max', f'|{short}| <= {par["b2mndr_ua_max"]:g} {unit}', np.abs(v), par['b2mndr_ua_max'])
        if par['b2npmo_ion_vlct_restrict'] != 0 and (zamin is None or zamin[i] != 0):
            na = ps['na'].reshape(-1, nCv)
            ns = na.shape[0]
            if am is None or len(am) != ns:
                print('note: no b2fstate with am found, the Mach limit on ua is skipped')
            else:
                charged = np.asarray(zamin) != 0 if zamin is not None else np.ones(ns, bool)
                pz = ps['ne'][:nCv] * ps['te'][:nCv] + (na[charged] * ps['ti'][:nCv]).sum(axis=0)
                rz = (MP * np.asarray(am)[charged, None] * na[charged]).sum(axis=0)
                M = par['b2npmo_ion_vlct_restrict_M']
                add('max', f'|{short}| <= {M:g}*sqrt(pz/rz)', np.abs(v), M * np.sqrt(pz / rz))
    elif name == 'po':
        r = par['b2nppo_restr_po']
        if r != 0:
            ratio = v / np.maximum(ps['te'][:nCv] / EV, 1e-30)           # po / Te[eV]
            mask = valid & (np.abs(ratio) >= abs(r) * (1 - tol)) & ((ratio < 0) if r < 0 else True)
            ext = np.nanmax(np.abs(ratio[valid]))
            out.append(('max', f'|po| <= {abs(r):g}*Te[eV]' + (' (po<0 only)' if r < 0 else ''),
                        mask, f'max reached {ext:.4g}'))
    if not out and not note_for(name):
        return label, unit, out, 'no limit set in b2mn.dat/defaults'
    return label, unit, out, note_for(name) if not out else None


def note_for(name):
    """Why a field with no limit has none (None when it can have one)."""
    if name in ('te', 'ti', 'tn', 'na', 'ua', 'po'):
        return None
    return 'no limit in the code for this field'


def plot_limits(fn, ps, geo, polys, sel, specs, par, am, zamin, tol, a, tmpdir):
    """One map per -q quantity: cells on a min (blue) / max (red) limit; says so when none is reached."""
    from matplotlib.colors import ListedColormap
    nCv = len(geo['cvX'])
    tag = fn.rsplit('.', 1)[1]
    t = float(ps['time'][0]) if 'time' in ps else np.nan
    valid = np.zeros(nCv, bool)
    valid[sel] = True
    cmap = ListedColormap(['#dddddd', '#2166ac', '#b2182b', '#762a83'])    # none, min, max, both
    fig, axs = plt.subplots(1, len(specs), figsize=(5.5 * len(specs) + 1, 8), sharey=True,
                            constrained_layout=True, squeeze=False)
    print(f'{fn}: t={t:.4g} s, cells within {tol:g} (relative) of a limit')
    for ax, spec in zip(axs[0], specs):
        label, unit, limits, note = quantity_limits(spec, ps, nCv, par, am, zamin, tol, valid)
        c = np.zeros(nCv, int)
        xl = 'R [m]'
        for kind_, text, mask, ext in limits:
            n = int(mask.sum())
            c[mask] |= 1 if kind_ == 'min' else 2          # bit 0: on a minimum, bit 1: on a maximum
            if n:
                R, Z = geo['cvX'][mask], geo['cvY'][mask]
                line = f'{text}: {n} cells, R {R.min():.3f}..{R.max():.3f}, Z {Z.min():.3f}..{Z.max():.3f}'
                xl += f'\n{text}: {n} cells'
            else:
                line = f'{text}: never reached ({ext})'
                xl += f'\n{text}: never reached ({ext})'
            print(f'    {label:<12} {line}')
        if not limits:
            print(f'    {label:<12} {note}')
            xl += f'\n{note}'
            head = f'{label}: no limit'
        elif not c.any():
            head = f'{label}: limit never reached'
        else:
            head = f'{label}: on a limit in {int((c > 0).sum())} cells'
        ax.add_collection(PolyCollection(polys, array=c[sel].astype(float), cmap=cmap, clim=(-0.5, 3.5),
                                         edgecolors='none'))
        ax.set_aspect('equal')
        ax.set_xlim(*(a.zoom[:2] if a.zoom else (np.min(geo['vxX']), np.max(geo['vxX']))))
        ax.set_ylim(*(a.zoom[2:] if a.zoom else (np.min(geo['vxY']), np.max(geo['vxY']))))
        ax.set_title(f'{head}\n(plasmastate.{tag}, t={t:.4g} s)')
        ax.set_xlabel(xl)
    axs[0][0].set_ylabel('Z [m]')
    if a.save or tmpdir:
        png = os.path.join(a.out if a.save else tmpdir,
                           f'limits_{"_".join(s.replace(":", "") for s in specs)}_{tag}.png')
        fig.savefig(png, dpi=130)
        plt.close(fig)
        return png
    return None


def main():
    here = os.getcwd()
    gm = os.path.join(here, 'b2fgmtry')
    if not os.path.exists(gm):
        gm = os.path.join(here, '..', 'baserun', 'b2fgmtry')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ids', nargs='*', help="plasmastate suffixes, e.g. 0012, or 'latest' (default: all)")
    ap.add_argument('-l', '--latest', action='store_true', help='only the most recent plasmastate file')
    ap.add_argument('-q', '--quantity', default='te,ti',
                    help="comma-separated fields, NAME or NAME:IDX for species-resolved (default: te,ti)")
    ap.add_argument('--list', action='store_true', help='list the fields in the plasmastate file and exit')
    ap.add_argument('-d', '--dir', default=os.path.join(here, 'b2mn.exe.dir'),
                    help='directory with the plasmastate.XXXX files (default: ./b2mn.exe.dir)')
    ap.add_argument('-g', '--geometry', default=gm,
                    help='b2fgmtry of the (unstructured) grid (default: ./b2fgmtry, else ../baserun/b2fgmtry)')
    ap.add_argument('-o', '--out', default=os.path.join(here, 'plasmastate_2d'),
                    help='output directory for --save (default: ./plasmastate_2d)')
    ap.add_argument('--save', action='store_true', help='write PNG/npz to --out instead of showing figures')
    ap.add_argument('--lin', action='store_true', help='linear colour scale (default: log for positive fields)')
    ap.add_argument('--vmin', type=float, default=None, help='lower colour limit, in the units shown in the title')
    ap.add_argument('--vmax', type=float, default=None, help='upper colour limit, in the units shown in the title')
    ap.add_argument('--zoom', nargs=4, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                    help='plot window in m: R range then Z range (default: whole mesh)')
    ap.add_argument('--limits', action='store_true',
                    help='for each -q quantity, plot where it sits on a min/max limit of the code')
    ap.add_argument('--b2mn', default=os.path.join(here, 'b2mn.dat'),
                    help='b2mn.dat to read the limits from (default: ./b2mn.dat)')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='relative distance to a limit that counts as on it (default: 1e-3)')
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

    if a.limits:
        par = read_b2mn(a.b2mn)
        print('limits: ' + ', '.join(f'{k}={par[k]:g}' for k in LIMIT_DEFAULTS))
        sf = find_b2fstate(here, a.dir)
        st = read_cf(sf, {'am', 'zamin'}) if sf else {}
        for fn in files:
            png = plot_limits(fn, read_plasmastate(fn), geo, polys, sel, specs, par, st.get('am'),
                              st.get('zamin'), a.tol, a, tmpdir)
            if png:
                shown.append(png)
        show_figures(a.save, tmpdir, shown)
        return

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
