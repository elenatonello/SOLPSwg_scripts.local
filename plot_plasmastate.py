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

Face-based blocks (fluxes fhi, fhe, fch, fna:IDX, ... and velocities vaecrb:IDX, ...) can be shown in three ways:
    --faces            every face as a coloured line segment (colour = signed face value; fluxes are integrated over
                       the face, fhi/fhe in W). Also the default when -q names only face-based blocks.
    --div              net flux OUT of each cell (sum over its faces; > 0: net outflow), on the cell polygons
    -v NAME[:IDX]      streamlines of the cell-averaged vector of a face block on top of the -q maps
Layout, as used by b2ursd: file order [all faces comp 0][all faces comp 1], face value = comp0 + comp1 (--comp 0|1
keeps one), positive from cell fcCv(f,1) to fcCv(f,2). Signed colours on a symmetric log scale (linear within +-1e-6 of
the largest |value|, or of --vmax); the terminal shows the largest |value| and its face/cell index (0-based).
    plot_plasmastate.py -q fhi,fhe 0011                 # faces (default for a face block)
    plot_plasmastate.py -q fhi --div 0011               # net outflow of fhi per cell
    plot_plasmastate.py -q fna:1 --comp 1 0011          # only component 1 of the D+ particle flux
    plot_plasmastate.py -q te -v fna:1 0011             # Te map with D+ particle-flux streamlines
--wall [FILE] draws the wall (structure.dat) on the maps and face views.
(plot_plasmastate_flux.py was merged into this script; plot_b2fstate.py shows the same for finished runs.)

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
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from solpsplt import build_polygons, read_cf, read_plasmastate, valid_cells  # noqa: F401  (re-exported: other scripts import them from here)
from solpsplt.draw import setup_display, show_figures  # noqa: F401

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
    ap.add_argument('-q', '--quantity', default=None,
                    help="comma-separated fields, NAME or NAME:IDX for species-resolved (default: te,ti; fhi with --faces/--div)")
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
    ap.add_argument('--wall', nargs='?', const='auto', metavar='FILE',
                    help='draw the wall: the structure.dat found next to the geometry (or in the current directory), or FILE')
    ap.add_argument('--faces', action='store_true', help='draw face blocks (fhi, fna:IDX, ...) as coloured face segments')
    ap.add_argument('--div', action='store_true', help='net outflow per cell of face blocks instead of face segments')
    ap.add_argument('--comp', type=int, choices=(0, 1), default=None,
                    help='face blocks: use only this flux component (default: comp0 + comp1)')
    ap.add_argument('-v', '--vector', default=None, metavar='NAME[:IDX]',
                    help='overlay streamlines of this face block (e.g. fna:1) on the -q maps')
    ap.add_argument('--density', type=float, default=1.0, help='streamline density for -v (default: 1)')
    ap.add_argument('--limits', action='store_true',
                    help='for each -q quantity, plot where it sits on a min/max limit of the code')
    ap.add_argument('--b2mn', default=os.path.join(here, 'b2mn.dat'),
                    help='b2mn.dat to read the limits from (default: ./b2mn.dat)')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='relative distance to a limit that counts as on it (default: 1e-3)')
    # accept negative numbers in exponent notation as values (--vmin -2e4), not as options
    ap._negative_number_matcher = re.compile(r'^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')
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
    face_mode = a.faces or a.div
    if a.quantity is None:
        a.quantity = 'fhi' if face_mode else 'te,ti'
    specs = [s.strip() for s in a.quantity.split(',') if s.strip()]
    if not face_mode and not a.limits:
        # a -q that names only face-based blocks (fhi, fna:1, ...) is shown as faces, as plot_plasmastate_flux.py did
        ps0 = read_plasmastate(files[0])
        names = [x.partition(':')[0] for x in specs]
        face_mode = a.faces = all(n in ps0 and len(ps0[n]) > nCv and kind(n, ps0[n], nCv) is None for n in names)

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

    wall = None
    if a.wall:
        from solpsplt import find_structure, read_structures
        from solpsplt.draw import draw_wall
        wf = a.wall if a.wall != 'auto' else (find_structure(os.path.dirname(os.path.abspath(a.geometry)))
                                              or find_structure(here))
        if wf is None:
            print('WARNING: no structure.dat found (use --wall FILE), no wall drawn')
        else:
            wall = read_structures(wf)
            print(f'wall: {wf}')

    mesh = None                                      # solpsplt mesh, only needed for face blocks / streamlines
    if face_mode or a.vector:
        from solpsplt import State, open_mesh
        from solpsplt.draw import draw_divergence, draw_faces, draw_streams
        mesh = open_mesh(a.geometry)

    def face_field(ps, spec):
        name, _, idx = spec.partition(':')
        try:
            return name + (f'[{idx}]' if idx else ''), State.from_blocks(mesh, ps).faces(name, int(idx or 0), a.comp)
        except KeyError as e:
            sys.exit(f"{e.args[0]} (see --list)")

    if face_mode:
        for fn in files:
            tag = fn.rsplit('.', 1)[1]
            ps = read_plasmastate(fn)
            t = float(ps['time'][0]) if 'time' in ps else np.nan
            fig, axs = plt.subplots(1, len(specs), figsize=(5.5 * len(specs) + 1, 8), sharey=True,
                                    constrained_layout=True, squeeze=False)
            for ax, spec in zip(axs[0], specs):
                label, F = face_field(ps, spec)
                if a.div:
                    vs, title = F.divergence(), f'net outflow of {label}'
                    col = draw_divergence(ax, polys, vs, a.vmax)
                else:
                    segs, vs = F.segments(a.comp)
                    title = f'{label} (+: cell1 -> cell2)'
                    col = draw_faces(ax, segs, vs, a.vmax)
                if wall:
                    draw_wall(ax, wall)
                ax.set_aspect('equal')
                ax.set_xlim(*(a.zoom[:2] if a.zoom else mesh.extent[:2]))
                ax.set_ylim(*(a.zoom[2:] if a.zoom else mesh.extent[2:]))
                ax.set_title(f'{title}\n(plasmastate.{tag}, t={t:.4g} s)')
                ax.set_xlabel('R [m]')
                fig.colorbar(col, ax=ax, shrink=0.8)
                i = int(np.argmax(np.abs(vs)))
                print(f'{fn} {title}: |max| = {vs[i]:.4g} at index {i}')
            axs[0][0].set_ylabel('Z [m]')
            if a.save or tmpdir:
                png = os.path.join(a.out if a.save else tmpdir,
                                   f'{"_".join(x.replace(":", "") for x in specs)}'
                                   f'{"_div" if a.div else "_faces"}_{tag}.png')
                fig.savefig(png, dpi=130)
                plt.close(fig)
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
            if wall:
                draw_wall(ax, wall)
            if a.vector:
                vec = face_field(ps, a.vector)[1].cell_vectors()
                draw_streams(ax, polys, mesh.centres, vec, a.zoom, a.density)
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
