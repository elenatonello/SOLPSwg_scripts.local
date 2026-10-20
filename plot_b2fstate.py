#!/usr/bin/env python3
"""2D maps of the final state of one or several SOLPS-ITER runs, with the direction of a flux or velocity field.

Works on structured (3.0.9: b2fgmtry + b2fstate) and unstructured (3.2.1: b2fgmtry + b2fstate) runs, also mixed in one
figure; the kind is taken from the geometry. Runs go side by side (columns), scalars one under the other (rows), and
the colour scale of a row is shared by all runs. A run can also be a single state file, e.g. the output of
generate_initialState_us.py or a b2fstati (-g gives its b2fgmtry). It replaces plot_b2fstate_us.py; for snapshots of a running job
(plasmastate.XXXX) use plot_plasmastate.py, which shares the same engine (the solpsplt package).

Scalar (-q)   any cell-centred block of the state: ne na te ti po ua tn kinrgy ...  (--list shows them).
              Species-resolved ones (na, ua, ...) take -s (default 1 = D+); several -s give one row each.
              Temperatures are shown in eV.
Vector (-v)   any face-based block: fluxes fna fhe fhi fch fne fni fna_mdf fna_32 fna_52 fna_eir ..., velocities
              vaecrb vadia wadia veecrb vedia uadia. Aliases: vecrb -> veecrb, vdia -> vedia. Species-resolved
              ones follow the species of each row (-s), or --vs if given.
--mode        how the vector is shown
                stream   streamlines over the scalar map (default): direction independent of the magnitude
                arrows   one fixed-length arrow per cell (every --stride-th), direction only
                quiver   arrows proportional to the vector (99th percentile drawn as --alen metres)
                faces    the faces themselves as coloured segments, signed value on a symmetric log scale
                div      net outflow per cell of the face flux (symmetric log scale)
              faces and div replace the scalar map (-q is ignored); --comp 0|1 keeps one face component.
The cell vector is the least-squares flux density (velocity) over the faces of the cell, see solpsplt/vectors.py;
velocity blocks of unstructured runs are treated like the fluxes and are not validated.

--source      auto (default): b2fstate, or for an unstructured run whose b2fstate is only the stub of a running job the
              latest plasmastate.NNNN;  b2fstate;  plasmastate (--snap NNNN);  aver (time-averaged scalars).

--line R0 Z0 R1 Z1   also plot the scalars along the straight line (R0, Z0) -> (R1, Z1) in a second figure, one panel per
              row, the runs overlaid: the line is marked on the maps. --line-species picks the species (default: all
              rows), --line-method linear (default, interpolation between cell centres) or cell (value of the cell),
              --line-n the number of points, --line-x R|Z|s the abscissa (default R for a horizontal line, Z for a
              vertical one, else the distance s). Gaps in a profile: the line leaves the mesh of that run.

Without -o the figure(s) are shown in a window, or opened in VS Code when there is no display (-o FILE.png writes
FILE.png and FILE_profile.png).

Examples:
    plot_b2fstate.py RUN_A RUN_B -q te -v fna                  # Te map + D+ particle-flux streamlines
    plot_b2fstate.py RUN -q na ua -s 4 -v fhi --zoom 0.6 1.0 -0.8 -0.3
    plot_b2fstate.py RUN -q ne -v vadia --vs 1 -o ne_vdia.png
    plot_b2fstate.py RUN -q te --wall                          # with the wall (structure.dat of the run)
    plot_b2fstate.py RUN -v fna --mode div                     # net D+ particle outflow per cell
    plot_b2fstate.py RUN -v fhi --mode faces --comp 1 -s 0
    plot_b2fstate.py b2fstate_new -g b2fgmtry_us -q ne te      # an explicit state file + geometry
    plot_b2fstate.py RUN_A RUN_B -q ua -s 1 --line 0.8 -0.6 1.0 -0.6   # D+ ua along a horizontal line
    plot_b2fstate.py RUN --list                                # blocks available in the state file
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from solpsplt import ALIASES, State, draw  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run directories, or state files (b2fstate, b2fstati, plasmastate.NNNN)')
    ap.add_argument('-g', '--geometry', help='b2fgmtry to use (default: the one of the run; needed for state files elsewhere)')
    ap.add_argument('--labels', nargs='*', help='titles of the runs')
    ap.add_argument('-q', '--quantity', nargs='*', default=['te'], help='scalar(s) to map (default te)')
    ap.add_argument('-s', '--species', nargs='*', type=int, default=[1],
                    help='species index (0-based) for species-resolved fields; several give one row each (default 1 = D+)')
    ap.add_argument('-v', '--vector', help='face-based block shown as streamlines / arrows / faces / divergence')
    ap.add_argument('--vs', type=int,
                    help='species of a species-resolved vector (default: the species of each row, i.e. of -s)')
    ap.add_argument('--mode', choices=['stream', 'arrows', 'quiver', 'faces', 'div'], default='stream')
    ap.add_argument('--comp', type=int, choices=(0, 1), help='faces/div: only this face component')
    ap.add_argument('--source', choices=['auto', 'b2fstate', 'plasmastate', 'aver'], default='auto')
    ap.add_argument('--snap', help="plasmastate/aver suffix, e.g. 0015 (default: latest)")
    ap.add_argument('--wall', nargs='?', const='auto', metavar='FILE',
                    help='draw the wall: the structure.dat of each run (looked for in the run, its baserun, and the family '
                         'directory), or FILE for all runs')
    ap.add_argument('--zoom', nargs=4, type=float, metavar=('R0', 'R1', 'Z0', 'Z1'))
    ap.add_argument('--log', action='store_true', help='logarithmic colour scale')
    ap.add_argument('--lin', action='store_true', help='linear colour scale (default for all but ne, na)')
    ap.add_argument('--vmin', type=float)
    ap.add_argument('--vmax', type=float, help='upper colour limit (faces/div: symmetric limit)')
    ap.add_argument('--clip', type=float, default=99.5, help='colour limit = this percentile over the row (default 99.5)')
    ap.add_argument('--cmap')
    ap.add_argument('--density', type=float, default=1.0, help='stream: density of the streamlines (default 1)')
    ap.add_argument('--lw', type=float, default=0.6, help='stream: line width (default 0.6)')
    ap.add_argument('--lw-flux', action='store_true', help='stream: line width grows with log |vector|')
    ap.add_argument('--color', default='k', help='stream/arrows colour (default k)')
    ap.add_argument('--res', type=float, default=0.003, help='stream: grid spacing in m (default 0.003)')
    ap.add_argument('--stride', type=int, default=2, help='arrows/quiver: every N-th cell (default 2)')
    ap.add_argument('--alen', type=float, default=0.03, help='arrows/quiver: arrow length in m (default 0.03)')
    ap.add_argument('--line', nargs=4, type=float, metavar=('R0', 'Z0', 'R1', 'Z1'),
                    help='scalars along the straight line (R0, Z0) -> (R1, Z1), in a second figure (see above)')
    ap.add_argument('--line-species', nargs='*', type=int, help='species shown in the line profiles (default: all rows)')
    ap.add_argument('--line-method', choices=['linear', 'cell'], default='linear')
    ap.add_argument('--line-n', type=int, default=200, help='points along the line (default 200)')
    ap.add_argument('--line-x', choices=['auto', 'R', 'Z', 's'], default='auto', help='abscissa of the line profiles')
    ap.add_argument('--list', action='store_true', help='list the scalar and face blocks of each run and exit')
    ap.add_argument('-o', '--output', help='save to this image (png/pdf) instead of showing it')
    # accept negative numbers in exponent notation as values (--vmin -2e4), not as options
    ap._negative_number_matcher = re.compile(r'^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')
    a = ap.parse_args()

    tmpdir = None
    if a.output:
        matplotlib.use('Agg')
    elif not a.list:
        tmpdir = draw.setup_display(False, None, prefix='plot_b2fstate_')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                                'savefig.dpi': 200, 'savefig.bbox': 'tight'})

    # ------------------------------------------------------------------ open the runs
    states = []
    for i, p in enumerate(a.runs):
        label = a.labels[i] if a.labels and i < len(a.labels) else Path(p).resolve().name
        try:
            st = State.from_run(p, a.source, a.snap, a.geometry)
            st.label = label
            if a.source == 'auto' and st.source != 'b2fstate':
                print(f'note: {label}: no complete b2fstate, using {st.file.name}')
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f'WARNING {label}: {e}')
            st = None
        states.append((label, st))

    if a.list:
        for label, st in states:
            if st is None:
                continue
            k = st.catalog()
            print(f'== {label}  [{st.mesh.kind}]  {st.file}')
            print('   scalars (species-resolved marked *): ' + ' '.join(n + ('*' if s else '') for n, s in k['scalar'].items()))
            print('   face blocks (species-resolved marked *): ' + ' '.join(n + ('*' if s else '') for n, s in k['vector'].items()))
        return
    live = [st for _, st in states if st is not None]
    if not live:
        sys.exit('no run could be opened')

    vname = ALIASES.get(a.vector, a.vector) if a.vector else None
    vs = a.vs if a.vs is not None else a.species[0]
    face_mode = a.mode in ('faces', 'div')
    if face_mode and not vname:
        ap.error(f'--mode {a.mode} needs -v')
    for st in live:                                       # one pass over each (large) text file
        want = ([] if face_mode else list(a.quantity)) + ([vname] if vname else [])
        try:
            st.load(want)
        except KeyError:
            pass

    def resolved(name, kind):
        for st in live:
            c = st.catalog()[kind].get(name)
            if c is not None:
                return c
        return None

    # ------------------------------------------------------------------ rows
    if face_mode:
        res = resolved(vname, 'vector')
        rows = [(vname, s) for s in (a.species if res else [vs])]
    else:
        rows, seen = [], set()
        for q in a.quantity:
            res = resolved(q, 'scalar')
            for s in a.species:
                key = (q, s if res else None)
                if key not in seen:
                    seen.add(key)
                    rows.append((q, s))
    nr, nc = len(rows), len(states)
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc + 1.3, 4.5 * nr), squeeze=False, constrained_layout=True)
    drawn = 0

    walls = {}                                           # label -> structures
    if a.wall:
        from solpsplt.wall import extent as wall_extent
        for label, st in states:
            if st is None:
                continue
            w, wf = st.wall(None if a.wall == 'auto' else a.wall)
            if w is None:
                print(f'WARNING {label}: no structure.dat found, no wall drawn')
            else:
                walls[label] = w
                print(f'wall of {label}: {wf}')
    ext = [st.mesh.cell_extent for st in live] + [wall_extent(w) for w in walls.values()]            # one common view for all runs, so panels are comparable
    pad = 0.01 if walls else 0.0                          # keep the wall lines inside the axes frame
    common = (min(e[0] for e in ext) - pad, max(e[1] for e in ext) + pad,
              min(e[2] for e in ext) - pad, max(e[3] for e in ext) + pad)

    def finish_axes(ax, st, i, j, bounds):
        ax.set_aspect('equal')
        ax.set_xlabel('R [m]' if i == nr - 1 else '')
        ax.set_ylabel('Z [m]' if j == 0 else '')
        R0, R1, Z0, Z1 = a.zoom if a.zoom else common
        ax.set_xlim(R0, R1)
        ax.set_ylim(Z0, Z1)

    for i, (name, s) in enumerate(rows):
        # data of the row, one entry per run
        data = []
        for label, st in states:
            if st is None:
                data.append(None)
                continue
            try:
                if face_mode:
                    F = st.faces(name, s, a.comp)
                    data.append(F.segments(a.comp) if a.mode == 'faces' else F.divergence())
                else:
                    data.append(st.scalar(name, s))
            except (KeyError, ValueError) as e:
                print(f'WARNING {label}: {name}: {e}')
                data.append(None)
        good = [d for d in data if d is not None]
        if not good:
            continue
        drawn += 1
        if face_mode:
            vals = [d[1] if a.mode == 'faces' else d for d in good]
            vmax = a.vmax or max(np.nanmax(np.abs(v)) for v in vals)
        else:
            norm, cmap = draw.pick_norm(good, name, a.log, a.lin, a.vmin, a.vmax, a.clip, a.cmap)
        mappable = None
        for j, ((label, st), d) in enumerate(zip(states, data)):
            ax = axs[i][j]
            if d is None:
                ax.set_title(f'{label}', fontsize=9)
                ax.text(0.5, 0.5, 'no valid data', ha='center', va='center', transform=ax.transAxes, color='firebrick')
                ax.set_axis_off()
                continue
            m = st.mesh
            resolved_here = (st.catalog()['vector' if face_mode else 'scalar'].get(name))
            sp = f' {st.species_name(s)}' if resolved_here else ''
            tag = f'{name}{sp}' + (f' ({a.mode})' if face_mode else '')
            src = '' if st.source == 'b2fstate' else f', {st.file.name}'
            ax.set_title(f'{label} ({m.kind[:6]}.{src})\n{tag}', fontsize=9)
            if face_mode:
                if a.mode == 'faces':
                    mappable = draw.draw_faces(ax, d[0], d[1], vmax)
                else:
                    mappable = draw.draw_divergence(ax, m.polys, d, vmax)
            else:
                mappable = draw.draw_cells(ax, m.polys, d, norm, cmap)
                if vname:
                    try:
                        vec = st.faces(vname, a.vs if a.vs is not None else s).cell_vectors()
                        if not np.nanmax(np.linalg.norm(vec, axis=1)) > 0:
                            print(f"WARNING {label}: '{vname}' is zero everywhere (drifts off?), no overlay")
                        elif a.mode == 'stream':
                            draw.draw_streams(ax, m.polys, m.centres, vec, a.zoom, a.density, a.lw, a.lw_flux,
                                              a.color, a.res)
                        else:
                            shape = (m.nx, m.ny) if m.kind == 'structured' else None
                            draw.draw_arrows(ax, m.centres, vec, shape, a.stride, a.alen, a.mode == 'quiver',
                                             color=a.color)
                    except (KeyError, ValueError) as e:
                        print(f'WARNING {label}: no overlay ({e})')
            if a.line:
                ax.plot([a.line[0], a.line[2]], [a.line[1], a.line[3]], '--', color='#d81b60', lw=1.5, zorder=6)
            if label in walls:
                draw.draw_wall(ax, walls[label])
            finish_axes(ax, st, i, j, m.extent)
        if mappable is not None:
            unit = {'te': 'eV', 'ti': 'eV', 'tn': 'eV', 'ua': 'm/s', 'ne': 'm^-3', 'na': 'm^-3', 'po': 'V'}.get(name, '')
            unit = f'[{unit}]' if unit else ''
            fig.colorbar(mappable, ax=[x for x in axs[i] if x.axison], shrink=0.8, pad=0.02,
                         label=f'{name} {unit}'.strip())

    if not drawn:
        sys.exit('nothing to plot: no run has valid data for the requested quantities (see --list)')
    if vname and not face_mode:
        st0 = live[0]
        res = st0.catalog()['vector'].get(vname)
        vl = vname + ('' if not res else (f' {st0.species_name(a.vs)}' if a.vs is not None else ' (species of each row)'))
        what = {'stream': 'streamlines', 'arrows': 'direction arrows', 'quiver': 'arrows ~ vector'}[a.mode]
        fig.supxlabel(f'{what}: {vl} (cell-centred vector, see --help)', fontsize=7, color='grey')

    figures = [(fig, 'plot_b2fstate.png', a.output)]
    if a.line and not face_mode:
        pf = line_figure(plt, states, live, rows, a, resolved)
        if pf is not None:
            out = None
            if a.output:
                o = Path(a.output)
                out = str(o.with_name(o.stem + '_profile' + (o.suffix or '.png')))
            figures.append((pf, 'plot_b2fstate_profile.png', out))
    elif a.line:
        print('note: --line is ignored for --mode faces/div')

    if a.output:
        for f, _, out in figures:
            f.savefig(out)
            print('wrote', out)
    elif tmpdir:
        paths = []
        for f, name, _ in figures:
            paths.append(str(Path(tmpdir) / name))
            f.savefig(paths[-1])
        draw.show_figures(False, tmpdir, paths)
    else:
        plt.show()


def line_figure(plt, states, live, rows, a, resolved):
    """Figure with the scalars of the rows (all runs overlaid) along the line a.line."""
    from solpsplt.sample import line_profile
    R0, Z0, R1, Z1 = a.line
    prow = [(q, s) for q, s in rows if a.line_species is None or not resolved(q, 'scalar') or s in a.line_species]
    if not prow:
        print('note: --line-species selects no row, no line profile')
        return None
    xsel = a.line_x if a.line_x != 'auto' else ('R' if Z0 == Z1 else 'Z' if R0 == R1 else 's')
    xlabel = {'R': 'R [m]', 'Z': 'Z [m]', 's': f'distance along the line from (R, Z) = ({R0:g}, {Z0:g}) [m]'}[xsel]
    units = {'te': 'eV', 'ti': 'eV', 'tn': 'eV', 'ua': 'm/s', 'ne': 'm$^{-3}$', 'na': 'm$^{-3}$', 'po': 'V'}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    pf, axs = plt.subplots(len(prow), 1, figsize=(6.4, 3.3 * len(prow)), squeeze=False, sharex=True,
                           constrained_layout=True)
    for k, (name, s) in enumerate(prow):
        ax = axs[k][0]
        signed = False
        for j, (label, st) in enumerate(states):
            if st is None:
                continue
            try:
                vals = st.scalar(name, s)
            except (KeyError, ValueError) as e:
                print(f'WARNING {label}: {name}: {e}')
                continue
            sd, pts, v = line_profile(st.mesh, vals, (R0, Z0), (R1, Z1), a.line_n, a.line_method)
            x = {'R': pts[:, 0], 'Z': pts[:, 1], 's': sd}[xsel]
            ax.plot(x, v, color=colors[j % len(colors)], label=label)
            signed |= bool(np.nanmin(v) < 0 < np.nanmax(v)) if np.isfinite(v).any() else False
            if k == 0:
                print(f'line profile, {label}: {np.isfinite(v).mean() * 100:.0f}% of the line is inside the mesh')
        if signed:
            ax.axhline(0, color='grey', lw=0.7, zorder=0)
        sp = f' {live[0].species_name(s)}' if resolved(name, 'scalar') else ''
        ax.set_ylabel(f"{name}{sp} [{units[name]}]" if name in units else f'{name}{sp}')
        ax.grid(alpha=0.25)
        ax.set_xlim(*sorted((x[0], x[-1])))
    axs[0][0].legend(frameon=False)
    axs[-1][0].set_xlabel(xlabel)
    pf.suptitle(f'along the line ({R0:g}, {Z0:g}) -> ({R1:g}, {Z1:g})', fontsize=10)
    return pf

if __name__ == '__main__':
    main()
