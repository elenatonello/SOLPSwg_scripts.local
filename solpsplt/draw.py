"""Drawing helpers shared by plot_b2fstate.py and plot_plasmastate.py.

Display      setup_display / show_figures   interactive window if possible, else PNGs opened in VS Code
Scalars      pick_norm, draw_cells
Vectors      draw_streams, draw_arrows      overlays of cell vectors (see vectors.FaceField.cell_vectors)
Faces        draw_faces, draw_divergence    signed face values / net outflow per cell on a symlog scale
"""
import os
import shutil
import subprocess
import tempfile

import matplotlib
import numpy as np

from .wall import draw_wall  # noqa: F401  (drawn with the other overlays)


# ---------------------------------------------------------------- display
def setup_display(save, out, prefix='plasmastate_'):
    """Choose the matplotlib backend; return a temp dir if figures must go through files for VS Code."""
    if save:
        matplotlib.use('Agg')
        os.makedirs(out, exist_ok=True)
        return None
    # interactive window if Tk can open one, otherwise files that are opened in VS Code
    try:
        matplotlib.use('TkAgg', force=True)
        import tkinter
        tkinter.Tk().destroy()
        return None
    except Exception:
        matplotlib.use('Agg', force=True)
        return tempfile.mkdtemp(prefix=prefix)


def show_figures(save, tmpdir, shown):
    """Show the figures: window (interactive backend), or, when they had to be written to tmpdir, open them in VS Code
    (`code -r`) and say plainly whether that worked."""
    if tmpdir:
        code = shutil.which('code')
        opened, why = False, 'the `code` command is not on the PATH (not a VS Code terminal?)'
        if code:
            try:
                r = subprocess.run([code, '-r', *shown], capture_output=True, text=True, timeout=30)
                opened = r.returncode == 0
                if not opened:
                    why = f'`code -r` failed: {(r.stderr or r.stdout).strip()[:200] or "exit status " + str(r.returncode)}'
            except (subprocess.TimeoutExpired, OSError) as e:
                why = f'`code -r` did not work: {e}'
        files = ', '.join(shown)
        if opened:
            print(f'No display available: opened in VS Code ({files}). Use ssh -X for a window, or -o FILE to save.')
        else:
            print(f'No display available and the figure could not be opened in VS Code: {why}.\n'
                  f'The figure is in {files} (open it from the VS Code explorer, or copy it with scp). '
                  f'Run from a terminal inside VS Code, or use ssh -X for a window.')
    elif not save:
        import matplotlib.pyplot as plt
        plt.show()


# ---------------------------------------------------------------- scalars
def pick_norm(arrays, name='', log=False, lin=False, vmin=None, vmax=None, clip=99.5, cmap=None,
              log_names=('ne', 'na', 'nap', 'ni')):
    """Colour normalisation and colormap shared by the arrays of one row.

    Positive densities (`log_names`) are logarithmic unless lin; fields of both signs are symmetric about 0
    (diverging colours); the rest linear. The upper limit is the `clip` percentile unless vmax is given.
    """
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    allv = np.concatenate([a[np.isfinite(a)] for a in arrays])
    positive = allv.min() > 0
    if (log or (positive and name in log_names and not lin)) and positive:
        hi = vmax or np.percentile(allv, clip)
        return LogNorm(vmin=vmin or hi * 1e-3, vmax=hi), cmap or 'inferno'
    if allv.min() < 0 < allv.max():
        m = vmax or np.percentile(np.abs(allv), clip)
        return TwoSlopeNorm(0, vmin=vmin if vmin is not None else -m, vmax=m), cmap or 'RdBu_r'
    return Normalize(vmin=vmin if vmin is not None else allv.min(),
                     vmax=vmax or np.percentile(allv, clip)), cmap or 'inferno'


def draw_cells(ax, polys, values, norm, cmap, **kw):
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(polys, array=values, cmap=cmap, norm=norm, edgecolors=kw.pop('edgecolors', 'face'),
                        linewidths=kw.pop('linewidths', 0.2), **kw)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    return pc


# ---------------------------------------------------------------- vector overlays
def _raster_mask(polys, xs, ys):
    """True where a regular-grid point lies inside some cell polygon."""
    from matplotlib.path import Path
    X, Y = np.meshgrid(xs, ys)
    mask = np.zeros(X.shape, bool)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    for p in polys:
        i0, i1 = np.searchsorted(xs, p[:, 0].min() - dx), np.searchsorted(xs, p[:, 0].max() + dx)
        j0, j1 = np.searchsorted(ys, p[:, 1].min() - dy), np.searchsorted(ys, p[:, 1].max() + dy)
        if i1 > i0 and j1 > j0:
            sub = np.column_stack([X[j0:j1, i0:i1].ravel(), Y[j0:j1, i0:i1].ravel()])
            mask[j0:j1, i0:i1] |= Path(p).contains_points(sub).reshape(j1 - j0, i1 - i0)
    return mask


def draw_streams(ax, polys, centres, vec, bounds=None, density=1.0, lw=0.6, lw_flux=False, color='k', res=0.003):
    """Streamlines of the cell vectors, interpolated linearly (triangulation of the cell centres) on a regular
    R-Z grid of spacing `res`; points outside every cell are masked, so lines stop at the mesh boundary.
    bounds=(R0, R1, Z0, Z1) restricts the grid (default: the mesh). lw_flux: line width follows log10|vec|."""
    import matplotlib.patheffects as pe
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    good = np.isfinite(vec).all(axis=1)
    allp = np.concatenate(polys)
    R0, R1, Z0, Z1 = bounds or (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())
    xs, ys = np.arange(R0, R1, res), np.arange(Z0, Z1, res)
    X, Y = np.meshgrid(xs, ys)
    tri = Triangulation(centres[good, 0], centres[good, 1])
    U = LinearTriInterpolator(tri, vec[good, 0])(X, Y)
    V = LinearTriInterpolator(tri, vec[good, 1])(X, Y)
    bad = ~_raster_mask(polys, xs, ys) | np.ma.getmaskarray(U) | np.ma.getmaskarray(V)
    U, V = np.ma.masked_where(bad, U), np.ma.masked_where(bad, V)
    width = lw
    if lw_flux:
        ref = np.nanpercentile(np.linalg.norm(vec[good], axis=1), 99)
        lo = np.log10(ref * 1e-4)
        with np.errstate(invalid='ignore', divide='ignore'):
            width = 0.3 + 1.7 * np.clip((np.log10(np.maximum(np.hypot(U, V), ref * 1e-4)) - lo)
                                        / (np.log10(ref) - lo), 0, 1)
    sp = ax.streamplot(xs, ys, U, V, density=density, color=color, linewidth=width, arrowsize=0.9,
                       arrowstyle='-|>', broken_streamlines=False)
    halo = [] if lw_flux else [pe.withStroke(linewidth=lw + 1.2, foreground='w', alpha=0.55)]   # dark + light maps
    for art in (sp.lines, sp.arrows):
        art.set_zorder(3)
        art.set_path_effects(halo)
    return sp


def draw_arrows(ax, centres, vec, shape=None, stride=2, length=0.03, proportional=False, ref=None,
                min_frac=0.01, color='k'):
    """Arrows at the cell centres: fixed length `length` m (direction only; cells with |v| below min_frac*ref are
    skipped) or, proportional=True, length ~ |v| with `ref` (default the 99th percentile) drawn as `length` m.
    shape=(nx, ny) lets stride thin out a structured mesh in both directions (cell order j*nx + i)."""
    mag = np.linalg.norm(vec, axis=1)
    if ref is None:
        ref = np.nanpercentile(mag[np.isfinite(mag)], 99)
    keep = np.isfinite(mag)
    if shape is not None:
        nx, ny = shape
        keep &= ((np.arange(nx) % stride == 0)[None, :] & (np.arange(ny) % stride == 0)[:, None]).ravel()
    else:
        keep &= (np.arange(len(vec)) % stride == 0)
    kw = dict(angles='xy', scale_units='xy', color=color, width=0.003, headwidth=3.5, headlength=4,
              headaxislength=3.5, zorder=3)
    if proportional:
        return ax.quiver(centres[keep, 0], centres[keep, 1], vec[keep, 0], vec[keep, 1],
                         scale=ref / length, pivot='tail', **kw)
    keep &= mag > min_frac * ref
    v = vec[keep] / mag[keep, None]
    return ax.quiver(centres[keep, 0], centres[keep, 1], v[:, 0], v[:, 1], scale=1 / length, pivot='mid', **kw)


# ---------------------------------------------------------------- faces / divergence
def _symlog(vals, vmax):
    from matplotlib.colors import SymLogNorm
    m = vmax if vmax else np.nanmax(np.abs(vals))
    return SymLogNorm(linthresh=max(m * 1e-6, 1e-30), vmin=-m, vmax=m, base=10)


def draw_faces(ax, segs, vals, vmax=None):
    """Every face as a coloured line segment (signed value, symmetric log scale, red/blue = +/-)."""
    from matplotlib.collections import LineCollection
    col = LineCollection(segs, array=vals, norm=_symlog(vals, vmax), cmap='RdBu_r', linewidths=1.5)
    ax.add_collection(col)
    return col


def draw_divergence(ax, polys, vals, vmax=None):
    """Net outflow per cell on the cell polygons (symmetric log scale)."""
    from matplotlib.collections import PolyCollection
    col = PolyCollection(polys, array=vals, norm=_symlog(vals, vmax), cmap='RdBu_r', edgecolors='none')
    ax.add_collection(col)
    return col
