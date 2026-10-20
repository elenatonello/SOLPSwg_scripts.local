"""Wall / structures of a run from the Carre `structure.dat`.

File format (coordinates in m):
        8                       number of structures
    $structures
    Structure    1
            -9                  number of points; negative = open polyline, positive = closed
         0.624  -0.300          R Z, one point per line
         ...
    Structure    2
    ...
(`structures.dat`, with an s, of some run directories is a different file: Carre segments in cm.)

read_structures(file)     -> list of Structure(points (n, 2), closed)
find_structure(*dirs)     -> the structure.dat found for a run directory (see below) or None
draw_wall(ax, structures) -> draws them on an axes
"""
from collections import namedtuple
from pathlib import Path

import numpy as np

Structure = namedtuple('Structure', 'points closed')


def read_structures(fn):
    """Structures of a structure.dat as a list of Structure(points (n, 2) [R, Z] in m, closed)."""
    lines = [l.split() for l in open(fn)]
    out, i = [], 0
    while i < len(lines):
        if lines[i] and lines[i][0].lower() == 'structure':
            k = int(lines[i + 1][0])
            n = abs(k)
            pts = np.array([[float(x) for x in l[:2]] for l in lines[i + 2:i + 2 + n]])
            out.append(Structure(pts, k > 0))
            i += 2 + n
        else:
            i += 1
    if not out:
        raise ValueError(f'{fn}: no structures found')
    return out


def find_structure(*dirs):
    """structure.dat for a run: looked for in each given directory, its baserun/, and the baserun/ and the directory
    that sit next to it (a run is <family>/<run>, with <family>/baserun/ and sometimes <family>/structure.dat)."""
    for d in dirs:
        d = Path(d).resolve()
        for cand in (d / 'structure.dat', d / 'baserun' / 'structure.dat', d.parent / 'baserun' / 'structure.dat',
                     d.parent / 'structure.dat'):
            if cand.is_file():
                return cand
    return None


def extent(structures):
    allp = np.concatenate([s.points for s in structures])
    return allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max()


def draw_wall(ax, structures, color='#222222', lw=1.4, zorder=5):
    """Draw the structures as lines (closed ones are closed)."""
    for s in structures:
        p = np.vstack([s.points, s.points[:1]]) if s.closed else s.points
        ax.plot(p[:, 0], p[:, 1], '-', color=color, lw=lw, zorder=zorder, solid_capstyle='round')
