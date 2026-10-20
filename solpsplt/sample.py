"""Values of a cell field along a straight line in the R-Z plane (structured and unstructured meshes alike).

    s, pts, v = line_profile(mesh, values, (R0, Z0), (R1, Z1), n=200, method='linear')

s       distance from the start of the line [m]            pts   the (n, 2) sample points [R, Z]
v       field at the points, NaN where the point is outside every cell of the mesh

method  'linear'  linear interpolation between the cell centres (Delaunay triangulation of the centres); points that lie
                  in a cell but outside the triangulation (within half a cell of the boundary) get the value of their cell
        'cell'    value of the cell containing the point (piecewise constant, steps at the cell faces)
"""
import numpy as np


def _bboxes(mesh):
    """Bounding boxes of the cell polygons (cached on the mesh)."""
    if not hasattr(mesh, '_bbox'):
        mesh._bbox = (np.array([p.min(axis=0) for p in mesh.polys]), np.array([p.max(axis=0) for p in mesh.polys]))
    return mesh._bbox


def find_cells(mesh, pts):
    """Index (into mesh.polys) of the cell containing each point, -1 outside the mesh."""
    from matplotlib.path import Path
    lo, hi = _bboxes(mesh)
    out = -np.ones(len(pts), int)
    for k, (r, z) in enumerate(pts):
        cand = np.flatnonzero((lo[:, 0] <= r) & (hi[:, 0] >= r) & (lo[:, 1] <= z) & (hi[:, 1] >= z))
        for c in cand:
            if Path(mesh.polys[c]).contains_point((r, z)):
                out[k] = c
                break
    return out


def line_profile(mesh, values, p0, p1, n=200, method='linear'):
    """Field `values` (one per interior cell, in mesh order) along the line p0 -> p1; see the module docstring."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)
    pts = p0 + t[:, None] * (p1 - p0)
    s = t * np.linalg.norm(p1 - p0)
    cells = find_cells(mesh, pts)
    inside = cells >= 0
    v = np.full(n, np.nan)
    v[inside] = np.asarray(values, float)[cells[inside]]                        # cell value
    if method == 'linear':
        from matplotlib.tri import LinearTriInterpolator, Triangulation
        good = np.isfinite(values)
        tri = Triangulation(mesh.centres[good, 0], mesh.centres[good, 1])
        lin = LinearTriInterpolator(tri, np.asarray(values, float)[good])(pts[:, 0], pts[:, 1])
        lin = np.ma.filled(lin, np.nan)
        use = inside & np.isfinite(lin)
        v[use] = lin[use]
    elif method != 'cell':
        raise ValueError(f"unknown method '{method}' (linear or cell)")
    return s, pts, v
