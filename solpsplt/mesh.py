"""Meshes of SOLPS-ITER runs: structured (3.0.9 style) and unstructured (3.2.1 style) behind one interface.

open_mesh(b2fgmtry) returns a StructuredMesh or an UnstructuredMesh (decided from the file header). Both have

    kind            'structured' | 'unstructured'
    polys           list of (nvert, 2) arrays [R, Z] of the interior cells (guard cells dropped)
    centres         (ncell, 2) cell centres, same order as polys
    ncell_all       number of cells of a state block, guard cells included
    cell_values(a, ns=None, species=None)   interior-cell values of a state block `a`
    extent          (Rmin, Rmax, Zmin, Zmax) of all vertices of the file
    cell_extent     the same for the interior cells only (use this for plot limits)

Cell order: structured, j (radial) outer and i (poloidal) inner; unstructured, the file order without the
guard cells (cells with fewer than 3 vertices).
"""
import numpy as np

from .cf import read_cf


def find_geometry(run):
    """First usable b2fgmtry of a run directory: b2mn.exe.dir, the run, ../baserun, baserun."""
    from pathlib import Path
    run = Path(run)
    for d in (run / 'b2mn.exe.dir', run, run.parent / 'baserun', run / 'baserun'):
        f = d / 'b2fgmtry'
        if f.is_file() and f.stat().st_size > 0:
            return f
    raise FileNotFoundError(f'no b2fgmtry found for {run}')


def build_polygons(geo):
    """Cell polygons of an unstructured b2fgmtry read with read_cf (needs cvX, cvVxP, cvVx, vxX, vxY).

    Returns (polys, ok): one polygon per cell (vertices sorted by angle) and the mask of cells that have
    one (>= 3 vertices); the others get a dummy triangle.
    """
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


def valid_cells(geo):
    """Mask of cells with at least 3 vertices (the rest are degenerate/boundary placeholders)."""
    return geo['cvVxP'][len(geo['cvX']):] >= 3


def _is_unstructured(geom_file):
    with open(geom_file) as f:
        return 'nCv' in ''.join(f.readline() for _ in range(6))


class StructuredMesh:
    kind = 'structured'

    def __init__(self, geom_file):
        self.geom_file = str(geom_file)
        g = read_cf(geom_file, {'nx,ny', 'crx', 'cry', 'rightix', 'rightiy', 'topix', 'topiy'})
        nx, ny = int(g['nx,ny'][0]), int(g['nx,ny'][1])
        cr = lambda a: a.reshape(nx + 2, ny + 2, 4, order='F')
        rz = np.stack([cr(g['crx']), cr(g['cry'])], axis=-1)           # (nx+2, ny+2, 4, 2), guard cells included
        self.nx, self.ny = nx, ny
        self.ncell_all = (nx + 2) * (ny + 2)
        self.corners = rz[1:nx + 1, 1:ny + 1]                          # (nx, ny, 4, 2): LL, LR, UL, UR
        # Neighbours across the right and top faces (b2fgmtry topology, Fortran numbering -1..nx -> file index + 1);
        # they differ from (ix+1, iy) / (ix, iy+1) at the topological cuts (core closure, private flux region).
        if all(k in g for k in ('rightix', 'rightiy', 'topix', 'topiy')):
            nb = lambda k: g[k].reshape(nx + 2, ny + 2, order='F')[1:nx + 1, 1:ny + 1] + 1
            self.right, self.top = (nb('rightix'), nb('rightiy')), (nb('topix'), nb('topiy'))
        else:                                                          # no topology in the file: plain neighbours
            ix, iy = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing='ij')
            self.right, self.top = (ix + 1, iy), (ix, iy + 1)
        order = [0, 1, 3, 2]                                           # polygon order LL, LR, UR, UL
        self.polys = [self.corners[i, j][order] for j in range(ny) for i in range(nx)]
        self.centres = self.corners.mean(axis=2).transpose(1, 0, 2).reshape(-1, 2)
        allp = self.corners.reshape(-1, 2)
        self.extent = self.cell_extent = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())

    def cell_values(self, a, ns=None, species=None):
        nx, ny = self.nx, self.ny
        if ns is None:
            return a.reshape(nx + 2, ny + 2, order='F')[1:nx + 1, 1:ny + 1].T.ravel()
        return a.reshape(nx + 2, ny + 2, ns, order='F')[1:nx + 1, 1:ny + 1, species].T.ravel()


class UnstructuredMesh:
    kind = 'unstructured'

    def __init__(self, geom_file):
        self.geom_file = str(geom_file)
        dims = 'nCi,nCg,nCv,nFc,nVx,nFs,nFt'
        g = read_cf(geom_file, {dims, 'cvX', 'cvY', 'cvVxP', 'cvVx', 'vxX', 'vxY', 'fcCv', 'fcVx'})
        self.nCv, self.nFc = int(g[dims][2]), int(g[dims][3])
        self.ncell_all = self.nCv
        self.polys_all, self.ok = build_polygons(g)
        self.polys = [self.polys_all[i] for i in np.flatnonzero(self.ok)]
        self.vx = np.column_stack([g['vxX'], g['vxY']])
        self.cen_all = np.column_stack([g['cvX'], g['cvY']])
        self.centres = self.cen_all[self.ok]
        self.fcCv = g['fcCv'].reshape(2, self.nFc) - 1                 # the two cells of each face (0-based)
        self.fcVx = g['fcVx'].reshape(2, self.nFc) - 1                 # the two end vertices of each face
        self.interior_index = -np.ones(self.nCv, int)                  # cell number -> index among interior cells
        self.interior_index[self.ok] = np.arange(self.ok.sum())
        self.extent = (self.vx[:, 0].min(), self.vx[:, 0].max(), self.vx[:, 1].min(), self.vx[:, 1].max())
        allp = np.concatenate(self.polys)            # extent of the interior cells (the vertex list can hold unused points)
        self.cell_extent = (allp[:, 0].min(), allp[:, 0].max(), allp[:, 1].min(), allp[:, 1].max())

    def cell_values(self, a, ns=None, species=None):
        if ns is None:
            return a[self.ok]
        return a.reshape(ns, -1)[species][self.ok]


def open_mesh(geom_file):
    """Mesh of a b2fgmtry, structured or unstructured according to its header."""
    return UnstructuredMesh(geom_file) if _is_unstructured(geom_file) else StructuredMesh(geom_file)
