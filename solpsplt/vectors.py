"""Face-based 2-component fields (particle / heat fluxes, drift velocities) of structured and unstructured meshes.

A FaceField gives three views of the same data:
    cell_vectors()          one vector (R, Z) per interior cell, from a least-squares fit over the faces of the cell
    segments(comp=None)     every face as a line segment with its signed value (for drawing the faces themselves)
    divergence()            net outflow per interior cell

Conventions
  structured    block (nx+2, ny+2, 2): component 0 = flux through the left (poloidal-index) face of cell (ix, iy),
                component 1 = through its bottom face, positive towards increasing ix / iy; index 0 is unused
                and the last index carries the boundary (target) faces. The flux out of a cell through its right /
                top face is the value stored for its neighbour there (mesh.right / mesh.top, from the b2fgmtry
                topology), which is not (ix+1, iy) / (ix, iy+1) at the cuts of the mesh.
  unstructured  block (2, nFc): the two components are summed (or one chosen, `comp`), positive from the first
                to the second cell of the face (fcCv).
Cell vector: the outward flux of a cell through its face f is  F_f = J . n_f A_f  with n_f the outward unit normal and
A_f = L_f * 2 pi R_f the face area for integrated fluxes (fna, fhe, ...), or A_f = 1 for velocities (vaecrb, ...,
taken as the normal component). J (flux density, or velocity) is the least-squares solution over the faces of the cell.
Velocity blocks of unstructured runs are treated like the fluxes and are not validated.
"""
import numpy as np


def _outward_vectors(va, vb, centre, velocity):
    """Face vectors a_f = n A for faces (va -> vb) seen from the cell with the given centre."""
    mid, t = 0.5 * (va + vb), vb - va
    L = np.hypot(t[:, 0], t[:, 1])
    with np.errstate(divide='ignore', invalid='ignore'):
        n = np.column_stack([t[:, 1], -t[:, 0]]) / L[:, None]
    n *= np.sign(np.einsum('ij,ij->i', n, mid - centre))[:, None]
    return n if velocity else n * (L * 2 * np.pi * mid[:, 0])[:, None]


class _LeastSquares:
    """Accumulates sum a a^T and sum a F per cell and solves the 2x2 systems."""

    def __init__(self, ncell):
        self.M00, self.M01, self.M11, self.b0, self.b1 = (np.zeros(ncell) for _ in range(5))

    def add(self, cell, out, a):
        good = np.isfinite(a).all(axis=1)
        cell, out, a = cell[good], out[good], a[good]
        np.add.at(self.M00, cell, a[:, 0] ** 2)
        np.add.at(self.M01, cell, a[:, 0] * a[:, 1])
        np.add.at(self.M11, cell, a[:, 1] ** 2)
        np.add.at(self.b0, cell, a[:, 0] * out)
        np.add.at(self.b1, cell, a[:, 1] * out)

    def solve(self):
        det = self.M00 * self.M11 - self.M01 ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            jx = (self.M11 * self.b0 - self.M01 * self.b1) / det
            jy = (self.M00 * self.b1 - self.M01 * self.b0) / det
        return np.column_stack([jx, jy])


class StructuredFaces:
    def __init__(self, mesh, V, velocity=False):
        self.mesh, self.velocity = mesh, velocity
        self.Fx, self.Fy = V[..., 0], V[..., 1]                        # (nx+2, ny+2)

    def cell_vectors(self):
        m, Fx, Fy = self.mesh, self.Fx, self.Fy
        nx, ny = m.nx, m.ny
        cr = m.corners
        ctr = cr.mean(axis=2)
        flat = lambda a: a.transpose(1, 0, *range(2, a.ndim)).reshape(nx * ny, *a.shape[2:])
        acc = _LeastSquares(nx * ny)
        idx = np.arange(nx * ny)
        for out, ia, ib in ((-Fx[1:nx + 1, 1:ny + 1], 0, 2), (Fx[m.right], 1, 3),               # left, right
                            (-Fy[1:nx + 1, 1:ny + 1], 0, 1), (Fy[m.top], 2, 3)):               # bottom, top
            va, vb = flat(cr[:, :, ia]), flat(cr[:, :, ib])
            acc.add(idx, flat(out), _outward_vectors(va, vb, flat(ctr), self.velocity))
        return acc.solve()

    def segments(self, comp=None):
        m, nx, ny = self.mesh, self.mesh.nx, self.mesh.ny
        cr = m.corners
        segs, vals = [], []
        if comp in (None, 0):
            segs += [cr[:, :, [0, 2]].reshape(-1, 2, 2), cr[nx - 1][:, [1, 3]].reshape(-1, 2, 2)]
            vals += [self.Fx[1:nx + 1, 1:ny + 1].reshape(-1), self.Fx[nx + 1, 1:ny + 1]]
        if comp in (None, 1):
            segs += [cr[:, :, [0, 1]].reshape(-1, 2, 2), cr[:, ny - 1][:, [2, 3]].reshape(-1, 2, 2)]
            vals += [self.Fy[1:nx + 1, 1:ny + 1].reshape(-1), self.Fy[1:nx + 1, ny + 1]]
        return np.concatenate(segs), np.concatenate(vals)

    def divergence(self):
        m = self.mesh
        nx, ny, Fx, Fy = m.nx, m.ny, self.Fx, self.Fy
        d = (Fx[m.right] - Fx[1:nx + 1, 1:ny + 1]) + (Fy[m.top] - Fy[1:nx + 1, 1:ny + 1])
        return d.T.ravel()


class UnstructuredFaces:
    def __init__(self, mesh, f, velocity=False):
        self.mesh, self.f, self.velocity = mesh, f, velocity           # f: (nFc,)

    def cell_vectors(self):
        m = self.mesh
        acc = _LeastSquares(int(m.ok.sum()))
        va, vb = m.vx[m.fcVx[0]], m.vx[m.fcVx[1]]
        for side, sign in ((0, 1.0), (1, -1.0)):                       # outward flux: +f for cell 1, -f for cell 2
            cells = m.fcCv[side]
            sel = m.interior_index[cells] >= 0
            acc.add(m.interior_index[cells][sel], (sign * self.f)[sel],
                    _outward_vectors(va[sel], vb[sel], m.cen_all[cells][sel], self.velocity))
        return acc.solve()

    def segments(self, comp=None):
        m = self.mesh
        return np.stack([m.vx[m.fcVx[0]], m.vx[m.fcVx[1]]], axis=1), self.f

    def divergence(self):
        m = self.mesh
        v = np.zeros(m.nCv)
        np.add.at(v, m.fcCv[0], self.f)                                # leaves cell 1
        np.add.at(v, m.fcCv[1], -self.f)                               # enters cell 2
        return v[m.ok]


def make_faces(mesh, block, ns=None, species=0, velocity=False, comp=None):
    """FaceField of a state block (1D array). ns: number of species if the block is species-resolved, else None."""
    if mesh.kind == 'structured':
        nx, ny = mesh.nx, mesh.ny
        if ns is None:
            V = block.reshape(nx + 2, ny + 2, 2, order='F')
        else:
            V = block.reshape(nx + 2, ny + 2, 2, ns, order='F')[..., species]
        return StructuredFaces(mesh, V, velocity)
    arr = block.reshape(2, mesh.nFc) if ns is None else block.reshape(ns, 2, mesh.nFc)[species]
    return UnstructuredFaces(mesh, arr.sum(axis=0) if comp is None else arr[comp], velocity)
