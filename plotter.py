#!/usr/bin/env python3
"""Lightweight Plotter class mirroring GOAT's plotting structure (GOAT: the code of the coupled
SOLPS-GOAT runs, see doc/goat_inputs_procedure.tex).

Library, not a command-line tool: used by plot_b2fstate_us.py. Provides a small, modular plotting API
built on matplotlib (one figure with one axes, created on construction) with an optional Delaunay
triangulation helper (uses SciPy when available). Unlike the other plotting scripts it does not
select a matplotlib backend; the default one is used.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


class Plotter:
    def __init__(self, figsize=(8, 6)):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError('matplotlib is required for plotting: pip install matplotlib') from e
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def scatter(self, x, y, c=None, s=12, cmap='viridis'):
        return self.ax.scatter(x, y, c=c, s=s, cmap=cmap)

    def tripcolor(self, x, y, triangles, values, cmap='viridis'):
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(x, y, triangles=triangles)
        return self.ax.tripcolor(triang, values, cmap=cmap, shading='flat')

    def delaunay(self, x, y):
        """Triangle vertex indices (n_tri, 3) of the Delaunay triangulation of the points, or None if
        SciPy is not installed. The triangulation covers the convex hull of the points."""
        try:
            from scipy.spatial import Delaunay
        except Exception:
            return None
        pts = np.column_stack([x, y])
        tri = Delaunay(pts)
        return tri.simplices

    def colorbar(self, mappable):
        self.fig.colorbar(mappable, ax=self.ax)

    def set_aspect_equal(self):
        self.ax.set_aspect('equal')

    def set_labels(self, xlabel='X', ylabel='Y'):
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def set_title(self, title: str):
        self.ax.set_title(title)

    def save(self, path: Path | str, dpi=150):
        self.fig.savefig(str(path), bbox_inches='tight', dpi=dpi)

    def show(self):
        self.plt.show()
