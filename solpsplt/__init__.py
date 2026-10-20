"""solpsplt - shared code of the SOLPS-ITER plotting tools in scripts.local.

    cf       readers of the formatted ('*cf:') and plasmastate files
    mesh     structured / unstructured meshes (cell polygons, faces)
    state    State: fields of a run from b2fstate or plasmastate, scalars and face blocks, species names
    vectors  face fluxes / velocities -> cell vectors, face segments, divergence
    wall     the wall / structures of a run (Carre structure.dat)
    draw     display handling, colour scales, maps, streamlines, arrows, faces

Tools built on it: plot_b2fstate.py (final state of one or several runs), plot_plasmastate.py (snapshots of a
running job). scripts.local is on PYTHONPATH, so `import solpsplt` works from anywhere.
"""
from .cf import read_cf, read_plasmastate, scan_cf
from .mesh import StructuredMesh, UnstructuredMesh, build_polygons, find_geometry, open_mesh, valid_cells
from .state import ALIASES, EV, State, is_velocity
from .vectors import make_faces
from .wall import find_structure, read_structures
from . import draw

__all__ = ['read_cf', 'read_plasmastate', 'scan_cf', 'StructuredMesh', 'UnstructuredMesh', 'build_polygons',
           'find_geometry', 'open_mesh', 'valid_cells', 'ALIASES', 'EV', 'State', 'is_velocity', 'make_faces',
           'find_structure', 'read_structures', 'draw']
