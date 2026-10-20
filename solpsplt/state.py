"""State of a run (fields on a mesh) from any of the SOLPS-ITER state files.

    st = State.from_run(run_dir)                       # b2fstate (structured / unstructured), see `source`
    st = State.from_run(run_dir, source='plasmastate', snap='0015')
    st = State.from_blocks(mesh, plasmastate_dict)     # already-read plasmastate blocks (plot_plasmastate)

    st.catalog()               {'scalar': {name: species-resolved?}, 'vector': {...}} from the block sizes
    st.scalar('te')            interior-cell values (temperatures in eV)       species: st.scalar('na', 1)
    st.faces('fna', 1)         FaceField (see vectors.py) of a face-based block
    st.species_name(4)         'C2+ (s4)'

`source` of State.from_run:
    auto          b2fstate; for an unstructured run whose b2fstate is only the stub written at the start of a
                  running job, the latest plasmastate.NNNN instead
    b2fstate      the b2fstate of the run (run dir, or b2mn.exe.dir)
    plasmastate   plasmastate.<snap> of b2mn.exe.dir (latest if snap is None); unstructured only
    aver          scalars of the time-averaged plasmastate_aver.<snap> (no ne: ne = sum zamin*na);
                  vector blocks still come from plasmastate.<snap>; unstructured only
Unstructured b2fstate and plasmastate have the same block names and layouts (flat, species outermost), so
everything downstream is the same for the sources.
"""
import glob
import re
from pathlib import Path

import numpy as np

from .cf import read_cf, read_plasmastate, scan_cf
from .mesh import find_geometry, open_mesh
from .vectors import make_faces
from .wall import find_structure, read_structures

EV = 1.602176634e-19
TEMPS = {'te', 'ti', 'tn'}                       # shown in eV
ALIASES = {'vecrb': 'veecrb', 'vdia': 'vedia'}   # names used in conversation -> block names
ELEMENTS = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 10: 'Ne', 18: 'Ar', 74: 'W'}


def latest_file(pattern):
    """The file matching the glob pattern with the highest trailing number, or None."""
    files = glob.glob(pattern)
    key = lambda p: int(re.findall(r'(\d+)$', p)[0]) if re.findall(r'(\d+)$', p) else -1
    return Path(max(files, key=key)) if files else None


def is_velocity(name):
    """v*, w*, u* blocks are velocities [m/s] (normal component on the faces); the others face-integrated fluxes."""
    return name[0] in 'vwu'


class State:
    def __init__(self, mesh, source, file=None, blocks=None, sizes=None, run=None, label=None):
        self.mesh, self.source, self.file, self.run = mesh, source, file, run
        self.label = label or (Path(run).name if run else '')
        self._blocks = dict(blocks or {})
        if sizes is None:
            sizes = {k: v.size for k, v in self._blocks.items() if v.dtype.kind == 'f'}
        self._sizes = sizes                              # name -> number of values, real blocks only
        self._species = None

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_blocks(cls, mesh, blocks, source='plasmastate', run=None, file=None):
        return cls(mesh, source, file=file, blocks=blocks, run=run)

    @classmethod
    def from_file(cls, file, geometry=None):
        """A single state file (b2fstate, b2fstati, the output of generate_initialState_us.py, plasmastate.NNNN);
        the geometry is `geometry` or the b2fgmtry found for the directory of the file."""
        file = Path(file).expanduser().resolve()
        mesh = open_mesh(geometry or find_geometry(file.parent))
        if file.name.startswith('plasmastate') and not file.name.startswith('plasmastate_aver'):
            st = cls(mesh, 'plasmastate', file=file, blocks=read_plasmastate(file), run=file.parent)
        else:
            st = cls._from_cf(mesh, file, 'b2fstate', file.parent)
        st.label = f'{file.parent.name}/{file.name}'
        return st

    @classmethod
    def from_run(cls, run, source='auto', snap=None, geometry=None):
        """State of a run directory (see the module docstring); `run` may also be a state file (from_file)."""
        run = Path(run).expanduser().resolve()
        if run.is_file():
            return cls.from_file(run, geometry)
        if not run.is_dir():
            raise FileNotFoundError(f'not a directory: {run}')
        mesh = open_mesh(geometry or find_geometry(run))
        dirs = (run / 'b2mn.exe.dir', run)
        if mesh.kind == 'structured':
            if source not in ('auto', 'b2fstate'):
                raise ValueError(f"structured runs only have a b2fstate (source '{source}' not available)")
            f = next((d / 'b2fstate' for d in (run, run / 'b2mn.exe.dir') if (d / 'b2fstate').is_file()), None)
            if f is None:
                raise FileNotFoundError(f'no b2fstate in {run}')
            return cls._from_cf(mesh, f, 'b2fstate', run)

        def snapshot(base):
            for d in dirs:
                f = (d / f'{base}.{snap}') if snap else latest_file(str(d / (base + '.*')))
                if f and Path(f).is_file():
                    return Path(f)
            return None

        if source in ('auto', 'b2fstate'):
            f = next((d / 'b2fstate' for d in (run, run / 'b2mn.exe.dir') if (d / 'b2fstate').is_file()), None)
            if f is not None and 'na' in scan_cf(f):
                return cls._from_cf(mesh, f, 'b2fstate', run)
            if source == 'b2fstate':
                raise ValueError(f'{f or run / "b2fstate"}: no complete b2fstate (run still going or failed?)')
            source = 'plasmastate'                       # auto: running job, use the latest snapshot
        if source == 'plasmastate':
            f = snapshot('plasmastate')
            if f is None:
                raise FileNotFoundError(f'no plasmastate.* in {run}')
            return cls(mesh, 'plasmastate', file=f, blocks=read_plasmastate(f), run=run)
        if source == 'aver':
            fa, fi = snapshot('plasmastate_aver'), snapshot('plasmastate')
            if fa is None:
                raise FileNotFoundError(f'no plasmastate_aver.* in {run}')
            blocks = dict(read_plasmastate(fi)) if fi else {}
            raw = read_cf(fa)
            for k, v in raw.items():
                if k.endswith('_mean') and k[:-5] in ('na', 'ua', 'te', 'ti', 'po'):
                    blocks[k[:-5]] = v
            blocks['ne'] = (raw['zamin'][:, None] * blocks['na'].reshape(len(raw['zamin']), -1)).sum(axis=0)
            return cls(mesh, 'aver', file=fa, blocks=blocks, run=run)
        raise ValueError(f"unknown source '{source}'")

    @classmethod
    def _from_cf(cls, mesh, f, source, run):
        cat = scan_cf(f)
        if 'na' not in cat:
            raise ValueError(f'{f} is incomplete ({Path(f).stat().st_size} bytes): the run wrote no valid state')
        sizes = {k: n for k, (t, n) in cat.items() if t == 'real'}
        return cls(mesh, source, file=f, sizes=sizes, run=run)

    # ---------------------------------------------------------------- blocks
    def load(self, names):
        """Read the blocks (text files: only the missing ones, in one pass over the file, plus the species header)."""
        missing = {n for n in names if n not in self._blocks}
        if missing and self.source == 'b2fstate':
            self._blocks.update(read_cf(self.file, missing | {'zn', 'zamin', 'am'}))
        miss = [n for n in names if n not in self._blocks]
        if miss:
            raise KeyError(f"block(s) {', '.join(miss)} not in {self.file or self.source}")

    def block(self, name):
        self.load([name])
        return self._blocks[name]

    def ns(self):
        n = self._sizes['na'] // self.mesh.ncell_all
        return n

    def species_arrays(self):
        """(zn, zamin, am) of the species, or None. From the state file itself, else the b2fstate of the run
        (a stub written at the start of a job is enough: it has the species header)."""
        if self._species is None:
            if 'zn' in self._blocks:
                s = self._blocks
            else:
                cands = [Path(self.file)] if self.file is not None else []
                if self.run:
                    cands += [Path(self.run) / 'b2fstate', Path(self.run) / 'b2mn.exe.dir' / 'b2fstate']
                s = {}
                for f in cands:
                    binary = f.name.startswith('plasmastate') and not f.name.startswith('plasmastate_aver')
                    if f.is_file() and not binary:
                        s = read_cf(f, {'zn', 'zamin', 'am'})
                        if 'zn' in s:
                            break
            self._species = (s['zn'], s['zamin'], s.get('am')) if 'zn' in s else False
        return self._species or None

    def species_name(self, s):
        sp = self.species_arrays()
        if sp is None:
            return f's{s}'
        zn, za, am = sp
        el = ELEMENTS.get(int(zn[s]), f'Z{int(zn[s])}')
        if int(zn[s]) == 1 and am is not None:
            el = {1: 'H', 2: 'D', 3: 'T'}.get(int(round(am[s])), 'H')
        c = int(za[s])
        return f"{el}{'' if c == 0 else ('+' if c == 1 else f'{c}+')} (s{s})"

    def catalog(self):
        """{'scalar': {name: species-resolved?}, 'vector': {...}} from the block sizes."""
        m, ns = self.mesh, self.ns()
        ncell = m.ncell_all
        nfc = getattr(m, 'nFc', None)
        sc, vc = {}, {}
        for k, n in self._sizes.items():
            if n == ncell:
                sc[k] = False
            elif ns > 1 and n == ncell * ns:
                sc[k] = True
            elif nfc is not None and n == 2 * nfc:
                vc[k] = False
            elif nfc is not None and n == 2 * nfc * ns:
                vc[k] = True
            elif nfc is None and n == 2 * ncell:
                vc[k] = False
            elif nfc is None and n == 2 * ncell * ns:
                vc[k] = True
        return {'scalar': sc, 'vector': vc}

    # ---------------------------------------------------------------- wall
    def wall(self, file=None):
        """Structures of the wall: from `file`, else the structure.dat found for the run (wall.find_structure).
        Returns (structures, path), or (None, None) when there is none."""
        f = Path(file) if file else find_structure(self.run) if self.run else None
        if f is None or not f.is_file():
            return None, None
        return read_structures(f), f

    # ---------------------------------------------------------------- fields
    def scalar(self, name, species=1):
        """Values of a cell scalar on the interior cells (temperatures converted to eV)."""
        kinds = self.catalog()['scalar']
        if name not in kinds:
            raise KeyError(f"'{name}' is not a cell scalar of {self.label or self.file} (see --list)")
        ns = self.ns() if kinds[name] else None
        if ns is not None and not 0 <= species < ns:
            raise ValueError(f'species must be 0..{ns - 1}')
        v = self.mesh.cell_values(self.block(name), ns, species)
        return v / EV if name in TEMPS else v

    def faces(self, name, species=1, comp=None):
        """FaceField of a face-based block (see vectors.py); aliases vecrb -> veecrb, vdia -> vedia."""
        name = ALIASES.get(name, name)
        kinds = self.catalog()['vector']
        if name not in kinds:
            raise KeyError(f"'{name}' is not a face block of {self.label or self.file} (see --list)")
        ns = self.ns() if kinds[name] else None
        if ns is not None and not 0 <= species < ns:
            raise ValueError(f'species must be 0..{ns - 1}')
        return make_faces(self.mesh, self.block(name), ns, species, is_velocity(name), comp)
