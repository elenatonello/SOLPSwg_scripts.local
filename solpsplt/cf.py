"""Readers for the SOLPS-ITER state / geometry files.

read_cf          formatted '*cf:' files (b2fgmtry, b2fstate, plasmastate_aver.NNNN): label line
                 `*cf: <type> <count> <name>` followed by the values, several per line
scan_cf          the same file, but only the block names, types and sizes (fast, no values parsed)
read_plasmastate plasmastate.NNNN: Fortran unformatted sequential records
"""
import struct

import numpy as np


def _skip_values(lines, i, n):
    """Index of the line after the data of a block of n values that starts at line i.

    Assumes the usual layout (a fixed number of values per line, fewer on the last one) and checks it,
    falling back to counting the values line by line.
    """
    if n <= 0:
        return i
    first = len(lines[i].split())
    if first:
        k = -(-n // first)
        j = i + k
        if (j <= len(lines) and len(lines[j - 1].split()) == n - (k - 1) * first
                and (j == len(lines) or lines[j].startswith('*cf:') or not lines[j].strip())):
            return j
    got = 0
    while got < n:
        got += len(lines[i].split())
        i += 1
    return i


def _to_array(vals, typ):
    if typ == 'real':
        try:
            return np.array(vals, dtype=float)
        except ValueError:                                  # Fortran 'D' exponent
            return np.array([v.replace('D', 'E').replace('d', 'e') for v in vals], dtype=float)
    return np.array(vals, dtype=int)


def read_cf(fn, want=None):
    """Formatted B2.5 '*cf:' file -> {name: array}. Only the blocks in `want` (all if None);
    real blocks give float arrays, integer ones int arrays, character blocks are skipped."""
    out = {}
    with open(fn) as f:
        lines = f.read().split('\n')
    i = 1                                                   # line 0 is the VERSION line
    while i < len(lines):
        if not lines[i].startswith('*cf:'):
            i += 1
            continue
        p = lines[i].split()
        typ, n, name = p[1], int(p[2]), p[3]
        i += 1
        if typ == 'char':
            i += 1
            continue
        if want is not None and name not in want:
            i = _skip_values(lines, i, n)
            continue
        vals = []
        while len(vals) < n:
            vals += lines[i].split()
            i += 1
        out[name] = _to_array(vals, typ)
    return out


def scan_cf(fn):
    """{name: (type, count)} of every block of a formatted file, without parsing the values."""
    out = {}
    with open(fn) as f:
        lines = f.read().split('\n')
    i = 1
    while i < len(lines):
        if not lines[i].startswith('*cf:'):
            i += 1
            continue
        p = lines[i].split()
        typ, n, name = p[1], int(p[2]), p[3]
        out[name] = (typ, n)
        i += 2 if typ == 'char' else 1
        if typ != 'char':
            i = _skip_values(lines, i, n)
    return out


def read_plasmastate(fn):
    """plasmastate.NNNN (Fortran unformatted records, little endian) -> {field name: array}."""
    out = {}
    with open(fn, 'rb') as f:
        def rec():
            h = f.read(4)
            if len(h) < 4:
                return None
            n = struct.unpack('<i', h)[0]
            d = f.read(n)
            f.read(4)
            return d
        rec()  # version string
        while True:
            h = rec()
            if h is None:
                break
            typ = h[8:16].decode().strip()
            cnt = struct.unpack('<i', h[16:20])[0]
            name = h[20:].decode().strip()
            d = rec() if cnt > 0 else b''
            if typ == 'real':
                out[name] = np.frombuffer(d, '<f8')
            elif typ == 'int':
                out[name] = np.frombuffer(d, '<i4')
    return out
