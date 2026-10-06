#!/usr/bin/env python3
"""b2state_io.py

Provides the B2StateIO helper class for reading/writing minimal
SOLPS `b2fstate` and `b2fgmtry` text files.

This module is intended for import by other scripts (e.g. the
`generate_initialState_us.py` CLI) so the parsing/writing logic can be
reused without duplicating code.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np


class B2StateIO:
    """Helper class providing minimal SOLPS b2fstate/gmtry read/write.

    Methods match the previous standalone functions used by the CLI.
    """

    def read_numbers_after(self, text: str, label: str, count: int):
        # search for the label anchored at the start of a line to avoid
        # accidental matches in free-form header text (e.g. filenames)
        # 1) try strict match at start of line (most robust)
        pat = re.compile(r"(?im)^[ \t]*" + re.escape(label) + r"(?=[ \t]|$)")
        m = pat.search(text)
        if not m:
            # 2) fallback: match label preceded by whitespace or newline
            #    (covers files where labels are indented or have leading markers)
            pat2 = re.compile(r"(?i)(?:^|\n|\s)" + re.escape(label) + r"(?=[ \t]|$)")
            m = pat2.search(text)
        if not m:
            raise ValueError(f"Label '{label}' not found in file")
        sub = text[m.end():]
        # accept Fortran-style D exponent as well as E
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", sub)
        if len(nums) < count:
            raise ValueError(f"Not enough numbers after label {label} (need {count}, got {len(nums)})")
        out = []
        for x in nums[:count]:
            xt = x.replace('D', 'E').replace('d', 'e')
            out.append(float(xt))
        return out

    def read_b2fgmtry_us(self, path: Path):
        txt = path.read_text()
        firstline = txt.splitlines()[0] if txt.splitlines() else ''
        dims = self.read_numbers_after(txt, 'nCi,nCg,nCv,nFc,nVx,nFs,nFt', 7)
        nCi, nCg, nCv, nFc, nVx, nFs, nFt = [int(x) for x in dims]
        gmtry = {'nCv': nCv, 'nFc': nFc}
        if firstline.upper().startswith('VERSION'):
            gmtry['version'] = firstline
        cvX = self.read_numbers_after(txt, 'cvX', nCv)
        cvY = self.read_numbers_after(txt, 'cvY', nCv)
        gmtry['cvX'] = np.array(cvX)
        gmtry['cvY'] = np.array(cvY)
        return gmtry

    def read_b2fgmtry_st(self, path: Path):
        txt = path.read_text()
        dims = self.read_numbers_after(txt, 'nx,ny', 2)
        nx, ny = [int(x) for x in dims]
        nx2 = nx + 2
        ny2 = ny + 2
        ncr = nx2 * ny2 * 4
        crx = self.read_numbers_after(txt, 'crx', ncr)
        cry = self.read_numbers_after(txt, 'cry', ncr)
        crx = np.array(crx).reshape((nx2, ny2, 4), order='F')
        cry = np.array(cry).reshape((nx2, ny2, 4), order='F')
        cvX = crx.mean(axis=2).ravel(order='F')
        cvY = cry.mean(axis=2).ravel(order='F')
        return {'nCv': cvX.size, 'cvX': cvX, 'cvY': cvY, 'nx': nx, 'ny': ny}

    def read_b2fstate_st(self, path: Path):
        txt = path.read_text()
        firstline = txt.splitlines()[0] if txt.splitlines() else ''
        version = ''
        if firstline.upper().startswith('VERSION'):
            version = firstline
        dims = self.read_numbers_after(txt, 'nx,ny,ns', 3)
        nx, ny, ns = [int(x) for x in dims]
        nx2 = nx + 2
        ny2 = ny + 2
        out = {'version': version, 'nx': nx, 'ny': ny, 'ns': ns}
        out['zamin'] = np.array(self.read_numbers_after(txt, 'zamin', ns))
        out['zamax'] = np.array(self.read_numbers_after(txt, 'zamax', ns))
        out['zn'] = np.array(self.read_numbers_after(txt, 'zn', ns))
        out['am'] = np.array(self.read_numbers_after(txt, 'am', ns))
        n_na = nx2 * ny2 * ns
        out['na'] = np.array(self.read_numbers_after(txt, 'na', n_na)).reshape((nx2, ny2, ns), order='F')
        n_ne = nx2 * ny2
        out['ne'] = np.array(self.read_numbers_after(txt, 'ne', n_ne)).reshape((nx2, ny2), order='F')
        out['ua'] = np.array(self.read_numbers_after(txt, 'ua', n_na)).reshape((nx2, ny2, ns), order='F')
        out['te'] = np.array(self.read_numbers_after(txt, 'te', n_ne)).reshape((nx2, ny2), order='F')
        out['ti'] = np.array(self.read_numbers_after(txt, 'ti', n_ne)).reshape((nx2, ny2), order='F')
        try:
            out['tn'] = np.array(self.read_numbers_after(txt, 'tn', n_ne)).reshape((nx2, ny2), order='F')
        except Exception:
            out['tn'] = out['ti'] * 0.0
        out['po'] = np.array(self.read_numbers_after(txt, 'po', n_ne)).reshape((nx2, ny2), order='F')
        return out

    def read_b2fstate_us(self, path: Path):
        """Read an unstructured b2fstate (fields are flat over nCv cells)."""
        version_line, blocks = self.parse_b2fstate_template(path)
        by_label = {b[0].strip().split()[-1]: b[1] for b in blocks}
        if 'nCv,nFc,ns' not in by_label or len(by_label['nCv,nFc,ns']) < 3:
            raise ValueError(f"{path}: no 'nCv,nFc,ns' header; not an unstructured b2fstate")
        nCv, nFc, ns = [int(x) for x in by_label['nCv,nFc,ns'][:3]]

        def field(name, shape):
            v = by_label.get(name)
            if v is None or np.size(v) != int(np.prod(shape)):
                raise ValueError(f"{path}: '{name}' has {0 if v is None else np.size(v)} values, expected {int(np.prod(shape))}")
            return np.asarray(v).reshape(shape, order='F')

        out = {'version': version_line, 'nCv': nCv, 'nFc': nFc, 'ns': ns}
        for meta in ('zamin', 'zamax', 'zn', 'am'):
            out[meta] = field(meta, (ns,))
        out['na'] = field('na', (nCv, ns))
        out['ua'] = field('ua', (nCv, ns))
        for name in ('ne', 'te', 'ti', 'po'):
            out[name] = field(name, (nCv,))
        try:
            out['tn'] = field('tn', (nCv,))
        except ValueError:
            out['tn'] = np.copy(out['ti'])
        return out

    def read_flux_tubes(self, path: Path):
        """Read the flux-tube cell lists of an unstructured b2fgmtry.

        Returns (tubes, closed): `tubes` is a list of 0-based cell-index
        arrays, each ordered along the tube; `closed[i]` is True when the
        first and last cell of tube i share a face (e.g. core rings).
        Ghost cells belong to no tube and are not returned.
        """
        _, blocks = self.parse_b2fstate_template(path)
        d = {b[0].strip().split()[-1]: b[1] for b in blocks}
        for key in ('nCi,nCg,nCv,nFc,nVx,nFs,nFt', 'ftCvP', 'ftCv', 'fcCv'):
            if key not in d:
                raise ValueError(f"{path}: '{key}' not found; cannot identify flux tubes")
        dims = d['nCi,nCg,nCv,nFc,nVx,nFs,nFt'].astype(int)
        nFc, nFt = int(dims[3]), int(dims[6])
        ptr = d['ftCvP'].astype(int)
        if ptr.size != 2 * nFt:
            raise ValueError(f"{path}: ftCvP has {ptr.size} values, expected {2 * nFt}")
        # ftCvP holds 1-based start offsets (first nFt) then cell counts (last nFt)
        start, count = ptr[:nFt], ptr[nFt:]
        cells = d['ftCv'].astype(int)
        # fcCv is (nFc, 2): the two cells on either side of each face
        fc = d['fcCv'].astype(int).reshape((nFc, 2), order='F')
        face_pairs = set(map(tuple, np.sort(fc, axis=1).tolist()))
        tubes, closed = [], []
        for s, n in zip(start, count):
            t = cells[s - 1:s - 1 + n] - 1
            tubes.append(t)
            closed.append(n > 2 and (min(t[0], t[-1]) + 1, max(t[0], t[-1]) + 1) in face_pairs)
        return tubes, closed

    @staticmethod
    def smooth_along_tubes(values, tubes, closed, sigma, log=False):
        """Gaussian-smooth `values` (nCv,) or (nCv, k) along each flux tube.

        `sigma` is in cells. Closed tubes wrap around; open tubes use a
        truncated, renormalised kernel so constants are preserved. With
        `log=True`, columns that are strictly positive over a tube are
        smoothed in log space (positive fields spanning decades).
        """
        vals = np.asarray(values, dtype=float)
        squeeze = vals.ndim == 1
        if squeeze:
            vals = vals[:, None]
        out = vals.copy()
        radius = int(np.ceil(3.0 * sigma))
        cache = {}
        for cells, is_closed in zip(tubes, closed):
            n = len(cells)
            if n < 2:
                continue
            key = (n, is_closed)
            if key not in cache:
                i = np.arange(n)
                dist = np.abs(i[:, None] - i[None, :])
                if is_closed:
                    dist = np.minimum(dist, n - dist)
                w = np.exp(-0.5 * (dist / sigma) ** 2) * (dist <= radius)
                cache[key] = w / w.sum(axis=1, keepdims=True)
            w = cache[key]
            v = vals[cells]
            res = w @ v
            if log:
                pos = (v > 0).all(axis=0)
                if pos.any():
                    res[:, pos] = np.exp(w @ np.log(v[:, pos]))
            out[cells] = res
        return out[:, 0] if squeeze else out

    def nearest_map(self, src_xy: np.ndarray, dst_xy: np.ndarray, chunk: int = 2000):
        # chunk over destination points so memory stays O(chunk * nSrc);
        # a dense nDst x nSrc matrix is prohibitive for two unstructured meshes
        idx = np.empty(len(dst_xy), dtype=np.intp)
        for i in range(0, len(dst_xy), chunk):
            d = dst_xy[i:i + chunk]
            dx = d[:, None, 0] - src_xy[None, :, 0]
            dy = d[:, None, 1] - src_xy[None, :, 1]
            idx[i:i + chunk] = np.argmin(dx * dx + dy * dy, axis=1)
        return idx

    def parse_b2fstate_template(self, path: Path):
        txt = path.read_text()
        lines = txt.splitlines()
        version_line = ''
        if lines and lines[0].upper().startswith('VERSION'):
            version_line = lines[0]
            lines = lines[1:]
        blocks = []
        i = 0
        # match numeric tokens including Fortran D exponent
        number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")
        while i < len(lines):
            line = lines[i].strip()
            if line == '':
                i += 1
                continue
            if line.startswith('*cf:'):
                label_line = lines[i]
                parts = line.split()
                label_token = parts[-1]
                i += 1
                nums = []
                raw_lines = []
                while i < len(lines) and (not lines[i].strip().startswith('*cf:')) and re.search(r'\d', lines[i]):
                    raw_lines.append(lines[i].rstrip('\n'))
                    nums.extend(number_re.findall(lines[i]))
                    i += 1
                vals = np.array([float(x.replace('D', 'E').replace('d', 'e')) for x in nums]) if nums else np.array([], dtype=float)
                blocks.append((label_line, vals, raw_lines))
                continue
            if re.match(r'^[A-Za-z0-9_,\s]+$', line):
                label_line = lines[i]
                i += 1
                if i < len(lines) and not re.search(r'\d', lines[i]):
                    s = lines[i].rstrip()
                    blocks.append((label_line, s, [s]))
                    i += 1
                else:
                    nums = []
                    raw_lines = []
                    while i < len(lines) and (not lines[i].strip().startswith('*cf:')) and re.search(r'\d', lines[i]):
                        raw_lines.append(lines[i].rstrip('\n'))
                        nums.extend(number_re.findall(lines[i]))
                        i += 1
                    vals = np.array([float(x.replace('D', 'E').replace('d', 'e')) for x in nums]) if nums else np.array([], dtype=float)
                    blocks.append((label_line, vals, raw_lines))
                continue
            i += 1
        return version_line, blocks

    def _format_real_fortran(self, val: float, width: int = 20, prec: int = 13) -> str:
        s = f"{val:.{prec}E}"
        s = re.sub(r'([E][+-])(\d+)$', lambda m: m.group(1) + m.group(2).zfill(3), s)
        if len(s) < width:
            s = s.rjust(width)
        elif len(s) > width:
            s = s[-width:]
        return s

    def write_from_template(self, path: Path, version_line: str, blocks, state_us: dict, gmtry_us: dict):
        nCv = int(gmtry_us['nCv'])
        nFc = int(gmtry_us.get('nFc', 0))
        ns = int(state_us['na'].shape[1])

        replacements = {}
        replacements['na'] = np.asarray(state_us['na']).reshape((nCv, ns), order='F')
        replacements['ne'] = np.asarray(state_us['ne']).reshape((nCv,), order='F')
        replacements['ua'] = np.asarray(state_us['ua']).reshape((nCv, ns), order='F')
        replacements['te'] = np.asarray(state_us['te']).reshape((nCv,), order='F')
        replacements['ti'] = np.asarray(state_us['ti']).reshape((nCv,), order='F')
        replacements['tn'] = np.asarray(state_us.get('tn', np.zeros_like(state_us['ti']))).reshape((nCv,), order='F')
        replacements['po'] = np.asarray(state_us['po']).reshape((nCv,), order='F')

        # match numeric tokens including Fortran D exponent
        number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?")

        def format_to_width(val, width):
            prec = 13
            s = f"{val:.{prec}E}"
            s = re.sub(r'([E][+-])(\d+)$', lambda mm: mm.group(1) + mm.group(2).zfill(3), s)
            while len(s) > width and prec > 0:
                prec -= 1
                s = f"{val:.{prec}E}"
                s = re.sub(r'([E][+-])(\d+)$', lambda mm: mm.group(1) + mm.group(2).zfill(3), s)
            if len(s) < width:
                s = s.rjust(width)
            elif len(s) > width:
                s = s[-width:]
            return s

        composed_lines = []
        if version_line:
            composed_lines.append(version_line)
        for block in blocks:
            if len(block) == 3:
                label_line, vals, raw_lines = block
                composed_lines.append(label_line)
                composed_lines.extend(raw_lines)
            else:
                label_line, vals = block
                composed_lines.append(label_line)
        composed_txt = '\n'.join(composed_lines) + '\n'

        repl_map = {k: iter(np.asarray(v).ravel(order='F').tolist()) for k, v in replacements.items()}

        out_parts = []
        pos = 0
        for block in blocks:
            if len(block) == 3:
                label_line, vals, raw_lines = block
                block_lines = [label_line] + raw_lines
            else:
                label_line, vals = block
                block_lines = [label_line]
            block_chunk = '\n'.join(block_lines) + '\n'
            idx = composed_txt.find(block_chunk, pos)
            if idx == -1:
                out_parts.append(composed_txt[pos:])
                pos = len(composed_txt)
                break
            out_parts.append(composed_txt[pos:idx])
            lab_token = label_line.strip().split()[-1]
            if lab_token.replace(' ', '') == 'nCv,nFc,ns' or lab_token == 'nCv,nFc,ns':
                if raw_lines:
                    rl = raw_lines[0]
                    tokens = list(re.finditer(r"\S+", rl))
                    newchars = list(rl)
                    nums = [nCv, nFc, ns]
                    for (m, num) in zip(tokens, nums):
                        s, e = m.start(), m.end()
                        w = e - s
                        txt = f"{int(num):{w}d}"
                        if len(txt) > w:
                            txt = txt[-w:]
                        newchars[s:e] = list(txt)
                    out_parts.append(label_line + '\n' + ''.join(newchars) + '\n')
                else:
                    out_parts.append(label_line + f"{nCv:12d} {nFc:12d} {ns:12d}\n")
            elif lab_token in repl_map and raw_lines:
                arr_iter = repl_map[lab_token]

                def repl_func(m):
                    try:
                        val = float(next(arr_iter))
                    except StopIteration:
                        val = 0.0
                    width = len(m.group(0))
                    return format_to_width(val, width)

                label_and_rest = block_chunk.split('\n', 1)
                if len(label_and_rest) == 2:
                    lbl, rest = label_and_rest
                    new_rest = number_re.sub(repl_func, rest)
                    out_parts.append(lbl + '\n' + new_rest)
                else:
                    out_parts.append(block_chunk)
            else:
                out_parts.append(block_chunk)
            pos = idx + len(block_chunk)

        out_parts.append(composed_txt[pos:])
        final_txt = ''.join(out_parts)
        Path(path).write_text(final_txt)
