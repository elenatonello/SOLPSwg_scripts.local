#!/usr/bin/env python3
"""Apply the wall-face part of the b2us "prep_geometry" step to an unstructured b2fgmtry text file.

b2ag does not read the b2us_prep_* parameters on the unstructured (traduit) path, so
b2us_prep_Qalfmin/Qalfmax/Pbs_reduce_fac in b2ag.dat have no effect. This script ports stage 1 of
that step from modules/B2.5/src/modules/b2us_prep.F (Qalf_mod_style = 0) to Python:

  wrk = |fcQalf(0)*fcBb(0)/fcBb(3)| (= Pbs/S) for the face of every guard cell with fcLbl > 0
  wrk >= Qalfmin : "wall face"
      wrk < Qalfmax : Pbs := Qalfmax*S*sign(Pbs);  Qalf(0) := Pbs/S*Bb(3)/Bb(0);  Qalf(1) keeps its sign
      then (--angles prep only, all wall faces): Qbet(0) = +-bz_sign, Qbet(1) = 0,
           Qgam = (|Qalf0|, +-Qalf1*bz_sign);  with --angles keep (default) Qgam is left alone and
           Qbet is recomputed from the new Qalf for the raised faces
  wrk <  Qalfmin : "grazing face": Pbs := Pbs_reduce_fac*Pbs, Qalf/Qbet recomputed from the new Pbs
  The scalars Qalfmin and Qalfmax stored in the file are set to the values used.

Not ported (yet): b2us_prep_mod_hc, b2us_prep_set_beta_0, b2us_prep_Qalf_mod_style = 1.

The script never changes the input file unless --inplace is given (then the original is kept once as
<file>.orig and every run starts from that copy, so the step is never applied twice). Unmodified
arrays are written back byte for byte; a round-trip self-test runs first.

Usage (from the baserun directory):
    modify_b2fgmtry_prep.py b2fgmtry --qmin 2e-3 --qmax 4e-3 --pbs-reduce 1e-6 -o b2fgmtry_prep
    modify_b2fgmtry_prep.py b2fgmtry --qmin 2e-3 --qmax 4e-3 --pbs-reduce 1e-6 --inplace
    modify_b2fgmtry_prep.py b2fgmtry --dry-run ...        # only print what would change
    modify_b2fgmtry_prep.py b2fgmtry_b2ag --qmin 1e-3 --qmax 8e-3 --scalars-only -o b2fgmtry_scalars   # edit only the two scalars
    modify_b2fgmtry_prep.py b2fgmtry --angles prep ...    # also overwrite Qbet/Qgam of all wall faces
                                                          # (what prep_geometry does; default is --angles keep)
"""
import argparse
import os
import shutil
import sys

import numpy as np

REALW, INTW, REALPL, INTPL = 22, 11, 6, 12


def fmt_real(v):
    m, e = ('%.13E' % v).split('E')
    return ('%' + str(REALW) + 's') % (m + 'E' + e[0] + e[1:].lstrip('0').rjust(3, '0'))


def fmt_int(v):
    return ('%' + str(INTW) + 'd') % v


class Gmtry:
    """b2fgmtry as a list of records; numeric arrays are parsed, everything else is kept verbatim."""

    def __init__(self, path):
        self.text = open(path, 'rb').read().decode('latin-1')
        lines = self.text.split('\n')
        self.recs = [['raw', lines[0]]]
        self.idx = {}
        i = 1
        while i < len(lines):
            ln = lines[i]
            if not ln.startswith('*cf:'):
                self.recs.append(['raw', ln])
                i += 1
                continue
            p = ln.split()
            typ, cnt, name = p[1], int(p[2]), p[3]
            i += 1
            if typ == 'char':
                self.recs.append(['raw', ln])
                self.recs.append(['raw', lines[i]])
                i += 1
                continue
            start = i
            vals = []
            while len(vals) < cnt:
                vals += lines[i].split()
                i += 1
            arr = np.array(vals, dtype=float if typ == 'real' else np.int64)
            self.idx[name] = len(self.recs)
            self.recs.append(['arr', ln, typ, name, arr, lines[start:i], False])

    def get(self, name):
        return self.recs[self.idx[name]][4]

    def set(self, name, values):
        r = self.recs[self.idx[name]]
        r[4] = np.asarray(values, dtype=r[4].dtype).reshape(r[4].shape)
        r[6] = True

    def render(self):
        out = []
        for r in self.recs:
            if r[0] == 'raw':
                out.append(r[1])
                continue
            out.append(r[1])
            if not r[6]:
                out.extend(r[5])
                continue
            f, per = (fmt_real, REALPL) if r[2] == 'real' else (fmt_int, INTPL)
            v = r[4]
            out.extend(''.join(f(x) for x in v[k:k + per]) for k in range(0, len(v), per))
        return '\n'.join(out)


def sign1(x):
    """Fortran sign(1.0, x)."""
    return np.where(x < 0, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('geometry', help='input b2fgmtry (unstructured, text)')
    ap.add_argument('--qmin', type=float, required=True, help='b2us_prep_Qalfmin')
    ap.add_argument('--qmax', type=float, required=True, help='b2us_prep_Qalfmax')
    ap.add_argument('--pbs-reduce', type=float, default=1e-3,
                    help='b2us_prep_Pbs_reduce_fac for grazing faces (code default 1e-3)')
    ap.add_argument('--angles', choices=('prep', 'keep'), default='keep',
                    help="'keep' (default): leave the stored connector angle (Qgam) of wall faces alone and "
                         "only update Pbs/Qalf (and Qbet = f(Qalf, Qgam) where Qalf changed); "
                         "'prep': overwrite Qbet/Qgam of ALL wall faces (beta = 0, gamma = alpha) as "
                         "prep_geometry does - on the traduit grid this changes ~130 more faces")
    ap.add_argument('--scalars-only', action='store_true',
                    help='only store Qalfmin/Qalfmax in the file; leave Pbs and all angles untouched '
                         '(the file then plots exactly like the input with --qmin/--qmax)')
    ap.add_argument('-o', '--output', default=None, help='output file (default <input>_prep)')
    ap.add_argument('--inplace', action='store_true', help='overwrite the input (original kept as .orig)')
    ap.add_argument('--dry-run', action='store_true', help='print the summary, write nothing')
    a = ap.parse_args()

    src = a.geometry
    if a.inplace:
        orig = a.geometry + '.orig'
        if os.path.exists(orig):
            src = orig
            print(f'using the saved original {orig}')
        elif not a.dry_run:
            shutil.copyfile(a.geometry, orig)
            src = orig
            print(f'saved the original as {orig}')
    out = a.geometry if a.inplace else (a.output or a.geometry + '_prep')
    if not a.inplace and os.path.abspath(out) == os.path.abspath(a.geometry) and not a.dry_run:
        sys.exit(f'refusing to overwrite the input {a.geometry} (-o points to it): the original would be lost '
                 'and a second run would apply the reduction twice. Use a different -o, or --inplace '
                 '(keeps the original as .orig).')

    G = Gmtry(src)
    if G.render() != G.text:
        sys.exit('round-trip self-test failed: this file cannot be rewritten byte for byte; aborting')
    print('round-trip self-test: OK (unmodified file reproduced byte for byte)')

    hdr = G.get('nCi,nCg,nCv,nFc,nVx,nFs,nFt')
    nCi, nCg, nCv, nFc = (int(hdr[k]) for k in (0, 1, 2, 3))
    cvFcP = G.get('cvFcP').reshape(2, -1)
    cvFc = G.get('cvFc')
    fcLbl = G.get('fcLbl')
    S = G.get('fcS')
    Bb = G.get('fcBb').reshape(4, nFc)
    Pbs0 = G.get('fcPbs')
    Qa0 = G.get('fcQalf').reshape(2, nFc)
    Qb0 = G.get('fcQbet').reshape(2, nFc)
    Qg0 = G.get('fcQgam').reshape(2, nFc)
    Eb = G.get('cvEb').reshape(3, nCv)
    old_qmin = float(G.get('Qalfmin')[0]) if 'Qalfmin' in G.idx else None
    old_qmax = float(G.get('Qalfmax')[0]) if 'Qalfmax' in G.idx else None
    if old_qmin is None:
        sys.exit('this b2fgmtry has no Qalfmin/Qalfmax arrays (older format); not supported')

    Pbs, Qa, Qb, Qg = Pbs0.copy(), Qa0.copy(), Qb0.copy(), Qg0.copy()
    gcells = np.arange(nCi, nCv)
    gf = cvFc[cvFcP[0, gcells] - 1] - 1                      # the face of each guard cell
    keep = fcLbl[gf] > 0
    gcells, gf = gcells[keep], gf[keep]
    bz = sign1(Eb[2, gcells])
    wrk = np.abs(Qa0[0, gf] * Bb[0, gf] / Bb[3, gf])
    wall = wrk >= a.qmin
    band = wall & (wrk < a.qmax)
    graz = ~wall

    # wall faces below Qalfmax: contact area raised to Qalfmax*S
    f = gf[band]
    Pbs[f] = a.qmax * S[f] * sign1(Pbs0[f])
    Qa[0, f] = np.clip(Pbs[f] / S[f] * Bb[3, f] / Bb[0, f], -1.0, 1.0)
    Qa[1, f] = np.where(Qa0[1, f] < 0, -1.0, 1.0) * np.sqrt(1.0 - Qa[0, f] ** 2)
    if a.angles == 'prep':
        # wall faces: angles as in prep_geometry (beta = 0, gamma = alpha)
        f, b = gf[wall], bz[wall]
        neg = Qa[0, f] < 0
        Qb[0, f] = np.where(neg, -1.0, 1.0) * b
        Qg[0, f] = np.where(neg, -Qa[0, f], Qa[0, f])
        Qg[1, f] = np.where(neg, -Qa[1, f], Qa[1, f]) * b
        Qb[1, f] = 0.0
    else:
        # keep the stored connector angle (Qgam); only restore Qbet = f(Qalf, Qgam) where Qalf changed.
        # (this relation holds to 1e-14 for all boundary faces of the unmodified file)
        f, b = gf[band], bz[band]
        Qb[0, f] = (Qg0[0, f] * Qa[0, f] + Qg0[1, f] * Qa[1, f] * b) * b
        Qb[1, f] = Qg0[1, f] * Qa[0, f] - Qg0[0, f] * Qa[1, f] * b
    # grazing faces: reduced contact area, angles recomputed from it
    f, b = gf[graz], bz[graz]
    Pbs[f] = a.pbs_reduce * Pbs0[f]
    Qa[0, f] = np.clip(Pbs[f] / S[f] * Bb[3, f] / Bb[0, f], -1.0, 1.0)
    Qa[1, f] = np.where(Qa0[1, f] < 0, -1.0, 1.0) * np.sqrt(1.0 - Qa[0, f] ** 2)
    Qb[0, f] = (Qg0[0, f] * Qa[0, f] + Qg0[1, f] * Qa[1, f] * b) * b
    Qb[1, f] = Qg0[1, f] * Qa[0, f] - Qg0[0, f] * Qa[1, f] * b

    # ---- summary
    print(f'\nQalfmin {old_qmin:g} -> {a.qmin:g}, Qalfmax {old_qmax:g} -> {a.qmax:g}, '
          f'Pbs_reduce_fac = {a.pbs_reduce:g}, angles = {a.angles}')
    print(f'guard faces with fcLbl > 0: {len(gf)}')
    print(f'  wall faces (wrk >= Qalfmin):          {int(wall.sum()):4d}   of which raised to Qalfmax*S: {int(band.sum())}')
    print(f'  grazing faces (wrk < Qalfmin):        {int(graz.sum()):4d}   (of which wrk == 0: {int((wrk[graz] == 0).sum())})')
    for name, new, old in (('fcPbs', Pbs, Pbs0), ('fcQalf(0)', Qa[0], Qa0[0]), ('fcQalf(1)', Qa[1], Qa0[1]),
                           ('fcQbet(0)', Qb[0], Qb0[0]), ('fcQbet(1)', Qb[1], Qb0[1]),
                           ('fcQgam(0)', Qg[0], Qg0[0]), ('fcQgam(1)', Qg[1], Qg0[1])):
        d = np.abs(new - old)
        print(f'  {name:<10s}: {int((d > 1e-14).sum()):5d} faces change, max |change| {d.max():.3g}')
    print('  grazing faces that are not core-type (wrk > 0): R, Z not available here; use plot_qalf_wall.py')
    # checks that prep_geometry asserts
    ok = (Qg[1].min() > -1.0 and Qg[1].max() < 1.0 and Qg[0].min() > 0.0 and Qg[0].max() <= 1.0)
    print(f'  prep_geometry range check on fcQgam: {"OK" if ok else "WOULD FAIL (min Qgam0 %.3g)" % Qg[0].min()}')
    norms = max(np.abs(np.hypot(*Qa) - 1).max(), np.abs(np.hypot(*Qb) - 1).max(), np.abs(np.hypot(*Qg) - 1).max())
    print(f'  unit-vector check on Qalf/Qbet/Qgam: max deviation {norms:.1e}')
    if a.dry_run:
        print('\n--dry-run: nothing written')
        return

    if a.scalars_only:
        print('\n--scalars-only: fcPbs, fcQalf, fcQbet, fcQgam are NOT modified (the summary above shows what the full prep would do)')
    else:
        G.set('fcPbs', Pbs)
        G.set('fcQalf', Qa.reshape(-1))
        G.set('fcQbet', Qb.reshape(-1))
        G.set('fcQgam', Qg.reshape(-1))
    G.set('Qalfmin', [a.qmin])
    G.set('Qalfmax', [a.qmax])
    with open(out, 'wb') as fh:
        fh.write(G.render().encode('latin-1'))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
