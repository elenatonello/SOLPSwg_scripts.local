#!/usr/bin/env python3
"""Track the onset and spread of a Te/Ti/po runaway through the plasmastate.XXXX snapshots of a run.

Per snapshot it prints the extrema of Te, Ti, po, |po|/Te and the parallel current fch_p, the number of
cells beyond the thresholds, and where the worst cell is. Then it follows the cells that go bad first
(po, Te, Ti, ne) through all snapshots, and ranks the quantities by which crossed its threshold first.

Run from the run directory (geometry from b2mn.exe.dir/b2fgmtry, ./b2fgmtry or ../baserun/b2fgmtry).

Usage:
    track_runaway.py                    # all snapshots, default thresholds
    track_runaway.py --te 500 --po 100  # thresholds [eV], [V]
    track_runaway.py --follow 12        # follow the 12 first-affected cells
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_plasmastate import EV, read_cf, read_plasmastate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-d', '--dir', default='b2mn.exe.dir')
    ap.add_argument('--te', type=float, default=1000.0, help='Te threshold [eV]')
    ap.add_argument('--ti', type=float, default=1000.0, help='Ti threshold [eV]')
    ap.add_argument('--po', type=float, default=100.0, help='|po| threshold [V]')
    ap.add_argument('--rpo', type=float, default=50.0, help='|po|/Te threshold')
    ap.add_argument('--follow', type=int, default=8, help='number of first-affected cells to follow')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, 'plasmastate.*')))
    if not files:
        sys.exit(f'no plasmastate files in {a.dir}')
    gm = next((p for p in (os.path.join(a.dir, 'b2fgmtry'), 'b2fgmtry', '../baserun/b2fgmtry')
               if os.path.exists(p)), None)
    if gm is None:
        sys.exit('b2fgmtry not found')
    geo = read_cf(gm, {'cvX', 'cvY', 'fcCv', 'nCi,nCg,nCv,nFc,nVx,nFs,nFt'})
    nCv = len(geo['cvX'])
    nFc = len(geo['fcCv']) // 2
    nCi = int(geo['nCi,nCg,nCv,nFc,nVx,nFs,nFt'][0])
    fcCv = geo['fcCv'].reshape(2, nFc) - 1

    snaps = []
    print(f'{nCv} cells ({nCi} interior), {nFc} faces, {len(files)} snapshots\n')
    hdr = (f'{"id":>4} {"t[s]":>9} {"maxTe":>9} {"maxTi":>9} {"po min":>10} {"po max":>10} '
           f'{"|po|/Te":>8} {"max|fch_p|":>10} | n(Te>) n(Ti>) n(|po|>) n(r>)  worst cell')
    print(hdr)
    for fn in files:
        s = read_plasmastate(fn)
        te, ti, po = s['te'] / EV, s['ti'] / EV, s['po']
        fp = np.abs(s['fch_p'][:nFc]) if 'fch_p' in s and len(s['fch_p']) >= nFc else np.zeros(1)
        r = np.abs(po) / te
        snaps.append(dict(id=fn.rsplit('.', 1)[1], t=float(s['time'][0]) if 'time' in s else np.nan,
                          te=te, ti=ti, po=po, ne=s['ne'], fp=fp))
        bad = (te > a.te) | (ti > a.ti) | (np.abs(po) > a.po)
        w = int(np.argmax(np.where(bad, np.maximum(np.maximum(te / a.te, ti / a.ti), np.abs(po) / a.po), 0)))
        print(f'{snaps[-1]["id"]:>4} {snaps[-1]["t"]:9.5f} {te.max():9.3g} {ti.max():9.3g} {po.min():10.3g} '
              f'{po.max():10.3g} {r.max():8.3g} {fp.max():10.3g} | {int((te > a.te).sum()):6d} '
              f'{int((ti > a.ti).sum()):6d} {int((np.abs(po) > a.po).sum()):8d} {int((r > a.rpo).sum()):5d}  '
              + (f'{w} (R={geo["cvX"][w]:.3f}, Z={geo["cvY"][w]:.3f})' if bad.any() else '-'))

    # first snapshot at which each indicator crosses its threshold
    print('\nFirst snapshot at which each indicator crosses its threshold:')
    inds = {f'Te>{a.te:g} eV': lambda d: (d['te'] > a.te).any(),
            f'Ti>{a.ti:g} eV': lambda d: (d['ti'] > a.ti).any(),
            f'|po|>{a.po:g} V': lambda d: (np.abs(d['po']) > a.po).any(),
            f'|po|/Te>{a.rpo:g}': lambda d: (np.abs(d['po']) / d['te'] > a.rpo).any()}
    for name, f in inds.items():
        first = next((d for d in snaps if f(d)), None)
        print(f'  {name:<16} ' + (f'snapshot {first["id"]} (t = {first["t"]:.5f} s)' if first else 'never'))

    # cells affected first
    first_bad = None
    for d in snaps:
        bad = (d['te'] > a.te) | (d['ti'] > a.ti) | (np.abs(d['po']) > a.po)
        if bad.any():
            first_bad = d
            break
    if first_bad is None:
        print('\nNo snapshot crosses the thresholds.')
        return
    bad = np.where((first_bad['te'] > a.te) | (first_bad['ti'] > a.ti) | (np.abs(first_bad['po']) > a.po))[0]
    sc = np.maximum(np.maximum(first_bad['te'][bad] / a.te, first_bad['ti'][bad] / a.ti),
                    np.abs(first_bad['po'][bad]) / a.po)
    cells = bad[np.argsort(-sc)][:a.follow]
    print(f'\nSnapshot {first_bad["id"]}: {len(bad)} affected cells; following the {len(cells)} worst')
    for c in cells:
        print(f'\ncell {c} ({"guard" if c >= nCi else "interior"}) R={geo["cvX"][c]:.3f} Z={geo["cvY"][c]:.3f}')
        print(f'  {"id":>4} {"t[s]":>9} {"Te[eV]":>9} {"Ti[eV]":>9} {"po[V]":>10} {"ne":>10}')
        for d in snaps:
            print(f'  {d["id"]:>4} {d["t"]:9.5f} {d["te"][c]:9.3g} {d["ti"][c]:9.3g} {d["po"][c]:10.3g} '
                  f'{d["ne"][c]:10.3e}')


if __name__ == '__main__':
    main()
