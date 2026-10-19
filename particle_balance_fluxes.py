#! /usr/bin/env python
"""Per-region breakdown of the wide-grid particle balance (companion of particle_balance_widegrid.py).

For each element (species of one element summed) it draws two panels versus time:
    top     fnareg of every boundary face region (raw sign: > 0 = particles leaving the domain); regions with
            non-zero balance weight are solid lines, with --all the weight-0 (internal) regions are drawn dotted
    bottom  the balance terms: net inflow (= - sum of the boundary fnareg), b2stbr (EIRENE source, charged
            states only), b2sext (external source) and the balance error fna + b2stbr + b2sext (black dashed).
            dN/dt is not included here; see particle_balance_widegrid.py for the full balance.
The balance weight is -1 on all boundary regions and 0 on internal ones (the face normal of every boundary face
points out of the domain on the wide grid). The boundary regions are the face regions whose faces all have a
guard cell as second cell, read from b2fgmtry (the file given as argument, else ./b2fgmtry, baserun/b2fgmtry or
../baserun/b2fgmtry). Without a b2fgmtry the SN default list is used (as in particle_balance_widegrid.py), WITHOUT checking
the topology, so give the file for any other grid. Region names (NAMES) are only known for SN; otherwise just the number is shown.
Also printed, per element: the time average over the last 10% of the outputs of each region flux, of the
weighted flux sum, b2stbr and b2sext.

Input: b2mn.exe.dir/b2tallies.nc if readable, else ./b2tallies.nc. A window is opened (TkAgg, needs a display); if DISPLAY
is not set each figure is saved to particle_balance_fluxes_<element>.png in the current directory before the window is requested.

Usage:
    particle_balance_fluxes.py [b2fgmtry] [element ...] [--all]
        element   only these elements (e.g. D C); default: all
        --all     also plot the regions with zero weight
"""
from __future__ import print_function
import netCDF4
import os
import matplotlib
matplotlib.use("TkAgg")
#if not os.getenv("DISPLAY"): matplotlib.use('Agg')
import matplotlib.pylab as plt
import sys
import numpy
import re

if any(a in ('-h', '--help') for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

args = [a for a in sys.argv[1:] if not a.startswith('--')]
show_all = '--all' in sys.argv

if os.access('b2mn.exe.dir/b2tallies.nc', os.R_OK):
    f = netCDF4.Dataset('b2mn.exe.dir/b2tallies.nc','r')
else:
    f = netCDF4.Dataset('b2tallies.nc','r')
vreg = f.dimensions['vreg'].size
freg = f.dimensions['freg'].size
times = numpy.ma.filled(f.variables['times'][:], 0)
species_names = f.variables['species']
species = [b''.join(species_names[i,:]).strip().decode('utf-8') for i in range(species_names.shape[0])]
elements = [re.sub('[-+0-9]','',s) for s in species]
mask = [re.match('[a-zA-Z]+0$',s) != None for s in species]
s = 0
bounds = []
for i in range(mask.count(True)-1):
    e = mask.index(True,s+1)
    bounds += [[s,e]]
    s = e
bounds += [[s,len(mask)]]

fnareg = numpy.ma.filled(f.variables['fnareg'][:], 0)
b2stbr_sna_reg = numpy.ma.filled(f.variables['b2stbr_sna_reg'][:], 0)
b2sext_sna_reg = numpy.ma.filled(f.variables['b2sext_sna_reg'][:], 0)

# Wide grid: the normal of every boundary face points out of the domain (second cell of the
# face is a guard cell), so fnareg > 0 on a boundary region means particles leaving the plasma.
# The balance weight is therefore -1 on all boundary regions and 0 on internal ones (the
# structured-grid +-1 x/y weights of particle_balance.py do not apply). The boundary regions
# are read from b2fgmtry (argument, ./, baserun/ or ../baserun/); SN defaults if none is found.
SN_BOUNDARY = [1,4,7,8,9,11,12,13]
SN_NAMES = {1:'W target', 2:'W div entrance', 3:'E div entrance', 4:'E target',
            5:'core cut', 6:'div cut', 7:'W PFR wall', 8:'core boundary',
            9:'E PFR wall', 10:'separatrix', 11:'W div-main wall', 12:'SOL-main wall',
            13:'E div-main wall'}

def read_cf(filename, names):
    out = {}
    cur = None
    buf = []
    for line in open(filename):
        if line.startswith('*cf:'):
            if cur in names: out[cur] = numpy.array(buf, dtype=float)
            p = line.split()
            cur = p[3] if len(p) > 3 else None
            buf = []
        elif cur in names:
            buf += line.split()
    if cur in names: out[cur] = numpy.array(buf, dtype=float)
    return out

def boundary_regions():
    for fn in args + ['b2fgmtry', 'baserun/b2fgmtry', '../baserun/b2fgmtry']:
        if not os.access(fn, os.R_OK): continue
        d = read_cf(fn, ['nCi,nCg,nCv,nFc,nVx,nFs,nFt', 'fcCv', 'fcReg'])
        if len(d) < 3: continue
        nCi = int(d['nCi,nCg,nCv,nFc,nVx,nFs,nFt'][0])
        fcCv = d['fcCv'].astype(int).reshape(2,-1).T
        fcReg = d['fcReg'].astype(int)
        bnd = []
        for r in range(1, freg):
            fs = numpy.where(fcReg == r)[0]
            if len(fs) > 0 and (fcCv[fs,1]-1 >= nCi).all(): bnd.append(r)
        print('Boundary regions from %s:' % fn, bnd)
        args.remove(fn) if fn in args else None
        return bnd
    print('b2fgmtry not found, using SN default boundary regions', SN_BOUNDARY)
    return SN_BOUNDARY

FULL_XY = numpy.zeros(freg, dtype=int)
FULL_XY[boundary_regions()] = -1
NAMES = SN_NAMES if vreg == 5 else {}

regions = [r for r in range(1,freg) if show_all or FULL_XY[r] != 0]

for b in bounds:
    el = elements[b[0]]
    if args and el not in args: continue
    fna_reg = fnareg[:,b[0]:b[1],:].sum(axis=1)            # time x freg
    fna = (fna_reg*FULL_XY).sum(axis=1)

    b2stbr = b2stbr_sna_reg[:,b[0]:b[1],0].copy()
    b2stbr[:,0] = 0 # remove the neutral
    if b2stbr.shape[1] > 2: b2stbr[:,-1] = 0 # remove the fully stripped species for He and up
    b2stbr = b2stbr.sum(axis=1)
    b2sext = b2sext_sna_reg[:,b[0]:b[1],0].sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))
    for r in regions:
        label = 'fcReg %d %s (w=%+d)' % (r, NAMES.get(r,''), FULL_XY[r])
        ax1.plot(times, fna_reg[:,r], label=label, ls='-' if FULL_XY[r] != 0 else ':')
    ax1.set_ylabel('fnareg [1/s]')
    ax1.set_title('%s: particle flux per face region (raw sign: >0 = leaving the domain at boundaries)' % el)
    ax1.legend(loc='best', fontsize='small', ncol=2)

    ax2.plot(times, fna, label='net inflow (-sum boundary fnareg)')
    ax2.plot(times, b2stbr, label='b2stbr')
    ax2.plot(times, b2sext, label='b2sext')
    ax2.plot(times, fna+b2stbr+b2sext, 'k--', label='balance error')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('[1/s]')
    ax2.legend(loc='best', fontsize='small')
    fig.tight_layout()

    print(el, 'time-averaged flux per region (last 10% of run):')
    n = max(1, len(times)//10)
    for r in regions:
        print('  fcReg %2d %-18s w=%+d  %12.4e' % (r, NAMES.get(r,''), FULL_XY[r], fna_reg[-n:,r].mean()))
    print('  weighted flux sum %12.4e  b2stbr %12.4e  b2sext %12.4e' % (fna[-n:].mean(), b2stbr[-n:].mean(), b2sext[-n:].mean()))
    if not os.getenv("DISPLAY"): fig.savefig('particle_balance_fluxes_%s.png' % el, dpi=120)
plt.show()
