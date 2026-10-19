#! /usr/bin/env python
"""Particle balance of a B2.5 run on the WIDE (unstructured) grid, per element.

On the wide grid the face normal of every boundary face points out of the domain (second cell of the
face, fcCv(iFc,2), is a guard cell), so a positive fnareg on a boundary region means particles leaving the
plasma. The balance is therefore
    - sum_{boundary regions} fnareg + b2stbr + b2sext - dN/dt = 0
with weight W = -1 on all boundary regions and 0 on internal ones, instead of the +-1 x/y-orientation
weights of particle_balance.py (valid for the structured grid only). Per element (species of one element
summed, neutral first):
    fnareg  face particle flux per face region (b2tallies.nc)
    b2stbr  EIRENE particle source, whole domain, charged states only (neutral and, for He and heavier, the
            fully stripped state are left out)
    b2sext  external particle source, whole domain
    N       ion content (nareg of the charged states; the neutral is excluded, as for b2stbr);
            dN/dt is its time derivative (numpy.gradient)

Three panels versus time, one line per element:
    1. (fna + b2stbr + b2sext) / max|fnareg|            normalised error, dN/dt neglected
    2. (fna + b2stbr + b2sext - dN/dt) / max|fnareg|    same with dN/dt (the proper balance)
    3. (fna + b2stbr + b2sext - dN/dt) / gross(t)       instantaneous error relative to the gross throughput,
                                                        gross = sum of the absolute sizes of all terms
The printed table gives, per element, the means over the last 10% of the outputs (<flux in>, <b2stbr+b2sext>,
<dN/dt>) and the three errors above (err/norm, err/norm with dN/dt, err/gross(t) with dN/dt).

The boundary regions are the face regions whose faces all have a guard cell as second cell, read from
b2fgmtry (the file given as argument, else ./b2fgmtry, baserun/b2fgmtry or ../baserun/b2fgmtry); a warning is
printed for regions with mixed orientation. If no b2fgmtry is found the SN defaults (SN_BOUNDARY) are used
WITHOUT checking the topology, so give the file for any other grid.
Input: b2mn.exe.dir/b2tallies.nc if readable, else ./b2tallies.nc. A window is opened (TkAgg, needs a display); if DISPLAY
is not set the figure is saved to particle_balance_widegrid.png in the current directory before the window is requested.

Usage:
    particle_balance_widegrid.py [b2fgmtry]
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

SN_BOUNDARY = [1,4,7,8,9,11,12,13]   # targets, PFR walls, core boundary, main chamber walls

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

def boundary_regions(freg):
    """Face regions whose faces all have a guard cell as second cell (outward normal)."""
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
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
            if len(fs) == 0: continue
            g2 = (fcCv[fs,1]-1 >= nCi)
            g1 = (fcCv[fs,0]-1 >= nCi)
            if g1.any(): print('WARNING: region %d has faces with the guard cell first (mixed orientation)' % r)
            if g2.all(): bnd.append(r)
        print('Boundary regions from %s:' % fn, bnd)
        return bnd
    print('b2fgmtry not found, using SN default boundary regions', SN_BOUNDARY)
    return SN_BOUNDARY

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
nareg = numpy.ma.filled(f.variables['nareg'][:], 0)

W = numpy.zeros(freg)
W[boundary_regions(freg)] = -1

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9,11))
for b in bounds:
    el = elements[b[0]]
    fr = fnareg[:,b[0]:b[1],:].sum(axis=1)
    fna = (fr*W).sum(axis=1)   # net inflow through the boundary

    b2stbr = b2stbr_sna_reg[:,b[0]:b[1],0].copy()
    b2stbr[:,0] = 0 # remove the neutral
    if b2stbr.shape[1] > 2: b2stbr[:,-1] = 0 # remove the fully stripped species for He and up
    b2stbr = b2stbr.sum(axis=1)
    b2sext = b2sext_sna_reg[:,b[0]:b[1],0].sum(axis=1)

    N = nareg[:,b[0]+1:b[1],0].sum(axis=1)    # ion content (neutral excluded, as for b2stbr)
    dNdt = numpy.gradient(N, times) if len(times) > 1 else numpy.zeros_like(N)

    fna_norm = numpy.max(numpy.abs(fnareg[:,b[0]:b[1],:]))
    err = (fna+b2stbr+b2sext)/fna_norm
    err_dn = (fna+b2stbr+b2sext-dNdt)/fna_norm
    # Instantaneous error: residual relative to the gross throughput at the same time, i.e. the sum
    # of the absolute sizes of all terms in the balance (boundary fluxes, sources and dN/dt).
    gross = (numpy.abs(fr)*numpy.abs(W)).sum(axis=1) + numpy.abs(b2stbr) + numpy.abs(b2sext) + numpy.abs(dNdt)
    err_inst = numpy.divide(fna+b2stbr+b2sext-dNdt, gross, out=numpy.full_like(gross, numpy.nan), where=gross > 0)
    n = max(1, len(times)//10)
    print('%-3s norm %.3e  <flux in> %.3e  <b2stbr+b2sext> %.3e  <dN/dt> %.3e  err/norm %.4f  err/norm (with dN/dt) %.4f  err/gross(t) (with dN/dt) %.4f'
          % (el, fna_norm, fna[-n:].mean(), (b2stbr+b2sext)[-n:].mean(), dNdt[-n:].mean(), err[-n:].mean(), err_dn[-n:].mean(), numpy.nanmean(err_inst[-n:])))
    l, = ax1.plot(times, err, label=el)
    ax2.plot(times, err_dn, label=el, color=l.get_color())
    ax3.plot(times, err_inst, label=el, color=l.get_color())
ax1.set_ylabel('normalised particle error')
ax1.set_title('Wide-grid particle balance (outward boundary faces)')
ax1.legend(loc=0)
ax2.set_ylabel('normalised error incl. dN/dt')
ax3.set_ylabel('instantaneous error incl. dN/dt\n(relative to gross throughput at t)')
ax3.set_xlabel('time [s]')
fig.tight_layout()
if not os.getenv("DISPLAY"): fig.savefig('particle_balance_widegrid.png', dpi=120)
plt.show()
