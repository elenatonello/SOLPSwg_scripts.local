#! /usr/bin/env python
"""Particle balance error of the CORE region on the STRUCTURED grid (copy of scripts/particle_balance_core.py).

Same idea as particle_balance.py, but only for the core volume region: per element it plots versus time
    (fna + b2stbr) / max|fnareg|
with
    fna    = sum of W * fnareg over the two face regions that bound the core, with opposite signs
             (SN: face regions 8 = core boundary and 10 = separatrix; W = FULL_XY below), summed over
             the species of the element
    b2stbr = B2.5 particle source from EIRENE in volume region 1 (b2stbr_sna_reg[..., 1], the core),
             summed over the charged states (neutral and, for He and heavier, the fully stripped state
             are left out)
There is no b2sext term and dN/dt is not subtracted. Input, window and structured-grid caveat as in
particle_balance.py (not valid for the wide grid).

Printed per element: name, normalisation, mean of (fna+b2stbr), mean normalised error, mean fna, mean b2stbr.
Topologies coded: SN, limiter, stellarator island, LFS snowflake, CDN, DDN (no linear).

Usage:
    particle_balance_core.py
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
import subprocess
import re

if any(a in ('-h', '--help') for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if os.access('b2mn.exe.dir/b2tallies.nc', os.R_OK):
    f = netCDF4.Dataset('b2mn.exe.dir/b2tallies.nc','r')
else:
    f = netCDF4.Dataset('b2tallies.nc','r')
vreg = f.dimensions['vreg'].size
freg = f.dimensions['freg'].size
ns = f.dimensions['ns'].size
time = f.dimensions['time'].size
times = f.variables['times']
species_names=f.variables['species']
species=[b''.join(species_names[i,:]).strip().decode('utf-8') for i in range(species_names.shape[0])]
elements=[re.sub('[-+0-9]','',species[i]) for i in range(len(species))]
mask=[re.match('[a-zA-Z]+0$',species[i])!=None for i in range(len(species))]
s=0
bounds=[]
for i in range(mask.count(True)-1):
    e=mask.index(True,s+1)
    bounds+=[[s,e]]
    s=e
bounds+=[[s,len(mask)]]

fnareg = f.variables['fnareg']
b2stbr_sna_reg = f.variables['b2stbr_sna_reg']

if vreg == 5: # SN topology
    FULL_XY = numpy.array([0,0,0,0,0,0,0,0,1,0,-1,0,0,0])
elif vreg == 3: # Limiter topology
    FULL_XY = numpy.array([0,0,0,0,1,-1,0])
elif vreg == 6: # Stellarator island topology
    FULL_XY = numpy.array([0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0])
elif vreg == 8: # LFS Snowflake topology
    FULL_XY = numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0])
elif freg == 27: # CDN topology
    FULL_XY = numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,1,0,-1,0,0,0])
elif freg == 28: # DDN topology
    FULL_XY = numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,1,0,-1,0,0,0])
else:
    raise ValueError('Value of vreg=%s not currently coded' % vreg)

for i in range(len(bounds)):
    fna = (fnareg[:,bounds[i][0]:bounds[i][1],:].sum(axis=1)*FULL_XY).sum(axis=1)
    b2stbr = b2stbr_sna_reg[:,bounds[i][0]:bounds[i][1],1].copy()
    b2stbr[:,0] = 0 # remove the neutral
    if b2stbr.shape[1] > 2: b2stbr[:,-1] = 0 # remove the fully stripped species for He and up
    b2stbr = b2stbr.sum(axis=1)
    fna_norm = numpy.max([numpy.max(numpy.abs(fnareg[:,bounds[i][0]:bounds[i][1],:]))])
    print(elements[bounds[i][0]], fna_norm, (fna+b2stbr).mean(), ((fna+b2stbr)/fna_norm).mean(), (fna).mean(), (b2stbr).mean())
    plt.plot(times[:],((fna+b2stbr)/fna_norm), label=elements[bounds[i][0]])
plt.xlabel('time [s]')
plt.ylabel('normalised particle error')
plt.title('Normalised particle error (core)')
plt.legend(loc=0)
plt.show()
