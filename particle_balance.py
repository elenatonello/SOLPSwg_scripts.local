#! /usr/bin/env python
"""Particle balance error of a B2.5 run on the STRUCTURED grid (copy of scripts/particle_balance.py).

For each element (D, C, ...; the species of one element are grouped, neutral first) it plots, versus
time, the particle balance residual
    fna + b2stbr + b2sext
normalised to the largest |fnareg| of that element, where
    fna    = sum over the boundary face regions of  W * fnareg   (net particle inflow through the
             boundary, summed over the species of the element; W = FULL_XY below)
    b2stbr = B2.5 particle source from EIRENE (b2stbr_sna_reg, whole-domain region 0), summed over the
             charged states; the neutral and the fully stripped state (He and heavier) are left out
    b2sext = external particle source (b2sext_sna_reg, region 0)
A perfectly converged run gives 0; a drift away from 0 means the particle balance is not closed.
dN/dt is NOT subtracted, so it is only meaningful for a quasi-steady run (the wide-grid version,
particle_balance_widegrid.py, also shows the error with dN/dt).

FULL_XY holds the weight W of every face region of b2tallies.nc (index = region number, 0 = whole
domain): non-zero only for the boundary regions, with the sign (+1/-1) of the x/y orientation of the
faces, chosen per topology (vreg, freg). It is the structured-grid convention and is WRONG for the
wide grid, where all boundary faces point outwards: use particle_balance_widegrid.py there.

Input: b2mn.exe.dir/b2tallies.nc if readable, else ./b2tallies.nc (run from the run directory).
Printed per element: name, normalisation, mean error, mean normalised error (means over all outputs).
Opens a plot window (TkAgg). Topologies coded: SN, limiter, linear, stellarator island, LFS snowflake,
CDN, DDN; anything else raises an error.

Usage:
    particle_balance.py
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
b2sext_sna_reg = f.variables['b2sext_sna_reg']

if vreg == 5: # SN topology
    FULL_XY = numpy.array([0,1,0,0,-1,0,0,1,1,1,0,-1,-1,-1])
elif vreg == 3: # Limiter topology
    FULL_XY = numpy.array([0,1,-1,0,1,0,-1])
elif vreg == 2: # Linear topology
    FULL_XY = numpy.array([0,1,-1,1,-1])
elif vreg == 6: # Stellarator island topology
    FULL_XY = numpy.array([0,1,0,0,-1,0,0,0,1,1,1,0,0,-1,0,0])
elif vreg == 8: # LFS Snowflake topology
    FULL_XY = numpy.array([0,1,0,0,-1,1,0,0,-1,0,0,0,0,0,1,1,1,0,-1,-1,-1,1,1,1,-1,-1,-1])
elif freg == 27 : # CDN topology
    FULL_XY = numpy.array([0,1,0,0,-1,1,0,0,-1,0,0,0,0,1,1,1,0,-1,-1,-1,1,1,1,0,-1,-1,-1])
elif freg == 28 : # DDN topology
    FULL_XY = numpy.array([0,1,0,0,-1,1,0,0,-1,0,0,0,0,0,1,1,1,0,-1,-1,-1,1,1,1,0,-1,-1,-1])
else:
    raise ValueError('Value of vreg=%s not currently coded' % vreg)

for i in range(len(bounds)):
    fna = (fnareg[:,bounds[i][0]:bounds[i][1],:].sum(axis=1)*FULL_XY).sum(axis=1)
    b2stbr = b2stbr_sna_reg[:,bounds[i][0]:bounds[i][1],0].copy()
    b2stbr[:,0] = 0 # remove the neutral
    if b2stbr.shape[1] > 2: b2stbr[:,-1] = 0 # remove the fully stripped species for He and up
    b2stbr = b2stbr.sum(axis=1)

    b2sext = b2sext_sna_reg[:,bounds[i][0]:bounds[i][1],0].copy()
    b2sext = b2sext.sum(axis=1)

    fna_norm = numpy.max(numpy.abs(fnareg[:,bounds[i][0]:bounds[i][1],:]))
    print(elements[bounds[i][0]], fna_norm, (fna+b2stbr+b2sext).mean(), ((fna+b2stbr+b2sext)/fna_norm).mean())
    plt.plot(times[:],((fna+b2stbr+b2sext)/fna_norm), label=elements[bounds[i][0]])
plt.xlabel('time [s]')
plt.ylabel('normalised particle error')
plt.title('Normalised particle error')
plt.legend(loc=0)
plt.show()
