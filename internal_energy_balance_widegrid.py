#!/usr/bin/env python
"""Internal energy balance for wide-grid runs (python version of scripts/internal_energy_balance_coupled).

internal_energy_balance_coupled picks the boundary fluxes of fhireg+fhereg by hand from the columns of
the structured-grid regions (E, N, W, S with +-1 orientation weights). On the wide grid the face normal of
every boundary face points out of the domain (second cell of the face, fcCv(iFc,2), is a guard cell), so
fhireg+fhereg on a boundary region is power leaving the plasma and the balance reads
    sum_{boundary regions} (fhireg+fhereg) - [volume sources] = 0
with weight +1 on all boundary regions and 0 on internal ones (same convention as particle_balance_widegrid.py).
Only the internal-energy fluxes fhi+fhe are used (no fhp/fhm: those belong to energy_balance_coupled).

Volume sources are the same as in internal_energy_balance_coupled, taken from region 0 (whole domain) and
summed over species where relevant:
  rsahireg rrahireg rqahereg rcxhireg  atomic/molecular ion+electron energy sources (ionisation, recombination, radiation, CX)
  b2divue b2divua b2exbe b2exba b2visa b2joule b2fraa b2wrong1-3   B2.5 terms that are not sources of internal energy
  b2stbr_she/shi_reg   B2.5 (EIRENE) electron/ion energy sources
  rsanareg rranareg b2stbr_sna_reg * potential   potential-energy terms from ionisation/recombination/EIRENE particle sources
residual = outflow - rsa - rra + rqa - rcx - b2divue - b2divua - b2exbe - b2exba - b2visa - b2joule - b2fraa
           - b2wrong1-3 - b2brhe - b2brhi + rsap - rrap + b2brnp
Note: the awk residual in internal_energy_balance_coupled writes b2bnrp (typo, undefined -> 0) in the sum, so b2brnp
is not actually included there; here it is included (use --no-b2brnp to reproduce the awk behaviour).

The boundary regions are read from b2fgmtry (in the run directory or in baserun/) when available, otherwise the
SN defaults are used. The ionisation potentials ('pot' line of the tallies) are read from run.log / run.log.gz /
b2.log in the run directory, otherwise from the --pot option or the hard-coded D+C defaults.
Input: b2mn.exe.dir/b2tallies.nc if readable, else ./b2tallies.nc (run from the run directory).
Output: a table of the time-averaged terms (mean over the last 10% of the outputs, W), the residual and
residual/gross, the outflow through each boundary region, and a window (TkAgg) with three panels:
boundary outflow vs volume sources, the individual source terms, and the instantaneous error
residual/gross(t) with gross = sum of the absolute sizes of all terms. dN/dt of the internal energy is not
included, so the balance is only meaningful for a quasi-steady run.

Usage:
    internal_energy_balance_widegrid.py [b2fgmtry] [--no-b2brnp] [--log=run.log] [--pot=p0,p1,...]
        b2fgmtry      path to the geometry (default: ./b2fgmtry, baserun/b2fgmtry, ../baserun/b2fgmtry)
        --no-b2brnp   leave b2brnp out of the residual (reproduces the awk script)
        --log=FILE    log to read the 'pot' line from (default: run.log, run.log.gz, b2.log, b2.log.gz)
        --pot=LIST    comma-separated ionisation potentials in eV, one per species (overrides the log)
"""
from __future__ import print_function
import netCDF4
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import sys
import gzip
import re
import numpy

if any(a in ('-h', '--help') for a in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

SN_BOUNDARY = [1,4,7,8,9,11,12,13]   # targets, PFR walls, core boundary, main chamber walls
QE = 1.602176634e-19
POT_DEFAULT = [0.0, 13.602, 0.0, 11.26, 35.643, 83.531, 148.02, 540.12, 1030.1]   # D0 D+ C0..C6+ (cumulative)

opts = [a for a in sys.argv[1:] if a.startswith('--')]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
use_b2brnp = '--no-b2brnp' not in opts

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

def read_pot(ns):
    """Ionisation potentials [eV] per species from the 'pot' line printed in the tallies of the log."""
    for o in opts:
        if o.startswith('--pot='): return numpy.array(o[6:].split(','), dtype=float)
    logs = [o[6:] for o in opts if o.startswith('--log=')] + ['run.log', 'run.log.gz', 'b2.log', 'b2.log.gz']
    for fn in logs:
        if not os.access(fn, os.R_OK): continue
        op = gzip.open if fn.endswith('.gz') else open
        with op(fn, 'rb') as fh:
            for line in fh:
                if line.startswith(b'pot '):
                    pot = numpy.array(line.split()[1:], dtype=float)
                    if len(pot) == ns:
                        print('Potentials from %s:' % fn, pot)
                        return pot
    print('pot not found in the logs, using D+C defaults', POT_DEFAULT[:ns])
    return numpy.array((POT_DEFAULT + [0.0]*ns)[:ns])

if os.access('b2mn.exe.dir/b2tallies.nc', os.R_OK):
    f = netCDF4.Dataset('b2mn.exe.dir/b2tallies.nc','r')
else:
    f = netCDF4.Dataset('b2tallies.nc','r')
freg = f.dimensions['freg'].size
ns = f.dimensions['ns'].size
times = numpy.ma.filled(f.variables['times'][:], 0)

def reg0(name):
    """Whole-domain (region 0) value, summed over species if the variable has a species index."""
    v = numpy.ma.filled(f.variables[name][:], 0)
    return v[:,:,0].sum(axis=1) if v.ndim == 3 else v[:,0]

def reg0_pot(name, shift):
    """Region 0 value per species weighted with the potential: pot[k+shift] (0 beyond the last species)."""
    v = numpy.ma.filled(f.variables[name][:], 0)[:,:,0]
    p = numpy.zeros(ns)
    src = pot[shift:ns] * QE
    p[:len(src)] = src
    return (v*p).sum(axis=1)

pot = read_pot(ns)

W = numpy.zeros(freg)
W[boundary_regions(freg)] = 1
fh = numpy.ma.filled(f.variables['fhireg'][:], 0) + numpy.ma.filled(f.variables['fhereg'][:], 0)
bflux = fh*W                      # power through each boundary region, positive = leaving the plasma
outflow = bflux.sum(axis=1)

rsa, rra, rqa, rcx = [reg0(n) for n in ('rsahireg', 'rrahireg', 'rqahereg', 'rcxhireg')]
b2 = {n: reg0(n) for n in ('b2divue', 'b2divua', 'b2exbe', 'b2exba', 'b2visa', 'b2joule', 'b2fraa', 'b2wrong1', 'b2wrong2', 'b2wrong3')}
b2brhe = reg0('b2stbr_she_reg')
b2brhi = reg0('b2stbr_shi_reg')
rsap = reg0_pot('rsanareg', 1)
rrap = reg0_pot('rranareg', 0)
b2brnp = reg0_pot('b2stbr_sna_reg', 1)

src = (rsa + rra - rqa + rcx + sum(b2.values()) + b2brhe + b2brhi - rsap + rrap
       - (b2brnp if use_b2brnp else 0))
resid = outflow - src
gross = (numpy.abs(bflux).sum(axis=1) + numpy.abs(rsa) + numpy.abs(rra) + numpy.abs(rqa) + numpy.abs(rcx)
         + sum(numpy.abs(v) for v in b2.values()) + numpy.abs(b2brhe) + numpy.abs(b2brhi)
         + numpy.abs(rsap) + numpy.abs(rrap) + (numpy.abs(b2brnp) if use_b2brnp else 0))
err_inst = numpy.divide(resid, gross, out=numpy.full_like(gross, numpy.nan), where=gross > 0)

n = max(1, len(times)//10)
print('Mean over the last %d outputs (W):' % n)
terms = [('outflow (fhi+fhe)', outflow), ('rsa', rsa), ('rra', rra), ('rqa', rqa), ('rcx', rcx)] + \
        list(b2.items()) + [('b2brhe', b2brhe), ('b2brhi', b2brhi), ('rsap', rsap), ('rrap', rrap), ('b2brnp', b2brnp)]
for name, v in terms:
    print('  %-18s %12.4e' % (name, v[-n:].mean()))
print('  %-18s %12.4e' % ('residual', resid[-n:].mean()))
print('  %-18s %12.4f' % ('residual/gross', numpy.nanmean(err_inst[-n:])))
print('Boundary region fluxes fhi+fhe (W):', ' '.join('%d:%.3e' % (r, bflux[-n:, r].mean()) for r in range(freg) if W[r]))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9,11))
ax1.plot(times, outflow, label='boundary outflow fhi+fhe')
ax1.plot(times, src, label='volume sources')
ax1.set_ylabel('Power (W)')
ax1.set_title('Wide-grid internal energy balance (outward boundary faces)')
ax1.legend(loc=0)
for name, v in terms[1:]:
    if numpy.abs(v).max() > 0: ax2.plot(times, v, label=name)
ax2.set_ylabel('source terms (W)')
ax2.legend(loc=0, ncol=3, fontsize='small')
ax3.plot(times, err_inst)
ax3.set_ylabel('instantaneous error\n(relative to gross throughput at t)')
ax3.set_xlabel('time [s]')
fig.tight_layout()
plt.show()
