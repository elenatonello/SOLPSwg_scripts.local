#!/usr/bin/env python3
"""Sketches of how a wall face is treated as a function of Qalfmin, Qalfmax and Pbs_reduce_fac.

Figure 1 (qalf_regimes.png): for three example parameter sets, as a function of the face's ORIGINAL contact
fraction w0 = Pbs/S:  (row 1) the regime (grazing / band / normal), (row 2) the stored Pbs/S the geometry prep
leaves in the file, (row 3) the contact fraction the boundary-condition sinks effectively use.
Figure 2 (qalf_bc_regimes.png): what each boundary condition does in the three regimes.

The statements follow the code (b2us_prep.F wall-face block; b2stbc_phys.F branches on w < Qalfmin).
Usage:  sketch_qalf_regimes.py [outdir]     (default: directory of this script)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

RED, ORANGE, BLUE, GREEN, GREY = '#d62728', '#ff9f1c', '#1f77b4', '#2ca02c', '#555555'
SETS = [  # (title, Qalfmin, Qalfmax, Pbs_reduce_fac)
    ('A: Qalfmin = Qalfmax = 1e-3,  f_red = 1e-3', 1e-3, 1e-3, 1e-3),
    ('B: Qalfmin = 1e-3,  Qalfmax = 8e-3,  f_red = 1e-6', 1e-3, 8e-3, 1e-6),
    ('C: Qalfmin = 2e-3,  Qalfmax = 4e-3,  f_red = 1e-6', 2e-3, 4e-3, 1e-6),
]
PITCH = 3e-3          # example Bpol/B for the "clipped" curve


def stored_pbs(w0, qmin, qmax, f):
    """Pbs/S written by the prep for a face with original contact fraction w0."""
    return np.where(w0 < qmin, f * w0, np.where(w0 < qmax, qmax, w0))


def bc_fraction(w0, qmin, qmax, f, prep=True):
    """Contact fraction the boundary-condition sinks use (angle-based w for the branch test).
    prep=True : Pbs / Qalf modified as above. prep=False: only the scalars Qalfmin/Qalfmax edited."""
    if prep:
        w_file = stored_pbs(w0, qmin, qmax, f)       # unclipped: Qalf0*Bx/B follows Pbs/S
    else:
        w_file = w0
    return np.where(w_file < qmin, qmax, w_file)      # grazing branch -> floor Qalfmax


def fig_regimes(outdir):
    w0 = np.logspace(-5, -1, 600)
    fig, axs = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[0.55, 1.5, 1.5]))
    fig.suptitle('Treatment of a wall face as a function of its original contact fraction $w_0=P_{bs}/S$ '
                 'and of Qalfmin, Qalfmax, Pbs_reduce_fac\n'
                 'row 3: sink $\\propto$ (this fraction)$\\times S\\,c_s\\,n\\ldots$.  With the prep the sink is '
                 '$\\max(w_0,\\mathrm{Qalfmax})$ (continuous); with only the scalars edited it jumps at Qalfmin when '
                 'Qalfmax > Qalfmin.', fontsize=11)
    for j, (title, qmin, qmax, f) in enumerate(SETS):
        # ---- row 1: regime strip
        ax = axs[0, j]
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1e-5, 1e-1)
        ax.set_ylim(0, 1)
        ax.axvspan(1e-5, qmin, color=RED, alpha=0.35)
        if qmax > qmin:
            ax.axvspan(qmin, qmax, color=ORANGE, alpha=0.45)
        ax.axvspan(qmax, 1e-1, color=BLUE, alpha=0.25)
        for q, lab in ((qmin, 'Qalfmin'), (qmax, 'Qalfmax')):
            if q == qmin == qmax and lab == 'Qalfmax':
                continue
            ax.axvline(q, color='k', lw=1)
        ax.text(np.sqrt(1e-5 * qmin), 0.5, 'GRAZING\n$w_0<$Qalfmin', ha='center', va='center', fontsize=8.5)
        if qmax > qmin:
            ax.text(np.sqrt(qmin * qmax), 0.5, 'BAND', ha='center', va='center', fontsize=8.5)
        ax.text(np.sqrt(qmax * 1e-1), 0.5, 'NORMAL\n$w_0\\geq$Qalfmax', ha='center', va='center', fontsize=8.5)
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.text(qmin, 0.06, ' Qalfmin' if qmax > qmin else ' Qalfmin = Qalfmax', rotation=90, ha='right', va='bottom',
                fontsize=7.5)
        if qmax > qmin:
            ax.text(qmax, 0.06, ' Qalfmax', rotation=90, ha='right', va='bottom', fontsize=7.5)
        # ---- row 2: stored Pbs/S
        ax = axs[1, j]
        y = stored_pbs(w0, qmin, qmax, f)
        ax.loglog(w0, w0, '--', color=GREY, lw=1, label='unchanged ($=w_0$)')
        ax.loglog(w0, y, color='k', lw=2.2, label='stored $P_{bs}/S$ after the prep')
        m = w0 <= PITCH
        ax.loglog(w0[m], np.minimum(y, PITCH)[m], ':', color=RED, lw=2.4,
                  label=f'angle-based $w=|Q_\\alpha^{{(0)}}B_x/B|$ for a face with $B_x/B={PITCH:g}$ ($w_0\\leq B_x/B$)')
        ax.axvline(qmin, color='k', lw=0.8)
        ax.axvline(qmax, color='k', lw=0.8)
        ax.axvspan(1e-5, qmin, color=RED, alpha=0.12)
        if qmax > qmin:
            ax.axvspan(qmin, qmax, color=ORANGE, alpha=0.18)
        ax.set_xlim(1e-5, 1e-1)
        ax.set_ylim(1e-12, 3e-1)
        ax.set_ylabel('stored  $P_{bs}/S$')
        ax.grid(alpha=0.25, which='major')
        ax.text(1.3e-5, 2e-11, f'grazing: $P_{{bs}}\\to f_{{red}}P_{{bs}}$\n($\\times${f:g})', fontsize=8, color=RED)
        if qmax > qmin:
            ax.text(np.sqrt(qmin * qmax), qmax * 2.5, 'raised\nto Qalfmax', ha='center', va='bottom',
                    fontsize=8, color='#b36b00')
        if j == 0:
            ax.legend(fontsize=7.2, loc='upper left', framealpha=0.9)
        # ---- row 3: BC sink fraction
        ax = axs[2, j]
        ax.loglog(w0, w0, '--', color=GREY, lw=1, label='$w_0$ (true contact fraction)')
        ax.loglog(w0, bc_fraction(w0, qmin, qmax, f, prep=True), color='k', lw=2.2,
                  label='geometry prep applied (Pbs, Qalf modified)')
        ax.loglog(w0, bc_fraction(w0, qmin, qmax, f, prep=False), color=GREEN, lw=1.8, ls='-.',
                  label='only the scalars edited (Pbs, Qalf untouched)')
        ax.axvline(qmin, color='k', lw=0.8)
        ax.axvline(qmax, color='k', lw=0.8)
        ax.axvspan(1e-5, qmin, color=RED, alpha=0.12)
        if qmax > qmin:
            ax.axvspan(qmin, qmax, color=ORANGE, alpha=0.18)
        ax.set_xlim(1e-5, 1e-1)
        ax.set_ylim(1e-5, 2e-1)
        ax.set_xlabel('original contact fraction  $w_0=P_{bs}/S$')
        ax.set_ylabel('effective fraction in the BC sinks')
        ax.grid(alpha=0.25, which='major')
        ax.text(1.3e-5, 3.5e-2, 'grazing: floor = Qalfmax', fontsize=8, color=RED)
        if j == 0:
            ax.legend(fontsize=7.2, loc='lower right', framealpha=0.9)
    out = os.path.join(outdir, 'qalf_regimes.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)


def face_panel(ax, title, color, opening, sink, lines, pinned=False):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.45, 1.12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10.5, fontweight='bold', color=color)
    ax.add_patch(Rectangle((-1.5, 0.0), 3.0, 1.0, fc='#eaf2fb', ec='none'))
    ax.add_patch(Rectangle((-1.5, -0.55), 3.0, 0.55, fc='#f3e9d2', ec='k', lw=0.8))
    ax.add_patch(Rectangle((-1.5, -0.62), 3.0, 0.07, fc='#cccccc', ec='none', hatch='////'))
    ax.plot([-1.5, 1.5], [0, 0], 'k', lw=3)
    ax.text(-1.45, 0.88, 'interior cell', fontsize=8, color=GREY, va='top')
    ax.text(-1.45, -0.08, 'guard cell (boundary condition acts here)', fontsize=7.5, color=GREY, va='top')
    # opening
    ax.plot([-opening / 2, opening / 2], [0.04, 0.04], color=RED, lw=5, solid_capstyle='butt')
    # plasma -> wall arrows (sink)
    for dx in np.linspace(-opening / 2, opening / 2, 3 if opening > 0.5 else 1):
        ax.add_patch(FancyArrowPatch((dx, 0.7), (dx, -0.35), arrowstyle='-|>', mutation_scale=11, color=GREEN,
                                     lw=1.2 + 3.5 * sink, alpha=0.9))
    if pinned:
        ax.text(0.95, -0.28, '$u,\\ \\phi$ := interior', fontsize=8, color='#7a3db8', ha='center',
                bbox=dict(fc='white', ec='#7a3db8', pad=2))
    for k, (txt, col) in enumerate(lines):
        ax.text(-1.55, -0.78 - 0.14 * k, txt, fontsize=7.6, color=col, va='top', ha='left')


def fig_bc(outdir):
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.6), constrained_layout=True)
    fig.suptitle('What the wall boundary conditions do in each regime (test: stored $w=|Q_\\alpha^{(0)}B_x/B|$ versus Qalfmin)',
                 fontsize=11.5)
    face_panel(axs[0], 'NORMAL  ($w\\geq$ Qalfmin)', BLUE, 1.2, 1.0,
               [('particles : $-S\\,m_{par}\\,c_s\\,w$', 'k'),
                ('momentum : $u=\\pm m_{par}c_s\\,S w/|P_{bs}|$ (Mach cond.)', 'k'),
                ('electron E: $s_1=|P_{bs}|$,  sheath $\\exp(-e\\Delta\\phi/T_e)$', 'k'),
                ('ion E     : $-{\\rm enipar1}\\,\\sum m_{par}n S c_s w$', 'k'),
                ('potential : sheath current continuity with $S\\,w$', 'k'),
                ('bulk face : fluxes $\\propto P_{bs}$, $Q_\\alpha$', GREY)])
    face_panel(axs[1], 'BAND  (after the prep: raised to Qalfmax)', ORANGE, 1.2, 0.8,
               [('same formulas as NORMAL, with $w=Q_{\\alpha,max}$ (not $w_0$)', 'k'),
                ('=> sink larger by Qalfmax/$w_0$, momentum $u=\\pm m_{par}c_s$', 'k'),
                ('clipped if $B_x/B<Q_{\\alpha,max}$:', RED),
                ('  $w=B_x/B$ (particles, ion E, potential)', RED),
                ('  $P_{bs}/S=Q_{\\alpha,max}$ (electron E)', RED),
                ('  $u=\\pm m_{par}c_s\\,(B_x/B)/Q_{\\alpha,max}$: subsonic', RED)])
    face_panel(axs[2], 'GRAZING  ($w<$ Qalfmin)', RED, 0.12, 0.35,
               [('particles : $-S\\,m_{par}\\,c_s\\,Q_{\\alpha,max}$ (floor)', 'k'),
                ('momentum : $u=u_{interior}$ (weight $V_{BIG}$)', 'k'),
                ('electron E: $s_1=S\\,Q_{\\alpha,max}$', 'k'),
                ('ion E     : $-1.5\\,S\\,Q_{\\alpha,max}\\,c_s\\,n_i$', 'k'),
                ('potential : $\\phi=\\phi_{interior}$ (no sheath current)', 'k'),
                ('bulk face : $P_{bs}$, $Q_\\alpha^{(0)}\\times f_{red}$ if prep', GREY)], pinned=True)
    out = os.path.join(outdir, 'qalf_bc_regimes.png')
    fig.savefig(out, dpi=140)
    print('wrote', out)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    fig_regimes(outdir)
    fig_bc(outdir)


if __name__ == '__main__':
    main()
