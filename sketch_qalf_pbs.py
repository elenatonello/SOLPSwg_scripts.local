#!/usr/bin/env python3
"""Schematic (not to scale) of what Qalf / Pbs mean on a wall face and what the prep step does to them.

Poloidal-plane view of one wall face: the wall (hatched), the face normal n, the poloidal projection of B
(blue) at angle alpha to the normal, and the "opening" Pbs/S = cos(alpha) * Bpol/B through which plasma can
leave along B (red bracket). Top row: three kinds of faces before the prep. Bottom row: what the prep does.

Usage:  sketch_qalf_pbs.py [out.png]       (default: qalf_sketch.png next to this script)
Needs no input files: the numbers in the panels are illustrative examples, not read from a run.
Background: qalf_pbs_notes.tex / .pdf in this directory.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

BLUE, RED, GREY, GREEN = '#1f77b4', '#d62728', '#555555', '#2ca02c'


def panel(ax, title, alpha_deg, opening, lines, sink=1.0, sink_label=None, tor=False, dashed=False,
          alpha_note=None):
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.42, 1.62)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10.5, fontweight='bold')
    # plasma and wall
    ax.add_patch(plt.Rectangle((-1.4, 0), 2.8, 1.62, fc='#eaf2fb', ec='none', zorder=0))
    ax.add_patch(plt.Rectangle((-1.4, -0.22), 2.8, 0.22, fc='#dddddd', ec='none', hatch='////', zorder=1))
    ax.plot([-1.4, 1.4], [0, 0], color='k', lw=3, zorder=3)
    ax.text(1.38, -0.17, 'wall (face area S)', ha='right', va='center', fontsize=8, color=GREY)
    # face normal
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1.0), arrowstyle='-|>', mutation_scale=12, color=GREY, lw=1.5,
                                 zorder=4))
    ax.text(0.04, 1.0, 'n', fontsize=10, color=GREY, va='bottom')
    # poloidal B direction
    a = np.radians(alpha_deg)
    tip = (np.sin(a) * 1.0, np.cos(a) * 1.0)
    ax.add_patch(FancyArrowPatch((0, 0), tip, arrowstyle='-|>', mutation_scale=14, color=BLUE, lw=2.2, zorder=5))
    ax.text(tip[0] + 0.05, tip[1] + 0.03, r'$B_{pol}$', color=BLUE, fontsize=10, ha='left', va='bottom')
    ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=90 - alpha_deg, theta2=90, color=BLUE, lw=1.2, zorder=5))
    lx, ly = (-0.2, 0.5) if alpha_deg < 15 else (0.55 * np.sin(a / 2), 0.55 * np.cos(a / 2))
    ax.text(lx, ly, alpha_note or fr'$\alpha={alpha_deg:.0f}^\circ$', color=BLUE, fontsize=9, ha='center',
            va='center', bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.5))
    if tor:
        ax.plot([-0.9], [0.62], 'o', mfc='none', mec=BLUE, ms=13, mew=1.5, zorder=5)
        ax.plot([-0.9], [0.62], '.', color=BLUE, ms=5, zorder=5)
        ax.text(-0.9, 0.35, r'$B_{tor}\gg B_{pol}$', color=BLUE, fontsize=8, ha='center')
    # opening on the wall
    x0, x1 = -opening / 2 + 0.0, opening / 2
    ax.plot([x0, x1], [-0.06, -0.06], color=RED, lw=5 if opening > 0.05 else 3, solid_capstyle='butt', zorder=6,
            linestyle='--' if dashed else '-')
    ax.text(0, -0.31, r'opening $P_{bs}/S$', color=RED, fontsize=8.5, ha='center', va='center')
    # sink arrows through the opening (thickness ~ opening)
    if sink > 0:
        for dx in np.linspace(x0 + 0.02, x1 - 0.02, 3 if opening > 0.2 else 1):
            ax.add_patch(FancyArrowPatch((dx, 0.35), (dx, 0.03), arrowstyle='-|>', mutation_scale=10,
                                         color=GREEN, lw=1.5 + 3 * sink, alpha=0.85, zorder=4,
                                         linestyle='--' if dashed else '-'))
    if sink_label:
        ax.text(-1.35, 1.45, sink_label, fontsize=8, color=GREEN, va='top', ha='left')
    for k, (txt, col) in enumerate(lines):
        ax.text(-1.4, 1.57 - 0.125 * k, txt, fontsize=7.4, color=col, ha='left', va='top')


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             'qalf_sketch.png')
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.2), constrained_layout=True)
    fig.suptitle(r'$P_{bs}/S = |Q_\alpha^{(0)}|\cdot B_x/B = \cos\alpha \cdot B_{pol}/B$:   '
                 'the opening through which plasma can leave along B   (schematic, not to scale)',
                 fontsize=11.5)
    # ---- top row: before the prep
    panel(axs[0, 0], '(a) normal wall face', 60, 0.75,
          [(r'$\cos\alpha=0.5$,  $B_{pol}/B\approx0.07$', 'k'), (r'$P_{bs}/S\approx3\times10^{-2}$ (typical, example)', RED),
           ('sheath sink ∝ wrk: full strength', GREEN)], sink=1.0)
    panel(axs[0, 1], '(b) tangent wall (B_pol runs along the wall)', 89, 0.13,
          [(r'$\cos\alpha=0.015$,  $B_{pol}/B\approx0.055$', 'k'), (r'$P_{bs}/S\approx8\times10^{-4}$ (example)', RED),
           ('large alpha, normal pitch', GREEN)], sink=0.12)
    panel(axs[0, 2], '(c) low pitch (B almost purely toroidal)', 6, 0.13,
          [(r'$\cos\alpha\approx0.99$, $B_{pol}/B\approx1.2\times10^{-3}$', 'k'),
           (r'$P_{bs}/S\approx1.2\times10^{-3}$', RED), ('small alpha, tiny pitch', GREEN)],
          sink=0.12, tor=True)
    # ---- bottom row: after the prep (example: Pbs_reduce=1e-6; Qalfmin < w < Qalfmax raised)
    panel(axs[1, 0], '(d) normal face, wrk ≥ Qalfmax: unchanged', 60, 0.75,
          [('nothing modified', 'k'), (r'$P_{bs}/S$ and $Q_\alpha$ as before', RED)], sink=1.0)
    panel(axs[1, 1], r'(e) grazing, wrk < Qalfmin: REDUCED', 89.9999, 0.02,
          [(r'$P_{bs}\leftarrow10^{-6}P_{bs}$, $Q_\alpha^{(0)}\to10^{-6}$ ($\alpha\to90^\circ$)', 'k'),
           (r'$P_{bs}/S\sim10^{-9}$: parallel outlet closed', RED),
           (r'BC floor sink: $1.5\,S\,Q_{\alpha,max}\,c_s\,n\,T$', GREEN),
           (r'floor $\propto Q_{\alpha,max}$;  $\phi$, $u$ pinned to interior', GREEN)],
          sink=0.05, dashed=True, alpha_note=r'$\alpha\to90^\circ$')
    panel(axs[1, 2], r'(f) band, Qalfmin ≤ wrk < Qalfmax: RAISED', 25, 0.4,
          [(r'$P_{bs}\leftarrow Q_{\alpha,max}S$, $Q_\alpha^{(0)}$ recomputed', 'k'),
           ('sink grows by Qalfmax / wrk', GREEN),
           (r'clipped if $B_{pol}/B<Q_{\alpha,max}$:', 'k'),
           (r'$P_{bs}/S=Q_{\alpha,max}>\cos\alpha\,B_{pol}/B$', RED)], sink=0.5, alpha_note=r'$\alpha\downarrow$')
    for ax in axs[0]:
        ax.annotate('', xy=(0.5, -0.02), xytext=(0.5, -0.02), xycoords='axes fraction')
    fig.text(0.5, 0.505, '▼  geometry prep (modify_b2fgmtry_prep.py, or prep_geometry in b2us)  ▼',
             ha='center', fontsize=11, fontweight='bold', color=GREY)
    fig.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
