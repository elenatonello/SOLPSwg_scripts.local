#!/usr/bin/env python
# coding: utf-8
"""
plot_omp_profiles.py

Compares radial profiles, at a chosen poloidal cut, across one or more
SOLPS-ITER runs. Cuts available: OMP (outer midplane), the outer divertor
target, and the inner divertor target.

Each run is identified by its top-level directory, which must contain a
2d_profiles/ subdirectory (as produced by the `2d_profiles` script) with
files such as ne3da.last10, te3da.last10, na3da0.last10, etc. The suffix
before ".last10" ('a', 'r' or 'l') denotes the cut; each file has two
whitespace-separated columns: distance from the separatrix [m] and the
(time-averaged) quantity value.

Usage:
    python plot_omp_profiles.py run_dir1 run_dir2 ... [options]

Examples:
    python plot_omp_profiles.py runs/A runs/B --labels "case A" "case B"
    python plot_omp_profiles.py runs/A runs/B -q ne te ti na -o omp.png
    python plot_omp_profiles.py runs/A runs/B -q na --species 0 1 --show
    python plot_omp_profiles.py runs/A --location od -q ne te ti
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

# Categorical palette (fixed order, CVD-validated), one color per run.
PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
LINESTYLES = ["-", "--", "-.", ":"]

# quantity key -> (file base, y-axis label, species-resolved?, log scale by default?)
QUANTITIES = {
    "ne": ("ne3d", "$n_e$ [m$^{-3}$]", False, True),
    "te": ("te3d", "$T_e$ [eV]", False, False),
    "ti": ("ti3d", "$T_i$ [eV]", False, False),
    "po": ("po3d", "potential [V]", False, False),
    "na": ("na3d", "$n_s$ [m$^{-3}$]", True, True),
    "vs": ("vs3d", r"$v_\parallel$ [m/s]", True, False),
    "vx": ("vx3d", "$v_x$ [m/s]", True, False),
    "vy": ("vy3d", "$v_y$ [m/s]", True, False),
    "dn": ("dn3d", "$D_n$ [m$^2$/s]", True, False),
    "dp": ("dp3d", "$D_p$ [m$^2$/s]", True, False),
    "an": ("an3d", "anomalous coeff.", False, False),
    "ke": ("ke3d", r"$\kappa_e$", False, False),
    "ki": ("ki3d", r"$\kappa_i$", False, False),
    "mn": ("mn3d", "Mach number", False, False),
}
DEFAULT_QUANTITIES = ["ne", "te", "ti", "po"]

# location key -> (file suffix, display name for axis labels)
LOCATIONS = {
    "omp": ("a", "OMP"),
    "od": ("r", "outer divertor target"),
    "id": ("l", "inner divertor target"),
}
DEFAULT_LOCATION = "omp"


def load_profile(path):
    ds, val = np.loadtxt(path, unpack=True)
    order = np.argsort(ds)
    return ds[order], val[order]


def find_species_files(profiles_dir, base, suffix):
    prefix = base + suffix
    found = []
    for f in profiles_dir.glob(f"{prefix}*.last10"):
        idx_str = f.name[len(prefix):-len(".last10")]
        if idx_str.isdigit():
            found.append((int(idx_str), f))
    found.sort()
    return found


def plot_quantity(ax, qty, run_dirs, labels, species_filter, location, log_override=None):
    base, ylabel, species_resolved, log_scale = QUANTITIES[qty]
    if log_override is not None:
        log_scale = log_override
    suffix, location_label = LOCATIONS[location]
    plotted = False

    for run_idx, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        color = PALETTE[run_idx % len(PALETTE)]
        profiles_dir = run_dir / "2d_profiles"
        if not profiles_dir.is_dir():
            print(f"warning: {profiles_dir} not found, skipping run '{label}'", file=sys.stderr)
            continue

        if species_resolved:
            species_files = find_species_files(profiles_dir, base, suffix)
            if species_filter is not None:
                species_files = [(i, f) for i, f in species_files if i in species_filter]
            if not species_files:
                print(f"warning: no {base}{suffix}<N>.last10 files found for run '{label}'", file=sys.stderr)
                continue
            for sp_idx, f in species_files:
                ds, val = load_profile(f)
                ls = LINESTYLES[sp_idx % len(LINESTYLES)]
                ax.plot(ds, val, color=color, linestyle=ls, linewidth=2,
                         marker=".", markersize=5,
                         label=f"{label} (species {sp_idx})")
                plotted = True
        else:
            f = profiles_dir / f"{base}{suffix}.last10"
            if not f.is_file():
                print(f"warning: {f} not found, skipping run '{label}'", file=sys.stderr)
                continue
            ds, val = load_profile(f)
            ax.plot(ds, val, color=color, linewidth=2, marker=".", markersize=5, label=label)
            plotted = True

    ax.axvline(0.0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_xlabel(f"distance from separatrix at {location_label} [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(qty)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    if log_scale:
        ax.set_yscale("log")
    if plotted:
        ax.legend(fontsize=8)
    return plotted


def main():
    parser = argparse.ArgumentParser(
        description="Compare radial profiles at the OMP or a divertor target across SOLPS-ITER runs "
                    "(from the 2d_profiles/*.last10 files).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path,
                         help="Run directories, each containing a 2d_profiles subdirectory")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Legend label per run (default: directory name)")
    parser.add_argument("-q", "--quantities", nargs="+", default=DEFAULT_QUANTITIES,
                         choices=sorted(QUANTITIES), metavar="QUANTITY",
                         help=f"Quantities to plot, any of: {sorted(QUANTITIES)}")
    parser.add_argument("--location", default=DEFAULT_LOCATION, choices=sorted(LOCATIONS),
                         help="Poloidal cut to plot: omp (outer midplane), "
                              "od (outer divertor target), id (inner divertor target). "
                              "Not every quantity is written out at every location "
                              "(e.g. ke/ki only at omp) -- missing files are skipped with a warning.")
    parser.add_argument("--species", nargs="+", type=int, default=None,
                         help="Species indices to plot for species-resolved quantities "
                              "(default: all found)")
    parser.add_argument("--log-scale", nargs="+", default=[], choices=sorted(QUANTITIES),
                         metavar="QUANTITY", help="Force these quantities onto a log y-axis")
    parser.add_argument("--linear-scale", nargs="+", default=[], choices=sorted(QUANTITIES),
                         metavar="QUANTITY", help="Force these quantities onto a linear y-axis "
                                                   "(overrides their log-by-default setting)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Also save the figure to this image file (default: don't save, just show)")
    parser.add_argument("--no-show", action="store_true",
                         help="Don't display the plot interactively (useful with --output on a headless node)")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.run_dirs):
        parser.error("--labels must list exactly one label per run directory")
    labels = args.labels or [d.name for d in args.run_dirs]

    clash = set(args.log_scale) & set(args.linear_scale)
    if clash:
        parser.error(f"quantities in both --log-scale and --linear-scale: {sorted(clash)}")
    log_overrides = {q: True for q in args.log_scale}
    log_overrides.update({q: False for q in args.linear_scale})

    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    ncols = 2 if len(args.quantities) > 1 else 1
    nrows = -(-len(args.quantities) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    species_filter = set(args.species) if args.species is not None else None
    for ax, qty in zip(flat_axes, args.quantities):
        plot_quantity(ax, qty, args.run_dirs, labels, species_filter, args.location,
                      log_override=log_overrides.get(qty))
    for ax in flat_axes[len(args.quantities):]:
        ax.axis("off")

    fig.suptitle(f"{LOCATIONS[args.location][1]} profile comparison")
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
