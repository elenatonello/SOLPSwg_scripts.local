#!/usr/bin/env python
# coding: utf-8
"""
watch_omp_profiles.py

Live monitor for radial profiles at a chosen poloidal cut (OMP, IMP, outer
or inner divertor target), reading directly from a run's b2time.nc while
the run is still going (unlike 2d_profiles/.last10 files, which are only
written once the run finishes, b2time.nc is appended to continuously, so
this is what lets you watch profiles evolve live).

Each run is identified by its top-level directory, which must contain
b2time.nc (or b2mn.exe.dir/b2time.nc). The plot re-reads the file and
redraws on a fixed interval until the window is closed or you hit Ctrl+C.

Works with b2time.nc from both SOLPS-ITER 3.2.1 and 3.0.9. Differences handled:
  * 3.2.1 stores the radial coordinate as ds{a,i,l,r} variables inside b2time.nc;
    3.0.9 only writes it as text files ds{a,i,l,r} in the run directory, which are
    read instead.
  * 3.2.1 stores na/dn/vs/vx/vy/dp with a species axis (time, ns, ny); 3.0.9 stores
    dn/dp/vs/vx/vy without one (time, ny) and has no na3d* at all. Species-less
    variables are drawn as a single line (and ignore --species); quantities missing
    from the file are reported once on stderr and skipped.

Usage:
    python watch_omp_profiles.py run_dir1 [run_dir2 ...] [options]

Examples:
    python watch_omp_profiles.py runs/A
    python watch_omp_profiles.py runs/A runs/B --labels "case A" "case B"
    python watch_omp_profiles.py runs/A -q ne te ti na --species 0 1 --interval 5
    python watch_omp_profiles.py runs/A --location od -q ne te ti
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import netCDF4 as nc
import numpy as np

PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
LINESTYLES = ["-", "--", "-.", ":"]

# quantity key -> (netcdf variable base, y-axis label, species-resolved?, log scale?, convert J->eV?)
# b2time.nc stores raw solver units (temperatures in Joules); the eV conversion normally
# happens in the `2d_profiles` shell script's postprocessing, which this script bypasses
# by reading b2time.nc directly, so it has to do that conversion itself.
QUANTITIES = {
    "ne": ("ne3d", "$n_e$ [m$^{-3}$]", False, False, False),
    "te": ("te3d", "$T_e$ [eV]", False, False, True),
    "ti": ("ti3d", "$T_i$ [eV]", False, False, True),
    "po": ("po3d", "potential [V]", False, False, False),
    "na": ("na3d", "$n_s$ [m$^{-3}$]", True, False, False),
    "vs": ("vs3d", r"$v_\parallel$ [m/s]", True, False, False),
    "vx": ("vx3d", "$v_x$ [m/s]", True, False, False),
    "vy": ("vy3d", "$v_y$ [m/s]", True, False, False),
    "dn": ("dn3d", "$D_n$ [m$^2$/s]", True, False, False),
    "dp": ("dp3d", "$D_p$ [m$^2$/s]", True, False, False),
    "an": ("an3d", "anomalous coeff.", False, False, False),
    "ke": ("ke3d", r"$\kappa_e$", False, False, False),
    "ki": ("ki3d", r"$\kappa_i$", False, False, False),
    "mn": ("mn3d", "Mach number", False, False, False),
}
DEFAULT_QUANTITIES = ["ne", "te", "ti", "po"]

# location key -> (netcdf variable/ds suffix, display name for axis labels)
LOCATIONS = {
    "omp": ("a", "OMP"),
    "imp": ("i", "IMP"),
    "od": ("r", "outer divertor target"),
    "id": ("l", "inner divertor target"),
}
DEFAULT_LOCATION = "omp"

TIME_VAR_CANDIDATES = ["timesa", "times"]
EV = 1.602176634e-19

# location suffix -> name of the spatial dimension in b2time.nc
SPATIAL_DIMS = {"a": "nya", "i": "nyi", "l": "nybl", "r": "nybr"}

_WARNED = set()


def warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"warning: {msg}", file=sys.stderr)


def find_b2time(run_dir):
    for candidate in (run_dir / "b2mn.exe.dir" / "b2time.nc", run_dir / "b2time.nc"):
        if candidate.is_file():
            return candidate
    return None


def read_ds(ds_nc, b2time_path, run_dir, suffix):
    """Radial coordinate for the chosen cut: the ds{suffix} variable in b2time.nc
    (3.2.1) or, failing that, the ds{suffix} text file next to it / in the run
    directory (3.0.9). Returns None if neither is usable."""
    var = f"ds{suffix}"
    if var in ds_nc.variables:
        return np.asarray(ds_nc.variables[var][:])
    n_expected = None
    dim = SPATIAL_DIMS.get(suffix)
    if dim in ds_nc.dimensions:
        n_expected = len(ds_nc.dimensions[dim])
    for folder in (b2time_path.parent, run_dir, run_dir / "b2mn.exe.dir"):
        path = folder / var
        if not path.is_file():
            continue
        try:
            ds = np.atleast_1d(np.loadtxt(path))
        except ValueError:
            continue
        if n_expected is not None and ds.shape[0] != n_expected:
            warn_once(f"{path} has {ds.shape[0]} points but b2time.nc has {n_expected} "
                      f"along {dim}; ignoring it")
            continue
        return ds
    warn_once(f"no radial coordinate '{var}' in {b2time_path} or as a file under {run_dir}")
    return None


def series_style(sp_idx, run_label):
    """Linestyle and legend text for one series; sp_idx None = no species axis."""
    if sp_idx is None:
        return "-", run_label
    return LINESTYLES[sp_idx % len(LINESTYLES)], f"{run_label} (species {sp_idx})"


def window_mean(values, window):
    n = values.shape[0]
    k = min(window, n)
    return values[n - k:, ...].mean(axis=0)


def last_and_prev_avg(values, window):
    """Split a time-history array into its last step and the mean of the
    `window` steps immediately before it (excludes the last step itself)."""
    n = values.shape[0]
    if n < 2:
        return values[-1, ...], values[-1, ...]
    k = min(window, n - 1)
    return values[-1, ...], values[n - 1 - k:n - 1, ...].mean(axis=0)


def relative_delta_pct(last, prev_avg):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(prev_avg != 0, (last - prev_avg) / prev_avg * 100.0, np.nan)


def read_quantity(ds_nc, base, suffix, window, convert_ev):
    var = f"{base}{suffix}"
    if var not in ds_nc.variables:
        return None
    val = window_mean(np.asarray(ds_nc.variables[var][:]), window)
    if convert_ev:
        val = val / EV
    return val


def read_species_quantity(ds_nc, base, suffix, window, species_filter, n_spatial):
    prefix = f"{base}{suffix}"

    # Convention A: one variable per species, e.g. na3da0, na3da1, ...
    found = []
    for name in ds_nc.variables:
        if name.startswith(prefix):
            idx_str = name[len(prefix):]
            if idx_str.isdigit():
                found.append((int(idx_str), name))
    found.sort()
    if found:
        if species_filter is not None:
            found = [(i, name) for i, name in found if i in species_filter]
        return [(i, window_mean(np.asarray(ds_nc.variables[name][:]), window)) for i, name in found]

    # Convention B: one variable with an extra species axis, e.g. na3da(time, ns, nya)
    # (axis order not assumed -- matched by size against the spatial (dsa) dimension).
    if prefix not in ds_nc.variables:
        return []
    raw = np.asarray(ds_nc.variables[prefix][:])

    # Convention C: no species axis at all, e.g. 3.0.9 dn3da(time, nya)
    if raw.ndim == 2:
        return [(None, window_mean(raw, window))] if raw.shape[1] == n_spatial else []
    if raw.ndim != 3:
        return []
    avg = window_mean(raw, window)
    if avg.shape[0] == n_spatial:
        spatial_axis = 0
    elif avg.shape[1] == n_spatial:
        spatial_axis = 1
    else:
        return []
    nsp = avg.shape[1 - spatial_axis]
    result = []
    for sp in range(nsp):
        if species_filter is not None and sp not in species_filter:
            continue
        val = avg[:, sp] if spatial_axis == 0 else avg[sp, :]
        result.append((sp, val))
    return result


def read_quantity_delta(ds_nc, base, suffix, window, convert_ev):
    var = f"{base}{suffix}"
    if var not in ds_nc.variables:
        return None
    last, prev_avg = last_and_prev_avg(np.asarray(ds_nc.variables[var][:]), window)
    if convert_ev:
        last, prev_avg = last / EV, prev_avg / EV
    return relative_delta_pct(last, prev_avg)


def read_species_quantity_delta(ds_nc, base, suffix, window, species_filter, n_spatial):
    prefix = f"{base}{suffix}"

    found = []
    for name in ds_nc.variables:
        if name.startswith(prefix):
            idx_str = name[len(prefix):]
            if idx_str.isdigit():
                found.append((int(idx_str), name))
    found.sort()
    if found:
        if species_filter is not None:
            found = [(i, name) for i, name in found if i in species_filter]
        result = []
        for i, name in found:
            last, prev_avg = last_and_prev_avg(np.asarray(ds_nc.variables[name][:]), window)
            result.append((i, relative_delta_pct(last, prev_avg)))
        return result

    if prefix not in ds_nc.variables:
        return []
    raw = np.asarray(ds_nc.variables[prefix][:])
    if raw.ndim == 2:
        if raw.shape[1] != n_spatial:
            return []
        last, prev_avg = last_and_prev_avg(raw, window)
        return [(None, relative_delta_pct(last, prev_avg))]
    if raw.ndim != 3:
        return []
    last, prev_avg = last_and_prev_avg(raw, window)
    if last.shape[0] == n_spatial:
        spatial_axis = 0
    elif last.shape[1] == n_spatial:
        spatial_axis = 1
    else:
        return []
    nsp = last.shape[1 - spatial_axis]
    result = []
    for sp in range(nsp):
        if species_filter is not None and sp not in species_filter:
            continue
        if spatial_axis == 0:
            l, p = last[:, sp], prev_avg[:, sp]
        else:
            l, p = last[sp, :], prev_avg[sp, :]
        result.append((sp, relative_delta_pct(l, p)))
    return result


def draw(fig, axes, quantities, run_dirs, labels, species_filter, window, location, delta_axes=None):
    any_ok = False
    status_bits = []
    suffix, location_label = LOCATIONS[location]

    if delta_axes is None:
        delta_axes = [None] * len(axes)

    for ax, delta_ax, qty in zip(axes, delta_axes, quantities):
        ax.clear()
        if delta_ax is not None:
            delta_ax.clear()
        base, ylabel, species_resolved, log_scale, convert_ev = QUANTITIES[qty]
        plotted = False
        delta_plotted = False

        for run_idx, (run_dir, label) in enumerate(zip(run_dirs, labels)):
            color = PALETTE[run_idx % len(PALETTE)]
            b2time_path = find_b2time(run_dir)
            if b2time_path is None:
                if ax is axes[0]:
                    print(f"warning: no b2time.nc found under {run_dir}", file=sys.stderr)
                continue
            try:
                with nc.Dataset(b2time_path) as ds_nc:
                    dsa = read_ds(ds_nc, b2time_path, run_dir, suffix)
                    if dsa is None:
                        continue
                    order = np.argsort(dsa)
                    nt = None
                    for tvar in TIME_VAR_CANDIDATES:
                        if tvar in ds_nc.variables:
                            nt = ds_nc.variables[tvar].shape[0]
                            break
                    if ax is axes[0]:
                        status_bits.append(f"{label}: {nt if nt is not None else '?'} steps")

                    if species_resolved:
                        series = read_species_quantity(
                            ds_nc, base, suffix, window, species_filter, len(dsa))
                        if not series and not any(n.startswith(f"{base}{suffix}")
                                                  for n in ds_nc.variables):
                            warn_once(f"{base}{suffix} not in {b2time_path}; skipping '{qty}' for {label}")
                        for sp_idx, val in series:
                            ls, lab = series_style(sp_idx, label)
                            ax.plot(dsa[order], val[order], color=color, linestyle=ls,
                                    linewidth=2, label=lab)
                            plotted = True
                        if delta_ax is not None:
                            for sp_idx, dval in read_species_quantity_delta(
                                    ds_nc, base, suffix, window, species_filter, len(dsa)):
                                ls, lab = series_style(sp_idx, label)
                                delta_ax.plot(dsa[order], dval[order], color=color, linestyle=ls,
                                              linewidth=2, label=lab)
                                delta_plotted = True
                    else:
                        val = read_quantity(ds_nc, base, suffix, window, convert_ev)
                        if val is None:
                            warn_once(f"{base}{suffix} not in {b2time_path}; skipping '{qty}' for {label}")
                        else:
                            ax.plot(dsa[order], val[order], color=color, linewidth=2, label=label)
                            plotted = True
                        if delta_ax is not None:
                            dval = read_quantity_delta(ds_nc, base, suffix, window, convert_ev)
                            if dval is not None:
                                delta_ax.plot(dsa[order], dval[order], color=color,
                                              linewidth=2, label=label)
                                delta_plotted = True
            except OSError as exc:
                print(f"warning: could not read {b2time_path} ({exc}), will retry", file=sys.stderr)
                continue

        ax.axvline(0.0, color="#c3c2b7", linewidth=1, zorder=0)
        ax.set_xlabel(f"distance from separatrix at {location_label} [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(qty)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        if log_scale:
            ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        if plotted:
            any_ok = True

        if delta_ax is not None:
            delta_ax.axhline(0.0, color="#c3c2b7", linewidth=1, zorder=0)
            delta_ax.axvline(0.0, color="#e1e0d9", linewidth=1, zorder=0)
            delta_ax.set_xlabel(f"distance from separatrix at {location_label} [m]")
            delta_ax.set_ylabel("last step vs prev-N avg [%]")
            delta_ax.set_title(f"{qty} — Δ last step")
            delta_ax.grid(True, color="#e1e0d9", linewidth=0.8)
            if delta_ax.get_legend_handles_labels()[0]:
                delta_ax.legend(fontsize=8)

    stamp = time.strftime("%H:%M:%S")
    status = "  |  ".join(status_bits)
    fig.suptitle(f"{location_label} profiles (live, last {window} steps averaged) — {stamp}\n{status}")
    fig.canvas.draw_idle()
    return any_ok


def main():
    parser = argparse.ArgumentParser(
        description="Live-watch radial profiles (OMP, IMP or a divertor target) from a running SOLPS-ITER "
                    "case; reads b2time.nc directly and redraws every --interval seconds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path,
                         help="Run directories, each containing (or whose b2mn.exe.dir contains) b2time.nc")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Legend label per run (default: directory name)")
    parser.add_argument("-q", "--quantities", nargs="+", default=DEFAULT_QUANTITIES,
                         choices=sorted(QUANTITIES), metavar="QUANTITY",
                         help=f"Quantities to plot, any of: {sorted(QUANTITIES)}")
    parser.add_argument("--location", default=DEFAULT_LOCATION, choices=sorted(LOCATIONS),
                         help="Poloidal cut to watch: omp, imp (midplanes), "
                              "od (outer divertor target), id (inner divertor target). "
                              "Not every quantity is written at every location "
                              "(e.g. ke/ki only at omp/imp) -- missing ones are just skipped.")
    parser.add_argument("--species", nargs="+", type=int, default=None,
                         help="Species indices to plot for species-resolved quantities "
                              "(default: all found); variables without a species axis "
                              "(3.0.9) are always plotted")
    parser.add_argument("--window", type=int, default=10,
                         help="Number of most recent timesteps to average over")
    parser.add_argument("--interval", type=float, default=10.0,
                         help="Seconds between refreshes")
    parser.add_argument("--cycles", type=int, default=None,
                         help="Stop after this many refreshes (default: run until window closed / Ctrl+C)")
    parser.add_argument("--delta", action="store_true",
                         help="Also show, per quantity, the %% change of the last timestep "
                              "relative to the average of the previous --window steps")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.run_dirs):
        parser.error("--labels must list exactly one label per run directory")
    labels = args.labels or [d.name for d in args.run_dirs]

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.ion()

    nq = len(args.quantities)
    if args.delta:
        fig, axes = plt.subplots(nq, 2, figsize=(13.0, 4.5 * nq), squeeze=False)
        plot_axes = [axes[i, 0] for i in range(nq)]
        delta_axes = [axes[i, 1] for i in range(nq)]
    else:
        ncols = 2 if nq > 1 else 1
        nrows = -(-nq // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)
        flat_axes = axes.flatten()
        for ax in flat_axes[nq:]:
            ax.axis("off")
        plot_axes = flat_axes[:nq]
        delta_axes = None

    species_filter = set(args.species) if args.species is not None else None

    fig.show()
    cycle = 0
    try:
        while plt.fignum_exists(fig.number):
            draw(fig, plot_axes, args.quantities, args.run_dirs, labels, species_filter, args.window,
                 args.location, delta_axes=delta_axes)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            cycle += 1
            if args.cycles is not None and cycle >= args.cycles:
                break
            plt.pause(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
