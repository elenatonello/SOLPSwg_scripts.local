#!/usr/bin/env python3
"""Summarise and plot the B2.5 residual norms stored in one or more b2ftrace files.

Usage:
    b2ftrace_residuals.py RUN_DIR [RUN_DIR ...] [--plot] [--last N]    # table (+ overlay plot of the runs)
    b2ftrace_residuals.py RUN_DIR [RUN_DIR ...] --list                 # field titles of the first run
    b2ftrace_residuals.py RUN_DIR [RUN_DIR ...] --by-eq [--save out.png]
    b2ftrace_residuals.py RUN_DIR [RUN_DIR ...] --log [--last N]

A table is always printed (also with --plot / --by-eq): per run, one row per residual group with the last value
(mean of the last N steps), the minimum so far and the trend. The trend compares the mean of the last N steps
with the N before (falling = still converging, ~0 = stalled/converged, rising = problem); N is --last
(default 500) and is also the averaging window of --log.

--plot draws one window with the runs overlaid (max over species, 6 panels: continuity, parallel
momentum, electron heat, potential, pressure correction, Te correction; x axis = step in b2ftrace).

--by-eq draws one figure per run with one panel per equation (continuity,
parallel momentum, total momentum, electron heat, atom heat, potential and the
corrections), with a line per species (D0, D1+, C0, C1+, ...).

--log reads run.log instead (works on a RUNNING job): steps done, CPU per step,
share of the step spent in Eirene, estimated remaining time, and whether
eirene_underrelax is active. Residual mode also works on a running job: it
reads the live RUN_DIR/b2mn.exe.dir/b2ftrace when that is the newest copy.

RUN_DIR is a run directory (containing b2ftrace) or a b2ftrace file itself.
Read-only: figures are shown in a window (TkAgg, needs a display; without DISPLAY the Agg backend is used
and nothing can be shown, so give --save); files are written only with --save. With --save out.png the files
are out_overlay.png (--plot) and out_eq_<run>.png (--by-eq, one per run).

b2ftrace layout (formatted "cf" file):
    header : ns, label, nfld, then nfld x (linlog, title, label)
    body   : one block per B2.5 step: "*cf: real <nfld> data" + nfld numbers
Field titles (e.g. "norm of residual of continuity equation, species 3") are read
from the header, so the script follows whatever b2mwqt_style / ns the run used.

Residual groups plotted / tabulated (max over species where there are several):
    continuity, parallel momentum, total momentum, electron heat, atom heat,
    fluid-neutral heat, potential, plus the pressure/velocity/Te/Ti corrections.
"""
import argparse
import os
import re
import sys

import numpy as np


def read_cf_blocks(path):
    """Yield (type, count, name, [lines]) for every '*cf:' block in the file."""
    head = None
    body = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("*cf:"):
                if head is not None:
                    yield head + (body,)
                parts = line.split()
                # '*cf:  <type>  <count>  <name>'
                head = (parts[1], int(parts[2]), parts[3] if len(parts) > 3 else "")
                body = []
            elif head is not None:
                body.append(line.rstrip("\n"))
    if head is not None:
        yield head + (body,)


def parse_b2ftrace(path):
    titles = []
    rows = []
    nfld = None
    for typ, cnt, name, body in read_cf_blocks(path):
        if name == "nfld":
            nfld = int(body[0].split()[0])
        elif name == "title" and typ == "char":
            titles.append(body[0].strip())
        elif name == "data" and typ == "real":
            vals = np.array(" ".join(body).replace("D", "E").split(), dtype=float)
            if vals.size == cnt:  # drop a half-written final block (live run)
                rows.append(vals)
    if nfld is None or not rows:
        raise SystemExit("%s: no data blocks found" % path)
    return titles, np.vstack(rows)


ELEMENTS = {2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 10: "Ne",
            18: "Ar", 26: "Fe", 74: "W"}


def species_labels(path, ns):
    """Species names (D0, D1+, C0, C1+, ...) built from zn/am/zamin of the sibling b2ftrack file.

    Falls back to sp0, sp1, ... when no b2ftrack is found."""
    meta = {}
    # b2ftrace itself has no species data; its sibling b2ftrack (same dir, or the run dir) does
    d = os.path.dirname(os.path.abspath(path))
    for track in (os.path.join(d, "b2ftrack"), os.path.join(os.path.dirname(d), "b2ftrack")):
        if os.path.isfile(track):
            for typ, cnt, name, body in read_cf_blocks(track):
                if name in ("zn", "am", "zamin") and typ == "real":
                    meta[name] = np.array(" ".join(body).replace("D", "E").split(), dtype=float)
            if meta:
                break
    try:
        out = []
        for i in range(ns):
            zn, am, q = int(round(meta["zn"][i])), meta["am"][i], int(round(meta["zamin"][i]))
            el = ({1: "H", 2: "D", 3: "T"}.get(int(round(am)), "H") if zn == 1
                  else ELEMENTS.get(zn, "Z%d" % zn))
            out.append("%s%s" % (el, "%d+" % q if q > 0 else "0"))
        return out
    except (KeyError, IndexError):
        return ["sp%d" % i for i in range(ns)]


GROUPS = [
    ("continuity res.",   r"residual of continuity equation"),
    ("par. momentum res.", r"residual of parallel momentum equation, species"),
    ("total momentum res.", r"residual of total parallel momentum"),
    ("electron heat res.", r"residual of electron heat"),
    ("atom heat res.",    r"residual of all atom heat"),
    ("fl.neutral heat res.", r"residual of the fluid neutral heat"),
    ("potential res.",    r"residual of electric potential"),
    ("pressure corr.",    r"correction of pressure"),
    ("velocity corr.",    r"correction of velocity"),
    ("Te corr.",          r"correction of electron temperature"),
    ("Ti corr.",          r"correction of ion temperature"),
]


def build_groups(titles):
    """Return {group label: [column indices]} for the groups present in this file."""
    out = {}
    for label, pat in GROUPS:
        cols = [i for i, t in enumerate(titles) if re.search(pat, t)]
        if cols:
            out[label] = cols
    return out


def group_series(data, cols):
    """Largest |value| over the columns `cols` (e.g. over species) at every step."""
    sub = np.abs(data[:, cols])
    return sub.max(axis=1)


def trace_path(arg):
    """Run dir -> the newest of RUN/b2ftrace and RUN/b2mn.exe.dir/b2ftrace.

    While a job runs, the live trace is in b2mn.exe.dir; it is moved up to the
    run directory only when the job ends.
    """
    if not os.path.isdir(arg):
        if not os.path.isfile(arg):
            raise SystemExit("not found: %s" % arg)
        return arg
    cands = [os.path.join(arg, "b2mn.exe.dir", "b2ftrace"), os.path.join(arg, "b2ftrace")]
    cands = [c for c in cands if os.path.isfile(c)]
    if not cands:
        raise SystemExit("no b2ftrace in %s or %s/b2mn.exe.dir" % (arg, arg))
    return max(cands, key=os.path.getmtime)


def get_plt():
    """pyplot with the TkAgg backend (interactive window); Agg if there is no display."""
    import matplotlib
    if os.environ.get("DISPLAY"):
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
        print("note: no DISPLAY set, figures cannot be shown (use --save FILE, or ssh -X)")
    import matplotlib.pyplot as plt
    return plt


def plot_by_equation(label, path, titles, data, out=None):
    """One figure per run: one panel per equation, one line per species where relevant."""
    plt = get_plt()

    panels = [  # (panel title, regex on field titles, per species?)
        ("continuity residual", r"residual of continuity equation", True),
        ("parallel momentum residual", r"residual of parallel momentum equation, species", True),
        ("total momentum residual", r"residual of total parallel momentum", False),
        ("electron heat residual", r"residual of electron heat", False),
        ("atom heat residual", r"residual of all atom heat", False),
        ("potential residual", r"residual of electric potential", False),
        ("pressure correction", r"correction of pressure", True),
        ("velocity correction", r"correction of velocity, species", True),
        ("Te / Tatom / potential corrections",
         r"correction of (electron temperature|all atom temperature|electric potential)", False),
    ]
    ns = sum(1 for t in titles if re.search(r"residual of continuity equation", t))
    names = species_labels(path, ns)
    x = np.arange(data.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    for ax, (ptitle, pat, per_sp) in zip(axes.flat, panels):
        cols = [i for i, t in enumerate(titles) if re.search(pat, t)]
        for c in cols:
            y = np.abs(data[:, c])
            if not np.any(y > 0):
                continue  # e.g. neutral species with no equation solved
            m = re.search(r"species\s+(\d+)", titles[c])
            lab = names[int(m.group(1))] if (m and int(m.group(1)) < len(names)) else \
                titles[c].replace("norm of ", "").replace("correction of ", "")
            ax.semilogy(x, np.where(y > 0, y, np.nan), lw=0.9, label=lab)
        ax.set_title(ptitle)
        ax.set_xlabel("step in b2ftrace")
        ax.grid(True, which="both", alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2 if per_sp else 1)
    fig.suptitle("%s  (%d steps)" % (label, data.shape[0]))
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=110)
        print("wrote %s" % out)


def log_report(arg, n):
    """Progress report from run.log (or a log file given directly); n = averaging window in steps."""
    p = os.path.join(arg, "run.log") if os.path.isdir(arg) else arg
    if not os.path.isfile(p):
        raise SystemExit("not found: %s" % p)
    re_iter = re.compile(r"^\s*ITER\s+(\d+)\s+TIME\s+(\S+)\s+NTIM\s+(\d+)\s+DTIM\s+(\S+)")
    re_step = re.compile(r"b2-step-cpu\s*=\s*(\S+)\s+elapsed\s+(\S+)\s+estimated remaining CPU time \(s\)\s*=\s*(\S+)")
    re_eir = re.compile(r"EIRENE USED\s+(\S+)\s+CPU SECONDS")
    re_rel = re.compile(r"EIRENE source under-relaxation\s+(\d+)\s+(\d+)")
    it = []; step = []; eir = []; rel = None; remain = None
    with open(p, "r", errors="replace") as fh:
        for line in fh:
            if "ITER" in line:
                m = re_iter.match(line)
                if m:
                    it.append((int(m.group(1)), float(m.group(2)), int(m.group(3)),
                               float(m.group(4))))
                    continue
            if "b2-step-cpu" in line:
                m = re_step.search(line)
                if m:
                    step.append(float(m.group(2)))
                    remain = float(m.group(3))
                    continue
            if "EIRENE USED" in line:
                m = re_eir.search(line)
                if m:
                    eir.append(float(m.group(1)))
                    continue
            if "source under-relaxation" in line:
                m = re_rel.search(line)
                if m:
                    rel = (int(m.group(1)), int(m.group(2)))
    label = os.path.basename(os.path.dirname(os.path.abspath(p)))
    print("=== %s (run.log)" % label)
    if not it:
        print("  no 'ITER ... TIME' lines yet")
        return
    i, t, nt, dt = it[-1]
    print("  step %d of %d   false time %.4g s   dtim %.3g" % (i, nt, t, dt))
    if step:
        w = step[-n:]
        print("  elapsed per step (last %d): mean %.1f s" % (len(w), np.mean(w)))
    if eir and step:
        k = min(len(eir), len(step), n)
        e = np.mean(eir[-k:]); b = np.mean(step[-k:])
        print("  Eirene CPU per call: %.1f s  = %.0f%% of the step" % (e, 100 * e / b))
    if remain is not None:
        print("  code's own estimate of remaining CPU time: %.1f h" % (remain / 3600.0))
    if rel:
        print("  under-relaxation ACTIVE: r=%d, history counter %d" % rel)
    else:
        print("  under-relaxation: no 'source under-relaxation' lines (scheme not active)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs or b2ftrace files")
    ap.add_argument("--plot", action="store_true",
                    help="show a window overlaying the runs (max over species, 6 panels: continuity, "
                         "parallel momentum, electron heat, potential, pressure and Te corrections)")
    ap.add_argument("--save", metavar="PNG",
                    help="additionally save the figure(s): NAME_overlay.png for --plot, "
                         "NAME_eq_<run>.png for --by-eq (NAME = PNG without extension)")
    ap.add_argument("--last", type=int, default=500,
                    help="window (steps) for the trend: mean of last N vs the N before (default 500); "
                         "also the averaging window of --log")
    ap.add_argument("--by-eq", action="store_true",
                    help="show one window per run, one panel per equation "
                         "(species resolved)")
    ap.add_argument("--list", action="store_true",
                    help="list the numbered field titles of the (first) run and exit")
    ap.add_argument("--log", action="store_true",
                    help="report progress/timing from run.log (works on a running job)")
    args = ap.parse_args()

    if args.log:
        for r in args.runs:
            log_report(r, args.last)
        return

    runs = []
    paths = []
    for r in args.runs:
        p = trace_path(r)
        titles, data = parse_b2ftrace(p)
        d = os.path.dirname(os.path.abspath(p))
        if os.path.basename(d) == "b2mn.exe.dir":
            d = os.path.dirname(d)
        label = os.path.basename(d) or p
        runs.append((label, titles, data))
        paths.append(p)

    if args.list:
        for i, t in enumerate(runs[0][1]):
            print("%3d  %s" % (i + 1, t))
        return

    n = args.last
    for label, titles, data in runs:
        groups = build_groups(titles)
        print("=== %s : %d steps (fields: %d)" % (label, data.shape[0], data.shape[1]))
        print("%-22s %12s %12s %9s" % ("quantity (max over species)", "last", "min so far",
                                       "trend"))
        for g, cols in groups.items():
            s = group_series(data, cols)
            last = s[-min(n, len(s)):].mean()
            trend = ""
            if len(s) >= 2 * n:
                prev = s[-2 * n:-n].mean()
                trend = "%+.0f%%" % (100.0 * (last / prev - 1.0)) if prev > 0 else "n/a"
            print("%-22s %12.3e %12.3e %9s" % (g, last, s.min(), trend))
        print("  trend = change of the mean over the last %d steps vs the %d before;" % (n, n))
        print("  falling = still converging, ~0 = stalled/converged, rising = problem.\n")

    shown = False
    base, ext = os.path.splitext(args.save) if args.save else ("", "")
    ext = ext or ".png"

    if args.by_eq:
        for (label, titles, data), p in zip(runs, paths):
            out = "%s_eq_%s%s" % (base, label, ext) if args.save else None
            plot_by_equation(label, p, titles, data, out)
            shown = True

    if args.plot:
        plt = get_plt()

        panels = ["continuity res.", "par. momentum res.", "electron heat res.",
                  "potential res.", "pressure corr.", "Te corr."]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
        for ax, g in zip(axes.flat, panels):
            for label, titles, data in runs:
                cols = build_groups(titles).get(g)
                if cols:
                    s = group_series(data, cols)
                    ax.semilogy(np.arange(len(s)), np.where(s > 0, s, np.nan), lw=0.8,
                                label=label)
            ax.set_title(g)
            ax.set_xlabel("step in b2ftrace")
            ax.grid(True, which="both", alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        if args.save:
            out = "%s_overlay%s" % (base, ext)
            fig.savefig(out, dpi=110)
            print("wrote %s" % out)
        shown = True

    if shown and os.environ.get("DISPLAY"):
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
