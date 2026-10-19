#!/usr/bin/env python3
"""
update_spt_input.py — edit the reflection-model blocks (*** 6a) of an EIRENE
input file, in place, across input.eir / input_avg.eir / input_fin.eir /
input_hybrid.eir at once (those found in the directory). The files are
rewritten without a backup: keep a copy (or use git) if you may want the
originals back.

Rules applied to every SURFMOD_* block inside "*** 6a. General data for
reflection model" (name line + 4 raw numeric lines: ints / material / recyc
/ sputter):

  1. Temperature-dependent chemical-sputtering block, appended right after
     the 4 raw numeric lines. Temperature is read from the "_<N>K" suffix
     of the SURFMOD name (e.g. SURFMOD_1_CARBON_SPT_580K -> 580 K):
       - T > 100 degC : append the block for D+, D and D2
       - T <= 100 degC: append the block for D+ only
     Each species block is (fixed total line width per keyword, regardless
     of species width, so columns stay aligned for D+/D/D2):
         ISRC     spec           2
         LCHSPNWL spec           0
         RECYCC   spec       3.50000E-02
         ESPUTC   spec       1.00000E+00
     with "spec" replaced by D+ / D / D2.
     If a block has no "_<N>K" suffix, temperature is unknown and rule 1 is
     skipped for it (logged as a warning) rather than guessed.
     Idempotent: a block that already has an ISRC line is left unchanged.

  2. Independently of rule 1: if the block's RECYCT value (2nd column of
     the recyc line) is <= 0.99, append
         RECYCT   D2       1.00000E+00
     right after whatever rule 1 produced (if RECYCT > 0.99, nothing is
     added). Idempotent: skipped if that exact line is already the last
     line of the block.

Usage:
    update_spt_input.py DIR                 # auto-finds and edits
                                             # input.eir, input_avg.eir,
                                             # input_fin.eir, input_hybrid.eir in DIR
    update_spt_input.py FILE [FILE ...]     # edits the given files
"""
import sys
import re
from pathlib import Path

AUTO_NAMES = ["input.eir", "input_avg.eir", "input_fin.eir", "input_hybrid.eir"]

# (keyword, value, total line width). Width is fixed per keyword so that
# the value column stays put regardless of the species field's width
# (D+ = 2 chars, D = 1 char, D2 = 2 chars).
SPT_FIELDS = [
    ("ISRC", "2", 23),
    ("LCHSPNWL", "0", 23),
    ("RECYCC", "3.50000E-02", 29),
    ("ESPUTC", "1.00000E+00", 29),
]
KEYWORD_WIDTH = 9  # keyword left-justified into this many columns before the species


def fmt_line(keyword, spec, val, total_width):
    prefix = f"{keyword:<{KEYWORD_WIDTH}}{spec}"
    spaces = total_width - len(prefix) - len(val)
    return prefix + " " * spaces + val


def spt_block(spec):
    return [fmt_line(kw, spec, val, width) for kw, val, width in SPT_FIELDS]


RECYCT_D2_LINE = fmt_line("RECYCT", "D2", "1.00000E+00", 29)

TEMP_RE = re.compile(r"_(\d+)K$")


def field0(line):
    return line.split(None, 1)[0] if line.split() else ""


def recyct_of(sb):
    """RECYCT = 2nd column of the recyc line (index 3: name,ints,material,recyc)."""
    if len(sb) < 4:
        return None
    parts = sb[3].split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def process_block(sb, name, log):
    sb = list(sb)

    # --- Rule 1: temperature-dependent chemical-sputtering block. ---
    already = any(field0(l) == "ISRC" for l in sb)
    if already:
        log.append("    rule 1: ISRC block already present (unchanged)")
    else:
        m = TEMP_RE.search(name)
        if not m:
            log.append("    rule 1: WARNING: no _<N>K suffix in name; temperature unknown, skipped")
        else:
            temp_c = int(m.group(1)) - 273.15
            specs = ["D+", "D", "D2"] if temp_c > 100 else ["D+"]
            for sp in specs:
                sb += spt_block(sp)
            log.append(f"    rule 1: T={m.group(1)}K ({temp_c:.1f} degC) -> appended block(s) for {specs}")

    # --- Rule 2: RECYCT <= 0.99 -> append "RECYCT D2 1.0" after everything else. ---
    # recyct is read from the original recyc line, which is always sb[3]
    # (rule 1 only ever appends lines after this point).
    recyct = recyct_of(sb)
    if recyct is None:
        log.append("    rule 2: WARNING: could not parse RECYCT; skipped")
    elif recyct > 0.99:
        log.append(f"    rule 2: RECYCT={recyct} > 0.99 -> not added")
    else:
        if sb and sb[-1].strip() == RECYCT_D2_LINE.strip():
            log.append(f"    rule 2: RECYCT={recyct} <= 0.99 -> already present (unchanged)")
        else:
            sb.append(RECYCT_D2_LINE)
            log.append(f"    rule 2: RECYCT={recyct} <= 0.99 -> appended RECYCT D2 1.0")

    return sb


def block_span(lines, headers, tag):
    for j, i in enumerate(headers):
        if lines[i].startswith(tag):
            end = headers[j + 1] if j + 1 < len(headers) else len(lines)
            return i, end
    raise SystemExit(f"ERROR: block header not found: {tag!r}")


def process_file(path):
    print(f"=== {path} ===")
    with open(path, "r", newline="") as f:
        text = f.read()
    lines = text.split("\n")

    headers = [i for i, l in enumerate(lines) if l.startswith("***")]
    s6, e6 = block_span(lines, headers, "*** 6a.")
    surf_idx = [i for i in range(s6, e6) if lines[i].startswith("SURFMOD_")]
    if not surf_idx:
        raise SystemExit(f"ERROR: no SURFMOD_* blocks found inside *** 6a. in {path}")

    new_lines = lines[:surf_idx[0]]
    for j, s in enumerate(surf_idx):
        e = surf_idx[j + 1] if j + 1 < len(surf_idx) else e6
        name = lines[s].strip()
        log = [f"  {name}:"]
        new_lines += process_block(lines[s:e], name, log)
        print("\n".join(log))
    new_lines += lines[e6:]

    with open(path, "w", newline="") as f:
        f.write("\n".join(new_lines))
    print(f"Wrote: {path}")
    print("-" * 60)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    args = [Path(a) for a in sys.argv[1:]]
    if len(args) == 1 and args[0].is_dir():
        files = [args[0] / n for n in AUTO_NAMES if (args[0] / n).is_file()]
        missing = [n for n in AUTO_NAMES if not (args[0] / n).is_file()]
        if missing:
            print(f"Note: not found in {args[0]}, skipping: {missing}")
        if not files:
            raise SystemExit(f"ERROR: none of {AUTO_NAMES} found in {args[0]}")
    else:
        files = args
        for f in files:
            if not f.is_file():
                raise SystemExit(f"ERROR: file not found: {f}")

    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
