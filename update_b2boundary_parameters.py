#!/usr/bin/env python3
"""
update_b2boundary_parameters.py — apply a fixed set of targeted, in-place
edits to an existing b2.boundary.parameters file. Every other line
(bcchar, bcstart/bcend, bcpot/potpar, the other bccon/conpar/bcmom/mompar
species rows, lcbs, nniso, bcenk/enkpar, ...) is left byte-for-byte
untouched, and edited lines keep their original column spacing: each
value's own field width/padding is preserved, and only fields whose
value actually changes are re-rendered (widening only if the new value
no longer fits the original slot).

Edits 1-4 change the FIRST value (column 1, 1-based) of the array, which is
the CORE boundary segment (bcstart(1) = -21 in the reference file). Edit 6
(bccon(0,1)) instead changes two specific entries of its row, given as 0-based
indices (index 1 = the 2nd value). Both conventions are used below; they are
stated explicitly in each item.

Edits applied (the original values are those found in
path/to/your/run/dir/b2.boundary.parameters BEFORE
this script was run on it; that file has since been edited and no longer shows them):

  1. enepar(1,1): first value (core), originally 1.000E+06 -> --SHEE value
     (left untouched if --SHEE is not given).
  2. enipar(1,1): first value (core), originally 1.000E+06 -> --SHII value
     (left untouched if --SHII is not given).
  3. bcene: first value (core), originally 8 -> 16.
  4. bceni: first value (core), originally 8 -> 27.
  5. conpar(0,1,1): ALL values -> 0 (originally 0,3.0E+19,0,0,0,0,0,0,0).
  6. bccon(0,1): index 1 (0-based; "species 1", i.e. the 2nd
     species, originally 1) -> 27, AND the last entry (originally 8)
     -> 27. Row becomes 5,27,8,8,8,8,8,8,27.

The file is edited in place and NO backup is kept: keep a copy (or use git) if you may want the
original back. The script stops with an error if one of the arrays above is not found.

This mapping (row bccon(0,1)/conpar(0,1,1) = the core segment; column
index = 0-based species order D0,D+1,C0,C+1..C+6 as listed in
SOLPS_FLUIDS/b2ai.dat) was inferred from the fact that edits 3-6 are all
described together as core-boundary edits, and rows bccon(0,1)/
conpar(0,1,1) are the only rows in the reference file that already
differ from the rest of their block. Double-check the printed
before/after values against your intent before trusting the output.

Usage:
    update_b2boundary_parameters.py DIR [--SHEE X] [--SHII Y]
        # edits DIR/b2.boundary.parameters in place
    update_b2boundary_parameters.py FILE [--SHEE X] [--SHII Y]
        # edits FILE directly
"""
import argparse
import re
from pathlib import Path

FIELD_RE = re.compile(r"[^,]*,")


def _field_value(field):
    """The bare number/string inside one 'field' (leading ws + value +
    trailing ws + comma), i.e. with padding and the trailing comma
    stripped."""
    return field[:-1].strip()


def _rebuild_field(field, new_value):
    """Re-render one field with a (possibly different) value, preserving
    the field's original leading whitespace and, as long as the new
    value fits, its original total width (so later columns stay aligned
    with unrelated rows in the file). If the new value is wider than the
    original slot, the field simply grows rather than being truncated."""
    core = field[:-1]
    leading_ws = core[: len(core) - len(core.lstrip())]
    new_trailing_len = max(0, len(core) - len(leading_ws) - len(new_value))
    return leading_ws + new_value + " " * new_trailing_len + ","


def replace_array_line(lines, name, transform):
    """Find the line 'NAME=v1,v2,...' (NAME matched exactly, e.g. 'bcene'
    or 'enepar(1,1)') and replace its comma-separated values via
    transform(values) -> new_values, preserving each field's original
    column spacing (only the fields whose value actually changes shift
    the padding, and only if the new value is wider than the original).
    Returns (old_values, new_values) or raises if NAME is not found."""
    pattern = re.compile(r"^(\s*" + re.escape(name) + r"=)(.*)$")
    for i, line in enumerate(lines):
        m = pattern.match(line.rstrip("\n"))
        if m:
            prefix, rest = m.group(1), m.group(2)
            fields = FIELD_RE.findall(rest)
            values = [_field_value(f) for f in fields]
            new_values = transform(list(values))
            new_fields = [_rebuild_field(f, v) for f, v in zip(fields, new_values)]
            lines[i] = prefix + "".join(new_fields) + "\n"
            return values, new_values
    raise SystemExit(f"ERROR: could not find a '{name}=' line to edit")


def fmt_exp(x):
    return f"{x:.3E}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="run directory containing b2.boundary.parameters, or the file itself")
    parser.add_argument("--SHEE", type=float, default=None, help="sheath electron energy transmission value for enepar(1,1) core column (replaces 1.0E+06)")
    parser.add_argument("--SHII", type=float, default=None, help="sheath ion energy transmission value for enipar(1,1) core column (replaces 1.0E+06)")
    args = parser.parse_args()

    b2b_path = args.path / "b2.boundary.parameters" if args.path.is_dir() else args.path
    if not b2b_path.is_file():
        raise SystemExit(f"ERROR: file not found: {b2b_path}")

    lines = b2b_path.read_text().splitlines(keepends=True)

    edits = []

    if args.SHEE is not None:
        old, new = replace_array_line(lines, "enepar(1,1)", lambda v: [fmt_exp(args.SHEE)] + v[1:])
        edits.append(("enepar(1,1)", old, new))
    else:
        print("--SHEE not given: enepar(1,1) left unchanged")

    if args.SHII is not None:
        old, new = replace_array_line(lines, "enipar(1,1)", lambda v: [fmt_exp(args.SHII)] + v[1:])
        edits.append(("enipar(1,1)", old, new))
    else:
        print("--SHII not given: enipar(1,1) left unchanged")

    old, new = replace_array_line(lines, "bcene", lambda v: ["16"] + v[1:])
    edits.append(("bcene", old, new))

    old, new = replace_array_line(lines, "bceni", lambda v: ["27"] + v[1:])
    edits.append(("bceni", old, new))

    old, new = replace_array_line(lines, "conpar(0,1,1)", lambda v: ["0.00"] * len(v))
    edits.append(("conpar(0,1,1)", old, new))

    def bccon_transform(v):
        v = list(v)
        v[1] = "27"
        v[-1] = "27"
        return v

    old, new = replace_array_line(lines, "bccon(0,1)", bccon_transform)
    edits.append(("bccon(0,1)", old, new))

    for name, old, new in edits:
        print(f"{name}: {old} -> {new}")

    b2b_path.write_text("".join(lines))
    print(f"Wrote: {b2b_path}")


if __name__ == "__main__":
    main()
