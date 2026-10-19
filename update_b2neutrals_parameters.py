#!/usr/bin/env python3
"""
update_b2neutrals_parameters.py — apply targeted, in-place edits to an
existing b2.neutrals.parameters. Everything not touched below is left
byte-for-byte untouched, and edited numeric lines keep their original
column spacing (only the changed field's padding is adjusted, growing
only if the new value no longer fits its original slot).

Two independent edits, either or both may be requested:

  1. --EIRENE_MOD N
     Append " eirene_mod= N," to the &NEUTRALS namelist, just before the
     closing '/' (SOLPS's "call Eirene every N full B2.5 iterations"
     switch, default 1). If an eirene_mod= line already exists, its
     value is updated in place instead of appending a duplicate
     (idempotent).

  2. --GASPUFF X
     Set the gas puff flux (particles/s) in userfluxparm(1,1), i.e. the
     column of that array matching the stratum flagged 'C' (constant
     flux) in crcstra=. In the reference file
     (path/to/your/run/dir/b2.neutrals.parameters)
     that is stratum 17 (crcstra index 16, 0-based), originally
     3.000E+22. The 'C' column is auto-detected from crcstra= on every
     run (not hardcoded), and the script refuses to guess if crcstra
     has zero or more than one 'C' entries — pass --puff-column
     (1-based) to disambiguate in that case.

The file is edited in place and NO backup is kept: keep a copy (or use git) if you may want the
original back.

Usage:
    update_b2neutrals_parameters.py DIR  [--EIRENE_MOD N] [--GASPUFF X] [--puff-column N]
        # edits DIR/b2.neutrals.parameters in place
    update_b2neutrals_parameters.py FILE [--EIRENE_MOD N] [--GASPUFF X] [--puff-column N]
        # edits FILE directly
"""
import argparse
import re
from pathlib import Path

FIELD_RE = re.compile(r"[^,]*,")


def _field_value(field):
    """The bare value inside one 'field' (leading ws + value + trailing
    ws + comma), i.e. with padding and the trailing comma stripped."""
    return field[:-1].strip()


def _rebuild_field(field, new_value):
    """Re-render one field with a (possibly different) value, preserving
    the field's original leading whitespace and, as long as the new
    value fits, its original total width. If the new value is wider than
    the original slot, the field simply grows."""
    core = field[:-1]
    leading_ws = core[: len(core) - len(core.lstrip())]
    new_trailing_len = max(0, len(core) - len(leading_ws) - len(new_value))
    return leading_ws + new_value + " " * new_trailing_len + ","


def _find_line(lines, name):
    pattern = re.compile(r"^(\s*" + re.escape(name) + r"=)(.*)$")
    for i, line in enumerate(lines):
        m = pattern.match(line.rstrip("\n"))
        if m:
            return i, m.group(1), m.group(2)
    return None, None, None


def fmt_exp(x):
    return f"{x:.3E}"


def apply_eirene_mod(lines, value):
    pattern = re.compile(r"^(\s*)eirene_mod(\s*=\s*)(\S+?),?\s*$", re.IGNORECASE)

    for i, line in enumerate(lines):
        m = pattern.match(line.rstrip("\n"))
        if m:
            old_value = m.group(3)
            lines[i] = f"{m.group(1)}eirene_mod{m.group(2)}{value},\n"
            return "updated", old_value

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "/":
            lines.insert(i, f" eirene_mod= {value},\n")
            return "appended", None

    raise SystemExit("ERROR: could not find the namelist terminator ('/') to append eirene_mod before")


def find_puff_column(lines):
    """0-based column index of the stratum flagged 'C' (constant flux /
    gas puff) in crcstra=."""
    i, _, rest = _find_line(lines, "crcstra")
    if i is None:
        raise SystemExit("ERROR: could not find a 'crcstra=' line")

    values = [_field_value(f) for f in FIELD_RE.findall(rest)]
    matches = [idx for idx, v in enumerate(values) if v.strip("'\"") == "C"]

    if not matches:
        raise SystemExit(
            "ERROR: no crcstra entry equal to 'C' (gas puff stratum) found; "
            "pass --puff-column to set it explicitly"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"ERROR: multiple crcstra entries equal to 'C' found at columns "
            f"{[m + 1 for m in matches]} (1-based); pass --puff-column to disambiguate"
        )
    return matches[0]


def set_column(lines, name, index, new_value):
    i, prefix, rest = _find_line(lines, name)
    if i is None:
        raise SystemExit(f"ERROR: could not find a '{name}=' line")

    fields = FIELD_RE.findall(rest)
    if index >= len(fields):
        raise SystemExit(f"ERROR: column {index + 1} out of range for '{name}=' ({len(fields)} columns)")

    old_value = _field_value(fields[index])
    fields[index] = _rebuild_field(fields[index], new_value)
    lines[i] = prefix + "".join(fields) + "\n"
    return old_value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="run directory containing b2.neutrals.parameters, or the file itself")
    parser.add_argument("--EIRENE_MOD", type=int, default=None, help="value for eirene_mod (frequency of Eirene calls, in full B2.5 iterations)")
    parser.add_argument("--GASPUFF", type=float, default=None, help="gas puff flux (particles/s) to set in userfluxparm(1,1) at the crcstra='C' column")
    parser.add_argument("--puff-column", type=int, default=None, help="1-based column override for the gas-puff stratum (auto-detected from crcstra='C' by default)")
    args = parser.parse_args()

    if args.EIRENE_MOD is None and args.GASPUFF is None:
        raise SystemExit("ERROR: nothing to do; pass --EIRENE_MOD and/or --GASPUFF")

    b2n_path = args.path / "b2.neutrals.parameters" if args.path.is_dir() else args.path
    if not b2n_path.is_file():
        raise SystemExit(f"ERROR: file not found: {b2n_path}")

    lines = b2n_path.read_text().splitlines(keepends=True)

    if args.EIRENE_MOD is not None:
        action, old_value = apply_eirene_mod(lines, args.EIRENE_MOD)
        if action == "updated":
            print(f"eirene_mod: {old_value} -> {args.EIRENE_MOD} (updated existing line)")
        else:
            print(f"eirene_mod= {args.EIRENE_MOD}, (appended before namelist terminator)")

    if args.GASPUFF is not None:
        column = (args.puff_column - 1) if args.puff_column is not None else find_puff_column(lines)
        new_value = fmt_exp(args.GASPUFF)
        old_value = set_column(lines, "userfluxparm(1,1)", column, new_value)
        print(f"userfluxparm(1,1) column {column + 1} (gas puff): {old_value} -> {new_value}")

    b2n_path.write_text("".join(lines))
    print(f"Wrote: {b2n_path}")


if __name__ == "__main__":
    main()
