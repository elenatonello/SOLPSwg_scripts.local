#!/usr/bin/env python3
"""
write_b2transport_parameters.py — write b2.transport.parameters,
generalized over the number of species.

Reproduces exactly the reference file found in
runs/TCV/TCV62807/SILO/noDRIFTS_330008/b2.transport.parameters:

     &transport
     flag_dna=1, parm_dna=9*0.3,
     flag_dpa=0, parm_dpa=9*0,
     flag_vla=0, parm_vla=9*0,
     flag_vsa=1, parm_vsa=9*0.3,
     flag_hci=1, parm_hci=9*1.0,
     flag_hce=1, parm_hce=1.0,
     flag_sig=0, parm_sig=0,
     flag_alf=0, parm_alf=0,
     cflme= 3.00E-01,
     cflmi= 1.00E+01,
     cflmv= 5.00E-01,
     /

Only parm_dna and parm_hci use the Fortran "N*value" shorthand sized to
the number of SPECIES (9 in the reference); parm_dpa/parm_vla/parm_vsa
keep their fixed "2*..." shorthand as found (unrelated to species count)
and are copied verbatim.

Number of species is read, in order:
    1. --nspecies N (explicit override)
    2. DIR/../SOLPS_FLUIDS  (whitespace-separated species names, count them)
    3. DIR/../baserun/b2ai.dat  (the integer under the "*dimens (ns; ...)"
       header)
Unlike the region count in write_b2numerics_F/T.py, there is no safe
default here — an incorrect species count silently corrupts the physics
setup — so the script errors out if none of the above are available.

Usage:
    write_b2transport_parameters.py DIR [--nspecies N]
        # writes DIR/b2.transport.parameters
"""
import argparse
from pathlib import Path

TEMPLATE = """ &transport
 flag_dna=1, parm_dna={ns}*0.3,
 flag_dpa=0, parm_dpa={ns}*0,
 flag_vla=0, parm_vla={ns}*0,
 flag_vsa=1, parm_vsa={ns}*0.3,
 flag_hci=1, parm_hci={ns}*1.0,
 flag_hce=1, parm_hce=1.0,
 flag_sig=0, parm_sig=0,
 flag_alf=0, parm_alf=0,
 cflme= 3.00E-01,
 cflmi= 1.00E+01,
 cflmv= 5.00E-01,
 cflal= 5.00E-01,
 cflab= 5.00E-01,
 /
"""


def detect_nspecies(dir_path, explicit):
    if explicit is not None:
        return explicit, f"using --nspecies={explicit}"

    fluids_path = dir_path.parent / "SOLPS_FLUIDS"
    if fluids_path.is_file():
        n = len(fluids_path.read_text().split())
        return n, f"read nspecies={n} from {fluids_path}"

    b2ai_path = dir_path.parent / "baserun" / "b2ai.dat"
    if b2ai_path.is_file():
        lines = b2ai_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("*dimens"):
                for j in range(i + 1, min(i + 4, len(lines))):
                    tokens = lines[j].split()
                    if tokens:
                        n = int(tokens[0])
                        return n, f"read nspecies={n} from {b2ai_path}"

    raise SystemExit(
        f"ERROR: could not determine nspecies; provide --nspecies, or "
        f"ensure {fluids_path} or {b2ai_path} exists"
    )


def build_transport(nspecies):
    # the reference file has one cosmetic trailing space after "parm_dna=...,"
    # (a leftover from whatever tool generated it); reproduced explicitly
    # here since editors strip literal trailing whitespace from source.
    text = TEMPLATE.format(ns=nspecies)
    return text.replace(f"parm_dna={nspecies}*0.3,\n", f"parm_dna={nspecies}*0.3, \n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="run directory to write b2.transport.parameters into")
    parser.add_argument("--nspecies", type=int, default=None, help="number of species (overrides SOLPS_FLUIDS/b2ai.dat auto-detection)")
    args = parser.parse_args()

    dir_path = args.dir
    nspecies, note = detect_nspecies(dir_path, args.nspecies)
    print(note)

    out_path = dir_path / "b2.transport.parameters"
    out_path.write_text(build_transport(nspecies))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
