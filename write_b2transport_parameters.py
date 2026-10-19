#!/usr/bin/env python3
"""
write_b2transport_parameters.py — write b2.transport.parameters,
generalized over the number of species.

Based on the b2.transport.parameters of a TCV run (path/to/your/run/dir/). The script
writes the following, where 9 is replaced by the number of species:

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
     cflal= 5.00E-01,
     cflab= 5.00E-01,
     /

The per-species parameters parm_dna, parm_dpa, parm_vla, parm_vsa and parm_hci use the Fortran
"N*value" shorthand sized to the number of SPECIES (9 in the reference); every other value
(flags, parm_hce, parm_sig, parm_alf, the cfl* limits, including the cflal/cflab lines that the
reference file did not have) is fixed text of TEMPLATE. Edit TEMPLATE for other transport
coefficients; the script always overwrites the whole file.

Number of species is read, in order:
    1. --nspecies N (explicit override)
    2. DIR/../SOLPS_FLUIDS  (whitespace-separated species names, count them)
    3. DIR/../baserun/b2ai.dat  (the integer under the "*dimens (ns; ...)"
       header)
Unlike the region count in write_b2numerics_parameters.py, there is no safe
default here — an incorrect species count silently corrupts the physics
setup — so the script errors out if none of the above are available.

Anomalous pinch (vla)
    By default the pinch is off (flag_vla=0, parm_vla=0), as above. With --pinch V the pinch is switched
    on with an INWARD (towards the core) speed V in m/s for all species. The sign of parm_vla is not
    simply +V: in the unstructured B2.5 the direction of a positive vla is set by the mesh
    (radial face coefficient = fcS*fcQalf(:,1)*vla, flux positive from fcCv(:,1) to fcCv(:,2)), and it
    differs between meshes. The script therefore reads the geometry, determines which way a positive
    vla points (B2StateIO.vla_inward_sign in b2state_io.py) and writes
        flag_vla=1, parm_vla=N*(+V or -V)
    so that the pinch is always inward. Use a negative V for an outward pinch.
    The geometry is DIR/../baserun/b2fgmtry (then DIR/b2fgmtry), or --gmtry FILE.

Usage:
    write_b2transport_parameters.py DIR [--nspecies N] [--pinch V] [--gmtry b2fgmtry]
        # writes DIR/b2.transport.parameters
"""
import argparse
import sys
from pathlib import Path

_thisdir = Path(__file__).resolve().parent
if str(_thisdir) not in sys.path:
    sys.path.insert(0, str(_thisdir))

TEMPLATE = """ &transport
 flag_dna=1, parm_dna={ns}*0.3,
 flag_dpa=0, parm_dpa={ns}*0,
 flag_vla={vla_flag}, parm_vla={ns}*{vla_val},
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


def find_gmtry(dir_path, explicit):
    if explicit is not None:
        return explicit
    for cand in (dir_path.parent / "baserun" / "b2fgmtry", dir_path / "b2fgmtry"):
        if cand.is_file():
            return cand
    raise SystemExit(
        f"ERROR: --pinch needs the geometry; provide --gmtry, or ensure "
        f"{dir_path.parent / 'baserun' / 'b2fgmtry'} or {dir_path / 'b2fgmtry'} exists"
    )


def inward_vla(gmtry, speed):
    """parm_vla value that gives an inward pinch of `speed` m/s on mesh `gmtry`."""
    from b2state_io import B2StateIO
    try:
        sign, cons = B2StateIO().vla_inward_sign(gmtry)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")
    note = (f"geometry {gmtry}: positive vla points {'towards' if sign > 0 else 'away from'} the core "
            f"({cons * 100:.1f}% of radial faces agree) -> parm_vla = {sign * speed:g}")
    if cons < 0.99:
        note += "\nWARNING: the radial direction is not uniform over the mesh; a single sign for parm_vla is not exact"
    return sign * speed, note


def build_transport(nspecies, vla=None):
    # the reference file has one cosmetic trailing space after "parm_dna=...,"
    # (a leftover from whatever tool generated it); reproduced explicitly
    # here since editors strip literal trailing whitespace from source.
    if vla is None or vla == 0:
        text = TEMPLATE.format(ns=nspecies, vla_flag=0, vla_val=0)
    else:
        text = TEMPLATE.format(ns=nspecies, vla_flag=1, vla_val=repr(float(vla)))
    return text.replace(f"parm_dna={nspecies}*0.3,\n", f"parm_dna={nspecies}*0.3, \n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="run directory to write b2.transport.parameters into")
    parser.add_argument("--nspecies", type=int, default=None, help="number of species (overrides SOLPS_FLUIDS/b2ai.dat auto-detection)")
    parser.add_argument("--pinch", type=float, default=0.0, metavar="V",
                        help="switch the anomalous pinch on with an inward speed V [m/s] for all species; the sign of "
                             "parm_vla is chosen from the geometry so that V>0 is always towards the core (default 0 = off)")
    parser.add_argument("--gmtry", type=Path, default=None,
                        help="b2fgmtry used to determine the sign of vla (default DIR/../baserun/b2fgmtry, then DIR/b2fgmtry)")
    args = parser.parse_args()

    dir_path = args.dir
    nspecies, note = detect_nspecies(dir_path, args.nspecies)
    print(note)

    vla = None
    if args.pinch != 0.0:
        vla, vla_note = inward_vla(find_gmtry(dir_path, args.gmtry), args.pinch)
        print(vla_note)

    out_path = dir_path / "b2.transport.parameters"
    out_path.write_text(build_transport(nspecies, vla))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
