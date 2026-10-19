#!/usr/bin/env python3
"""
write_b2numerics_parameters.py — write both b2.numerics.parameters_F and
b2.numerics.parameters_T, generalized over the number of poloidal
regions.

Based on the reference files of a TCV run (path/to/your/run/dir/). The two
files differ in which equations are solved: in _F the continuity, potential and
electron/ion energy solves are all off (solveco, solvepo, solveee, solveei = f;
only the solvemo momentum pattern is on); in _T potential and electron/ion
energy are on and the continuity pattern is the same as for momentum. solvekt
and solvezt are off in both. What is written is exactly:

  b2.numerics.parameters_F:
    &numerics
    solveco(0,1)=f,f,f,f,f,f,f,f,f,
    ...                              (one line per region)
    solvemo(0,1)=f,t,f,t,t,t,t,t,t,
    ...                              (one line per region)
    solvepo(1)=f,
    ...                              (one line per region)
    solveee(1)=f,
    solveei(1)=f,
    solvekt(1)=f,
    solvezt(1)=f,
    /

  b2.numerics.parameters_T:
    &numerics
    solveco(0,1)=f,t,f,t,t,t,t,t,t,
    ...                              (one line per region)
    solvemo(0,1)=f,t,f,t,t,t,t,t,t,
    ...                              (one line per region)
    solvepo(1)=t,
    ...                              (one line per region)
    solveee(1)=t,
    solveei(1)=t,
    solvekt(1)=f,
    solvezt(1)=f,
    dtco(0,1)=9*100.0,
    dtee(1)=50.0,
    dtei(1)=50.0,
    corr_core_dn(0) = 9*0.0075,
    corr_core_dt  = 0.0075,
    /

where 9 in dtco is the number of species (NSPECIES, see below). The per-species true/false patterns for solveco/solvemo (9 entries,
matching D0,D+1,C0,C+1..C+6) and the two trailing "corr_core_dn/dt" lines in
the _T file are fixed text and are NOT re-derived from the species list —
only the number of REGIONS (how many solveco/solvemo/solvepo/... lines are
written) is made configurable, since that is what varies between grid
topologies. Edit the *_PATTERN constants (and build_numerics_t) for another
species set. sna_corr is deliberately NOT written (it does not work on the wide grid).
dtco/dtee/dtei are written once (region 1 only) and only in the _T file.

Number of species (for dtco): explicit --nspecies N, otherwise the same detection as
write_b2transport_parameters.py (DIR/../SOLPS_FLUIDS, then DIR/../baserun/b2ai.dat); an error
if neither is available.

Number of regions:
    - explicit  --nregions N
    - otherwise read the first integer from a NREG file, checked at
      DIR/../NREG then DIR/NREG (the file simply contains e.g. "4 13")
    - otherwise default to 4 and print a WARNING, since 4 regions is
      typical of a single-null (SN) configuration. This default may be
      wrong for double-null or other topologies — always check the
      warning if you did not pass --nregions or provide a NREG file.

Usage:
    write_b2numerics_parameters.py DIR [--nregions N] [--nspecies N]
        # writes DIR/b2.numerics.parameters_F and DIR/b2.numerics.parameters_T
"""
import argparse
from pathlib import Path

from write_b2transport_parameters import detect_nspecies

F_SOLVECO_PATTERN = "f,f,f,f,f,f,f,f,f,"
F_SOLVEMO_PATTERN = "f,t,f,t,t,t,t,t,t,"

T_SOLVECO_PATTERN = "f,t,f,t,t,t,t,t,t,"
T_SOLVEMO_PATTERN = "f,t,f,t,t,t,t,t,t,"


def detect_nregions(dir_path, explicit):
    if explicit is not None:
        return explicit, f"using --nregions={explicit}"

    for candidate in (dir_path.parent / "NREG", dir_path / "NREG"):
        if candidate.is_file():
            tokens = candidate.read_text().split()
            if tokens:
                n = int(tokens[0])
                return n, f"read nregions={n} from {candidate}"

    n = 4
    print(
        f"WARNING: no NREG file found at {dir_path.parent / 'NREG'} or "
        f"{dir_path / 'NREG'}; assuming nregions={n} (typical of a "
        f"single-null (SN) configuration). Pass --nregions to override "
        f"if this grid is not SN."
    )
    return n, f"defaulted nregions={n}"


def build_numerics_f(nregions):
    lines = ["&numerics"]
    lines += [f"solveco(0,{i})={F_SOLVECO_PATTERN}" for i in range(1, nregions + 1)]
    lines += [f"solvemo(0,{i})={F_SOLVEMO_PATTERN}" for i in range(1, nregions + 1)]
    lines += [f"solvepo({i})=f," for i in range(1, nregions + 1)]
    lines += [f"solveee({i})=f," for i in range(1, nregions + 1)]
    lines += [f"solveei({i})=f," for i in range(1, nregions + 1)]
    lines += [f"solvekt({i})=f," for i in range(1, nregions + 1)]
    lines += [f"solvezt({i})=f," for i in range(1, nregions + 1)]
    lines.append("/")
    return "\n".join(lines) + "\n"


def build_numerics_t(nregions, nspecies):
    lines = ["&numerics"]
    lines += [f"solveco(0,{i})={T_SOLVECO_PATTERN}" for i in range(1, nregions + 1)]
    lines += [f"solvemo(0,{i})={T_SOLVEMO_PATTERN}" for i in range(1, nregions + 1)]
    lines += [f"solvepo({i})=t," for i in range(1, nregions + 1)]
    lines += [f"solveee({i})=t," for i in range(1, nregions + 1)]
    lines += [f"solveei({i})=t," for i in range(1, nregions + 1)]
    lines += [f"solvekt({i})=f," for i in range(1, nregions + 1)]
    lines += [f"solvezt({i})=f," for i in range(1, nregions + 1)]
    lines.append(f"dtco(0,1)={nspecies}*100.0,")
    lines.append(f"dtee(1)=50.0,")
    lines.append(f"dtei(1)=50.0,")
    lines.append(f"corr_core_dn(0) = 9*0.0075,")
    lines.append(f"corr_core_dt  = 0.0075,")
    lines.append("/")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="run directory to write b2.numerics.parameters_F/_T into")
    parser.add_argument("--nregions", type=int, default=None, help="number of regions (overrides NREG auto-detection)")
    parser.add_argument("--nspecies", type=int, default=None,
                        help="number of species for dtco (overrides SOLPS_FLUIDS/b2ai.dat auto-detection)")
    args = parser.parse_args()

    dir_path = args.dir
    nregions, note = detect_nregions(dir_path, args.nregions)
    print(note)

    nspecies, note = detect_nspecies(dir_path, args.nspecies)
    print(note)

    f_path = dir_path / "b2.numerics.parameters_F"
    f_path.write_text(build_numerics_f(nregions))
    print(f"Wrote: {f_path}")

    t_path = dir_path / "b2.numerics.parameters_T"
    t_path.write_text(build_numerics_t(nregions, nspecies))
    print(f"Wrote: {t_path}")


if __name__ == "__main__":
    main()
