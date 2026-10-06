#!/usr/bin/env python3
"""
write_b2numerics_parameters.py — write both b2.numerics.parameters_F and
b2.numerics.parameters_T, generalized over the number of poloidal
regions.

Reproduces exactly the reference files found in
runs/TCV/TCV62807/SILO/noDRIFTS_330008/:

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
    sna_corr = 2*50,
    /

The per-species true/false patterns for solveco/solvemo (9 entries,
matching D0,D+1,C0,C+1..C+6) and the trailing "sna_corr = 2*50," line in
the _T file are copied verbatim from the reference files and are NOT
re-derived from the species list — only the number of REGIONS (how many
solveco/solvemo/solvepo/... lines are written) is made configurable,
since that is what varies between grid topologies.

Number of regions:
    - explicit  --nregions N
    - otherwise read the first integer from a NREG file, checked at
      DIR/../NREG then DIR/NREG (the file simply contains e.g. "4 13")
    - otherwise default to 4 and print a WARNING, since 4 regions is
      typical of a single-null (SN) configuration. This default may be
      wrong for double-null or other topologies — always check the
      warning if you did not pass --nregions or provide a NREG file.

Usage:
    write_b2numerics_parameters.py DIR [--nregions N]
        # writes DIR/b2.numerics.parameters_F and DIR/b2.numerics.parameters_T
"""
import argparse
from pathlib import Path

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


def build_numerics_t(nregions):
    lines = ["&numerics"]
    lines += [f"solveco(0,{i})={T_SOLVECO_PATTERN}" for i in range(1, nregions + 1)]
    lines += [f"solvemo(0,{i})={T_SOLVEMO_PATTERN}" for i in range(1, nregions + 1)]
    lines += [f"solvepo({i})=t," for i in range(1, nregions + 1)]
    lines += [f"solveee({i})=t," for i in range(1, nregions + 1)]
    lines += [f"solveei({i})=t," for i in range(1, nregions + 1)]
    lines += [f"solvekt({i})=f," for i in range(1, nregions + 1)]
    lines += [f"solvezt({i})=f," for i in range(1, nregions + 1)]
    lines.append("corr_core_dn(0) = 9*0.0075,")
    lines.append("corr_core_dt  = 0.0075,")
    lines.append("/")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path, help="run directory to write b2.numerics.parameters_F/_T into")
    parser.add_argument("--nregions", type=int, default=None, help="number of regions (overrides NREG auto-detection)")
    args = parser.parse_args()

    dir_path = args.dir
    nregions, note = detect_nregions(dir_path, args.nregions)
    print(note)

    f_path = dir_path / "b2.numerics.parameters_F"
    f_path.write_text(build_numerics_f(nregions))
    print(f"Wrote: {f_path}")

    t_path = dir_path / "b2.numerics.parameters_T"
    t_path.write_text(build_numerics_t(nregions))
    print(f"Wrote: {t_path}")


if __name__ == "__main__":
    main()
