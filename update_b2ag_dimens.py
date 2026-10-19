#!/usr/bin/env python3
"""
update_b2ag_dimens.py — (re)write b2ag.dat as the unstructured-grid template,
with nCi,nFc,nVx filled in from the traduit.out.b2us in the same folder.

This overwrites b2ag.dat entirely (whatever was there before, including the
old structured "*dimens (nx, ny, nx1, ny1)" format and any parameter you had
changed in it) with:

    *format
        unstructured
    *dimens         (nCi,nFc,nVx; free format)
        <nCi>       <nFc>       <nVx>
    *param          (param(0:99); free format)
     -1.0, 99*0.0
    'b2agfs_min_pitch'    '0.01'
    'b2agdr_redef_gmtry'  '0'
    'b2agfs_geometry'     'traduit.out.b2us'
    'b2agmt_style'        '1'
    'b2us_prep_mod_hc'    '1'
    'b2us_prep_Qalfmax'   '0.004'
    'b2us_prep_Qalfmin'   '0.002'
    'b2us_prep_Pbs_reduce_fac' '1e-6'
    'b2us_prep_set_beta_0'       '1'

NOTE: the b2us_prep_* parameters are written for completeness but have no effect on the
unstructured (traduit) path, because b2ag does not read them there. To apply the Qalfmin/Qalfmax/
Pbs_reduce_fac wall-face step to the geometry use modify_b2fgmtry_prep.py on the b2fgmtry.
The parameter values (Qalfmin 0.002, Qalfmax 0.004, Pbs_reduce_fac 1e-6, ...) are fixed text in TEMPLATE
in this script; edit it to change them.

Usage:
    update_b2ag_dimens.py DIR                 # writes DIR/b2ag.dat,
                                               # reading DIR/traduit.out.b2us
    update_b2ag_dimens.py B2AG_DAT [TRADUIT]  # explicit paths; TRADUIT
                                               # defaults to traduit.out.b2us
                                               # next to B2AG_DAT
"""
import re
import sys
from pathlib import Path

TEMPLATE = """*format
    unstructured
*dimens         (nCi,nFc,nVx; free format)
{dimens}
*param          (param(0:99); free format)
 -1.0, 99*0.0
'b2agfs_min_pitch'    '0.01'
'b2agdr_redef_gmtry'  '0'
'b2agfs_geometry'     'traduit.out.b2us'
'b2agmt_style'        '1'
'b2us_prep_mod_hc'    '1'
'b2us_prep_Qalfmax'   '0.004'
'b2us_prep_Qalfmin'   '0.002'
'b2us_prep_Pbs_reduce_fac' '1e-6'
'b2us_prep_set_beta_0'       '1'
"""


def parse_traduit_counts(path):
    """Return (nCi, nFc, nVx) from traduit.out.b2us's '*cf: int 6 nCi,nFc,...' header."""
    with open(path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("*cf:") and "nCi" in line and "nFc" in line and "nVx" in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                nums = re.findall(r"[-+]?\d+", lines[j])
                if len(nums) >= 3:
                    return int(nums[0]), int(nums[1]), int(nums[2])
    raise SystemExit(f"ERROR: could not find nCi,nFc,nVx header in {path}")


def build_b2ag(nCi, nFc, nVx):
    dimens = "{:>8}{:>11}{:>11}".format(nCi, nFc, nVx)
    return TEMPLATE.format(dimens=dimens)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    arg0 = Path(sys.argv[1])
    if len(sys.argv) == 2 and arg0.is_dir():
        b2ag_path = arg0 / "b2ag.dat"
        traduit_path = arg0 / "traduit.out.b2us"
    else:
        b2ag_path = arg0
        traduit_path = Path(sys.argv[2]) if len(sys.argv) > 2 else b2ag_path.parent / "traduit.out.b2us"

    if not traduit_path.is_file():
        raise SystemExit(f"ERROR: file not found: {traduit_path}")

    nCi, nFc, nVx = parse_traduit_counts(traduit_path)
    print(f"From {traduit_path}: nCi={nCi}, nFc={nFc}, nVx={nVx}")

    if b2ag_path.is_file():
        old = b2ag_path.read_text()
        if old.lstrip().startswith("*format") and "unstructured" in old:
            print(f"{b2ag_path}: already unstructured, rewriting it from the template (dimens updated, "
                  "other parameters reset to the template values)")
        else:
            print(f"{b2ag_path}: not unstructured yet, converting")
    else:
        print(f"{b2ag_path}: does not exist yet, creating")

    b2ag_path.write_text(build_b2ag(nCi, nFc, nVx))
    print(f"Wrote: {b2ag_path}")


if __name__ == "__main__":
    main()
