#!/usr/bin/env python3
"""
Standalone script: Generate uinput.crr from traduit.out.b2us

Usage:
    python traduit_to_uinput_crr.py <traduit_file> [output_file]

If output_file is omitted, defaults to 'uinput.crr' in the same directory as traduit_file.
Run without arguments or with -h/--help to print this text.

The script extracts grid counts (nCi, nFc, nCg) from the traduit header and generates
a pseudo carre.out file suitable for uinp input in coupled EIRENE cases.

Description:
    uinput.crr is a pseudo carre.out. It contains no real geometry: uinp's uincrr
    only reads it to derive NCVP/NCIP/NFCP, which reach Eirene through deck block 14.
    The coupled b2mn aborts in test_couple unless these equal B2's nCv/nFc, so the file
    is built backwards from the grid counts in the traduit header (the line after
    '*cf: ... nCi,nFc ...': nCi nFc nVx nCg ...):

      - n_bnd   = nCg                          (boundary / guard cells)
      - n_degen = 4*nCi + n_bnd - 2*nFc        (degenerate faces; inverts uincrr's
                                                face formula; error if negative)
      - 'nreg=' is always written as 3, with 3 regions: region 1 is sized to hold
        all the cells, regions 2 and 3 are 3x3 placeholders. The topology (SN/DN)
        is not used: only the three counts matter, so the same file is produced
        for any topology.
      - 'cellflags' blocks flag exactly nCi internal cells and nCg + n_degen
        boundary cells; 'degenerate faces' blocks mark n_degen entries.
      - nptseg = (24, 24, 72) and npr = (12, 20, 20) only feed uinp's informational
        block-2 sizes (AUG-representative defaults, not derived from the grid).

Exit status: 0 on success, 1 on missing arguments, unreadable/malformed traduit
header, inconsistent grid counts, or write failure.
"""

import sys
from pathlib import Path
from typing import Tuple


def read_traduit_grid_counts(traduit_path: Path) -> Tuple[int, int, int]:
    """Extract nCi, nFc, nCg from traduit.out.b2us header.
    
    Returns:
        (nci, nfc, ncg) tuple
        
    Raises:
        ValueError: if header line not found or format is unexpected
    """
    with open(traduit_path, 'r') as f:
        tlines = f.readlines()
    
    # Find the header line starting with "*cf:" containing "nCi,nFc"
    try:
        hidx = next(i for i, l in enumerate(tlines)
                    if l.startswith("*cf:") and "nCi,nFc" in l)
    except StopIteration:
        raise ValueError("Could not find traduit header line with '*cf:' and 'nCi,nFc'")
    
    # Extract values from next line
    try:
        values = [int(v) for v in tlines[hidx + 1].split()[:4]]
        nci, nfc, _, ncg = values
    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse traduit header values: {e}")
    
    return nci, nfc, ncg


def write_uinput_crr(n_cells_internal: int,
                     n_cells_total: int,
                     n_faces: int,
                     nptseg: Tuple[int, int, int] = (24, 24, 72),
                     npr: Tuple[int, int, int] = (12, 20, 20)) -> str:
    """Generate pseudo carre.out content.
    
    Synthesizes uinput.crr with cellflags blocks engineered so that
    uinp's uincrr reproduces the EXACT grid counts — nicells = n_cells_internal,
    boundary-face flags = n_cells_total - n_cells_internal + ndfaces,
    and n_faces matches exactly.
    
    Args:
        n_cells_internal: nCi from traduit (internal cells)
        n_cells_total: nCv from traduit (internal + guard cells)
        n_faces: nFc from traduit (total faces)
        nptseg: toroidal segment counts per region (default: AUG-representative)
        npr: radial/poloidal point counts per region (default: AUG-representative)
        
    Returns:
        Formatted uinput.crr content as string
        
    Raises:
        ValueError: if grid counts are inconsistent
    """
    n_int = n_cells_internal
    n_bnd = n_cells_total - n_cells_internal
    
    # Compute degenerate faces: invert uincrr's face formula
    # nFc = (4*nCi + nBnd - 2*nDegen)/2 + nBnd
    n_degen = 4 * n_int + n_bnd - 2 * n_faces
    if n_degen < 0:
        raise ValueError(f"grid counts inconsistent: nFc={n_faces} "
                        f"exceeds the quad-mesh bound for nCi={n_int}, "
                        f"nCg={n_bnd}")
    
    # Compute region dimensions
    nprad = (25, 3, 3)
    cap0 = max(n_int, n_bnd)
    nppol0 = cap0 // (nprad[0] - 1) + 2
    nppol = (nppol0, 3, 3)
    
    # Validate nptseg
    if any(v > 99 for v in nptseg):
        raise ValueError("nptseg must be <= 99 (uincrr reads i3 after a11)")
    
    lines = []
    
    # Write mesh parameters
    for i, v in enumerate(nptseg, start=1):
        lines.append(f"nptseg({i}) = {v:2d}")
    for i, v in enumerate(npr, start=1):
        lines.append(f"npr({i}) ={v:3d}")
    lines.append(f"'nreg='{3:5d}")
    
    # Distribute cells across regions
    remaining_int = n_int
    remaining_bnd = n_bnd + n_degen      # raw flags; uincrr subtracts ndfaces
    remaining_deg = n_degen
    
    for reg in range(3):
        lines.append(f"'region:'{reg+1:5d}")
        lines.append(f"'nppol='{nppol[reg]:5d}  'nprad='{nprad[reg]:5d}")
        lines.append("'cellflags'")
        
        ncell = (nppol[reg] - 1) * (nprad[reg] - 1)
        for _ in range(ncell):
            # Flag 1: internal cell (1 if remaining_int > 0, else 0)
            f1 = 1 if remaining_int > 0 else 0
            remaining_int -= f1
            
            # Flag 2: boundary cell (1 if remaining_bnd > 0, else 0)
            f2 = 1 if remaining_bnd > 0 else 0
            remaining_bnd -= f2
            
            # Flags 3-5: unused
            lines.append(f"{f1:4d}{f2:4d}{0:4d}{0:4d}{0:4d}")
        
        # Degenerate faces
        lines.append("'degenerate faces'")
        for _ in range(ncell):
            row = [0, 0, 0, 0]
            for c in range(4):
                if remaining_deg > 0:
                    row[c] = 1
                    remaining_deg -= 1
            lines.append("".join(f"{v:4d}" for v in row))
    
    if remaining_int > 0 or remaining_bnd > 0 or remaining_deg > 0:
        raise RuntimeError("pseudo carre.out region capacity too small")
    
    lines.append("fin")
    lines.append("fin")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Entry point: parse arguments and generate uinput.crr."""
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    traduit_path = Path(sys.argv[1])
    
    if not traduit_path.is_file():
        print(f"Error: {traduit_path} not found", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = traduit_path.parent / "uinput.crr"
    
    # Extract grid counts
    try:
        nci, nfc, ncg = read_traduit_grid_counts(traduit_path)
    except ValueError as e:
        print(f"Error reading traduit file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Extracted from {traduit_path.name}:")
    print(f"  nCi (internal cells) = {nci}")
    print(f"  nFc (total faces)    = {nfc}")
    print(f"  nCg (guard cells)    = {ncg}")
    print(f"  nCv (total cells)    = {nci + ncg}")
    
    # Generate uinput.crr
    try:
        uinput_crr = write_uinput_crr(n_cells_internal=nci,
                                      n_cells_total=nci + ncg,
                                      n_faces=nfc)
    except ValueError as e:
        print(f"Error generating uinput.crr: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write output
    try:
        output_path.write_text(uinput_crr)
        print(f"\nGenerated: {output_path}")
        print(f"File size: {len(uinput_crr)} bytes, {len(uinput_crr.splitlines())} lines")
    except IOError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
