#!/usr/bin/env python3
"""
Convert a translator `.dgo` file (sections `p1`, `p2`, `fclbl`) into a Carre `structure.dat` file.

Usage:
  python scripts/dgo_to_structure.py --in dg.dgo --out structure.dat

Behavior:
 - Parses `p1` and `p2` sections as segment endpoints (X Y [Z]) and `fclbl` as integer labels.
 - Coordinates in the `.dgo` are expected to be in millimetres (mm) and will be converted to metres (m) in the output `structure.dat`.
 - Groups segments by positive `fclbl` (if `fclbl` missing, all segments get label 1).
 - Assembles connected segments into chains/loops using exact or near-equality (tolerance).
 - Emits `structure.dat` compatible with Carre's `fcrstro` expectations:
     First line: number of structures
     Then `$structures`, then `Structure <n>` blocks, each with a signed count
 - Closed loops are written with positive count and first point repeated at end; open chains use negative count.

This is a best-effort converter; complex topologies might need manual inspection.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict, deque
from typing import List, Tuple, Dict

Point = Tuple[float, float]

def parse_dgo(path: str):
    p1: List[Tuple[float,float,float]] = []
    p2: List[Tuple[float,float,float]] = []
    fclbl: List[int] = []

    section = None
    sec_re = re.compile(r"^\s*(p1|p2|fclbl)\b", re.IGNORECASE)

    def to_nums(line: str):
        line = re.split(r"[!#]", line, maxsplit=1)[0].strip()
        if not line:
            return []
        parts = re.split(r"[,,\s]+", line.strip())
        vals = []
        for p in parts:
            if p == '':
                continue
            try:
                vals.append(float(p))
            except ValueError:
                pass
        return vals

    with open(path, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n')
            m = sec_re.match(line)
            if m:
                section = m.group(1).lower()
                continue
            # treat a new alphabetic header at line start as end of a numeric section
            if re.match(r"^\s*[A-Za-z]", line) and not sec_re.match(line):
                section = None
                continue
            if section is None:
                continue
            nums = to_nums(line)
            if not nums:
                continue
            if section == 'p1' or section == 'p2':
                if len(nums) >= 2:
                    # input coordinates are in mm; convert to metres
                    x = float(nums[0]) * 1e-3
                    y = float(nums[1]) * 1e-3
                    z = (float(nums[2]) if len(nums) >=3 else 0.0) * 1e-3
                    if section == 'p1':
                        p1.append((x,y,z))
                    else:
                        p2.append((x,y,z))
            elif section == 'fclbl':
                try:
                    fclbl.append(int(round(nums[0])))
                except Exception:
                    fclbl.append(0)

    return p1,p2,fclbl

def key_pt(pt: Tuple[float,float,float], tol=1e-6):
    # discard Z for Carre structures (R,Z -> we take X as R, Y as Z)
    x,y,_ = pt
    # quantize using tolerance
    return (round(x/tol)*tol, round(y/tol)*tol)

def assemble_loops(segments: List[Tuple[Point,Point]], tol=1e-6):
    # Directed chaining: respect segment orientation p1->p2
    key_of = lambda p: (round(p[0]/tol)*tol, round(p[1]/tol)*tol)
    # map p1_key -> list of segment indices that start at that key
    start_map: Dict[Point, List[int]] = defaultdict(list)
    for i,(a,b) in enumerate(segments):
        start_map[key_of(a)].append(i)

    used = set()
    structures: List[Tuple[List[Point], bool]] = []

    for i in range(len(segments)):
        if i in used:
            continue
        # start a chain from segment i using directed links
        a,b = segments[i]
        chain = [a, b]
        used.add(i)
        # extend forward by finding segments whose p1 == current p2
        cur_key = key_of(chain[-1])
        while True:
            next_indices = start_map.get(cur_key, [])
            found = None
            for j in next_indices:
                if j in used:
                    continue
                found = j
                break
            if found is None:
                break
            # append found segment keeping p1->p2 order
            _, nb = segments[found]
            chain.append(nb)
            used.add(found)
            cur_key = key_of(nb)

        # attempt backward extension: find segment whose p2 equals current chain start
        # we must search all segments whose p2 matches chain[0]
        # build reverse map lazily
        rev_map = defaultdict(list)
        for idx,(sa,sb) in enumerate(segments):
            rev_map[key_of(sb)].append(idx)
        while True:
            prev_indices = rev_map.get(key_of(chain[0]), [])
            found = None
            for j in prev_indices:
                if j in used:
                    continue
                found = j
                break
            if found is None:
                break
            # prepend this segment keeping p1->p2 order: insert its p1 before current start
            pa,_ = segments[found]
            chain.insert(0, pa)
            used.add(found)

        closed = (len(chain) >= 3) and (key_of(chain[0]) == key_of(chain[-1]))
        structures.append((chain, closed))

    return structures

def write_structure(path: str, structures: List[Tuple[int, List[Point], bool]]):
    # structures: list of (label, pts, closed)
    total = len(structures)
    count_width = 10
    label_width = 5
    with open(path, 'w') as f:
        # align total with the per-structure count column
        f.write(f"{total:{count_width}d}\n")
        f.write("$structures\n")
        for label, pts, closed in structures:
            # 'Structure' line with label right-aligned in a small field
            f.write(f"Structure{label:{label_width}d}\n")
            if closed:
                l = len(pts)-1 if (len(pts) >=2 and pts[0]==pts[-1]) else len(pts)
                count = l+1
            else:
                count = -len(pts)
            # write count right-aligned in same width as the total
            f.write(f"{count:{count_width}d}\n")
            # coordinates: two columns, each width 22 with 15 decimals, aligned
            for x,y in pts:
                f.write(f"{x:22.15f}  {y:22.15f}\n")
            if closed:
                f.write(f"{pts[0][0]:22.15f}  {pts[0][1]:22.15f}\n")
        f.write("$end\n")

def convert(infile: str, outfile: str, tol: float=1e-6):
    p1,p2,fclbl = parse_dgo(infile)
    if not p1 or not p2:
        raise SystemExit("No p1/p2 sections found or empty file")
    if len(p1) != len(p2):
        raise SystemExit(f"Counts differ: p1={len(p1)} p2={len(p2)}")
    n = len(p1)
    if not fclbl:
        fclbl = [1]*n
    if len(fclbl) != n:
        raise SystemExit(f"fclbl length mismatch: expected {n}, got {len(fclbl)}")

    # group segments by positive fclbl (ignore non-positive)
    segs_by_label: Dict[int, List[Tuple[Point,Point]]] = defaultdict(list)
    for i in range(n):
        lbl = fclbl[i]
        if lbl <= 0:
            continue
        a = (p1[i][0], p1[i][1])
        b = (p2[i][0], p2[i][1])
        segs_by_label[lbl].append((a,b))

    all_structures: List[Tuple[int, List[Point], bool]] = []
    for lbl in sorted(segs_by_label.keys()):
        segs = segs_by_label[lbl]
        structs = assemble_loops(segs, tol=tol)
        for pts, closed in structs:
            all_structures.append((lbl, pts, closed))

    if not all_structures:
        raise SystemExit("No structures assembled (no positive fclbl found)")

    # determine whether the assembled structures can be concatenated into a closed contour
    def close(a: Point, b: Point, tol: float=tol) -> bool:
        return abs(a[0]-b[0]) <= tol and abs(a[1]-b[1]) <= tol

    # build start/end list
    items = []
    for idx, (lbl, pts, closed) in enumerate(all_structures):
        if not pts:
            continue
        start = pts[0]
        end = pts[-1]
        items.append((idx, lbl, start, end))

    # adjacency: i -> j if end_i == start_j within tol
    adj = {it[0]: [] for it in items}
    for i, _, si, ei in items:
        for j, _, sj, ej in items:
            if i == j:
                continue
            if close(ei, sj):
                adj[i].append(j)

    # try to find a Hamiltonian cycle covering all items
    N = len(items)
    order_idxs = None

    def dfs(path):
        nonlocal order_idxs
        if order_idxs is not None:
            return True
        if len(path) == N:
            # check closure back to start
            last = path[-1]
            first = path[0]
            last_end = next(x[3] for x in items if x[0] == last)
            first_start = next(x[2] for x in items if x[0] == first)
            if close(last_end, first_start):
                order_idxs = list(path)
                return True
            return False
        cur = path[-1]
        for nb in adj.get(cur, []):
            if nb in path:
                continue
            path.append(nb)
            if dfs(path):
                return True
            path.pop()
        return False

    # try each possible start
    for start_item in adj.keys():
        dfs([start_item])
        if order_idxs is not None:
            break

    if order_idxs is None:
        print("Global contour closed: False")
        print("No single closed ordering found among assembled structures.")
    else:
        # map indices back to labels for display
        label_order = [next(x[1] for x in items if x[0] == idx) for idx in order_idxs]
        print("Global contour closed: True")
        print("Closed order (labels):", label_order)

    write_structure(outfile, all_structures)

def main():
    ap = argparse.ArgumentParser(description='Convert dgo -> structure.dat')
    ap.add_argument('--in','-i', dest='infile', default='dg.dgo', help='input dgo file')
    ap.add_argument('--out','-o', dest='outfile', default='structure.dat', help='output structure.dat')
    ap.add_argument('--tol', type=float, default=1e-6, help='point matching tolerance')
    args = ap.parse_args()
    convert(args.infile, args.outfile, tol=args.tol)

if __name__ == '__main__':
    main()
