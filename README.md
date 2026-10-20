# scripts.local

Helper scripts for SOLPS-ITER runs on the unstructured ("wide") grid. Every Python script prints
its full usage with `-h` / `--help`; the details below are only an overview.

## Requirements

- Python 3 with `numpy`, `matplotlib` and `netCDF4` (`scipy` is not needed).
- Most plotting scripts open a Tk window (`ssh -X` or a local display). Without a display some of
  them write PNG files instead (see each `--help`); `--save` always writes files.
- Unless stated otherwise, **run the scripts from the run directory**
  (`path/to/your/run/dir/`): they look for `b2mn.exe.dir/`, `b2fgmtry` (or `../baserun/b2fgmtry`),
  `b2tallies.nc`, `run.log`, ... relative to the current directory.
- Scripts that edit or write files (`update_*`, `write_*`, `modify_b2fgmtry_prep.py`) overwrite in
  place and keep no backup (except `modify_b2fgmtry_prep.py --inplace`, which keeps `.orig`): keep
  a copy or use git.

## Preparing a run (grid and input files)

| Script | What it does |
|---|---|
| `dgo_to_structure.py` | Converts a translator `.dgo` file (`p1`, `p2`, `fclbl`) to a Carre `structure.dat`. Open segments only for now. |
| `traduit_to_uinput_crr.py` | Generates a pseudo `uinput.crr` from `traduit.out.b2us` (SN topology only) for coupled EIRENE cases. |
| `update_b2ag_dimens.py` | (Re)writes `b2ag.dat` as the unstructured template with nCi, nFc, nVx from `traduit.out.b2us`. |
| `modify_b2fgmtry_prep.py` | Applies the wall-face part of the `b2us` geometry prep (Qalfmin / Qalfmax / Pbs_reduce_fac) to an unstructured `b2fgmtry`; optional R/Z boxes. |
| `write_b2numerics_parameters.py` | Writes `b2.numerics.parameters_F` and `_T` for a given number of regions and species. |
| `write_b2transport_parameters.py` | Writes `b2.transport.parameters` for a given number of species. With `--pinch V` it switches on the anomalous pinch with an inward speed V, choosing the sign of `parm_vla` from the sign of `fcQalf` in the mesh. |
| `update_b2boundary_parameters.py` | Targeted edits of `b2.boundary.parameters` (core boundary conditions, sheath transmission `--SHEE`, `--SHII`). |
| `update_b2neutrals_parameters.py` | Sets `eirene_mod` and the gas-puff flux in `b2.neutrals.parameters`. |
| `update_spt_input.py` | Adds the temperature-dependent chemical-sputtering / RECYCT blocks to the reflection-model section of the EIRENE `input*.eir` files. |
| `generate_initialState_us.py` | Builds an unstructured `b2fstate` by mapping fields (na, ne, te, ti, po) from another solution (structured or unstructured) by nearest neighbour, with optional smoothing along flux tubes. Uses `b2state_io.py`. |

## Monitoring a run

| Script | What it does |
|---|---|
| `monitor_run_log.sh` | Health of the potential solve in `run.log`, in bins of steps (stalled solves, clamp messages). |
| `b2ftrace_residuals.py` | Residual norms from `b2ftrace` (table, overlay plot, per-equation plot) and progress / timing from `run.log` (`--log`). Works on a running job. |
| `watch_omp_profiles.py` | Live radial profiles (OMP, IMP, targets) read from `b2time.nc` of a running job. |
| `plot_omp_profiles.py` | Compares profiles of several finished runs from their `2d_profiles/*.last10` files. |
| `2dt.py` | Plots time series from `b2time.nc`. |
| `track_runaway.py` | Follows a Te / Ti / po runaway through the `plasmastate.XXXX` snapshots. |
| `plasmastate_extrema.py` | Max / min of plasmastate quantities versus time, with location. |

## Looking at the solution

| Script | What it does |
|---|---|
| `plot_b2fstate.py` | 2D maps of the final state (`b2fstate`) of one or several runs, structured (3.0.9) and unstructured (3.2.1) side by side, also from a single state file (`-g` for its geometry). Any cell scalar (`-q ne te ua ...`, species with `-s`, default 1 = D+) plus a face-based vector (`-v fna fhi vaecrb ...`) as streamlines, direction arrows, the face values themselves or the net outflow per cell (`--mode`). `--line R0 Z0 R1 Z1` marks a straight line on the maps and adds a figure with the scalars along it (runs overlaid). `--wall` draws the wall from the run's `structure.dat` (Carre format; looked for in the run, its `baserun/` and the family directory). `--list` shows the blocks of a state. Replaces `plot_b2fstate_us.py`. |
| `plot_plasmastate.py` | The same for the `plasmastate.XXXX` snapshots of a *running* job (monitoring / debugging): cell fields on the 2D mesh with max / min markers, `--limits` shows the cells sitting on a limit enforced by B2.5, `--faces` / `--div` draw face blocks (fhi, fhe, fna:IDX, ...) as face segments or net outflow per cell, `-v NAME:IDX` adds streamlines, `--wall` the wall. Replaces `plot_plasmastate_flux.py`. |
| `plot_qalf_wall.py` | Classifies the wall faces (grazing / band / normal) from Qalf and Pbs, compares with an original geometry (`--ref`) and shows the boundary conditions that result (`--show bc`). |

## Balances (read `b2tallies.nc`)

| Script | What it does |
|---|---|
| `particle_balance_widegrid.py` | Particle balance per element for the wide grid (outward boundary faces), with and without dN/dt. |
| `particle_balance_fluxes.py` | The same balance broken down by boundary face region. |
| `internal_energy_balance_widegrid.py` | Internal energy balance for the wide grid (python version of `scripts/internal_energy_balance_coupled`). |
| `particle_balance.py`, `particle_balance_core.py` | Copies of the stock structured-grid scripts. **Not valid on the wide grid.** |

## Libraries and notes

- `solpsplt/`: the shared code of the plotting tools (readers for `b2fstate` / `plasmastate` / `b2fgmtry`, structured and
  unstructured meshes, face fluxes -> cell vectors, streamlines, display handling). `scripts.local` is on `PYTHONPATH`, so
  `import solpsplt` works anywhere:

  ```python
  import solpsplt as sp
  st = sp.State.from_run('path/to/run')        # b2fstate; plasmastate: source='plasmastate', snap='0015'
  te = st.scalar('te'); na = st.scalar('na', 4)           # interior cells, Te in eV; species 4 = C2+
  vec = st.faces('fna', 1).cell_vectors()                 # D+ particle flux density vector per cell, (R, Z)
  st.mesh.polys, st.mesh.centres                          # cell polygons / centres, same order as the values
  ```
  Structured runs need the topology (`rightix`, `topix`, ...) of the `b2fgmtry`; they are read from it.
- `b2state_io.py`: imported by `generate_initialState_us.py` and `write_b2transport_parameters.py`, not run directly.
- `sketch_qalf_pbs.py`, `sketch_qalf_regimes.py`: schematic figures of Qalf / Pbs and the regimes
  (no input needed). Background in `qalf_pbs_notes.pdf` (source: `.tex`).
- `doc/goat_inputs_procedure.tex`: procedure for generating the GOAT inputs and checking the grid.
