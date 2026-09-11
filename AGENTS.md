# AGENTS.md - Hypersonic Body CFD

## Quick Start

```bash
uv sync                          # Install deps (Python 3.13, numpy, scipy, gmsh, ezdxf, matplotlib)
uv run pytest tests/ -v          # Run all tests (~565)
uv run pytest tests/test_case_config.py -v  # Run single test file
uv run python run.py --case apollo-cm --mach 15.6 --full2d  # Full2D simulation
uv run python run.py --case apollo-cm --thermal --thermal-time 200  # Thermal analysis
```

## Architecture

**Pipeline flow:** `geometry -> mesh -> SU2 -> postprocess -> validation -> thermal`

**Entry points:**
- `run.py` -- Unified CLI (replaces 6 legacy scripts). Use `--full2d` for entire body, default is axisymmetric half-body.
- `run_aoa_sweep.py` -- Angle of attack parametric study (separate workflow).

**Source modules (`src/`):**
| Module | Purpose | Key file |
|--------|---------|----------|
| `geometry/` | Blunt body contour, presets, DXF loader | `blunt_body.py` (generate_contour) |
| `cfd/` | Mesh gen, SU2 config, solver, postprocess | `mesh.py`, `solver.py`, `config.py` |
| `pipeline/` | Case config, stage orchestration | `case_config.py`, `stages.py` |
| `physics/` | US Standard Atmosphere, real-gas | `atmosphere.py` |
| `validation/` | Sutton-Graves, Billig, Newtonian | `billig.py`, `compare.py` |
| `thermal/` | 1D/2D heat equation, ablation | `solver_2d.py`, `config.py` |
| `viz/` | Contour plots, convergence, geometry | `contour.py`, `mesh.py` |

**Key data flow:**
```
BluntBodyConfig (geometry/presets.py)
  -> generate_contour() -> (x, r) arrays
  -> generate_body_mesh() -> .su2 mesh file
  -> SU2HypersonicConfig.write() -> config.cfg
  -> SU2Solver.run_stages() -> flow.vtu, history.csv
  -> postprocess + validation -> results.json, plots
  -> thermal analysis -> thermal_results.json, temperature plots
```

## Conventions

- **No em dashes** anywhere in any file.
- **C++ conventions** (for any C++ code): 4-space indent, K&R braces, snake_case locals, PascalCase types.
- **Python**: snake_case, type hints, docstrings on public functions.
- **Plots** go to `docs/assets/images/{case}/`. **Artifacts** go to `output/{case}/`.
- **Tests**: 565+ tests in `tests/`. Use `uv run pytest tests/test_FILE.py -v` for single file.

## Geometry -- Critical Details

The Apollo CM preset uses **DXF-verified** dimensions. Key values that are NOT obvious:
- `base_radius=0.219` (not 0.03 -- the cone tapers to ~0.22m, not the full diameter)
- `max_radius=1.924` (fillet-cone junction, not the fillet arc apex at 1.956)
- Fillet uses **internal tangency** (R-Rf) because the heat shield is concave, not external (R+Rf).
- Base fillet center is at `r=0.025` (near axis), not `r=0.450`.

If generating geometry for a different blunt body, check `geometry/apollo_2d.dxf` for the reference shape.

## Mesh

Two modes:
- **Axisymmetric** (default): Half-body, elliptical O-grid farfield. Faster, fewer cells.
- **Full2D** (`--full2d`): Entire body, full ellipse farfield. Use for visualization or AoA studies.

The O-grid mesh has quality issues (~88% bad cells) for full2D. The simpler distance-based mesh (0% bad cells, 0.97 mean quality) is used for production runs. See `generate_body_mesh()` in `src/cfd/mesh.py`.

## SU2 Solver

- **Version**: SU2 v8.4.0 (binary at `~/SU2_CFD/bin/SU2_CFD` or `SU2_CFD_BIN` env var)
- **Config**: `src/cfd/config.py` generates `.cfg` files. `AXISYMMETRIC=YES` for half-body, `NO` for full2D.
- **Convergence**: 4-stage Mach ramp (M=2 -> M=6.5 -> M=11.1 -> M=15.6). See `src/cfd/convergence.py`.
- **Critical SU2 v8.4 quirk**: `CFL_ADAPT_PARAM` requires `cfl_adapt_max >= 1.0`. Using values like 0.05 causes "CFL adaption factor up should be greater than 1.0" error.

## Output Structure

```
output/{case}/
  mesh/               # .su2 mesh files
  su2/{mach}/         # Per-Mach: config.cfg, history.csv, flow.vtu, results.json
  postprocess/        # postprocess.json
  validation/         # validation.json
  thermal/            # thermal_results.json (if --thermal enabled)
```

## Thermal Analysis

The thermal module solves the 2D axisymmetric heat equation through the heat shield wall with temperature-dependent properties and optional charring ablation.

**CLI flags:**
- `--thermal` -- Enable thermal analysis after SU2
- `--material {avcoat,pica}` -- Heat shield material (default: avcoat)
- `--wall-thickness FLOAT` -- Wall thickness in meters (default: 0.05)
- `--thermal-time FLOAT` -- Simulation duration in seconds (default: 100)
- `--ablation` -- Enable charring ablation model
- `--thermal-output DIR` -- Custom thermal output directory

**Example:**
```bash
uv run python run.py --case apollo-cm --thermal --thermal-time 200 --ablation
```

**Key files:**
| File | Purpose |
|------|---------|
| `src/thermal/config.py` | ThermalConfig2D, AblationConfig dataclasses |
| `src/thermal/solver_2d.py` | 2D implicit backward Euler solver |
| `src/thermal/heat_flux.py` | VTU heat flux extraction, synthetic distributions |
| `src/thermal/materials.py` | Temperature-dependent material properties |
| `src/thermal/ablation.py` | Charring ablation model (pyrolysis kinetics) |
| `src/thermal/results.py` | Result dataclasses, JSON serialization |
| `src/viz/thermal.py` | Wall temp, through-wall profiles, contour plots |
| `src/validation/thermal_validation.py` | Validation against Sutton-Graves, physical bounds |

**Thermal results JSON structure:**
```json
{
  "config": { "material": "avcoat", "wall_thickness_m": 0.05 },
  "results": { "T_max_wall_K": 2500, "T_max_back_K": 500, "q_total_J_m2": 1e6 },
  "surface": { "s_m": [...], "q_surface_W_m2": [...] },
  "profiles": { "z_m": [...], "T_initial_K": [...], "T_final_K": [...] }
}
```

**Data flow:**
```
SU2 flow.vtu -> extract_heat_flux_from_vtu() -> q_surface(s)
  -> ThermalConfig2D -> ThermalSolver2D.solve() -> ThermalResult2D
  -> save_thermal_results_2d() -> thermal_results.json
  -> plot_wall_temperature(), plot_through_wall_profiles(), plot_temperature_contour()
```

## Tests

- Run all: `uv run pytest tests/ -v`
- Run single: `uv run pytest tests/test_convergence.py -v`
- Integration tests (marked `@pytest.mark.integration`) require actual SU2 binary.
- Test files mirror source modules: `test_billig.py` tests `src/validation/billig.py`.

## Common Pitfalls

1. **SU2 crashes immediately** -- Check `cfl_adapt_max >= 1.0` in convergence config.
2. **No bow shock in results** -- Solver diverged. Check mesh quality (0% bad cells needed) and CFL values.
3. **Wrong geometry dimensions** -- Verify against DXF in `geometry/apollo_2d.dxf`, not the code defaults.
4. **Axisymmetric but showing full body** -- Use `--full2d` flag and set `domain_type="full2d"` in MeshConfig.
5. **Tests fail after geometry changes** -- Update expected values in `tests/test_presets.py` and `tests/test_blunt_body_config.py`.
