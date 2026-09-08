# AGENTS.md - Hypersonic Body CFD

## Quick Start

```bash
uv sync                          # Install deps (Python 3.13, numpy, scipy, gmsh, ezdxf, matplotlib)
uv run pytest tests/ -v          # Run all tests (~565)
uv run pytest tests/test_case_config.py -v  # Run single test file
uv run python run.py --case apollo-cm --mach 15.6 --full2d  # Full2D simulation
```

## Architecture

**Pipeline flow:** `geometry -> mesh -> SU2 -> postprocess -> validation`

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
| `viz/` | Contour plots, convergence, geometry | `contour.py`, `mesh.py` |

**Key data flow:**
```
BluntBodyConfig (geometry/presets.py)
  -> generate_contour() -> (x, r) arrays
  -> generate_body_mesh() -> .su2 mesh file
  -> SU2HypersonicConfig.write() -> config.cfg
  -> SU2Solver.run_stages() -> flow.vtu, history.csv
  -> postprocess + validation -> results.json, plots
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
