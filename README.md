# Hypersonic Blunt Body Aerothermodynamics

Phase 1: Analytical and geometric foundation for hypersonic flow simulation over spherically-blunted cones.

This project simulates hypersonic flow (M=5, 8, 12) over spherically-blunted cones, anchored by an Apollo Command Module headline case. Validation is performed against Sutton-Graves stagnation heating, Billig shock standoff, and modified Newtonian pressure distributions. 2D axisymmetric SU2 RANS comes in later phases.

## Quick Start

```bash
uv sync
uv run pytest tests/ -v
uv run python run_geometry.py --case apollo-cm
```

## Project Structure

- `src/geometry/` - Blunt body configuration and contour generation
- `src/validation/` - Analytical correlations (Sutton-Graves, Billig, Newtonian, shock relations)
- `src/physics/` - Standard atmosphere and real-gas properties
- `src/pipeline/` - Case configuration and stage execution
- `src/viz/` - Reentry orange/amber visualization theme
- `tests/` - pytest test suite
