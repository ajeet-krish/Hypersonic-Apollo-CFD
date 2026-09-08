# Research: 3D Hypersonic CFD Simulation Feasibility - Apollo CM

**Date:** September 2026
**Domain:** CFD, Gmsh mesh generation, SU2 solver, aerospace engineering
**Objective:** Assess feasibility of transitioning from 2D axisymmetric to full 3D simulation using STEP geometry

---

## Executive Summary

Building a 3D hypersonic CFD simulation for the Apollo CM using a STEP file is **technically feasible** but involves significant complexity and computational cost. The project already has the key ingredient: `geometry/apollo_3d.step` (a valid STEP file generated from Autodesk). The main challenges are:

1. **3D mesh generation** from STEP using Gmsh OpenCASCADE kernel (proven approach)
2. **Computational cost** increase of 50-100x over 2D axisymmetric
3. **Convergence difficulties** at Mach 15.6 in 3D (requires careful strategy)
4. **Questionable physics benefit** for an axisymmetric body at zero angle of attack

**Recommendation:** Proceed with 3D only for angle-of-attack studies or non-axisymmetric features. For zero-AoA, the 2D axisymmetric simulation captures all relevant physics at 1/100th the cost.

---

## 1. 3D Mesh Generation from STEP File

### 1.1 Gmsh STEP Import (OpenCASCADE Kernel)

**Key Finding:** Gmsh natively supports STEP import via the OpenCASCADE geometry kernel. This is the recommended approach for 3D CFD mesh generation.

**Python API approach:**
```python
import gmsh
gmsh.initialize()
gmsh.model.add("apollo_3d")

# Load STEP file using OpenCASCADE
v = gmsh.model.occ.importShapes("geometry/apollo_3d.step")
gmsh.model.occ.synchronize()

# Generate 3D volume mesh
gmsh.model.mesh.generate(3)
gmsh.write("mesh.su2")
gmsh.finalize()
```

**Source:** Gmsh documentation (t20.py tutorial), Gmsh 4.15.2 manual, piclas DSMC cone tutorial

| Feature | Status | Notes |
|---------|--------|-------|
| STEP import | Supported | Via OpenCASCADE kernel (`occ.importShapes`) |
| Boolean operations | Supported | Fragment, cut, fuse for domain creation |
| Physical groups | Supported | Map to SU2 boundary markers |
| Boundary layers | Supported | Via `gmsh.model.mesh.field.add("BoundaryLayer")` |
| Size fields | Supported | Ball, Box, Distance, MathEval fields |
| Export to SU2 | Supported | `gmsh -format su2` preserves physical group names |

### 1.2 3D Mesh Generation Strategy for Hypersonic Blunt Body

**Two main approaches:**

#### Approach A: boolean subtraction (Recommended)
1. Import STEP body as a solid
2. Create outer domain (sphere, cylinder, or box) using OpenCASCADE
3. Subtract body from domain (`gmsh.model.occ.fragments` or `gmsh.model.occ.cut`)
4. Apply boundary layer inflation on body surfaces
5. Apply size fields for shock/wake refinement
6. Generate tetrahedral volume mesh

```python
# Pseudocode for boolean approach
body = gmsh.model.occ.importShapes("apollo_3d.step")
# Create farfield sphere
farfield = gmsh.model.occ.addSphere(cx, cy, cz, radius)
# Subtract body from farfield
fluid_domain, _ = gmsh.model.occ.cut([(3, farfield)], [(3, body[0][1])])
gmsh.model.occ.synchronize()
# Add boundary layers on body surfaces
# Generate mesh
gmsh.model.mesh.generate(3)
```

**Source:** piclas DSMC cone 3D tutorial, AeroRare Studio documentation

#### Approach B: Extrusion from 2D
1. Generate 2D axisymmetric mesh (existing capability)
2. Revolve/extrude around axis using Gmsh
3. Add azimuthal refinement for 3D effects

**Advantage:** Leverages existing 2D mesh expertise
**Disadvantage:** Less flexible for complex geometry, may have quality issues at axis

### 1.3 Cell Count Requirements for 3D

| Mesh Tier | 2D Cells | Estimated 3D Cells | Notes |
|-----------|----------|-------------------|-------|
| Draft | 42,401 | 500K - 1M | Minimum for qualitative results |
| Standard | 150K | 2M - 5M | Production quality |
| High | 400K | 10M - 20M | Research/publishable quality |

**Key insight:** 3D cell count is approximately (2D cells) * (circumferential points) * (radial factor). For a 360-degree model with 60 azimuthal points: 42,401 * 60 * (some factor for volume meshing) = ~2-5M cells minimum.

**Reference:** NASA TFAWS study used 1.04M elements for 2D axisymmetric; 3D hypersonic studies typically use 3-20M cells.

### 1.4 Mesh Quality Metrics for 3D

| Metric | Target | Acceptable | Critical for |
|--------|--------|------------|--------------|
| Orthogonal quality | > 0.85 | > 0.7 | All CFD |
| Skewness | < 0.2 | < 0.4 | Shock capture |
| Aspect ratio | < 20 (BL) | < 50 | Boundary layer |
| First cell height | y+ < 1 | y+ < 3 | RANS heat flux |
| Growth ratio | < 1.2 | < 1.3 | Boundary layer |
| Minimum angle | > 18 | > 12 | Tet quality |

**Reference:** GridPro reentry vehicle meshing guide, NASA mesh adaptation studies

---

## 2. SU2 v8.4.0 3D Compatibility

### 2.1 3D RANS Support

**Key Finding:** SU2 v8.4.0 fully supports 3D RANS simulations. The same solver handles 2D and 3D; the mesh dimension determines the physics.

| Feature | 2D | 3D | Notes |
|---------|----|----|-------|
| RANS (SA model) | Yes | Yes | Same configuration |
| Euler/Navier-Stokes | Yes | Yes | Same configuration |
| AXISYMMETRIC option | Yes | No | 3D doesn't use this flag |
| MARKER_SYM | Yes | Yes | For symmetry planes |
| MARKER_FAR | Yes | Yes | Farfield BC |
| MARKER_ISOTHERMAL | Yes | Yes | Wall BC |
| MUSCL reconstruction | Yes | Yes | Second-order accuracy |
| CFL adaptation | Yes | Yes | Same mechanism |
| Restart capability | Yes | Yes | Same format |

**Source:** SU2 config_template.cfg, SU2 GitHub issues #1373, #2438

### 2.2 3D Boundary Conditions

For a full 3D Apollo CM simulation:

| Boundary | SU2 Marker | Notes |
|----------|------------|-------|
| Body wall | `MARKER_ISOTHERMAL` | No-slip, T=2500K |
| Farfield (inflow/outflow) | `MARKER_FAR` | Characteristic-based BC |
| Symmetry plane (optional) | `MARKER_SYM` | For half-body or quarter-body |
| Wake cut (if applicable) | `MARKER_FAR` | Treat as farfield |

**Critical:** For full 3D (360 degrees), use `MARKER_FAR` for all outer boundaries. SU2 automatically determines inflow vs outflow based on local flow direction.

**For angle-of-attack studies:** Two symmetry planes can be used:
- XY plane (if body is symmetric about XY)
- XZ plane (if body is symmetric about XZ)
This reduces cell count by 4x.

### 2.3 3D Convergence Challenges

**Key Finding:** 3D hypersonic convergence is significantly harder than 2D.

| Challenge | 2D | 3D | Mitigation |
|-----------|----|----|------------|
| CFL stability | Moderate | Difficult | Lower initial CFL (0.0005-0.001) |
| Iterations to converge | 5K-15K | 20K-50K | Multi-stage Mach ramping |
| Divergence risk | Low | Medium-High | CFL adaptation, restart capability |
| Shock stability | Good | Challenging | ROE flux, first-order initial |
| Turbulence init | Important | Critical | SA viscosity ratio > 10 |

**Source:** SU2-NEMO Apollo CM simulations (Maier et al. 2021), Babaji & Sahin 2021

**Recommended convergence strategy for 3D M=15.6:**
1. Stage 1: M=2.0, CFL=0.0005, 10K iters, first-order
2. Stage 2: M=5.0, CFL=0.001, 10K iters, first-order
3. Stage 3: M=10.0, CFL=0.002, 15K iters, first-order
4. Stage 4: M=15.6, CFL=0.003, 20K iters, first-order
5. Stage 5: M=15.6, CFL=0.005, 20K iters, second-order (MUSCL)

### 2.4 Known SU2 v8.4 Issues

1. **Axisymmetric source term bugs** (issues #1373, #2438): Not relevant for full 3D, but be aware if using symmetry planes
2. **CFL_ADAPT_PARAM**: Must have `cfl_adapt_max >= 1.0` (already handled in codebase)
3. **Heat flux at symmetry axis**: Can have numerical artifacts in axisymmetric mode; full 3D avoids this

---

## 3. Computational Requirements

### 3.1 Cell Count vs Memory vs Time

| Mesh Size | Cells | RAM (est.) | Time/iter (est.) | Total time (est.) |
|-----------|-------|------------|------------------|-------------------|
| Coarse | 500K | 4-8 GB | 0.5-1 sec | 3-8 hours |
| Medium | 2M | 16-32 GB | 2-5 sec | 12-48 hours |
| Fine | 5M | 32-64 GB | 5-15 sec | 2-7 days |
| Very fine | 20M | 128-256 GB | 20-60 sec | 1-4 weeks |

**Notes:**
- Time estimates assume single-core performance (SU2 serial mode)
- MPI parallelization can reduce time by 4-10x on multi-core
- Memory scales linearly with cell count
- Hypersonic convergence requires more iterations than subsonic

### 3.2 Hardware Requirements

| Configuration | Coarse (500K) | Medium (2M) | Fine (5M) |
|---------------|---------------|-------------|-----------|
| CPU cores | 4-8 | 8-16 | 16-32 |
| RAM | 8 GB | 32 GB | 64 GB |
| Storage | 10 GB | 50 GB | 200 GB |
| Laptop feasible? | Yes (slow) | Marginal | No |
| HPC needed? | No | Recommended | Yes |

**Reference:** 
- Pampero HiFi re-entry study: 200 CPU cores, 144,000 CPU-hours for high-fidelity structured mesh
- DSMC Apollo study: 150 hours on 48-core server for 4M cells
- hyStrath Mars reentry: up to 4,608 cores for 20M cells

### 3.3 Cost-Benefit Analysis: 2D vs 3D

| Criterion | 2D Axisymmetric | Full 3D (360) | 3D Quarter (90) |
|-----------|-----------------|---------------|-----------------|
| Cell count | 42K | 2-5M | 500K-1.25M |
| Compute time | 30 min | 12-48 hours | 3-12 hours |
| RAM required | 1 GB | 32-64 GB | 8-16 GB |
| Laptop run? | Yes | No | Yes (slow) |
| Physics captured | All (at AoA=0) | All | All |
| AoA capability | No (full2d only) | Yes | Yes |
| Non-axisymmetric | No | Yes | Yes |
| Wake capture | Good | Excellent | Good |

---

## 4. 3D Mesh Strategies

### 4.1 Structured Hex vs Unstructured Tet

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| Structured hex | Better shock alignment, lower diffusion | Complex topology, hard to automate | For expert users |
| Unstructured tet | Easy to generate, flexible | More cells needed, higher diffusion | **Recommended** |
| Hybrid (tet + prism BL) | Best of both | More complex setup | Ideal but complex |

**For initial 3D development:** Use unstructured tetrahedral with boundary layer inflation (prism layers near wall). This is the most straightforward approach with Gmsh.

### 4.2 O-grid vs C-grid for 3D

| Topology | 2D Performance | 3D Complexity | Notes |
|----------|----------------|---------------|-------|
| O-grid | Good | Moderate | Body-conforming, good for capsules |
| C-grid | Better wake | High | Better wake capture, complex 3D topology |
| Spherical farfield | Good | Simple | **Recommended for 3D** |

**For 3D:** Use a spherical or cylindrical farfield boundary. This is natural for blunt body reentry and simplifies the mesh topology significantly.

### 4.3 Boundary Layer Meshing for 3D

**Gmsh BoundaryLayer field for 3D:**
```python
gmsh.model.mesh.field.add("BoundaryLayer", bl_tag)
gmsh.model.mesh.field.setNumber(bl_tag, "hfar", 0.5)  # Far-field size
gmsh.model.mesh.field.setNumber(bl_tag, "hwall_n", 1e-6 * R_nose)  # First cell height
gmsh.model.mesh.field.setNumber(bl_tag, "hwall_t", 0.01)  # Tangential size
gmsh.model.mesh.field.setNumber(bl_tag, "ratio", 1.1)  # Growth ratio
gmsh.model.mesh.field.setNumber(bl_tag, "thickness", 0.05)  # BL thickness
```

**Key parameters for hypersonic BL:**
- First cell height: y+ < 1, typically 1e-6 * R_nose
- Number of BL layers: 30-60
- Growth ratio: 1.1-1.2
- BL thickness: 5-10% of body length

### 4.4 Wake Capture in 3D

**3D wake structures that 2D cannot capture:**
- Helical vortex shedding
- Asymmetric wake instabilities
- 3D turbulence structures
- Cross-flow effects at angle of attack

**Mesh refinement strategy:**
1. Refined region immediately behind body (0-5 D downstream)
2. Gradual coarsening in mid-wake (5-15 D)
3. Coarse far-wake (15-30 D)
4. Use Box fields for cylindrical refinement zones

---

## 5. 3D vs 2D Comparison

### 5.1 What Additional Physics Does 3D Capture?

| Physics | 2D Axisymmetric | Full 3D | Importance |
|---------|-----------------|---------|------------|
| Bow shock | Captured | Captured | Same |
| Boundary layer | Captured | Captured | Same |
| Stagnation point | Captured | Captured | Same |
| Heat shield heating | Captured | Captured | Same |
| Conical afterbody | Captured | Captured | Same |
| Wake recirculation | Captured (2D) | Captured (3D) | Different topology |
| AoA effects | Limited (full2d) | Full | Critical for AoA |
| Non-axisymmetric | Not possible | Full | Only if geometry requires |
| 3D turbulence | Not captured | Captured | Important for Re > 10^6 |
| Helical instabilities | Not captured | Captured | May matter for wake |

### 5.2 Is 3D Necessary for Apollo CM?

**For zero angle of attack (AoA = 0):**
- **No.** The Apollo CM is an axisymmetric body. At AoA = 0, the flow is perfectly axisymmetric.
- 2D axisymmetric simulation captures ALL relevant physics.
- 3D would give identical results at 100x the cost.

**For angle of attack (AoA > 0):**
- **Yes, if full 3D effects are needed.** The 2D "full2d" mode treats the body as a 2D cross-section, which is not correct for AoA.
- 3D captures the asymmetric shock structure, cross-flow, and 3D separation.
- **However:** For moderate AoA (< 15 deg), 2D "full2d" with freestream rotation is a reasonable approximation for initial studies.

**For non-axisymmetric features:**
- If the real Apollo CM has non-axisymmetric features (e.g., thruster ports, asymmetric TPS), then 3D is required.
- The STEP file appears to be a clean axisymmetric body of revolution.

### 5.3 Advantages and Disadvantages

**Advantages of 3D:**
1. Correct physics for AoA studies
2. Captures 3D turbulence and wake instabilities
3. Required for non-axisymmetric geometries
4. Better validation against 3D experimental data
5. Can use symmetry planes to reduce cost

**Disadvantages of 3D:**
1. 50-100x more cells than 2D
2. 50-100x longer compute time
3. Requires HPC or large workstation
4. More complex mesh generation
5. Harder to converge at hypersonic speeds
6. More difficult to debug and validate
7. Diminishing returns for axisymmetric body at AoA=0

---

## 6. Implementation Architecture

### 6.1 Codebase Modifications Required

**New modules needed:**

| Module | Purpose | Complexity |
|--------|---------|------------|
| `src/cfd/mesh_3d.py` | 3D mesh generation from STEP | High |
| `src/cfd/config_3d.py` | 3D SU2 configuration | Low (mostly same) |
| `src/cfd/solver_3d.py` | 3D solver interface | Low (same interface) |
| `src/viz/contour_3d.py` | 3D VTU visualization | Medium |
| `src/geometry/step_loader.py` | STEP file loading/query | Low |

**Modified modules:**

| Module | Changes Required |
|--------|------------------|
| `src/cfd/mesh_config.py` | Add 3D domain types, cell count estimates |
| `src/cfd/config.py` | Add 3D-specific options, symmetry plane handling |
| `src/pipeline/stages.py` | Add 3D mesh generation stage |
| `run.py` | Add `--3d` flag, 3D workflow |
| `src/viz/mesh.py` | 3D mesh visualization |

### 6.2 3D Mesh Generation Module

**Core function signature:**
```python
def generate_3d_mesh(
    step_file: Path,
    output_path: Path,
    mesh_config: Mesh3DConfig,
    mach: float,
) -> Path:
    """Generate 3D SU2 mesh from STEP file.
    
    Args:
        step_file: Path to STEP geometry file
        output_path: Output .su2 mesh file path
        mesh_config: 3D mesh configuration
        mach: Freestream Mach number for shock refinement
    
    Returns:
        Path to generated mesh file
    """
```

**Mesh3DConfig dataclass:**
```python
@dataclass
class Mesh3DConfig:
    """Configuration for 3D mesh generation."""
    domain_type: str = "spherical"  # spherical, cylindrical, box
    farfield_radius: float = 30.0  # in body lengths
    n_boundary_layers: int = 40
    first_cell_height: float = 1e-6  # m, or None for auto
    bl_growth_ratio: float = 1.15
    max_cell_size: float = 2.0  # m
    min_cell_size: float = 0.05  # m
    shock_refinement: bool = True
    wake_refinement: bool = True
    use_symmetry: bool = False  # Use symmetry planes
    symmetry_planes: int = 0  # 0=full, 1=half, 2=quarter
```

### 6.3 Pipeline Integration

**Current pipeline:**
```
geometry -> mesh -> SU2 -> postprocess -> validation
```

**3D pipeline:**
```
STEP file -> geometry_3d -> mesh_3d -> SU2 -> postprocess_3d -> validation_3d
```

**Key integration points:**
1. `run.py`: Add `--3d` and `--step-file` arguments
2. `pipeline/stages.py`: Add `mesh_3d` stage
3. `cfd/config.py`: Add `as_3d()` method to SU2HypersonicConfig
4. `cfd/solver.py`: No changes needed (same interface)
5. `viz/`: Add 3D contour and mesh visualization

### 6.4 Testing and Validation Approach

**Phase 1: Unit tests**
- Test STEP file loading with OpenCASCADE
- Test boolean operations (body subtraction from farfield)
- Test boundary layer generation
- Test mesh quality metrics for 3D

**Phase 2: Integration tests**
- Generate 3D mesh from Apollo CM STEP file
- Verify mesh quality meets targets
- Export to SU2 format and verify markers

**Phase 3: Validation**
- Compare 2D axisymmetric results with 3D at AoA=0
- Should match to within 1-2% for forces, 5% for heat flux
- Validate against experimental data (Bertin 1966, Apollo wind tunnel)

**Phase 4: Extension**
- AoA sweep studies
- Non-axisymmetric features
- Time-accurate simulations

---

## 7. Reference Papers and Data

### 7.1 Apollo CM CFD Studies

1. **Babaji & Sahin (2021)** - "Aerodynamic Analysis of Flow Around Apollo Reentry Capsule Using SU2"
   - Used SU2 with pyAMG mesh adaptation
   - Mach 2.98-10.18, Re up to 2.45e7
   - Good agreement with Apollo wind tunnel data (Bertin 1966)
   - DOI: AIAC-2021-117

2. **Maier et al. (2021)** - "SU2-NEMO: An Open-Source Framework for High-Mach Nonequilibrium Flows"
   - Simulated Apollo CM at 45 deg AoA using SU2-NEMO
   - Demonstrated 3D RANS capability for reentry vehicles
   - DOI: 10.3390/aerospace8070193

3. **Sasanapuri et al. (2013)** - "Numerical Simulation of Massively Separated Flow over Apollo Command Module"
   - Used ANSYS Fluent for 3D separated flow
   - SST k-omega turbulence model
   - Captured unsteady vortex shedding
   - DOI: 10.2514/6.2013-643

### 7.2 Mesh Generation References

4. **piclas DSMC Cone Tutorial** - 3D mesh generation with Gmsh for hypersonic cone
   - Demonstrated STEP import via OpenCASCADE
   - Used `Mesh.Algorithm3D = 7` and `Mesh.SubdivisionAlgorithm = 2`
   - URL: piclas.readthedocs.io

5. **GridPro (2020)** - "Know Your Mesh for Reentry Vehicles"
   - Recommended 60-90 points in normal direction for heat flux
   - Cell Reynolds number of 2-3 for first cell spacing
   - 160 points axial/circumferential for grid convergence

6. **NASA TFAWS (2023)** - Grid dependence study for reentry heating
   - 250K, 1.04M, 4.42M element meshes
   - Medium mesh (1.04M) sufficient for engineering accuracy
   - Cell Reynolds number of 8 for first cell height

### 7.3 Gmsh STEP Import References

7. **Gmsh Tutorial t20** - STEP import and manipulation
   - Demonstrated OpenCASCADE kernel for STEP import
   - Boolean operations for geometry partitioning
   - URL: gitlab.onelab.info/gmsh/gmsh/blob/master/tutorials/python/t20.py

8. **FEATool Tutorial** - CAD File Import and Mesh Generation
   - Step-by-step GUIDE for STEP import in Gmsh
   - URL: featool.com/tutorial/2017/11/06/Gmsh-CAD-STEP-File-Import-and-Mesh-Generation-Tutorial/

### 7.4 Computational Cost References

9. **Pampero HiFi (2025)** - Spacecraft re-entry CFD
   - 200 CPU cores, 144,000 CPU-hours for high-fidelity structured mesh
   - 2 months engineering work for mesh setup
   - 1 month computation time

10. **hyStrath Mars Reentry (2024)** - DSMC/CFD hybrid
    - Up to 4,608 cores for 20M cells
    - Adaptive mesh refinement essential for efficiency

---

## 8. Verification Checklist

- [x] STEP file exists at `geometry/apollo_3d.step` (254 lines, valid ISO-10303-21)
- [x] Gmsh supports STEP import via OpenCASCADE (documented, tested)
- [x] SU2 v8.4.0 supports 3D RANS (confirmed in documentation)
- [x] Existing 2D code can be extended for 3D (modular architecture)
- [x] Computational requirements identified (500K-5M cells, 8-64 GB RAM)
- [x] Convergence strategy defined (5-stage Mach ramp, conservative CFL)
- [x] Validation approach defined (compare with 2D at AoA=0, then extend)
- [x] Risk factors identified (convergence difficulty, computational cost)
- [x] Cost-benefit analysis complete (3D only needed for AoA or non-axisymmetric)

---

## 9. Final Recommendation

### Proceed with 3D if:
1. **Angle-of-attack studies are required** (AoA > 0)
2. **Non-axisymmetric features exist** in the real geometry
3. **HPC resources are available** (16+ cores, 32+ GB RAM)
4. **Research/publication quality** results are needed
5. **Validation against 3D experimental data** is planned

### Stay with 2D axisymmetric if:
1. **Zero angle of attack** is the primary use case
2. **Rapid turnaround** is needed (30 min vs 12-48 hours)
3. **Laptop computation** is preferred
4. **Parametric studies** require many runs
5. **Physics are adequately captured** by axisymmetric assumption

### Hybrid approach (Recommended):
1. **Keep 2D axisymmetric as primary** for production runs
2. **Add 3D capability** for AoA studies and validation
3. **Use symmetry planes** (quarter-body) to reduce 3D cost
4. **Validate 3D against 2D** at AoA=0 before extending to AoA sweep

---

*Report prepared by Literature Research Specialist*
*Last updated: September 2026*
