# Research: C-Grid Mesh Generation for Hypersonic Blunt Body CFD

## Executive Summary

This report investigates the transition from an O-grid to C-grid topology for hypersonic blunt body CFD simulations using Gmsh. The research covers theoretical foundations, practical implementation considerations, and SU2 boundary condition requirements. Key findings indicate that C-grids offer superior wake resolution and shock-capturing capabilities for blunt body flows, though they require more complex geometry construction.

---

## 1. C-Grid vs O-Grid Topology

### 1.1 Fundamental Differences

**O-Grid Topology:**
- Grid lines form closed loops completely encircling the body
- Body is described by one family of grid lines (η = const.)
- Radial lines (ξ = const.) extend from body to farfield
- Creates a "closed" topology around the body

**C-Grid Topology:**
- Grid lines form C-shaped loops beginning and ending in the far wake
- Grid wraps around the body surface and extends downstream to capture the wake
- Creates an "open" topology with a coordinate cut (wake line)
- Body is enclosed by one family of grid lines, with the wake region naturally included

### 1.2 Key Differences Summary

| Feature | O-Grid | C-Grid |
|---------|--------|--------|
| **Topology** | Closed loops around body | Open loops with wake cut |
| **Body Description** | Single η = const. line | η = const. lines wrap around |
| **Wake Resolution** | Poor - grid lines expand rapidly | Excellent - natural clustering |
| **Leading Edge** | Good orthogonality | Better orthogonality |
| **Trailing Edge** | Poor resolution for sharp edges | Excellent resolution |
| **Grid Orthogonality** | Good near body | Better overall |
| **Cell Count** | Lower for same body resolution | Higher (additional wake cells) |
| **Boundary Conditions** | Simpler - no wake cut | Complex - wake cut requires special treatment |

---

## 2. Why C-Grid is Superior for Hypersonic Blunt Body Flows

### 2.1 Bow Shock Capture

**Advantages of C-Grid for Shock Capture:**
1. **Grid Alignment**: C-grid lines naturally align with the shock wave direction in the forebody region
2. **Shock Clustering**: Grid points can be concentrated along the expected shock path
3. **Smooth Transition**: Better cell quality through the shock layer from body to farfield
4. **Reduced Numerical Diffusion**: Aligned grids reduce smearing of the shock discontinuity

**Research Evidence:**
- Lutton (1989) demonstrated that C-grids eliminate total pressure loss spikes at trailing edges compared to O-grids
- Arnone et al. (1992) showed C-grids provide better shock resolution in transonic flows
- GridPro (2024) demonstrated that shock-aligned C-grids significantly improve shock capture accuracy for hypersonic capsules

### 2.2 Wake Formation Resolution

**Critical for Blunt Body Flows:**
1. **Recirculation Zone**: C-grids naturally resolve the base recirculation region
2. **Shear Layer**: Grid clustering along the wake shear layer improves accuracy
3. **Wake Throat**: Better resolution of the wake neck (minimum pressure region)
4. **Base Pressure**: More accurate prediction of base drag (major component for blunt bodies)

**Research Evidence:**
- NASA studies (2024) showed that mesh topology significantly affects wake heating predictions
- Ferfouri et al. (2025) found C-grids better capture base drag for artillery projectiles
- Lutton (1989) identified poor wake resolution as a major weakness of O-grids

### 2.3 Boundary Layer Resolution

**C-Grid Advantages:**
1. **Orthogonality**: Better orthogonality at the body surface reduces numerical errors
2. **Stretching Control**: More natural grid stretching from wall to shock
3. **Y+ Compliance**: Easier to achieve y+ < 1 requirements for RANS
4. **Transition Prediction**: Better resolution of laminar-turbulent transition

---

## 3. C-Grid Geometric Structure

### 3.1 Domain Boundaries

A typical C-grid for a blunt body consists of:

```
                    Upper Farfield (inflow/outflow)
                            ↑
                            |
    Wake Cut (coordinate cut) → → → → → → → → → → → → → → →
                            ↑                                    ↓
                            |                                    |
    Body Surface ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
                            |                                    |
                            |                                    ↓
                    Lower Farfield (inflow/outflow)
```

### 3.2 Detailed Boundary Specification

**For Hypersonic Blunt Body (Apollo CM):**

1. **Inflow Boundaries (Upper/Lower Farfield)**:
   - Supersonic inflow conditions
   - Characteristic-based boundary conditions
   - Flow variables specified from freestream

2. **Outflow Boundary (Downstream Farfield)**:
   - Supersonic outflow - extrapolation from interior
   - No information propagates upstream in supersonic flow

3. **Body Surface**:
   - No-slip, isothermal wall condition
   - Wall temperature specified (e.g., 2500 K for AVCOAT)
   - Heat flux or temperature boundary condition

4. **Wake Cut (Coordinate Cut)**:
   - Periodic or interpolation boundary condition
   - Upper and lower surfaces must be treated as separate boundaries
   - Special treatment required for SU2 implementation

5. **Symmetry Plane (Axisymmetric Mode)**:
   - Symmetry boundary condition along r = 0
   - Only for axisymmetric half-body simulations

### 3.3 Physical Groups for SU2

```python
# Example physical group structure for C-grid
physical_groups = {
    "body": body_surface_curves,      # No-slip wall
    "inflow": upper_lower_farfield,   # Supersonic inflow
    "outflow": downstream_farfield,   # Supersonic outflow
    "wake_cut": wake_line_curves,     # Coordinate cut (periodic/interpolation)
    "sym": symmetry_axis,            # Symmetry plane (if axisymmetric)
    "fluid": all_surfaces            # Computational domain
}
```

---

## 4. SU2 Boundary Conditions for C-Grid

### 4.1 Available Boundary Condition Types

**For Hypersonic C-Grid Blunt Body:**

| Boundary | SU2 Option | Description |
|----------|------------|-------------|
| **Body Wall** | `MARKER_ISOTHERMAL` | No-slip, fixed temperature wall |
| **Inflow** | `MARKER_FAR` | Characteristic-based farfield (subsonic/supersonic) |
| **Outflow** | `MARKER_FAR` or `MARKER_SUPERSONIC_OUTLET` | Extrapolation for supersonic flow |
| **Wake Cut** | `MARKER_PERIODIC` or interpolation | Coordinate cut treatment |
| **Symmetry** | `MARKER_SYM` | Symmetry plane (axisymmetric) |

### 4.2 Configuration File Example

```cfg
% -------------------- BOUNDARY CONDITIONS ---------------------
% Body wall (isothermal)
MARKER_ISOTHERMAL= ( body, 2500.0 )

% Farfield boundaries (inflow/outflow)
MARKER_FAR= ( inflow, outflow )

% Symmetry axis (if axisymmetric)
MARKER_SYM= ( sym )

% Wake cut (coordinate cut) - requires special handling
% Option 1: Treat as two separate boundaries with interpolation
% Option 2: Use periodic boundary conditions
% Option 3: Use marker farfield with appropriate treatment
```

### 4.3 Critical SU2 v8.4 Considerations

**Known Issues and Solutions:**

1. **CFL Adaptation**: `CFL_ADAPT_PARAM` requires `cfl_adapt_max >= 1.0`
2. **Wake Cut Treatment**: SU2 does not natively support C-grid wake cuts
   - **Workaround 1**: Treat wake cut as two separate farfield boundaries
   - **Workaround 2**: Use interpolation between upper and lower wake surfaces
   - **Workaround 3**: Split C-grid into multiple blocks with explicit coupling

3. **Axisymmetric Mode**: 
   - Set `AXISYMMETRIC= YES` in config
   - Only works with half-body domain (r >= 0)
   - Full2D requires `AXISYMMETRIC= NO`

### 4.4 Recommended Approach for SU2

**Option A: Single Block with Interpolation**
```cfg
MARKER_FAR= ( inflow_upper, inflow_lower, outflow )
MARKER_INTERPOLATION= ( wake_upper, wake_lower )
```

**Option B: Multiple Block Approach**
- Block 1: Upper half with symmetry
- Block 2: Lower half with symmetry
- Explicit coupling at wake cut

**Option C: Modified O-Grid with Wake Extension**
- Use O-grid near body
- Extend grid downstream with wake-fitted topology
- Hybrid approach combining benefits of both

---

## 5. Axisymmetric vs Full2D Modes

### 5.1 Axisymmetric Mode (r >= 0)

**Characteristics:**
- Half-body domain only (upper half)
- Symmetry boundary condition along r = 0
- 2D mesh represents axisymmetric 3D flow
- Reduced computational cost

**C-Grid Implementation:**
```
Domain: 0 <= r <= r_max
Boundary conditions:
- Body: No-slip wall (upper surface)
- Symmetry: r = 0 line
- Inflow: Upper farfield
- Outflow: Downstream farfield
- Wake: Cut along symmetry plane
```

**SU2 Configuration:**
```cfg
AXISYMMETRIC= YES
MARKER_SYM= ( symmetry_axis )
MARKER_ISOTHERMAL= ( body, 2500.0 )
MARKER_FAR= ( inflow, outflow )
```

### 5.2 Full2D Mode

**Characteristics:**
- Complete body domain (both upper and lower halves)
- No symmetry boundary condition
- 2D mesh represents 2D planar flow
- Higher computational cost

**C-Grid Implementation:**
```
Domain: -r_max <= r <= r_max
Boundary conditions:
- Body: No-slip wall (complete body surface)
- Inflow: Upper and lower farfield
- Outflow: Downstream farfield
- Wake: Cut line (if applicable)
```

**SU2 Configuration:**
```cfg
AXISYMMETRIC= NO
MARKER_ISOTHERMAL= ( body, 2500.0 )
MARKER_FAR= ( inflow_upper, inflow_lower, outflow )
```

### 5.3 Mode Selection Guidelines

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Zero angle of attack | Axisymmetric | Reduced cost, accurate results |
| Non-zero angle of attack | Full2D | Required for asymmetric flow |
| Visualization | Full2D | Complete body view |
| Parametric studies | Axisymmetric | Faster turnaround |
| Wake analysis | Full2D | Complete wake structure |

---

## 6. Gmsh Implementation Challenges

### 6.1 Geometry Construction Challenges

**Challenge 1: Complex Curves**
- Body surface: Sphere + cone combination
- Farfield: C-shaped boundary with wake extension
- Wake cut: Requires precise point placement

**Solution Approach:**
```python
# Example Gmsh Python API structure
import gmsh

# Initialize
gmsh.initialize()
gmsh.model.add("c_grid_blunt_body")

# Define points
# Body surface points
# Farfield points (C-shaped)
# Wake cut points

# Create curves
# Body curves
# Farfield curves (upper, lower, outflow)
# Wake cut curves

# Create surfaces
# Body surface
# Farfield surface(s)
# Wake region

# Define physical groups
gmsh.model.geo.addPhysicalGroup(1, body_curves, name="body")
gmsh.model.geo.addPhysicalGroup(1, inflow_curves, name="inflow")
gmsh.model.geo.addPhysicalGroup(1, outflow_curves, name="outflow")
gmsh.model.geo.addPhysicalGroup(1, wake_cut_curves, name="wake_cut")
gmsh.model.geo.addPhysicalGroup(2, all_surfaces, name="fluid")
```

### 6.2 Mesh Quality Challenges

**Challenge 1: Cell Quality at Wake Cut**
- High skewness possible at coordinate cut
- Need careful point distribution

**Challenge 2: Transition from Body to Farfield**
- Size ratio between body cells and farfield cells
- Need smooth grading

**Challenge 3: Shock Alignment**
- Grid should align with expected shock location
- Requires knowledge of shock shape

**Solutions:**
1. Use transfinite interpolation for structured regions
2. Apply boundary layer inflation near body
3. Use size fields for shock refinement
4. Optimize mesh with Netgen or similar

### 6.3 Implementation Strategy

**Phase 1: Geometry Generation**
```python
def generate_cgrid_geometry(config, mesh_config):
    """Generate C-grid geometry for blunt body."""
    
    # 1. Generate body contour
    x_body, r_body = generate_contour(config)
    
    # 2. Compute C-grid farfield
    # Upper boundary
    # Lower boundary  
    # Outflow boundary
    # Wake cut line
    
    # 3. Create boundary layer offset
    x_offset, r_offset = compute_offset_contour(x_body, r_body, bl_thickness)
    
    return geometry_data
```

**Phase 2: Mesh Generation**
```python
def generate_cgrid_mesh(geometry_data, mesh_config):
    """Generate mesh using Gmsh."""
    
    # 1. Create geometry in Gmsh
    # 2. Define transfinite constraints
    # 3. Apply boundary layer inflation
    # 4. Generate mesh
    # 5. Optimize mesh quality
    
    return mesh_file
```

**Phase 3: SU2 Configuration**
```python
def generate_su2_config(mesh_config, flow_config):
    """Generate SU2 configuration file."""
    
    # 1. Set boundary conditions
    # 2. Configure solver parameters
    # 3. Set output options
    
    return config_file
```

---

## 7. Reference Papers and Best Practices

### 7.1 Key Research Papers

1. **Lutton, M.J. (1989)**: "Comparison of C- and O-Grid Generation Methods Using a NACA 0012 Airfoil"
   - Demonstrated C-grid advantages for wake resolution
   - Identified O-grid weaknesses in trailing edge region

2. **Arnone, A., Liou, M.S., Povinelli, L.A. (1992)**: "Transonic Cascade Flow Calculations Using Non-Periodic C-Type Grids"
   - Introduced non-periodic C-grids for better orthogonality
   - Showed improved shock capture with C-topology

3. **Ferfouri et al. (2025)**: "Performance Analysis of Grid Topologies and RANS Turbulence Models..."
   - Compared O-grid and C-grid for artillery projectiles
   - C-grid showed better base drag prediction

4. **NASA (2024)**: "Aeroheating Predictions for a Hypersonic, Turbulent Near-Wake"
   - Investigated mesh topology effects on wake heating
   - Found stacked-block meshes improve wake resolution

5. **GridPro (2024)**: "Fast and Accurate Hypersonic CFD Simulations..."
   - Demonstrated shock-aligned mesh generation
   - Showed significant improvement with structured grids

### 7.2 Best Practices

**For Hypersonic Blunt Body C-Grids:**

1. **Body Surface Resolution**
   - First cell height: y+ < 1 (typically 1e-6 * R_nose)
   - Growth ratio: 1.10-1.15
   - Minimum 50 cells in boundary layer

2. **Shock Alignment**
   - Use Billig correlation for shock standoff distance
   - Align grid with expected shock location
   - Apply refinement near shock

3. **Wake Region**
   - Cluster points along wake shear layer
   - Extend domain sufficiently downstream (10-15 body diameters)
   - Resolve recirculation zone

4. **Farfield Boundaries**
   - Inflow: 8-10 body lengths upstream
   - Outflow: 12-15 body diameters downstream
   - Lateral: 8-10 body radii

5. **Mesh Quality Metrics**
   - Orthogonal quality: > 0.9
   - Skewness: < 0.15
   - Aspect ratio: < 10 in boundary layer

### 7.3 Implementation Recommendations

**For Current Project (Apollo CM):**

1. **Phase 1: Validation**
   - Start with axisymmetric mode
   - Validate against existing O-grid results
   - Compare bow shock location and shape

2. **Phase 2: Refinement**
   - Implement shock-aligned refinement
   - Add wake region refinement
   - Optimize mesh quality

3. **Phase 3: Extension**
   - Add full2D capability
   - Implement angle of attack studies
   - Validate against experimental data

---

## 8. Conclusions and Recommendations

### 8.1 Key Findings

1. **C-Grid Superiority**: C-grids offer significant advantages for hypersonic blunt body flows, particularly in wake resolution and shock capture.

2. **Implementation Complexity**: C-grids require more complex geometry construction but provide better solution quality.

3. **SU2 Compatibility**: C-grids can be implemented in SU2 with appropriate boundary condition treatment, though wake cut requires special handling.

4. **Mesh Quality**: Proper implementation yields high-quality meshes with excellent orthogonality and controlled stretching.

### 8.2 Recommendations

1. **Proceed with C-Grid Implementation**: The benefits outweigh the implementation complexity for this application.

2. **Start with Axisymmetric Mode**: Begin with half-body simulations to validate the approach.

3. **Implement in Phases**: Geometry generation, mesh generation, SU2 configuration, and validation.

4. **Validate Thoroughly**: Compare with existing O-grid results and experimental data.

5. **Document Lessons Learned**: Create implementation guide for future projects.

---

## 9. Implementation Roadmap

### Phase 1: Geometry Generation (2-3 days)
- [ ] Define C-grid boundary points
- [ ] Create body surface curves
- [ ] Create farfield curves
- [ ] Create wake cut line
- [ ] Define physical groups

### Phase 2: Mesh Generation (3-4 days)
- [ ] Implement transfinite interpolation
- [ ] Add boundary layer inflation
- [ ] Apply shock refinement
- [ ] Generate and optimize mesh
- [ ] Validate mesh quality

### Phase 3: SU2 Configuration (1-2 days)
- [ ] Define boundary conditions
- [ ] Configure solver parameters
- [ ] Set up convergence strategy
- [ ] Test with simple cases

### Phase 4: Validation (2-3 days)
- [ ] Run axisymmetric simulations
- [ ] Compare with O-grid results
- [ ] Validate against experimental data
- [ ] Document findings

**Total Estimated Time: 8-12 days**

---

## 10. Appendix: Gmsh Code Structure

### 10.1 Basic C-Grid Geometry

```python
import gmsh
import numpy as np

def create_cgrid_geometry(config, mesh_config):
    """Create C-grid geometry for blunt body."""
    
    gmsh.initialize()
    gmsh.model.add("c_grid_blunt_body")
    
    # Body contour
    x_body, r_body = generate_contour(config)
    
    # C-grid farfield parameters
    upstream = mesh_config.upstream_factor * config.R_nose
    downstream = mesh_config.downstream_factor * 2 * config.max_radius
    lateral = mesh_config.lateral_factor * config.R_nose
    
    # Create points
    # Body points
    body_pts = []
    for i in range(len(x_body)):
        pt = gmsh.model.geo.addPoint(x_body[i], r_body[i], 0)
        body_pts.append(pt)
    
    # Farfield points (C-shaped)
    # Upper boundary
    # Lower boundary
    # Outflow boundary
    
    # Create curves
    # Body curves
    body_curves = []
    for i in range(len(body_pts) - 1):
        curve = gmsh.model.geo.addLine(body_pts[i], body_pts[i+1])
        body_curves.append(curve)
    
    # Farfield curves
    # ...
    
    # Create surface
    # ...
    
    gmsh.model.geo.synchronize()
    
    return gmsh
```

### 10.2 Mesh Generation

```python
def generate_cgrid_mesh(gmsh, mesh_config):
    """Generate mesh with transfinite interpolation."""
    
    # Set transfinite constraints
    gmsh.model.geo.mesh.setTransfiniteSurface(surface)
    gmsh.model.geo.mesh.setRecombine(2, surface)
    
    # Set boundary layer inflation
    # ...
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Optimize mesh
    gmsh.model.mesh.optimize("Netgen")
    
    # Export
    gmsh.write("c_grid_mesh.su2")
    
    gmsh.finalize()
```

---

*Report prepared: September 2026*
*Research scope: C-grid topology for hypersonic blunt body CFD*
*Domain: Gmsh mesh generation, SU2 solver, Apollo CM geometry*