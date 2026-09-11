#!/usr/bin/env python3
"""Overnight 3D RANS - Apollo CM M=15.6.

Mesh: Draft tier, NO Netgen optimization (fast generation)
Stages: 4-stage Mach ramp M=2->5->10->15.6
Total: 12,500 iterations

Usage: nohup uv run python scripts/run_3d_overnight_v2.py > overnight_run.log 2>&1 &
"""
import sys
import time
import math
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gmsh
from geometry.presets import apollo_cm
from geometry.blunt_body import generate_contour
from physics.atmosphere import standard_atmosphere

OUTPUT_DIR = Path("output/apollo-cm-3d-overnight")
MESH_DIR = OUTPUT_DIR / "mesh"
SU2_DIR = OUTPUT_DIR / "su2"

STAGES = [
    {"mach": 2.0,  "cfl": 0.001, "iterations": 2000, "name": "M2_init"},
    {"mach": 5.0,  "cfl": 0.003, "iterations": 2500, "name": "M5_supersonic"},
    {"mach": 10.0, "cfl": 0.005, "iterations": 3000, "name": "M10_hypersonic"},
    {"mach": 15.6, "cfl": 0.007, "iterations": 5000, "name": "M15_final"},
]


def generate_mesh() -> Path:
    """Generate 3D mesh WITHOUT Netgen optimization."""
    print("\n" + "=" * 60)
    print("  STEP 1: Generate 3D Mesh (no Netgen)")
    print("=" * 60)

    body_config = apollo_cm()
    x_contour, r_contour = generate_contour(body_config)
    x_mm = x_contour * 1000.0
    r_mm = r_contour * 1000.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("wind_tunnel")

    n = 50
    idx = np.linspace(0, len(x_mm) - 1, n, dtype=int)
    x_ds, r_ds = x_mm[idx], r_mm[idx]

    pts = []
    for i in range(n):
        pts.append(gmsh.model.occ.addPoint(float(x_ds[i]), float(r_ds[i]), 0.0))
    pts.append(gmsh.model.occ.addPoint(float(x_ds[-1]), 0.0, 0.0))
    pts.append(gmsh.model.occ.addPoint(float(x_ds[0]), 0.0, 0.0))
    gmsh.model.occ.synchronize()

    spline = gmsh.model.occ.addSpline(pts[:n])
    line_base = gmsh.model.occ.addLine(pts[n - 1], pts[n])
    line_axis = gmsh.model.occ.addLine(pts[n], pts[n + 1])
    gmsh.model.occ.synchronize()

    wire = gmsh.model.occ.addWire([spline, line_base, line_axis])
    surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()

    angle = 2.0 * math.pi - 0.01
    gmsh.model.occ.revolve([(2, surface)], 0, 0, 0, 1, 0, 0, angle)
    gmsh.model.occ.synchronize()

    body_vol = [t for d, t in gmsh.model.getEntities(3) if d == 3][0]

    R_nose_mm = body_config.R_nose * 1000.0
    body_diameter_mm = body_config.max_radius * 2 * 1000.0
    x_min = float(x_mm.min()) - 10.0 * R_nose_mm
    x_max = float(x_mm.max()) + 20.0 * body_diameter_mm
    radius = 8.0 * R_nose_mm
    cyl_len = x_max - x_min

    box_tag = gmsh.model.occ.addBox(x_min, -radius, -radius, cyl_len, 2 * radius, 2 * radius)
    gmsh.model.occ.synchronize()

    print("  Boolean subtract...")
    gmsh.model.occ.cut([(3, box_tag)], [(3, body_vol)])
    gmsh.model.occ.synchronize()

    fluid_vols = [t for d, t in gmsh.model.getEntities(3) if d == 3]
    all_surfs = [t for d, t in gmsh.model.getEntities(2) if d == 2]
    body_surfs = []
    farfield_surfs = []
    body_bbox = (float(x_mm.min()), -float(r_mm.max()), -float(r_mm.max()),
                 float(x_mm.max()), float(r_mm.max()), float(r_mm.max()))

    for st in all_surfs:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, st)
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        in_body = (body_bbox[0] - 10 <= cx <= body_bbox[3] + 10 and
                   body_bbox[1] - 10 <= cy <= body_bbox[4] + 10 and
                   body_bbox[2] - 10 <= cz <= body_bbox[5] + 10)
        if in_body:
            body_surfs.append(st)
        else:
            farfield_surfs.append(st)

    print(f"  Surfs: body={len(body_surfs)}, farfield={len(farfield_surfs)}")

    if body_surfs:
        gmsh.model.geo.addPhysicalGroup(2, body_surfs, name="body")
    if farfield_surfs:
        gmsh.model.geo.addPhysicalGroup(2, farfield_surfs, name="farfield")
    gmsh.model.geo.addPhysicalGroup(3, fluid_vols, name="fluid")
    gmsh.model.geo.synchronize()

    dist_tag = 100
    gmsh.model.mesh.field.add("Distance", dist_tag)
    if body_surfs:
        gmsh.model.mesh.field.setNumbers(dist_tag, "SurfacesList", body_surfs)

    size_tag = 200
    gmsh.model.mesh.field.add("MathEval", size_tag)
    gmsh.model.mesh.field.setString(
        size_tag, "F",
        f"50.0 * Exp(Min(F{dist_tag}, {radius}) * Log(2000.0/50.0) / {radius})",
    )

    bg_tag = 300
    gmsh.model.mesh.field.add("Constant", bg_tag)
    gmsh.model.mesh.field.setNumber(bg_tag, "VIn", 2000.0)
    gmsh.model.mesh.field.setNumber(bg_tag, "VOut", 2000.0)

    min_tag = 999
    gmsh.model.mesh.field.add("Min", min_tag)
    gmsh.model.mesh.field.setNumbers(min_tag, "FieldsList", [size_tag, bg_tag])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_tag)

    print("  Meshing...")
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    t0 = time.time()
    gmsh.model.mesh.generate(3)
    print(f"  Mesh generated in {time.time() - t0:.1f}s")

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"  Nodes: {n_nodes}")

    # NO optimization - export directly
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    msh_path = MESH_DIR / "mesh.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()

    # Convert to SU2
    import meshio
    msh = meshio.read(str(msh_path))
    tetra_cells = [c for c in msh.cells if c.type == "tetra"]
    all_tetra = np.vstack([c.data for c in tetra_cells])
    points = msh.points

    body_tris = []
    farfield_tris = []
    for i, c in enumerate(msh.cells):
        if c.type == "triangle":
            phys = msh.cell_data["gmsh:physical"][i]
            tag = int(phys[0]) if len(phys) > 0 else 0
            for tri in c.data:
                if tag == 1:
                    body_tris.append(tri)
                elif tag == 2:
                    farfield_tris.append(tri)

    su2_path = MESH_DIR / "mesh.su2"
    with open(su2_path, "w") as f:
        f.write(f"NDIME= 3\n")
        f.write(f"NELEM= {len(all_tetra)}\n")
        for elem in all_tetra:
            f.write(f"10 {elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
        f.write(f"NPOIN= {len(points)}\n")
        for pt in points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
        f.write(f"NMARK= 2\n")
        f.write(f"MARKER_TAG= body\n")
        f.write(f"MARKER_ELEMS= {len(body_tris)}\n")
        for tri in body_tris:
            f.write(f"5 {tri[0]} {tri[1]} {tri[2]}\n")
        f.write(f"MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {len(farfield_tris)}\n")
        for tri in farfield_tris:
            f.write(f"5 {tri[0]} {tri[1]} {tri[2]}\n")

    size_mb = su2_path.stat().st_size / 1e6
    print(f"  SU2 mesh: {su2_path} ({size_mb:.1f} MB)")
    print(f"  Elements: {len(all_tetra):,}, Nodes: {len(points):,}")
    print(f"  Body tris: {len(body_tris)}, Farfield tris: {len(farfield_tris)}")

    return su2_path


def run_stage(stage: dict, workdir: Path, mesh_path: Path, is_restart: bool = False) -> bool:
    """Run a single SU2 stage."""
    import shutil

    workdir.mkdir(parents=True, exist_ok=True)

    if not is_restart:
        dest = workdir / "mesh.su2"
        if not dest.exists() or mesh_path.stat().st_size != dest.stat().st_size:
            shutil.copy2(mesh_path, dest)

    atm = standard_atmosphere(30000.0)
    V_inf = atm.speed_of_sound * stage["mach"]
    reynolds = atm.density * V_inf * 1.0 / atm.dynamic_viscosity
    restart_line = "RESTART_SOL= YES" if is_restart else "RESTART_SOL= NO"

    config = f"""% 3D RANS - Apollo CM M={stage['mach']:.1f}
SOLVER= RANS
KIND_TURB_MODEL= SA
FREESTREAM_TURBULENCEINTENSITY= 0.05
FREESTREAM_TURB2LAMVISCRATIO= 10.0
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT
MATH_PROBLEM= DIRECT
{restart_line}
AXISYMMETRIC= NO
GAMMA_VALUE= 1.4
GAS_CONSTANT= 287.058
SYSTEM_MEASUREMENTS= SI
MACH_NUMBER= {stage['mach']}
AOA= 0.0
FREESTREAM_PRESSURE= {atm.pressure:.2f}
FREESTREAM_TEMPERATURE= {atm.temperature:.2f}
FREESTREAM_DENSITY= {atm.density:.6f}
FREESTREAM_VISCOSITY= {atm.dynamic_viscosity:.6e}
REYNOLDS_NUMBER= {reynolds:.0f}
REYNOLDS_LENGTH= 1.0
MARKER_ISOTHERMAL= ( body, 2500.0 )
MARKER_FAR= ( farfield )
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= NO
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
ENTROPY_FIX_COEFF= 0.1
TIME_DISCRE_FLOW= EULER_IMPLICIT
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-6
LINEAR_SOLVER_ITER= 100
MGLEVEL= 0
ITER= {stage['iterations']}
CFL_NUMBER= {stage['cfl']}
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.0001, 1.0, 0.5, 1.2 )
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -6.0
CONV_STARTITER= 100
REF_AREA= 1.0
REF_LENGTH= 1.0
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )
OUTPUT_FILES= ( RESTART, PARAVIEW )
VOLUME_FILENAME= flow
HISTORY_OUTPUT= ( ITER, RMS_RES, LIFT, DRAG )
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
"""
    with open(workdir / "config.cfg", "w") as f:
        f.write(config)

    print(f"\n  Running {stage['name']}: M={stage['mach']}, CFL={stage['cfl']}, "
          f"ITER={stage['iterations']}")

    su2_binary = Path.home() / "SU2_CFD" / "bin" / "SU2_CFD"
    t0 = time.time()
    result = subprocess.run(
        [str(su2_binary), "config.cfg"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=stage["iterations"] * 30,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED: SU2 returned {result.returncode}")
        if result.stderr:
            # Print last few lines of stderr
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
        return False

    history_path = workdir / "history.csv"
    if history_path.exists():
        with open(history_path) as f:
            lines = f.readlines()
        if len(lines) > 2:
            last = lines[-1].strip().split(",")
            rms = float(last[3]) if len(last) > 3 else 999
            iters = len(lines) - 1
            print(f"  Done: {iters} iters, RMS={rms:.4f}, {elapsed:.0f}s")
            return rms < -5.0

    print(f"  Completed in {elapsed:.0f}s")
    return True


def main():
    t_total = time.time()
    print("=" * 60)
    print("  OVERNIGHT 3D RANS - Apollo CM M=15.6")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Stages: {len(STAGES)}, Total iters: {sum(s['iterations'] for s in STAGES):,}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    mesh_path = generate_mesh()

    print("\n" + "=" * 60)
    print("  STEP 2: Mach Ramp")
    print("=" * 60)

    restart = False
    for stage in STAGES:
        stage_dir = SU2_DIR / stage["name"]
        run_stage(stage, stage_dir, mesh_path, is_restart=restart)
        restart = True

    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"  ALL DONE: {elapsed / 3600:.1f} hours")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for stage in STAGES:
        h = SU2_DIR / stage["name"] / "history.csv"
        if h.exists():
            with open(h) as f:
                lines = f.readlines()
            if len(lines) > 2:
                last = lines[-1].strip().split(",")
                rms = float(last[3]) if len(last) > 3 else "N/A"
                print(f"  {stage['name']:20s}: {len(lines)-1:5d} iters, RMS={rms}")
        else:
            print(f"  {stage['name']:20s}: NO OUTPUT")


if __name__ == "__main__":
    main()
