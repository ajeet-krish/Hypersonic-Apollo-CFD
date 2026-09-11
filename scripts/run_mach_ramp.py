#!/usr/bin/env python3
"""Run 4-stage Mach ramp on existing 3D mesh.

Uses the already-generated mesh (2.7M elements) and runs:
  Stage 1: M=2.0,  CFL=0.001, 2000 iters
  Stage 2: M=5.0,  CFL=0.003, 2500 iters
  Stage 3: M=10.0, CFL=0.005, 3000 iters
  Stage 4: M=15.6, CFL=0.007, 5000 iters

Usage: nohup uv run python scripts/run_mach_ramp.py > overnight_run.log 2>&1 &
"""
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from physics.atmosphere import standard_atmosphere

OUTPUT_DIR = Path("output/apollo-cm-3d-ramp")
SU2_DIR = OUTPUT_DIR / "su2"
MESH_SRC = Path("output/apollo-cm/mesh/apollo_cm_3d.su2")

STAGES = [
    {"mach": 2.0,  "cfl": 0.001, "iterations": 2000, "name": "M2_init"},
    {"mach": 5.0,  "cfl": 0.003, "iterations": 2500, "name": "M5_supersonic"},
    {"mach": 10.0, "cfl": 0.005, "iterations": 3000, "name": "M10_hypersonic"},
    {"mach": 15.6, "cfl": 0.007, "iterations": 5000, "name": "M15_final"},
]


def run_stage(stage, workdir, mesh_path, is_restart=False):
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

    # Stream output to log file
    result = subprocess.run(
        [str(su2_binary), "config.cfg"],
        cwd=str(workdir),
        timeout=stage["iterations"] * 30,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED: SU2 returned {result.returncode}")
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
    print("  3D RANS MACH RAMP - Apollo CM M=15.6")
    print("=" * 60)

    if not MESH_SRC.exists():
        print(f"  ERROR: Mesh not found at {MESH_SRC}")
        return 1

    print(f"  Mesh: {MESH_SRC} ({MESH_SRC.stat().st_size / 1e6:.1f} MB)")
    print(f"  Stages: {len(STAGES)}, Total iters: {sum(s['iterations'] for s in STAGES):,}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    restart = False
    for stage in STAGES:
        stage_dir = SU2_DIR / stage["name"]
        run_stage(stage, stage_dir, MESH_SRC, is_restart=restart)
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
