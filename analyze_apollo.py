import sys
sys.path.insert(0, '/Users/ajeet/Projects/Hypersonic Body CFD/src')
import gmsh
import numpy as np

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.merge('/Users/ajeet/Projects/Hypersonic Body CFD/output/apollo-cm/cad_draft.step')
gmsh.model.occ.synchronize()

# Get all entities
surfaces = gmsh.model.getEntities(dim=2)
curves = gmsh.model.getEntities(dim=1)
points = gmsh.model.getEntities(dim=0)

print(f"Surfaces: {len(surfaces)}")
print(f"Curves: {len(curves)}")
print(f"Points: {len(points)}")

# Get bounding box
print("\n=== SURFACE BOUNDING BOXES ===")
for dim, tag in surfaces:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
    print(f"Surface {tag}: bbox = ({xmin:.4f}, {ymin:.4f}, {zmin:.4f}) to ({xmax:.4f}, {ymax:.4f}, {zmax:.4f})")
    print(f"  Width: {xmax-xmin:.4f}, Height: {ymax-ymin:.4f}")

# Get curve lengths
print("\n=== CURVE LENGTHS ===")
for dim, tag in curves:
    length = gmsh.model.occ.getLength((dim, tag))
    bbox = gmsh.model.occ.getBoundingBox(dim, tag)
    print(f"Curve {tag}: length={length:.4f}, bbox=({bbox[0]:.4f},{bbox[1]:.4f}) to ({bbox[4]:.4f},{bbox[5]:.4f})")

# Get points
print("\n=== POINTS ===")
for dim, tag in points:
    x, y, z = gmsh.model.occ.getPoint((dim, tag))
    print(f"Point {tag}: ({x:.4f}, {y:.4f}, {z:.4f})")

# Overall bounding box
print("\n=== OVERALL BOUNDING BOX ===")
all_boxes = []
for dim, tag in surfaces:
    all_boxes.append(gmsh.model.occ.getBoundingBox(dim, tag))
if all_boxes:
    xmin = min(b[0] for b in all_boxes)
    ymin = min(b[1] for b in all_boxes)
    zmin = min(b[2] for b in all_boxes)
    xmax = max(b[3] for b in all_boxes)
    ymax = max(b[4] for b in all_boxes)
    zmax = max(b[5] for b in all_boxes)
    print(f"Overall: ({xmin:.4f}, {ymin:.4f}, {zmin:.4f}) to ({xmax:.4f}, {ymax:.4f}, {zmax:.4f})")
    print(f"Width (X): {xmax-xmin:.4f}")
    print(f"Depth (Y): {ymax-ymin:.4f}")
    print(f"Height (Z): {zmax-zmin:.4f}")

gmsh.finalize()
