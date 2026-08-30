"""Tests for Gmsh mesh generation (requires gmsh Python package)."""
from pathlib import Path

import pytest

try:
    import gmsh  # noqa: F401

    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

from cfd.mesh_config import MeshConfig
from geometry.presets import apollo_cm, generic

# Mark all tests in this module as requiring gmsh
pytestmark = pytest.mark.skipif(
    not HAS_GMSH, reason="gmsh Python package not installed"
)


@pytest.fixture
def draft_config() -> MeshConfig:
    """Draft-tier MeshConfig for fast test meshes."""
    return MeshConfig.for_tier("draft")


@pytest.fixture
def generic_body():
    """Generic blunt body geometry config."""
    return generic()


@pytest.fixture
def apollo_body():
    """Apollo CM blunt body geometry config."""
    return apollo_cm()


class TestGenerateBodyMesh:
    """Tests for generate_body_mesh function."""

    def test_generic_draft_produces_su2(
        self, generic_body, draft_config, tmp_path: Path,
    ):
        """Draft mesh for generic preset should produce a valid .su2 file."""
        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import validate_su2_mesh

        output = tmp_path / "generic_draft.su2"
        result = generate_body_mesh(generic_body, draft_config, mach=8.0, output_path=output)

        assert result.exists()
        assert result.stat().st_size > 0
        assert validate_su2_mesh(result) is True

    def test_apollo_draft_produces_su2(
        self, apollo_body, draft_config, tmp_path: Path,
    ):
        """Draft mesh for Apollo CM should produce a valid .su2 file."""
        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import validate_su2_mesh

        output = tmp_path / "apollo_draft.su2"
        result = generate_body_mesh(apollo_body, draft_config, mach=8.0, output_path=output)

        assert result.exists()
        assert result.stat().st_size > 0
        assert validate_su2_mesh(result) is True

    def test_mesh_has_markers(
        self, generic_body, draft_config, tmp_path: Path,
    ):
        """Generated mesh should have body, farfield, and sym markers."""
        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import check_mesh_quality

        output = tmp_path / "markers_test.su2"
        generate_body_mesh(generic_body, draft_config, mach=8.0, output_path=output)

        quality = check_mesh_quality(output)
        assert quality["has_body_marker"] is True
        assert quality["has_farfield_marker"] is True
        assert quality["has_sym_marker"] is True

    def test_mesh_nonzero_cell_count(
        self, generic_body, draft_config, tmp_path: Path,
    ):
        """Generated mesh should have a reasonable number of cells."""
        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import check_mesh_quality

        output = tmp_path / "cell_count.su2"
        generate_body_mesh(generic_body, draft_config, mach=8.0, output_path=output)

        quality = check_mesh_quality(output)
        assert quality["n_cells"] > 100, "Mesh should have more than 100 cells"
        assert quality["mean_quality"] > 0.0

    def test_no_shock_refinement(
        self, generic_body, tmp_path: Path,
    ):
        """Mesh without shock refinement should still produce valid mesh."""
        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import validate_su2_mesh

        config = MeshConfig.for_tier("draft", shock_refinement=False)
        output = tmp_path / "no_shock.su2"
        result = generate_body_mesh(generic_body, config, mach=8.0, output_path=output)

        assert result.exists()
        assert validate_su2_mesh(result) is True

    def test_returns_output_path(
        self, generic_body, draft_config, tmp_path: Path,
    ):
        """Should return the output path."""
        from cfd.mesh import generate_body_mesh

        output = tmp_path / "return_test.su2"
        result = generate_body_mesh(generic_body, draft_config, mach=8.0, output_path=output)
        assert result == output


class TestTierDifferentiation:
    """Integration tests verifying tiers produce different mesh densities."""

    def test_tiers_produce_different_meshes(
        self, generic_body, tmp_path: Path,
    ):
        """Draft should have fewer cells than standard, standard fewer than high."""
        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import check_mesh_quality

        counts: dict[str, int] = {}
        for tier in ("draft", "standard", "high"):
            config = MeshConfig.for_tier(tier)
            output = tmp_path / f"{tier}_tier.su2"
            generate_body_mesh(generic_body, config, mach=8.0, output_path=output)
            quality = check_mesh_quality(output)
            counts[tier] = quality["n_cells"]

        assert counts["draft"] < counts["standard"], (
            f"Draft ({counts['draft']}) should have fewer cells than "
            f"standard ({counts['standard']})"
        )
        assert counts["standard"] < counts["high"], (
            f"Standard ({counts['standard']}) should have fewer cells than "
            f"high ({counts['high']})"
        )

    def test_boundary_layer_effectiveness(
        self, generic_body, tmp_path: Path,
    ):
        """Cells near the wall should be much smaller than far-field cells."""
        import numpy as np

        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import _element_area, _parse_su2_mesh

        config = MeshConfig.for_tier("draft")
        output = tmp_path / "bl_test.su2"
        generate_body_mesh(generic_body, config, mach=8.0, output_path=output)

        nodes, elements, markers = _parse_su2_mesh(output)
        R_nose = generic_body.R_nose

        # Get body surface node coordinates to compute distance to body
        body_elems = markers.get("body", [])
        assert len(body_elems) > 0, "No body marker elements found"
        body_node_ids = set()
        for elem in body_elems:
            for nid in elem[1:]:
                body_node_ids.add(nid)
        body_coords = nodes[list(body_node_ids)]

        # Compute element areas and centroid distances to body for all elements
        near_areas: list[float] = []
        far_areas: list[float] = []
        body_length = generic_body.computed_body_length

        for elem in elements:
            area = abs(_element_area(nodes, elem))
            if area < 1e-15:
                continue
            # Element centroid
            node_ids = elem[1:]
            centroid = nodes[node_ids].mean(axis=0)
            x_c = centroid[0]
            # Distance to body surface (approximate: min distance to body nodes)
            dists = np.linalg.norm(body_coords[:, :2] - centroid[:2], axis=1)
            min_dist = float(np.min(dists))

            if min_dist < 0.5 * R_nose:
                near_areas.append(area)
            elif x_c > body_length + 3.0 * R_nose:
                far_areas.append(area)

        assert len(near_areas) > 0, "No near-wall elements found"
        assert len(far_areas) > 0, "No far-field elements found"

        avg_near = sum(near_areas) / len(near_areas)
        avg_far = sum(far_areas) / len(far_areas)

        # BL elements should be significantly smaller than far-field
        assert avg_near < 0.3 * avg_far, (
            f"Near-wall avg area ({avg_near:.8f}) should be < 30% of "
            f"far-field avg area ({avg_far:.8f}). BL refinement may be inert."
        )

    def test_shock_refinement_effect(
        self, generic_body, tmp_path: Path,
    ):
        """Cell density near Billig standoff should be higher than far from it."""
        import numpy as np

        from cfd.mesh import generate_body_mesh
        from cfd.mesh_quality import _parse_su2_mesh
        from validation.billig import billig_blunted_cone

        config = MeshConfig.for_tier("draft")
        output = tmp_path / "shock_test.su2"
        generate_body_mesh(generic_body, config, mach=8.0, output_path=output)

        nodes, _elements, _markers = _parse_su2_mesh(output)
        R_nose = generic_body.R_nose

        # Billig standoff position
        standoff = billig_blunted_cone(R_nose, 8.0)
        shock_x = standoff.delta

        # Measure average element size near shock vs far downstream
        # Near shock: x in [shock_x - 0.5*R_nose, shock_x + 0.5*R_nose], r < 2*R_nose
        # Far downstream: x > body_length, r < 2*R_nose
        body_length = generic_body.computed_body_length

        near_shock_nodes = []
        far_downstream_nodes = []
        for node in nodes:
            x, r = node[0], node[1]
            if r < 2.0 * R_nose:
                if abs(x - shock_x) < 0.5 * R_nose:
                    near_shock_nodes.append(node)
                elif x > body_length + 2.0 * R_nose:
                    far_downstream_nodes.append(node)

        # Compute average nearest-neighbor distance for each region
        def avg_nn_distance(pts: list) -> float:
            if len(pts) < 2:
                return float("inf")
            arr = np.array(pts)
            dists = np.linalg.norm(arr[:, np.newaxis, :] - arr[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            nn = np.min(dists, axis=1)
            return float(np.mean(nn))

        if near_shock_nodes and far_downstream_nodes:
            shock_size = avg_nn_distance(near_shock_nodes)
            far_size = avg_nn_distance(far_downstream_nodes)
            # Near shock should be refined (smaller cells)
            assert shock_size < far_size, (
                f"Near-shock avg size ({shock_size:.6f}) should be < "
                f"far-downstream avg size ({far_size:.6f})"
            )
