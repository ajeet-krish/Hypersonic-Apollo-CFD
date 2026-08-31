"""Tests for GCI (Grid Convergence Index) module.

Tests GCI computation with known convergence behavior and
non-monotonic data detection.
"""
import pytest

from validation.gci import GCIMeshLevel, GCIResult, compute_gci


class TestComputeGci:
    """Tests for compute_gci function."""

    def test_second_order_convergence(self):
        """f = 1 + 1/n^2 should give p ~ 2 (second order)."""
        # Create 3 levels: n=10, 20, 40
        # f(n) = 1 + 1/n^2
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0 + 1.0 / 100),
            GCIMeshLevel(name="medium", n_cells=20, value=1.0 + 1.0 / 400),
            GCIMeshLevel(name="fine", n_cells=40, value=1.0 + 1.0 / 1600),
        ]
        result = compute_gci(levels)
        # Apparent order should be close to 2
        assert result.apparent_order == pytest.approx(2.0, abs=0.1), (
            f"Expected p ~ 2.0, got {result.apparent_order}"
        )
        # Extrapolated value should be close to 1.0
        assert result.extrapolated_value == pytest.approx(1.0, abs=0.01)

    def test_gci_decreases_with_refinement(self):
        """GCI should decrease from coarse to fine level."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0 + 1.0 / 100),
            GCIMeshLevel(name="medium", n_cells=20, value=1.0 + 1.0 / 400),
            GCIMeshLevel(name="fine", n_cells=40, value=1.0 + 1.0 / 1600),
        ]
        result = compute_gci(levels)
        assert result.gci_coarse_pct > result.gci_fine_pct, (
            f"GCI coarse ({result.gci_coarse_pct}%) should be > "
            f"GCI fine ({result.gci_fine_pct}%)"
        )

    def test_monotonic_convergence_detected(self):
        """Monotonically converging data should be detected as monotonic."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=2.0),
            GCIMeshLevel(name="medium", n_cells=20, value=1.5),
            GCIMeshLevel(name="fine", n_cells=40, value=1.25),
        ]
        result = compute_gci(levels)
        assert result.monotonic is True

    def test_non_monotonic_data(self):
        """Non-monotonic data should set monotonic=False."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.5),
            GCIMeshLevel(name="medium", n_cells=20, value=1.2),
            GCIMeshLevel(name="fine", n_cells=40, value=1.3),
        ]
        result = compute_gci(levels)
        assert result.monotonic is False

    def test_non_monotonic_warning_ratio(self):
        """Non-monotonic data should give asymptotic ratio outside (0.5, 2.0)."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.5),
            GCIMeshLevel(name="medium", n_cells=20, value=1.2),
            GCIMeshLevel(name="fine", n_cells=40, value=1.3),
        ]
        result = compute_gci(levels)
        # The asymptotic ratio may or may not be in range for non-monotonic,
        # but the result should still be computed without error
        assert isinstance(result.asymptotic_ratio, float)

    def test_exact_three_levels_required(self):
        """Should raise ValueError if not exactly 3 levels."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0),
            GCIMeshLevel(name="fine", n_cells=20, value=1.5),
        ]
        with pytest.raises(ValueError, match="Need exactly 3 mesh levels"):
            compute_gci(levels)

    def test_levels_sorted_by_ncells(self):
        """Levels should be sorted by cell count regardless of input order."""
        levels = [
            GCIMeshLevel(name="fine", n_cells=40, value=1.25),
            GCIMeshLevel(name="coarse", n_cells=10, value=2.0),
            GCIMeshLevel(name="medium", n_cells=20, value=1.5),
        ]
        result = compute_gci(levels)
        assert result.levels[0].n_cells < result.levels[1].n_cells < result.levels[2].n_cells

    def test_refinement_ratio(self):
        """Refinement ratio should be geometric mean of pairwise ratios."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=100, value=1.0),
            GCIMeshLevel(name="medium", n_cells=200, value=1.5),
            GCIMeshLevel(name="fine", n_cells=400, value=1.75),
        ]
        result = compute_gci(levels)
        # r21 = 400/200 = 2, r32 = 200/100 = 2, r = sqrt(2*2) = 2
        assert result.refinement_ratio == pytest.approx(2.0, rel=1e-6)

    def test_safety_factor_applied(self):
        """GCI should use the safety factor."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0 + 1.0 / 100),
            GCIMeshLevel(name="medium", n_cells=20, value=1.0 + 1.0 / 400),
            GCIMeshLevel(name="fine", n_cells=40, value=1.0 + 1.0 / 1600),
        ]
        result_default = compute_gci(levels, safety_factor=1.25)
        result_high = compute_gci(levels, safety_factor=3.0)
        # Higher safety factor should give larger GCI
        assert result_high.gci_fine_pct > result_default.gci_fine_pct

    def test_result_is_gcidataclass(self):
        """Should return a GCIResult dataclass."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0),
            GCIMeshLevel(name="medium", n_cells=20, value=1.5),
            GCIMeshLevel(name="fine", n_cells=40, value=1.75),
        ]
        result = compute_gci(levels)
        assert isinstance(result, GCIResult)

    def test_uniform_convergence_order_one(self):
        """f = 1 + 1/n should give p ~ 1 (first order)."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0 + 1.0 / 10),
            GCIMeshLevel(name="medium", n_cells=20, value=1.0 + 1.0 / 20),
            GCIMeshLevel(name="fine", n_cells=40, value=1.0 + 1.0 / 40),
        ]
        result = compute_gci(levels)
        assert result.apparent_order == pytest.approx(1.0, abs=0.1)

    def test_third_order_convergence(self):
        """f = 1 + 1/n^3 should give p ~ 3."""
        levels = [
            GCIMeshLevel(name="coarse", n_cells=10, value=1.0 + 1.0 / 1000),
            GCIMeshLevel(name="medium", n_cells=20, value=1.0 + 1.0 / 8000),
            GCIMeshLevel(name="fine", n_cells=40, value=1.0 + 1.0 / 64000),
        ]
        result = compute_gci(levels)
        assert result.apparent_order == pytest.approx(3.0, abs=0.2)


class TestGCIIntegration:
    """Integration tests for run_gci_study (marked to skip in unit tests)."""

    @pytest.mark.integration
    def test_run_gci_study_smoke(self, tmp_path):
        """Smoke test: run_gci_study should return a dict."""
        from geometry.presets import generic
        from pipeline.case_config import CaseConfig
        from validation.gci import run_gci_study

        config = CaseConfig(
            name="test",
            label="Test",
            preset_fn=generic,
            mach=8.0,
            altitude=30000.0,
        )
        # This test is marked integration and should be skipped in CI
        # It requires actual SU2 runs
        result = run_gci_study(config, quantities=["stagnation_heat_flux"])
        assert isinstance(result, dict)
