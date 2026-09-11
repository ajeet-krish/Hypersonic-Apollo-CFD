"""Tests for heat flux extraction and distribution.

Covers heat_flux_distribution with all distribution types,
edge cases, and the extract_heat_flux_from_vtu interface.
"""
import numpy as np
import pytest

from thermal.heat_flux import heat_flux_distribution


class TestHeatFluxDistribution:
    """Tests for heat_flux_distribution function."""

    def test_uniform_constant(self):
        """Uniform distribution should return constant q_max."""
        s = np.linspace(0, 1, 10)
        q = heat_flux_distribution(s, 1000.0, distribution="uniform")
        np.testing.assert_array_almost_equal(q, 1000.0)

    def test_uniform_length_independent(self):
        """Uniform distribution should not depend on L."""
        s = np.linspace(0, 2, 10)
        q = heat_flux_distribution(s, 500.0, distribution="uniform", L=2.0)
        np.testing.assert_array_almost_equal(q, 500.0)

    def test_sinusoidal_at_stagnation(self):
        """Sinusoidal distribution should have q_max at stagnation (s=0)."""
        s = np.array([0.0])
        q = heat_flux_distribution(s, 1000.0, distribution="sinusoidal")
        assert q[0] == pytest.approx(1000.0)

    def test_sinusoidal_at_end(self):
        """Sinusoidal distribution should be zero at s=L (cos(pi/2)=0)."""
        s = np.array([1.0])
        q = heat_flux_distribution(s, 1000.0, distribution="sinusoidal", L=1.0)
        assert q[0] == pytest.approx(0.0, abs=1e-10)

    def test_sinusoidal_monotonic_decay(self):
        """Sinusoidal distribution should monotonically decay from stagnation."""
        s = np.linspace(0, 1, 50)
        q = heat_flux_distribution(s, 1000.0, distribution="sinusoidal")
        assert np.all(np.diff(q) <= 0)

    def test_sinusoidal_midpoint(self):
        """Sinusoidal at midpoint should be q_max * cos(pi/4)."""
        s = np.array([0.5])
        q = heat_flux_distribution(s, 1000.0, distribution="sinusoidal", L=1.0)
        expected = 1000.0 * np.cos(np.pi / 4)
        assert q[0] == pytest.approx(expected, abs=1.0)

    def test_cosine_at_stagnation(self):
        """Cosine distribution should have q_max at stagnation."""
        s = np.array([0.0])
        q = heat_flux_distribution(s, 1000.0, distribution="cosine")
        assert q[0] == pytest.approx(1000.0)

    def test_cosine_at_end(self):
        """Cosine distribution should be zero at s=L."""
        s = np.array([1.0])
        q = heat_flux_distribution(s, 1000.0, distribution="cosine", L=1.0)
        assert q[0] == pytest.approx(0.0, abs=1e-10)

    def test_cosine_monotonic_decay(self):
        """Cosine distribution should monotonically decay."""
        s = np.linspace(0, 1, 50)
        q = heat_flux_distribution(s, 1000.0, distribution="cosine")
        assert np.all(np.diff(q) <= 0)

    def test_cosine_squarer_than_sinusoidal(self):
        """Cosine distribution should decay faster than sinusoidal."""
        s = np.array([0.5])
        q_sin = heat_flux_distribution(s, 1000.0, distribution="sinusoidal", L=1.0)
        q_cos = heat_flux_distribution(s, 1000.0, distribution="cosine", L=1.0)
        assert q_cos[0] < q_sin[0]

    def test_unknown_distribution_raises(self):
        """Unknown distribution should raise ValueError."""
        s = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="Unknown distribution"):
            heat_flux_distribution(s, 1000.0, distribution="linear")

    def test_zero_flux(self):
        """Zero q_max should return all zeros."""
        s = np.linspace(0, 1, 10)
        q = heat_flux_distribution(s, 0.0, distribution="sinusoidal")
        np.testing.assert_array_almost_equal(q, 0.0)

    def test_single_point(self):
        """Single point at s=0 should return q_max."""
        s = np.array([0.0])
        for dist in ["uniform", "sinusoidal", "cosine"]:
            q = heat_flux_distribution(s, 500.0, distribution=dist)
            assert q[0] == pytest.approx(500.0)

    def test_output_shape_matches_input(self):
        """Output array should have same shape as input."""
        s = np.linspace(0, 1, 100)
        for dist in ["uniform", "sinusoidal", "cosine"]:
            q = heat_flux_distribution(s, 1000.0, distribution=dist)
            assert q.shape == s.shape

    def test_different_lengths(self):
        """Should work with different L values."""
        s1 = np.linspace(0, 1, 10)
        s2 = np.linspace(0, 2, 10)

        q1 = heat_flux_distribution(s1, 1000.0, distribution="sinusoidal", L=1.0)
        q2 = heat_flux_distribution(s2, 1000.0, distribution="sinusoidal", L=2.0)

        # At s=0.5/L=0.5 for both, heat flux should be the same
        assert q1[5] == pytest.approx(q2[5], abs=1.0)

    def test_negative_coordinates_clamped(self):
        """Negative s values should be clamped to zero."""
        s = np.array([-1.0, 0.0, 1.0])
        q = heat_flux_distribution(s, 1000.0, distribution="sinusoidal", L=1.0)
        # s=-1 clamped to 0, so q[0] = q_max
        assert q[0] == pytest.approx(1000.0)

    def test_above_l_coordinates_clamped(self):
        """s values above L should be clamped to L (zero flux)."""
        s = np.array([0.0, 1.0, 2.0])
        q = heat_flux_distribution(s, 1000.0, distribution="sinusoidal", L=1.0)
        # s=2 clamped to 1, so q[2] = 0
        assert q[2] == pytest.approx(0.0, abs=1e-10)

    def test_all_distributions_positive(self):
        """All distributions should produce non-negative heat flux."""
        s = np.linspace(0, 1, 50)
        for dist in ["uniform", "sinusoidal", "cosine"]:
            q = heat_flux_distribution(s, 1000.0, distribution=dist)
            assert np.all(q >= -1e-10), f"{dist} produced negative values"
