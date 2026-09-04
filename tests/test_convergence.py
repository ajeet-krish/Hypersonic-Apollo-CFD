"""Tests for multi-stage convergence strategy.

Covers ConvergenceStage defaults, for_mach() factory thresholds,
mach_ramp() interpolation, and stage parameter invariants.
"""
import pytest

from cfd.convergence import ConvergenceStage, ConvergenceStrategy


# ---------------------------------------------------------------------------
# ConvergenceStage dataclass
# ---------------------------------------------------------------------------

class TestConvergenceStage:
    """Tests for ConvergenceStage dataclass defaults and construction."""

    def test_default_values(self):
        stage = ConvergenceStage(
            name="test", mach=5.0, cfl=0.01, iterations=1000,
        )
        assert stage.muscl is False
        assert stage.linear_solver == "BCGSTAB"
        assert stage.linear_solver_error == pytest.approx(1e-2)
        assert stage.linear_solver_iter == 20
        assert stage.cfl_adapt_min == pytest.approx(0.0005)
        assert stage.cfl_adapt_max == pytest.approx(0.05)
        assert stage.cfl_adapt_decrease == pytest.approx(0.5)
        assert stage.cfl_adapt_increase == pytest.approx(1.5)

    def test_custom_values(self):
        stage = ConvergenceStage(
            name="custom",
            mach=10.0,
            cfl=0.05,
            iterations=5000,
            muscl=True,
            linear_solver="FGMRES",
            linear_solver_error=1e-4,
            linear_solver_iter=40,
            cfl_adapt_min=0.001,
            cfl_adapt_max=0.1,
            cfl_adapt_decrease=0.3,
            cfl_adapt_increase=2.0,
        )
        assert stage.mach == 10.0
        assert stage.cfl == 0.05
        assert stage.iterations == 5000
        assert stage.muscl is True
        assert stage.linear_solver == "FGMRES"
        assert stage.linear_solver_error == pytest.approx(1e-4)
        assert stage.linear_solver_iter == 40
        assert stage.cfl_adapt_min == pytest.approx(0.001)
        assert stage.cfl_adapt_max == pytest.approx(0.1)
        assert stage.cfl_adapt_decrease == pytest.approx(0.3)
        assert stage.cfl_adapt_increase == pytest.approx(2.0)

    def test_name_preserved(self):
        stage = ConvergenceStage(
            name="M=5.0 first-order", mach=5.0, cfl=0.005, iterations=3000,
        )
        assert stage.name == "M=5.0 first-order"


# ---------------------------------------------------------------------------
# ConvergenceStrategy.for_mach() factory
# ---------------------------------------------------------------------------

class TestConvergenceStrategyForMach:
    """Tests for ConvergenceStrategy.for_mach() factory method."""

    # --- Single stage (M <= 5) ---

    def test_mach_1_single_stage(self):
        s = ConvergenceStrategy.for_mach(1.0)
        assert len(s.stages) == 1

    def test_mach_3_single_stage(self):
        s = ConvergenceStrategy.for_mach(3.0)
        assert len(s.stages) == 1

    def test_mach_5_boundary_single(self):
        s = ConvergenceStrategy.for_mach(5.0)
        assert len(s.stages) == 1

    def test_single_stage_mach_matches_target(self):
        s = ConvergenceStrategy.for_mach(4.0)
        assert s.stages[0].mach == pytest.approx(4.0)

    # --- Two stages (5 < M <= 10) ---

    def test_mach_5_001_two_stage(self):
        s = ConvergenceStrategy.for_mach(5.001)
        assert len(s.stages) == 2

    def test_mach_8_two_stage(self):
        s = ConvergenceStrategy.for_mach(8.0)
        assert len(s.stages) == 2

    def test_mach_10_boundary_two(self):
        s = ConvergenceStrategy.for_mach(10.0)
        assert len(s.stages) == 2

    def test_two_stage_first_mach_is_5(self):
        s = ConvergenceStrategy.for_mach(8.0)
        assert s.stages[0].mach == pytest.approx(5.0)

    def test_two_stage_last_mach_is_target(self):
        s = ConvergenceStrategy.for_mach(8.0)
        assert s.stages[1].mach == pytest.approx(8.0)

    def test_two_stage_mach_monotonic(self):
        s = ConvergenceStrategy.for_mach(7.5)
        machs = [st.mach for st in s.stages]
        assert machs == sorted(machs)

    # --- Three stages (M > 10) ---

    def test_mach_10_001_three_stage(self):
        s = ConvergenceStrategy.for_mach(10.001)
        assert len(s.stages) == 3

    def test_mach_15_6_three_stage(self):
        s = ConvergenceStrategy.for_mach(15.6)
        assert len(s.stages) == 3

    def test_mach_25_three_stage(self):
        s = ConvergenceStrategy.for_mach(25.0)
        assert len(s.stages) == 3

    def test_three_stage_first_mach_is_5(self):
        s = ConvergenceStrategy.for_mach(15.6)
        assert s.stages[0].mach == pytest.approx(5.0)

    def test_three_stage_second_mach_is_10(self):
        s = ConvergenceStrategy.for_mach(15.6)
        assert s.stages[1].mach == pytest.approx(10.0)

    def test_three_stage_last_mach_is_target(self):
        s = ConvergenceStrategy.for_mach(15.6)
        assert s.stages[2].mach == pytest.approx(15.6)

    # --- Invariants across all Mach ranges ---

    def test_stage_mach_progression(self):
        s = ConvergenceStrategy.for_mach(15.6)
        machs = [st.mach for st in s.stages]
        assert machs == sorted(machs)
        assert machs[-1] == pytest.approx(15.6)

    def test_all_stages_first_order(self):
        s = ConvergenceStrategy.for_mach(15.6)
        for st in s.stages:
            assert st.muscl is False

    def test_all_stages_bcgstab(self):
        s = ConvergenceStrategy.for_mach(15.6)
        for st in s.stages:
            assert st.linear_solver == "BCGSTAB"

    def test_stage_cfl_in_valid_range(self):
        s = ConvergenceStrategy.for_mach(15.6)
        for st in s.stages:
            assert 0.001 <= st.cfl <= 0.1

    def test_stage_iterations_positive(self):
        s = ConvergenceStrategy.for_mach(15.6)
        for st in s.stages:
            assert st.iterations > 0

    def test_stage_names_non_empty(self):
        s = ConvergenceStrategy.for_mach(15.6)
        for st in s.stages:
            assert len(st.name) > 0


# ---------------------------------------------------------------------------
# ConvergenceStrategy.mach_ramp() factory
# ---------------------------------------------------------------------------

class TestConvergenceStrategyMachRamp:
    """Tests for ConvergenceStrategy.mach_ramp() factory method."""

    def test_ramp_3_stages(self):
        s = ConvergenceStrategy.mach_ramp(15.0, n_stages=3)
        assert len(s.stages) == 3

    def test_ramp_2_stages_minimum(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=1)
        assert len(s.stages) == 2

    def test_ramp_4_stages(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=4)
        assert len(s.stages) == 4

    def test_ramp_first_stage_starts_at_m2(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=4)
        assert s.stages[0].mach == pytest.approx(2.0)

    def test_ramp_last_stage_is_target(self):
        s = ConvergenceStrategy.mach_ramp(15.0, n_stages=3)
        assert s.stages[-1].mach == pytest.approx(15.0)

    def test_ramp_mach_increases(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=5)
        machs = [st.mach for st in s.stages]
        assert machs == sorted(machs)

    def test_ramp_even_spacing(self):
        s = ConvergenceStrategy.mach_ramp(14.0, n_stages=3)
        # Stages: M=2, M=8, M=14 (evenly spaced from 2 to 14)
        assert s.stages[0].mach == pytest.approx(2.0)
        assert s.stages[1].mach == pytest.approx(8.0)
        assert s.stages[2].mach == pytest.approx(14.0)

    def test_ramp_last_stage_more_iterations(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=3)
        assert s.stages[-1].iterations > s.stages[0].iterations

    def test_ramp_last_stage_tighter_tolerance(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=3)
        assert s.stages[-1].linear_solver_error < s.stages[0].linear_solver_error

    def test_ramp_last_stage_more_linear_iterations(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=3)
        assert s.stages[-1].linear_solver_iter > s.stages[0].linear_solver_iter

    def test_ramp_all_first_order(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=5)
        for st in s.stages:
            assert st.muscl is False

    def test_ramp_all_bcgstab(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=5)
        for st in s.stages:
            assert st.linear_solver == "BCGSTAB"

    def test_ramp_cfl_positive(self):
        s = ConvergenceStrategy.mach_ramp(20.0, n_stages=5)
        for st in s.stages:
            assert st.cfl > 0

    def test_ramp_stage_names_contain_mach(self):
        s = ConvergenceStrategy.mach_ramp(10.0, n_stages=2)
        for st in s.stages:
            assert "M=" in st.name

    def test_ramp_very_large_target(self):
        s = ConvergenceStrategy.mach_ramp(30.0, n_stages=3)
        assert len(s.stages) == 3
        assert s.stages[-1].mach == pytest.approx(30.0)

    def test_ramp_small_target(self):
        s = ConvergenceStrategy.mach_ramp(3.0, n_stages=2)
        assert len(s.stages) == 2
        assert s.stages[0].mach == pytest.approx(2.0)
        assert s.stages[1].mach == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestConvergenceEdgeCases:
    """Edge cases for convergence strategy."""

    def test_negative_mach_single_stage(self):
        s = ConvergenceStrategy.for_mach(-1.0)
        assert len(s.stages) == 1

    def test_zero_mach_single_stage(self):
        s = ConvergenceStrategy.for_mach(0.0)
        assert len(s.stages) == 1

    def test_very_high_mach_three_stage(self):
        s = ConvergenceStrategy.for_mach(100.0)
        assert len(s.stages) == 3
        assert s.stages[-1].mach == pytest.approx(100.0)

    def test_exact_boundary_mach_5(self):
        s = ConvergenceStrategy.for_mach(5.0)
        assert len(s.stages) == 1
        # Just above boundary
        s2 = ConvergenceStrategy.for_mach(5.0 + 1e-9)
        assert len(s2.stages) == 2

    def test_exact_boundary_mach_10(self):
        s = ConvergenceStrategy.for_mach(10.0)
        assert len(s.stages) == 2
        # Just above boundary
        s2 = ConvergenceStrategy.for_mach(10.0 + 1e-9)
        assert len(s2.stages) == 3
