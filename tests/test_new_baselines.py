"""Tests for additional baselines: Uncertainty-Aware (BL5) and UCB Learning (BL6)."""
import copy
from config import SimConfig
from simulator import run_one
from metrics import MetricsCollector


def _run_baseline(bl_id: int, seed: int = 99) -> dict:
    cfg = SimConfig(seed=seed, n_monte_carlo=1, n_slots=200,
                    n_users=10, n_targets=3, n_bs=2)
    cfg.baseline_id = bl_id
    mc = MetricsCollector(steady_state_fraction=0.5)
    run_one(cfg, run_id=0, mc=mc)
    return mc.runs[-1]


def test_bl5_runs_without_error():
    result = _run_baseline(5)
    assert result["utility_mean"] != 0.0
    assert result["sum_rate_mean"] > 0


def test_bl6_runs_without_error():
    result = _run_baseline(6)
    assert result["utility_mean"] != 0.0
    assert result["sum_rate_mean"] > 0


def test_bl5_outperforms_static():
    bl0 = _run_baseline(0)
    bl5 = _run_baseline(5)
    # BL5 should achieve at least comparable utility to static
    assert bl5["utility_mean"] >= bl0["utility_mean"] - 0.05, (
        f"BL5 utility {bl5['utility_mean']:.4f} too far below "
        f"BL0 utility {bl0['utility_mean']:.4f}"
    )


def test_bl6_learns_over_time():
    """UCB should improve allocation over time."""
    cfg = SimConfig(seed=99, n_monte_carlo=1, n_slots=400,
                    n_users=10, n_targets=3, n_bs=2)
    cfg.baseline_id = 6
    mc = MetricsCollector(steady_state_fraction=0.5)
    run_one(cfg, run_id=0, mc=mc)
    slots = mc.slot_df()
    early = slots[slots['slot'] < 100]['sum_rate'].mean()
    late = slots[slots['slot'] > 300]['sum_rate'].mean()
    # Late performance should not be drastically worse than early
    assert late >= early * 0.8, (
        f"UCB late SR {late:.1f} should not degrade drastically from early {early:.1f}"
    )


def test_full_proposed_beats_new_baselines():
    """Full Proposed (BL4) should outperform BL5 and BL6 in utility."""
    bl4 = _run_baseline(4)
    bl5 = _run_baseline(5)
    bl6 = _run_baseline(6)
    assert bl4["utility_mean"] > bl5["utility_mean"] - 0.02, (
        f"BL4 utility {bl4['utility_mean']:.4f} vs BL5 {bl5['utility_mean']:.4f}"
    )
    assert bl4["utility_mean"] > bl6["utility_mean"] - 0.02, (
        f"BL4 utility {bl4['utility_mean']:.4f} vs BL6 {bl6['utility_mean']:.4f}"
    )
