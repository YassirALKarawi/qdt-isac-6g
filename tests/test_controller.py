"""
Controller integration tests — verifies scientific behaviour across the
full baseline set (BL0-7) and the ablation mode (BL=-1).
"""
import pytest

from config import SimConfig
from simulator import run_one
from metrics import MetricsCollector


def _run_baseline(bl_id: int, seed: int = 99, **overrides) -> dict:
    cfg = SimConfig(seed=seed, n_monte_carlo=1, n_slots=200,
                    n_users=10, n_targets=3, n_bs=2, **overrides)
    cfg.baseline_id = bl_id
    mc = MetricsCollector(steady_state_fraction=0.5)
    run_one(cfg, run_id=0, mc=mc)
    return mc.runs[-1]


def test_full_proposed_beats_static_in_utility():
    bl0 = _run_baseline(0)
    bl4 = _run_baseline(4)
    assert bl4["utility_mean"] > bl0["utility_mean"], (
        f"BL4 utility {bl4['utility_mean']:.4f} should exceed "
        f"BL0 utility {bl0['utility_mean']:.4f}")


def test_full_proposed_has_reasonable_energy():
    bl0 = _run_baseline(0)
    bl4 = _run_baseline(4)
    assert bl4["energy_norm_mean"] <= bl0["energy_norm_mean"] + 0.02


def test_dt_methods_have_nonzero_twin_error():
    for bl in (2, 4):
        r = _run_baseline(bl)
        assert r["twin_err_mean"] > 0.0


def test_security_aware_methods_have_detection_activity():
    r = _run_baseline(4)
    assert r["det_rate_mean"] > 0.0 or r["fa_rate_mean"] > 0.0


def test_predictor_and_robust_baselines_run():
    for bl in (5, 6, 7):
        r = _run_baseline(bl)
        assert r["sum_rate_mean"] >= 0.0


def test_ablation_mode_with_full_flags_runs():
    cfg = SimConfig(seed=99, n_monte_carlo=1, n_slots=200,
                    n_users=10, n_targets=3, n_bs=2)
    cfg.baseline_id = -1
    mc = MetricsCollector(steady_state_fraction=0.5)
    run_one(cfg, run_id=0, mc=mc)
    r = mc.runs[-1]
    assert r["utility_mean"] > -1.0


def test_ablation_no_dt_zeros_twin_error():
    cfg = SimConfig(seed=99, n_monte_carlo=1, n_slots=200,
                    n_users=10, n_targets=3, n_bs=2)
    cfg.baseline_id = -1
    cfg.use_twin = False
    cfg.use_trust_gating = False
    cfg.use_screening = False
    cfg.use_adaptive_weights = False
    cfg.use_mismatch_comp = False
    mc = MetricsCollector(steady_state_fraction=0.5)
    run_one(cfg, run_id=0, mc=mc)
    r = mc.runs[-1]
    assert r["twin_err_mean"] == 0.0
