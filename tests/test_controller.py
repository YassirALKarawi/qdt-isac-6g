"""
Controller integration test.
Verifies that the Full Proposed method (BL4) achieves higher composite
utility than Static ISAC (BL0) under identical conditions.
"""
import copy
from config import SimConfig
from simulator import run_one
from metrics import MetricsCollector


def _run_baseline(bl_id: int, seed: int = 99) -> dict:
    """Run a short simulation and return steady-state summary."""
    cfg = SimConfig(seed=seed, n_monte_carlo=1, n_slots=200,
                    n_users=10, n_targets=3, n_bs=2)
    cfg.baseline_id = bl_id
    mc = MetricsCollector(steady_state_fraction=0.5)
    run_one(cfg, run_id=0, mc=mc)
    return mc.runs[-1]


def test_full_proposed_beats_static_in_utility():
    bl0 = _run_baseline(0)
    bl4 = _run_baseline(4)
    assert bl4["utility_mean"] > bl0["utility_mean"], (
        f"BL4 utility {bl4['utility_mean']:.4f} should exceed "
        f"BL0 utility {bl0['utility_mean']:.4f}"
    )


def test_full_proposed_has_lower_energy():
    bl0 = _run_baseline(0)
    bl4 = _run_baseline(4)
    assert bl4["energy_norm_mean"] <= bl0["energy_norm_mean"] + 0.01, (
        f"BL4 energy {bl4['energy_norm_mean']:.4f} should not exceed "
        f"BL0 energy {bl0['energy_norm_mean']:.4f} significantly"
    )


def test_dt_methods_have_nonzero_twin_error():
    bl2 = _run_baseline(2)
    bl4 = _run_baseline(4)
    assert bl2["twin_err_mean"] > 0.01, "DT method should have measurable twin error"
    assert bl4["twin_err_mean"] > 0.01, "Full Proposed should have measurable twin error"


def test_security_aware_methods_have_detection():
    bl4 = _run_baseline(4)
    # With anomaly_prob=0.08, there should be some detections over 200 slots
    assert bl4["det_rate_mean"] > 0.0 or bl4["fa_rate_mean"] > 0.0, (
        "Security-aware method should have nonzero detection or FA activity"
    )
