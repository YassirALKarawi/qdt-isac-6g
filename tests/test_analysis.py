"""Tests for formal analysis: trust bounds, utility bounds, convergence."""
import numpy as np
from config import SimConfig
from analysis import (
    steady_state_trust_bound,
    utility_loss_bound,
    monotonic_degradation_curve,
    trust_gating_conservativeness,
    feedback_loop_stability,
    compute_all_bounds,
)


def test_trust_bound_within_valid_range():
    cfg = SimConfig()
    tb = steady_state_trust_bound(cfg)
    assert 0.0 < tb.tau_lower <= tb.tau_upper <= 1.0
    assert tb.convergence_rate > 0
    assert tb.mixing_time_slots > 0


def test_trust_bound_higher_with_lower_anomaly():
    cfg_low = SimConfig(anomaly_prob=0.02)
    cfg_high = SimConfig(anomaly_prob=0.20)
    tb_low = steady_state_trust_bound(cfg_low)
    tb_high = steady_state_trust_bound(cfg_high)
    assert tb_low.tau_lower > tb_high.tau_lower


def test_utility_loss_increases_with_delay():
    cfg = SimConfig()
    ub_short = utility_loss_bound(cfg, delay_slots=2)
    ub_long = utility_loss_bound(cfg, delay_slots=20)
    assert ub_long.delta_j_upper > ub_short.delta_j_upper


def test_utility_loss_zero_at_zero_delay():
    cfg = SimConfig()
    ub = utility_loss_bound(cfg, delay_slots=0)
    assert abs(ub.delta_j_upper) < 1e-10


def test_monotonic_degradation():
    cfg = SimConfig()
    delays, losses = monotonic_degradation_curve(cfg)
    # Should be monotonically non-decreasing
    diffs = np.diff(losses)
    assert np.all(diffs >= -1e-10), "Degradation curve should be monotonic"


def test_gating_conservativeness():
    trust_vals = np.array([0.9, 0.8, 0.3, 0.2, 0.6, 0.4, 0.1])
    result = trust_gating_conservativeness(trust_vals, threshold=0.5)
    assert 0 <= result['gated_fraction'] <= 1
    assert result['avg_trust_when_gated'] < 0.5


def test_stability_analysis():
    rng = np.random.default_rng(42)
    # Converging utility series
    noise = rng.normal(0, 0.01, 500)
    utility = 0.5 + 0.2 * np.exp(-np.arange(500) / 100) + noise
    result = feedback_loop_stability(utility, window=50)
    assert 'stable' in result
    assert 'variance_trend' in result


def test_compute_all_bounds_structure():
    cfg = SimConfig()
    bounds = compute_all_bounds(cfg)
    assert 'trust_bound' in bounds
    assert 'utility_bound' in bounds
    assert 'degradation_curve' in bounds
    assert len(bounds['degradation_curve']['delays']) > 0
