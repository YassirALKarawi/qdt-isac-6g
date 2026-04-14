"""Security / trust-aware gating tests — scientific behaviour checks."""
import numpy as np
import pytest

from config import SimConfig
from security import SecurityModel
from digital_twin import DigitalTwin
from network import create_network


def _build(seed: int = 42):
    cfg = SimConfig(seed=seed, n_users=6, n_targets=3, n_bs=2,
                    n_slots=80, n_monte_carlo=1, anomaly_prob=0.2)
    rng = np.random.default_rng(seed)
    _, users, targets = create_network(cfg, rng)
    sec = SecurityModel(cfg, rng)
    sec.init(users, targets)
    twin = DigitalTwin(cfg, rng)
    twin.init(users, targets)
    return cfg, rng, sec, twin, users, targets


def test_security_initialisation():
    cfg, _, sec, _, _, _ = _build()
    assert sec is not None
    assert all(abs(v - cfg.trust_init) < 1e-9 for v in sec.trust.values())


def test_trust_decay_under_anomaly_is_configured():
    cfg = SimConfig()
    assert cfg.trust_decay_rate > 0
    assert cfg.trust_recovery_rate > 0
    assert 0 <= cfg.anomaly_prob <= 1


def test_gate_becomes_more_conservative_when_trust_drops():
    """Scientific: trust-aware gating should scale the deployed action
    down as deployment confidence decreases."""
    cfg, _, sec, twin, users, _ = _build()
    uid = users[0].user_id
    # High trust baseline
    sec.trust[f"u{uid}"] = 1.0
    rbs_hi, pw_hi, fb_hi = sec.gate_action(uid, proposed_rbs=30,
                                             proposed_power_scale=1.0,
                                             twin=twin)
    # Low trust
    sec.trust[f"u{uid}"] = 0.2
    rbs_lo, pw_lo, fb_lo = sec.gate_action(uid, proposed_rbs=30,
                                             proposed_power_scale=1.0,
                                             twin=twin)
    assert rbs_lo <= rbs_hi, (rbs_lo, rbs_hi)
    assert pw_lo <= pw_hi + 1e-9, (pw_lo, pw_hi)
    assert fb_lo is True


def test_hard_floor_triggers_fallback():
    cfg, _, sec, twin, users, _ = _build()
    uid = users[0].user_id
    sec.trust[f"u{uid}"] = 0.05  # well below hard floor
    rbs, pw, fb = sec.gate_action(uid, proposed_rbs=40,
                                    proposed_power_scale=1.0, twin=twin)
    assert fb is True
    assert rbs <= 40
    assert sec.fallback_deployment_ratio() > 0.0


def test_unsafe_action_suppression_counter_rises():
    cfg, _, sec, twin, users, _ = _build()
    uid = users[0].user_id
    sec.trust[f"u{uid}"] = 0.1
    # Propose an unsafe (large RB) action
    for _ in range(5):
        sec.gate_action(uid, proposed_rbs=int(cfg.n_resource_blocks * 0.9),
                         proposed_power_scale=1.0, twin=twin)
    assert sec.n_unsafe_suppressed > 0
    assert sec.unsafe_action_suppression_rate() > 0.0
