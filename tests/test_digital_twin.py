"""Tests for the imperfect digital twin — scientific behaviour checks."""
import numpy as np
import pytest
from config import SimConfig
from digital_twin import DigitalTwin
from network import create_network


def _mk_twin(delay=5, sinr_noise=3.0, pos_noise=5.0, seed=42):
    cfg = SimConfig(seed=seed, n_slots=60, n_monte_carlo=1,
                    n_users=8, n_targets=3, n_bs=2,
                    twin_sync_delay_slots=delay,
                    twin_sinr_noise_std=sinr_noise,
                    twin_pos_noise_std=pos_noise)
    rng = np.random.default_rng(seed)
    _, users, targets = create_network(cfg, rng)
    twin = DigitalTwin(cfg, rng)
    twin.init(users, targets)
    return cfg, rng, twin, users, targets


def test_twin_initialisation():
    _, _, twin, _, _ = _mk_twin()
    assert twin is not None
    assert len(twin.user_st) > 0


def test_twin_fidelity_in_range():
    cfg, _, twin, users, targets = _mk_twin()
    for slot in range(10):
        for u in users:
            u.move(cfg.slot_duration)
        twin.push(slot, users, targets)
        twin.update(slot, users, targets)
    assert 0.0 <= twin.twin_fidelity <= 1.0


def test_higher_twin_delay_does_not_improve_mismatch():
    """Scientific: increasing sync delay must not reduce mismatch for
    a moving ground truth — we accept equality, forbid strict improvement."""
    mismatches = []
    for delay in (1, 40):
        cfg, _, twin, users, targets = _mk_twin(delay=delay, seed=123)
        for slot in range(40):
            for u in users:
                u.move(cfg.slot_duration)
            for t in targets:
                t.move(cfg.slot_duration)
            twin.push(slot, users, targets)
            twin.update(slot, users, targets)
        mismatches.append(twin.twin_mismatch_mean)
    assert mismatches[1] >= mismatches[0] - 1e-6, (
        f"Expected larger or equal mismatch at high delay, "
        f"got low={mismatches[0]:.4f}, high={mismatches[1]:.4f}")


def test_stale_state_penalty_monotone_in_delay():
    a_cfg, _, twin_a, users_a, targets_a = _mk_twin(delay=1, seed=7)
    b_cfg, _, twin_b, users_b, targets_b = _mk_twin(delay=40, seed=7)
    for slot in range(50):
        for u in users_a:
            u.move(a_cfg.slot_duration)
        for u in users_b:
            u.move(b_cfg.slot_duration)
        twin_a.push(slot, users_a, targets_a)
        twin_a.update(slot, users_a, targets_a)
        twin_b.push(slot, users_b, targets_b)
        twin_b.update(slot, users_b, targets_b)
    assert twin_b.stale_state_penalty >= twin_a.stale_state_penalty


def test_twin_staleness_decay_valid():
    cfg = SimConfig()
    assert 0 < cfg.twin_state_decay < 1
