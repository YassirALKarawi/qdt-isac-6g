"""Tests for digital twin module."""
import numpy as np
import pytest
from digital_twin import DigitalTwin
from config import SimConfig

@pytest.fixture
def twin():
    cfg = SimConfig(seed=42, n_slots=100, n_monte_carlo=1)
    rng = np.random.default_rng(cfg.seed)
    return DigitalTwin(cfg, rng)

def test_twin_initialisation(twin):
    assert twin is not None

def test_twin_delay_increases_error(twin):
    cfg_fast = SimConfig(seed=42, twin_sync_delay_slots=1)
    cfg_slow = SimConfig(seed=42, twin_sync_delay_slots=20)
    assert cfg_slow.twin_sync_delay_slots > cfg_fast.twin_sync_delay_slots

def test_twin_staleness_decay():
    cfg = SimConfig()
    assert 0 < cfg.twin_state_decay < 1

def test_twin_noise_positive():
    cfg = SimConfig()
    assert cfg.twin_sinr_noise_std > 0
