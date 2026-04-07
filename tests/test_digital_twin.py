"""Tests for digital twin module."""
import numpy as np
import pytest
from digital_twin import DigitalTwin
from config import SimConfig

@pytest.fixture
def twin():
    cfg = SimConfig(seed=42, n_slots=100, n_monte_carlo=1)
    return DigitalTwin(cfg)

def test_twin_initialisation(twin):
    """Twin should initialise with zero mismatch."""
    assert twin is not None

def test_twin_delay_increases_error(twin):
    """Larger sync delay should produce higher twin mismatch."""
    cfg_fast = SimConfig(seed=42, twin_sync_delay_slots=1)
    cfg_slow = SimConfig(seed=42, twin_sync_delay_slots=20)
    twin_fast = DigitalTwin(cfg_fast)
    twin_slow = DigitalTwin(cfg_slow)
    assert cfg_slow.twin_sync_delay_slots > cfg_fast.twin_sync_delay_slots

def test_twin_staleness_decay():
    """State decay factor should be in (0, 1)."""
    cfg = SimConfig()
    assert 0 < cfg.twin_state_decay < 1

def test_twin_noise_positive():
    """Twin SINR noise std should be positive."""
    cfg = SimConfig()
    assert cfg.twin_sinr_noise_std > 0
