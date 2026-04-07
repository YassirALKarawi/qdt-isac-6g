"""Tests for security module."""
import numpy as np
import pytest
from security import SecurityModel
from config import SimConfig

@pytest.fixture
def sec():
    cfg = SimConfig(seed=42)
    rng = np.random.default_rng(cfg.seed)
    return SecurityModel(cfg, rng)

def test_security_initialisation(sec):
    assert sec is not None

def test_trust_init_is_one():
    cfg = SimConfig()
    assert cfg.trust_init == 1.0

def test_trust_decay_under_anomaly():
    cfg = SimConfig()
    assert cfg.trust_decay_rate > 0
    assert cfg.trust_recovery_rate > 0

def test_anomaly_prob_in_range():
    cfg = SimConfig()
    assert 0 <= cfg.anomaly_prob <= 1
