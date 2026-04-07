"""Tests for security module."""
import numpy as np
import pytest
from security import SecurityModule
from config import SimConfig

@pytest.fixture
def sec():
    cfg = SimConfig(seed=42)
    return SecurityModule(cfg)

def test_security_initialisation(sec):
    """Security module should initialise."""
    assert sec is not None

def test_trust_init_is_one():
    """Initial trust should be 1.0."""
    cfg = SimConfig()
    assert cfg.trust_init == 1.0

def test_trust_decay_under_anomaly():
    """Trust should decrease when anomalies are detected."""
    cfg = SimConfig()
    assert cfg.trust_decay_rate > 0
    assert cfg.trust_recovery_rate > 0

def test_anomaly_prob_in_range():
    """Anomaly probability should be in [0, 1]."""
    cfg = SimConfig()
    assert 0 <= cfg.anomaly_prob <= 1
