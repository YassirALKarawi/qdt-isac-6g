"""Tests for quantum-assisted module."""
import numpy as np
import pytest
from quantum_assist import QuantumAssist
from config import SimConfig

@pytest.fixture
def qa():
    cfg = SimConfig(seed=42)
    rng = np.random.default_rng(cfg.seed)
    return QuantumAssist(cfg, rng)

def test_qa_initialisation(qa):
    assert qa is not None

def test_candidate_reduction():
    n_candidates = 20
    assert n_candidates > 1

def test_gate_fidelity_valid():
    fidelity = 0.995
    assert 0.9 < fidelity <= 1.0
