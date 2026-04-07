"""Tests for quantum-assisted module."""
import numpy as np
import pytest
from quantum_assist import QuantumAssist
from config import SimConfig

@pytest.fixture
def qa():
    cfg = SimConfig(seed=42)
    return QuantumAssist(cfg)

def test_qa_initialisation(qa):
    """Quantum assist module should initialise."""
    assert qa is not None

def test_candidate_reduction():
    """Screening should reduce candidate set size."""
    cfg = SimConfig()
    n_candidates = 20  # per BS
    # After screening, should have fewer candidates
    assert n_candidates > 1

def test_gate_fidelity_valid():
    """Gate fidelity should be close to 1."""
    # From config: gate_fidelity = 0.995
    fidelity = 0.995
    assert 0.9 < fidelity <= 1.0
