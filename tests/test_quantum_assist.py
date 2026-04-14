"""Quantum-assisted screening tests — scientific behaviour checks."""
import numpy as np
import pytest

from config import SimConfig
from digital_twin import DigitalTwin
from network import create_network
from quantum_assist import QuantumAssist


def _build(n_candidates=32, shortlist=6, enabled=True, seed=42):
    cfg = SimConfig(seed=seed, n_users=6, n_targets=3, n_bs=2,
                    n_slots=60, n_monte_carlo=1,
                    qa_n_candidates=n_candidates,
                    qa_shortlist_size=shortlist,
                    qa_enabled=enabled)
    rng = np.random.default_rng(seed)
    _, users, targets = create_network(cfg, rng)
    twin = DigitalTwin(cfg, rng)
    twin.init(users, targets)
    qa = QuantumAssist(cfg, rng)
    return cfg, rng, qa, twin, users, targets


def test_screening_reduces_full_evaluations():
    """Scientific: screening must reduce the number of full candidate
    evaluations compared to exhaustive (classical) search."""
    cfg_s, _, qa_s, twin, users, _ = _build(n_candidates=32, shortlist=6,
                                              enabled=True)
    trust = {f"u{u.user_id}": 1.0 for u in users}
    for _ in range(20):
        cands = qa_s.generate(len(users), cfg_s.n_resource_blocks,
                                twin, [u.user_id for u in users])
        qa_s.search(cands, twin, trust)
    total_screened = qa_s.q_evals + qa_s.c_evals

    cfg_c, _, qa_c, twin2, users2, _ = _build(n_candidates=32, shortlist=6,
                                                enabled=False, seed=42)
    trust2 = {f"u{u.user_id}": 1.0 for u in users2}
    for _ in range(20):
        cands = qa_c.generate(len(users2), cfg_c.n_resource_blocks,
                                twin2, [u.user_id for u in users2])
        qa_c.search(cands, twin2, trust2)
    total_classical = qa_c.q_evals + qa_c.c_evals

    assert total_screened < total_classical, (
        f"Expected screened evals < classical; got {total_screened} "
        f"vs {total_classical}")


def test_candidate_reduction_ratio_below_one_when_enabled():
    cfg, _, qa, twin, users, _ = _build(n_candidates=40, shortlist=5,
                                          enabled=True)
    trust = {f"u{u.user_id}": 1.0 for u in users}
    for _ in range(10):
        cands = qa.generate(len(users), cfg.n_resource_blocks,
                              twin, [u.user_id for u in users])
        qa.search(cands, twin, trust)
    assert qa.candidate_reduction_ratio() < 1.0
    assert qa.search_cost_reduction() > 0.0


def test_rank_percentile_in_unit_range():
    cfg, _, qa, twin, users, _ = _build(enabled=True)
    trust = {f"u{u.user_id}": 1.0 for u in users}
    for _ in range(10):
        cands = qa.generate(len(users), cfg.n_resource_blocks,
                              twin, [u.user_id for u in users])
        qa.search(cands, twin, trust)
    assert 0.0 <= qa.selected_action_rank_percentile() <= 1.0


def test_gate_fidelity_valid():
    cfg = SimConfig()
    assert 0.9 < cfg.qa_gate_fidelity <= 1.0
