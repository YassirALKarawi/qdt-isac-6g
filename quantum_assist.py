"""
Quantum-Assisted Candidate Screening for ISAC Resource Allocation.
Realistic: Grover-inspired search with decoherence penalty.
Key: generates BETTER candidates using twin info, then quantum search finds best.
"""
import numpy as np
from typing import List, Dict, Optional
from config import SimConfig
from digital_twin import DigitalTwin


class Candidate:
    __slots__ = ['cid', 'rb_alloc', 'sense_frac', 'score']
    def __init__(self, cid):
        self.cid = cid
        self.rb_alloc: Dict[int, int] = {}
        self.sense_frac: float = 0.2
        self.score: float = 0.0


class QuantumAssist:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_cand = cfg.qa_n_candidates
        self.enabled = cfg.qa_enabled
        self.q_evals = 0
        self.c_evals = 0

    def generate(self, n_users: int, n_rbs: int, twin: DigitalTwin,
                 user_ids: list = None) -> List[Candidate]:
        """Generate diverse candidates. user_ids = actual UIDs served by this BS."""
        if user_ids is None:
            user_ids = list(range(n_users))
        cands = []
        est_sinrs = {}
        est_conf = {}
        for i, uid in enumerate(user_ids):
            s = twin.user_st.get(uid)
            est_sinrs[i] = s.est_sinr if s else 0.0
            est_conf[i] = s.confidence if s else 0.5

        n = len(user_ids)
        base_per_user = max(1, n_rbs // max(n, 1))

        for cid in range(self.n_cand):
            c = Candidate(cid)
            c.sense_frac = 0.10 + (cid / self.n_cand) * 0.20

            strat = cid % 5
            if strat == 0:
                for i, uid in enumerate(user_ids): c.rb_alloc[uid] = base_per_user
            elif strat == 1:
                avg_sinr = np.mean(list(est_sinrs.values()))
                for i, uid in enumerate(user_ids):
                    delta = -1 if est_sinrs[i] > avg_sinr+3 else (1 if est_sinrs[i] < avg_sinr-3 else 0)
                    c.rb_alloc[uid] = max(1, base_per_user + delta)
            elif strat == 2:
                ranked = sorted(range(n), key=lambda x: est_sinrs[x], reverse=True)
                top_n = max(1, n // 4)
                for i, uid in enumerate(user_ids):
                    bonus = 2 if i in ranked[:top_n] else 0
                    c.rb_alloc[uid] = max(1, base_per_user + bonus)
            elif strat == 3:
                for i, uid in enumerate(user_ids):
                    conf = est_conf[i]
                    delta = 1 if conf > 0.7 else (-1 if conf < 0.3 else 0)
                    c.rb_alloc[uid] = max(1, base_per_user + delta)
            else:
                for i, uid in enumerate(user_ids):
                    jitter = self.rng.integers(-1, 2)
                    c.rb_alloc[uid] = max(1, base_per_user + jitter)

            cands.append(c)
        return cands

    def score_candidate(self, c: Candidate, twin: DigitalTwin,
                        trust: Dict[str, float],
                        w_c: float, w_s: float, w_sec: float, w_e: float) -> float:
        """Score a candidate using twin state and trust."""
        comm_u = 0.0
        for uid, nrb in c.rb_alloc.items():
            s = twin.user_st.get(uid)
            if s:
                sinr_lin = 10**(s.est_sinr / 10)
                se = np.log2(1 + max(sinr_lin, 1e-6))
                t = trust.get(f"u{uid}", 1.0)
                comm_u += se * nrb * t * s.confidence
        sense_u = c.sense_frac * 10
        sec_u = sum(trust.get(f"u{uid}", 1.0) for uid in c.rb_alloc)
        energy_u = -0.005 * sum(c.rb_alloc.values())
        return w_c*comm_u + w_s*sense_u + w_sec*sec_u + w_e*energy_u

    def search(self, cands: List[Candidate], twin: DigitalTwin,
               trust: Dict[str, float]) -> Candidate:
        """Quantum-amplified search: finds best candidate faster."""
        w_c, w_s, w_sec, w_e = (self.cfg.weight_comm, self.cfg.weight_sense,
                                 self.cfg.weight_sec, self.cfg.weight_energy)
        for c in cands:
            c.score = self.score_candidate(c, twin, trust, w_c, w_s, w_sec, w_e)

        if not self.enabled:
            self.c_evals += len(cands)
            return max(cands, key=lambda x: x.score)

        # Grover-inspired: with probability p_success, find global best
        N = len(cands)
        n_layers = int(np.ceil(np.sqrt(N)))
        gate_time = 0.05  # us per gate
        circ_time = n_layers * N * gate_time
        decay = np.exp(-circ_time / self.cfg.qa_coherence_us)
        p_success = self.cfg.qa_gate_fidelity**n_layers * decay

        scores = np.array([c.score for c in cands])
        ranked = np.argsort(scores)[::-1]

        if self.rng.random() < p_success:
            best_idx = ranked[0]
        else:
            # Near-optimal fallback
            rank = min(self.rng.geometric(0.6), N-1)
            best_idx = ranked[rank]

        self.q_evals += int(np.sqrt(N) * self.cfg.qa_speedup)
        return cands[best_idx]

    def reset(self):
        self.q_evals = 0; self.c_evals = 0
