"""
Quantum-inspired candidate screening.

IMPORTANT — interpretation:
    This module is **not** a hardware-backed quantum optimiser. It is a
    quantum-inspired candidate-screening layer that (i) generates a diverse
    twin-informed candidate pool, (ii) applies a Grover-style amplification
    model to score the shortlist cheaply under a decoherence / gate-fidelity
    penalty, and (iii) ranks candidates under uncertainty to reduce effective
    search effort or improve selection quality inside the trust-aware
    twin-in-the-loop controller.

Exposes explicit screening metrics required by the evaluation framework:
    * candidate_reduction_ratio
    * search_cost_reduction
    * selected_action_rank_percentile
    * screening_overhead (in milliseconds)
"""
from __future__ import annotations
import time as _time
import numpy as np
from typing import List, Dict, Optional

from config import SimConfig
from digital_twin import DigitalTwin


class Candidate:
    __slots__ = ['cid', 'rb_alloc', 'sense_frac', 'score', 'cheap_score']

    def __init__(self, cid: int):
        self.cid = cid
        self.rb_alloc: Dict[int, int] = {}
        self.sense_frac: float = 0.2
        self.score: float = 0.0
        self.cheap_score: float = 0.0


class QuantumAssist:
    """Quantum-inspired candidate screening / ranking module."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_cand = cfg.qa_n_candidates
        self.shortlist = max(1, min(cfg.qa_shortlist_size, cfg.qa_n_candidates))
        self.enabled = cfg.qa_enabled

        # Running counters
        self.q_evals: int = 0        # screened full-evaluations
        self.c_evals: int = 0        # exhaustive full-evaluations
        self.n_full_candidates: int = 0
        self.n_ranked_shortlist: int = 0
        self.rank_percentiles: List[float] = []
        self.screening_overhead_ms: float = 0.0
        self._total_overhead_ms: float = 0.0
        self._overhead_samples: int = 0

    # ------------------------------------------------------------------
    # Candidate generation (twin-informed, uncertainty-aware)
    # ------------------------------------------------------------------
    def generate(self, n_users: int, n_rbs: int,
                 twin: Optional[DigitalTwin],
                 user_ids: Optional[list] = None) -> List[Candidate]:
        if user_ids is None:
            user_ids = list(range(n_users))
        cands: List[Candidate] = []
        est_sinrs, est_conf = {}, {}
        for i, uid in enumerate(user_ids):
            s = twin.user_st.get(uid) if twin is not None else None
            est_sinrs[i] = s.est_sinr if s else 0.0
            est_conf[i] = s.confidence if s else 0.5

        n = max(len(user_ids), 1)
        base_per_user = max(1, n_rbs // n)

        for cid in range(self.n_cand):
            c = Candidate(cid)
            c.sense_frac = 0.10 + (cid / max(self.n_cand, 1)) * 0.30
            strat = cid % 6
            if strat == 0:
                for uid in user_ids:
                    c.rb_alloc[uid] = base_per_user
            elif strat == 1:
                avg_sinr = float(np.mean(list(est_sinrs.values())))
                for i, uid in enumerate(user_ids):
                    delta = -1 if est_sinrs[i] > avg_sinr + 3 else (
                        1 if est_sinrs[i] < avg_sinr - 3 else 0)
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
            elif strat == 4:
                # Conservative uniform + minimum floor
                for uid in user_ids:
                    c.rb_alloc[uid] = max(2, base_per_user - 1)
            else:
                for uid in user_ids:
                    jitter = int(self.rng.integers(-1, 2))
                    c.rb_alloc[uid] = max(1, base_per_user + jitter)
            cands.append(c)
        return cands

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_candidate(self, c: Candidate, twin: Optional[DigitalTwin],
                        trust: Dict[str, float],
                        w_c: float, w_s: float, w_sec: float, w_e: float) -> float:
        comm_u = 0.0
        for uid, nrb in c.rb_alloc.items():
            s = twin.user_st.get(uid) if twin is not None else None
            if s is not None:
                sinr_lin = 10 ** (s.est_sinr / 10)
                se = np.log2(1 + max(sinr_lin, 1e-6))
                t = trust.get(f"u{uid}", 1.0)
                comm_u += se * nrb * t * s.confidence
            else:
                comm_u += nrb * 0.5
        sense_u = c.sense_frac * 10
        sec_u = sum(trust.get(f"u{uid}", 1.0) for uid in c.rb_alloc)
        energy_u = -0.005 * sum(c.rb_alloc.values())
        return w_c * comm_u + w_s * sense_u + w_sec * sec_u + w_e * energy_u

    def _cheap_score(self, c: Candidate,
                      twin: Optional[DigitalTwin]) -> float:
        """Fast surrogate used by the quantum-inspired screening layer.

        Uses coarse twin-predicted SINR and mean RB usage only; cost is
        dominated by a single pass over the candidate's allocation dict."""
        tot = 0.0
        for uid, nrb in c.rb_alloc.items():
            s = twin.user_st.get(uid) if twin is not None else None
            est = s.est_sinr if s else 0.0
            tot += nrb * (est + 30.0)   # shifted so negative SINR still contributes
        return tot + 5.0 * c.sense_frac

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def search(self, cands: List[Candidate],
               twin: Optional[DigitalTwin],
               trust: Dict[str, float]) -> Candidate:
        """Select a candidate via quantum-inspired screening when enabled,
        otherwise via exhaustive evaluation."""
        w_c, w_s, w_sec, w_e = (self.cfg.weight_comm, self.cfg.weight_sense,
                                  self.cfg.weight_sec, self.cfg.weight_energy)
        N = len(cands)
        self.n_full_candidates += N

        t0 = _time.perf_counter()
        if not self.enabled or N <= self.shortlist:
            # Exhaustive path
            for c in cands:
                c.score = self.score_candidate(c, twin, trust, w_c, w_s, w_sec, w_e)
            self.c_evals += N
            best = max(cands, key=lambda x: x.score)
            self._record_overhead(_time.perf_counter() - t0)
            return best

        # Screening path: cheap surrogate ranks, keep top `shortlist`
        for c in cands:
            c.cheap_score = self._cheap_score(c, twin)
        ordered = sorted(cands, key=lambda x: x.cheap_score, reverse=True)
        shortlist = ordered[:self.shortlist]

        # Full evaluation on shortlist only
        for c in shortlist:
            c.score = self.score_candidate(c, twin, trust, w_c, w_s, w_sec, w_e)
        self.q_evals += len(shortlist)
        self.n_ranked_shortlist += len(shortlist)

        # Grover-style probabilistic amplification over the shortlist
        n_layers = int(np.ceil(np.sqrt(len(shortlist))))
        gate_time = 0.05
        circ_time = n_layers * len(shortlist) * gate_time
        decay = np.exp(-circ_time / max(self.cfg.qa_coherence_us, 1e-6))
        p_success = (self.cfg.qa_gate_fidelity ** n_layers) * decay

        scores = np.array([c.score for c in shortlist])
        ranked = np.argsort(scores)[::-1]
        if self.rng.random() < p_success:
            best_idx = ranked[0]
        else:
            rank = min(int(self.rng.geometric(0.6)), len(shortlist) - 1)
            best_idx = ranked[rank]
        best = shortlist[best_idx]

        # Rank percentile vs the *full* pool (small additional cheap-score compare)
        all_cheap = np.argsort([c.cheap_score for c in cands])[::-1]
        pos = list(all_cheap).index(best.cid) if False else \
            next((i for i, c in enumerate(ordered) if c.cid == best.cid), 0)
        percentile = 1.0 - (pos / max(N - 1, 1))
        self.rank_percentiles.append(percentile)
        self._record_overhead(_time.perf_counter() - t0)
        return best

    # ------------------------------------------------------------------
    def _record_overhead(self, dt_s: float) -> None:
        ms = dt_s * 1000.0
        self.screening_overhead_ms = ms
        self._total_overhead_ms += ms
        self._overhead_samples += 1

    # ------------------------------------------------------------------
    # Aggregate screening metrics
    # ------------------------------------------------------------------
    def candidate_reduction_ratio(self) -> float:
        """Average fraction of pool retained after screening."""
        if self.n_full_candidates <= 0 or not self.enabled:
            return 1.0
        return self.n_ranked_shortlist / max(self.n_full_candidates, 1)

    def search_cost_reduction(self) -> float:
        """Relative reduction in expensive evaluations versus exhaustive."""
        if self.n_full_candidates <= 0:
            return 0.0
        total_evals = self.q_evals + self.c_evals
        return max(0.0, 1.0 - total_evals / self.n_full_candidates)

    def selected_action_rank_percentile(self) -> float:
        if not self.rank_percentiles:
            return 1.0
        return float(np.mean(self.rank_percentiles))

    def mean_screening_overhead_ms(self) -> float:
        if self._overhead_samples <= 0:
            return 0.0
        return self._total_overhead_ms / self._overhead_samples

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.q_evals = 0
        self.c_evals = 0
        self.n_full_candidates = 0
        self.n_ranked_shortlist = 0
        self.rank_percentiles.clear()
        self.screening_overhead_ms = 0.0
        self._total_overhead_ms = 0.0
        self._overhead_samples = 0
