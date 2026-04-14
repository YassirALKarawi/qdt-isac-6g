"""
Imperfect Digital Twin model.

Models four sources of twin imperfection:
  * synchronisation delay (slots)
  * position / SINR measurement noise
  * state staleness (aging of last update)
  * twin fidelity degradation (captured via confidence)

Exposes explicit twin metrics required by the evaluation framework:
  * twin_mismatch_mean, twin_mismatch_std
  * twin_fidelity            (1 - normalised mismatch)
  * stale_state_penalty      (mean staleness across tracked entities)
  * confidence (per-entity)
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import deque

from config import SimConfig
from network import MobileUser, SensingTarget, Position


class TwinState:
    __slots__ = ['est_pos', 'est_sinr', 'last_update',
                 'staleness', 'confidence', 'mismatch']

    def __init__(self):
        self.est_pos: Optional[Position] = None
        self.est_sinr: float = 0.0
        self.last_update: int = -1
        self.staleness: float = 0.0
        self.confidence: float = 1.0
        self.mismatch: float = 0.0   # instantaneous per-entity mismatch


class DigitalTwin:
    """Imperfect digital twin with explicit fidelity / mismatch tracking."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.user_st: Dict[int, TwinState] = {}
        self.target_st: Dict[int, TwinState] = {}
        self._buf: deque = deque()

        # Aggregates maintained per slot
        self.twin_error: float = 0.0         # normalised aggregate
        self.twin_mismatch_mean: float = 0.0
        self.twin_mismatch_std: float = 0.0
        self.twin_fidelity: float = 1.0
        self.stale_state_penalty: float = 0.0

    # ------------------------------------------------------------------
    def init(self, users: List[MobileUser], targets: List[SensingTarget]):
        ns = self.cfg.twin_pos_noise_std
        for u in users:
            s = TwinState()
            s.est_pos = Position(u.pos.x + self.rng.normal(0, ns),
                                 u.pos.y + self.rng.normal(0, ns))
            s.est_sinr = u.sinr_db + self.rng.normal(0, self.cfg.twin_sinr_noise_std)
            s.last_update = 0
            s.confidence = 0.9
            self.user_st[u.user_id] = s
        for t in targets:
            s = TwinState()
            s.est_pos = Position(t.pos.x + self.rng.normal(0, ns * 1.5),
                                 t.pos.y + self.rng.normal(0, ns * 1.5))
            s.est_sinr = t.sensing_snr_db + self.rng.normal(0, self.cfg.twin_sinr_noise_std)
            s.last_update = 0
            s.confidence = 0.8
            self.target_st[t.target_id] = s

    def push(self, slot: int,
             users: List[MobileUser],
             targets: List[SensingTarget]) -> None:
        """Record a ground-truth observation to be consumed after sync delay."""
        self._buf.append({
            'slot': slot,
            'u': {u.user_id: (u.pos.copy(), u.sinr_db) for u in users},
            't': {t.target_id: (t.pos.copy(), t.sensing_snr_db) for t in targets},
        })

    def update(self, slot: int,
               users: List[MobileUser],
               targets: List[SensingTarget]) -> None:
        """Consume buffered observations and recompute confidence / mismatch."""
        delay = max(0, int(self.cfg.twin_sync_delay_slots))
        while self._buf and slot - self._buf[0]['slot'] >= delay:
            self._apply(self._buf.popleft(), slot)

        per_entity: List[float] = []
        stales: List[float] = []

        for u in users:
            s = self.user_st.get(u.user_id)
            if not s or not s.est_pos:
                continue
            age = slot - s.last_update
            s.staleness = 1.0 - self.cfg.twin_state_decay ** max(age, 0)
            fresh = 1.0 - s.staleness
            p_err = u.pos.distance_to(s.est_pos)
            p_acc = max(0.0, 1.0 - p_err / 100.0)
            s_err = abs(u.sinr_db - s.est_sinr)
            s_acc = max(0.0, 1.0 - s_err / 30.0)
            s.confidence = max(0.05, fresh * 0.4 + p_acc * 0.35 + s_acc * 0.25)
            # Normalised instantaneous mismatch
            s.mismatch = 0.7 * (p_err / self.cfg.area_size) + 0.3 * (s_err / 40.0)
            per_entity.append(s.mismatch)
            stales.append(s.staleness)

        for t in targets:
            s = self.target_st.get(t.target_id)
            if not s or not s.est_pos:
                continue
            age = slot - s.last_update
            s.staleness = 1.0 - self.cfg.twin_state_decay ** max(age, 0)
            fresh = 1.0 - s.staleness
            p_err = t.pos.distance_to(s.est_pos)
            p_acc = max(0.0, 1.0 - p_err / 150.0)
            s.confidence = max(0.05, fresh * 0.5 + p_acc * 0.5)
            s.mismatch = p_err / self.cfg.area_size
            per_entity.append(s.mismatch)
            stales.append(s.staleness)

        if per_entity:
            arr = np.array(per_entity, dtype=float)
            self.twin_mismatch_mean = float(arr.mean())
            self.twin_mismatch_std = float(arr.std())
            # Fidelity: 1 - clipped normalised mismatch, floored
            self.twin_fidelity = max(self.cfg.twin_fidelity_floor,
                                     1.0 - min(1.0, self.twin_mismatch_mean * 3.0))
            self.twin_error = float(arr.mean())
        else:
            self.twin_mismatch_mean = 0.0
            self.twin_mismatch_std = 0.0
            self.twin_fidelity = 1.0
            self.twin_error = 0.0

        self.stale_state_penalty = float(np.mean(stales)) if stales else 0.0

    # ------------------------------------------------------------------
    def _apply(self, obs: Dict, slot: int) -> None:
        ns = self.cfg.twin_pos_noise_std
        sn = self.cfg.twin_sinr_noise_std
        for uid, (pos, sinr) in obs['u'].items():
            s = self.user_st.setdefault(uid, TwinState())
            s.est_pos = Position(pos.x + self.rng.normal(0, ns),
                                 pos.y + self.rng.normal(0, ns))
            s.est_sinr = sinr + self.rng.normal(0, sn)
            s.last_update = slot
            s.staleness = 0.0
        for tid, (pos, snr) in obs['t'].items():
            s = self.target_st.setdefault(tid, TwinState())
            s.est_pos = Position(pos.x + self.rng.normal(0, ns * 1.5),
                                 pos.y + self.rng.normal(0, ns * 1.5))
            s.est_sinr = snr + self.rng.normal(0, sn)
            s.last_update = slot
            s.staleness = 0.0

    # ------------------------------------------------------------------
    def avg_confidence(self) -> float:
        all_c = [s.confidence for s in self.user_st.values()]
        all_c += [s.confidence for s in self.target_st.values()]
        return float(np.mean(all_c)) if all_c else 0.5

    def user_mismatch(self, uid: int) -> float:
        s = self.user_st.get(uid)
        return s.mismatch if s else 0.0

    def reset(self) -> None:
        self.user_st.clear()
        self.target_st.clear()
        self._buf.clear()
        self.twin_error = 0.0
        self.twin_mismatch_mean = 0.0
        self.twin_mismatch_std = 0.0
        self.twin_fidelity = 1.0
        self.stale_state_penalty = 0.0
