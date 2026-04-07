"""
Imperfect Digital Twin: sync delay, noisy estimates, staleness, actual error tracking.
"""
import numpy as np
from typing import List, Optional, Dict
from collections import deque
from config import SimConfig
from network import MobileUser, SensingTarget, Position


class TwinState:
    __slots__ = ['est_pos', 'est_sinr', 'last_update', 'staleness', 'confidence']
    def __init__(self):
        self.est_pos: Optional[Position] = None
        self.est_sinr: float = 0.0
        self.last_update: int = -1
        self.staleness: float = 0.0
        self.confidence: float = 1.0


class DigitalTwin:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.user_st: Dict[int, TwinState] = {}
        self.target_st: Dict[int, TwinState] = {}
        self._buf: deque = deque()
        self.twin_error: float = 0.0   # normalised aggregate

    def init(self, users: List[MobileUser], targets: List[SensingTarget]):
        ns = self.cfg.twin_pos_noise_std
        for u in users:
            s = TwinState()
            s.est_pos = Position(u.pos.x + self.rng.normal(0, ns),
                                  u.pos.y + self.rng.normal(0, ns))
            s.est_sinr = u.sinr_db + self.rng.normal(0, self.cfg.twin_sinr_noise_std)
            s.last_update = 0; s.confidence = 0.9
            self.user_st[u.user_id] = s
        for t in targets:
            s = TwinState()
            s.est_pos = Position(t.pos.x + self.rng.normal(0, ns*1.5),
                                  t.pos.y + self.rng.normal(0, ns*1.5))
            s.est_sinr = t.sensing_snr_db + self.rng.normal(0, self.cfg.twin_sinr_noise_std)
            s.last_update = 0; s.confidence = 0.8
            self.target_st[t.target_id] = s

    def push(self, slot: int, users: List[MobileUser], targets: List[SensingTarget]):
        self._buf.append({
            'slot': slot,
            'u': {u.user_id: (u.pos.copy(), u.sinr_db) for u in users},
            't': {t.target_id: (t.pos.copy(), t.sensing_snr_db) for t in targets}
        })

    def update(self, slot: int, users: List[MobileUser], targets: List[SensingTarget]):
        delay = self.cfg.twin_sync_delay_slots
        # consume delayed observations
        while self._buf and slot - self._buf[0]['slot'] >= delay:
            obs = self._buf.popleft()
            self._apply(obs, slot)
        # Confidence = f(freshness, position_accuracy, SINR_accuracy)
        for u in users:
            s = self.user_st.get(u.user_id)
            if not s or not s.est_pos: continue
            age = slot - s.last_update
            s.staleness = 1.0 - self.cfg.twin_state_decay ** age
            fresh = 1.0 - s.staleness
            p_err = u.pos.distance_to(s.est_pos)
            p_acc = max(0.0, 1.0 - p_err / 100.0)
            s_err = abs(u.sinr_db - s.est_sinr)
            s_acc = max(0.0, 1.0 - s_err / 30.0)
            s.confidence = max(0.05, fresh*0.4 + p_acc*0.35 + s_acc*0.25)
        for t in targets:
            s = self.target_st.get(t.target_id)
            if not s or not s.est_pos: continue
            age = slot - s.last_update
            s.staleness = 1.0 - self.cfg.twin_state_decay ** age
            fresh = 1.0 - s.staleness
            p_err = t.pos.distance_to(s.est_pos)
            p_acc = max(0.0, 1.0 - p_err / 150.0)
            s.confidence = max(0.05, fresh*0.5 + p_acc*0.5)
        # compute actual twin error from ground truth
        self.twin_error = self._error(users, targets)

    def _apply(self, obs, slot):
        ns = self.cfg.twin_pos_noise_std
        sn = self.cfg.twin_sinr_noise_std
        for uid, (pos, sinr) in obs['u'].items():
            if uid not in self.user_st:
                self.user_st[uid] = TwinState()
            s = self.user_st[uid]
            s.est_pos = Position(pos.x + self.rng.normal(0, ns),
                                  pos.y + self.rng.normal(0, ns))
            s.est_sinr = sinr + self.rng.normal(0, sn)
            s.last_update = slot
            s.staleness = 0.0
            # confidence will be recomputed in update() from actual accuracy
        for tid, (pos, snr) in obs['t'].items():
            if tid not in self.target_st:
                self.target_st[tid] = TwinState()
            s = self.target_st[tid]
            s.est_pos = Position(pos.x + self.rng.normal(0, ns*1.5),
                                  pos.y + self.rng.normal(0, ns*1.5))
            s.est_sinr = snr + self.rng.normal(0, sn)
            s.last_update = slot
            s.staleness = 0.0

    def _error(self, users, targets) -> float:
        """Normalised position + SINR error vs ground truth."""
        errs = []
        for u in users:
            s = self.user_st.get(u.user_id)
            if s and s.est_pos:
                pos_err = u.pos.distance_to(s.est_pos) / self.cfg.area_size
                sinr_err = abs(u.sinr_db - s.est_sinr) / 40.0
                errs.append(0.7*pos_err + 0.3*sinr_err)
        for t in targets:
            s = self.target_st.get(t.target_id)
            if s and s.est_pos:
                pos_err = t.pos.distance_to(s.est_pos) / self.cfg.area_size
                errs.append(pos_err)
        return np.mean(errs) if errs else 0.0

    def avg_confidence(self):
        all_c = [s.confidence for s in self.user_st.values()]
        all_c += [s.confidence for s in self.target_st.values()]
        return np.mean(all_c) if all_c else 0.5

    def reset(self):
        self.user_st.clear(); self.target_st.clear()
        self._buf.clear(); self.twin_error = 0.0
