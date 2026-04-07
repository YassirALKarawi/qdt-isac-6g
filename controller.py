"""
Controller: 5 methods with proper differentiation.
  0: Static equal allocation
  1: Adaptive rebalancing from measurements (no twin)
  2: DT-guided PF allocation (no quantum)
  3: DT + quantum candidate search (no security)
  4: Full: DT + quantum + security-aware closed-loop
"""
import numpy as np
import time as _time
from typing import List, Optional, Dict
from config import SimConfig
from network import BaseStation, MobileUser, SensingTarget
from channel import ChannelModel
from communication import CommModel
from sensing import SensingModel
from digital_twin import DigitalTwin
from security import SecurityModel
from quantum_assist import QuantumAssist


class Controller:
    def __init__(self, cfg, ch, comm, sense, twin, sec, qa, rng):
        self.cfg = cfg
        self.ch = ch
        self.comm: CommModel = comm
        self.sense: SensingModel = sense
        self.twin: Optional[DigitalTwin] = twin
        self.sec: Optional[SecurityModel] = sec
        self.qa: Optional[QuantumAssist] = qa
        self.rng = rng
        self.bl = cfg.baseline_id
        self.w_c = cfg.weight_comm
        self.w_s = cfg.weight_sense
        self.w_sec = cfg.weight_sec
        self.w_e = cfg.weight_energy
        self.latency_ms = 0.0

    def step(self, slot, bss, users, targets):
        t0 = _time.perf_counter()
        # ALL baselines face the same attack environment
        if self.sec:
            self.sec.inject(slot, users, targets)
        methods = [self._bl0, self._bl1, self._bl2, self._bl3, self._bl4]
        methods[self.bl](slot, bss, users, targets)
        # Track active RBs per BS for energy accounting
        for bs in bss:
            bs.active_rbs = sum(bs.rb_alloc.values())
        self.latency_ms = (_time.perf_counter() - t0) * 1000

    # === BL0: Static ISAC (no detection, no mitigation) ===
    def _bl0(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)

    # === BL1: Adaptive ISAC (rebalance from measurements, no twin) ===
    def _bl1(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        for bs in bss:
            if len(bs.served_users) < 2: continue
            sinrs = {}
            for uid in bs.served_users:
                u = next(u for u in users if u.user_id == uid)
                sinrs[uid] = u.sinr_db
            avg_s = np.mean(list(sinrs.values()))
            for uid in bs.served_users:
                if sinrs[uid] < avg_s - 3:
                    bs.rb_alloc[uid] = min(bs.rb_alloc.get(uid,1)+2, self.cfg.n_resource_blocks//2)
                elif sinrs[uid] > avg_s + 5:
                    bs.rb_alloc[uid] = max(1, bs.rb_alloc.get(uid,1)-1)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)

    # === BL2: DT-guided PF (detect but no quantum) ===
    def _bl2(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        self.twin.push(slot, users, targets)
        self.twin.update(slot, users, targets)
        # Detect anomalies
        if self.sec:
            self.sec.detect(self.twin, users, targets)
        # Twin-guided allocation: base + small shift
        for bs in bss:
            if len(bs.served_users) < 2: continue
            base = max(1, self.cfg.n_resource_blocks // len(bs.served_users))
            for uid in bs.served_users:
                s = self.twin.user_st.get(uid)
                if s and s.confidence > 0.2:
                    sinr_est = max(s.est_sinr, -20)
                    avg_se = np.log2(1 + 10**(sinr_est/10))
                    delta = -1 if avg_se > 4 else (1 if avg_se < 2 else 0)
                    bs.rb_alloc[uid] = max(1, base + delta)
                else:
                    bs.rb_alloc[uid] = base
        self.comm.evaluate(bss, users)

    # === BL3: DT + Quantum (attack-unaware: no anomaly detection/mitigation) ===
    # This baseline has digital twin + quantum search but is BLIND to attacks.
    # It allocates resources using quantum-optimised candidates with
    # twin-predicted channel states, but cannot detect or respond to
    # spoofing/jamming. Trust is assumed perfect (worst case for security).
    def _bl3(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        self.twin.push(slot, users, targets)
        self.twin.update(slot, users, targets)
        # Attack-unaware: trust all entities equally
        naive_trust = {f"u{u.user_id}": 1.0 for u in users}
        for bs in bss:
            if len(bs.served_users) < 1: continue
            cands = self.qa.generate(len(bs.served_users), self.cfg.n_resource_blocks,
                                      self.twin, bs.served_users)
            best = self.qa.search(cands, self.twin, naive_trust)
            for uid in bs.served_users:
                if uid in best.rb_alloc:
                    bs.rb_alloc[uid] = best.rb_alloc[uid]
        self.comm.evaluate(bss, users)

    # === BL4: Full Proposed (DT + Quantum + Security + Power Adaptation) ===
    def _bl4(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        self.twin.push(slot, users, targets)
        self.twin.update(slot, users, targets)
        # Detect anomalies
        self.sec.detect(self.twin, users, targets)
        # Quantum-assisted per-BS allocation with trust
        for bs in bss:
            if len(bs.served_users) < 1: continue
            cands = self.qa.generate(len(bs.served_users), self.cfg.n_resource_blocks,
                                      self.twin, bs.served_users)
            best = self.qa.search(cands, self.twin, self.sec.trust)
            for uid in bs.served_users:
                base = best.rb_alloc.get(uid, 1)
                t = self.sec.trust.get(f"u{uid}", 1.0)
                factor = 1.0 if t > 0.5 else (0.5 + t)
                bs.rb_alloc[uid] = max(1, int(base * factor))
            # Power adaptation: reduce Tx power if avg SINR margin is large
            avg_sinr_bs = np.mean([
                next(u.sinr_db for u in users if u.user_id == uid)
                for uid in bs.served_users
            ]) if bs.served_users else 0
            sinr_margin = avg_sinr_bs - self.cfg.sinr_threshold_db
            if sinr_margin > 15:  # excess margin → save power
                bs.tx_pow = max(bs.max_tx_pow * 0.6, bs.tx_pow * 0.95)
            elif sinr_margin < 5:  # tight margin → restore power
                bs.tx_pow = min(bs.max_tx_pow, bs.tx_pow * 1.05)
            # Sensing power fraction adaptation
            avg_pd = self.sense.avg_pd(targets)
            if avg_pd < 0.6:
                bs.sense_power_frac = min(0.40, bs.sense_power_frac + 0.01)
            elif avg_pd > 0.9:
                bs.sense_power_frac = max(0.10, bs.sense_power_frac - 0.01)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        self._adapt(users, targets, slot)

    def _adapt(self, users, targets, slot):
        if slot % 200 != 0 or slot == 0: return
        out = self.comm.outage_rate(users)
        pd = self.sense.avg_pd(targets)
        tr = self.sec.avg_trust() if self.sec else 1.0
        lr = self.cfg.control_lr
        if out > 0.15: self.w_c = min(0.6, self.w_c + lr)
        if pd < 0.7:   self.w_s = min(0.5, self.w_s + lr)
        if tr < 0.7:   self.w_sec = min(0.4, self.w_sec + lr)
        total = self.w_c + self.w_s + self.w_sec + self.w_e
        self.w_c /= total; self.w_s /= total
        self.w_sec /= total; self.w_e /= total
