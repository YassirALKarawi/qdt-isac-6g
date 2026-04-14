"""
Trust-aware digital-twin-in-the-loop controller.

Baselines (selected via `SimConfig.baseline_id`):
    -1 : ablation mode — composed from ablation flags
     0 : Static ISAC                          (no adaptation)
     1 : Reactive Adaptive ISAC               (measurement-driven rebalancing)
     2 : DT-guided (no quantum, no gating)
     3 : DT + QA screening (no trust gating)
     4 : Full Proposed — trust-aware DT-in-the-loop + QA screening
     5 : Predictor-based uncertainty-aware (no DT)
     6 : Robust min-max heuristic (no DT / no QA)
     7 : Learning-based epsilon-greedy bandit controller

Every method shares the same measurement plumbing; baselines 0-4 retain their
original roles and behaviour. Baselines 5-7 are additional strong algorithmic
benchmarks so that the proposed method is not only evaluated against weak
internal baselines.

The ablation mode (baseline_id == -1) drives behaviour from the flags
`use_twin`, `use_trust_gating`, `use_screening`, `use_adaptive_weights`,
`use_mismatch_comp`.
"""
from __future__ import annotations
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
    def __init__(self, cfg: SimConfig, ch, comm, sense, twin, sec, qa, rng):
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

        # Predictor state (BL5)
        self._pred_sinr: Dict[int, float] = {}
        self._pred_var: Dict[int, float] = {}
        # Bandit state (BL7): arm = (rb_delta, power_scale)
        self._arms = [(-2, 1.0), (-1, 1.0), (0, 1.0),
                      (+1, 0.9), (+2, 0.8), (+3, 0.7)]
        self._arm_rewards = np.zeros(len(self._arms))
        self._arm_counts = np.zeros(len(self._arms))
        self._epsilon = 0.15

    # ------------------------------------------------------------------
    def step(self, slot: int, bss, users, targets) -> None:
        t0 = _time.perf_counter()
        if self.sec:
            self.sec.inject(slot, users, targets)

        if self.bl == -1:
            self._bl_ablation(slot, bss, users, targets)
        else:
            methods = {0: self._bl0, 1: self._bl1, 2: self._bl2, 3: self._bl3,
                       4: self._bl4, 5: self._bl5, 6: self._bl6, 7: self._bl7}
            methods.get(self.bl, self._bl0)(slot, bss, users, targets)

        for bs in bss:
            bs.active_rbs = sum(bs.rb_alloc.values())
        self.latency_ms = (_time.perf_counter() - t0) * 1000

    # ==================================================================
    # BL0 — Static ISAC
    # ==================================================================
    def _bl0(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)

    # ==================================================================
    # BL1 — Reactive adaptive ISAC
    # ==================================================================
    def _bl1(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        for bs in bss:
            if len(bs.served_users) < 2:
                continue
            sinrs = {uid: next(u.sinr_db for u in users if u.user_id == uid)
                      for uid in bs.served_users}
            avg_s = float(np.mean(list(sinrs.values())))
            for uid in bs.served_users:
                if sinrs[uid] < avg_s - 3:
                    bs.rb_alloc[uid] = min(bs.rb_alloc.get(uid, 1) + 2,
                                             self.cfg.n_resource_blocks // 2)
                elif sinrs[uid] > avg_s + 5:
                    bs.rb_alloc[uid] = max(1, bs.rb_alloc.get(uid, 1) - 1)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)

    # ==================================================================
    # BL2 — DT-guided (no QA, no gating)
    # ==================================================================
    def _bl2(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        self.twin.push(slot, users, targets)
        self.twin.update(slot, users, targets)
        if self.sec:
            self.sec.detect(self.twin, users, targets)
        for bs in bss:
            if len(bs.served_users) < 2:
                continue
            base = max(1, self.cfg.n_resource_blocks // len(bs.served_users))
            for uid in bs.served_users:
                s = self.twin.user_st.get(uid)
                if s and s.confidence > 0.2:
                    sinr_est = max(s.est_sinr, -20)
                    avg_se = np.log2(1 + 10 ** (sinr_est / 10))
                    delta = -1 if avg_se > 4 else (1 if avg_se < 2 else 0)
                    bs.rb_alloc[uid] = max(1, base + delta)
                else:
                    bs.rb_alloc[uid] = base
        self.comm.evaluate(bss, users)

    # ==================================================================
    # BL3 — DT + QA (no trust gating)
    # ==================================================================
    def _bl3(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        self.twin.push(slot, users, targets)
        self.twin.update(slot, users, targets)
        naive_trust = {f"u{u.user_id}": 1.0 for u in users}
        for bs in bss:
            if not bs.served_users:
                continue
            cands = self.qa.generate(len(bs.served_users),
                                       self.cfg.n_resource_blocks,
                                       self.twin, bs.served_users)
            best = self.qa.search(cands, self.twin, naive_trust)
            for uid in bs.served_users:
                if uid in best.rb_alloc:
                    bs.rb_alloc[uid] = best.rb_alloc[uid]
        self.comm.evaluate(bss, users)

    # ==================================================================
    # BL4 — Full Proposed (trust-aware, DT + QA + adaptive)
    # ==================================================================
    def _bl4(self, slot, bss, users, targets):
        self._trust_aware_qa_step(slot, bss, users, targets,
                                    use_twin=True, use_qa=True,
                                    use_gating=True, use_adapt=True,
                                    use_mismatch=True)

    # ==================================================================
    # BL5 — Predictor-based uncertainty-aware (no DT)
    # ==================================================================
    def _bl5(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        # EWMA predictor on measured SINR
        alpha = 0.2
        for u in users:
            prev = self._pred_sinr.get(u.user_id, u.sinr_db)
            prev_v = self._pred_var.get(u.user_id, 4.0)
            new_mean = alpha * u.sinr_db + (1 - alpha) * prev
            resid = u.sinr_db - prev
            new_var = max(0.5, 0.8 * prev_v + 0.2 * resid * resid)
            self._pred_sinr[u.user_id] = new_mean
            self._pred_var[u.user_id] = new_var
        for bs in bss:
            if not bs.served_users:
                continue
            base = max(1, self.cfg.n_resource_blocks // len(bs.served_users))
            for uid in bs.served_users:
                mu = self._pred_sinr.get(uid, 0.0)
                var = self._pred_var.get(uid, 4.0)
                # Uncertainty-aware conservative allocation:
                # high variance → bump RBs (safety margin); high mean → trim.
                delta = 0
                if var > 6.0:
                    delta += 1
                if mu < 0:
                    delta += 1
                if mu > 15:
                    delta -= 1
                bs.rb_alloc[uid] = max(1, base + delta)
        self.comm.evaluate(bss, users)

    # ==================================================================
    # BL6 — Robust min-max heuristic controller
    # ==================================================================
    def _bl6(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        # Protect worst user per BS: allocate more RBs to the weakest
        # served UE, within a min-max budget.
        for bs in bss:
            if not bs.served_users:
                continue
            n = len(bs.served_users)
            base = max(1, self.cfg.n_resource_blocks // n)
            sinrs = {uid: next(u.sinr_db for u in users if u.user_id == uid)
                      for uid in bs.served_users}
            worst_uid = min(sinrs, key=sinrs.get)
            best_uid = max(sinrs, key=sinrs.get)
            for uid in bs.served_users:
                if uid == worst_uid:
                    bs.rb_alloc[uid] = min(self.cfg.n_resource_blocks,
                                            base + 3)
                elif uid == best_uid and n > 2:
                    bs.rb_alloc[uid] = max(1, base - 1)
                else:
                    bs.rb_alloc[uid] = base
            # Small sensing boost when link margin is healthy
            avg_sinr = float(np.mean(list(sinrs.values())))
            if avg_sinr > 12:
                bs.sense_power_frac = min(0.35, bs.sense_power_frac + 0.01)
        self.comm.evaluate(bss, users)

    # ==================================================================
    # BL7 — Learning-based epsilon-greedy bandit controller
    # ==================================================================
    def _bl7(self, slot, bss, users, targets):
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        # Shared arm across BSs (simple formulation for comparability)
        if self.rng.random() < self._epsilon:
            arm = int(self.rng.integers(0, len(self._arms)))
        else:
            # argmax(avg reward)
            avg = np.where(self._arm_counts > 0,
                            self._arm_rewards / np.maximum(self._arm_counts, 1),
                            0.0)
            arm = int(np.argmax(avg))
        rb_delta, pw_scale = self._arms[arm]
        reward_proxy = 0.0
        for bs in bss:
            if not bs.served_users:
                continue
            base = max(1, self.cfg.n_resource_blocks // len(bs.served_users))
            for uid in bs.served_users:
                bs.rb_alloc[uid] = max(1, base + rb_delta)
            bs.tx_pow = max(bs.max_tx_pow * 0.5, bs.max_tx_pow * pw_scale)
            reward_proxy += float(np.mean([
                next(u.sinr_db for u in users if u.user_id == uid)
                for uid in bs.served_users
            ]))
        self.comm.evaluate(bss, users)
        # Reward: sum-rate normalised plus sensing Pd, minus energy
        sr = self.comm.sum_rate(users) / 1e6
        pd = self.sense.avg_pd(targets)
        reward = 0.001 * sr + 0.4 * pd - 0.1 * pw_scale
        self._arm_rewards[arm] += reward
        self._arm_counts[arm] += 1
        # Anneal exploration
        self._epsilon = max(0.02, self._epsilon * 0.999)

    # ==================================================================
    # Ablation — composed from flags
    # ==================================================================
    def _bl_ablation(self, slot, bss, users, targets):
        c = self.cfg
        self._trust_aware_qa_step(
            slot, bss, users, targets,
            use_twin=c.use_twin,
            use_qa=c.use_screening,
            use_gating=c.use_trust_gating,
            use_adapt=c.use_adaptive_weights,
            use_mismatch=c.use_mismatch_comp,
        )

    # ==================================================================
    # Shared trust-aware DT-in-the-loop step
    # ==================================================================
    def _trust_aware_qa_step(self, slot, bss, users, targets,
                              use_twin: bool, use_qa: bool,
                              use_gating: bool, use_adapt: bool,
                              use_mismatch: bool) -> None:
        self.comm.assign_users(bss, users)
        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)

        if use_twin and self.twin is not None:
            self.twin.push(slot, users, targets)
            self.twin.update(slot, users, targets)
            if self.sec is not None:
                self.sec.detect(self.twin, users, targets)

        for bs in bss:
            if not bs.served_users:
                continue

            if use_qa and self.qa is not None and use_twin and self.twin is not None:
                cands = self.qa.generate(len(bs.served_users),
                                           self.cfg.n_resource_blocks,
                                           self.twin, bs.served_users)
                trust = self.sec.trust if (self.sec and use_gating) else \
                        {f"u{uid}": 1.0 for uid in bs.served_users}
                best = self.qa.search(cands, self.twin, trust)
                proposed = {uid: best.rb_alloc.get(uid, 1) for uid in bs.served_users}
            elif use_twin and self.twin is not None:
                base = max(1, self.cfg.n_resource_blocks // len(bs.served_users))
                proposed = {}
                for uid in bs.served_users:
                    s = self.twin.user_st.get(uid)
                    if s and s.confidence > 0.2:
                        sinr_est = max(s.est_sinr, -20)
                        avg_se = np.log2(1 + 10 ** (sinr_est / 10))
                        delta = -1 if avg_se > 4 else (1 if avg_se < 2 else 0)
                        proposed[uid] = max(1, base + delta)
                    else:
                        proposed[uid] = base
            else:
                base = max(1, self.cfg.n_resource_blocks // len(bs.served_users))
                proposed = {uid: base for uid in bs.served_users}

            # Twin mismatch compensation: scale down allocation for high-mismatch UEs
            if use_mismatch and self.twin is not None:
                for uid, rbs in list(proposed.items()):
                    mis = self.twin.user_mismatch(uid)
                    if mis > 0.15:
                        proposed[uid] = max(1, int(rbs * max(0.5, 1.0 - mis)))

            pw_scale = 1.0
            for uid in bs.served_users:
                prb = proposed.get(uid, 1)
                if use_gating and self.sec is not None:
                    rbs, pw_scale_u, _ = self.sec.gate_action(
                        uid, prb, pw_scale, self.twin)
                else:
                    rbs, pw_scale_u = prb, pw_scale
                bs.rb_alloc[uid] = rbs

            # Per-BS Tx power adaptation (retained from BL4 original)
            if bs.served_users:
                avg_sinr_bs = float(np.mean([
                    next(u.sinr_db for u in users if u.user_id == uid)
                    for uid in bs.served_users
                ]))
                sinr_margin = avg_sinr_bs - self.cfg.sinr_threshold_db
                if sinr_margin > 15:
                    bs.tx_pow = max(bs.max_tx_pow * 0.6, bs.tx_pow * 0.95)
                elif sinr_margin < 5:
                    bs.tx_pow = min(bs.max_tx_pow, bs.tx_pow * 1.05)
                avg_pd = self.sense.avg_pd(targets)
                if avg_pd < 0.6:
                    bs.sense_power_frac = min(0.40, bs.sense_power_frac + 0.01)
                elif avg_pd > 0.9:
                    bs.sense_power_frac = max(0.10, bs.sense_power_frac - 0.01)

        self.comm.evaluate(bss, users)
        self.sense.evaluate(bss, targets)
        if use_adapt:
            self._adapt(users, targets, slot)

    # ==================================================================
    def _adapt(self, users, targets, slot) -> None:
        if slot % 200 != 0 or slot == 0:
            return
        out = self.comm.outage_rate(users)
        pd = self.sense.avg_pd(targets)
        tr = self.sec.avg_trust() if self.sec else 1.0
        lr = self.cfg.control_lr
        if out > 0.15:
            self.w_c = min(0.6, self.w_c + lr)
        if pd < 0.7:
            self.w_s = min(0.5, self.w_s + lr)
        if tr < 0.7:
            self.w_sec = min(0.4, self.w_sec + lr)
        total = self.w_c + self.w_s + self.w_sec + self.w_e
        self.w_c /= total
        self.w_s /= total
        self.w_sec /= total
        self.w_e /= total
