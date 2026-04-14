"""
Trust-and-security layer.

Responsibilities:
  * inject cyber-physical anomalies (spoofing, jamming, burst, persistent)
  * detect anomalies using telemetry / twin mismatch evidence
  * maintain per-entity trust scores with EWMA dynamics
  * provide trust-aware deployment gating that can downscale or fall back
    on proposed control actions when the deployment-confidence is low
  * expose aggregate safety / robustness metrics required by the evaluation
    framework (fallback_deployment_ratio, unsafe_action_suppression_rate,
    trust_degradation_rate, anomaly_containment_score)
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple

from config import SimConfig
from network import MobileUser, SensingTarget
from digital_twin import DigitalTwin


class SecurityModel:
    # Named attack types for anomaly scenarios
    _SPOOF, _JAM = 'spoof', 'jam'

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.trust: Dict[str, float] = {}
        # (key, type, severity, age)
        self.active_attacks: list = []
        self.n_detected = 0
        self.n_missed = 0
        self.n_false = 0
        self._score_ewma: Dict[str, float] = {}
        self._ewma_alpha: float = 0.1

        # Trust-aware deployment accounting
        self.n_deploys: int = 0
        self.n_fallback_deploys: int = 0
        self.n_unsafe_suppressed: int = 0
        self.n_unsafe_attempts: int = 0
        self.n_safe_slots: int = 0
        self.n_total_slots: int = 0

        # Per-slot trust history for degradation rate
        self._trust_history: list = []

        # Containment score accumulator
        self._contain_num: float = 0.0
        self._contain_den: float = 0.0

    # ------------------------------------------------------------------
    def init(self, users, targets):
        for u in users:
            self.trust[f"u{u.user_id}"] = self.cfg.trust_init
        for t in targets:
            self.trust[f"t{t.target_id}"] = self.cfg.trust_init

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------
    def inject(self, slot: int,
               users: List[MobileUser],
               targets: List[SensingTarget]) -> None:
        """Inject anomalies per the configured scenario."""
        mode = self.cfg.anomaly_mode
        prob = self.cfg.anomaly_prob

        # Age / expire
        max_age = 30 if mode == "persistent" else 15
        self.active_attacks = [(k, ty, sv, ag + 1)
                               for k, ty, sv, ag in self.active_attacks
                               if ag < max_age]
        active_keys = {a[0] for a in self.active_attacks}
        jam_keys = {a[0]: a[2] for a in self.active_attacks if a[1] == self._JAM}

        for u in users:
            key = f"u{u.user_id}"
            if key in jam_keys:
                sev = jam_keys[key]
                jam_db = self.cfg.anomaly_jam_db * sev
                u.jam_interference_w = 10 ** (jam_db / 10) * self.cfg.noise_power_w()
            else:
                u.jam_interference_w = 0.0

        if mode == "none" or prob <= 0.0:
            return

        # Burst mode: during injection burst window, attack probability is ×3
        burst_factor = 1.0
        if mode == "burst":
            cycle = slot % 200
            burst_factor = 3.0 if cycle < 30 else 0.2

        for u in users:
            key = f"u{u.user_id}"
            if key in active_keys:
                continue
            if self.rng.random() < prob * burst_factor:
                sev = self.rng.uniform(0.4, 1.0)
                ty = self._pick_attack_type(mode)
                if ty == self._JAM:
                    jam_db = self.cfg.anomaly_jam_db * sev
                    u.jam_interference_w = 10 ** (jam_db / 10) * self.cfg.noise_power_w()
                else:
                    off = self.cfg.anomaly_spoof_m * sev
                    u.pos.x = np.clip(u.pos.x + self.rng.uniform(-off, off),
                                       0, self.cfg.area_size)
                    u.pos.y = np.clip(u.pos.y + self.rng.uniform(-off, off),
                                       0, self.cfg.area_size)
                self.active_attacks.append((key, ty, sev, 0))

        for t in targets:
            key = f"t{t.target_id}"
            if key in active_keys:
                continue
            if self.rng.random() < prob * 0.5 * burst_factor:
                sev = self.rng.uniform(0.3, 1.0)
                off = self.cfg.anomaly_spoof_m * sev
                t.pos.x = np.clip(t.pos.x + self.rng.uniform(-off, off),
                                   0, self.cfg.area_size)
                t.pos.y = np.clip(t.pos.y + self.rng.uniform(-off, off),
                                   0, self.cfg.area_size)
                self.active_attacks.append((key, self._SPOOF, sev, 0))

    def _pick_attack_type(self, mode: str) -> str:
        if mode == "spoof":
            return self._SPOOF
        if mode == "jam":
            return self._JAM
        if mode in ("persistent", "burst", "mixed"):
            return self._JAM if self.rng.random() < 0.5 else self._SPOOF
        return self._SPOOF

    # ------------------------------------------------------------------
    # Anomaly detection + trust dynamics
    # ------------------------------------------------------------------
    def detect(self, twin: DigitalTwin,
               users: List[MobileUser],
               targets: List[SensingTarget]) -> None:
        atk_keys = {a[0] for a in self.active_attacks}
        pos_base = self.cfg.twin_pos_noise_std * 3
        sinr_base = self.cfg.twin_sinr_noise_std * 2

        for u in users:
            key = f"u{u.user_id}"
            s = twin.user_st.get(u.user_id)
            if not s or not s.est_pos:
                continue
            pos_sc = u.pos.distance_to(s.est_pos) / (pos_base + 1)
            sinr_sc = abs(u.sinr_db - s.est_sinr) / (sinr_base + 1)
            score = 0.5 * pos_sc + 0.5 * sinr_sc

            ek = key + "_c"
            old_e = self._score_ewma.get(ek, 0.3)
            thr = max(1.5, old_e * 1.5 + 0.3)
            self._score_ewma[ek] = (self._ewma_alpha * score
                                     + (1 - self._ewma_alpha) * old_e)

            flagged = score > thr
            real = key in atk_keys
            if flagged and real:
                self.n_detected += 1
            elif flagged and not real:
                self.n_false += 1
            elif not flagged and real:
                self.n_missed += 1

            if flagged:
                exc = min(score / thr, 3.0)
                self.trust[key] = max(0.05, self.trust.get(key, 1)
                                       * (1 - self.cfg.trust_decay_rate * exc))
            else:
                tau = self.trust.get(key, 1.0)
                self.trust[key] = tau + self.cfg.trust_recovery_rate * (1 - tau)

        for t in targets:
            key = f"t{t.target_id}"
            s = twin.target_st.get(t.target_id)
            if not s or not s.est_pos:
                continue
            score = t.pos.distance_to(s.est_pos) / (pos_base * 1.5 + 1)
            ek = key + "_c"
            old_e = self._score_ewma.get(ek, 0.3)
            thr = max(1.5, old_e * 1.5 + 0.3)
            self._score_ewma[ek] = (self._ewma_alpha * score
                                     + (1 - self._ewma_alpha) * old_e)
            flagged = score > thr
            real = key in atk_keys
            if flagged and real:
                self.n_detected += 1
            elif flagged and not real:
                self.n_false += 1
            elif not flagged and real:
                self.n_missed += 1
            if flagged:
                self.trust[key] = max(0.05, self.trust.get(key, 1)
                                       * (1 - self.cfg.trust_decay_rate * 1.5))
            else:
                tau = self.trust.get(key, 1.0)
                self.trust[key] = tau + self.cfg.trust_recovery_rate * (1 - tau)

        # Snapshot trust history (slot-level)
        self._trust_history.append(self.avg_trust())

    # ------------------------------------------------------------------
    # Trust-aware deployment gating
    # ------------------------------------------------------------------
    def deployment_confidence(self, uid: int, twin: DigitalTwin) -> float:
        """Combine trust, twin-mismatch and EWMA anomaly evidence into a
        deployment-confidence value in [0,1]."""
        tau = self.trust.get(f"u{uid}", 1.0)
        ek = f"u{uid}_c"
        evid = self._score_ewma.get(ek, 0.3)
        evid_factor = max(0.0, 1.0 - min(1.0, evid / 3.0))
        mismatch = 0.0
        if twin is not None:
            mismatch = twin.user_mismatch(uid)
        mis_factor = max(0.0, 1.0 - min(1.0, mismatch * 4.0))
        conf = 0.5 * tau + 0.25 * evid_factor + 0.25 * mis_factor
        return float(np.clip(conf, 0.0, 1.0))

    def gate_action(self,
                    uid: int,
                    proposed_rbs: int,
                    proposed_power_scale: float,
                    twin: DigitalTwin) -> Tuple[int, float, bool]:
        """Apply trust-aware gating to a proposed per-user action.

        Returns (rbs, power_scale, was_fallback).
        """
        self.n_deploys += 1
        conf = self.deployment_confidence(uid, twin)

        # Consider a proposed action "unsafe" if it demands high RBs / power
        # while deployment confidence is low.
        unsafe = (proposed_rbs > self.cfg.n_resource_blocks * 0.5
                  and conf < self.cfg.trust_gate_threshold)
        if unsafe:
            self.n_unsafe_attempts += 1

        if conf < self.cfg.trust_hard_floor:
            # Hard fallback — strong suppression
            rbs = max(1, int(proposed_rbs * self.cfg.fallback_rb_scale * 0.5))
            pwr = proposed_power_scale * self.cfg.fallback_power_scale * 0.7
            self.n_fallback_deploys += 1
            if unsafe:
                self.n_unsafe_suppressed += 1
            return rbs, pwr, True

        if conf < self.cfg.trust_gate_threshold:
            # Soft fallback — scale down smoothly
            scale = (conf - self.cfg.trust_hard_floor) / max(
                1e-9, self.cfg.trust_gate_threshold - self.cfg.trust_hard_floor)
            rb_scale = self.cfg.fallback_rb_scale + (1 - self.cfg.fallback_rb_scale) * scale
            pw_scale = self.cfg.fallback_power_scale + (1 - self.cfg.fallback_power_scale) * scale
            rbs = max(1, int(proposed_rbs * rb_scale))
            pwr = proposed_power_scale * pw_scale
            self.n_fallback_deploys += 1
            if unsafe:
                self.n_unsafe_suppressed += 1
            return rbs, pwr, True

        return proposed_rbs, proposed_power_scale, False

    # ------------------------------------------------------------------
    # Slot-level accounting (called after evaluate)
    # ------------------------------------------------------------------
    def end_slot(self, n_active_attacks: int, slot_utility: float) -> None:
        self.n_total_slots += 1
        if n_active_attacks == 0:
            self.n_safe_slots += 1
        # Containment score: fraction of utility retained during attack slots
        if n_active_attacks > 0:
            self._contain_num += max(0.0, slot_utility)
            self._contain_den += 1.0

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    def avg_trust(self) -> float:
        return float(np.mean(list(self.trust.values()))) if self.trust else 1.0

    def det_rate(self) -> float:
        t = self.n_detected + self.n_missed
        return self.n_detected / max(t, 1)

    def fa_rate(self) -> float:
        t = self.n_detected + self.n_false
        return self.n_false / max(t, 1)

    def fallback_deployment_ratio(self) -> float:
        return self.n_fallback_deploys / max(self.n_deploys, 1)

    def unsafe_action_suppression_rate(self) -> float:
        return self.n_unsafe_suppressed / max(self.n_unsafe_attempts, 1)

    def trust_degradation_rate(self) -> float:
        """Average slot-over-slot downward drift in trust (positive = worse)."""
        if len(self._trust_history) < 2:
            return 0.0
        arr = np.array(self._trust_history)
        drops = np.clip(arr[:-1] - arr[1:], 0, None)
        return float(drops.mean())

    def anomaly_containment_score(self, baseline_utility: float = 0.0) -> float:
        """Mean utility retained during attack slots, clipped to [0,1]."""
        if self._contain_den <= 0:
            return 1.0
        mean_u = self._contain_num / self._contain_den
        if baseline_utility <= 1e-9:
            return float(np.clip(mean_u, 0.0, 1.0))
        return float(np.clip(mean_u / baseline_utility, 0.0, 1.0))

    def safe_control_persistence(self) -> float:
        return self.n_safe_slots / max(self.n_total_slots, 1)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.trust.clear()
        self.active_attacks.clear()
        self.n_detected = self.n_missed = self.n_false = 0
        self._score_ewma.clear()
        self.n_deploys = self.n_fallback_deploys = 0
        self.n_unsafe_suppressed = self.n_unsafe_attempts = 0
        self.n_safe_slots = self.n_total_slots = 0
        self._trust_history.clear()
        self._contain_num = self._contain_den = 0.0
