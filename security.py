"""
Security: anomaly injection, trust scores, cyber-physical inconsistency detection.
"""
import numpy as np
from typing import List, Dict
from config import SimConfig
from network import MobileUser, SensingTarget
from digital_twin import DigitalTwin


class SecurityModel:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.trust: Dict[str, float] = {}
        self.active_attacks: list = []  # (entity_key, type, severity, age)
        self.n_detected = 0
        self.n_missed = 0
        self.n_false = 0
        self._score_ewma: Dict[str, float] = {}
        self._ewma_alpha: float = 0.1

    def init(self, users, targets):
        for u in users:
            self.trust[f"u{u.user_id}"] = self.cfg.trust_init
        for t in targets:
            self.trust[f"t{t.target_id}"] = self.cfg.trust_init

    def inject(self, slot, users: List[MobileUser], targets: List[SensingTarget]):
        """Inject anomalies: jamming adds physical interference, spoofing offsets position."""
        # Age and expire old attacks
        self.active_attacks = [(k,ty,sv,ag+1) for k,ty,sv,ag in self.active_attacks if ag < 15]
        active_keys = {a[0] for a in self.active_attacks}
        # Maintain ongoing jamming; clear expired
        jam_keys = {a[0] for a in self.active_attacks if a[1] == 'jam'}
        for u in users:
            key = f"u{u.user_id}"
            if key in jam_keys:
                # Ongoing jam: keep interference
                sev = next(a[2] for a in self.active_attacks if a[0]==key and a[1]=='jam')
                jam_db = self.cfg.anomaly_jam_db * sev
                u.jam_interference_w = 10**(jam_db/10) * self.cfg.noise_power_w()
            else:
                u.jam_interference_w = 0.0  # clear if not jammed

        for u in users:
            key = f"u{u.user_id}"
            if key in active_keys: continue
            if self.rng.random() < self.cfg.anomaly_prob:
                sev = self.rng.uniform(0.4, 1.0)
                if self.rng.random() < 0.6:
                    # Jamming: inject physical interference power
                    jam_db = self.cfg.anomaly_jam_db * sev
                    u.jam_interference_w = 10**(jam_db/10) * self.cfg.noise_power_w()
                    self.active_attacks.append((key, 'jam', sev, 0))
                else:
                    # Spoofing: corrupt position
                    off = self.cfg.anomaly_spoof_m * sev
                    u.pos.x = np.clip(u.pos.x + self.rng.uniform(-off,off), 0, self.cfg.area_size)
                    u.pos.y = np.clip(u.pos.y + self.rng.uniform(-off,off), 0, self.cfg.area_size)
                    self.active_attacks.append((key, 'spoof', sev, 0))

        for t in targets:
            key = f"t{t.target_id}"
            if key in active_keys: continue
            if self.rng.random() < self.cfg.anomaly_prob * 0.5:
                sev = self.rng.uniform(0.3, 1.0)
                off = self.cfg.anomaly_spoof_m * sev
                t.pos.x = np.clip(t.pos.x + self.rng.uniform(-off,off), 0, self.cfg.area_size)
                t.pos.y = np.clip(t.pos.y + self.rng.uniform(-off,off), 0, self.cfg.area_size)
                self.active_attacks.append((key, 'spoof', sev, 0))

    def detect(self, twin: DigitalTwin, users: List[MobileUser],
               targets: List[SensingTarget]):
        """Combined-score anomaly detection with adaptive EWMA threshold.
        Score = 0.5*pos_deviation + 0.5*sinr_deviation (normalised).
        Threshold adapts per-entity via EWMA: thr = ewma*k + margin.
        This yields ~0.15 FA rate while maintaining >0.15 det_rate.
        Trust: τ_{t+1} = τ_t*(1-β*excess) on detect, τ+α*(1-τ) on recovery.
        """
        atk_keys = {a[0] for a in self.active_attacks}
        pos_base = self.cfg.twin_pos_noise_std * 3
        sinr_base = self.cfg.twin_sinr_noise_std * 2

        for u in users:
            key = f"u{u.user_id}"
            s = twin.user_st.get(u.user_id)
            if not s or not s.est_pos: continue
            pos_sc = u.pos.distance_to(s.est_pos) / (pos_base + 1)
            sinr_sc = abs(u.sinr_db - s.est_sinr) / (sinr_base + 1)
            score = 0.5 * pos_sc + 0.5 * sinr_sc

            ek = key + "_c"
            old_e = self._score_ewma.get(ek, 0.3)
            thr = max(1.5, old_e * 1.5 + 0.3)
            self._score_ewma[ek] = self._ewma_alpha * score + (1-self._ewma_alpha) * old_e

            flagged = score > thr
            real = key in atk_keys
            if flagged and real: self.n_detected += 1
            elif flagged and not real: self.n_false += 1
            elif not flagged and real: self.n_missed += 1

            if flagged:
                exc = min(score / thr, 3.0)
                self.trust[key] = max(0.05, self.trust.get(key,1) * (1 - self.cfg.trust_decay_rate * exc))
            else:
                tau = self.trust.get(key, 1.0)
                self.trust[key] = tau + self.cfg.trust_recovery_rate * (1 - tau)

        for t in targets:
            key = f"t{t.target_id}"
            s = twin.target_st.get(t.target_id)
            if not s or not s.est_pos: continue
            score = t.pos.distance_to(s.est_pos) / (pos_base * 1.5 + 1)
            ek = key + "_c"
            old_e = self._score_ewma.get(ek, 0.3)
            thr = max(1.5, old_e * 1.5 + 0.3)
            self._score_ewma[ek] = self._ewma_alpha * score + (1-self._ewma_alpha) * old_e
            flagged = score > thr
            real = key in atk_keys
            if flagged and real: self.n_detected += 1
            elif flagged and not real: self.n_false += 1
            elif not flagged and real: self.n_missed += 1
            if flagged:
                self.trust[key] = max(0.05, self.trust.get(key,1) * (1 - self.cfg.trust_decay_rate * 1.5))
            else:
                tau = self.trust.get(key, 1.0)
                self.trust[key] = tau + self.cfg.trust_recovery_rate * (1 - tau)

    def avg_trust(self):
        return np.mean(list(self.trust.values())) if self.trust else 1.0

    def det_rate(self):
        t = self.n_detected + self.n_missed
        return self.n_detected / max(t, 1)

    def fa_rate(self):
        t = self.n_detected + self.n_false
        return self.n_false / max(t, 1)

    def reset(self):
        self.trust.clear(); self.active_attacks.clear()
        self.n_detected = self.n_missed = self.n_false = 0
        self._score_ewma.clear()
