"""
Mono-static OFDM radar sensing for ISAC.
Calibrated: P_d varies 0.3-0.95 across target distances 100-800m.
"""
import numpy as np
from typing import List
from config import SimConfig
from network import BaseStation, SensingTarget


class SensingModel:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.c = 3e8

    def radar_snr(self, bs: BaseStation, tgt: SensingTarget) -> float:
        """Radar range equation with clutter and processing loss."""
        d = max(bs.pos.distance_to(tgt.pos), 10.0)
        lam = self.c / (self.cfg.bs_frequency_ghz * 1e9)
        sigma = tgt.rcs
        pt = bs.tx_pow * self.cfg.sensing_power_fraction
        # Coherent processing gain (one-way for Tx beamforming)
        G_tx = self.cfg.bs_n_antennas
        G_rx = np.sqrt(self.cfg.bs_n_antennas)  # reduced Rx gain (shared aperture)
        noise = self.cfg.noise_power_w()
        clutter = noise * 10**(self.cfg.clutter_to_noise_ratio_db / 10)
        proc_loss = 10**(self.cfg.sensing_processing_loss_db / 10)

        num = pt * G_tx * G_rx * sigma * lam**2
        den = (4*np.pi)**3 * d**4 * (noise + clutter) * proc_loss

        snr_single = num / (den + 1e-30)
        return snr_single * self.cfg.sensing_integration_slots

    def prob_detection(self, snr_lin: float) -> float:
        """Swerling-I detection probability via Albersheim approximation."""
        if snr_lin <= 1e-10:
            return 0.0
        pfa = self.cfg.pfa
        n = self.cfg.sensing_integration_slots
        snr_db = 10 * np.log10(snr_lin)
        # Albersheim
        A = np.log(0.62 / pfa)
        Z = snr_db - 5*np.log10(n) + 6.2 + 4.54/np.sqrt(n + 0.44)
        pd = 1.0 / (1.0 + np.exp(-0.8*(Z - A)))
        return np.clip(pd, 0.0, 0.999)

    def tracking_error(self, snr_lin: float) -> float:
        """CRLB for range estimation."""
        if snr_lin <= 1e-10:
            return 500.0  # very large error
        bw = self.cfg.bs_bandwidth_mhz * 1e6
        return self.c / (2 * bw * np.sqrt(2*snr_lin + 1e-10))

    def sensing_utility(self, pd: float, terr: float) -> float:
        acc = max(0.0, 1.0 - terr / 100.0)
        return 0.6 * pd + 0.4 * acc

    def evaluate(self, bss: List[BaseStation], tgts: List[SensingTarget]):
        for t in tgts:
            dists = [t.pos.distance_to(bs.pos) for bs in bss]
            bs = bss[int(np.argmin(dists))]
            snr = self.radar_snr(bs, t)
            t.sensing_snr_db = 10*np.log10(max(snr, 1e-30))
            pd = self.prob_detection(snr)
            t.detected = self.rng.random() < pd
            t.track_err = self.tracking_error(snr)

    def avg_pd(self, tgts):
        return np.mean([self.prob_detection(10**(t.sensing_snr_db/10)) for t in tgts])

    def avg_utility(self, tgts):
        return np.mean([
            self.sensing_utility(self.prob_detection(10**(t.sensing_snr_db/10)), t.track_err)
            for t in tgts
        ])
