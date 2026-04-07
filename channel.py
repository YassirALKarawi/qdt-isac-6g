"""
Time-varying wireless channel: path loss, shadowing, small-scale fading, AR(1).
"""
import numpy as np
from config import SimConfig

class ChannelModel:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._fading: dict = {}  # (tx,rx) -> complex array

    def path_loss_db(self, d_m: float) -> float:
        d = max(d_m, 10.0)
        f = self.cfg.bs_frequency_ghz
        return 32.4 + 20*np.log10(f) + 10*self.cfg.path_loss_exponent*np.log10(d)

    def small_scale(self, tx: int, rx: int, na: int, los: bool = False) -> np.ndarray:
        key = (tx, rx)
        rho = self.cfg.channel_temporal_corr
        if los:
            k = 10**(self.cfg.rician_k_db/10)
            inn = (np.sqrt(k/(1+k)) * np.ones(na) +
                   np.sqrt(1/(2*(1+k))) * (self.rng.standard_normal(na) +
                                            1j*self.rng.standard_normal(na)))
        else:
            inn = (1/np.sqrt(2)) * (self.rng.standard_normal(na) +
                                     1j*self.rng.standard_normal(na))
        if key in self._fading and self._fading[key].shape[0] == na:
            h = rho * self._fading[key] + np.sqrt(1-rho**2) * inn
        else:
            h = inn
        self._fading[key] = h
        return h

    def gain_linear(self, dist: float, tx: int, rx: int, na: int, los=False) -> float:
        pl = self.path_loss_db(dist)
        shadow = self.rng.normal(0, self.cfg.shadow_std_db)
        h = self.small_scale(tx, rx, na, los)
        bf = np.sum(np.abs(h)**2)  # beamforming gain
        g_db = -pl - shadow + 10*np.log10(bf + 1e-30)
        return 10**(g_db/10)

    def sinr(self, sig_gain: float, intf_gains: list, ptx: float, noise: float) -> float:
        return ptx*sig_gain / (sum(ptx*g for g in intf_gains) + noise + 1e-30)

    def reset(self):
        self._fading.clear()
