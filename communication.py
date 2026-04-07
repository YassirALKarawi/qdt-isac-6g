"""
Downlink communication: user-BS association, SINR, throughput, outage.
"""
import numpy as np
from typing import List
from config import SimConfig
from network import BaseStation, MobileUser
from channel import ChannelModel


class CommModel:
    def __init__(self, cfg: SimConfig, ch: ChannelModel):
        self.cfg = cfg
        self.ch = ch

    def assign_users(self, bss: List[BaseStation], users: List[MobileUser]):
        for bs in bss:
            bs.served_users.clear()
            bs.rb_alloc.clear()
        for u in users:
            dists = [u.pos.distance_to(bs.pos) for bs in bss]
            u.serving_bs = int(np.argmin(dists))
            bss[u.serving_bs].served_users.append(u.user_id)
        for bs in bss:
            n = max(len(bs.served_users), 1)
            per = max(1, self.cfg.n_resource_blocks // n)
            for uid in bs.served_users:
                bs.rb_alloc[uid] = per

    def compute_sinr(self, user: MobileUser, bss: List[BaseStation]) -> float:
        sbs = bss[user.serving_bs]
        d = user.pos.distance_to(sbs.pos)
        p_los = min(1.0, 18.0/max(d,1)) + np.exp(-d/63)*(1 - min(1.0, 18.0/max(d,1)))
        los = self.ch.rng.random() < p_los
        sg = self.ch.gain_linear(d, sbs.bs_id, user.user_id+1000, sbs.n_ant, los)
        ig = []
        for bs in bss:
            if bs.bs_id == sbs.bs_id: continue
            di = user.pos.distance_to(bs.pos)
            ig.append(self.ch.gain_linear(di, bs.bs_id, user.user_id+1000, bs.n_ant))
        prb_pow = sbs.tx_pow / self.cfg.n_resource_blocks
        # Include jamming interference if user is under attack
        jam = user.jam_interference_w
        return self.ch.sinr(sg, ig, prb_pow, self.cfg.noise_power_w() + jam)

    def throughput(self, sinr_lin: float, n_rbs: int) -> float:
        if sinr_lin <= 0: return 0.0
        bw_rb = (self.cfg.bs_bandwidth_mhz * 1e6) / self.cfg.n_resource_blocks
        se = min(np.log2(1 + sinr_lin), self.cfg.max_se)
        return bw_rb * n_rbs * se

    def evaluate(self, bss: List[BaseStation], users: List[MobileUser]):
        thr_lin = 10**(self.cfg.sinr_threshold_db / 10)
        for u in users:
            sl = self.compute_sinr(u, bss)
            u.sinr_db = 10*np.log10(max(sl, 1e-30))
            n_rbs = bss[u.serving_bs].rb_alloc.get(u.user_id, 1)
            u.throughput = self.throughput(sl, n_rbs)
            u.outage = sl < thr_lin

    def sum_rate(self, users): return sum(u.throughput for u in users)
    def avg_sinr(self, users): return np.mean([u.sinr_db for u in users])
    def outage_rate(self, users): return np.mean([float(u.outage) for u in users])
