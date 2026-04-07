"""
Network entities: base stations, mobile users, sensing targets.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from config import SimConfig


@dataclass
class Position:
    x: float
    y: float
    def distance_to(self, o: 'Position') -> float:
        return np.sqrt((self.x-o.x)**2 + (self.y-o.y)**2)
    def as_array(self): return np.array([self.x, self.y])
    def copy(self): return Position(self.x, self.y)


class BaseStation:
    def __init__(self, bs_id: int, pos: Position, cfg: SimConfig):
        self.bs_id = bs_id
        self.pos = pos
        self.max_tx_pow = cfg.tx_power_w()
        self.tx_pow = cfg.tx_power_w()        # current Tx power (adaptable)
        self.n_ant = cfg.bs_n_antennas
        self.served_users: list = []
        self.rb_alloc: dict = {}  # uid -> n_rbs
        self.sense_power_frac: float = cfg.sensing_power_fraction
        self.active_rbs: int = 0  # for energy accounting

    def total_power_w(self, cfg: SimConfig) -> float:
        """Total BS power: PA + circuit + DSP + sensing compute."""
        pa_power = self.tx_pow / cfg.pa_efficiency
        circuit = cfg.circuit_power_w
        dsp = self.active_rbs * cfg.compute_power_per_rb_w
        sense_compute = cfg.sensing_compute_overhead_w * self.sense_power_frac
        return pa_power + circuit + dsp + sense_compute


class MobileUser:
    def __init__(self, uid: int, pos: Position, cfg: SimConfig, rng: np.random.Generator):
        self.user_id = uid
        self.pos = pos
        self.cfg = cfg
        self.rng = rng
        self.speed = rng.uniform(*cfg.user_speed_range)
        self.direction = rng.uniform(0, 2*np.pi)
        self._wp: Optional[Position] = None
        self.serving_bs: int = 0
        self.sinr_db: float = -20.0
        self.throughput: float = 0.0
        self.outage: bool = False
        self.jam_interference_w: float = 0.0  # injected by attacker, persists through eval

    def move(self, dt: float):
        a = self.cfg.area_size
        if self._wp is None or self.pos.distance_to(self._wp) < 5:
            self._wp = Position(self.rng.uniform(0,a), self.rng.uniform(0,a))
            dx, dy = self._wp.x - self.pos.x, self._wp.y - self.pos.y
            self.direction = np.arctan2(dy, dx)
            self.speed = self.rng.uniform(*self.cfg.user_speed_range)
        s = self.speed * dt
        self.pos.x = np.clip(self.pos.x + s*np.cos(self.direction), 0, a)
        self.pos.y = np.clip(self.pos.y + s*np.sin(self.direction), 0, a)


class SensingTarget:
    def __init__(self, tid: int, pos: Position, cfg: SimConfig, rng: np.random.Generator):
        self.target_id = tid
        self.pos = pos
        self.cfg = cfg
        self.rng = rng
        self.speed = rng.uniform(*cfg.target_speed_range)
        self.direction = rng.uniform(0, 2*np.pi)
        self._wp: Optional[Position] = None
        self.rcs = cfg.radar_cross_section
        self.detected: bool = False
        self.sensing_snr_db: float = -10.0
        self.track_err: float = 100.0

    def move(self, dt: float):
        a = self.cfg.area_size
        if self._wp is None or self.pos.distance_to(self._wp) < 10:
            self._wp = Position(self.rng.uniform(0,a), self.rng.uniform(0,a))
            dx, dy = self._wp.x - self.pos.x, self._wp.y - self.pos.y
            self.direction = np.arctan2(dy, dx)
            self.speed = self.rng.uniform(*self.cfg.target_speed_range)
        s = self.speed * dt
        self.pos.x = np.clip(self.pos.x + s*np.cos(self.direction), 0, a)
        self.pos.y = np.clip(self.pos.y + s*np.sin(self.direction), 0, a)


def create_network(cfg: SimConfig, rng: np.random.Generator):
    a = cfg.area_size
    gs = int(np.ceil(np.sqrt(cfg.n_bs)))
    sp = a / (gs + 1)
    bss, idx = [], 0
    for i in range(gs):
        for j in range(gs):
            if idx >= cfg.n_bs: break
            bss.append(BaseStation(idx, Position((i+1)*sp, (j+1)*sp), cfg))
            idx += 1
    users = [MobileUser(i, Position(rng.uniform(0,a), rng.uniform(0,a)), cfg, rng)
             for i in range(cfg.n_users)]
    targets = [SensingTarget(i, Position(rng.uniform(0,a), rng.uniform(0,a)), cfg, rng)
               for i in range(cfg.n_targets)]
    return bss, users, targets
