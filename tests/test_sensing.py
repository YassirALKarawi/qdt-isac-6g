import numpy as np
from config import SimConfig
from network import BaseStation, Position, SensingTarget
from sensing import SensingModel


def test_higher_sensing_power_increases_pd():
    rng = np.random.default_rng(123)
    cfg = SimConfig(seed=123)

    bs = BaseStation(0, Position(100.0, 100.0), cfg)
    tgt = SensingTarget(0, Position(200.0, 200.0), cfg, rng)
    sensing = SensingModel(cfg, rng)

    bs.sense_power_frac = 0.10
    snr_low = sensing.radar_snr(bs, tgt)
    pd_low = sensing.prob_detection(snr_low)

    bs.sense_power_frac = 0.30
    snr_high = sensing.radar_snr(bs, tgt)
    pd_high = sensing.prob_detection(snr_high)

    assert snr_high > snr_low
    assert pd_high >= pd_low


def test_pd_decreases_with_distance():
    rng = np.random.default_rng(42)
    cfg = SimConfig(seed=42)
    sensing = SensingModel(cfg, rng)

    bs = BaseStation(0, Position(0.0, 0.0), cfg)
    tgt_near = SensingTarget(0, Position(100.0, 0.0), cfg, rng)
    tgt_far = SensingTarget(1, Position(800.0, 0.0), cfg, rng)

    snr_near = sensing.radar_snr(bs, tgt_near)
    snr_far = sensing.radar_snr(bs, tgt_far)

    assert snr_near > snr_far
    assert sensing.prob_detection(snr_near) > sensing.prob_detection(snr_far)
