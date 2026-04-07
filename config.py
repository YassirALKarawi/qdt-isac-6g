"""
Configuration — Quantum-Assisted Digital Twin ISAC 6G Open RAN.
Calibration:
  Area 1000m, 4 BS => avg UE-BS ~250m, target at 100-800m.
  Sensing: P_d ≈ 0.4-0.95 depending on range, RCS, clutter.
  Anomaly 8% => trust drops noticeably over simulation.
  Twin delay 5 slots => staleness visible with mobility.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class SimConfig:
    seed: int = 42
    # Area
    area_size: float = 1000.0
    n_bs: int = 4
    n_users: int = 40
    n_targets: int = 10
    # Time
    slot_duration: float = 0.01
    n_slots: int = 3000
    n_monte_carlo: int = 50
    # BS
    bs_tx_power_dbm: float = 46.0
    bs_n_antennas: int = 64
    bs_frequency_ghz: float = 3.5
    bs_bandwidth_mhz: float = 100.0
    n_resource_blocks: int = 50
    # Channel
    path_loss_exponent: float = 3.5
    shadow_std_db: float = 8.0
    rician_k_db: float = 5.0
    channel_temporal_corr: float = 0.95
    noise_figure_db: float = 7.0
    thermal_noise_dbm_hz: float = -174.0
    # Mobility
    user_speed_range: Tuple[float, float] = (1.0, 5.0)
    target_speed_range: Tuple[float, float] = (5.0, 20.0)
    # Communication
    sinr_threshold_db: float = 0.0
    max_se: float = 8.0
    # Sensing
    radar_cross_section: float = 1.0
    pfa: float = 1e-4
    sensing_integration_slots: int = 4
    sensing_power_fraction: float = 0.20
    sensing_processing_loss_db: float = 3.0
    clutter_to_noise_ratio_db: float = 5.0
    # Digital twin
    twin_sync_delay_slots: int = 5
    twin_pos_noise_std: float = 5.0
    twin_sinr_noise_std: float = 3.0
    twin_state_decay: float = 0.97
    # Security
    anomaly_prob: float = 0.08
    trust_init: float = 1.0
    trust_decay_rate: float = 0.03
    trust_recovery_rate: float = 0.025
    anomaly_jam_db: float = 15.0
    anomaly_spoof_m: float = 80.0
    # Quantum
    qa_n_candidates: int = 20
    qa_speedup: float = 1.4
    qa_coherence_us: float = 100.0
    qa_gate_fidelity: float = 0.995
    qa_enabled: bool = True
    # Control weights (for composite utility J)
    weight_comm: float = 0.35
    weight_sense: float = 0.25
    weight_sec: float = 0.25
    weight_energy: float = 0.15
    control_lr: float = 0.01
    # Energy model
    circuit_power_w: float = 5.0        # per-BS static circuit power
    compute_power_per_rb_w: float = 0.05 # DSP power per active RB
    sensing_compute_overhead_w: float = 2.0 # extra power for radar processing
    pa_efficiency: float = 0.35          # power amplifier efficiency
    # Baseline
    baseline_id: int = 4
    results_dir: str = "results"
    plots_dir: str = "plots"

    def noise_power_w(self) -> float:
        bw = self.bs_bandwidth_mhz * 1e6
        n_dbm = self.thermal_noise_dbm_hz + 10*np.log10(bw) + self.noise_figure_db
        return 10**(n_dbm/10) * 1e-3

    def tx_power_w(self) -> float:
        return 10**(self.bs_tx_power_dbm/10) * 1e-3


SWEEP_CONFIGS = {
    "user_density": {"param": "n_users", "values": [10, 20, 40, 60, 80]},
    "mobility": {"param": "user_speed_range",
                 "values": [(0.5,1.0),(1.0,5.0),(5.0,15.0),(15.0,30.0)]},
    "anomaly_prob": {"param": "anomaly_prob",
                     "values": [0.0, 0.02, 0.05, 0.10, 0.20]},
    "twin_delay": {"param": "twin_sync_delay_slots",
                   "values": [0, 2, 5, 10, 20, 50]},
    "weight_sweep": {"param": "weight_comm",
                     "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
    "scalability": {"param": "n_bs", "values": [2, 4, 8, 12, 16]},
    "quantum_onoff": {"param": "qa_enabled", "values": [False, True]},
    "twin_fidelity": {"param": "twin_sinr_noise_std",
                      "values": [0.5, 1.0, 3.0, 6.0, 10.0]},
    # --- Sensing stress tests ---
    "target_speed": {"param": "target_speed_range",
                     "values": [(1.0,3.0),(5.0,20.0),(20.0,50.0),(50.0,100.0)]},
    "clutter": {"param": "clutter_to_noise_ratio_db",
                "values": [0.0, 3.0, 5.0, 10.0, 15.0]},
    "sensing_power": {"param": "sensing_power_fraction",
                      "values": [0.05, 0.10, 0.20, 0.30, 0.40]},
    "target_density": {"param": "n_targets",
                       "values": [2, 5, 10, 20, 40]},
}
