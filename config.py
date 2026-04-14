"""
Configuration — Trust-Aware Quantum-Assisted Digital Twin ISAC 6G Open RAN.

This module defines the simulation configuration dataclass (`SimConfig`),
parameter sweeps (`SWEEP_CONFIGS`) and experiment profiles (`PROFILES`).

The simulator is organised around six experiment families:
    - baseline       : full baseline comparison (8 methods)
    - ablation       : module-by-module isolation of contributions
    - anomaly        : robustness under cyber-physical anomalies
    - twin_delay     : graceful degradation under twin imperfection
    - scalability    : sweeps over network/search-space scale
    - runtime        : runtime / complexity evaluation

Each family maps to its own output directory under `results/` and its own
reproducibility metadata, recorded automatically in `metadata.json`.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any, List
import copy
import numpy as np


# =============================================================================
# Core simulation configuration
# =============================================================================
@dataclass
class SimConfig:
    # --- Reproducibility ----------------------------------------------------
    seed: int = 42
    n_monte_carlo: int = 50
    n_slots: int = 3000
    slot_duration: float = 0.01
    steady_state_fraction: float = 0.5

    # --- Topology -----------------------------------------------------------
    area_size: float = 1000.0
    n_bs: int = 4
    n_users: int = 40
    n_targets: int = 10

    # --- Base station -------------------------------------------------------
    bs_tx_power_dbm: float = 46.0
    bs_n_antennas: int = 64
    bs_frequency_ghz: float = 3.5
    bs_bandwidth_mhz: float = 100.0
    n_resource_blocks: int = 50

    # --- Channel ------------------------------------------------------------
    path_loss_exponent: float = 3.5
    shadow_std_db: float = 8.0
    rician_k_db: float = 5.0
    channel_temporal_corr: float = 0.95
    noise_figure_db: float = 7.0
    thermal_noise_dbm_hz: float = -174.0

    # --- Mobility -----------------------------------------------------------
    user_speed_range: Tuple[float, float] = (1.0, 5.0)
    target_speed_range: Tuple[float, float] = (5.0, 20.0)

    # --- Communication ------------------------------------------------------
    sinr_threshold_db: float = 0.0
    max_se: float = 8.0

    # --- Sensing ------------------------------------------------------------
    radar_cross_section: float = 1.0
    pfa: float = 1e-4
    sensing_integration_slots: int = 4
    sensing_power_fraction: float = 0.20
    sensing_processing_loss_db: float = 3.0
    clutter_to_noise_ratio_db: float = 5.0

    # --- Digital twin -------------------------------------------------------
    twin_sync_delay_slots: int = 5
    twin_pos_noise_std: float = 5.0
    twin_sinr_noise_std: float = 3.0
    twin_state_decay: float = 0.97
    twin_fidelity_target: float = 0.9   # nominal fidelity reference
    twin_fidelity_floor: float = 0.3    # minimum fidelity clip
    twin_stale_penalty_weight: float = 0.5

    # --- Security / trust ---------------------------------------------------
    anomaly_prob: float = 0.08
    anomaly_mode: str = "mixed"     # {mixed, spoof, jam, burst, persistent, none}
    trust_init: float = 1.0
    trust_decay_rate: float = 0.03
    trust_recovery_rate: float = 0.025
    anomaly_jam_db: float = 15.0
    anomaly_spoof_m: float = 80.0
    # Trust-aware deployment gating
    trust_gate_threshold: float = 0.6
    trust_hard_floor: float = 0.35
    fallback_rb_scale: float = 0.6
    fallback_power_scale: float = 0.7

    # --- Quantum-assisted candidate screening -------------------------------
    qa_enabled: bool = True
    qa_n_candidates: int = 20
    qa_shortlist_size: int = 6
    qa_speedup: float = 1.4             # effective sqrt-N amplification factor
    qa_coherence_us: float = 100.0
    qa_gate_fidelity: float = 0.995

    # --- Controller weights --------------------------------------------------
    weight_comm: float = 0.35
    weight_sense: float = 0.25
    weight_sec: float = 0.25
    weight_energy: float = 0.15
    control_lr: float = 0.01

    # --- Energy model -------------------------------------------------------
    circuit_power_w: float = 5.0
    compute_power_per_rb_w: float = 0.05
    sensing_compute_overhead_w: float = 2.0
    pa_efficiency: float = 0.35

    # --- Baseline selection -------------------------------------------------
    # -1: ablation mode (uses the ablation flags directly)
    #  0: Static ISAC
    #  1: Reactive Adaptive ISAC
    #  2: DT-guided (no QA, no trust gating)
    #  3: DT + QA (no security / no trust gating)
    #  4: Full Proposed (trust-aware DT + quantum-assisted screening)
    #  5: Predictor-based uncertainty-aware controller (no DT)
    #  6: Robust min-max heuristic controller
    #  7: Learning-based epsilon-greedy bandit controller
    baseline_id: int = 4

    # --- Ablation flags (only used when baseline_id == -1) ------------------
    use_twin: bool = True
    use_trust_gating: bool = True
    use_screening: bool = True
    use_adaptive_weights: bool = True
    use_mismatch_comp: bool = True

    # --- Experiment metadata (filled by runners) ----------------------------
    experiment_family: str = "adhoc"
    profile_name: str = "default"
    scenario_name: str = "default"
    sweep_variable: str = ""

    # --- Output paths -------------------------------------------------------
    results_dir: str = "results"
    plots_dir: str = "figures"

    # --- Derived helpers ----------------------------------------------------
    def noise_power_w(self) -> float:
        bw = self.bs_bandwidth_mhz * 1e6
        n_dbm = self.thermal_noise_dbm_hz + 10*np.log10(bw) + self.noise_figure_db
        return 10**(n_dbm/10) * 1e-3

    def tx_power_w(self) -> float:
        return 10**(self.bs_tx_power_dbm/10) * 1e-3

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view of configuration (for metadata logging)."""
        d = asdict(self)
        # coerce tuples into lists for JSON
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def clone(self, **overrides) -> "SimConfig":
        c = copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(c, k, v)
        return c


# =============================================================================
# Parameter sweeps (used by experiment runners)
# =============================================================================
SWEEP_CONFIGS: Dict[str, Dict[str, Any]] = {
    # --- Anomaly / security ---
    "anomaly_prob": {"param": "anomaly_prob",
                      "values": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]},
    "anomaly_mode": {"param": "anomaly_mode",
                      "values": ["none", "spoof", "jam", "burst",
                                  "persistent", "mixed"]},

    # --- Twin imperfection ---
    "twin_delay": {"param": "twin_sync_delay_slots",
                    "values": [0, 2, 5, 10, 20, 50, 100]},
    "twin_fidelity": {"param": "twin_sinr_noise_std",
                       "values": [0.5, 1.0, 3.0, 6.0, 10.0, 15.0]},
    "twin_pos_noise": {"param": "twin_pos_noise_std",
                        "values": [0.5, 2.0, 5.0, 10.0, 20.0]},

    # --- Scale / complexity ---
    "user_density": {"param": "n_users",
                      "values": [10, 20, 40, 60, 80, 120]},
    "target_density": {"param": "n_targets",
                        "values": [2, 5, 10, 20, 40]},
    "bs_density": {"param": "n_bs",
                    "values": [2, 4, 8, 12, 16]},
    "candidate_pool": {"param": "qa_n_candidates",
                        "values": [4, 8, 16, 32, 64, 128]},
    "shortlist_ratio": {"param": "qa_shortlist_size",
                         "values": [2, 4, 6, 8, 12, 16]},

    # --- Physical / ISAC ---
    "mobility": {"param": "user_speed_range",
                  "values": [(0.5, 1.0), (1.0, 5.0), (5.0, 15.0),
                              (15.0, 30.0), (30.0, 60.0)]},
    "target_speed": {"param": "target_speed_range",
                      "values": [(1.0, 3.0), (5.0, 20.0),
                                  (20.0, 50.0), (50.0, 100.0)]},
    "clutter": {"param": "clutter_to_noise_ratio_db",
                 "values": [0.0, 3.0, 5.0, 10.0, 15.0]},
    "sensing_power": {"param": "sensing_power_fraction",
                       "values": [0.05, 0.10, 0.20, 0.30, 0.40]},
    "weight_sweep": {"param": "weight_comm",
                      "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},

    # --- Quantum ---
    "quantum_onoff": {"param": "qa_enabled", "values": [False, True]},
}


# =============================================================================
# Experiment profiles — default overrides per experiment family
# =============================================================================
PROFILES: Dict[str, Dict[str, Any]] = {
    # Baseline comparison: full method list, balanced defaults, longer runs.
    "profile_baseline": {
        "experiment_family": "baseline",
        "n_monte_carlo": 20,
        "n_slots": 1500,
        "anomaly_prob": 0.08,
        "twin_sync_delay_slots": 5,
    },
    # Ablation: ablation mode (baseline_id=-1), isolated toggles.
    "profile_ablation": {
        "experiment_family": "ablation",
        "n_monte_carlo": 20,
        "n_slots": 1500,
        "anomaly_prob": 0.10,
        "twin_sync_delay_slots": 8,
    },
    # Anomaly: sweep anomaly_prob / anomaly_mode on Full Proposed and strong
    # baselines.
    "profile_anomaly": {
        "experiment_family": "anomaly",
        "n_monte_carlo": 15,
        "n_slots": 1200,
    },
    # Twin delay / fidelity: sweep twin imperfection severity.
    "profile_twin_delay": {
        "experiment_family": "twin_delay",
        "n_monte_carlo": 15,
        "n_slots": 1200,
        "anomaly_prob": 0.05,
    },
    # Scalability: sweep network size / search space.
    "profile_scalability": {
        "experiment_family": "scalability",
        "n_monte_carlo": 8,
        "n_slots": 800,
    },
    # Runtime / complexity: shorter runs, per-slot timing.
    "profile_runtime": {
        "experiment_family": "runtime",
        "n_monte_carlo": 5,
        "n_slots": 600,
    },
}


# =============================================================================
# Twin imperfection regimes (used by twin-delay experiment)
# =============================================================================
TWIN_REGIMES: Dict[str, Dict[str, Any]] = {
    "low": {
        "twin_sync_delay_slots": 1,
        "twin_pos_noise_std": 1.0,
        "twin_sinr_noise_std": 1.0,
    },
    "medium": {
        "twin_sync_delay_slots": 8,
        "twin_pos_noise_std": 5.0,
        "twin_sinr_noise_std": 3.0,
    },
    "severe": {
        "twin_sync_delay_slots": 30,
        "twin_pos_noise_std": 15.0,
        "twin_sinr_noise_std": 8.0,
    },
}


# =============================================================================
# Anomaly scenarios (used by anomaly-robustness campaign)
# =============================================================================
ANOMALY_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "no_attack":      {"anomaly_prob": 0.00, "anomaly_mode": "none"},
    "low_anomaly":    {"anomaly_prob": 0.03, "anomaly_mode": "mixed"},
    "medium_anomaly": {"anomaly_prob": 0.08, "anomaly_mode": "mixed"},
    "high_anomaly":   {"anomaly_prob": 0.20, "anomaly_mode": "mixed"},
    "burst":          {"anomaly_prob": 0.12, "anomaly_mode": "burst"},
    "persistent":     {"anomaly_prob": 0.10, "anomaly_mode": "persistent"},
    "spoof_dominant": {"anomaly_prob": 0.12, "anomaly_mode": "spoof"},
    "jam_dominant":   {"anomaly_prob": 0.12, "anomaly_mode": "jam"},
    "mixed_attack":   {"anomaly_prob": 0.15, "anomaly_mode": "mixed"},
}


# =============================================================================
# Ablation variants
# =============================================================================
# Each variant is a dict of ablation flag overrides.
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "no_dt": {
        "use_twin": False, "use_trust_gating": False,
        "use_screening": False, "use_adaptive_weights": False,
        "use_mismatch_comp": False,
    },
    "dt_only": {
        "use_twin": True, "use_trust_gating": False,
        "use_screening": False, "use_adaptive_weights": False,
        "use_mismatch_comp": False,
    },
    "dt_trust": {
        "use_twin": True, "use_trust_gating": True,
        "use_screening": False, "use_adaptive_weights": False,
        "use_mismatch_comp": False,
    },
    "dt_screening": {
        "use_twin": True, "use_trust_gating": False,
        "use_screening": True, "use_adaptive_weights": False,
        "use_mismatch_comp": False,
    },
    "dt_trust_screening": {
        "use_twin": True, "use_trust_gating": True,
        "use_screening": True, "use_adaptive_weights": False,
        "use_mismatch_comp": False,
    },
    "full": {
        "use_twin": True, "use_trust_gating": True,
        "use_screening": True, "use_adaptive_weights": True,
        "use_mismatch_comp": True,
    },
}
