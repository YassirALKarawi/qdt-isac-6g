"""
Discrete-time closed-loop simulator for the trust-aware quantum-assisted
digital twin ISAC framework.

Emits rich slot-level telemetry required by the evaluation framework:
  * communication / sensing / trust / twin / screening metrics
  * derived utility, robustness and energy-utility trade-off
  * per-slot runtime and candidate-evaluation counts
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict

from config import SimConfig
from channel import ChannelModel
from network import create_network
from communication import CommModel
from sensing import SensingModel
from digital_twin import DigitalTwin
from security import SecurityModel
from quantum_assist import QuantumAssist
from controller import Controller
from metrics import MetricsCollector


# Extended baseline registry (including ablation mode -1 and new BL5-7)
BL_NAMES: Dict[int, str] = {
    -1: "Ablation",
    0: "Static ISAC",
    1: "Reactive Adaptive",
    2: "DT-guided",
    3: "DT+QA (no Sec)",
    4: "Full Proposed",
    5: "Predictor-UA",
    6: "Robust Heuristic",
    7: "Learning Bandit",
}

# Which baselines use digital twin / QA respectively
_TWIN_BASELINES = {2, 3, 4}
_QA_BASELINES = {3, 4}


def _baseline_uses_twin(cfg: SimConfig) -> bool:
    if cfg.baseline_id == -1:
        return cfg.use_twin
    return cfg.baseline_id in _TWIN_BASELINES


def _baseline_uses_qa(cfg: SimConfig) -> bool:
    if cfg.baseline_id == -1:
        return cfg.use_screening
    return cfg.baseline_id in _QA_BASELINES


def run_one(cfg: SimConfig, run_id: int, mc: MetricsCollector,
             extra: Optional[dict] = None) -> None:
    rng = np.random.default_rng(cfg.seed + run_id * 997)
    ch = ChannelModel(cfg, rng)
    bss, users, targets = create_network(cfg, rng)
    comm = CommModel(cfg, ch)
    sense = SensingModel(cfg, rng)

    use_twin = _baseline_uses_twin(cfg)
    use_qa = _baseline_uses_qa(cfg)

    twin = DigitalTwin(cfg, rng) if use_twin else None
    sec = SecurityModel(cfg, rng)   # all baselines face the same attack environment
    qa = QuantumAssist(cfg, rng) if use_qa else None

    comm.assign_users(bss, users)
    comm.evaluate(bss, users)
    sense.evaluate(bss, targets)
    if twin:
        twin.init(users, targets)
    sec.init(users, targets)

    ctrl = Controller(cfg, ch, comm, sense, twin, sec, qa, rng)

    for slot in range(cfg.n_slots):
        for u in users:
            u.move(cfg.slot_duration)
        for t in targets:
            t.move(cfg.slot_duration)

        ctrl.step(slot, bss, users, targets)

        # --- Primary metrics ---
        sr = comm.sum_rate(users) / 1e6  # Mbps
        total_energy_w = sum(bs.total_power_w(cfg) for bs in bss)
        E_max = cfg.n_bs * (cfg.tx_power_w() / cfg.pa_efficiency
                             + cfg.circuit_power_w
                             + cfg.n_resource_blocks * cfg.compute_power_per_rb_w
                             + cfg.sensing_compute_overhead_w * 0.4)
        energy_norm = total_energy_w / (E_max + 1e-10)

        w_c, w_s, w_sec, w_e = ctrl.w_c, ctrl.w_s, ctrl.w_sec, ctrl.w_e
        R_max = cfg.n_bs * cfg.bs_bandwidth_mhz * cfg.max_se
        R_norm = min(sr / (R_max + 1e-10), 1.0)
        S_norm = sense.avg_utility(targets)

        has_det = (cfg.baseline_id in (2, 4)) or \
                  (cfg.baseline_id == -1 and cfg.use_twin)
        if has_det:
            T_norm = sec.avg_trust()
        else:
            n_attacks = len([a for a in sec.active_attacks if a[3] < 10])
            n_ent = cfg.n_users + cfg.n_targets
            T_norm = max(0.0, 1.0 - 1.5 * n_attacks / (n_ent + 1e-10))

        utility = w_c * R_norm + w_s * S_norm + w_sec * T_norm - w_e * energy_norm

        # --- Twin-related metrics ---
        twin_err = twin.twin_error if twin else 0.0
        twin_conf = twin.avg_confidence() if twin else 1.0
        twin_mismatch = twin.twin_mismatch_mean if twin else 0.0
        twin_mismatch_std = twin.twin_mismatch_std if twin else 0.0
        twin_fidelity = twin.twin_fidelity if twin else 1.0
        stale_penalty = twin.stale_state_penalty if twin else 0.0

        # --- Screening metrics ---
        if qa is not None:
            cand_reduction = qa.candidate_reduction_ratio()
            cost_reduction = qa.search_cost_reduction()
            rank_perc = qa.selected_action_rank_percentile()
            screen_ms = qa.screening_overhead_ms
            total_evals = qa.q_evals + qa.c_evals
            search_cost = qa.q_evals / max(total_evals, 1) if total_evals else 1.0
        else:
            cand_reduction = 1.0
            cost_reduction = 0.0
            rank_perc = 1.0
            screen_ms = 0.0
            search_cost = 1.0

        # --- Trust-aware deployment metrics ---
        fallback_ratio = sec.fallback_deployment_ratio()
        suppress_rate = sec.unsafe_action_suppression_rate()

        # --- Adaptation / robustness derived metrics ---
        w_init = [cfg.weight_comm, cfg.weight_sense,
                  cfg.weight_sec, cfg.weight_energy]
        w_now = [ctrl.w_c, ctrl.w_s, ctrl.w_sec, ctrl.w_e]
        adaptation_gain = sum(abs(a - b) for a, b in zip(w_init, w_now))

        n_active = len([a for a in sec.active_attacks if a[3] < 10])
        robustness_gain = utility / (1.0 + 0.1 * n_active) if n_active > 0 else utility

        sec.end_slot(n_active, utility)

        mc.record({
            'slot': slot,
            'sum_rate': sr,
            'avg_tput': sr / max(len(users), 1),
            'avg_sinr': comm.avg_sinr(users),
            'outage': comm.outage_rate(users),
            'sense_util': S_norm,
            'avg_pd': sense.avg_pd(targets),
            'twin_err': twin_err,
            'twin_conf': twin_conf,
            'twin_mismatch': twin_mismatch,
            'twin_mismatch_std': twin_mismatch_std,
            'twin_fidelity': twin_fidelity,
            'stale_state_penalty': stale_penalty,
            'trust': sec.avg_trust(),
            'det_rate': sec.det_rate(),
            'fa_rate': sec.fa_rate(),
            'latency_ms': ctrl.latency_ms,
            'energy': total_energy_w,
            'energy_norm': energy_norm,
            'utility': utility,
            'search_cost': search_cost,
            'search_cost_reduction': cost_reduction,
            'candidate_reduction_ratio': cand_reduction,
            'selected_action_rank_percentile': rank_perc,
            'screening_overhead_ms': screen_ms,
            'fallback_deployment_ratio': fallback_ratio,
            'unsafe_action_suppression_rate': suppress_rate,
            'adaptation_gain': adaptation_gain,
            'robustness_gain': robustness_gain,
            'n_active_attacks': n_active,
        })

    extras = dict(extra) if extra else {}
    # Attach end-of-run trust/robustness aggregates as run-level fields
    extras.setdefault("trust_degradation_rate", sec.trust_degradation_rate())
    extras.setdefault("safe_control_persistence", sec.safe_control_persistence())
    extras.setdefault("anomaly_containment_score",
                       sec.anomaly_containment_score())
    if qa is not None:
        extras.setdefault("mean_screening_overhead_ms",
                           qa.mean_screening_overhead_ms())
    mc.end_run(run_id, cfg.baseline_id, extras)
    ch.reset()
    if twin:
        twin.reset()
    if sec:
        sec.reset()
    if qa:
        qa.reset()


def run_mc(cfg: SimConfig, mc: MetricsCollector,
            extra: Optional[dict] = None, verbose: bool = True) -> None:
    name = BL_NAMES.get(cfg.baseline_id, f"BL{cfg.baseline_id}")
    if verbose:
        print(f"\n  [{name}] MC={cfg.n_monte_carlo}, Slots={cfg.n_slots}")
    for r in range(cfg.n_monte_carlo):
        run_one(cfg, r, mc, extra)
        if verbose and mc.runs:
            last = mc.runs[-1]
            sr = last.get('sum_rate_mean', 0.0)
            pd = last.get('avg_pd_mean', 0.0)
            te = last.get('twin_err_mean', 0.0)
            tr = last.get('trust_mean', 0.0)
            fb = last.get('fallback_deployment_ratio_mean', 0.0)
            print(f"    Run {r+1}/{cfg.n_monte_carlo}: "
                  f"SR={sr:.0f} Pd={pd:.3f} TwErr={te:.4f} "
                  f"Trust={tr:.3f} Fallback={fb:.3f}")
