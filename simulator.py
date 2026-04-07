"""
Discrete-time closed-loop simulator.
"""
import numpy as np
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

BL_NAMES = {0:"Static ISAC", 1:"Adaptive ISAC", 2:"DT (no QA)",
            3:"DT+QA (no Sec)", 4:"Full Proposed"}


def run_one(cfg: SimConfig, run_id: int, mc: MetricsCollector, extra=None):
    rng = np.random.default_rng(cfg.seed + run_id * 997)
    ch = ChannelModel(cfg, rng)
    bss, users, targets = create_network(cfg, rng)
    comm = CommModel(cfg, ch)
    sense = SensingModel(cfg, rng)

    use_twin = cfg.baseline_id >= 2
    use_qa   = cfg.baseline_id in [3, 4]

    twin = DigitalTwin(cfg, rng) if use_twin else None
    sec  = SecurityModel(cfg, rng)  # ALL baselines face attacks
    qa   = QuantumAssist(cfg, rng) if use_qa else None

    # Initial eval to seed twin
    comm.assign_users(bss, users)
    comm.evaluate(bss, users)
    sense.evaluate(bss, targets)
    if twin: twin.init(users, targets)
    sec.init(users, targets)

    ctrl = Controller(cfg, ch, comm, sense, twin, sec, qa, rng)

    for slot in range(cfg.n_slots):
        # Mobility
        for u in users: u.move(cfg.slot_duration)
        for t in targets: t.move(cfg.slot_duration)
        # Control step
        ctrl.step(slot, bss, users, targets)
        # Metrics
        sr = comm.sum_rate(users) / 1e6  # Mbps
        # --- Energy model (variable per-BS) ---
        total_energy_w = sum(bs.total_power_w(cfg) for bs in bss)
        E_max = cfg.n_bs * (cfg.tx_power_w()/cfg.pa_efficiency + cfg.circuit_power_w +
                            cfg.n_resource_blocks*cfg.compute_power_per_rb_w +
                            cfg.sensing_compute_overhead_w*0.4)
        energy_norm = total_energy_w / (E_max + 1e-10)
        # --- Publication-grade composite utility J ---
        # Uses controller's ADAPTIVE weights (not fixed config)
        w_c, w_s, w_sec, w_e = ctrl.w_c, ctrl.w_s, ctrl.w_sec, ctrl.w_e
        R_max = cfg.n_bs * cfg.bs_bandwidth_mhz * cfg.max_se
        R_norm = min(sr / (R_max + 1e-10), 1.0)
        S_norm = sense.avg_utility(targets)
        has_det = cfg.baseline_id in [2, 4]
        if has_det:
            T_norm = sec.avg_trust()
        else:
            n_attacks = len([a for a in sec.active_attacks if a[3] < 10])
            n_ent = cfg.n_users + cfg.n_targets
            T_norm = max(0.0, 1.0 - 1.5 * n_attacks / (n_ent + 1e-10))

        utility = w_c * R_norm + w_s * S_norm + w_sec * T_norm - w_e * energy_norm

        # --- Derived metrics for paper Results section ---
        # search_cost: quantum evals / classical evals (lower = better)
        if qa:
            total_evals = qa.q_evals + qa.c_evals
            search_cost = qa.q_evals / max(total_evals, 1)
        else:
            search_cost = 1.0  # classical = full cost

        # adaptation_gain: weight divergence from initial (shows controller learning)
        w_init = [cfg.weight_comm, cfg.weight_sense, cfg.weight_sec, cfg.weight_energy]
        w_now = [ctrl.w_c, ctrl.w_s, ctrl.w_sec, ctrl.w_e]
        adaptation_gain = sum(abs(a - b) for a, b in zip(w_init, w_now))

        # robustness_gain: utility maintained despite active attacks
        n_active = len([a for a in sec.active_attacks if a[3] < 10])
        robustness_gain = utility / (1.0 + 0.1 * n_active) if n_active > 0 else utility

        mc.record({
            'slot': slot,
            'sum_rate': sr,
            'avg_tput': sr / max(len(users), 1),
            'avg_sinr': comm.avg_sinr(users),
            'outage': comm.outage_rate(users),
            'sense_util': S_norm,
            'avg_pd': sense.avg_pd(targets),
            'twin_err': twin.twin_error if twin else 0.0,
            'twin_conf': twin.avg_confidence() if twin else 1.0,
            'trust': sec.avg_trust(),
            'det_rate': sec.det_rate(),
            'fa_rate': sec.fa_rate(),
            'latency_ms': ctrl.latency_ms,
            'energy': total_energy_w,
            'energy_norm': energy_norm,
            'utility': utility,
            'search_cost': search_cost,
            'adaptation_gain': adaptation_gain,
            'robustness_gain': robustness_gain,
        })

    mc.end_run(run_id, cfg.baseline_id, extra)
    ch.reset()
    if twin: twin.reset()
    if sec: sec.reset()
    if qa: qa.reset()


def run_mc(cfg: SimConfig, mc: MetricsCollector, extra=None, verbose=True):
    name = BL_NAMES.get(cfg.baseline_id, f"BL{cfg.baseline_id}")
    if verbose:
        print(f"\n  [{name}] MC={cfg.n_monte_carlo}, Slots={cfg.n_slots}")
    for r in range(cfg.n_monte_carlo):
        run_one(cfg, r, mc, extra)
        if verbose:
            last = mc.runs[-1] if mc.runs else {}
            sr = last.get('sum_rate_mean', 0)
            pd = last.get('avg_pd_mean', 0)
            te = last.get('twin_err_mean', 0)
            tr = last.get('trust_mean', 0)
            print(f"    Run {r+1}/{cfg.n_monte_carlo}: "
                  f"SR={sr:.0f} Pd={pd:.3f} TwErr={te:.4f} Trust={tr:.3f}")
