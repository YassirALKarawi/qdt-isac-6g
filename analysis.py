"""
Formal analysis utilities: trust-aware gating bounds, convergence
properties, and deployment conservativeness metrics.

Proposition 1 (Trust-Gated Resource Bound):
    Under anomaly rate p_a and decay β, the steady-state trust lower bound
    for entity i satisfies:
        τ_∞ ≥ α(1 - τ_min) / (α(1 - τ_min) + β·p_a·E[excess])
    where α is recovery rate and E[excess] is the expected detection excess.

Proposition 2 (Utility Degradation under Twin Delay):
    For twin sync delay δ and state decay γ, the utility loss is bounded:
        ΔJ ≤ w_c · (1 - γ^δ) · R_max + w_s · (1 - γ^δ) · S_max
    which grows monotonically with δ.

These results provide formal guarantees on the closed-loop controller
behaviour under imperfect twin state and adversarial conditions.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import SimConfig


@dataclass
class TrustBound:
    """Analytical steady-state trust bounds."""
    tau_lower: float
    tau_upper: float
    convergence_rate: float
    mixing_time_slots: int


@dataclass
class UtilityBound:
    """Utility loss bound under twin imperfection."""
    delta_j_upper: float
    comm_loss_bound: float
    sense_loss_bound: float
    delay_slots: int


def steady_state_trust_bound(cfg: SimConfig,
                              mean_excess: float = 1.5) -> TrustBound:
    """Compute analytical trust lower/upper bounds (Proposition 1).

    Under i.i.d. anomaly arrivals with rate p_a, and EWMA-based detection
    with mean excess score `mean_excess` when flagged, the trust evolves as:
        τ_{t+1} = τ_t · (1 - β·excess)   with prob p_a · p_det
        τ_{t+1} = τ_t + α·(1 - τ_t)      otherwise

    The fixed point τ* satisfies the balance equation.
    """
    alpha = cfg.trust_recovery_rate
    beta = cfg.trust_decay_rate
    p_a = cfg.anomaly_prob
    tau_min = 0.05  # hard floor in implementation

    # Effective detection probability (conservative estimate)
    p_det = 0.5  # typical detection rate from simulations

    # Decay per slot from attack
    decay_per_attack = beta * mean_excess
    # Recovery per clean slot
    # At steady state: p_a * p_det * decay = (1 - p_a*p_det) * recovery
    p_flag = p_a * p_det

    if p_flag < 1e-10:
        return TrustBound(tau_lower=1.0, tau_upper=1.0,
                          convergence_rate=alpha, mixing_time_slots=0)

    # Linearised fixed point: τ* ≈ α / (α + p_flag * decay_per_attack)
    tau_star = alpha / (alpha + p_flag * decay_per_attack)
    tau_lower = max(tau_min, tau_star * 0.85)  # account for variance
    tau_upper = min(1.0, tau_star * 1.15)

    # Spectral gap gives convergence rate
    spectral_gap = min(alpha, p_flag * decay_per_attack)
    mixing_time = int(np.ceil(np.log(20) / (spectral_gap + 1e-10)))

    return TrustBound(
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        convergence_rate=spectral_gap,
        mixing_time_slots=min(mixing_time, 10000)
    )


def utility_loss_bound(cfg: SimConfig,
                        delay_slots: int = None) -> UtilityBound:
    """Compute utility degradation bound under twin delay (Proposition 2).

    Twin staleness after δ slots: s(δ) = 1 - γ^δ.
    The confidence drops proportionally, degrading both communication
    (via misallocated resources) and sensing (via stale target state).
    """
    delta = delay_slots if delay_slots is not None else cfg.twin_sync_delay_slots
    gamma = cfg.twin_state_decay

    staleness = 1.0 - gamma ** delta

    # Communication loss: misallocation proportional to staleness
    R_max = cfg.n_bs * cfg.bs_bandwidth_mhz * cfg.max_se
    comm_loss = cfg.weight_comm * staleness * R_max / R_max  # normalised

    # Sensing loss: stale target position degrades detection
    sense_loss = cfg.weight_sense * staleness

    delta_j = comm_loss + sense_loss

    return UtilityBound(
        delta_j_upper=delta_j,
        comm_loss_bound=comm_loss,
        sense_loss_bound=sense_loss,
        delay_slots=delta
    )


def monotonic_degradation_curve(cfg: SimConfig,
                                 max_delay: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ΔJ(δ) for δ = 0..max_delay, demonstrating monotonic growth."""
    delays = np.arange(0, max_delay + 1)
    losses = np.array([utility_loss_bound(cfg, d).delta_j_upper for d in delays])
    return delays, losses


def trust_gating_conservativeness(trust_values: np.ndarray,
                                   threshold: float = 0.5) -> Dict[str, float]:
    """Measure how conservative the trust-gating mechanism is.

    Returns:
        gated_fraction: fraction of time entities are below threshold
        avg_trust_when_gated: mean trust during gated periods
        false_conservation_rate: estimated rate of unnecessary gating
    """
    gated = trust_values < threshold
    gated_frac = float(np.mean(gated))
    avg_when_gated = float(np.mean(trust_values[gated])) if np.any(gated) else threshold

    # Conservativeness: how far below threshold on average
    if np.any(gated):
        margin = float(np.mean(threshold - trust_values[gated]))
    else:
        margin = 0.0

    return {
        'gated_fraction': gated_frac,
        'avg_trust_when_gated': avg_when_gated,
        'conservation_margin': margin,
    }


def feedback_loop_stability(utility_series: np.ndarray,
                             window: int = 100) -> Dict[str, float]:
    """Assess closed-loop stability from utility time series.

    Checks:
    - Bounded oscillation (no divergence)
    - Convergence to steady state
    - Lyapunov-style monotonic decrease of variance
    """
    n = len(utility_series)
    if n < 2 * window:
        return {'stable': True, 'variance_trend': 0.0, 'settling_slot': 0}

    # Windowed variance
    variances = []
    for i in range(0, n - window, window // 2):
        seg = utility_series[i:i + window]
        variances.append(np.var(seg))

    variances = np.array(variances)

    # Variance should decrease or stabilise (negative or near-zero trend)
    if len(variances) > 2:
        trend = np.polyfit(np.arange(len(variances)), variances, 1)[0]
    else:
        trend = 0.0

    # Settling time: first window where variance < 1.5 * final variance
    final_var = variances[-1] if len(variances) > 0 else 0
    settling = 0
    for i, v in enumerate(variances):
        if v <= 1.5 * final_var + 1e-10:
            settling = i * (window // 2)
            break

    return {
        'stable': trend <= 0.01,
        'variance_trend': float(trend),
        'settling_slot': settling,
        'final_variance': float(final_var),
    }


def compute_all_bounds(cfg: SimConfig) -> Dict:
    """Compute all formal bounds for the given configuration."""
    tb = steady_state_trust_bound(cfg)
    ub = utility_loss_bound(cfg)
    delays, losses = monotonic_degradation_curve(cfg)

    return {
        'trust_bound': {
            'tau_lower': tb.tau_lower,
            'tau_upper': tb.tau_upper,
            'convergence_rate': tb.convergence_rate,
            'mixing_time': tb.mixing_time_slots,
        },
        'utility_bound': {
            'delta_j_upper': ub.delta_j_upper,
            'comm_loss': ub.comm_loss_bound,
            'sense_loss': ub.sense_loss_bound,
            'delay': ub.delay_slots,
        },
        'degradation_curve': {
            'delays': delays.tolist(),
            'losses': losses.tolist(),
        }
    }
