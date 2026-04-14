"""
Trust-Aware Quantum-Assisted Digital Twin ISAC — 6G Open RAN Simulator.

Command-line entry point for all six experiment families.

Examples
--------
    # Run every experiment family with default profiles
    python main.py --all

    # Baseline comparison (all 8 baselines)
    python main.py --family baseline

    # Ablation study (uses baseline_id=-1 with flag variants)
    python main.py --family ablation

    # Anomaly-robustness sweep
    python main.py --family anomaly

    # Twin delay / fidelity sweep with regimes
    python main.py --family twin_delay

    # Scalability (users, targets, BS, candidate pool, shortlist)
    python main.py --family scalability

    # Runtime / complexity
    python main.py --family runtime

    # Quick smoke run (small MC × few slots)
    python main.py --family baseline --quick

    # Single baseline, direct run
    python main.py --baseline 4 --mc 5 --slots 300
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd

from config import SimConfig
from metrics import MetricsCollector
from simulator import run_mc, BL_NAMES

from experiments import (
    run_baseline_experiment,
    run_ablation_experiment,
    run_anomaly_sweep,
    run_twin_delay_sweep,
    run_scalability_experiment,
    run_runtime_experiment,
)
from plotting import (
    plot_baseline_bars, plot_radar_multiobjective,
    plot_trust_trajectory, plot_twin_mismatch_trajectory,
    plot_sweep_metric, plot_ablation_summary,
    plot_search_cost_vs_shortlist, plot_energy_utility,
    plot_runtime_vs_candidates,
    make_baseline_table, make_runtime_table,
)


FAMILIES = ("baseline", "ablation", "anomaly", "twin_delay",
            "scalability", "runtime")


def _apply_quick(cfg: SimConfig) -> SimConfig:
    cfg.n_monte_carlo = 2
    cfg.n_slots = 150
    return cfg


def _apply_overrides(cfg: SimConfig, args) -> SimConfig:
    if args.mc is not None:
        cfg.n_monte_carlo = args.mc
    if args.slots is not None:
        cfg.n_slots = args.slots
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def _make_figures_for_baseline(result) -> None:
    d = result.output_dir / "figures"
    plot_baseline_bars(result.summary, str(d))
    plot_radar_multiobjective(result.summary, str(d))
    plot_energy_utility(result.summary, str(d))
    # Trust and twin mismatch trajectories from slot data
    if result.slots is not None and not result.slots.empty:
        plot_trust_trajectory(result.slots, str(d))
        plot_twin_mismatch_trajectory(result.slots, str(d))
    # Summary table
    tbl = make_baseline_table(result.summary)
    if not tbl.empty:
        tbl.to_csv(result.output_dir / "summary_table.csv", index=False)
        print("\n" + tbl.to_string(index=False))


def _make_figures_for_ablation(result) -> None:
    d = result.output_dir / "figures"
    plot_ablation_summary(result.summary, str(d))


def _make_figures_for_anomaly(result) -> None:
    d = result.output_dir / "figures"
    if result.summary.empty:
        return
    # Use anomaly_prob from run extras if present
    sub = result.summary.copy()
    if 'anomaly_prob' in sub.columns:
        plot_sweep_metric(sub, str(d), 'anomaly_prob', 'utility',
                           'Utility', xlabel='Anomaly probability')
        plot_sweep_metric(sub, str(d), 'anomaly_prob', 'trust',
                           'Trust', xlabel='Anomaly probability')
        plot_sweep_metric(sub, str(d), 'anomaly_prob', 'fallback_deployment_ratio',
                           'Fallback deployment ratio',
                           xlabel='Anomaly probability')


def _make_figures_for_twin_delay(result) -> None:
    d = result.output_dir / "figures"
    if result.summary.empty:
        return
    sub_delay = result.summary[result.summary.get('sweep', '') == 'twin_delay'] \
                if 'sweep' in result.summary.columns else pd.DataFrame()
    sub_fid = result.summary[result.summary.get('sweep', '') == 'twin_fidelity'] \
              if 'sweep' in result.summary.columns else pd.DataFrame()
    if not sub_delay.empty:
        plot_sweep_metric(sub_delay, str(d), 'sweep_value', 'utility',
                           'Utility', xlabel='Twin sync delay (slots)',
                           filename='utility_vs_twin_delay')
        plot_sweep_metric(sub_delay, str(d), 'sweep_value', 'avg_pd',
                           'Avg $P_d$', xlabel='Twin sync delay (slots)',
                           filename='pd_vs_twin_delay')
        plot_sweep_metric(sub_delay, str(d), 'sweep_value', 'twin_mismatch',
                           'Twin mismatch', xlabel='Twin sync delay (slots)',
                           filename='mismatch_vs_twin_delay')
    if not sub_fid.empty:
        plot_sweep_metric(sub_fid, str(d), 'sweep_value', 'avg_pd',
                           'Avg $P_d$', xlabel='Twin SINR noise (dB)',
                           filename='pd_vs_twin_fidelity')
        plot_sweep_metric(sub_fid, str(d), 'sweep_value', 'utility',
                           'Utility', xlabel='Twin SINR noise (dB)',
                           filename='utility_vs_twin_fidelity')


def _make_figures_for_scalability(result) -> None:
    d = result.output_dir / "figures"
    if result.summary.empty:
        return
    for sw in ('user_density', 'target_density', 'bs_density',
                'candidate_pool', 'shortlist_ratio'):
        sub = result.summary[result.summary.get('sweep', '') == sw] \
              if 'sweep' in result.summary.columns else pd.DataFrame()
        if sub.empty:
            continue
        plot_sweep_metric(sub, str(d), 'sweep_value', 'utility',
                           'Utility', xlabel=sw,
                           filename=f'scalability_{sw}_utility')
        plot_sweep_metric(sub, str(d), 'sweep_value', 'latency_ms',
                           'Latency (ms)', xlabel=sw,
                           filename=f'scalability_{sw}_latency')


def _make_figures_for_runtime(result) -> None:
    d = result.output_dir / "figures"
    if result.summary.empty:
        return
    plot_runtime_vs_candidates(result.summary, str(d))
    plot_search_cost_vs_shortlist(result.summary, str(d))
    # Summary table
    per_bl = result.summary[result.summary.get('kind') == 'per_baseline'] \
             if 'kind' in result.summary.columns else result.summary
    if not per_bl.empty:
        tbl = make_runtime_table(per_bl)
        tbl.to_csv(result.output_dir / "runtime_table.csv", index=False)
        print("\n" + tbl.to_string(index=False))


RUNNERS = {
    "baseline":    (run_baseline_experiment,    _make_figures_for_baseline),
    "ablation":    (run_ablation_experiment,    _make_figures_for_ablation),
    "anomaly":     (run_anomaly_sweep,          _make_figures_for_anomaly),
    "twin_delay":  (run_twin_delay_sweep,       _make_figures_for_twin_delay),
    "scalability": (run_scalability_experiment, _make_figures_for_scalability),
    "runtime":     (run_runtime_experiment,     _make_figures_for_runtime),
}


def run_family(family: str, cfg: SimConfig, verbose: bool = True):
    runner, plot_fn = RUNNERS[family]
    print("=" * 72)
    print(f"  Running experiment family: {family}")
    print("=" * 72)
    result = runner(cfg, verbose=verbose)
    plot_fn(result)
    print(f"\n[{family}] results in {result.output_dir}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Trust-Aware Quantum-Assisted Digital Twin ISAC 6G Simulator")
    ap.add_argument('--family', type=str, default=None, choices=FAMILIES,
                     help="Run a single experiment family")
    ap.add_argument('--all', action='store_true',
                     help="Run every experiment family")
    ap.add_argument('--baseline', type=int, default=None,
                     help="Run a single baseline (direct, no experiment family)")
    ap.add_argument('--quick', action='store_true',
                     help="Quick smoke run (small MC × few slots)")
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mc', type=int, default=None,
                     help="Override number of Monte Carlo runs")
    ap.add_argument('--slots', type=int, default=None,
                     help="Override number of slots per run")
    ap.add_argument('--output', type=str, default='results',
                     help="Base results directory")
    args = ap.parse_args()

    cfg = SimConfig()
    if args.quick:
        cfg = _apply_quick(cfg)
    cfg = _apply_overrides(cfg, args)
    cfg.results_dir = args.output

    t0 = time.time()

    if args.baseline is not None:
        cfg.baseline_id = args.baseline
        odir = Path(cfg.results_dir) / f"bl{args.baseline}"
        odir.mkdir(parents=True, exist_ok=True)
        mc = MetricsCollector(steady_state_fraction=cfg.steady_state_fraction)
        run_mc(cfg, mc, verbose=True)
        mc.save(str(odir), metadata={
            "family": "adhoc_baseline",
            "baseline_id": args.baseline,
            "baseline_name": BL_NAMES.get(args.baseline,
                                             f"BL{args.baseline}"),
            "seed": cfg.seed,
            "n_monte_carlo": cfg.n_monte_carlo,
            "n_slots": cfg.n_slots,
            "config": cfg.to_dict(),
        })
    elif args.all:
        for fam in FAMILIES:
            run_family(fam, cfg)
    elif args.family:
        run_family(args.family, cfg)
    else:
        # Default = baseline family only
        run_family("baseline", cfg)

    print(f"\nDone in {time.time()-t0:.1f}s | Results under {cfg.results_dir}/")


if __name__ == '__main__':
    main()
