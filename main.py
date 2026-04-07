"""
Main: Quantum-Assisted Digital Twin ISAC 6G Simulator.
Usage:
  python main.py                  # Full run
  python main.py --quick          # Quick test
  python main.py --baseline 4     # Single baseline
  python main.py --sweep anomaly_prob
"""
import argparse, copy, time
from pathlib import Path
import numpy as np, pandas as pd
from config import SimConfig, SWEEP_CONFIGS
from simulator import run_mc, BL_NAMES
from metrics import MetricsCollector
from plotting import plot_bars, plot_time, plot_cdf, plot_sweep, make_table


def run_baselines(cfg, odir, pdir, verbose=True):
    all_sum = pd.DataFrame()
    all_slots = {}
    for bl in range(5):
        c = copy.copy(cfg); c.baseline_id = bl
        mc = MetricsCollector(steady_state_fraction=0.5)
        run_mc(c, mc, verbose=verbose)
        s = mc.summary_df()
        all_sum = pd.concat([all_sum, s], ignore_index=True)
        sl = mc.slot_df()
        if not sl.empty:
            all_slots[str(bl)] = sl.tail(cfg.n_slots)
        mc.save(odir, f"bl{bl}_")
    all_sum.to_csv(f"{odir}/all_summary.csv", index=False)
    # Plots
    plot_bars(all_sum, pdir)
    if '4' in all_slots: plot_time(all_slots['4'], pdir, 4)
    if all_slots:
        for m, xl in [('sum_rate','Sum Rate (Mbps)'),('avg_pd','P_d'),
                       ('trust','Trust'),('twin_err','Twin Error')]:
            try: plot_cdf(all_slots, m, xl, pdir)
            except: pass
    tbl = make_table(all_sum)
    tbl.to_csv(f"{odir}/table.csv", index=False)
    if verbose:
        print("\n" + "="*100)
        print(tbl.to_string(index=False))
        print("="*100)
    return all_sum


def run_one_sweep(cfg, name, odir, pdir, verbose=True):
    if name not in SWEEP_CONFIGS:
        print(f"Unknown sweep: {name}"); return
    si = SWEEP_CONFIGS[name]
    param, vals = si['param'], si['values']
    if verbose: print(f"\n### SWEEP: {name} ({param}) = {vals}")
    bl_data = {}
    for bl in [0, 2, 4]:
        bl_sums = pd.DataFrame()
        for v in vals:
            c = copy.copy(cfg); c.baseline_id = bl
            setattr(c, param, v)
            mc = MetricsCollector(steady_state_fraction=0.5)
            run_mc(c, mc, verbose=verbose)
            s = mc.summary_df()
            s['sweep_value'] = v if not isinstance(v, tuple) else str(v)
            bl_sums = pd.concat([bl_sums, s], ignore_index=True)
        bl_data[str(bl)] = bl_sums
    for m, yl in [('sum_rate','Sum Rate (Mbps)'),('avg_pd','P_d'),
                   ('utility','Overall Utility'),('trust','Trust')]:
        try:
            sv = vals if not isinstance(vals[0], tuple) else [str(v) for v in vals]
            plot_sweep(name, sv, bl_data, m, yl, pdir)
        except Exception as e:
            print(f"  Plot warning: {e}")


def main():
    ap = argparse.ArgumentParser(description="QDT-ISAC 6G Simulator")
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--baseline', type=int, default=None)
    ap.add_argument('--sweep', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mc', type=int, default=None)
    ap.add_argument('--slots', type=int, default=None)
    ap.add_argument('--output', type=str, default='results')
    args = ap.parse_args()

    cfg = SimConfig(seed=args.seed)
    if args.quick: cfg.n_monte_carlo = 3; cfg.n_slots = 300
    if args.mc: cfg.n_monte_carlo = args.mc
    if args.slots: cfg.n_slots = args.slots
    odir, pdir = args.output, "figures"
    Path(odir).mkdir(exist_ok=True); Path(pdir).mkdir(exist_ok=True)

    t0 = time.time()
    print("="*60)
    print(f"  QDT-ISAC Simulator | MC={cfg.n_monte_carlo} Slots={cfg.n_slots} Seed={cfg.seed}")
    print("="*60)

    if args.baseline is not None:
        cfg.baseline_id = args.baseline
        mc = MetricsCollector(steady_state_fraction=0.5)
        run_mc(cfg, mc, verbose=True)
        mc.save(odir, f"bl{args.baseline}_")
    elif args.sweep:
        run_one_sweep(cfg, args.sweep, odir, pdir)
    else:
        run_baselines(cfg, odir, pdir)
        for sw in ['user_density', 'anomaly_prob', 'twin_delay',
                    'target_speed', 'clutter', 'sensing_power', 'target_density',
                    'scalability', 'quantum_onoff', 'twin_fidelity', 'mobility',
                    'weight_sweep']:
            run_one_sweep(cfg, sw, odir, pdir)

    print(f"\nDone in {time.time()-t0:.1f}s | Results: {odir}/ | Figures: {pdir}/")

if __name__ == '__main__':
    main()
