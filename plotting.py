"""
Publication-quality plotting layer.

Every figure is saved as both PNG and PDF using consistent labelling,
colouring, line-style conventions, and sizing suited to two-column IEEE
journal layouts.

Required figures supported:
  * baseline_comparison           — baseline bar chart (95% CI)
  * radar_multiobjective          — multi-objective radar plot
  * trust_trajectory              — trust score over time
  * twin_mismatch_trajectory      — twin mismatch over time
  * utility_vs_anomaly            — utility vs anomaly probability
  * utility_vs_twin_delay         — utility vs twin delay
  * pd_vs_twin_fidelity           — detection probability vs twin fidelity
  * runtime_vs_candidates         — runtime vs candidate-set size
  * scalability_users / targets   — scalability plots
  * ablation_summary              — ablation bar summary
  * search_cost_vs_shortlist      — search cost reduction vs shortlist ratio
  * energy_utility_tradeoff       — energy vs utility Pareto
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 200,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 1.8, 'lines.markersize': 6,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

BL: Dict[int, str] = {
    -1: "Ablation",
    0: "Static ISAC", 1: "Reactive Adaptive", 2: "DT-guided",
    3: "DT+QA (no Sec)", 4: "Full Proposed",
    5: "Predictor-UA", 6: "Robust Heuristic", 7: "Learning Bandit",
}
MK: Dict[int, str] = {0: 's', 1: '^', 2: 'D', 3: 'v', 4: 'o',
                      5: 'P', 6: 'X', 7: '*', -1: '.'}
CL: Dict[int, str] = {
    0: '#d62728', 1: '#ff7f0e', 2: '#2ca02c', 3: '#9467bd',
    4: '#1f77b4', 5: '#17becf', 6: '#8c564b', 7: '#e377c2', -1: '#7f7f7f',
}
LS: Dict[int, str] = {0: '--', 1: '-.', 2: ':', 3: '--',
                      4: '-', 5: '-', 6: '-.', 7: ':', -1: '-'}


def _save(fig, d: str, name: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{d}/{name}.pdf")
    fig.savefig(f"{d}/{name}.png")
    plt.close(fig)


def _mean_col(df: pd.DataFrame, col: str) -> str:
    """Prefer `<col>_mean` if present, otherwise raw `<col>`."""
    if f"{col}_mean" in df.columns:
        return f"{col}_mean"
    return col


# =============================================================================
# Baseline comparison
# =============================================================================
def plot_baseline_bars(sdf: pd.DataFrame, d: str,
                        filename: str = "baseline_comparison") -> None:
    if sdf.empty:
        return
    metrics = [
        ('sum_rate', 'Sum Rate (Mbps)'),
        ('avg_pd',   'Avg $P_d$'),
        ('sense_util', 'Sensing Util.'),
        ('trust',    'Avg Trust'),
        ('utility',  'Overall Utility'),
        ('twin_err', 'Twin Error'),
        ('energy_norm', 'Energy (norm.)'),
    ]
    bls = sorted(int(b) for b in sdf['baseline_id'].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4.5))
    for i, (col, lab) in enumerate(metrics):
        ax = axes[i]
        c = _mean_col(sdf, col)
        means, cis = [], []
        for b in bls:
            bd = sdf[sdf['baseline_id'] == b]
            if c in bd.columns and not bd.empty:
                mu = bd[c].mean()
                sd = bd[c].std()
                n = max(len(bd), 1)
                means.append(mu)
                cis.append(1.96 * sd / np.sqrt(n))
            else:
                means.append(0); cis.append(0)
        ax.bar(range(len(bls)), means, yerr=cis, capsize=3, width=0.6,
               color=[CL.get(b, '#888') for b in bls],
               edgecolor='k', linewidth=0.5)
        ax.set_xticks(range(len(bls)))
        ax.set_xticklabels([BL.get(b, str(b)) for b in bls],
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(lab)
    plt.suptitle('Baseline Comparison (95% CI)', fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, d, filename)


# =============================================================================
# Radar plot
# =============================================================================
def plot_radar_multiobjective(sdf: pd.DataFrame, d: str,
                                filename: str = "radar_multiobjective") -> None:
    if sdf.empty:
        return
    # Normalise each metric across baselines so the radar is in [0,1]
    axes_metrics = [
        ('sum_rate',  'Comm.'),
        ('avg_pd',    'Sensing'),
        ('trust',     'Trust'),
        ('utility',   'Utility'),
        ('robustness_gain', 'Robustness'),
        ('energy_norm', 'Energy (inv.)'),
    ]
    bls = sorted(int(b) for b in sdf['baseline_id'].unique())
    vals: Dict[int, List[float]] = {b: [] for b in bls}
    for col, _ in axes_metrics:
        c = _mean_col(sdf, col)
        if c not in sdf.columns:
            for b in bls:
                vals[b].append(0.0)
            continue
        series: Dict[int, float] = {}
        for b in bls:
            bd = sdf[sdf['baseline_id'] == b]
            series[b] = float(bd[c].mean()) if not bd.empty else 0.0
        lo, hi = min(series.values()), max(series.values())
        rng = max(hi - lo, 1e-9)
        for b in bls:
            v = (series[b] - lo) / rng
            if col == 'energy_norm':
                v = 1.0 - v   # invert so higher = better
            vals[b].append(v)

    labels = [a[1] for a in axes_metrics]
    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    theta += theta[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for b in bls:
        v = vals[b] + vals[b][:1]
        ax.plot(theta, v, color=CL.get(b, 'gray'),
                label=BL.get(b, str(b)), lw=1.8)
        ax.fill(theta, v, color=CL.get(b, 'gray'), alpha=0.08)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_title('Multi-Objective Comparison (normalised)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    _save(fig, d, filename)


# =============================================================================
# Time-series trajectories
# =============================================================================
def _plot_timeseries(slot_df: pd.DataFrame, metric: str, ylabel: str,
                      d: str, filename: str,
                      group_col: str = "baseline_id", window: int = 50) -> None:
    if slot_df.empty or metric not in slot_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for gid, sub in slot_df.groupby(group_col):
        try:
            i = int(gid)
        except Exception:
            i = 0
        sub = sub.sort_values('slot')
        y = sub[metric].rolling(window, min_periods=1).mean()
        lab = BL.get(i, str(gid))
        ax.plot(sub['slot'], y, color=CL.get(i, 'gray'),
                ls=LS.get(i, '-'), label=lab, lw=1.5)
    ax.set_xlabel('Slot')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8)
    _save(fig, d, filename)


def plot_trust_trajectory(slot_df: pd.DataFrame, d: str,
                           filename: str = "trust_trajectory") -> None:
    _plot_timeseries(slot_df, 'trust', 'Avg Trust Score', d, filename)


def plot_twin_mismatch_trajectory(slot_df: pd.DataFrame, d: str,
                                     filename: str = "twin_mismatch_trajectory") -> None:
    col = 'twin_mismatch' if 'twin_mismatch' in slot_df.columns else 'twin_err'
    _plot_timeseries(slot_df, col, 'Twin Mismatch', d, filename)


# =============================================================================
# Sweep plots (utility vs X, Pd vs X, …)
# =============================================================================
def plot_sweep_metric(sum_df: pd.DataFrame, d: str,
                       x_col: str, metric: str, ylabel: str,
                       xlabel: Optional[str] = None,
                       filename: Optional[str] = None) -> None:
    if sum_df.empty or x_col not in sum_df.columns:
        return
    c = _mean_col(sum_df, metric)
    if c not in sum_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for bl, sub in sum_df.groupby('baseline_id'):
        try:
            b = int(bl)
        except Exception:
            b = 0
        xs, ys, cis = [], [], []
        for v, grp in sub.groupby(x_col):
            if grp[c].empty:
                continue
            xs.append(v)
            ys.append(float(grp[c].mean()))
            sd = float(grp[c].std()) if len(grp) > 1 else 0.0
            cis.append(1.96 * sd / max(np.sqrt(len(grp)), 1))
        if not xs:
            continue
        # sort numerically if possible
        try:
            order = np.argsort([float(x) for x in xs])
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            cis = [cis[i] for i in order]
        except Exception:
            pass
        ax.errorbar(xs, ys, yerr=cis, marker=MK.get(b, 'o'),
                     color=CL.get(b, 'gray'), ls=LS.get(b, '-'),
                     label=BL.get(b, str(b)), capsize=3)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8)
    fn = filename or f"sweep_{x_col}_{metric}"
    _save(fig, d, fn)


# =============================================================================
# Ablation bar summary
# =============================================================================
def plot_ablation_summary(sdf: pd.DataFrame, d: str,
                            filename: str = "ablation_summary") -> None:
    if sdf.empty or 'variant' not in sdf.columns:
        return
    metrics = [('utility', 'Utility'),
                ('avg_pd', 'Avg $P_d$'),
                ('trust', 'Trust'),
                ('robustness_gain', 'Robustness'),
                ('search_cost_reduction', 'Search Cost Red.'),
                ('energy_norm', 'Energy (norm.)')]
    variants = sdf['variant'].unique().tolist()
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4.5))
    for i, (col, lab) in enumerate(metrics):
        ax = axes[i]
        c = _mean_col(sdf, col)
        means, cis = [], []
        for v in variants:
            bd = sdf[sdf['variant'] == v]
            if c in bd.columns and not bd.empty:
                mu = bd[c].mean(); sd = bd[c].std(); n = max(len(bd), 1)
                means.append(mu); cis.append(1.96 * sd / np.sqrt(n))
            else:
                means.append(0); cis.append(0)
        ax.bar(range(len(variants)), means, yerr=cis, capsize=3, width=0.6,
               color='#1f77b4', edgecolor='k', linewidth=0.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(lab)
    plt.suptitle('Ablation Summary', fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, d, filename)


# =============================================================================
# Search-cost reduction vs shortlist ratio
# =============================================================================
def plot_search_cost_vs_shortlist(sum_df: pd.DataFrame, d: str,
                                     filename: str = "search_cost_vs_shortlist"
                                     ) -> None:
    if sum_df.empty or 'sweep' not in sum_df.columns:
        return
    sub = sum_df[sum_df['sweep'] == 'shortlist_ratio']
    if sub.empty:
        return
    c = _mean_col(sub, 'search_cost_reduction')
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs, ys, cis = [], [], []
    for v, grp in sub.groupby('sweep_value'):
        if c not in grp.columns or grp[c].empty:
            continue
        xs.append(v)
        ys.append(float(grp[c].mean()))
        sd = float(grp[c].std()) if len(grp) > 1 else 0.0
        cis.append(1.96 * sd / max(np.sqrt(len(grp)), 1))
    if xs:
        order = np.argsort([float(x) for x in xs])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        cis = [cis[i] for i in order]
        ax.errorbar(xs, ys, yerr=cis, marker='o', color='#1f77b4',
                     label='Full Proposed', capsize=3)
    ax.set_xlabel('Shortlist size')
    ax.set_ylabel('Search-cost reduction')
    ax.legend(loc='best')
    _save(fig, d, filename)


# =============================================================================
# Energy-utility trade-off
# =============================================================================
def plot_energy_utility(sdf: pd.DataFrame, d: str,
                          filename: str = "energy_utility_tradeoff") -> None:
    if sdf.empty:
        return
    ex = _mean_col(sdf, 'energy_norm')
    uy = _mean_col(sdf, 'utility')
    if ex not in sdf.columns or uy not in sdf.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for bl, sub in sdf.groupby('baseline_id'):
        try:
            b = int(bl)
        except Exception:
            b = 0
        ax.scatter(sub[ex], sub[uy], marker=MK.get(b, 'o'),
                    color=CL.get(b, 'gray'), s=70,
                    label=BL.get(b, str(b)), edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Energy (norm.)')
    ax.set_ylabel('Utility')
    ax.legend(loc='best', fontsize=8)
    _save(fig, d, filename)


# =============================================================================
# Runtime plots
# =============================================================================
def plot_runtime_vs_candidates(sum_df: pd.DataFrame, d: str,
                                 filename: str = "runtime_vs_candidates") -> None:
    if sum_df.empty or 'sweep' not in sum_df.columns:
        return
    sub = sum_df[sum_df['sweep'] == 'candidate_pool']
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    c = _mean_col(sub, 'latency_ms')
    xs, ys, cis = [], [], []
    for v, grp in sub.groupby('sweep_value'):
        xs.append(v)
        ys.append(float(grp[c].mean()))
        sd = float(grp[c].std()) if len(grp) > 1 else 0.0
        cis.append(1.96 * sd / max(np.sqrt(len(grp)), 1))
    if xs:
        order = np.argsort([float(x) for x in xs])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        cis = [cis[i] for i in order]
        ax.errorbar(xs, ys, yerr=cis, marker='o', color='#1f77b4', capsize=3)
    ax.set_xlabel('Candidate pool size')
    ax.set_ylabel('Per-slot latency (ms)')
    _save(fig, d, filename)


# =============================================================================
# Summary table
# =============================================================================
def make_baseline_table(sdf: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ('sum_rate_mean', 'Sum Rate'),
        ('avg_tput_mean', 'Avg Tput'),
        ('avg_pd_mean', 'P_d'),
        ('sense_util_mean', 'Sense Util'),
        ('twin_err_mean', 'Twin Err'),
        ('twin_fidelity_mean', 'Twin Fid.'),
        ('trust_mean', 'Trust'),
        ('fallback_deployment_ratio_mean', 'Fallback'),
        ('outage_mean', 'Outage'),
        ('energy_norm_mean', 'Energy'),
        ('latency_ms_mean', 'Latency ms'),
        ('search_cost_reduction_mean', 'SearchRed.'),
        ('robustness_gain_mean', 'Robustness'),
        ('utility_mean', 'Utility'),
    ]
    rows = []
    for bl in sorted(int(b) for b in sdf['baseline_id'].unique()):
        bd = sdf[sdf['baseline_id'] == bl]
        r = {'Method': BL.get(bl, f'BL{bl}')}
        for c, n in cols:
            if c in bd.columns:
                r[n] = f"{bd[c].mean():.3f}±{bd[c].std():.3f}"
            else:
                r[n] = "N/A"
        rows.append(r)
    return pd.DataFrame(rows)


def make_runtime_table(sdf: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ('latency_ms_mean', 'Lat. ms'),
        ('screening_overhead_ms_mean', 'Screen ms'),
        ('search_cost_reduction_mean', 'SearchRed.'),
        ('candidate_reduction_ratio_mean', 'PoolRatio'),
        ('selected_action_rank_percentile_mean', 'Rank %ile'),
        ('utility_mean', 'Utility'),
        ('energy_norm_mean', 'Energy'),
    ]
    rows = []
    for bl in sorted(int(b) for b in sdf['baseline_id'].unique()):
        bd = sdf[sdf['baseline_id'] == bl]
        r = {'Method': BL.get(bl, f'BL{bl}')}
        for c, n in cols:
            if c in bd.columns:
                r[n] = f"{bd[c].mean():.3f}"
            else:
                r[n] = "N/A"
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# Back-compat thin wrappers (old API used by legacy code paths)
# =============================================================================
def plot_bars(sdf, d):
    return plot_baseline_bars(sdf, d)


def plot_time(slots_df, d, bl=4):
    metrics = [('sum_rate', 'Sum Rate (Mbps)'),
                ('avg_pd', 'P_d'),
                ('sense_util', 'Sensing Util'),
                ('twin_err', 'Twin Error'),
                ('trust', 'Trust'),
                ('utility', 'Utility')]
    w = 50
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, (col, lab) in enumerate(metrics):
        ax = axes[i // 3][i % 3]
        if col not in slots_df.columns:
            continue
        y = slots_df[col].rolling(w, min_periods=1).mean()
        ax.plot(slots_df['slot'], y, color=CL.get(bl, 'gray'), lw=1.5)
        ax.set_xlabel('Slot'); ax.set_ylabel(lab)
    plt.tight_layout()
    _save(fig, d, f"time_evolution_bl{bl}")


def plot_cdf(slots_dict, metric, xlabel, d):
    fig, ax = plt.subplots(figsize=(7, 5))
    for bl_s, df in slots_dict.items():
        try:
            b = int(bl_s)
        except Exception:
            b = 0
        if metric not in df.columns:
            continue
        data = df[metric].dropna().values
        if len(data) == 0:
            continue
        s = np.sort(data)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s),
                 label=BL.get(b, ''),
                 color=CL.get(b, 'gray'), ls=LS.get(b, '-'))
    ax.set_xlabel(xlabel); ax.set_ylabel('CDF'); ax.legend()
    _save(fig, d, f'cdf_{metric}')


def plot_sweep(name, vals, bl_data, metric, ylabel, d):
    fig, ax = plt.subplots(figsize=(7, 5))
    for bl_s, sdf in bl_data.items():
        try:
            b = int(bl_s)
        except Exception:
            b = 0
        xs, ys, cis = [], [], []
        col = f'{metric}_mean'
        if col not in sdf.columns:
            continue
        for v in vals:
            sub = sdf[sdf.get('sweep_value') == v]
            if sub.empty:
                continue
            xs.append(v); ys.append(float(sub[col].mean()))
            sd = float(sub[col].std()) if len(sub) > 1 else 0.0
            cis.append(1.96 * sd / max(np.sqrt(len(sub)), 1))
        if ys:
            ax.errorbar(xs, ys, yerr=cis, label=BL.get(b, ''),
                         marker=MK.get(b, 'o'), color=CL.get(b, 'gray'),
                         ls=LS.get(b, '-'), capsize=3)
    ax.set_xlabel(name); ax.set_ylabel(ylabel); ax.legend()
    _save(fig, d, f'sweep_{name}_{metric}')


def make_table(sdf):
    return make_baseline_table(sdf)
