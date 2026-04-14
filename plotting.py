"""
Publication-quality plots for IEEE JSAC.
Consistent styling, proper typography, grid layout, and PDF/PNG dual export.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from pathlib import Path
from typing import Dict

# IEEE-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BL = {0: "Static ISAC", 1: "Adaptive ISAC", 2: "DT (no QA)",
      3: "DT+QA (no Sec)", 4: "Full Proposed",
      5: "Uncertainty-Aware", 6: "UCB Learning"}
MK = {0: 's', 1: '^', 2: 'D', 3: 'v', 4: 'o', 5: 'P', 6: 'X'}
CL = {0: '#d62728', 1: '#ff7f0e', 2: '#2ca02c', 3: '#9467bd', 4: '#1f77b4',
      5: '#8c564b', 6: '#e377c2'}
LS = {0: '--', 1: '-.', 2: ':', 3: '--', 4: '-', 5: '-.', 6: ':'}


def _save(fig, d, name):
    Path(d).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{d}/{name}.pdf")
    fig.savefig(f"{d}/{name}.png")
    plt.close(fig)


def plot_bars(sdf: pd.DataFrame, d: str):
    """Baseline comparison bar charts with 95% CI error bars."""
    metrics = [
        ('sum_rate_mean', 'Sum Rate\n(Mbps)'),
        ('avg_pd_mean', 'Avg $P_d$'),
        ('sense_util_mean', 'Sensing\nUtility'),
        ('trust_mean', 'Avg Trust'),
        ('utility_mean', 'Overall\nUtility'),
        ('twin_err_mean', 'Twin\nError'),
        ('energy_norm_mean', 'Energy\n(norm)'),
    ]
    bls = sorted(sdf['baseline_id'].unique())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(24, 5))

    for i, (col, lab) in enumerate(metrics):
        ax = axes[i]
        ms, cs = [], []
        for b in bls:
            bd = sdf[sdf['baseline_id'] == b]
            if col in bd.columns:
                ms.append(bd[col].mean())
                n = len(bd)
                cs.append(1.96 * bd[col].std() / np.sqrt(n + 1e-10))
            else:
                ms.append(0)
                cs.append(0)
        bars = ax.bar(range(len(bls)), ms, yerr=cs, capsize=3, width=0.65,
                      color=[CL.get(b, '#999') for b in bls],
                      edgecolor='white', linewidth=0.8,
                      error_kw={'linewidth': 1.2, 'capthick': 1.0})
        # Highlight the best
        if 'err' not in col.lower():
            best_idx = int(np.argmax(ms))
        else:
            best_idx = int(np.argmin(ms))
        bars[best_idx].set_edgecolor('#2C3E50')
        bars[best_idx].set_linewidth(2.0)

        ax.set_xticks(range(len(bls)))
        ax.set_xticklabels([BL.get(b, f'BL{b}') for b in bls],
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(lab)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        ax.grid(axis='x', visible=False)

    fig.suptitle('Baseline Comparison (95% CI)', fontsize=14,
                 fontweight='bold', y=1.03)
    plt.tight_layout()
    _save(fig, d, 'baseline_comparison')


def plot_time(sdf: pd.DataFrame, d: str, bl: int = 4):
    """Time evolution with rolling mean +/- std shading."""
    w = 50
    metrics = [
        ('sum_rate', 'Sum Rate (Mbps)'),
        ('avg_pd', '$P_d$'),
        ('sense_util', 'Sensing Utility'),
        ('twin_err', 'Twin Error'),
        ('trust', 'Trust Score'),
        ('utility', 'Overall Utility $J$'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    color = CL.get(bl, '#1f77b4')

    for i, (col, lab) in enumerate(metrics):
        ax = axes[i // 3][i % 3]
        if col not in sdf.columns:
            continue
        sm = sdf[col].rolling(w, min_periods=1).mean()
        sd = sdf[col].rolling(w, min_periods=1).std().fillna(0)
        x = sdf['slot']
        ax.plot(x, sm, color=color, lw=1.8)
        ax.fill_between(x, sm - sd, sm + sd, alpha=0.12, color=color)
        ax.set_xlabel('Slot')
        ax.set_ylabel(lab)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f'Time Evolution \u2014 {BL.get(bl, "")}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, d, f'time_evolution_bl{bl}')


def plot_cdf(slots_dict: Dict[str, pd.DataFrame], metric: str,
             xlabel: str, d: str):
    """CDF across baselines with proper styling."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for bl_s, df in sorted(slots_dict.items(), key=lambda x: int(x[0])):
        bl = int(bl_s)
        data = df[metric].dropna().values
        if len(data) == 0:
            continue
        s = np.sort(data)
        cdf = np.arange(1, len(s) + 1) / len(s)
        lw = 2.5 if bl == 4 else 1.6
        ax.plot(s, cdf, label=BL.get(bl, f'BL{bl}'),
                color=CL.get(bl, 'gray'), ls=LS.get(bl, '-'),
                lw=lw, marker=MK.get(bl, 'o'),
                markevery=max(1, len(s) // 8), markersize=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.set_ylim([0, 1.02])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    _save(fig, d, f'cdf_{metric}')


def plot_sweep(name, vals, bl_data: Dict, metric, ylabel, d):
    """Parameter sweep with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for bl_s, sdf in sorted(bl_data.items(), key=lambda x: int(x[0])):
        bl = int(bl_s)
        ms, cs, xs = [], [], []
        col = f'{metric}_mean'
        for v in vals:
            sub = sdf[sdf['sweep_value'] == v]
            if sub.empty or col not in sub:
                continue
            ms.append(sub[col].mean())
            cs.append(1.96 * sub[col].std() / np.sqrt(len(sub) + 1e-10))
            xs.append(v)
        if ms:
            lw = 2.5 if bl == 4 else 1.6
            ax.errorbar(xs, ms, yerr=cs, label=BL.get(bl, f'BL{bl}'),
                        marker=MK.get(bl, 'o'),
                        color=CL.get(bl, 'gray'), ls=LS.get(bl, '-'),
                        capsize=3, lw=lw, markeredgewidth=1.0,
                        elinewidth=1.0, capthick=0.8)

    ax.set_xlabel(name.replace('_', ' ').title())
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    _save(fig, d, f'sweep_{name}_{metric}')


def plot_ablation(ablation_df: pd.DataFrame, d: str):
    """Ablation study horizontal bar chart."""
    if 'ablation' not in ablation_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ablation_df.groupby('ablation')['utility_mean'].mean()
    names = names.sort_values(ascending=True)

    colors = ['#1f77b4' if n == 'Full Proposed' else '#95a5a6'
              for n in names.index]
    bars = ax.barh(range(len(names)), names.values, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names.index, fontsize=10)
    ax.set_xlabel('Overall Utility $J$')
    ax.set_title('Ablation Study \u2014 Component Contribution',
                 fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.grid(axis='y', visible=False)
    for bar, val in zip(bars, names.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    _save(fig, d, 'ablation_study')


def plot_degradation_curve(delays, losses, d: str):
    """Utility degradation vs twin delay (formal bound)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(delays, losses, 'o-', color='#E74C3C', lw=2.0,
            markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    ax.fill_between(delays, 0, losses, alpha=0.08, color='#E74C3C')
    ax.set_xlabel('Twin Sync Delay $\\delta$ (slots)')
    ax.set_ylabel('Utility Loss Upper Bound $\\Delta J$')
    ax.set_title('Monotonic Degradation (Proposition 2)', fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    _save(fig, d, 'degradation_bound')


def make_table(sdf: pd.DataFrame) -> pd.DataFrame:
    """Summary table for all baselines."""
    cols = {
        'sum_rate_mean': 'Sum Rate',
        'avg_tput_mean': 'Avg Tput',
        'avg_pd_mean': 'P_d',
        'sense_util_mean': 'Sense Util',
        'twin_err_mean': 'Twin Err',
        'trust_mean': 'Trust',
        'outage_mean': 'Outage',
        'energy_norm_mean': 'Energy(norm)',
        'search_cost_mean': 'Search Cost',
        'robustness_gain_mean': 'Robustness',
        'utility_mean': 'Utility',
    }
    rows = []
    for bl in sorted(sdf['baseline_id'].unique()):
        bd = sdf[sdf['baseline_id'] == bl]
        r = {'Method': BL.get(bl, f'BL{bl}')}
        for c, n in cols.items():
            if c in bd:
                r[n] = f"{bd[c].mean():.3f}\u00B1{bd[c].std():.3f}"
            else:
                r[n] = "N/A"
        rows.append(r)
    return pd.DataFrame(rows)
