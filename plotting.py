"""
Publication-quality plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from typing import Dict

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 1.8, 'lines.markersize': 6,
})

BL = {0:"Static ISAC", 1:"Adaptive ISAC", 2:"DT (no QA)",
      3:"DT+QA (no Sec)", 4:"Full Proposed",
      5:"Uncertainty-Aware", 6:"UCB Learning"}
MK = {0:'s', 1:'^', 2:'D', 3:'v', 4:'o', 5:'P', 6:'X'}
CL = {0:'#d62728', 1:'#ff7f0e', 2:'#2ca02c', 3:'#9467bd', 4:'#1f77b4',
      5:'#8c564b', 6:'#e377c2'}
LS = {0:'--', 1:'-.', 2:':', 3:'--', 4:'-', 5:'-.', 6:':'}


def _save(fig, d, name):
    Path(d).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{d}/{name}.pdf"); fig.savefig(f"{d}/{name}.png")
    plt.close(fig)


def plot_bars(sdf: pd.DataFrame, d: str):
    """Baseline comparison bar charts."""
    metrics = [('sum_rate_mean','Sum Rate\n(Mbps)'),
               ('avg_pd_mean','Avg P_d'),
               ('sense_util_mean','Sensing\nUtility'),
               ('trust_mean','Avg Trust'),
               ('utility_mean','Overall\nUtility'),
               ('twin_err_mean','Twin\nError'),
               ('energy_norm_mean','Energy\n(norm)')]
    bls = sorted(sdf['baseline_id'].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4.5))
    for i, (col, lab) in enumerate(metrics):
        ax = axes[i]
        ms, cs = [], []
        for b in bls:
            bd = sdf[sdf['baseline_id']==b]
            ms.append(bd[col].mean() if col in bd else 0)
            n = len(bd)
            cs.append(1.96*bd[col].std()/np.sqrt(n+1e-10) if col in bd else 0)
        ax.bar(range(len(bls)), ms, yerr=cs, capsize=3, width=0.6,
               color=[CL[b] for b in bls], edgecolor='k', linewidth=0.5)
        ax.set_xticks(range(len(bls)))
        ax.set_xticklabels([BL[b] for b in bls], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(lab); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Baseline Comparison (95% CI)', fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, d, 'baseline_comparison')


def plot_time(sdf: pd.DataFrame, d: str, bl: int = 4):
    """Time evolution of key metrics."""
    w = 50
    metrics = [('sum_rate','Sum Rate (Mbps)'), ('avg_pd','P_d'),
               ('sense_util','Sensing Utility'), ('twin_err','Twin Error'),
               ('trust','Trust Score'), ('utility','Overall Utility')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, (col, lab) in enumerate(metrics):
        ax = axes[i//3][i%3]
        if col not in sdf.columns: continue
        sm = sdf[col].rolling(w, min_periods=1).mean()
        sd = sdf[col].rolling(w, min_periods=1).std().fillna(0)
        ax.plot(sdf['slot'], sm, color=CL[bl], lw=1.5)
        ax.fill_between(sdf['slot'], sm-sd, sm+sd, alpha=0.15, color=CL[bl])
        ax.set_xlabel('Slot'); ax.set_ylabel(lab)
    plt.suptitle(f'Time Evolution — {BL.get(bl,"")}', fontsize=13)
    plt.tight_layout()
    _save(fig, d, f'time_evolution_bl{bl}')


def plot_cdf(slots_dict: Dict[str, pd.DataFrame], metric: str, xlabel: str, d: str):
    """CDF across baselines."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for bl_s, df in slots_dict.items():
        bl = int(bl_s)
        data = df[metric].dropna().values
        s = np.sort(data)
        ax.plot(s, np.arange(1,len(s)+1)/len(s),
                label=BL.get(bl,''), color=CL.get(bl,'gray'), ls=LS.get(bl,'-'))
    ax.set_xlabel(xlabel); ax.set_ylabel('CDF')
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, d, f'cdf_{metric}')


def plot_sweep(name, vals, bl_data: Dict, metric, ylabel, d):
    """Parameter sweep plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for bl_s, sdf in bl_data.items():
        bl = int(bl_s)
        ms, cs, xs = [], [], []
        col = f'{metric}_mean'
        for v in vals:
            sub = sdf[sdf['sweep_value']==v]
            if sub.empty or col not in sub: continue
            ms.append(sub[col].mean())
            cs.append(1.96*sub[col].std()/np.sqrt(len(sub)+1e-10))
            xs.append(v)
        if ms:
            ax.errorbar(xs, ms, yerr=cs, label=BL.get(bl,''), marker=MK.get(bl,'o'),
                       color=CL.get(bl,'gray'), ls=LS.get(bl,'-'), capsize=3)
    ax.set_xlabel(name); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, d, f'sweep_{name}_{metric}')


def make_table(sdf: pd.DataFrame) -> pd.DataFrame:
    cols = {'sum_rate_mean':'Sum Rate', 'avg_tput_mean':'Avg Tput',
            'avg_pd_mean':'P_d', 'sense_util_mean':'Sense Util',
            'twin_err_mean':'Twin Err', 'trust_mean':'Trust',
            'outage_mean':'Outage', 'energy_norm_mean':'Energy(norm)',
            'search_cost_mean':'Search Cost', 'robustness_gain_mean':'Robustness',
            'utility_mean':'Utility'}
    rows = []
    for bl in sorted(sdf['baseline_id'].unique()):
        bd = sdf[sdf['baseline_id']==bl]
        r = {'Method': BL.get(bl, f'BL{bl}')}
        for c, n in cols.items():
            if c in bd:
                r[n] = f"{bd[c].mean():.3f}±{bd[c].std():.3f}"
            else:
                r[n] = "N/A"
        rows.append(r)
    return pd.DataFrame(rows)
