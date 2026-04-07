"""
Metrics collection and aggregation.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional

METRIC_NAMES = [
    'slot', 'sum_rate', 'avg_tput', 'avg_sinr', 'outage',
    'sense_util', 'avg_pd', 'twin_err', 'twin_conf',
    'trust', 'det_rate', 'fa_rate', 'latency_ms', 'energy',
    'utility'
]

class MetricsCollector:
    def __init__(self):
        self._cur: List[Dict] = []
        self.runs: List[Dict] = []
        self.all_slots: List[Dict] = []

    def record(self, m: dict):
        self._cur.append(m)

    def end_run(self, run_id: int, bl_id: int, extra: Optional[Dict] = None):
        if not self._cur: return
        df = pd.DataFrame(self._cur)
        n = len(df)
        ss = df.iloc[n//2:]  # steady-state
        s = {'run_id': run_id, 'baseline_id': bl_id}
        for col in df.columns:
            if col == 'slot': continue
            s[f'{col}_mean'] = ss[col].mean()
            s[f'{col}_std'] = ss[col].std()
        if extra: s.update(extra)
        self.runs.append(s)
        self.all_slots.extend(self._cur)
        self._cur.clear()

    def summary_df(self): return pd.DataFrame(self.runs)
    def slot_df(self): return pd.DataFrame(self.all_slots)

    def save(self, outdir: str, prefix: str = ""):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        sdf = self.summary_df()
        if not sdf.empty:
            sdf.to_csv(f"{outdir}/{prefix}summary.csv", index=False)
        sldf = self.slot_df()
        if not sldf.empty:
            sldf.to_csv(f"{outdir}/{prefix}slots.csv", index=False)

    def reset(self):
        self._cur.clear(); self.runs.clear(); self.all_slots.clear()
