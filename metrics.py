"""
Metrics collection and aggregation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


METRIC_NAMES = [
    "slot", "sum_rate", "avg_tput", "avg_sinr", "outage",
    "sense_util", "avg_pd", "twin_err", "twin_conf",
    "trust", "det_rate", "fa_rate", "latency_ms",
    "energy", "energy_norm", "utility",
]


class MetricsCollector:
    def __init__(self, steady_state_fraction: float = 0.5):
        self._cur: List[Dict] = []
        self.runs: List[Dict] = []
        self.all_slots: List[Dict] = []
        self.steady_state_fraction = steady_state_fraction

    def record(self, m: dict):
        self._cur.append(m)

    def end_run(self, run_id: int, bl_id: int, extra: Optional[Dict] = None):
        if not self._cur:
            return

        df = pd.DataFrame(self._cur)
        n = len(df)
        start_idx = int(n * self.steady_state_fraction)
        ss = df.iloc[start_idx:]

        s = {
            "run_id": run_id,
            "baseline_id": bl_id,
            "steady_state_start_slot": int(df.iloc[start_idx]["slot"]) if len(df) else 0,
        }

        for col in df.columns:
            if col == "slot":
                continue
            s[f"{col}_mean"] = ss[col].mean()
            s[f"{col}_std"] = ss[col].std()

        # Confidence intervals for key metrics
        for col in ["sum_rate", "avg_pd", "trust", "utility", "energy"]:
            if col in ss.columns and len(ss) > 1:
                std = ss[col].std()
                s[f"{col}_ci95"] = 1.96 * std / np.sqrt(len(ss))
            elif col in ss.columns:
                s[f"{col}_ci95"] = 0.0

        if extra:
            s.update(extra)

        self.runs.append(s)
        self.all_slots.extend(self._cur)
        self._cur.clear()

    def summary_df(self):
        return pd.DataFrame(self.runs)

    def slot_df(self):
        return pd.DataFrame(self.all_slots)

    def save(self, outdir: str, prefix: str = ""):
        Path(outdir).mkdir(parents=True, exist_ok=True)
        sdf = self.summary_df()
        if not sdf.empty:
            sdf.to_csv(f"{outdir}/{prefix}summary.csv", index=False)
        sldf = self.slot_df()
        if not sldf.empty:
            sldf.to_csv(f"{outdir}/{prefix}slots.csv", index=False)

    def reset(self):
        self._cur.clear()
        self.runs.clear()
        self.all_slots.clear()
