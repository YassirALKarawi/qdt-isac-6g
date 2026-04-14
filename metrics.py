"""
Metrics collection and aggregation with steady-state, CI95, and derived
metrics suited to publication-grade reporting.
"""
from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Iterable


# Slot-level columns that are always recorded (others are discovered at runtime)
BASE_METRIC_NAMES: List[str] = [
    "slot", "sum_rate", "avg_tput", "avg_sinr", "outage",
    "sense_util", "avg_pd", "twin_err", "twin_conf",
    "twin_mismatch", "twin_mismatch_std", "twin_fidelity", "stale_state_penalty",
    "trust", "det_rate", "fa_rate", "latency_ms",
    "energy", "energy_norm", "utility",
    "search_cost", "search_cost_reduction", "candidate_reduction_ratio",
    "selected_action_rank_percentile", "screening_overhead_ms",
    "fallback_deployment_ratio", "unsafe_action_suppression_rate",
    "adaptation_gain", "robustness_gain",
]

# Metrics that get explicit 95% CI recorded
CI95_METRICS: List[str] = [
    "sum_rate", "avg_pd", "trust", "utility", "energy", "energy_norm",
    "twin_err", "twin_mismatch", "twin_fidelity", "latency_ms",
    "search_cost", "search_cost_reduction", "candidate_reduction_ratio",
    "selected_action_rank_percentile", "screening_overhead_ms",
    "fallback_deployment_ratio", "unsafe_action_suppression_rate",
    "adaptation_gain", "robustness_gain",
]


def ci95(std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return 0.0
    return 1.96 * std / math.sqrt(n)


class MetricsCollector:
    """Slot-level recorder that produces run-summary rows with steady-state
    mean / std / 95% CI for every numeric metric."""

    def __init__(self, steady_state_fraction: float = 0.5):
        self._cur: List[Dict] = []
        self.runs: List[Dict] = []
        self.all_slots: List[Dict] = []
        self.steady_state_fraction = steady_state_fraction

    def record(self, m: dict) -> None:
        self._cur.append(m)

    def end_run(self, run_id: int, bl_id: int,
                 extra: Optional[Dict] = None) -> None:
        if not self._cur:
            return

        df = pd.DataFrame(self._cur)
        n = len(df)
        start_idx = int(n * self.steady_state_fraction)
        ss = df.iloc[start_idx:]

        s: Dict = {
            "run_id": run_id,
            "baseline_id": bl_id,
            "steady_state_start_slot": int(df.iloc[start_idx]["slot"])
                                         if len(df) else 0,
            "n_slots": int(n),
        }

        numeric_cols = ss.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == "slot":
                continue
            mean = ss[col].mean()
            std = ss[col].std()
            s[f"{col}_mean"] = float(mean) if np.isfinite(mean) else 0.0
            s[f"{col}_std"] = float(std) if np.isfinite(std) else 0.0

        # Explicit CI95 on selected metrics
        for col in CI95_METRICS:
            if col in ss.columns and len(ss) > 1:
                s[f"{col}_ci95"] = ci95(float(ss[col].std()), int(len(ss)))
            elif col in ss.columns:
                s[f"{col}_ci95"] = 0.0

        # Energy-utility trade-off per run: utility per unit of normalised energy
        if "utility_mean" in s and "energy_norm_mean" in s:
            denom = s["energy_norm_mean"] + 1e-9
            s["energy_utility_tradeoff"] = s["utility_mean"] / denom

        if extra:
            s.update(extra)

        self.runs.append(s)
        self.all_slots.extend(self._cur)
        self._cur.clear()

    # ------------------------------------------------------------------
    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.runs)

    def slot_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.all_slots)

    # ------------------------------------------------------------------
    def save(self, outdir: str, prefix: str = "",
             metadata: Optional[Dict] = None) -> None:
        """Save summary.csv, slots.csv and metadata.json under `outdir`.

        If `prefix` is set, both CSVs are prefixed."""
        Path(outdir).mkdir(parents=True, exist_ok=True)
        sdf = self.summary_df()
        if not sdf.empty:
            sdf.to_csv(f"{outdir}/{prefix}summary.csv", index=False)
        sldf = self.slot_df()
        if not sldf.empty:
            sldf.to_csv(f"{outdir}/{prefix}slots.csv", index=False)
        if metadata is not None:
            # metadata.json is always unprefixed (one per experiment folder)
            meta_path = Path(outdir) / f"{prefix}metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    def reset(self) -> None:
        self._cur.clear()
        self.runs.clear()
        self.all_slots.clear()


# =============================================================================
# Cross-run aggregation helpers
# =============================================================================
def aggregate_across_runs(df: pd.DataFrame, group_cols: Iterable[str],
                           metric_cols: Iterable[str]) -> pd.DataFrame:
    """Aggregate mean / std / CI95 across MC runs grouped by `group_cols`."""
    out_rows = []
    for keys, sub in df.groupby(list(group_cols)):
        row: Dict = {}
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys):
                row[k] = v
        else:
            row[list(group_cols)[0]] = keys
        n = len(sub)
        for m in metric_cols:
            col = f"{m}_mean" if f"{m}_mean" in sub.columns else m
            if col in sub.columns:
                vals = sub[col].to_numpy(dtype=float)
                mu = float(np.nanmean(vals))
                sd = float(np.nanstd(vals))
                row[f"{m}_mu"] = mu
                row[f"{m}_sd"] = sd
                row[f"{m}_ci95"] = ci95(sd, n)
        row["n_runs"] = n
        out_rows.append(row)
    return pd.DataFrame(out_rows)
