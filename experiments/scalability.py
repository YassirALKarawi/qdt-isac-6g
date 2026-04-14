"""
Scalability study (experiment family: scalability).

Sweeps over users, targets, base stations, candidate pool size, and
shortlist size. Measures how utility, sensing quality, runtime, and
screening efficiency evolve with problem scale.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from config import SimConfig, SWEEP_CONFIGS
from metrics import MetricsCollector
from simulator import run_mc, BL_NAMES

from .common import (apply_profile, ensure_outdir, make_metadata,
                      save_metadata, ExperimentResult)


DEFAULT_BASELINES: List[int] = [0, 2, 4]
DEFAULT_SWEEPS: List[str] = [
    "user_density", "target_density", "bs_density",
    "candidate_pool", "shortlist_ratio",
]


def run_scalability_experiment(cfg: Optional[SimConfig] = None,
                                 sweeps: Iterable[str] = DEFAULT_SWEEPS,
                                 baselines: Iterable[int] = DEFAULT_BASELINES,
                                 output_dir: Optional[str] = None,
                                 verbose: bool = True) -> ExperimentResult:
    cfg = apply_profile(cfg or SimConfig(), "profile_scalability")
    cfg.scenario_name = "scalability"

    outdir = Path(output_dir) if output_dir else ensure_outdir(
        cfg.results_dir, "scalability")

    all_sum, all_sl = [], []
    baselines = list(baselines)
    for name in sweeps:
        spec = SWEEP_CONFIGS.get(name)
        if spec is None:
            print(f"[scalability] unknown sweep: {name}")
            continue
        param, values = spec["param"], spec["values"]
        for v in values:
            for bl in baselines:
                c = cfg.clone(baseline_id=bl, **{param: v})
                c.scenario_name = f"{name}_{v}"
                c.sweep_variable = param
                # Shortlist cannot exceed pool size
                if c.qa_shortlist_size > c.qa_n_candidates:
                    c.qa_shortlist_size = c.qa_n_candidates
                mc = MetricsCollector(steady_state_fraction=c.steady_state_fraction)
                if verbose:
                    print(f"\n[scalability] sweep={name} {param}={v} "
                          f"baseline={BL_NAMES.get(bl,bl)}")
                run_mc(c, mc, verbose=verbose,
                        extra={"sweep": name, "sweep_value": v,
                                "baseline_id": bl})
                sdf = mc.summary_df()
                sldf = mc.slot_df()
                prefix = f"{name}_{v}_bl{bl}_"
                mc.save(str(outdir), prefix=prefix,
                         metadata=make_metadata(c, family="scalability",
                                                  scenario=f"{name}_{v}",
                                                  sweep_variable=param))
                if not sdf.empty:
                    sdf = sdf.copy()
                    sdf["sweep"] = name
                    sdf["sweep_value"] = v
                    sdf["baseline_id"] = bl
                    all_sum.append(sdf)
                if not sldf.empty:
                    sldf = sldf.copy()
                    sldf["sweep"] = name
                    sldf["sweep_value"] = v
                    sldf["baseline_id"] = bl
                    all_sl.append(sldf)

    summary = pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame()
    slots = pd.concat(all_sl, ignore_index=True) if all_sl else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(outdir / "summary.csv", index=False)
    if not slots.empty:
        slots.to_csv(outdir / "slots.csv", index=False)

    md = make_metadata(cfg, family="scalability",
                         scenario="scalability",
                         sweep_variable="scale_params",
                         extra={"sweeps": list(sweeps),
                                 "baselines": baselines})
    save_metadata(outdir, md)
    return ExperimentResult(family="scalability", output_dir=outdir,
                              summary=summary, slots=slots, metadata=md)
