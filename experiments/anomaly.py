"""
Anomaly-robustness sweep (experiment family: anomaly).

Runs each scenario in `ANOMALY_SCENARIOS` for a configurable set of
baselines (default: Full Proposed + strong benchmarks + Static ISAC).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List

import pandas as pd

from config import SimConfig, ANOMALY_SCENARIOS
from metrics import MetricsCollector
from simulator import run_mc, BL_NAMES

from .common import (apply_profile, ensure_outdir, make_metadata,
                      save_metadata, ExperimentResult)


DEFAULT_BASELINES: List[int] = [0, 4, 5, 6]
DEFAULT_SCENARIOS: List[str] = list(ANOMALY_SCENARIOS.keys())


def run_anomaly_sweep(cfg: Optional[SimConfig] = None,
                       baselines: Iterable[int] = DEFAULT_BASELINES,
                       scenarios: Iterable[str] = DEFAULT_SCENARIOS,
                       output_dir: Optional[str] = None,
                       verbose: bool = True) -> ExperimentResult:
    cfg = apply_profile(cfg or SimConfig(), "profile_anomaly")
    cfg.scenario_name = "anomaly_sweep"

    outdir = Path(output_dir) if output_dir else ensure_outdir(
        cfg.results_dir, "anomaly")

    all_summary: List[pd.DataFrame] = []
    all_slots: List[pd.DataFrame] = []
    scenarios = list(scenarios)
    baselines = list(baselines)

    for sc_name in scenarios:
        sc = ANOMALY_SCENARIOS.get(sc_name)
        if sc is None:
            print(f"[anomaly] unknown scenario: {sc_name}")
            continue
        for bl in baselines:
            c = cfg.clone(baseline_id=bl, **sc)
            c.scenario_name = sc_name
            mc = MetricsCollector(steady_state_fraction=c.steady_state_fraction)
            if verbose:
                print(f"\n[anomaly] scenario={sc_name} baseline="
                      f"{BL_NAMES.get(bl,bl)}")
            run_mc(c, mc, verbose=verbose,
                    extra={"scenario": sc_name,
                            "anomaly_mode": sc.get("anomaly_mode"),
                            "anomaly_prob": sc.get("anomaly_prob")})
            sdf = mc.summary_df()
            sldf = mc.slot_df()
            prefix = f"{sc_name}_bl{bl}_"
            mc.save(str(outdir), prefix=prefix,
                     metadata=make_metadata(c, family="anomaly",
                                              scenario=sc_name,
                                              sweep_variable="anomaly_scenario"))
            if not sdf.empty:
                sdf = sdf.copy()
                sdf["scenario"] = sc_name
                sdf["baseline_id"] = bl
                all_summary.append(sdf)
            if not sldf.empty:
                sldf = sldf.copy()
                sldf["scenario"] = sc_name
                sldf["baseline_id"] = bl
                all_slots.append(sldf)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary \
              else pd.DataFrame()
    slots = pd.concat(all_slots, ignore_index=True) if all_slots \
            else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(outdir / "summary.csv", index=False)
    if not slots.empty:
        slots.to_csv(outdir / "slots.csv", index=False)

    md = make_metadata(cfg, family="anomaly",
                         scenario="anomaly_sweep",
                         sweep_variable="anomaly_scenario",
                         extra={"scenarios": scenarios,
                                 "baselines": baselines,
                                 "scenario_specs": {k: ANOMALY_SCENARIOS[k]
                                                      for k in scenarios}})
    save_metadata(outdir, md)
    return ExperimentResult(family="anomaly", output_dir=outdir,
                              summary=summary, slots=slots, metadata=md)
