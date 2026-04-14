"""
Baseline comparison experiment (experiment family: baseline).

Runs every baseline in `BL_NAMES` on the same scenario and writes:

    results/baseline/summary.csv        (MC-averaged per-run summaries)
    results/baseline/slots.csv          (slot-level telemetry, all baselines)
    results/baseline/metadata.json
    results/baseline/bl{id}_summary.csv (per-baseline summary)
    results/baseline/bl{id}_slots.csv   (per-baseline slot telemetry)
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List

import pandas as pd

from config import SimConfig
from metrics import MetricsCollector
from simulator import run_mc, BL_NAMES

from .common import (apply_profile, ensure_outdir, make_metadata,
                      save_metadata, ExperimentResult)


DEFAULT_BASELINES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]


def run_baseline_experiment(cfg: Optional[SimConfig] = None,
                              baselines: Iterable[int] = DEFAULT_BASELINES,
                              output_dir: Optional[str] = None,
                              verbose: bool = True) -> ExperimentResult:
    cfg = apply_profile(cfg or SimConfig(), "profile_baseline")
    cfg.scenario_name = "baseline_comparison"
    baselines = list(baselines)

    outdir = Path(output_dir) if output_dir else ensure_outdir(
        cfg.results_dir, "baseline")
    outdir.mkdir(parents=True, exist_ok=True)

    all_summary: List[pd.DataFrame] = []
    all_slots: List[pd.DataFrame] = []
    for bl in baselines:
        c = cfg.clone(baseline_id=bl)
        mc = MetricsCollector(steady_state_fraction=c.steady_state_fraction)
        run_mc(c, mc, verbose=verbose)
        sdf = mc.summary_df()
        sldf = mc.slot_df()
        # Persist per-baseline
        mc.save(str(outdir), prefix=f"bl{bl}_",
                 metadata=make_metadata(c, family="baseline",
                                          scenario=f"baseline_{BL_NAMES.get(bl, bl)}"))
        if not sdf.empty:
            all_summary.append(sdf)
        if not sldf.empty:
            sldf = sldf.copy()
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

    md = make_metadata(cfg, family="baseline",
                         scenario="baseline_comparison",
                         sweep_variable="baseline_id",
                         extra={"baselines": baselines,
                                 "baseline_names":
                                     {str(b): BL_NAMES.get(b, f"BL{b}")
                                      for b in baselines}})
    save_metadata(outdir, md)

    return ExperimentResult(family="baseline", output_dir=outdir,
                              summary=summary, slots=slots, metadata=md)
