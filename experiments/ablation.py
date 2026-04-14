"""
Ablation experiment (experiment family: ablation).

Runs the simulator in ablation mode (baseline_id == -1) with the ablation
flags toggled according to `ABLATION_VARIANTS`. The variants isolate the
effect of every major component: digital twin guidance, trust-aware
deployment, quantum-assisted screening, adaptive weight updates, and twin
mismatch compensation.

Variants (see config.ABLATION_VARIANTS):
    no_dt, dt_only, dt_trust, dt_screening, dt_trust_screening, full
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List

import pandas as pd

from config import SimConfig, ABLATION_VARIANTS
from metrics import MetricsCollector
from simulator import run_mc

from .common import (apply_profile, ensure_outdir, make_metadata,
                      save_metadata, ExperimentResult)


DEFAULT_VARIANTS: List[str] = [
    "no_dt", "dt_only", "dt_trust", "dt_screening",
    "dt_trust_screening", "full",
]


def run_ablation_experiment(cfg: Optional[SimConfig] = None,
                              variants: Iterable[str] = DEFAULT_VARIANTS,
                              output_dir: Optional[str] = None,
                              verbose: bool = True) -> ExperimentResult:
    cfg = apply_profile(cfg or SimConfig(), "profile_ablation")
    cfg.scenario_name = "ablation"

    outdir = Path(output_dir) if output_dir else ensure_outdir(
        cfg.results_dir, "ablation")

    all_summary: List[pd.DataFrame] = []
    all_slots: List[pd.DataFrame] = []
    for v in variants:
        flags = ABLATION_VARIANTS.get(v)
        if flags is None:
            print(f"[ablation] unknown variant: {v}")
            continue
        c = cfg.clone(baseline_id=-1, **flags)
        c.scenario_name = f"ablation_{v}"
        mc = MetricsCollector(steady_state_fraction=c.steady_state_fraction)
        if verbose:
            print(f"\n[ablation] variant = {v}  flags = {flags}")
        run_mc(c, mc, verbose=verbose, extra={"variant": v})
        sdf = mc.summary_df()
        sldf = mc.slot_df()
        mc.save(str(outdir), prefix=f"abl_{v}_",
                 metadata=make_metadata(c, family="ablation",
                                          scenario=f"ablation_{v}",
                                          sweep_variable="ablation_variant",
                                          extra={"variant": v, "flags": flags}))
        if not sdf.empty:
            sdf = sdf.copy()
            sdf["variant"] = v
            all_summary.append(sdf)
        if not sldf.empty:
            sldf = sldf.copy()
            sldf["variant"] = v
            all_slots.append(sldf)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary \
              else pd.DataFrame()
    slots = pd.concat(all_slots, ignore_index=True) if all_slots \
            else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(outdir / "summary.csv", index=False)
    if not slots.empty:
        slots.to_csv(outdir / "slots.csv", index=False)

    md = make_metadata(cfg, family="ablation",
                         scenario="ablation_campaign",
                         sweep_variable="ablation_variant",
                         extra={"variants": list(variants),
                                 "variant_flags":
                                     {v: ABLATION_VARIANTS[v]
                                      for v in variants if v in ABLATION_VARIANTS}})
    save_metadata(outdir, md)
    return ExperimentResult(family="ablation", output_dir=outdir,
                              summary=summary, slots=slots, metadata=md)
