"""
Twin-delay / twin-fidelity robustness sweep (experiment family: twin_delay).

Sweeps over synchronization delay, fidelity (sinr noise), and position
noise for the full proposed method and strong benchmarks, and also
exercises the three named twin imperfection regimes (low/medium/severe).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from config import SimConfig, SWEEP_CONFIGS, TWIN_REGIMES
from metrics import MetricsCollector
from simulator import run_mc, BL_NAMES

from .common import (apply_profile, ensure_outdir, make_metadata,
                      save_metadata, ExperimentResult)


DEFAULT_BASELINES: List[int] = [0, 2, 4, 5]
DEFAULT_SWEEPS: List[str] = ["twin_delay", "twin_fidelity", "twin_pos_noise"]


def _sweep_one(cfg: SimConfig, name: str, baselines: List[int],
                outdir: Path, verbose: bool) -> (pd.DataFrame, pd.DataFrame):
    spec = SWEEP_CONFIGS[name]
    param, values = spec["param"], spec["values"]
    all_sum, all_sl = [], []
    for v in values:
        for bl in baselines:
            c = cfg.clone(baseline_id=bl, **{param: v})
            c.scenario_name = f"{name}_{v}"
            c.sweep_variable = param
            mc = MetricsCollector(steady_state_fraction=c.steady_state_fraction)
            if verbose:
                print(f"\n[twin_delay] sweep={name} {param}={v} "
                      f"baseline={BL_NAMES.get(bl,bl)}")
            run_mc(c, mc, verbose=verbose,
                    extra={"sweep": name, "sweep_value": v, "baseline_id": bl})
            sdf = mc.summary_df()
            sldf = mc.slot_df()
            prefix = f"{name}_{v}_bl{bl}_"
            mc.save(str(outdir), prefix=prefix,
                     metadata=make_metadata(c, family="twin_delay",
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
    return (pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame(),
            pd.concat(all_sl, ignore_index=True) if all_sl else pd.DataFrame())


def run_twin_delay_sweep(cfg: Optional[SimConfig] = None,
                          sweeps: Iterable[str] = DEFAULT_SWEEPS,
                          baselines: Iterable[int] = DEFAULT_BASELINES,
                          include_regimes: bool = True,
                          output_dir: Optional[str] = None,
                          verbose: bool = True) -> ExperimentResult:
    cfg = apply_profile(cfg or SimConfig(), "profile_twin_delay")
    cfg.scenario_name = "twin_imperfection"

    outdir = Path(output_dir) if output_dir else ensure_outdir(
        cfg.results_dir, "twin_delay")

    all_sum, all_sl = [], []
    baselines = list(baselines)

    for name in sweeps:
        if name not in SWEEP_CONFIGS:
            print(f"[twin_delay] unknown sweep: {name}")
            continue
        s, sl = _sweep_one(cfg, name, baselines, outdir, verbose)
        if not s.empty:
            all_sum.append(s)
        if not sl.empty:
            all_sl.append(sl)

    if include_regimes:
        for regime, params in TWIN_REGIMES.items():
            for bl in baselines:
                c = cfg.clone(baseline_id=bl, **params)
                c.scenario_name = f"regime_{regime}"
                c.sweep_variable = "twin_regime"
                mc = MetricsCollector(steady_state_fraction=c.steady_state_fraction)
                if verbose:
                    print(f"\n[twin_delay] regime={regime} "
                          f"baseline={BL_NAMES.get(bl,bl)}")
                run_mc(c, mc, verbose=verbose,
                        extra={"regime": regime, "baseline_id": bl})
                sdf = mc.summary_df()
                sldf = mc.slot_df()
                prefix = f"regime_{regime}_bl{bl}_"
                mc.save(str(outdir), prefix=prefix,
                         metadata=make_metadata(c, family="twin_delay",
                                                  scenario=f"regime_{regime}",
                                                  sweep_variable="twin_regime",
                                                  extra={"regime_params": params}))
                if not sdf.empty:
                    sdf = sdf.copy()
                    sdf["regime"] = regime
                    sdf["baseline_id"] = bl
                    all_sum.append(sdf)
                if not sldf.empty:
                    sldf = sldf.copy()
                    sldf["regime"] = regime
                    sldf["baseline_id"] = bl
                    all_sl.append(sldf)

    summary = pd.concat(all_sum, ignore_index=True) if all_sum else pd.DataFrame()
    slots = pd.concat(all_sl, ignore_index=True) if all_sl else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(outdir / "summary.csv", index=False)
    if not slots.empty:
        slots.to_csv(outdir / "slots.csv", index=False)

    md = make_metadata(cfg, family="twin_delay",
                         scenario="twin_imperfection",
                         sweep_variable="twin_params",
                         extra={"sweeps": list(sweeps),
                                 "regimes": list(TWIN_REGIMES.keys())
                                              if include_regimes else [],
                                 "baselines": baselines})
    save_metadata(outdir, md)
    return ExperimentResult(family="twin_delay", output_dir=outdir,
                              summary=summary, slots=slots, metadata=md)
