"""
Shared utilities for experiment runners.
"""
from __future__ import annotations
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config import SimConfig, PROFILES


@dataclass
class ExperimentResult:
    """Return type for experiment runners."""
    family: str
    output_dir: Path
    summary: pd.DataFrame
    slots: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def apply_profile(cfg: SimConfig, profile_name: str) -> SimConfig:
    """Return a copy of `cfg` with profile defaults applied.

    Profile values are only applied for fields that the caller left at their
    SimConfig default value — explicit caller overrides always win. This lets
    tests pass tiny `n_monte_carlo` / `n_slots` while production runs use
    the profile defaults."""
    c = copy.deepcopy(cfg)
    defaults = SimConfig()
    prof = PROFILES.get(profile_name, {})
    for k, v in prof.items():
        # Keep user override; otherwise use profile default
        if getattr(c, k) == getattr(defaults, k):
            setattr(c, k, v)
    c.profile_name = profile_name
    c.experiment_family = prof.get("experiment_family", c.experiment_family)
    return c


def make_metadata(cfg: SimConfig,
                    *,
                    family: str,
                    scenario: str = "",
                    sweep_variable: str = "",
                    extra: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a reproducibility metadata dict for `metadata.json`."""
    md: Dict[str, Any] = {
        "family": family,
        "scenario": scenario or cfg.scenario_name,
        "sweep_variable": sweep_variable or cfg.sweep_variable,
        "baseline_id": cfg.baseline_id,
        "seed": cfg.seed,
        "n_monte_carlo": cfg.n_monte_carlo,
        "n_slots": cfg.n_slots,
        "steady_state_fraction": cfg.steady_state_fraction,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": cfg.profile_name,
        "ablation_flags": {
            "use_twin": cfg.use_twin,
            "use_trust_gating": cfg.use_trust_gating,
            "use_screening": cfg.use_screening,
            "use_adaptive_weights": cfg.use_adaptive_weights,
            "use_mismatch_comp": cfg.use_mismatch_comp,
        },
        "config": cfg.to_dict(),
    }
    if extra:
        md.update(extra)
    return md


def save_metadata(outdir: Path, metadata: Dict[str, Any],
                   filename: str = "metadata.json") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / filename, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def ensure_outdir(base: str, family: str) -> Path:
    p = Path(base) / family
    p.mkdir(parents=True, exist_ok=True)
    return p
