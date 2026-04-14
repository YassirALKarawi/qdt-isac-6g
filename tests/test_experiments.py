"""Tests for the experiment-runner package.

These are fast smoke tests that verify the six experiment families can
be invoked with tiny MC/slot counts and produce expected output files
(summary.csv, slots.csv, metadata.json) under `results/<family>/`.
"""
import json
from pathlib import Path

import pytest

from config import SimConfig
from experiments import (
    run_baseline_experiment,
    run_ablation_experiment,
    run_anomaly_sweep,
    run_twin_delay_sweep,
    run_scalability_experiment,
    run_runtime_experiment,
)


def _tiny_cfg(tmp_path, **kw):
    cfg = SimConfig(seed=1, n_monte_carlo=1, n_slots=50,
                    n_users=6, n_targets=3, n_bs=2, **kw)
    cfg.results_dir = str(tmp_path)
    return cfg


def _assert_outputs(outdir: Path):
    assert (outdir / "summary.csv").exists(), f"missing summary.csv in {outdir}"
    assert (outdir / "metadata.json").exists(), f"missing metadata.json in {outdir}"
    md = json.loads((outdir / "metadata.json").read_text())
    assert "family" in md and "seed" in md and "n_monte_carlo" in md


def test_baseline_experiment_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r = run_baseline_experiment(cfg, baselines=[0, 4], verbose=False)
    _assert_outputs(r.output_dir)
    assert not r.summary.empty


def test_ablation_experiment_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r = run_ablation_experiment(cfg, variants=["no_dt", "full"], verbose=False)
    _assert_outputs(r.output_dir)
    assert not r.summary.empty


def test_anomaly_sweep_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r = run_anomaly_sweep(cfg, baselines=[4],
                           scenarios=["no_attack", "high_anomaly"],
                           verbose=False)
    _assert_outputs(r.output_dir)


def test_twin_delay_sweep_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r = run_twin_delay_sweep(cfg, sweeps=["twin_delay"],
                              baselines=[4],
                              include_regimes=False, verbose=False)
    _assert_outputs(r.output_dir)


def test_scalability_experiment_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r = run_scalability_experiment(cfg, sweeps=["shortlist_ratio"],
                                     baselines=[4], verbose=False)
    _assert_outputs(r.output_dir)


def test_runtime_experiment_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    r = run_runtime_experiment(cfg, baselines=[0, 4],
                                 sweeps=["shortlist_ratio"], verbose=False)
    _assert_outputs(r.output_dir)
