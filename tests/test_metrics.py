"""Metrics collector tests."""
import math
import json
from pathlib import Path

import pandas as pd

from metrics import MetricsCollector, ci95, aggregate_across_runs


def test_metrics_summary_created():
    mc = MetricsCollector(steady_state_fraction=0.5)
    for i in range(10):
        mc.record({
            "slot": i,
            "sum_rate": 100 + i,
            "avg_pd": 0.7,
            "trust": 0.9,
            "utility": 0.4,
            "energy": 1.0,
            "energy_norm": 0.5,
        })
    mc.end_run(run_id=0, bl_id=4)
    df = mc.summary_df()
    assert len(df) == 1
    assert "sum_rate_mean" in df.columns
    assert "avg_pd_ci95" in df.columns
    assert "energy_utility_tradeoff" in df.columns


def test_steady_state_fraction():
    mc = MetricsCollector(steady_state_fraction=0.8)
    for i in range(100):
        mc.record({"slot": i, "sum_rate": float(i), "avg_pd": 0.5,
                    "trust": 1.0, "utility": 0.3, "energy": 1.0,
                    "energy_norm": 0.5})
    mc.end_run(run_id=0, bl_id=0)
    df = mc.summary_df()
    assert df["sum_rate_mean"].iloc[0] > 80
    assert df["steady_state_start_slot"].iloc[0] == 80


def test_ci95_math():
    assert ci95(0.0, 10) == 0.0
    assert ci95(1.0, 1) == 0.0
    assert math.isclose(ci95(2.0, 100), 1.96 * 2.0 / 10.0, rel_tol=1e-6)


def test_metadata_saved(tmp_path):
    mc = MetricsCollector(steady_state_fraction=0.5)
    for i in range(20):
        mc.record({"slot": i, "sum_rate": 100.0, "avg_pd": 0.8,
                    "trust": 0.95, "utility": 0.4, "energy": 0.9,
                    "energy_norm": 0.5})
    mc.end_run(run_id=0, bl_id=4)
    mc.save(str(tmp_path), metadata={"family": "unit_test", "seed": 1,
                                       "n_monte_carlo": 1, "n_slots": 20})
    meta = json.loads((Path(tmp_path) / "metadata.json").read_text())
    assert meta["family"] == "unit_test"
    assert meta["seed"] == 1
    assert (Path(tmp_path) / "summary.csv").exists()
    assert (Path(tmp_path) / "slots.csv").exists()


def test_aggregate_across_runs_basic():
    df = pd.DataFrame({
        "baseline_id": [0, 0, 4, 4],
        "utility_mean": [0.2, 0.22, 0.4, 0.42],
    })
    agg = aggregate_across_runs(df, group_cols=["baseline_id"],
                                  metric_cols=["utility"])
    assert "utility_mu" in agg.columns
    assert "utility_ci95" in agg.columns
    assert len(agg) == 2
