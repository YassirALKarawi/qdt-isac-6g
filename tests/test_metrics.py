from metrics import MetricsCollector


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
        })

    mc.end_run(run_id=0, bl_id=4)
    df = mc.summary_df()

    assert len(df) == 1
    assert "sum_rate_mean" in df.columns
    assert "avg_pd_ci95" in df.columns


def test_steady_state_fraction():
    mc = MetricsCollector(steady_state_fraction=0.8)

    for i in range(100):
        mc.record({"slot": i, "sum_rate": float(i), "avg_pd": 0.5,
                    "trust": 1.0, "utility": 0.3, "energy": 1.0})

    mc.end_run(run_id=0, bl_id=0)
    df = mc.summary_df()

    # Steady-state starts at slot 80, so mean should be ~89.5
    assert df["sum_rate_mean"].iloc[0] > 80
