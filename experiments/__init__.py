"""
Experiment runners for the six evaluation families:

    run_baseline_experiment()      -> results/baseline/
    run_ablation_experiment()      -> results/ablation/
    run_anomaly_sweep()            -> results/anomaly/
    run_twin_delay_sweep()         -> results/twin_delay/
    run_scalability_experiment()   -> results/scalability/
    run_runtime_experiment()       -> results/runtime/

Each runner writes at least `summary.csv`, `slots.csv` and `metadata.json`
for full reproducibility. Metadata includes scenario name, baseline ID,
sweep variable, seed, number of Monte Carlo runs, and the key configuration
parameters.
"""
from .common import apply_profile, make_metadata, ExperimentResult
from .baseline import run_baseline_experiment
from .ablation import run_ablation_experiment
from .anomaly import run_anomaly_sweep
from .twin_delay import run_twin_delay_sweep
from .scalability import run_scalability_experiment
from .runtime import run_runtime_experiment

__all__ = [
    "apply_profile",
    "make_metadata",
    "ExperimentResult",
    "run_baseline_experiment",
    "run_ablation_experiment",
    "run_anomaly_sweep",
    "run_twin_delay_sweep",
    "run_scalability_experiment",
    "run_runtime_experiment",
]
