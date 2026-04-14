<p align="center">
  <img src="figures/banner.png" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/YassirALKarawi/qdt-isac-6g/actions/workflows/ci.yml"><img src="https://github.com/YassirALKarawi/qdt-isac-6g/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/python-3.9+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="MIT"/></a>
  <a href="#"><img src="https://img.shields.io/badge/status-Publication--Oriented-blue.svg?style=flat-square" alt="Status"/></a>
</p>

<p align="center">
  <b>Trust-Aware Quantum-Assisted Digital Twin Control for Secure Adaptive ISAC in 6G Open RAN</b><br/>
  A publication-oriented simulation framework accompanying an IEEE JSAC Special Issue submission.
</p>

---

## 1. What This Repository Is

This repository is **not** a generic ISAC prototype. It is a **publication-grade
experimental platform** designed to support a single, focused scientific
contribution:

> **Trust-aware closed-loop digital-twin control for secure and adaptive
> ISAC in 6G Open RAN, with quantum-inspired candidate screening used as a
> supporting mechanism for uncertainty-aware action ranking and search-cost
> reduction** — *not* as a claim of universal quantum advantage.

Five claims anchor the simulation campaign. Every experiment family in the
platform was designed to probe at least one of them:

| # | Claim | Primary experiment family |
|---|-------|---------------------------|
| 1 | Digital-twin guidance adds measurable value beyond reactive control | `baseline`, `ablation` |
| 2 | Trust-aware deployment is essential under anomaly or twin mismatch | `anomaly`, `twin_delay` |
| 3 | Quantum-assisted screening reduces effective search burden / improves action quality under uncertainty | `ablation`, `scalability`, `runtime` |
| 4 | The framework degrades gracefully under stale-twin and cyber-physical conditions | `twin_delay`, `anomaly` |
| 5 | Gains do not come at unreasonable energy or runtime cost | `baseline`, `runtime` |

---

## 2. Experiment Families

The simulator is organised as **six independent experiment families**, each
with its own runner, configuration profile, output directory and metadata:

| Family | Runner | Profile | Output directory |
|--------|--------|---------|------------------|
| **Baseline comparison** | `run_baseline_experiment()` | `profile_baseline` | `results/baseline/` |
| **Ablation** | `run_ablation_experiment()` | `profile_ablation` | `results/ablation/` |
| **Anomaly robustness** | `run_anomaly_sweep()` | `profile_anomaly` | `results/anomaly/` |
| **Twin imperfection** | `run_twin_delay_sweep()` | `profile_twin_delay` | `results/twin_delay/` |
| **Scalability** | `run_scalability_experiment()` | `profile_scalability` | `results/scalability/` |
| **Runtime / complexity** | `run_runtime_experiment()` | `profile_runtime` | `results/runtime/` |

Every experiment directory always contains:

```
summary.csv     # MC-averaged run-level summaries (mean, std, 95% CI)
slots.csv       # slot-level telemetry for every run
metadata.json   # scenario, baseline ID, sweep variable, seed,
                # n_monte_carlo, ablation flags, full config
```

---

## 3. Baselines

The baseline set has been deliberately widened so that the proposed method is
evaluated against **stronger algorithmic alternatives**, not just weak internal
controls.

| ID | Baseline | Role |
|----|----------|------|
| 0 | Static ISAC | Fixed equal allocation — lower bound |
| 1 | Reactive Adaptive ISAC | Measurement-driven rebalancing |
| 2 | DT-guided (no QA) | Digital-twin-predicted allocation without screening |
| 3 | DT + QA, attack-unaware | Twin + screening without security awareness |
| 4 | **Full Proposed** | Trust-aware DT + QA screening + adaptive weights |
| 5 | Predictor-based uncertainty-aware | EWMA SINR predictor with conservative margin — *no DT* |
| 6 | Robust min-max heuristic | Worst-user protection — *no DT, no QA* |
| 7 | Learning-based bandit (ε-greedy) | Simple online learner over action arms |

Baseline `-1` (*ablation mode*) drives behaviour from five independent flags:
`use_twin`, `use_trust_gating`, `use_screening`, `use_adaptive_weights`,
`use_mismatch_comp`. The ablation campaign enumerates:

```
no_dt · dt_only · dt_trust · dt_screening · dt_trust_screening · full
```

---

## 4. Metrics

Every numeric metric is reported with **mean, standard deviation, and 95%
confidence interval**, aggregated over a well-defined steady-state window
(`steady_state_fraction`, recorded in each summary row as
`steady_state_start_slot`).

**Twin**: `twin_mismatch_mean`, `twin_mismatch_std`, `twin_fidelity`,
`stale_state_penalty`, `twin_conf`.

**Security / trust-aware deployment**: `fallback_deployment_ratio`,
`unsafe_action_suppression_rate`, `trust_degradation_rate`,
`anomaly_containment_score`, `safe_control_persistence`.

**Quantum-inspired screening**: `candidate_reduction_ratio`,
`search_cost_reduction`, `selected_action_rank_percentile`,
`screening_overhead_ms`.

**Derived**: `energy_utility_tradeoff`, `adaptation_gain`, `robustness_gain`.

---

## 5. Interpretation of the Quantum-Assisted Component

This is important and is deliberately conservative:

- The `quantum_assist.py` module is a **quantum-inspired candidate screening
  and action-ranking layer**.
- It is **not** a hardware-backed quantum optimiser and makes **no claim of
  universal quantum advantage**.
- Its purpose in this framework is to (i) reduce the number of expensive full
  candidate evaluations via a fast surrogate shortlist, (ii) expose a
  Grover-style amplification model with explicit decoherence / gate-fidelity
  penalties, and (iii) produce an uncertainty-aware action ranking inside the
  trust-aware twin-in-the-loop controller.
- Its value is demonstrated only through measurable screening metrics
  (`candidate_reduction_ratio`, `search_cost_reduction`, `rank_percentile`,
  `screening_overhead_ms`).

---

## 6. Quick Start

```bash
git clone https://github.com/YassirALKarawi/qdt-isac-6g.git
cd qdt-isac-6g
pip install -r requirements.txt

# Smoke test (tiny MC × few slots)
python main.py --family baseline --quick

# Run every experiment family with default profiles
python main.py --all

# Individual families
python main.py --family baseline
python main.py --family ablation
python main.py --family anomaly
python main.py --family twin_delay
python main.py --family scalability
python main.py --family runtime

# Ad-hoc single-baseline run
python main.py --baseline 4 --mc 10 --slots 1000
```

All experiment output is written under `--output` (default `results/`) in
the family sub-directories described above.

---

## 7. Reproducibility Workflow

- **Python:** 3.9+
- **Deterministic seeds:** `SimConfig(seed=...)`; each MC run uses
  `seed + run_id * 997` to decorrelate while remaining reproducible.
- **Monte Carlo:** `n_monte_carlo` per baseline, aggregated with 95% CI.
- **Steady-state aggregation:** latter `steady_state_fraction` of the time
  horizon, with the exact start slot stored per-run.
- **Metadata:** each experiment directory has a `metadata.json` recording
  scenario, baseline ID, sweep variable, seed, number of MC runs, the full
  configuration dict, and ablation flags.
- **Tests:** `python -m pytest tests/ -v` covers scientific behaviours
  (sensing power → Pd, twin delay → mismatch, trust drop → conservative
  gating, screening → fewer evaluations).

---

## 8. Composite Utility

$$J = w_c \cdot R_{\text{norm}} + w_s \cdot S + w_{\text{sec}} \cdot T - w_e \cdot E_{\text{norm}}$$

| Symbol | Component | Source |
|:------:|:----------|:------|
| $R_{\text{norm}}$ | Normalised sum-rate | `communication.py` |
| $S$ | Sensing utility (P_d + tracking) | `sensing.py` |
| $T$ | Trust score (EWMA) | `security.py` |
| $E_{\text{norm}}$ | Energy (PA + DSP + circuit + sensing compute) | `network.py` |

Weights adapt online in `controller._adapt()` based on observed outage,
detection, and trust levels.

---

## 9. Project Layout

```
qdt-isac-6g/
├── config.py               # SimConfig, profiles, sweeps, scenarios, regimes
├── channel.py              # Path loss + shadowing + Rician fading + AR(1)
├── network.py              # BS / UE / sensing target entities + mobility + energy
├── communication.py        # OFDMA SINR, throughput, outage
├── sensing.py              # Mono-static OFDM radar (Swerling-I P_d, CRLB)
├── digital_twin.py         # Imperfect DT (delay, noise, staleness, fidelity)
├── security.py             # Anomaly injection, EWMA detection, trust gating
├── quantum_assist.py       # Quantum-inspired candidate screening
├── controller.py           # 8 baselines + ablation mode
├── simulator.py            # Closed-loop simulation engine
├── metrics.py              # MC-level aggregation (mean, std, 95% CI)
├── plotting.py             # Publication-quality figures (PNG + PDF)
├── main.py                 # CLI entry point
├── experiments/            # Experiment runners (six families)
│   ├── baseline.py
│   ├── ablation.py
│   ├── anomaly.py
│   ├── twin_delay.py
│   ├── scalability.py
│   ├── runtime.py
│   └── common.py
├── tests/                  # Scientific-behaviour + unit tests (pytest)
├── figures/                # Legacy / generated plots
├── results/                # Output CSV + metadata per experiment family
└── paper/                  # Companion manuscript
```

---

## 10. Figures (Publication-Grade)

Each experiment runner triggers a corresponding plotting hook. Every figure
is saved as **both PNG and PDF** with consistent labelling and sizing.

| Figure | Path |
|--------|------|
| Baseline bar chart (95% CI) | `results/baseline/figures/baseline_comparison.{png,pdf}` |
| Multi-objective radar | `results/baseline/figures/radar_multiobjective.{png,pdf}` |
| Trust trajectory | `results/baseline/figures/trust_trajectory.{png,pdf}` |
| Twin mismatch trajectory | `results/baseline/figures/twin_mismatch_trajectory.{png,pdf}` |
| Energy–utility trade-off | `results/baseline/figures/energy_utility_tradeoff.{png,pdf}` |
| Utility vs anomaly prob. | `results/anomaly/figures/sweep_anomaly_prob_utility.{png,pdf}` |
| Utility vs twin delay | `results/twin_delay/figures/utility_vs_twin_delay.{png,pdf}` |
| Pd vs twin fidelity | `results/twin_delay/figures/pd_vs_twin_fidelity.{png,pdf}` |
| Scalability plots | `results/scalability/figures/scalability_*.{png,pdf}` |
| Ablation summary | `results/ablation/figures/ablation_summary.{png,pdf}` |
| Search-cost vs shortlist ratio | `results/runtime/figures/search_cost_vs_shortlist.{png,pdf}` |
| Runtime vs candidate-set size | `results/runtime/figures/runtime_vs_candidates.{png,pdf}` |

---

## 11. Limitations

This platform is a research simulator, not a deployment stack:

- Quantum component is quantum-*inspired* (see §5), not hardware-backed.
- PHY / scheduling / channel models are abstracted for tractability and
  are sufficient for comparative algorithmic evaluation but not for
  standards-compliant link-level validation.
- Security metrics are scenario-driven robustness indicators, not
  intrusion-detection benchmarks against CVE-backed adversaries.
- The ISAC model uses mono-static OFDM radar with a Swerling-I approximation
  and a CRLB-style tracking proxy.

---

## 12. Citation

```bibtex
@article{alkarawi2026qdt,
  title   = {Trust-Aware Quantum-Assisted Digital Twin Control for Secure
             and Adaptive ISAC in 6G Open RAN},
  author  = {Al-Karawi, Yassir Ameen Ahmed},
  journal = {Submitted to IEEE Journal on Selected Areas in Communications},
  year    = {2026},
  note    = {Under review}
}
```

---

## 13. License

Simulation code: MIT — see [LICENSE](LICENSE).
Manuscript in `paper/`: © 2026 the author, all rights reserved until
publication (copyright transfers to IEEE on acceptance).
