<p align="center">
  <img src="figures/banner.png" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/YassirALKarawi/qdt-isac-6g/actions/workflows/ci.yml"><img src="https://github.com/YassirALKarawi/qdt-isac-6g/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/python-3.9+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="MIT"/></a>
  <a href="#-example-results"><img src="https://img.shields.io/badge/status-Research_Prototype-orange.svg?style=flat-square" alt="Status"/></a>
  <a href="https://github.com/YassirALKarawi/qdt-isac-6g/stargazers"><img src="https://img.shields.io/github/stars/YassirALKarawi/qdt-isac-6g?style=flat-square&color=yellow" alt="Stars"/></a>
</p>

<p align="center">
  A publication-oriented discrete-time simulation framework for evaluating
  <b>quantum-assisted / quantum-inspired digital twin</b> control architectures in
  <b>Integrated Sensing and Communication (ISAC)</b> systems for <b>6G Open RAN</b> networks.
  <br/>
  <sub>Designed for comparative algorithmic evaluation, reproducible experiments, and research prototyping.
  Not intended to represent a full standards-compliant physical-layer or deployment-grade Open RAN stack.</sub>
</p>

---

## 🏛️ Architecture

<p align="center">
  <img src="figures/architecture.png" width="850"/>
</p>


### Digital Twin Module

<p align="center">
  <img src="figures/digital_twin_module.png" width="850"/>
</p>

### Quantum-Assisted Engine

<p align="center">
  <img src="figures/quantum_engine.png" width="850"/>
</p>

### Security Module

<p align="center">
  <img src="figures/security_module.png" width="850"/>
</p>

### Closed-Loop Control Flow

<p align="center">
  <img src="figures/control_loop.png" width="850"/>
</p>

### ISAC Signal Model

<p align="center">
  <img src="figures/isac_signal_model.png" width="850"/>
</p>

---

## ✨ Key Features

| Module | Description |
|:-------|:-----------|
| **6G Open RAN** | 4 BS × 64 antennas, 40 UE, 10 sensing targets, 1 km² area |
| **Channel** | 3GPP path loss, log-normal shadowing, Rician fading, AR(1) temporal correlation |
| **Communication** | OFDMA downlink, per-RB SINR, Shannon throughput, outage detection |
| **Sensing** | Mono-static OFDM radar, Swerling-I P_d, CRLB range estimation, clutter |
| **Digital Twin** | Imperfect: sync delay, measurement noise, SINR estimation error, state staleness |
| **Security** | EWMA anomaly detection (dual-channel), Bayesian trust dynamics, jamming + spoofing |
| **Quantum Assist** | Quantum-inspired candidate screening for uncertainty-aware action ranking |
| **Controller** | Closed-loop adaptive: per-BS power control, sensing/comm split, trust gating |
| **Evaluation** | 50 Monte Carlo runs, 3000 slots, 12+ parameter sweeps, 95% CI |

---

## ⚠️ Current Limitations

This framework is intentionally abstraction-driven and should be interpreted as a research simulation platform rather than a full deployment stack.

- The quantum module is **quantum-inspired / quantum-assisted** at the algorithmic level, not a hardware-backed quantum optimisation engine
- The sensing model is simplified and intended for comparative ISAC evaluation
- The interference and scheduling models are abstracted for tractability
- Security metrics are scenario-driven and should be interpreted as resilience indicators rather than full intrusion-detection benchmarks

---

## 📊 Results

### Multi-Objective Performance

<p align="center">
  <img src="figures/radar_comparison.png" width="550"/>
</p>

The Full Proposed method achieves the best overall trade-off across communication, sensing, security, and energy efficiency.

### Example Results (illustrative configuration: 5 MC × 500 slots)

<p align="center">
  <img src="figures/baseline_bars.png" width="850"/>
</p>

| Method | Sum Rate (Mbps) | P_d | Trust | Energy | Utility |
|:-------|:-:|:-:|:-:|:-:|:-:|
| Static ISAC | 1064 ± 121 | 0.600 | 1.000 | 0.996 | 0.276 |
| Adaptive ISAC | 976 ± 106 | 0.559 | 1.000 | 0.997 | 0.264 |
| DT (no QA) | 1009 ± 76 | 0.546 | 0.839 | 0.997 | 0.352 |
| DT+QA (no Sec) | 1051 ± 130 | 0.524 | 1.000 | 0.998 | 0.263 |
| **Full Proposed** | **1086 ± 95** | **0.743** | 0.837 | **0.965** | **0.397** |

### Time Evolution — Full Proposed Method

<p align="center">
  <img src="figures/time_evolution.png" width="850"/>
</p>

### Cumulative Distribution Functions

<p align="center">
  <img src="figures/cdf_all.png" width="850"/>
</p>

---

## 🔬 Sensitivity Analysis

### Attack Intensity Sweep

<p align="center">
  <img src="figures/sweep_anomaly.png" width="850"/>
</p>

The Full Proposed method maintains superior utility under increasing anomaly probability, demonstrating the value of security-aware closed-loop control.

### Digital Twin Delay Sweep

<p align="center">
  <img src="figures/sweep_twin_delay.png" width="850"/>
</p>

Performance degrades gracefully as twin synchronisation delay increases, validating the framework's robustness to imperfect state information.

---

## 🏗️ Baselines

| ID | Method | Digital Twin | Quantum Assist | Security | Description |
|:--:|:-------|:--:|:--:|:--:|:-----------|
| 0 | Static ISAC | ✗ | ✗ | ✗ | Fixed equal allocation, no adaptation |
| 1 | Adaptive ISAC | ✗ | ✗ | ✗ | Measurement-based rebalancing |
| 2 | DT-guided | ✓ | ✗ | ✓ | Twin-predicted PF allocation |
| 3 | DT+QA (attack-unaware) | ✓ | ✓ | ✗ | Quantum search, blind to attacks |
| 4 | **Full Proposed** | ✓ | ✓ | ✓ | Complete closed-loop system |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YassirALKarawi/qdt-isac-6g.git
cd qdt-isac-6g

# Install
pip install -r requirements.txt

# Quick test (3 MC, 300 slots)
python main.py --quick

# Full run (50 MC, 3000 slots)
python main.py --mc 50 --slots 3000

# Single baseline
python main.py --baseline 4 --mc 10 --slots 1000

# Parameter sweep
python main.py --sweep anomaly_prob
python main.py --sweep twin_delay
python main.py --sweep clutter
```

---

## 🔄 Reproducibility

- **Python version:** 3.9+
- **Seed-controlled** execution via `SimConfig(seed=...)`
- **Monte Carlo averaging** supported via `--mc`
- **Steady-state aggregation** computed over the latter 50% of the simulated time horizon
- **Slot-level and run-level** results exported to CSV under `results/`
- **Figures** saved under `plots/`
- **Tests:** `python -m pytest tests/ -v`

---

## 📐 Composite Utility

$$J = w_c \cdot R_{\text{norm}} + w_s \cdot S + w_{\text{sec}} \cdot T - w_e \cdot E_{\text{norm}}$$

| Symbol | Component | Source |
|:------:|:----------|:------|
| $R_{\text{norm}}$ | Normalised sum-rate | `communication.py` |
| $S$ | Sensing utility (P_d + tracking) | `sensing.py` |
| $T$ | Trust score (Bayesian adaptive) | `security.py` |
| $E_{\text{norm}}$ | Energy consumption (PA + DSP + circuit) | `network.py` |

Weights adapt online via the closed-loop controller based on observed outage, detection, and trust levels.

---

<details>
<summary><h2>📁 Project Structure</h2></summary>

```
qdt-isac-6g/
├── config.py            # Simulation parameters + 12 sweep configs
├── channel.py           # Path loss, shadowing, Rician fading, AR(1)
├── network.py           # BS, UE, targets + mobility + energy model
├── communication.py     # SINR, throughput, outage (OFDMA)
├── sensing.py           # Radar SNR, Swerling-I P_d, CRLB
├── digital_twin.py      # Imperfect DT: delay, noise, staleness
├── security.py          # EWMA detection + Bayesian trust
├── quantum_assist.py    # Quantum-inspired candidate search
├── controller.py        # 5 baseline controllers
├── simulator.py         # Discrete-time simulation engine
├── metrics.py           # CSV/JSON metrics export + CI95
├── plotting.py          # Publication-quality figures
├── main.py              # CLI entry point
├── tests/               # Verification tests (pytest)
├── figures/             # Generated plots
├── results/             # Output CSV/JSON
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

</details>

---

<details>
<summary><h2>🔧 Available Sweeps (12 experiments)</h2></summary>

| Sweep | Parameter | Values |
|:------|:----------|:-------|
| `user_density` | n_users | 10, 20, 40, 60, 80 |
| `anomaly_prob` | anomaly_prob | 0.0 – 0.20 |
| `twin_delay` | twin_sync_delay_slots | 0 – 50 |
| `clutter` | clutter_to_noise_ratio_db | 0 – 15 dB |
| `target_speed` | target_speed_range | 1 – 100 m/s |
| `sensing_power` | sensing_power_fraction | 0.05 – 0.40 |
| `target_density` | n_targets | 2 – 40 |
| `scalability` | n_bs | 2 – 16 |
| `quantum_onoff` | qa_enabled | True / False |
| `twin_fidelity` | twin_sinr_noise_std | 0.5 – 10 dB |
| `mobility` | user_speed_range | pedestrian – vehicular |
| `weight_sweep` | weight_comm | 0.1 – 0.7 |

</details>

---

## 📖 Citation

```bibtex
@article{alkarawi2026qdt,
  title   = {Quantum-Assisted Digital Twin for Closed-Loop Secure and 
             Adaptive ISAC in 6G Open RAN},
  author  = {Al-Karawi, Yassir Ameen Ahmed},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2026}
}
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
