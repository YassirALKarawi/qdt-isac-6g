# QDT-ISAC: Quantum-Assisted Digital Twin for Secure Adaptive ISAC in 6G Open RAN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade discrete-time simulation framework for evaluating **Quantum-Assisted Digital Twin** architectures in **Integrated Sensing and Communication (ISAC)** systems for **6G Open RAN** networks.

## Key Features

- **Closed-loop ISAC simulator** with 10 ms slot resolution over 3000 time slots
- **Imperfect Digital Twin** with synchronisation delay, measurement noise, and state staleness
- **Quantum-assisted resource allocation** using Grover-inspired candidate screening with realistic decoherence
- **EWMA-based anomaly detection** with Bayesian trust dynamics
- **Adaptive power control** and sensing/communication resource balancing
- **5 baseline methods** for rigorous comparative evaluation
- **12+ parameter sweeps** for comprehensive sensitivity analysis
- **Publication-quality plots** with 95% confidence intervals

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  6G Open RAN │───▶│  Digital Twin │───▶│   Quantum    │
│  (4 BS, 40   │    │  (Imperfect) │    │   Assist     │
│   UE, 10 TG) │    │              │    │  (Grover)    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│           Adaptive Controller (Closed-Loop)           │
│  • Security-aware trust gating                        │
│  • Per-BS power adaptation                            │
│  • Sensing/comm resource split                        │
└──────────────────────────────────────────────────────┘
```

## Baselines

| ID | Method | DT | Quantum | Security |
|----|--------|----|---------|----------|
| 0 | Static ISAC | ✗ | ✗ | ✗ |
| 1 | Adaptive ISAC | ✗ | ✗ | ✗ |
| 2 | DT-guided (no QA) | ✓ | ✗ | ✓ |
| 3 | DT+QA (attack-unaware) | ✓ | ✓ | ✗ |
| 4 | **Full Proposed** | ✓ | ✓ | ✓ |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/qdt-isac-6g.git
cd qdt-isac-6g

# Install dependencies
pip install -r requirements.txt

# Quick test (3 MC runs, 300 slots)
python main.py --quick

# Full JSAC-grade run (50 MC, 3000 slots)
python main.py --mc 50 --slots 3000

# Single baseline
python main.py --baseline 4 --mc 10 --slots 1000

# Parameter sweep
python main.py --sweep anomaly_prob
python main.py --sweep twin_delay
python main.py --sweep clutter
```

## Results (5 MC × 500 slots)

| Method | Sum Rate (Mbps) | P_d | Trust | Energy | Utility |
|--------|:-:|:-:|:-:|:-:|:-:|
| Static ISAC | 1064 ± 121 | 0.600 | 1.000 | 0.996 | 0.276 |
| Adaptive ISAC | 976 ± 106 | 0.559 | 1.000 | 0.997 | 0.264 |
| DT (no QA) | 1009 ± 76 | 0.546 | 0.839 | 0.997 | 0.352 |
| DT+QA (no Sec) | 1051 ± 130 | 0.524 | 1.000 | 0.998 | 0.263 |
| **Full Proposed** | **1086 ± 95** | **0.743** | 0.837 | **0.965** | **0.397** |

## Composite Utility

$$J = w_c \cdot R_{\text{norm}} + w_s \cdot S + w_{\text{sec}} \cdot T - w_e \cdot E_{\text{norm}}$$

where:
- $R_{\text{norm}}$: normalised sum-rate
- $S$: sensing utility (detection probability + tracking accuracy)
- $T$: trust score (security-aware, adaptive threshold)
- $E_{\text{norm}}$: normalised energy consumption

## Project Structure

```
qdt-isac-6g/
├── config.py           # All simulation parameters + sweep configs
├── channel.py          # Path loss, shadowing, Rician fading, AR(1)
├── network.py          # BS, UE, target entities + mobility
├── communication.py    # SINR, throughput, outage (OFDMA downlink)
├── sensing.py          # Radar SNR, Swerling-I P_d, CRLB tracking
├── digital_twin.py     # Imperfect DT: delay, noise, staleness
├── security.py         # EWMA anomaly detection + Bayesian trust
├── quantum_assist.py   # Grover-inspired candidate screening
├── controller.py       # 5 baseline controllers
├── simulator.py        # Discrete-time simulation engine
├── metrics.py          # Metrics collection + CSV/JSON export
├── plotting.py         # Publication-quality matplotlib figures
├── main.py             # CLI entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## Available Sweeps

| Sweep | Parameter | Values |
|-------|-----------|--------|
| `user_density` | n_users | 10, 20, 40, 60, 80 |
| `anomaly_prob` | anomaly_prob | 0.0 – 0.20 |
| `twin_delay` | twin_sync_delay_slots | 0 – 50 |
| `clutter` | clutter_to_noise_ratio_db | 0 – 15 dB |
| `target_speed` | target_speed_range | 1–100 m/s |
| `sensing_power` | sensing_power_fraction | 0.05 – 0.40 |
| `target_density` | n_targets | 2 – 40 |
| `mobility` | user_speed_range | pedestrian – vehicular |
| `scalability` | n_bs | 2 – 16 |
| `quantum_onoff` | qa_enabled | True / False |
| `twin_fidelity` | twin_sinr_noise_std | 0.5 – 10 dB |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{alkarawi2026qdt,
  title={Quantum-Assisted Digital Twin for Closed-Loop Secure and 
         Adaptive ISAC in 6G Open RAN},
  author={Al-Karawi, Yassir},
  journal={IEEE Journal on Selected Areas in Communications},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
