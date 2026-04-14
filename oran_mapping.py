"""
Open RAN architecture alignment: maps framework components to O-RAN
control interfaces and functional splits.

This module provides the conceptual mapping between the simulation
framework and the O-RAN Alliance reference architecture (O-RAN.WG1),
including placement of DT functions, control actions, and telemetry
flows within the near-RT RIC / non-RT RIC hierarchy.

Reference architecture:
    ┌──────────────────────────────────────────────────┐
    │              Non-RT RIC (SMO)                     │
    │  ┌──────────────────┐  ┌──────────────────────┐  │
    │  │ DT Orchestrator  │  │ Trust Policy Engine   │  │
    │  │ (twin lifecycle, │  │ (anomaly thresholds,  │  │
    │  │  sync scheduling)│  │  gating policy)       │  │
    │  └───────┬──────────┘  └──────────┬───────────┘  │
    │          │ A1 (policy)            │ A1 (intent)  │
    └──────────┼────────────────────────┼──────────────┘
               │                        │
    ┌──────────┼────────────────────────┼──────────────┐
    │          ▼       Near-RT RIC      ▼              │
    │  ┌──────────────────┐  ┌──────────────────────┐  │
    │  │ ISAC xApp        │  │ Security xApp        │  │
    │  │ (QA candidate    │  │ (EWMA detection,     │  │
    │  │  search, RB/     │  │  trust update,       │  │
    │  │  power control)  │  │  resource gating)    │  │
    │  └───────┬──────────┘  └──────────┬───────────┘  │
    │          │ E2 (control)           │ E2 (report)  │
    └──────────┼────────────────────────┼──────────────┘
               │                        │
    ┌──────────┼────────────────────────┼──────────────┐
    │          ▼       O-DU / O-RU      ▼              │
    │  ┌──────────────────────────────────────────┐    │
    │  │  ISAC waveform (OFDM radar + DL comms)   │    │
    │  │  Telemetry: SINR, P_d, position, energy  │    │
    │  └──────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────┘
"""
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class ORANLayer(Enum):
    NON_RT_RIC = "non_rt_ric"
    NEAR_RT_RIC = "near_rt_ric"
    O_DU = "o_du"
    O_RU = "o_ru"


class InterfaceType(Enum):
    A1 = "A1"    # non-RT RIC → near-RT RIC (policy/intent)
    E2 = "E2"    # near-RT RIC ↔ O-DU (control/report)
    O1 = "O1"    # SMO → all (management)
    OPEN_FH = "Open Fronthaul"  # O-DU ↔ O-RU


@dataclass
class ComponentMapping:
    """Maps a simulation module to its O-RAN functional placement."""
    sim_module: str
    oran_layer: ORANLayer
    oran_function: str
    interface_in: InterfaceType
    interface_out: InterfaceType
    telemetry: List[str]
    actions: List[str]
    latency_class: str  # 'real-time' | 'near-real-time' | 'non-real-time'
    description: str


# Canonical mapping of framework modules to O-RAN architecture
ARCHITECTURE_MAP: List[ComponentMapping] = [
    ComponentMapping(
        sim_module="digital_twin.DigitalTwin",
        oran_layer=ORANLayer.NON_RT_RIC,
        oran_function="DT Orchestrator rApp",
        interface_in=InterfaceType.O1,
        interface_out=InterfaceType.A1,
        telemetry=[
            "UE position estimates",
            "SINR estimates",
            "target tracking state",
            "confidence scores",
        ],
        actions=[
            "twin sync schedule adjustment",
            "staleness threshold update",
            "confidence decay parameter tuning",
        ],
        latency_class="non-real-time",
        description=(
            "Maintains imperfect digital replica of network state. "
            "Receives telemetry via O1, pushes state predictions and "
            "confidence scores to near-RT RIC via A1 policy enrichment."
        ),
    ),
    ComponentMapping(
        sim_module="controller.Controller (BL4)",
        oran_layer=ORANLayer.NEAR_RT_RIC,
        oran_function="ISAC Resource Allocation xApp",
        interface_in=InterfaceType.A1,
        interface_out=InterfaceType.E2,
        telemetry=[
            "per-UE SINR reports",
            "per-target P_d reports",
            "RB utilisation",
            "BS power levels",
        ],
        actions=[
            "RB allocation per UE",
            "sensing power fraction per BS",
            "Tx power adaptation",
            "weight rebalancing",
        ],
        latency_class="near-real-time",
        description=(
            "Receives twin state and trust scores via A1, executes "
            "quantum-assisted candidate search, sends control actions "
            "to O-DU via E2. Loop period: 10-100 ms."
        ),
    ),
    ComponentMapping(
        sim_module="security.SecurityModel",
        oran_layer=ORANLayer.NEAR_RT_RIC,
        oran_function="Security Monitoring xApp",
        interface_in=InterfaceType.E2,
        interface_out=InterfaceType.A1,
        telemetry=[
            "EWMA anomaly scores",
            "per-entity trust scores",
            "detection/false-alarm counts",
            "active attack registry",
        ],
        actions=[
            "trust score updates",
            "resource gating decisions",
            "anomaly threshold adaptation",
            "alert escalation to non-RT RIC",
        ],
        latency_class="near-real-time",
        description=(
            "Compares live telemetry against twin predictions to detect "
            "anomalies (jamming, spoofing). Updates trust scores that "
            "gate resource allocation in the ISAC xApp."
        ),
    ),
    ComponentMapping(
        sim_module="quantum_assist.QuantumAssist",
        oran_layer=ORANLayer.NEAR_RT_RIC,
        oran_function="QA Acceleration Module (within ISAC xApp)",
        interface_in=InterfaceType.A1,
        interface_out=InterfaceType.E2,
        telemetry=[
            "candidate scores",
            "quantum vs classical eval counts",
            "search overhead (ms)",
        ],
        actions=[
            "candidate generation (twin-informed)",
            "Grover-inspired search execution",
            "best-candidate selection",
        ],
        latency_class="near-real-time",
        description=(
            "Embedded within the ISAC xApp. Uses twin-predicted state "
            "to generate informed candidates, then applies quantum-"
            "amplified search to select the best allocation."
        ),
    ),
    ComponentMapping(
        sim_module="sensing.SensingModel + communication.CommModel",
        oran_layer=ORANLayer.O_DU,
        oran_function="ISAC Signal Processing",
        interface_in=InterfaceType.E2,
        interface_out=InterfaceType.OPEN_FH,
        telemetry=[
            "radar SNR per target",
            "detection outcomes",
            "tracking error estimates",
            "per-UE SINR and throughput",
        ],
        actions=[
            "OFDM waveform configuration",
            "beamforming weight application",
            "sensing integration window",
        ],
        latency_class="real-time",
        description=(
            "Executes joint ISAC waveform: DL communication and mono-"
            "static OFDM radar. Reports measurements upward via E2."
        ),
    ),
]


def telemetry_flow_summary() -> Dict[str, List[str]]:
    """Summarise telemetry flows by interface."""
    flows: Dict[str, List[str]] = {}
    for comp in ARCHITECTURE_MAP:
        key_in = comp.interface_in.value
        key_out = comp.interface_out.value
        flows.setdefault(key_in, []).extend(
            [f"{comp.sim_module} ← {t}" for t in comp.telemetry]
        )
        flows.setdefault(key_out, []).extend(
            [f"{comp.sim_module} → {a}" for a in comp.actions]
        )
    return flows


def control_loop_latency_budget() -> Dict[str, Dict]:
    """Estimate end-to-end control loop latency budget."""
    return {
        'twin_update': {
            'path': 'O-DU → (O1) → non-RT RIC DT',
            'typical_ms': '100-1000',
            'sim_param': 'twin_sync_delay_slots × slot_duration',
        },
        'anomaly_detection': {
            'path': 'O-DU → (E2) → Security xApp',
            'typical_ms': '10-50',
            'sim_param': 'per-slot in controller.step()',
        },
        'resource_allocation': {
            'path': 'ISAC xApp → (E2) → O-DU',
            'typical_ms': '10-100',
            'sim_param': 'controller.latency_ms',
        },
        'full_loop': {
            'path': 'sense → twin update → detect → allocate → apply',
            'typical_ms': '50-200 (near-RT) or 1000+ (non-RT twin refresh)',
            'sim_param': 'end-to-end measured in simulator',
        },
    }


def print_architecture_summary():
    """Print human-readable architecture mapping."""
    print("=" * 72)
    print("  O-RAN Architecture Mapping")
    print("=" * 72)
    for comp in ARCHITECTURE_MAP:
        print(f"\n  [{comp.oran_layer.value}] {comp.oran_function}")
        print(f"    Sim module:  {comp.sim_module}")
        print(f"    Latency:     {comp.latency_class}")
        print(f"    Interface:   {comp.interface_in.value} → "
              f"{comp.interface_out.value}")
        print(f"    Telemetry:   {', '.join(comp.telemetry[:2])}...")
        print(f"    Actions:     {', '.join(comp.actions[:2])}...")
    print("\n" + "=" * 72)
