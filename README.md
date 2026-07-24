# Satellite-Based Quantum Pseudonym Relay: Simulation Code

This repository contains the complete simulation code for the paper: **Satellite-Based Quantum Pseudonym Relay: Performance and Security Analysis**.

The package simulates a **Quantum Pseudonym System (QPS)** that uses satellite-based Quantum Key Distribution (QKD) to provision pseudonym tokens, a token-buffer consumption model with weather resilience, an orbital CRDT revocation layer, and a timing side-channel adversary analysis with Poisson cover-traffic mitigation.

---

## Reproducibility

All figures and data tables in the paper can be reproduced by running:

```bash
python3 run_satellite_demo.py --full
```

The pipeline runs approximately 10–30 minutes on a modern workstation and writes all outputs to `outputs/graphs/` (PNG figures) and `outputs/data/` (CSV data). See [Pipeline Overview](#pipeline-overview) for a detailed breakdown of each simulation step.

---

## System Requirements

- **Linux** or **macOS** (native). On **Windows**, use [WSL](https://learn.microsoft.com/en-us/windows/wsl/) or a Linux VM.
- **Python** ≥ 3.10
- **NetSquid** — free-ware quantum-network simulator ([netsquid.org](https://netsquid.org))

The non-QKD modules (token buffer, CRDT, adversary) use only the Python standard library and can be imported on any platform. Only the photon-level BB84 simulation requires NetSquid.

---

## Installation

### 1. NetSquid account

Register at [netsquid.org](https://netsquid.org) to obtain your credentials. NetSquid is required for the photon-level BB84 decoy-state backend.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

NetSquid is pulled from a private PyPI during install — see `requirements.txt` for credentials setup instructions.

---

## Pipeline Overview

The simulation runs in the following order, each step corresponding to a figure or analysis in the paper:

```
 Step   | Name                         | Outputs     | Module
--------|------------------------------|-------------|--------------------------------------------
 G1     | Token issuance during pass   | G1          | satellite_qkd/pass_simulator.py
 G2     | QBER vs turbulence           | G2          | satellite_qkd/sweep_turbulence.py
 Alt    | Altitude sweep               | —           | satellite_qkd/sweep_altitude.py
 Orb    | Orbital sensitivity          | —           | satellite_qkd/orbital_dynamics.py
 G3     | Buffer level trajectory      | G3          | token_buffer/analysis.py
 G4     | Cloudy-pass resilience       | G4          | token_buffer/analysis.py
 G8     | Weather DoS                  | G8          | token_buffer/weather_dos.py
 G5     | Revocation convergence       | G5          | orbital_crdt/analysis.py
 G6     | False-accept timing          | G6          | orbital_crdt/analysis.py
 Race   | Revocation race demonstrator | —           | orbital_crdt/revocation_race.py
 Hybrid | Hybrid CRDT topology         | —           | orbital_crdt/simulation.py
 M1     | Mitigation transcript build  | —           | adversary/timing_analysis.py
 G7a    | Linkability (unmitigated)    | G7 (solid)  | adversary/timing_analysis.py
 M2     | Cover traffic mitigation     | —           | token_buffer/simulation.py
 G7b    | Linkability (cover traffic)  | G7 (dashed) | adversary/timing_analysis.py
 G9     | Yearly survival              | G9          | token_buffer/analysis.py
 G10    | Staleness sensitivity        | G10         | token_buffer/analysis.py
 Var    | NetSquid variance char       | —           | satellite_qkd/netsquid_backend.py
```

After all steps, `summary.json` is written with acceptance checks, provenance metadata, and all intermediate results.

---

## Output Reference

### Figures

| Output File                                    | Description                                      |
|------------------------------------------------|--------------------------------------------------|
| `G1_token_issuance.png`                 | Cumulative token issuance during one zenith pass |
| `G2_qber_vs_turbulence.png`            | QBER across the atmospheric turbulence range     |
| `G3_buffer_level.png`                   | Token-buffer level trajectory over 7 days        |
| `G4_cloudy_pass_resilience.png`         | Survivable cloudy passes vs consumption rate     |
| `G5_revocation_convergence.png`         | CRDT convergence time vs orbital period          |
| `G6_false_accept_timing.png`            | False-accept window vs revoke-presentation delay |
| `G7_linkability_advantage.png`          | Linkability scores (unmitigated)                 |
| `G7_linkability_cover_traffic.png`      | Linkability scores (with cover traffic)          |
| `G8_weather_dos.png`                    | Degraded probability vs storm severity           |
| `G9_yearly_survival.png`               | Buffer survival probability over 1 year          |
| `G10_staleness_sensitivity.png`        | Sensitivity to token expiry window               |
| `altitude_sweep_tokens.png`            | Token yield vs satellite altitude                 |

### Data files

| File                                  | Description                              |
|---------------------------------------|------------------------------------------|
| `linkability_advantage.csv`           | G7 data (unmitigated)                    |
| `linkability_cover_traffic.csv`       | G7 data (with cover traffic)             |
| `contract_B_randomized_*_per_hour.csv` | Contract B rows (random consumption)    |
| `contract_B_cover_50_per_hour.csv`   | Contract B rows (cover traffic)          |
| `contract_C_presentation_transcripts.csv` | Contract C transcript data           |
| `cloudy_pass_resilience.csv`          | G4 raw data                              |
| `weather_dos_sweep.csv`               | G8 raw data                              |
| `crdt_convergence.csv`                | G5 raw data                              |
| `false_accept_timing.csv`             | G6 raw data                              |
| `sweep_turbulence.csv`                | G2 raw data                              |
| `sweep_altitude.csv`                  | Altitude sweep raw data                  |
| `revocation_race_log.json`            | Revocation race events                   |
| `revocation_race_log.csv`             | Revocation race events (tabular)         |
| `orbital_sensitivity.json`            | Orbital model comparison                 |
| `netsquid_variance.json`              | NetSquid Monte Carlo variance            |
| `buffer_timeseries_*_per_hour.csv`    | Buffer level time series                 |
| `hybrid_convergence_comparison.csv`   | Hybrid CRDT topology comparison          |
| `byzantine_scenario_results.json`     | Byzantine fault scenario results         |
| `yearly_survival.csv`                 | G9 raw data                              |
| `staleness_sensitivity.csv`          | G10 raw data                             |
| `pass_clear_zenith.json`              | Clear-sky pass parameters                |

---

## Module Reference

### `satellite_qkd/` — Satellite QKD Workstream

| File                     | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `channel_model.py`       | Free-space channel physics: atmospheric loss, depolarization, channel state |
| `orbital_dynamics.py`    | Pass geometry (sinusoidal, Gaussian, Keplerian), link budget, key-rate estimation |
| `bb84_satellite.py`      | Decoy-state BB84 simulation orchestration over one-second orbital intervals |
| `netsquid_backend.py`    | NetSquid weighted-Monte-Carlo backend for photon-level BB84              |
| `pass_simulator.py`      | Single satellite-pass simulation orchestrator                           |
| `sweep_turbulence.py`    | Sweep atmospheric turbulence strength `Cₙ²` and measure QBER/token yield |
| `sweep_altitude.py`      | Sweep satellite altitude and measure token yield                        |

**Entry point:** `simulate_pass()` — run one satellite pass with the NetSquid backend.

### `token_buffer/` — Token Buffer Workstream

| File               | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `buffer.py`        | `Token` and `TokenBuffer` — capacity-limited buffer with FIFO/random consumption and staleness expiry |
| `pass_schedule.py` | `PassWindow`, `WeatherMarkovChain` — pass scheduling and multi-state (4-state) Markov weather model |
| `simulation.py`    | Discrete-event 7-day buffer simulation with configurable consumption policies including cover traffic |
| `analysis.py`      | Graph generation: G3 (buffer level), G4 (cloudy-pass resilience), G9 (yearly survival), G10 (staleness sensitivity) |
| `weather_dos.py`   | Weather-based denial-of-service analysis (extended storm events)        |

**Entry point:** `simulate_buffer()` — run the token-buffer model over a configurable time horizon.

### `orbital_crdt/` — Orbital CRDT Revocation Workstream

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `node.py`             | `GSetNode` — grow-only set CRDT with revocation semantics               |
| `scheduler.py`        | `SatelliteVisibilityScheduler` — periodic visibility windows for LEO satellites |
| `simulation.py`       | `simulate_three_node_convergence()` — CRDT convergence with satellite + terrestrial links |
| `revocation_race.py`  | `run_revocation_race()` — the 7-step revocation race demonstration      |
| `analysis.py`         | Graph generation: G5 (convergence sweep), G6 (false-accept timing curve)|

**Entry points:**
- `simulate_three_node_convergence()` — measure CRDT propagation delay
- `run_revocation_race()` — demonstrate false-accept then reject

### `adversary/` — Timing Side-Channel Adversary

| File                | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `timing_analysis.py`| Contract C transcript builder, linkability advantage estimator, graph generators |

**Entry points:**
- `build_contract_c_transcripts()` — build adversarial observation transcripts from presentation events
- `estimate_linkability_advantage()` — compute within-pass and cross-pass linkage scores
- `graph_linkability()` — produce G7 linkability plots

### `common/` — Shared Utilities

| File          | Purpose                                                |
|---------------|--------------------------------------------------------|
| `io.py`       | File I/O helpers: `ensure_dir`, `write_json`, `write_csv` |
| `random.py`   | `poisson_count`, `noisy_count`, `percentile`            |
| `plotting.py` | `write_line_chart` — SVG/PNG line chart renderer with CI bands, thresholds, and bands |
| `statistics.py` | Bootstrap CI, Cohen's d, power analysis, KS normality test |

### Top-Level Scripts

| File                          | Purpose                                                    |
|-------------------------------|------------------------------------------------------------|
| `run_satellite_demo.py`       | Main pipeline runner — orchestrates all workstreams        |

---

## Configuration

All simulation parameters live in `satellite_qkd/config.json`. Key groups:

| Section          | Key Parameters                                                |
|------------------|--------------------------------------------------------------|
| `orbital`        | Altitude, Earth radius, ground-station separation, pass duration |
| `link`           | Wavelength, telescope apertures, atmospheric extinction, pointing loss |
| `qkd`            | Source rate, decoy-state parameters, detector specs, token size |
| `token_buffer`   | Buffer capacity, weather model/transitions, passes/day, cover traffic ratio |
| `crdt`           | Satellite/terrestrial latencies, merge interval               |
| `adversary`      | Number of pseudonym holders, verifier set, max sessions       |
| `sweep`          | Sweep ranges and trial counts for all parametric sweeps       |

---

## Running Individual Components

### Single satellite pass

```python
from satellite_qkd.pass_simulator import simulate_pass

result = simulate_pass(
    {"altitude_km": 500, "cn2": 1e-17, "max_elevation_deg": 90},
    seed=42,
)
print(result["summary"]["tokens_issued"], "tokens produced")
```

### Token-buffer simulation

```python
from token_buffer.simulation import simulate_buffer

sim = simulate_buffer(
    days=7,
    consumption_rate_per_hour=50,
    consumption_policy="random",      # or "fifo" or "random_cover"
    seed=123,
)
print(len(sim["contract_B"]), "presentation events")
```

### Linkability analysis

```python
from adversary.timing_analysis import build_contract_c_transcripts, estimate_linkability_advantage

transcripts = build_contract_c_transcripts(sim["presentation_events"], seed=777)
rows = estimate_linkability_advantage(transcripts, consumption_policy="random")
for row in rows:
    print(f"  {row['sessions_observed']:4d} sessions → cross-pass L = {row['cross_pass_advantage']:.3f}")
```

### CRDT revocation race

```python
from orbital_crdt.revocation_race import run_revocation_race

race = run_revocation_race()
print(f"False-accept window: {race['false_accept_window_min']:.1f} min")
print(f"After convergence: {'accepted' if race['after_accept'] else 'rejected'}")
```

---

## Running Tests

```bash
python3 -m unittest discover tests -v
```

| File                     | Coverage                                                |
|--------------------------|---------------------------------------------------------|
| `test_satellite_qkd.py`  | Slant range, atmospheric loss, QBER bands, pass yield   |
| `test_token_buffer.py`   | FIFO consumption, token expiry, simulation contract rows|
| `test_orbital_crdt.py`   | Satellite scheduler, convergence bound, revocation race |
| `test_adversary.py`      | Linkability advantage below threshold for random buffer |

---

## License

Copyright © 2026 Aryan Mondal

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
