"""Run the full Satellite QKD Pseudonym Relay simulation.

Usage:
    python3 run_satellite_demo.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _tqdm:  # type: ignore[no-redef]  # no-op fallback
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.n = 0
            self.total = 0

        def set_description(self, label: str) -> None:
            print(f"  {label}...")

        def update(self, n: int = 1) -> None:
            self.n += n
            if self.total:
                print(f"    {self.n}/{self.total}")

        def close(self) -> None:
            pass

from adversary.timing_analysis import build_contract_c_transcripts, graph_linkability
from common.io import ensure_dir, write_csv, write_json
from common.plotting import PALETTE, write_line_chart
from common.statistics import bootstrap_ci, validation_report
from orbital_crdt.analysis import graph_convergence, graph_false_accept
from orbital_crdt.revocation_race import run_revocation_race
from orbital_crdt.simulation import simulate_byzantine_scenario, simulate_three_node_convergence
from satellite_qkd.orbital_dynamics import compare_orbital_models, keplerian_elevation_profile, total_loss_timeseries, load_config
from satellite_qkd.pass_simulator import simulate_pass, write_pass_output
from satellite_qkd.sweep_altitude import altitude_points, run_altitude_sweep
from satellite_qkd.sweep_turbulence import run_turbulence_sweep
from satellite_qkd.netsquid_backend import characterize_variance
from token_buffer.analysis import graph_buffer_level, graph_resilience, graph_yearly_survival, graph_staleness_sensitivity
from token_buffer.simulation import simulate_buffer
from token_buffer.weather_dos import graph_weather_dos


def _git_hash() -> str:
    """Return the short SHA of the current git HEAD, or ``"unknown"``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent),
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_sha(config: dict[str, Any]) -> str:
    """Return a short SHA-256 digest of the sorted config JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


CN2_LABEL = "C<sub>n</sub><sup>2</sup>"
CN2_AXIS_LABEL = f"Turbulence {CN2_LABEL} (m<sup>-2/3</sup>)"
TOKEN_FORMULA = "N<sub>tok</sub> = K<sub>sec</sub> / 256 bits"


def _graph_g1(
    output_path: Path,
    config: dict[str, Any],
    seed: int,
    pulses_per_interval: int | None,
) -> dict[str, Any]:
    """Produce Graph G1: cumulative token issuance during a satellite pass."""
    sweep_cfg = config.get("sweep", {})
    cn2_values = [float(v) for v in sweep_cfg.get("cn2_g1_values", [1e-17, 1e-15, 1e-14])]
    n_trials = int(sweep_cfg.get("cn2_g1_trials", 10))
    token_size = int(config.get("qkd", {}).get("token_size_bits", 256))
    curves = []
    primary_result: dict[str, Any] | None = None
    for idx, cn2 in enumerate(cn2_values):
        all_curves: list[list[tuple[float, float]]] = []
        for trial in range(n_trials):
            result = simulate_pass(
                {
                    "pass_id": trial,
                    "altitude_km": 500.0,
                    "max_elevation_deg": 90.0,
                    "cn2": cn2,
                },
                config=config,
                seed=seed + idx * 1000 + trial,
                pulses_per_interval=pulses_per_interval,
            )
            if trial == 0 and idx == 0:
                primary_result = result
            cumulative = 0.0
            pts: list[tuple[float, float]] = []
            for row in result["timeseries"]:
                cumulative += float(row["secure_bits"]) / token_size
                pts.append((float(row["time_sec"]) / 60.0, cumulative))
            all_curves.append(pts)
        n_steps = min(len(c) for c in all_curves)
        median_pts = []
        ci_low = []
        ci_high = []
        for t in range(n_steps):
            step_vals = [c[t][1] for c in all_curves]
            boot = bootstrap_ci(step_vals, statistic="median", n_resamples=10000, seed=seed + t + 12345)
            median_pts.append((all_curves[0][t][0], boot["stat"]))
            ci_low.append((all_curves[0][t][0], boot["low"]))
            ci_high.append((all_curves[0][t][0], boot["high"]))
        curves.append(
            {
                "name": f"{CN2_LABEL}={cn2:.0e} m<sup>-2/3</sup>",
                "color": PALETTE[idx],
                "points": median_pts,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    write_line_chart(
        output_path,
        "Token Issuance During Satellite Pass",
        "Time t (min)",
        "Cumulative tokens N<sub>tok</sub>(t)",
        curves,
        y_min=0,
        subtitle=f"Median ± 95% CI over {n_trials} trials; {CN2_LABEL} values: {', '.join(f'{v:.0e}' for v in cn2_values)} m<sup>-2/3</sup>",
        column="double", markers=True,
    )
    assert primary_result is not None
    return primary_result


def _graph_g2(
    output_path: Path,
    csv_path: Path,
    config: dict[str, Any],
    seed: int,
    pulses_per_interval: int | None,
    trials: int | None,
) -> list[dict[str, float]]:
    """Produce Graph G2: QBER vs atmospheric turbulence sweep."""
    rows = run_turbulence_sweep(
        config=config,
        seed=seed,
        trials=trials,
        pulses_per_interval=pulses_per_interval,
    )
    write_csv(csv_path, rows)
    n_trials = int(rows[0]["trials"]) if rows else 0
    write_line_chart(
        output_path,
        "QBER vs Atmospheric Turbulence",
        CN2_AXIS_LABEL,
        "Mean QBER Q",
        [
            {
                "name": "mean Q",
                "points": [(row["Cn2"], row["mean_QBER"]) for row in rows],
                "ci_low": [(row["Cn2"], row["qber_ci95_low"]) for row in rows],
                "ci_high": [(row["Cn2"], row["qber_ci95_high"]) for row in rows],
            }
        ],
        x_scale="log",
        y_min=0,
        thresholds=[{"value": 0.11, "label": "Q<sub>abort</sub>=0.11", "color": "#D55E00"}],
        subtitle=f"Mean ± 95% CI over {n_trials} trials; Q = bit errors / sifted bits",
        column="double", markers=True,
    )
    return rows


def _write_altitude_sweep(
    graph_path: Path,
    csv_path: Path,
    config: dict[str, Any],
    seed: int,
    pulses_per_interval: int | None,
    trials: int | None,
) -> list[dict[str, float]]:
    """Produce altitude sweep: token yield vs satellite altitude."""
    rows = run_altitude_sweep(
        config=config,
        seed=seed,
        trials=trials,
        pulses_per_interval=pulses_per_interval,
    )
    write_csv(csv_path, rows)
    n_trials = int(rows[0]["trials"]) if rows else 0
    write_line_chart(
        graph_path,
        "Altitude Sweep Token Yield",
        "Altitude h (km)",
        "Mean tokens per pass N<sub>tok</sub>",
        [
            {
                "name": "mean N<sub>tok</sub>",
                "points": altitude_points(rows),
                "ci_low": [(row["altitude_km"], row["ci95_low"]) for row in rows],
                "ci_high": [(row["altitude_km"], row["ci95_high"]) for row in rows],
            }
        ],
        y_min=0,
        subtitle=f"N<sub>tok</sub>(h) = K<sub>sec</sub>(h) / 256 bits; shaded band = 95% CI over {n_trials} trials",
        column="double", markers=True,
    )
    return rows


def _acceptance_summary(
    clear_pass: dict[str, Any],
    strong_turbulence_pass: dict[str, Any],
    g4_rows: list[dict[str, float]],
    g7_rows: list[dict[str, float]],
    race: dict[str, Any],
    g7_cover_rows: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build acceptance check dict from pipeline results.

    Args:
        clear_pass: WS1 zenith pass simulation.
        strong_turbulence_pass: High-turbulence abort pass.
        g4_rows: WS2 cloud-pass resilience rows.
        g7_rows: WS3 adversary linkability rows (unmitigated).
        race: CRDT revocation race result.
        g7_cover_rows: WS3 adversary linkability rows (with cover traffic).

    Returns:
        Dict with ws1, ws2, ws3, adversary keys and individual checks.
    """
    timeseries = clear_pass["timeseries"]
    zenith_row = max(timeseries, key=lambda row: row["elevation_deg"])
    low_row = min(timeseries, key=lambda row: row["elevation_deg"])
    rate_50 = next(row for row in g4_rows if row["consumption_rate_per_hour"] == 50.0)
    rate_100 = next(row for row in g4_rows if row["consumption_rate_per_hour"] == 100.0)
    # Use converged advantage at max sessions (most data-rich estimate)
    cross_pass_at_1000 = next(row["cross_pass_advantage"] for row in g7_rows if row["sessions_observed"] == 1000.0)
    cover_cross_pass_at_1000 = next(
        (row["cross_pass_advantage"] for row in (g7_cover_rows or g7_rows) if row["sessions_observed"] == 1000.0),
        cross_pass_at_1000,
    )
    return {
        "ws1": {
            "clear_zenith_secure_key_bits": clear_pass["summary"]["total_secure_key_bits"],
            "clear_zenith_tokens": clear_pass["summary"]["tokens_issued"],
            "clear_zenith_key_yield_in_100_300_kbit_range": 100_000 <= clear_pass["summary"]["total_secure_key_bits"] <= 300_000,
            "zenith_expected_QBER": zenith_row["expected_QBER"],
            "zenith_QBER_in_0_02_0_04_range": 0.02 <= zenith_row["expected_QBER"] <= 0.04,
            "low_elevation_expected_QBER": low_row["expected_QBER"],
            "low_elevation_QBER_exceeds_0_08": low_row["expected_QBER"] > 0.08,
            "strong_turbulence_abort_fraction": 1.0 - strong_turbulence_pass["summary"]["pass_duration_above_threshold_sec"] / strong_turbulence_pass["summary"]["duration_sec"],
        },
        "ws2": {
            "survivable_cloudy_passes_at_50_per_hour": rate_50["cloudy_passes_survivable"],
            "survives_3_cloudy_passes_at_50_per_hour": rate_50["cloudy_passes_survivable"] >= 3.0,
            "survivable_cloudy_passes_at_100_per_hour": rate_100["cloudy_passes_survivable"],
            "degrades_after_1_to_2_cloudy_passes_at_100_per_hour": 1.0 <= rate_100["cloudy_passes_survivable"] <= 2.0,
        },
        "ws3": {
            "race_false_accept_window_min": race["false_accept_window_min"],
            "race_rejects_after_convergence": not race["after_accept"],
        },
        "adversary": {
            "unmitigated_cross_pass_advantage": cross_pass_at_1000,
            "unmitigated_cross_pass_advantage_below_0_55": cross_pass_at_1000 <= 0.55,
            "cover_traffic_cross_pass_advantage": cover_cross_pass_at_1000,
            "cross_pass_advantage_below_0_55": cover_cross_pass_at_1000 <= 0.55,
        },
    }


def run_full(
    output_dir: Path,
    seed: int,
    pulses_per_interval: int | None,
    sweep_pulses_per_interval: int | None,
    altitude_trials: int,
    turbulence_trials: int,
) -> dict[str, Any]:
    """Run the complete G1–G10 simulation pipeline.

    Orchestrates all workstreams in order, writes graphs and data files,
    and returns a summary dict with acceptance checks.
    """
    config = load_config()
    graph_dir = ensure_dir(output_dir / "graphs")
    data_dir = ensure_dir(output_dir / "data")

    steps = [
        "G1: token issuance pass",
        "G2: turbulence sweep",
        "Altitude sweep",
        "Orbital sensitivity + Keplerian",
        "G3: buffer level",
        "G4: cloudy-pass resilience",
        "G8: weather DoS",
        "G5: revocation convergence",
        "G6: false-accept timing",
        "Revocation race",
        "Hybrid CRDT topology",
        "Mitigation buffer + transcripts",
        "G7: linkability (no cover)",
        "Cover traffic mitigation",
        "G7: linkability (with cover)",
        "G9: yearly survival",
        "G10: staleness sensitivity",
        "NetSquid variance char",
        "Parameter table",
        "Strong-turbulence pass",
        "Writing summary",
    ]
    progress = _tqdm(total=len(steps), unit="step", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    def _step(label: str) -> None:
        progress.set_description(label)
        progress.update(1)

    clear_pass = _graph_g1(graph_dir / "G1_token_issuance.png", config, seed, pulses_per_interval)
    write_pass_output(clear_pass, data_dir / "pass_clear_zenith.json")
    _step(steps[0])

    g2_rows = _graph_g2(
        graph_dir / "G2_qber_vs_turbulence.png",
        data_dir / "sweep_turbulence.csv",
        config,
        seed + 100,
        sweep_pulses_per_interval,
        turbulence_trials,
    )
    _step(steps[1])

    altitude_rows = _write_altitude_sweep(
        graph_dir / "altitude_sweep_tokens.png",
        data_dir / "sweep_altitude.csv",
        config,
        seed + 200,
        sweep_pulses_per_interval,
        altitude_trials,
    )
    _step(steps[2])

    orbital_sensitivity = compare_orbital_models(altitude_km=500.0, cn2=1e-15)
    # Add Keplerian model comparison
    keplerian_elevation_profile(altitude_km=500.0)
    keplerian_result = simulate_pass(
        {"pass_id": 888, "altitude_km": 500.0, "max_elevation_deg": 90.0, "cn2": 1e-15},
        config=config, seed=seed + 250,
    )
    orbital_sensitivity.append({
        "model": "keplerian",
        "estimated_key_bits": keplerian_result["summary"]["total_secure_key_bits"],
        "estimated_tokens": keplerian_result["summary"]["tokens_issued"],
    })
    write_json(data_dir / "orbital_sensitivity.json", orbital_sensitivity)
    _step(steps[3])

    buffer_sims = graph_buffer_level(graph_dir / "G3_buffer_level.png", clear_pass["contract_A"], config=config, seed=seed + 300)
    for rate, sim in buffer_sims.items():
        write_csv(data_dir / f"buffer_timeseries_{rate}_per_hour.csv", sim["timeseries"])
        write_csv(data_dir / f"contract_B_{rate}_per_hour.csv", sim["contract_B"])
    _step(steps[4])

    g4_rows = graph_resilience(
        graph_dir / "G4_cloudy_pass_resilience.png",
        data_dir / "cloudy_pass_resilience.csv",
        clear_pass["contract_A"],
        config=config,
    )
    _step(steps[5])

    g8_rows = graph_weather_dos(
        graph_dir / "G8_weather_dos.png",
        data_dir / "weather_dos_sweep.csv",
        pass_contract_A=clear_pass["contract_A"],
        config=config,
    )
    _step(steps[6])

    g5_rows = graph_convergence(graph_dir / "G5_revocation_convergence.png", data_dir / "crdt_convergence.csv", config=config)
    _step(steps[7])

    g6_rows = graph_false_accept(graph_dir / "G6_false_accept_timing.png", data_dir / "false_accept_timing.csv", config=config)
    _step(steps[8])

    race = run_revocation_race(config=config)
    write_json(data_dir / "revocation_race_log.json", race)
    write_csv(data_dir / "revocation_race_log.csv", race["events"])
    _step(steps[9])

    # Hybrid CRDT topology comparison (satellite vs fiber backup)
    crdt_cfg = config.get("crdt", {})
    hybrid_results: list[dict[str, Any]] = []
    for period_min in [60.0, 90.0, 120.0, 180.0]:
        satellite_only = simulate_three_node_convergence(
            token_id="T-hybrid",
            origin="GS-A",
            orbital_period_min=period_min,
            next_pass_offset_min=period_min / 2.0,
            config=config,
        )
        hybrid_cfg = dict(config)
        hybrid_cfg["crdt"] = dict(crdt_cfg)
        hybrid_cfg["crdt"]["terrestrial_latency_max_ms"] = 50.0  # fiber backup
        fiber_backup = simulate_three_node_convergence(
            token_id="T-hybrid",
            origin="GS-A",
            orbital_period_min=period_min,
            config=hybrid_cfg,
            fiber_backup=True,
        )
        hybrid_results.append({
            "orbital_period_min": period_min,
            "satellite_only_convergence_min": satellite_only["max_convergence_time_sec"] / 60.0,
            "satellite_only_false_accept_min": satellite_only["false_accept_window_sec"] / 60.0,
            "hybrid_convergence_min": fiber_backup["max_convergence_time_sec"] / 60.0,
            "hybrid_false_accept_min": fiber_backup["false_accept_window_sec"] / 60.0,
        })
    write_csv(data_dir / "hybrid_convergence_comparison.csv", hybrid_results)

    # Byzantine scenario
    byzantine_results = {}
    for scenario, drop_flag, targets in [
        ("non_responding", True, None),
        ("selective_drop", True, ["T-other"]),
        ("honest", False, None),
    ]:
        byz_result = simulate_byzantine_scenario(
            token_id="T-secure",
            origin="GS-A",
            byzantine_node="GS-C",
            drop_revocations=drop_flag,
            target_token_ids=targets,
            config=config,
        )
        byzantine_results[scenario] = byz_result
    write_json(data_dir / "byzantine_scenario_results.json", byzantine_results)
    _step(steps[10])

    mitigation_buffer = simulate_buffer(
        days=7.0,
        consumption_rate_per_hour=50.0,
        pass_contract_A=clear_pass["contract_A"],
        config=config,
        consumption_policy="random",
        seed=seed + 400,
    )
    write_csv(data_dir / "contract_B_randomized_50_per_hour.csv", mitigation_buffer["contract_B"])
    transcripts = build_contract_c_transcripts(mitigation_buffer["presentation_events"], config=config, seed=seed + 500)
    write_csv(data_dir / "contract_C_presentation_transcripts.csv", transcripts)
    _step(steps[11])

    g7_rows = graph_linkability(
        graph_dir / "G7_linkability_advantage.png",
        data_dir / "linkability_advantage.csv",
        transcripts,
        consumption_policy="random",
        config=config,
    )
    _step(steps[12])

    # Cover traffic mitigation: run with random_cover policy
    cover_buffer = simulate_buffer(
        days=7.0,
        consumption_rate_per_hour=50.0,
        pass_contract_A=clear_pass["contract_A"],
        config=config,
        consumption_policy="random_cover",
        seed=seed + 450,
    )
    cover_transcripts = build_contract_c_transcripts(cover_buffer["presentation_events"], config=config, seed=seed + 550)
    g7_cover_rows = graph_linkability(
        graph_dir / "G7_linkability_cover_traffic.png",
        data_dir / "linkability_cover_traffic.csv",
        cover_transcripts,
        consumption_policy="random_cover",
        config=config,
    )
    write_csv(data_dir / "contract_B_cover_50_per_hour.csv", cover_buffer["contract_B"])
    _step(steps[13])

    # G9: Yearly survival
    g9_rows = graph_yearly_survival(
        graph_dir / "G9_yearly_survival.png",
        consumption_rate_per_hour=50.0,
        pass_contract_A=clear_pass["contract_A"],
        config=config,
        seed=seed + 700,
    )
    write_csv(data_dir / "yearly_survival.csv", g9_rows)
    _step(steps[14])

    # G10: Staleness sensitivity
    g10_rows = graph_staleness_sensitivity(
        graph_dir / "G10_staleness_sensitivity.png",
        pass_contract_A=clear_pass["contract_A"],
        config=config,
    )
    write_csv(data_dir / "staleness_sensitivity.csv", g10_rows)
    _step(steps[15])

    # NetSquid variance characterization
    variance_char = characterize_variance(config=config, seed=seed + 800)
    write_json(data_dir / "netsquid_variance.json", variance_char)
    _step(steps[16])

    _step(steps[17])

    strong_turbulence = simulate_pass(
        {
            "pass_id": 999,
            "altitude_km": 500.0,
            "max_elevation_deg": 90.0,
            "cn2": 1e-14,
        },
        config=config,
        seed=seed + 600,
        pulses_per_interval=pulses_per_interval,
    )
    _step(steps[18])

    summary = {
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_hash": _git_hash(),
            "config_sha256": _config_sha(config),
            "seed": seed,
            "backend": clear_pass["summary"]["backend"],
        },
        "outputs": {
            "graphs": str(graph_dir),
            "data": str(data_dir),
        },
        "config": config,
        "clear_pass": clear_pass["summary"],
        "g2_points": len(g2_rows),
        "altitude_points": len(altitude_rows),
        "orbital_sensitivity": orbital_sensitivity,
        "g5_points": len(g5_rows),
        "g6_points": len(g6_rows),
        "g8_points": len(g8_rows),
        "g9_points": len(g9_rows),
        "g10_points": len(g10_rows),
        "hybrid_crdt_results": hybrid_results,
        "variance_characterization": variance_char,
        "transcripts": len(transcripts),
        "cover_transcripts": len(cover_transcripts),
        "acceptance": _acceptance_summary(clear_pass, strong_turbulence, g4_rows, g7_rows, race, g7_cover_rows),
        "statistical_validation": {
            "clear_pass_QBER": validation_report([row["expected_QBER"] for row in clear_pass["timeseries"]]),
            "strong_turbulence_QBER": validation_report([row["expected_QBER"] for row in strong_turbulence["timeseries"]]),
        },
    }
    write_json(output_dir / "summary.json", summary)
    _step(steps[19])
    _step(steps[20])
    progress.close()
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simulation pipeline."""
    parser = argparse.ArgumentParser(description="Satellite QKD Pseudonym Relay simulation")
    parser.add_argument("--full", action="store_true", help="run the complete G1-G10 pipeline")
    parser.add_argument("--output-dir", default="outputs", help="directory for generated data and graphs")
    parser.add_argument("--seed", type=int, default=20260607, help="base random seed")
    parser.add_argument(
        "--pulses-per-interval",
        type=int,
        default=None,
        help="NetSquid Monte Carlo sample pulses per one-second interval",
    )
    parser.add_argument(
        "--sweep-pulses-per-interval",
        type=int,
        default=None,
        help="NetSquid sample pulses per interval for sweeps",
    )
    parser.add_argument("--altitude-trials", type=int, default=None, help="trials per altitude sweep point (default: from config sweep.altitude_trials)")
    parser.add_argument("--turbulence-trials", type=int, default=None, help="trials per turbulence sweep point (default: from config sweep.turbulence_trials)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.full:
        raise SystemExit("Pass --full to run the complete simulation pipeline.")
    summary = run_full(
        Path(args.output_dir),
        seed=args.seed,
        pulses_per_interval=args.pulses_per_interval,
        sweep_pulses_per_interval=args.sweep_pulses_per_interval,
        altitude_trials=args.altitude_trials,
        turbulence_trials=args.turbulence_trials,
    )
    print("Satellite demo complete")
    print(f"Graphs: {summary['outputs']['graphs']}")
    print(f"Data: {summary['outputs']['data']}")
    print(f"Summary: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
