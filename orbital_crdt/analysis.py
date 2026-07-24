"""Graph generation for orbital CRDT revocation behavior."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from common.io import write_csv
from common.plotting import write_line_chart
from common.random import percentile
from satellite_qkd.orbital_dynamics import load_config
from .simulation import simulate_three_node_convergence


def convergence_sweep(
    orbital_periods_min: list[float] | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Return Graph G5 rows with jitter-sampled latency for each orbital period."""
    cfg = config or load_config()
    sweep_cfg = cfg.get("sweep", {})
    crdt_cfg = cfg.get("crdt", {})
    periods = orbital_periods_min or [float(p) for p in sweep_cfg.get("orbital_periods_min", [60.0, 90.0, 120.0, 180.0])]
    n_samples = int(sweep_cfg.get("convergence_samples", 10))
    lat_min = float(crdt_cfg.get("terrestrial_latency_min_ms", 50.0)) / 1000.0
    lat_max = float(crdt_cfg.get("terrestrial_latency_max_ms", 200.0)) / 1000.0
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    for period in periods:
        fa_windows: list[float] = []
        conv_times: list[float] = []
        for _ in range(n_samples):
            jittered_cfg = {k: v for k, v in cfg.items()}
            jitter_ms = rng.uniform(lat_min * 1000.0, lat_max * 1000.0)
            jittered_cfg["crdt"] = dict(crdt_cfg)
            jittered_cfg["crdt"]["terrestrial_latency_max_ms"] = jitter_ms
            result = simulate_three_node_convergence(
                token_id=f"T-{int(period)}",
                origin="GS-A",
                orbital_period_min=period,
                next_pass_offset_min=period / 2.0,
                config=jittered_cfg,
            )
            fa_windows.append(result["false_accept_window_sec"] / 60.0)
            conv_times.append(result["max_convergence_time_sec"] / 60.0)
        rows.append(
            {
                "orbital_period_min": period,
                "false_accept_window_min": sum(fa_windows) / len(fa_windows),
                "fa_window_ci95_low": percentile(fa_windows, 0.025),
                "fa_window_ci95_high": percentile(fa_windows, 0.975),
                "max_convergence_time_min": sum(conv_times) / len(conv_times),
                "convergence_ci95_low": percentile(conv_times, 0.025),
                "convergence_ci95_high": percentile(conv_times, 0.975),
                "samples": float(n_samples),
            }
        )
    return rows


def graph_convergence(
    output_path: str | Path,
    csv_path: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Produce Graph G5."""
    rows = convergence_sweep(config=config)
    write_csv(csv_path, rows)
    write_line_chart(
        output_path,
        "Revocation Convergence vs Orbital Period",
        "Orbital period T<sub>orb</sub> (min)",
        "False-accept window W<sub>FA</sub> (min)",
        [{"name": "GS-C worst case W<sub>FA</sub>", "points": [(row["orbital_period_min"], row["false_accept_window_min"]) for row in rows]}],
        y_min=0,
        bands=[
            {
                "from": row["fa_window_ci95_low"],
                "to": row["fa_window_ci95_high"],
                "color": "#bfdbfe",
            }
            for row in rows
        ] if len(rows) > 0 and rows[0].get("fa_window_ci95_low", 0) != rows[0].get("fa_window_ci95_high", 0) else None,
        subtitle="W<sub>FA</sub> = t<sub>converge</sub> - t<sub>revoke</sub>; shaded band = 95% CI over latency jitter",
        column="double", markers=True,
    )
    return rows


def false_accept_curve(
    orbital_period_min: float = 94.6,
    pass_duration_min: float = 10.0,
    step_min: float = 5.0,
) -> list[dict[str, float]]:
    """Return Graph G6 rows."""
    rows: list[dict[str, float]] = []
    x = 0.0
    ramp_end = orbital_period_min / 2.0
    while x <= orbital_period_min + 1e-9:
        if x <= pass_duration_min:
            probability = 0.0
        elif x <= ramp_end:
            probability = min(1.0, (x - pass_duration_min) / max(1e-9, ramp_end - pass_duration_min))
        else:
            probability = 1.0
        rows.append({"minutes_since_last_pass": x, "false_accept_probability": probability})
        x += step_min
    if rows[-1]["minutes_since_last_pass"] < orbital_period_min:
        rows.append({"minutes_since_last_pass": orbital_period_min, "false_accept_probability": 1.0})
    return rows


def graph_false_accept(
    output_path: str | Path,
    csv_path: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Produce Graph G6."""
    cfg = config or load_config()
    orbital = cfg.get("orbital", {})
    token_cfg = cfg.get("token_buffer", {})
    rows = false_accept_curve(
        orbital_period_min=float(orbital.get("orbital_period_min", 94.6)),
        pass_duration_min=float(token_cfg.get("pass_duration_min", 10.0)),
    )
    write_csv(csv_path, rows)
    write_line_chart(
        output_path,
        "False-Accept Rate vs Pass Timing",
        "Time since last pass t (min)",
        "P<sub>FA</sub>(t)",
        [{"name": "GS-C P<sub>FA</sub>", "points": [(row["minutes_since_last_pass"], row["false_accept_probability"]) for row in rows]}],
        y_min=0,
        y_max=1,
        subtitle="P<sub>FA</sub>(t)=0 for t <= T<sub>pass</sub>; ramps to 1 by T<sub>orb</sub>/2",
        column="double", markers=False,
    )
    return rows
