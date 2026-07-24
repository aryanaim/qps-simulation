"""Altitude sweep for satellite QKD token yield."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Mapping

from common.random import percentile
from .orbital_dynamics import load_config
from .pass_simulator import simulate_pass


def run_altitude_sweep(
    altitudes_km: list[float] | None = None,
    trials: int | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int = 42,
    pulses_per_interval: int | None = None,
) -> list[dict[str, float]]:
    """Sweep altitude and return mean token yield with 95 percent intervals.

    Uses the NetSquid BB84 backend for all simulation trials.
    """
    cfg = dict(config or load_config())
    sweep_cfg = cfg.get("sweep", {})
    rng = random.Random(seed)
    n_trials = trials if trials is not None else int(sweep_cfg.get("altitude_trials", 50))
    altitudes = altitudes_km or [float(a) for a in sweep_cfg.get("altitudes_km", [400.0, 500.0, 600.0, 800.0, 1000.0])]
    rows: list[dict[str, float]] = []
    for altitude in altitudes:
        yields: list[float] = []
        for trial in range(n_trials):
            cn2 = 10.0 ** rng.uniform(-17.0, -15.0)
            max_elevation = rng.uniform(45.0, 90.0)
            result = simulate_pass(
                {
                    "pass_id": trial,
                    "altitude_km": altitude,
                    "cn2": cn2,
                    "max_elevation_deg": max_elevation,
                },
                config=cfg,
                seed=seed + trial + int(altitude),
                pulses_per_interval=pulses_per_interval,
            )
            yields.append(float(result["summary"]["tokens_issued"]))
        rows.append(
            {
                "altitude_km": altitude,
                "mean_tokens_per_pass": statistics.fmean(yields),
                "ci95_low": percentile(yields, 0.025),
                "ci95_high": percentile(yields, 0.975),
                "trials": float(n_trials),
            }
        )
    return rows


def altitude_points(rows: list[dict[str, float]]) -> list[tuple[float, float]]:
    """Return plotting points for altitude sweep output."""
    return [(row["altitude_km"], row["mean_tokens_per_pass"]) for row in rows]
