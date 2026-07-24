"""Atmospheric turbulence sweep for satellite QKD."""

from __future__ import annotations

import random
import statistics
from typing import Any, Mapping

from common.statistics import bootstrap_ci
from .orbital_dynamics import load_config
from .pass_simulator import simulate_pass


def logspace(start_exp: float, stop_exp: float, points: int) -> list[float]:
    """Return powers of ten between two exponents."""
    if points <= 1:
        return [10.0**start_exp]
    step = (stop_exp - start_exp) / (points - 1)
    return [10.0 ** (start_exp + i * step) for i in range(points)]


def run_turbulence_sweep(
    cn2_values: list[float] | None = None,
    trials: int | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int = 84,
    pulses_per_interval: int | None = None,
) -> list[dict[str, float]]:
    """Sweep Cn2 and return QBER/token statistics.

    Uses the NetSquid BB84 backend for all simulation trials.
    """
    cfg = dict(config or load_config())
    sweep_cfg = cfg.get("sweep", {})
    rng = random.Random(seed)
    n_trials = trials if trials is not None else int(sweep_cfg.get("turbulence_trials", 100))
    if cn2_values is None:
        values = logspace(
            float(sweep_cfg.get("cn2_log_start", -17.0)),
            float(sweep_cfg.get("cn2_log_stop", -13.0)),
            int(sweep_cfg.get("cn2_points", 10)),
        )
    else:
        values = cn2_values
    rows: list[dict[str, float]] = []
    for idx, cn2 in enumerate(values):
        qbers: list[float] = []
        tokens: list[float] = []
        abort_fraction: list[float] = []
        for trial in range(n_trials):
            max_elevation = rng.uniform(65.0, 75.0)
            result = simulate_pass(
                {
                    "pass_id": trial,
                    "altitude_km": 500.0,
                    "max_elevation_deg": max_elevation,
                    "cn2": cn2,
                },
                config=cfg,
                seed=seed + trial + idx * 1000,
                pulses_per_interval=pulses_per_interval,
            )
            qbers.append(float(result["summary"]["avg_QBER"]))
            tokens.append(float(result["summary"]["tokens_issued"]))
            above = float(result["summary"]["pass_duration_above_threshold_sec"])
            duration = float(result["summary"]["duration_sec"])
            abort_fraction.append(1.0 - above / max(1.0, duration))
        qber_ci = bootstrap_ci(qbers, statistic="mean", n_resamples=10000, seed=seed + idx)
        token_ci = bootstrap_ci(tokens, statistic="mean", n_resamples=10000, seed=seed + idx + 1000)
        rows.append(
            {
                "Cn2": cn2,
                "mean_QBER": qber_ci["stat"],
                "qber_ci95_low": qber_ci["low"],
                "qber_ci95_high": qber_ci["high"],
                "mean_tokens_per_pass": token_ci["stat"],
                "token_ci95_low": token_ci["low"],
                "token_ci95_high": token_ci["high"],
                "abort_fraction": statistics.fmean(abort_fraction),
                "trials": float(n_trials),
            }
        )
    return rows
