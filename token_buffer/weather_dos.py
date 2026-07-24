"""Weather-based denial-of-service analysis for the token buffer.

Models extended storm events (multi-day continuous cloud cover) that prevent
optical QKD, building on the baseline WeatherMarkovChain. The key output is
the probability that the buffer enters DEGRADED mode as a function of the
weather severity index (a composite of storm arrival rate and mean duration).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Mapping

from common.io import write_csv
from common.plotting import write_line_chart
from satellite_qkd.orbital_dynamics import load_config
from .buffer import Token, TokenBuffer
from .pass_schedule import PassWindow, schedule_from_config, weather_model_from_config


def _tokens_for_pass(pass_id: int, count: int, timestamp: float) -> list[Token]:
    return [Token(token_id=f"P{pass_id:04d}-T{idx:06d}", created_timestamp=timestamp, pass_id=pass_id) for idx in range(count)]


def _nominal_tokens_from_contract(contract_A: Mapping[str, Any] | None) -> int:
    if not contract_A:
        return 650
    return max(1, int(contract_A.get("tokens_issued", 650)))


def _storm_duration(rng: random.Random, mean_hours: float) -> float:
    """Sample storm duration in hours from a geometric-style distribution."""
    if mean_hours <= 0:
        return 0.0
    return math.ceil(rng.expovariate(1.0 / mean_hours))


def simulate_with_storms(
    days: float = 7.0,
    consumption_rate_per_hour: float = 50.0,
    pass_contract_A: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    storm_events_per_week: float = 1.0,
    mean_storm_hours: float = 12.0,
    force_cloudy_start: int | None = None,
    force_cloudy_count: int = 0,
    seed: int = 123,
) -> dict[str, Any]:
    """Run a minute-resolution buffer simulation with storm-event injection.

    Storm events are extended periods of continuous cloud cover that override
    the normal Markov weather. Between storms the baseline weather model applies.
    The storm arrival process is Poisson with rate ``storm_events_per_week``.
    """
    cfg = dict(config or load_config())
    token_cfg = cfg.get("token_buffer", {})
    rng = random.Random(seed)
    buffer = TokenBuffer(
        capacity=int(token_cfg.get("buffer_capacity", 10000)),
        staleness_max_hours=float(token_cfg.get("staleness_max_hours", 24.0)),
        consumption_policy="fifo",
        rng=rng,
    )
    passes = schedule_from_config(cfg, days=days, seed=seed)
    weather_model = weather_model_from_config(cfg, seed=seed + 1)
    weather_states = weather_model.generate(passes, force_cloudy_start=force_cloudy_start, force_cloudy_count=force_cloudy_count)
    pass_by_minute = {int(window.start_timestamp // 60): window for window in passes}
    nominal_tokens = _nominal_tokens_from_contract(pass_contract_A)
    total_minutes = int(days * 24.0 * 60.0)

    # Generate storm schedule: Poisson arrival of storms, geometric duration
    storm_rng = random.Random(seed + 1000)
    storm_active_until: float = 0.0
    mean_storm_interval_hours = max(0.001, (days * 7.0) / max(1.0, storm_events_per_week * days))
    next_storm_start_hours = storm_rng.expovariate(1.0 / mean_storm_interval_hours)

    storm_periods: list[tuple[float, float]] = []  # (start_hour, end_hour) for logging
    degraded_minutes = 0

    for minute in range(total_minutes + 1):
        timestamp = minute * 60.0
        hours = timestamp / 3600.0
        buffer.expire_stale(timestamp)

        # Check for storm start
        if storm_events_per_week > 0 and hours >= next_storm_start_hours and hours >= storm_active_until:
            duration_hours = _storm_duration(storm_rng, mean_storm_hours)
            storm_active_until = hours + duration_hours
            storm_periods.append((hours, storm_active_until))
            next_storm_start_hours = hours + storm_rng.expovariate(1.0 / mean_storm_interval_hours)

        tokens_added = 0
        in_storm = storm_events_per_week > 0 and hours < storm_active_until
        window: PassWindow | None = pass_by_minute.get(minute)
        if window and not in_storm:
            weather_state = weather_states.get(window.pass_id, "clear")
            mult = weather_model.throughput_multiplier(weather_state)
            if mult > 0:
                delivered = max(1, int(nominal_tokens * mult))
                tokens_added = buffer.add_tokens(_tokens_for_pass(window.pass_id, delivered, timestamp), timestamp)

        # Poisson consumption at the configured rate
        consumption_count = 0
        if consumption_rate_per_hour > 0:
            rate_per_minute = consumption_rate_per_hour / 60.0
            consumption_count = sum(1 for _ in range(3) if rng.random() < rate_per_minute / 3.0)
        for _ in range(consumption_count):
            if buffer.consume_token(timestamp) is None:
                break

        if buffer.is_degraded():
            degraded_minutes += 1

    return {
        "summary": {
            "days": days,
            "consumption_rate_per_hour": consumption_rate_per_hour,
            "storm_events_per_week": storm_events_per_week,
            "mean_storm_hours": mean_storm_hours,
            "degraded_minutes": degraded_minutes,
            "degraded_fraction": degraded_minutes / max(1, total_minutes),
            "n_storms": len(storm_periods),
        },
        "storm_periods": storm_periods,
    }


def run_weather_dos_sweep(
    consumption_rate_per_hour: float = 50.0,
    pass_contract_A: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, float]]:
    """Sweep over storm arrival rates and return degraded fraction for each.

    The Weather Severity Index (WSI) on the x-axis is the mean number of
    storm events per week. Each point is the average degraded fraction over
    ``n_replicates`` independent simulation runs.
    """
    cfg = dict(config or load_config())
    if seeds is None:
        seeds = [100, 200, 300, 400, 500]
    n_replicates = len(seeds)
    storm_rates = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    rows: list[dict[str, float]] = []
    for rate in storm_rates:
        fractions: list[float] = []
        for rep_idx, seed in enumerate(seeds):
            sim = simulate_with_storms(
                days=7.0,
                consumption_rate_per_hour=consumption_rate_per_hour,
                pass_contract_A=pass_contract_A,
                config=cfg,
                storm_events_per_week=rate,
                mean_storm_hours=12.0,
                seed=seed + rep_idx * 1000,
            )
            fractions.append(sim["summary"]["degraded_fraction"])
        rows.append(
            {
                "storm_events_per_week": rate,
                "mean_degraded_fraction": sum(fractions) / n_replicates,
                "min_degraded_fraction": min(fractions),
                "max_degraded_fraction": max(fractions),
                "n_replicates": n_replicates,
            }
        )
    return rows


def graph_weather_dos(
    output_path: str | Path,
    csv_path: str | Path,
    pass_contract_A: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Produce Graph G8: Token Availability vs Weather Severity Index."""
    rows = run_weather_dos_sweep(
        consumption_rate_per_hour=50.0,
        pass_contract_A=pass_contract_A,
        config=config,
    )
    write_csv(csv_path, rows)
    write_line_chart(
        output_path,
        "Token Availability vs Weather Severity",
        "Storm events per week",
        "Fraction of time in DEGRADED mode",
        [
            {
                "name": "mean degraded fraction",
                "points": [(r["storm_events_per_week"], r["mean_degraded_fraction"]) for r in rows],
                "ci_low": [(r["storm_events_per_week"], r["min_degraded_fraction"]) for r in rows],
                "ci_high": [(r["storm_events_per_week"], r["max_degraded_fraction"]) for r in rows],
            }
        ],
        y_min=0,
        subtitle=f"Mean over {rows[0]['n_replicates']:.0f} replicates; 12-hour mean storm duration; {rows[0]['n_replicates']:.0f} replicates each point",
        column="double", markers=True,
    )
    return rows
