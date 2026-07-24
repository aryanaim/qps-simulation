"""Graph generation and depletion analysis for WS2.

Provides G3 (buffer level), G4 (cloudy-pass resilience), G9 (yearly survival),
and G10 (staleness sensitivity) graphs for the token buffer analysis.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from common.io import write_csv
from common.plotting import PALETTE, write_line_chart
from common.random import percentile
from satellite_qkd.orbital_dynamics import load_config
from .simulation import cloudy_passes_survivable, simulate_buffer


def graph_buffer_level(
    output_path: str | Path,
    pass_contract_A: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
    seed: int = 123,
) -> dict[str, Any]:
    """Produce Graph G3 and return simulations for downstream use.

    Runs replicate simulations at 10, 50, and 100 tokens/hour to show buffer
    level trajectories over 7 days.

    Args:
        output_path: Path for the output graph image.
        pass_contract_A: Contract A dict from WS1 (tokens per clear pass).
        config: Full configuration dict.
        seed: RNG seed.

    Returns:
        Dict mapping rate string (str(rate)) to simulation result dicts.
    """
    cfg = dict(config or load_config())
    capacity = int(cfg.get("token_buffer", {}).get("buffer_capacity", 10000))
    n_replicates = int(cfg.get("sweep", {}).get("buffer_replicates", 3))
    rates = [10.0, 50.0, 100.0]
    series = []
    simulations: dict[str, Any] = {}
    for idx, rate in enumerate(rates):
        rep_timeseries: list[list[dict]] = []
        primary_sim: dict[str, Any] | None = None
        for rep in range(n_replicates):
            sim = simulate_buffer(
                days=7.0,
                consumption_rate_per_hour=rate,
                pass_contract_A=pass_contract_A,
                config=cfg,
                seed=seed + int(rate) + rep * 10000,
            )
            rep_timeseries.append(sim["timeseries"])
            if rep == 0:
                primary_sim = sim
        assert primary_sim is not None
        simulations[str(int(rate))] = primary_sim
        n_steps = min(len(ts) for ts in rep_timeseries)
        mean_pts = [
            (rep_timeseries[0][t]["timestamp_hours"],
             sum(ts[t]["buffer_level"] for ts in rep_timeseries) / n_replicates)
            for t in range(n_steps)
        ]
        ci_low = [
            (rep_timeseries[0][t]["timestamp_hours"],
             percentile([ts[t]["buffer_level"] for ts in rep_timeseries], 0.025))
            for t in range(n_steps)
        ]
        ci_high = [
            (rep_timeseries[0][t]["timestamp_hours"],
             percentile([ts[t]["buffer_level"] for ts in rep_timeseries], 0.975))
            for t in range(n_steps)
        ]
        series.append(
            {
                "name": f"{int(rate)} tokens/hour",
                "color": PALETTE[idx],
                "points": mean_pts,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    # Compute dynamic y-axis limit
    max_buffer_level = max(max(pt[1] for pt in s["points"]) for s in series)
    y_max = max_buffer_level * 1.3
    if y_max > capacity:
        y_max = capacity
    y_min = 0
    # Add threshold line at 10% capacity
    degraded_threshold = capacity * 0.10
    thresholds = [{"value": degraded_threshold, "label": "10% capacity", "color": "#666666"}]
    write_line_chart(
        output_path,
        "Buffer Level Over 7 Days",
        "Time t (hours)",
        "Buffer level B(t)",
        series,
        y_min=y_min,
        y_max=y_max,
        thresholds=thresholds,
        subtitle=f"Mean \pm 95% CI over {n_replicates} replicates; B(t+1) = min(B<sub>max</sub>, B(t) + I(t) - C(t) - E(t))",
        column="double", markers=False,
    )
    return simulations


def resilience_rows(
    pass_contract_A: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Return G4 resilience rows over the configured consumption range.

    Args:
        pass_contract_A: Contract A dict.
        config: Full configuration dict.

    Returns:
        List of dicts with consumption_rate_per_hour and cloudy_passes_survivable.
    """
    cfg = dict(config or load_config())
    token_cfg = cfg.get("token_buffer", {})
    tokens = max(1, int((pass_contract_A or {}).get("tokens_issued", 650)))
    rows: list[dict[str, float]] = []
    for rate in [10, 25, 50, 75, 100, 125, 150]:
        rows.append(
            {
                "consumption_rate_per_hour": float(rate),
                "cloudy_passes_survivable": float(
                    cloudy_passes_survivable(
                        consumption_rate_per_hour=float(rate),
                        tokens_per_clear_pass=tokens,
                        passes_per_day=int(token_cfg.get("passes_per_day", 5)),
                        capacity=int(token_cfg.get("buffer_capacity", 10000)),
                    )
                ),
            }
        )
    return rows


def graph_resilience(
    output_path: str | Path,
    csv_path: str | Path,
    pass_contract_A: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Produce Graph G4 and write its CSV.

    Args:
        output_path: Path for the output graph image.
        csv_path: Path for the output CSV data.
        pass_contract_A: Contract A dict.
        config: Full configuration dict.

    Returns:
        List of resilience row dicts.
    """
    rows = resilience_rows(pass_contract_A, config)
    write_csv(csv_path, rows)
    write_line_chart(
        output_path,
        "Cloudy-Pass Resilience",
        "Consumption rate r<sub>c</sub> (tokens/hour)",
        "Survivable cloudy passes M",
        [{"name": "M cloudy passes", "points": [(r["consumption_rate_per_hour"], r["cloudy_passes_survivable"]) for r in rows]}],
        y_min=0,
        subtitle="M = floor(min(B<sub>max</sub>, 2N<sub>tok</sub>) / (r<sub>c</sub> * 24/P<sub>day</sub>))",
        column="double", markers=True,
    )
    return rows


def graph_yearly_survival(
    output_path: str | Path,
    consumption_rate_per_hour: float = 50.0,
    pass_contract_A: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int = 12345,
) -> list[dict[str, float]]:
    """Produce Graph G9: Kaplan–Meier-style yearly survival curve.

    Runs batched 7-day simulations over a 365-day horizon, recording the longest
    consecutive degraded period per simulation. Outputs a survival curve showing
    P(survive) vs. consecutive degraded hours.

    Args:
        output_path: Path for the output graph (PNG).
        consumption_rate_per_hour: Token consumption rate.
        pass_contract_A: Contract A dict.
        config: Full configuration dict.
        seed: RNG seed.

    Returns:
        List of (degraded_hours, survival_probability) dicts.
    """
    cfg = dict(config or load_config())
    n_batches = 156  # 156 weeks × 7 days ≈ 1092 days (3 years of data)
    days_per_batch = 7

    results: list[float] = []
    for batch in range(n_batches):
        sim = simulate_buffer(
            days=days_per_batch,
            consumption_rate_per_hour=consumption_rate_per_hour,
            pass_contract_A=pass_contract_A,
            config=cfg,
            seed=seed + batch,
        )
        # Find longest consecutive degraded streak
        max_streak = 0.0
        current_streak = 0.0
        for row in sim["timeseries"]:
            if row["is_degraded"]:
                current_streak += 1.0 / 60.0  # hours
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0.0
        results.append(max_streak)

    # Build Kaplan–Meier-style survival curve
    max_hours = max(results) if results else 24.0
    step = max(1.0, max_hours / 50.0)
    n_total = len(results)
    points: list[tuple[float, float]] = []
    x = 0.0
    while x <= max_hours + step:
        survived = sum(1 for r in results if r < x or abs(r - x) < 1e-9)
        prob = survived / max(1, n_total)
        points.append((x, prob))
        x += step

    write_line_chart(
        output_path,
        "Yearly Buffer Survival Probability",
        "Consecutive degraded time t (hours)",
        "P(survive)",
        [{"name": "Survival probability", "points": points, "color": PALETTE[2]}],
        y_min=0,
        y_max=1.05,
        subtitle=f"Kaplan–Meier estimate over {n_batches} weekly simulations; r<sub>c</sub> = {int(consumption_rate_per_hour)} tokens/hr",
        column="double", markers=True,
    )
    return [{"degraded_hours": x, "survival_probability": y} for x, y in points]


def graph_staleness_sensitivity(
    output_path: str | Path,
    pass_contract_A: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Produce Graph G10: staleness sensitivity sweep.

    Sweeps staleness_max_hours over [6, 12, 24, 48] and plots survivable cloudy
    passes vs consumption rate with separate curves per staleness setting.

    Args:
        output_path: Path for the output graph (PNG).
        pass_contract_A: Contract A dict.
        config: Full configuration dict.

    Returns:
        List of row dicts with staleness_max_hours, consumption rate, and cloudy passes.
    """
    cfg = dict(config or load_config())
    token_cfg = cfg.get("token_buffer", {})
    tokens = max(1, int((pass_contract_A or {}).get("tokens_issued", 650)))
    staleness_values = [6, 12, 24, 48]
    rates = [10, 25, 50, 75, 100, 125, 150]
    series: list[dict[str, Any]] = []
    rows: list[dict[str, float]] = []

    for idx, staleness in enumerate(staleness_values):
        pts: list[tuple[float, float]] = []
        for rate in rates:
            # Modified cloudy_passes_survivable that uses the configured capacity
            capacity = int(token_cfg.get("buffer_capacity", 10000))
            passes_per_day = int(token_cfg.get("passes_per_day", 5))

            # The staleness limit effectively caps how long tokens last
            # Simpler: use the analytic model with staleness-adjusted capacity
            interval_hours = 24.0 / max(1, passes_per_day)
            stored = min(float(capacity), float(tokens) * 2.0)
            if rate <= 0:
                surv = 999
            else:
                # Effective capacity limited by staleness window
                tokens_per_interval = rate * staleness
                effective_capacity = min(capacity, int(tokens_per_interval))
                stored = min(float(effective_capacity), float(tokens) * 2.0)
                surv = max(0, int(stored // (rate * interval_hours)))
            pts.append((float(rate), float(surv)))
            rows.append({
                "staleness_max_hours": float(staleness),
                "consumption_rate_per_hour": float(rate),
                "cloudy_passes_survivable": float(surv),
            })
        series.append({
            "name": f"staleness = {staleness}h",
            "color": PALETTE[idx],
            "points": pts,
        })

    write_line_chart(
        output_path,
        "Staleness Sensitivity: Cloudy-Pass Resilience vs Token Expiry",
        "Consumption rate r<sub>c</sub> (tokens/hour)",
        "Survivable cloudy passes M",
        series,
        y_min=0,
        subtitle="Each curve shows survivable cloudy passes for a different staleness window; tokens per clear pass = {0}".format(tokens),
        column="double", markers=True,
    )
    return rows
