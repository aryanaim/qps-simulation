"""Discrete-event token-buffer simulation over a seven-day horizon.

Supports multi-state weather model, configurable consumption policies including
cover traffic, and produces Contract B data for adversary analysis.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from common.random import noisy_count, poisson_count
from satellite_qkd.orbital_dynamics import load_config
from .buffer import Token, TokenBuffer
from .pass_schedule import PassWindow, schedule_from_config, weather_model_from_config, WeatherMarkovChain


def _tokens_for_pass(pass_id: int, count: int, timestamp: float) -> list[Token]:
    """Create a list of Token objects for a given satellite pass.

    Args:
        pass_id: Satellite pass identifier.
        count: Number of tokens to create.
        timestamp: Creation timestamp.

    Returns:
        List of Token objects.
    """
    return [Token(token_id=f"P{pass_id:04d}-T{idx:06d}", created_timestamp=timestamp, pass_id=pass_id) for idx in range(count)]


def _nominal_tokens_from_contract(contract_A: Mapping[str, Any] | None) -> int:
    """Extract nominal tokens per clear pass from Contract A.

    Args:
        contract_A: Contract A dict from WS1 pass simulation.

    Returns:
        Integer number of tokens per clear pass (default 650).
    """
    if not contract_A:
        return 650
    return max(1, int(contract_A.get("tokens_issued", 650)))


def _generate_cover_events(
    real_events: list[dict[str, Any]],
    cover_ratio: float,
    inter_pass_interval: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate cover (dummy) presentation events to obscure burst timing.

    Dummy events have timestamps uniformly distributed across the inter-pass
    interval and ``pass_source_id = -(i+1)`` (unique negative per dummy) to
    prevent spurious cross-pass matches between consecutive dummy events.

    Args:
        real_events: Real presentation events.
        cover_ratio: Ratio of dummy to real events.
        inter_pass_interval: Time (seconds) between pass windows.
        rng: RNG instance.

    Returns:
        List of dummy presentation event dicts.
    """
    if cover_ratio <= 0.0 or not real_events:
        return []
    n_dummies = max(1, int(len(real_events) * cover_ratio))
    real_timestamps = [float(e["timestamp"]) for e in real_events]
    t_min = min(real_timestamps)
    t_max = max(real_timestamps)
    dummies: list[dict[str, Any]] = []
    for i in range(n_dummies):
        ts = rng.uniform(t_min, t_max)
        dummies.append({
            "timestamp": ts,
            "timestamp_hours": ts / 3600.0,
            "token_consumed_id": f"dummy-{i:06d}",
            "pass_source_id": -(i + 1),
            "token_age_hours": 0.0,
            "buffer_level_after": 0,
            "is_dummy": True,
        })
    return dummies


def simulate_buffer(
    days: float = 7.0,
    consumption_rate_per_hour: float = 50.0,
    pass_contract_A: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    consumption_policy: str = "fifo",
    force_cloudy_start: int | None = None,
    force_cloudy_count: int = 0,
    seed: int = 123,
) -> dict[str, Any]:
    """Run a minute-resolution token-buffer simulation.

    Uses the multi-state Markov weather model by default (configurable via
    ``token_buffer.weather_model`` in config.json). The 2-state model (legacy)
    is used when ``weather_model`` is not ``"multi_state"``.

    The ``consumption_policy`` parameter supports:
      - ``"fifo"``: consume oldest tokens first.
      - ``"random"``: consume randomly-selected tokens.
      - ``"random_cover"``: random consumption with cover traffic insertion.

    Args:
        days: Simulation duration in days.
        consumption_rate_per_hour: Mean token consumption rate.
        pass_contract_A: Contract A dict from WS1 (tokens per clear pass).
        config: Full configuration dict.
        consumption_policy: Token consumption order policy.
        force_cloudy_start: Force consecutive cloudy/storm passes starting at this index.
        force_cloudy_count: Number of passes to force cloudy.
        seed: RNG seed.

    Returns:
        Dict with timeseries, contract_B, presentation_events, passes, weather,
        staleness_hours, and summary.
    """
    cfg = dict(config or load_config())
    token_cfg = cfg.get("token_buffer", {})
    rng = random.Random(seed)
    buffer = TokenBuffer(
        capacity=int(token_cfg.get("buffer_capacity", 10000)),
        staleness_max_hours=float(token_cfg.get("staleness_max_hours", 24.0)),
        consumption_policy="random" if consumption_policy == "random_cover" else consumption_policy,
        rng=rng,
    )
    passes = schedule_from_config(cfg, days=days, seed=seed)
    weather = weather_model_from_config(cfg, seed=seed + 1)
    weather_states = weather.generate(passes, force_cloudy_start=force_cloudy_start, force_cloudy_count=force_cloudy_count)
    pass_by_minute = {int(window.start_timestamp // 60): window for window in passes}
    nominal_tokens = _nominal_tokens_from_contract(pass_contract_A)
    total_minutes = int(days * 24.0 * 60.0)
    timeseries: list[dict[str, Any]] = []
    contract_B: list[dict[str, Any]] = []
    presentation_events: list[dict[str, Any]] = []
    staleness_hours: list[float] = []
    degraded_minutes = 0

    for minute in range(total_minutes + 1):
        timestamp = minute * 60.0
        expired = buffer.expire_stale(timestamp)
        tokens_added = 0
        pass_id = ""
        weather_state = ""

        window: PassWindow | None = pass_by_minute.get(minute)
        if window:
            pass_id = window.pass_id
            weather_state = weather_states.get(window.pass_id, "clear")
            mult = weather.throughput_multiplier(weather_state)
            if mult > 0:
                delivered = noisy_count(max(1, int(nominal_tokens * mult)), rng, relative_sigma=0.18)
                tokens_added = buffer.add_tokens(_tokens_for_pass(window.pass_id, delivered, timestamp), timestamp)

        tokens_consumed = 0
        consumed_ids: list[str] = []
        source_pass_ids: list[str] = []
        consumption_count = poisson_count(consumption_rate_per_hour / 60.0, rng)
        for _ in range(consumption_count):
            token = buffer.consume_token(timestamp)
            if token is None:
                break
            tokens_consumed += 1
            consumed_ids.append(token.token_id)
            source_pass_ids.append(str(token.pass_id))
            age_hours = (timestamp - token.created_timestamp) / 3600.0
            staleness_hours.append(age_hours)
            event = {
                "timestamp": timestamp,
                "timestamp_hours": timestamp / 3600.0,
                "token_consumed_id": token.token_id,
                "pass_source_id": token.pass_id,
                "token_age_hours": age_hours,
                "buffer_level_after": buffer.level(),
                "is_dummy": False,
            }
            contract_B.append(event)
            presentation_events.append(event)

        if buffer.is_degraded():
            degraded_minutes += 1

        timeseries.append(
            {
                "timestamp": timestamp,
                "timestamp_hours": timestamp / 3600.0,
                "buffer_level": buffer.level(),
                "tokens_consumed": tokens_consumed,
                "tokens_added": tokens_added,
                "tokens_expired": expired,
                "pass_id": pass_id,
                "weather": weather_state,
                "is_degraded": buffer.is_degraded(),
                "token_consumed_id": ";".join(consumed_ids),
                "pass_source_id": ";".join(source_pass_ids),
            }
        )

    # Add cover traffic for random_cover policy
    if consumption_policy == "random_cover":
        interval_hours = 24.0 / max(1, int(token_cfg.get("passes_per_day", 5)))
        interval_sec = interval_hours * 3600.0
        cover_ratio = float(token_cfg.get("cover_traffic_ratio", 0.3))
        cover_events = _generate_cover_events(presentation_events, cover_ratio, interval_sec, rng)
        # Merge cover events into presentation events (sorted by timestamp)
        all_events = sorted(presentation_events + cover_events, key=lambda e: float(e["timestamp"]))
        presentation_events = all_events

    return {
        "timeseries": timeseries,
        "contract_B": contract_B,
        "presentation_events": presentation_events,
        "passes": [window.to_dict() for window in passes],
        "weather": weather_states,
        "staleness_hours": staleness_hours,
        "summary": {
            "days": days,
            "consumption_rate_per_hour": consumption_rate_per_hour,
            "nominal_tokens_per_clear_pass": nominal_tokens,
            "final_buffer_level": buffer.level(),
            "degraded_minutes": degraded_minutes,
            "degraded_hours": degraded_minutes / 60.0,
            "consumption_policy": consumption_policy,
        },
    }


def cloudy_passes_survivable(
    consumption_rate_per_hour: float,
    tokens_per_clear_pass: int,
    passes_per_day: int = 5,
    initial_clear_passes: int = 2,
    capacity: int = 10000,
) -> int:
    """Analytic resilience estimate for Graph S4.

    Computes the number of consecutive cloudy/storm passes the buffer can
    survive given the consumption rate and tokens per clear pass.

    Args:
        consumption_rate_per_hour: Token consumption rate.
        tokens_per_clear_pass: Tokens generated per clear pass.
        passes_per_day: Number of satellite passes per day.
        initial_clear_passes: Number of clear passes before cloudy streak.
        capacity: Maximum buffer capacity.

    Returns:
        Integer number of survivable cloudy passes.
    """
    interval_hours = 24.0 / max(1, passes_per_day)
    stored = min(float(capacity), float(tokens_per_clear_pass) * max(1, initial_clear_passes))
    if consumption_rate_per_hour <= 0:
        return 999
    return max(0, int(stored // (consumption_rate_per_hour * interval_hours)))
