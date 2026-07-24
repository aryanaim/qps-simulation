"""Satellite pass and weather scheduling for the token-buffer model.

Provides both a 2-state (clear/cloudy) and a 4-state (clear/thin_cloud/thick_cloud/storm)
Markov weather model. The 4-state model is the default for Q1-journal analysis.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping

# Type alias for weather state names
WeatherState = str

# All valid weather state names
VALID_STATES_2STATE = {"clear", "cloudy"}
VALID_STATES_4STATE = {"clear", "thin_cloud", "thick_cloud", "storm"}

# Default 4-state transition matrix (from config)
DEFAULT_4STATE_TRANSITIONS: dict[str, dict[str, float]] = {
    "clear":      {"clear": 0.70, "thin_cloud": 0.20, "thick_cloud": 0.08, "storm": 0.02},
    "thin_cloud": {"clear": 0.40, "thin_cloud": 0.35, "thick_cloud": 0.20, "storm": 0.05},
    "thick_cloud":{"clear": 0.15, "thin_cloud": 0.30, "thick_cloud": 0.40, "storm": 0.15},
    "storm":      {"clear": 0.10, "thin_cloud": 0.20, "thick_cloud": 0.40, "storm": 0.30},
}

# Throughput multipliers per weather state
DEFAULT_THROUGHPUT: dict[str, float] = {
    "clear": 1.0,
    "thin_cloud": 0.5,
    "thick_cloud": 0.1,
    "storm": 0.0,
}


@dataclass(frozen=True)
class PassWindow:
    """A usable satellite pass for one ground station."""

    pass_id: int
    start_timestamp: float
    duration_sec: float

    @property
    def end_timestamp(self) -> float:
        return self.start_timestamp + self.duration_sec

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def generate_pass_schedule(
    days: float = 7.0,
    passes_per_day: int = 5,
    pass_duration_min: float = 10.0,
    start_offset_hours: float = 0.0,
    jitter_min: float = 0.0,
    seed: int | None = None,
) -> list[PassWindow]:
    """Generate a ground-station pass schedule over a multi-day run.

    Args:
        days: Number of days to schedule.
        passes_per_day: Number of satellite passes per day.
        pass_duration_min: Duration of each pass in minutes.
        start_offset_hours: Offset from t=0 for the first pass.
        jitter_min: Random jitter (uniform, ±jitter_min min) added to each pass start.
        seed: RNG seed for reproducibility.

    Returns:
        List of PassWindow objects, one per pass.
    """
    if passes_per_day <= 0:
        raise ValueError("passes_per_day must be positive")
    rng = random.Random(seed)
    total_passes = int(days * passes_per_day)
    interval_sec = 24.0 * 3600.0 / passes_per_day
    windows: list[PassWindow] = []
    for pass_id in range(total_passes):
        jitter_sec = rng.uniform(-jitter_min * 60.0, jitter_min * 60.0) if jitter_min else 0.0
        start = start_offset_hours * 3600.0 + pass_id * interval_sec + jitter_sec
        windows.append(PassWindow(pass_id=pass_id, start_timestamp=max(0.0, start), duration_sec=pass_duration_min * 60.0))
    return windows


class WeatherMarkovChain:
    """Multi-state Markov weather model (2-state or 4-state).

    The 4-state model tracks clear, thin_cloud, thick_cloud, and storm states,
    each with configurable transition probabilities and throughput multipliers.
    The 2-state model uses the legacy clear/cloudy binary states.

    The throughput multiplier expresses the fraction of nominal token yield
    delivered in each weather state (e.g., 0.5 for thin_cloud, 0.0 for storm).
    """

    def __init__(
        self,
        clear_to_clear_probability: float | None = None,
        cloudy_to_clear_probability: float | None = None,
        initial_clear_probability: float = 0.70,
        seed: int | None = None,
        use_multi_state: bool = False,
        transitions: dict[str, dict[str, float]] | None = None,
        throughput: dict[str, float] | None = None,
    ) -> None:
        self.initial_clear_probability = initial_clear_probability
        self.use_multi_state = use_multi_state
        self.rng = random.Random(seed)

        if use_multi_state:
            self.states = sorted(VALID_STATES_4STATE)
            self.transitions = dict(transitions or DEFAULT_4STATE_TRANSITIONS)
            self.throughput = dict(throughput or DEFAULT_THROUGHPUT)
            # Normalize rows
            for key in self.transitions:
                row = self.transitions[key]
                total = sum(row.values())
                if total > 0:
                    for subkey in row:
                        row[subkey] = row[subkey] / total
        else:
            self.states = ["clear", "cloudy"]
            self.p_cc = clear_to_clear_probability if clear_to_clear_probability is not None else 0.75
            self.p_kc = cloudy_to_clear_probability if cloudy_to_clear_probability is not None else 0.50
            self.transitions = {}
            self.throughput = {"clear": 1.0, "cloudy": 0.0}

    def _next_state_2state(self, current: str) -> str:
        """Transition from current state using the 2-state Markov chain."""
        if current == "clear":
            return "clear" if self.rng.random() < self.p_cc else "cloudy"
        return "clear" if self.rng.random() < self.p_kc else "cloudy"

    def _next_state_4state(self, current: str) -> str:
        """Transition from current state using the 4-state transition matrix."""
        row = self.transitions.get(current, {})
        threshold = self.rng.random()
        cumulative = 0.0
        for next_state, prob in row.items():
            cumulative += prob
            if threshold <= cumulative:
                return next_state
        return list(row.keys())[-1] if row else "clear"

    def generate(
        self,
        passes: list[PassWindow],
        force_cloudy_start: int | None = None,
        force_cloudy_count: int = 0,
    ) -> dict[int, str]:
        """Return weather state for each pass id.

        Args:
            passes: List of pass windows to generate weather for.
            force_cloudy_start: If set, force the first N passes starting at this
                index to be cloudy (for controlled stress tests).
            force_cloudy_count: Number of passes to force cloudy.

        Returns:
            Dict mapping pass_id to weather state string.
        """
        if self.use_multi_state:
            current = self._pick_initial_4state()
        else:
            current = "clear" if self.rng.random() < self.initial_clear_probability else "cloudy"

        states: dict[int, str] = {}
        forced = set()
        if force_cloudy_start is not None and force_cloudy_count > 0:
            forced = set(range(force_cloudy_start, force_cloudy_start + force_cloudy_count))

        for window in passes:
            if window.pass_id in forced:
                states[window.pass_id] = "cloudy" if not self.use_multi_state else "storm"
                current = "cloudy" if not self.use_multi_state else "storm"
                continue
            states[window.pass_id] = current
            if self.use_multi_state:
                current = self._next_state_4state(current)
            else:
                current = self._next_state_2state(current)
        return states

    def _pick_initial_4state(self) -> str:
        """Pick the initial state using the clear probability weighted by state distribution."""
        if self.rng.random() < self.initial_clear_probability:
            return "clear"
        # Pick among non-clear states by their fraction of the non-clear total
        weights = {"thin_cloud": 0.5, "thick_cloud": 0.35, "storm": 0.15}
        threshold = self.rng.random()
        cumulative = 0.0
        for state, weight in weights.items():
            cumulative += weight
            if threshold <= cumulative:
                return state
        return "thick_cloud"

    def throughput_multiplier(self, state: str) -> float:
        """Return the throughput multiplier for a given weather state.

        Args:
            state: Weather state string (e.g., 'clear', 'thin_cloud', 'storm').

        Returns:
            Float multiplier in [0.0, 1.0] indicating fraction of nominal token yield.
        """
        if self.use_multi_state:
            return self.throughput.get(state, 0.0)
        return 1.0 if state == "clear" else 0.0


def schedule_from_config(config: Mapping[str, Any], days: float = 7.0, seed: int | None = None) -> list[PassWindow]:
    """Build a pass schedule from satellite_qkd/config.json token settings.

    Args:
        config: Full configuration dict (from satellite_qkd/config.json).
        days: Number of days to schedule.
        seed: RNG seed.

    Returns:
        List of PassWindow objects.
    """
    token_cfg = config.get("token_buffer", {})
    return generate_pass_schedule(
        days=days,
        passes_per_day=int(token_cfg.get("passes_per_day", 5)),
        pass_duration_min=float(token_cfg.get("pass_duration_min", 10.0)),
        seed=seed,
    )


def weather_model_from_config(
    config: Mapping[str, Any],
    seed: int | None = None,
) -> WeatherMarkovChain:
    """Build a WeatherMarkovChain from configuration.

    Respects the ``token_buffer.weather_model`` setting: ``"multi_state"`` uses
    the 4-state model with configurable transitions; any other value uses the
    legacy 2-state model.

    Args:
        config: Full configuration dict.
        seed: RNG seed.

    Returns:
        Configured WeatherMarkovChain instance.
    """
    token_cfg = config.get("token_buffer", {})
    use_multi = token_cfg.get("weather_model", "multi_state") == "multi_state"
    transitions_raw = token_cfg.get("weather_transitions")
    transitions: dict[str, dict[str, float]] | None = None
    if transitions_raw:
        transitions = {}
        for k, v in transitions_raw.items():
            transitions[str(k)] = {str(kk): float(vv) for kk, vv in v.items()}
    throughput_raw = token_cfg.get("weather_throughput")
    throughput: dict[str, float] | None = None
    if throughput_raw:
        throughput = {str(k): float(v) for k, v in throughput_raw.items()}
    return WeatherMarkovChain(
        clear_to_clear_probability=float(token_cfg.get("clear_to_clear_probability", 0.75)),
        cloudy_to_clear_probability=float(token_cfg.get("cloudy_to_clear_probability", 0.50)),
        initial_clear_probability=float(token_cfg.get("clear_sky_probability", 0.70)),
        seed=seed,
        use_multi_state=use_multi,
        transitions=transitions,
        throughput=throughput,
    )

