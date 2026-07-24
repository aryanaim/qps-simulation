"""Satellite visibility scheduler for intermittent CRDT anti-entropy."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VisibilityWindow:
    """A satellite visibility window."""

    start_timestamp: float
    duration_sec: float

    @property
    def end_timestamp(self) -> float:
        return self.start_timestamp + self.duration_sec

    def contains(self, timestamp: float) -> bool:
        return self.start_timestamp <= timestamp <= self.end_timestamp

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class SatelliteVisibilityScheduler:
    """Queue satellite messages until the next visibility window."""

    def __init__(
        self,
        windows: list[VisibilityWindow],
        satellite_latency_sec: float = 0.6,
    ) -> None:
        self.windows = sorted(windows, key=lambda item: item.start_timestamp)
        self.satellite_latency_sec = satellite_latency_sec

    @classmethod
    def periodic(
        cls,
        orbital_period_min: float = 94.6,
        pass_duration_min: float = 10.0,
        first_window_start_min: float = 0.0,
        horizon_min: float = 24.0 * 60.0,
        satellite_latency_sec: float = 0.6,
    ) -> "SatelliteVisibilityScheduler":
        """Create periodic visibility windows."""
        windows: list[VisibilityWindow] = []
        start = first_window_start_min * 60.0
        while start <= horizon_min * 60.0:
            windows.append(VisibilityWindow(start_timestamp=start, duration_sec=pass_duration_min * 60.0))
            start += orbital_period_min * 60.0
        return cls(windows, satellite_latency_sec=satellite_latency_sec)

    def is_visible(self, timestamp: float) -> bool:
        """Return True when timestamp falls inside a visibility window."""
        return any(window.contains(timestamp) for window in self.windows)

    def next_window_start(self, timestamp: float) -> float:
        """Return the next visibility start at or after timestamp."""
        for window in self.windows:
            if window.contains(timestamp):
                return timestamp
            if window.start_timestamp >= timestamp:
                return window.start_timestamp
        if not self.windows:
            raise ValueError("scheduler has no visibility windows")
        period = self.windows[-1].start_timestamp - self.windows[-2].start_timestamp if len(self.windows) > 1 else 94.6 * 60.0
        last = self.windows[-1].start_timestamp
        while last < timestamp:
            last += period
        return last

    def delivery_time(self, enqueue_timestamp: float) -> float:
        """Return when a satellite-queued message arrives."""
        if self.is_visible(enqueue_timestamp):
            return enqueue_timestamp + self.satellite_latency_sec
        return self.next_window_start(enqueue_timestamp) + self.satellite_latency_sec

