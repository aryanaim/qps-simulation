"""Free-space channel model for satellite-to-ground BB84.

This module exposes the physics needed by both backends:

* the research backend builds a real NetSquid ``QuantumChannel`` with
  time-varying satellite loss/noise models;
* the surrogate backend uses the same ``ChannelState`` conversions for fast
  smoke tests and large exploratory runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

try:  # pragma: no cover - exercised only in NetSquid environments.
    from netsquid.components import QuantumChannel as _NetSquidQuantumChannel
    from netsquid.components.models.qerrormodels import QuantumErrorModel as _NetSquidQuantumErrorModel

    NETSQUID_AVAILABLE = True
except Exception:  # pragma: no cover - expected in ordinary Python installs.
    _NetSquidQuantumChannel = object
    _NetSquidQuantumErrorModel = object
    NETSQUID_AVAILABLE = False


@dataclass(frozen=True)
class ChannelState:
    """Channel parameters for one time interval."""

    timestamp: float
    loss_dB: float
    error_prob: float
    transmission_probability: float
    depolarizing_probability: float


def transmission_probability_from_loss(loss_dB: float) -> float:
    """Convert positive attenuation in dB to photon transmission probability."""
    return 10.0 ** (-float(loss_dB) / 10.0)


def depolarizing_probability_from_qber(qber: float) -> float:
    """Convert BB84 QBER to a depolarizing noise probability."""
    return min(1.0, max(0.0, (4.0 / 3.0) * float(qber)))


def state_from_row(row: dict, fallback_timestamp: float = 0.0) -> ChannelState:
    """Convert an orbital row into a channel state."""
    loss = float(row.get("loss_dB", 0.0))
    error = float(row.get("expected_QBER", 0.0))
    return ChannelState(
        timestamp=float(row.get("timestamp", fallback_timestamp)),
        loss_dB=loss,
        error_prob=error,
        transmission_probability=transmission_probability_from_loss(loss),
        depolarizing_probability=depolarizing_probability_from_qber(error),
    )


class TimeVaryingSatelliteLossModel(_NetSquidQuantumErrorModel):
    """NetSquid error model that probabilistically drops photons by time slot."""

    def __init__(self, timeseries: Iterable[dict], update_interval_sec: float = 1.0, seed: int | None = None) -> None:
        if NETSQUID_AVAILABLE:
            super().__init__()
        self.timeseries = list(timeseries)
        self.update_interval_sec = max(float(update_interval_sec), 1e-9)
        self.rng = random.Random(seed)

    def _state(self, delta_time: float = 0.0, **kwargs: object) -> ChannelState:
        timestamp = float(kwargs.get("timestamp", delta_time) or 0.0)
        if not self.timeseries:
            return state_from_row({"timestamp": timestamp})
        index = int(max(0, round(timestamp / self.update_interval_sec)))
        index = min(index, len(self.timeseries) - 1)
        return state_from_row(self.timeseries[index], timestamp)

    def error_operation(self, qubits: list, delta_time: float = 0.0, **kwargs: object) -> None:
        state = self._state(delta_time, **kwargs)
        for index, qubit in enumerate(qubits):
            if qubit is not None and self.rng.random() > state.transmission_probability:
                qubits[index] = None


class TimeVaryingSatelliteNoiseModel(_NetSquidQuantumErrorModel):
    """NetSquid depolarizing noise model driven by the orbital QBER estimate."""

    def __init__(self, timeseries: Iterable[dict], update_interval_sec: float = 1.0, seed: int | None = None) -> None:
        if NETSQUID_AVAILABLE:
            super().__init__()
        self.timeseries = list(timeseries)
        self.update_interval_sec = max(float(update_interval_sec), 1e-9)
        self.rng = random.Random(seed)

    def _state(self, delta_time: float = 0.0, **kwargs: object) -> ChannelState:
        timestamp = float(kwargs.get("timestamp", delta_time) or 0.0)
        if not self.timeseries:
            return state_from_row({"timestamp": timestamp})
        index = int(max(0, round(timestamp / self.update_interval_sec)))
        index = min(index, len(self.timeseries) - 1)
        return state_from_row(self.timeseries[index], timestamp)

    def error_operation(self, qubits: list, delta_time: float = 0.0, **kwargs: object) -> None:
        if not NETSQUID_AVAILABLE:
            return
        import netsquid as ns

        qapi = ns.qubits.qubitapi
        state = self._state(delta_time, **kwargs)
        for qubit in qubits:
            if qubit is None or self.rng.random() >= state.depolarizing_probability:
                continue
            qapi.operate(qubit, self.rng.choice([ns.X, ns.Y, ns.Z]))


class FreeSpaceChannel(_NetSquidQuantumChannel):
    """Time-varying free-space channel with dB loss and QBER-like errors."""

    def __init__(
        self,
        name: str = "free_space_satellite",
        loss_dB: float = 0.0,
        error_prob: float = 0.0,
        timeseries: Iterable[dict] | None = None,
        update_interval_sec: float = 1.0,
    ) -> None:
        self._free_space_name = name
        self.loss_dB = float(loss_dB)
        self.error_prob = float(error_prob)
        self.update_interval_sec = float(update_interval_sec)
        self.timeseries = list(timeseries or [])
        if NETSQUID_AVAILABLE:
            try:
                super().__init__(name=name)
            except TypeError:
                super().__init__()
        else:
            super().__init__()

    @staticmethod
    def transmission_probability_from_loss(loss_dB: float) -> float:
        """Convert positive attenuation in dB to photon transmission probability."""
        return transmission_probability_from_loss(loss_dB)

    @staticmethod
    def depolarizing_probability_from_qber(qber: float) -> float:
        """Convert BB84 QBER to a depolarizing noise probability."""
        return depolarizing_probability_from_qber(qber)

    def current_state(self, timestamp: float = 0.0) -> ChannelState:
        """Return the channel state nearest to timestamp."""
        if self.timeseries:
            index = int(max(0, round(timestamp / max(self.update_interval_sec, 1e-9))))
            index = min(index, len(self.timeseries) - 1)
            row = self.timeseries[index]
            loss = float(row.get("loss_dB", self.loss_dB))
            error = float(row.get("expected_QBER", self.error_prob))
            ts = float(row.get("timestamp", timestamp))
        else:
            loss = self.loss_dB
            error = self.error_prob
            ts = timestamp
        return ChannelState(
            timestamp=ts,
            loss_dB=loss,
            error_prob=error,
            transmission_probability=transmission_probability_from_loss(loss),
            depolarizing_probability=depolarizing_probability_from_qber(error),
        )

    def as_netsquid_channel(self, delay: float = 0.0, seed: int | None = None):
        """Build a real NetSquid QuantumChannel for this time-varying link."""
        if not NETSQUID_AVAILABLE:
            raise RuntimeError("NetSquid is not installed; cannot create a QuantumChannel.")
        models = {
            "quantum_loss_model": TimeVaryingSatelliteLossModel(self.timeseries, self.update_interval_sec, seed=seed),
            "quantum_noise_model": TimeVaryingSatelliteNoiseModel(
                self.timeseries,
                self.update_interval_sec,
                seed=None if seed is None else seed + 1,
            ),
        }
        return _NetSquidQuantumChannel(name=getattr(self, "name", self._free_space_name), delay=delay, models=models)

    def transmit_count(self, photon_count: int, timestamp: float, rng: random.Random) -> tuple[int, int]:
        """Classical surrogate: return transmitted and errored photon counts."""
        state = self.current_state(timestamp)
        transmitted = 0
        errors = 0
        for _ in range(max(0, photon_count)):
            if rng.random() <= state.transmission_probability:
                transmitted += 1
                if rng.random() <= state.error_prob:
                    errors += 1
        return transmitted, errors
