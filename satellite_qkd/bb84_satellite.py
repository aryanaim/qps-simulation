"""Decoy-state BB84 simulation over the satellite free-space channel.

This module requires NetSquid. The ``simulate_bb84_over_timeseries`` function
delegates to the NetSquid weighted-Monte-Carlo backend for photon-level BB84
simulation with decoy-state analysis.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .netsquid_backend import NETSQUID_AVAILABLE, require_netsquid
from .netsquid_backend import simulate_bb84_over_timeseries_netsquid
from .orbital_dynamics import load_config


@dataclass(frozen=True)
class BB84IntervalResult:
    """Per-second BB84 output consumed by the pass simulator."""

    timestamp: float
    time_sec: float
    elevation_deg: float
    loss_dB: float
    expected_QBER: float
    measured_QBER: float
    raw_bits: int
    sifted_bits: int
    secure_bits: int
    aborted: bool
    backend: str = "netsquid"
    sampled_pulses: int = 0
    signal_gain: float = 0.0
    weak_gain: float = 0.0
    vacuum_gain: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def binary_entropy(x: float) -> float:
    """Binary entropy H(x), clipped to the physical interval."""
    x = min(1.0, max(0.0, x))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log2(x) + (1.0 - x) * math.log2(1.0 - x))


def qkd_parameters(config: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Extract numeric BB84 parameters from the shared config."""
    cfg = dict(config or load_config())
    qkd = cfg.get("qkd", {})
    return {
        "source_repetition_rate_hz": float(qkd.get("source_repetition_rate_hz", 100_000_000.0)),
        "mean_photon_number": float(qkd.get("mean_photon_number", 0.8)),
        "detector_efficiency": float(qkd.get("detector_efficiency", 0.12)),
        "dark_count_rate_hz": float(qkd.get("dark_count_rate_hz", 250.0)),
        "sifting_fraction": float(qkd.get("sifting_fraction", 0.5)),
        "error_correction_efficiency": float(qkd.get("error_correction_efficiency", 1.16)),
        "implementation_efficiency": float(qkd.get("implementation_efficiency", 0.18)),
        "qber_abort_threshold": float(qkd.get("qber_abort_threshold", 0.11)),
    }


def secure_key_rate_bps(
    loss_dB: float,
    qber: float,
    params: Mapping[str, float] | None = None,
) -> float:
    """Approximate asymptotic decoy-state BB84 secure key rate.

    Used by the NetSquid backend to compute the secure key from the Monte Carlo
    decoy-state gains.
    """
    p = dict(params or qkd_parameters())
    if qber >= p["qber_abort_threshold"]:
        return 0.0

    rep = p["source_repetition_rate_hz"]
    mu = p["mean_photon_number"]
    from .channel_model import FreeSpaceChannel
    eta_channel = FreeSpaceChannel.transmission_probability_from_loss(loss_dB)
    eta = eta_channel * p["detector_efficiency"]
    p_dark = p["dark_count_rate_hz"] / rep

    q_signal = 1.0 - math.exp(-mu * eta)
    q_dark = 2.0 * p_dark
    q_mu = min(1.0, q_signal + q_dark)
    q_1 = mu * math.exp(-mu) * eta
    e_1 = min(0.5, max(0.0, qber * 0.92 + 0.004))

    privacy_term = q_1 * (1.0 - binary_entropy(e_1))
    ec_term = p["error_correction_efficiency"] * q_mu * binary_entropy(qber)
    per_pulse = max(0.0, privacy_term - ec_term)
    return rep * p["sifting_fraction"] * p["implementation_efficiency"] * per_pulse


def require_backend() -> None:
    """Fail loudly if NetSquid is not available."""
    if not NETSQUID_AVAILABLE:
        require_netsquid()


def simulate_bb84_over_timeseries(
    timeseries: list[Mapping[str, float]],
    config: Mapping[str, Any] | None = None,
    seed: int | None = None,
    pulses_per_interval: int | None = None,
) -> list[BB84IntervalResult]:
    """Run NetSquid BB84 over a one-second orbital series.

    This function always uses the NetSquid backend. If NetSquid is not installed
    the call fails with a clear error message pointing to the QuTech package server.
    """
    require_backend()
    return simulate_bb84_over_timeseries_netsquid(
        timeseries,
        config=dict(config or load_config()),
        seed=seed,
        pulses_per_interval=pulses_per_interval,
    )
