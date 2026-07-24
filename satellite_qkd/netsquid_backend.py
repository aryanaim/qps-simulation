"""NetSquid BB84 backend for satellite QKD.

The 100 MHz source rate is far too high for one-qubit-per-pulse brute force
sweeps, so this backend uses weighted Monte Carlo: a configurable number of
representative pulses is simulated with NetSquid qubit preparation and
measurement in each orbital interval, then scaled to the physical pulse rate.
That is the normal production pattern for this kind of study: NetSquid handles
the quantum state transitions; the orbital/rate layer handles the bulk traffic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from .channel_model import FreeSpaceChannel, NETSQUID_AVAILABLE, depolarizing_probability_from_qber


class NetSquidUnavailableError(RuntimeError):
    """Raised when the strict NetSquid backend is requested but unavailable."""


@dataclass
class IntensityStats:
    """Observed decoy-state counts for one intensity class."""

    sent: int = 0
    emitted: int = 0
    detected: int = 0
    sifted: int = 0
    errors: int = 0

    @property
    def gain(self) -> float:
        return self.detected / self.sent if self.sent else 0.0

    @property
    def qber(self) -> float:
        return self.errors / self.sifted if self.sifted else 0.0


@dataclass
class IntervalStats:
    """Weighted NetSquid counts for one orbital interval."""

    raw_bits: int
    sifted_bits: int
    secure_bits: int
    measured_qber: float
    signal_gain: float
    weak_gain: float
    vacuum_gain: float
    sampled_pulses: int
    intensity_stats: dict[str, IntensityStats] = field(default_factory=dict)


def require_netsquid() -> None:
    """Fail clearly when NetSquid is not importable."""
    if not NETSQUID_AVAILABLE:
        raise NetSquidUnavailableError(
            "NetSquid is required for backend='netsquid'. Install it with access to "
            "QuTech's package server, for example: "
            "pip install --extra-index-url https://pypi.netsquid.org netsquid"
        )


def _intensity_schedule(config: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    """Return the decoy-state intensity schedule from config."""
    qkd = config.get("qkd", {})
    mu = float(qkd.get("mean_photon_number", 0.8))
    decoys = list(qkd.get("decoy_intensities", [0.1, 0.0]))
    weak = float(decoys[0]) if decoys else 0.1
    vacuum = float(decoys[1]) if len(decoys) > 1 else 0.0
    probs = qkd.get("decoy_probabilities", {"signal": 0.80, "weak": 0.15, "vacuum": 0.05})
    rows = [
        ("signal", mu, max(0.0, float(probs.get("signal", 0.80)))),
        ("weak", weak, max(0.0, float(probs.get("weak", 0.15)))),
        ("vacuum", vacuum, max(0.0, float(probs.get("vacuum", 0.05)))),
    ]
    total = sum(probability for _, _, probability in rows) or 1.0
    return [(label, intensity, probability / total) for label, intensity, probability in rows]


def _choose_intensity(schedule: list[tuple[str, float, float]], rng: random.Random) -> tuple[str, float]:
    """Pick a decoy intensity weighted by its probability in the schedule."""
    threshold = rng.random()
    cumulative = 0.0
    for label, intensity, probability in schedule:
        cumulative += probability
        if threshold <= cumulative:
            return label, intensity
    label, intensity, _ = schedule[-1]
    return label, intensity


def _prepare_qubit(ns: Any, qapi: Any, bit: int, basis: int):
    qubit = qapi.create_qubits(1)[0]
    if bit:
        qapi.operate(qubit, ns.X)
    if basis:
        qapi.operate(qubit, ns.H)
    return qubit


def _apply_depolarizing_noise(ns: Any, qapi: Any, qubit: Any, qber: float, rng: random.Random) -> None:
    if rng.random() < depolarizing_probability_from_qber(qber):
        qapi.operate(qubit, rng.choice([ns.X, ns.Y, ns.Z]))


def _measure(ns: Any, qapi: Any, qubit: Any, basis: int) -> int:
    observable = ns.X if basis else ns.Z
    outcome, _ = qapi.measure(qubit, observable)
    return int(outcome)


def simulate_netsquid_interval(
    sample: Mapping[str, float],
    config: Mapping[str, Any],
    rng: random.Random,
    pulses_per_interval: int,
    duration_sec: float = 1.0,
) -> IntervalStats:
    """Simulate one interval with NetSquid qubits and weighted decoy counts."""
    require_netsquid()
    import netsquid as ns

    qapi = ns.qubits.qubitapi
    ns.sim_reset()
    qkd = config.get("qkd", {})
    rep = float(qkd.get("source_repetition_rate_hz", 100_000_000.0))
    detector_efficiency = float(qkd.get("detector_efficiency", 0.12))
    dark_count_rate_hz = float(qkd.get("dark_count_rate_hz", 250.0))
    dark_probability = min(1.0, dark_count_rate_hz / rep)
    qber = float(sample["expected_QBER"])
    loss = float(sample["loss_dB"])
    eta = FreeSpaceChannel.transmission_probability_from_loss(loss) * detector_efficiency
    schedule = _intensity_schedule(config)
    stats = {label: IntensityStats() for label, _, _ in schedule}

    for _ in range(max(1, pulses_per_interval)):
        label, intensity = _choose_intensity(schedule, rng)
        bucket = stats[label]
        bucket.sent += 1
        alice_basis = 1 if rng.random() < 0.5 else 0
        bob_basis = 1 if rng.random() < 0.5 else 0
        alice_bit = 1 if rng.random() < 0.5 else 0
        emitted = rng.random() < (1.0 - math.exp(-intensity))
        detected = False
        measured_bit = 0

        if emitted:
            bucket.emitted += 1
            if rng.random() < eta:
                qubit = _prepare_qubit(ns, qapi, alice_bit, alice_basis)
                _apply_depolarizing_noise(ns, qapi, qubit, qber, rng)
                measured_bit = _measure(ns, qapi, qubit, bob_basis)
                detected = True

        if not detected and rng.random() < dark_probability:
            measured_bit = 1 if rng.random() < 0.5 else 0
            detected = True

        if not detected:
            continue

        bucket.detected += 1
        if alice_basis == bob_basis:
            bucket.sifted += 1
            if measured_bit != alice_bit:
                bucket.errors += 1

    scale = rep * duration_sec / max(1, pulses_per_interval)
    raw_bits = int(round(sum(bucket.detected for bucket in stats.values()) * scale))
    sifted_signal = stats["signal"].sifted
    sifted_bits = int(round(sifted_signal * scale))
    measured_qber = stats["signal"].qber or qber
    secure_bits = 0
    if measured_qber < float(qkd.get("qber_abort_threshold", 0.11)):
        from .bb84_satellite import secure_key_rate_bps, qkd_parameters

        skr_params = qkd_parameters(config)
        raw_secure = secure_key_rate_bps(loss, measured_qber, skr_params) * duration_sec
        secure_bits = min(sifted_bits, int(round(raw_secure)))

    return IntervalStats(
        raw_bits=raw_bits,
        sifted_bits=sifted_bits,
        secure_bits=secure_bits,
        measured_qber=measured_qber,
        signal_gain=stats["signal"].gain,
        weak_gain=stats["weak"].gain,
        vacuum_gain=stats["vacuum"].gain,
        sampled_pulses=pulses_per_interval,
        intensity_stats=stats,
    )


def simulate_bb84_over_timeseries_netsquid(
    timeseries: list[Mapping[str, float]],
    config: Mapping[str, Any],
    seed: int | None = None,
    pulses_per_interval: int | None = None,
) -> list:
    """Run the NetSquid BB84 backend over an orbital time series."""
    from .bb84_satellite import BB84IntervalResult

    require_netsquid()
    rng = random.Random(seed)
    # Build the actual NetSquid channel once so model construction/API
    # compatibility is validated by research runs.
    FreeSpaceChannel(timeseries=timeseries).as_netsquid_channel(seed=seed)
    qkd = config.get("qkd", {})
    sample_count = int(pulses_per_interval or qkd.get("netsquid_pulses_per_interval", 2048))
    results = []
    threshold = float(qkd.get("qber_abort_threshold", 0.11))
    for sample in timeseries:
        stats = simulate_netsquid_interval(sample, config, rng, sample_count)
        aborted = stats.measured_qber >= threshold
        results.append(
            BB84IntervalResult(
                timestamp=float(sample.get("timestamp", 0.0)),
                time_sec=float(sample.get("time_sec", 0.0)),
                elevation_deg=float(sample.get("elevation_deg", 0.0)),
                loss_dB=float(sample["loss_dB"]),
                expected_QBER=float(sample["expected_QBER"]),
                measured_QBER=stats.measured_qber,
                raw_bits=stats.raw_bits,
                sifted_bits=stats.sifted_bits,
                secure_bits=0 if aborted else stats.secure_bits,
                aborted=aborted,
                backend="netsquid",
                sampled_pulses=stats.sampled_pulses,
                signal_gain=stats.signal_gain,
                weak_gain=stats.weak_gain,
                vacuum_gain=stats.vacuum_gain,
            )
        )
    return results


def characterize_variance(
    config: Mapping[str, Any] | None = None,
    n_runs: int = 10,
    pulses_per_interval: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Characterize the coefficient of variation of total secure key bits.

    Runs ``n_runs`` zenith pass simulations with different seeds and reports
    ``std / mean`` (CV). If CV > 0.3, computes the pulses/interval needed for
    CV < 0.2.

    Args:
        config: Full configuration dict.
        n_runs: Number of replicate runs for variance estimation.
        pulses_per_interval: NetSquid samples per interval.
        seed: Base RNG seed.

    Returns:
        Dict with cv, mean_key_bits, std_key_bits, pulses_per_interval, and
        recommended_pulses (if CV > 0.3).
    """
    from .pass_simulator import simulate_pass

    cfg = dict(config or load_config())
    sample_count = pulses_per_interval or int(cfg.get("qkd", {}).get("netsquid_pulses_per_interval", 2048))
    key_bits: list[float] = []

    for run in range(n_runs):
        result = simulate_pass(
            {"pass_id": run, "altitude_km": 500.0, "max_elevation_deg": 90.0, "cn2": 1e-17},
            config=cfg,
            seed=seed + run * 100,
            pulses_per_interval=sample_count,
        )
        key_bits.append(float(result["summary"]["total_secure_key_bits"]))

    if len(key_bits) < 2:
        return {
            "n_runs": n_runs,
            "error": "insufficient runs for variance estimate",
        }

    mean_bits = sum(key_bits) / len(key_bits)
    variance = sum((b - mean_bits) ** 2 for b in key_bits) / (len(key_bits) - 1)
    std_bits = math.sqrt(variance)
    cv = std_bits / mean_bits if mean_bits > 0 else 0.0

    result: dict[str, Any] = {
        "n_runs": n_runs,
        "pulses_per_interval": sample_count,
        "mean_key_bits": round(mean_bits, 1),
        "std_key_bits": round(std_bits, 1),
        "coefficient_of_variation": round(cv, 4),
        "key_bits_by_run": [round(b, 1) for b in key_bits],
    }

    if cv > 0.3 and cv > 0.0:
        # Estimate pulses needed: CV ∝ 1/sqrt(N), so N_needed ≈ N * (CV / 0.2)^2
        recommended = int(sample_count * (cv / 0.2) ** 2)
        result["cv_exceeds_threshold"] = True
        result["recommended_pulses_per_interval"] = recommended
    else:
        result["cv_exceeds_threshold"] = False

    return result
