"""Orbital dynamics and free-space link budget for satellite QKD.

Uses a simplified sinusoidal pass geometry with explicit equations
and no non-stdlib dependencies. Downstream code consumes one-second
loss/QBER samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from common.io import read_json


CONFIG_PATH = Path(__file__).with_name("config.json")


def _binary_entropy(x: float) -> float:
    """Binary entropy H(x), clipped to [0, 1]."""
    x = min(1.0, max(0.0, x))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log2(x) + (1.0 - x) * math.log2(1.0 - x))


def _estimate_secure_key_bits_per_second(loss_dB: float, qber: float, config: dict[str, Any]) -> float:
    """Estimate secure key bits for a one-second orbital interval.

    Uses the asymptotic decoy-state BB84 formula. All constants are pulled
    from the shared config so this stays in sync with the NetSquid backend
    without importing it.
    """
    qkd = config.get("qkd", {})
    abort = float(qkd.get("qber_abort_threshold", 0.11))
    if qber >= abort:
        return 0.0

    rep = float(qkd.get("source_repetition_rate_hz", 100_000_000.0))
    mu = float(qkd.get("mean_photon_number", 0.8))
    det_eff = float(qkd.get("detector_efficiency", 0.12))
    dark_rate = float(qkd.get("dark_count_rate_hz", 250.0))
    sift = float(qkd.get("sifting_fraction", 0.5))
    f_ec = float(qkd.get("error_correction_efficiency", 1.16))
    impl = float(qkd.get("implementation_efficiency", 0.18))

    eta_channel = 10.0 ** (-loss_dB / 10.0)
    eta = eta_channel * det_eff
    p_dark = dark_rate / rep

    q_signal = 1.0 - math.exp(-mu * eta)
    q_dark = 2.0 * p_dark
    q_mu = min(1.0, q_signal + q_dark)
    q_1 = mu * math.exp(-mu) * eta
    e_1 = min(0.5, max(0.0, qber * 0.92 + 0.004))

    privacy_term = q_1 * (1.0 - _binary_entropy(e_1))
    ec_term = f_ec * q_mu * _binary_entropy(qber)
    per_pulse = max(0.0, privacy_term - ec_term)
    return rep * sift * impl * per_pulse


@dataclass(frozen=True)
class PassParameters:
    """Parameters for one satellite pass."""

    pass_id: int = 0
    start_timestamp: float = 0.0
    altitude_km: float = 500.0
    earth_radius_km: float = 6371.0
    pass_duration_sec: int = 600
    min_elevation_deg: float = 20.0
    max_elevation_deg: float = 90.0
    cn2: float = 1e-17
    extinction_coefficient_per_km: float = 0.07
    atmospheric_height_km: float = 8.0
    satellite_aperture_m: float = 0.3
    ground_telescope_m: float = 1.0
    wavelength_m: float = 850e-9
    pointing_loss_dB: float = 2.5
    calibration_loss_dB: float = 14.0
    optical_misalignment_error: float = 0.02


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the satellite QKD configuration JSON."""
    return dict(read_json(path))


def pass_parameters_from_config(
    config: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> PassParameters:
    """Build PassParameters from config.json and keyword overrides."""
    cfg = dict(config or load_config())
    orbital = cfg.get("orbital", {})
    link = cfg.get("link", {})
    qkd = cfg.get("qkd", {})
    params = PassParameters(
        altitude_km=float(orbital.get("altitude_km", 500.0)),
        earth_radius_km=float(orbital.get("earth_radius_km", 6371.0)),
        pass_duration_sec=int(orbital.get("pass_duration_sec", 600)),
        min_elevation_deg=float(orbital.get("min_elevation_deg", 20.0)),
        max_elevation_deg=float(orbital.get("max_elevation_deg", 90.0)),
        extinction_coefficient_per_km=float(link.get("extinction_coefficient_per_km", 0.07)),
        atmospheric_height_km=float(link.get("atmospheric_height_km", 8.0)),
        satellite_aperture_m=float(link.get("satellite_aperture_m", 0.3)),
        ground_telescope_m=float(link.get("ground_telescope_m", 1.0)),
        wavelength_m=float(link.get("wavelength_m", 850e-9)),
        pointing_loss_dB=float(link.get("pointing_loss_dB", 2.5)),
        calibration_loss_dB=float(link.get("calibration_loss_dB", 14.0)),
        optical_misalignment_error=float(qkd.get("optical_misalignment_error", 0.02)),
    )
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise TypeError(f"Unknown pass parameter: {key}")
        params = replace(params, **{key: value})
    return params


def elevation_angle(
    t: float,
    h: float,
    lat: float = 0.0,
    lon: float = 0.0,
    orbital_elements: Mapping[str, float] | None = None,
) -> float:
    """Return the simplified elevation angle in degrees at time t.

    The latitude, longitude, altitude, and orbital elements are accepted to
    preserve the full propagator interface. In this implementation, orbital
    elements may contain min_elevation_deg, max_elevation_deg, and
    pass_duration_sec.
    """
    del h, lat, lon
    elements = dict(orbital_elements or {})
    duration = float(elements.get("pass_duration_sec", 600.0))
    min_el = float(elements.get("min_elevation_deg", 20.0))
    max_el = float(elements.get("max_elevation_deg", 90.0))
    if duration <= 0:
        return min_el
    phase = min(1.0, max(0.0, t / duration))
    return min_el + (max_el - min_el) * math.sin(math.pi * phase)


def elevation_angle_linear(
    t: float,
    h: float = 500.0,
    lat: float = 0.0,
    lon: float = 0.0,
    orbital_elements: Mapping[str, float] | None = None,
) -> float:
    """Linear ramp elevation model: rise and fall at constant angular rate.

    Used for sensitivity analysis — comparison with the sinusoidal model
    quantifies how much the pass geometry simplification affects results.
    """
    del h, lat, lon
    elements = dict(orbital_elements or {})
    duration = float(elements.get("pass_duration_sec", 600.0))
    min_el = float(elements.get("min_elevation_deg", 20.0))
    max_el = float(elements.get("max_elevation_deg", 90.0))
    if duration <= 0:
        return min_el
    phase = min(1.0, max(0.0, t / duration))
    if phase <= 0.5:
        return min_el + (max_el - min_el) * (phase / 0.5)
    return min_el + (max_el - min_el) * ((1.0 - phase) / 0.5)


def elevation_angle_gaussian(
    t: float,
    h: float = 500.0,
    lat: float = 0.0,
    lon: float = 0.0,
    orbital_elements: Mapping[str, float] | None = None,
) -> float:
    """Gaussian-like elevation model: slower rise/decay near horizon, faster near zenith.

    Used for sensitivity analysis — comparison with the sinusoidal model
    quantifies how much the pass geometry simplification affects results.
    """
    del h, lat, lon
    elements = dict(orbital_elements or {})
    duration = float(elements.get("pass_duration_sec", 600.0))
    min_el = float(elements.get("min_elevation_deg", 20.0))
    max_el = float(elements.get("max_elevation_deg", 90.0))
    if duration <= 0:
        return min_el
    phase = min(1.0, max(0.0, t / duration))
    # Gaussian centered at mid-pass with sigma = duration/4
    sigma = 0.25
    gauss = math.exp(-((phase - 0.5) ** 2) / (2.0 * sigma**2))
    return min_el + (max_el - min_el) * gauss


def slant_range(elevation: float, h: float, earth_radius_km: float = 6371.0) -> float:
    """Compute satellite slant range R(theta), returned in kilometers."""
    theta = math.radians(max(elevation, 1e-6))
    sin_theta = math.sin(theta)
    return -earth_radius_km * sin_theta + math.sqrt(
        (earth_radius_km * sin_theta) ** 2 + 2.0 * earth_radius_km * h + h**2
    )


def geometric_loss_dB(
    R: float,
    D_tx: float,
    D_rx: float,
    wavelength: float,
) -> float:
    """Compute positive geometric attenuation from diffraction coupling."""
    if R <= 0:
        return float("inf")
    range_m = R * 1000.0
    coupling = (math.pi * D_tx * D_rx / (4.0 * wavelength * range_m)) ** 2
    coupling = min(1.0, max(coupling, 1e-300))
    return -10.0 * math.log10(coupling)


def atmospheric_loss_dB(
    elevation: float,
    alpha_ext: float,
    H_atm: float,
) -> float:
    """Compute positive Beer-Lambert atmospheric attenuation in dB."""
    sin_theta = math.sin(math.radians(max(elevation, 1e-6)))
    transmittance = math.exp(-alpha_ext * H_atm / sin_theta)
    return -10.0 * math.log10(max(transmittance, 1e-300))


def fried_parameter_m(
    elevation: float,
    Cn2: float,
    wavelength: float,
    atmospheric_height_m: float = 8000.0,
) -> float:
    """Compute the Fried coherence parameter r0 in meters."""
    k = 2.0 * math.pi / wavelength
    sec_z = 1.0 / math.sin(math.radians(max(elevation, 1e-6)))
    integrated_cn2 = max(Cn2, 1e-30) * atmospheric_height_m
    value = 0.423 * (k**2) * sec_z * integrated_cn2
    return max(value, 1e-300) ** (-3.0 / 5.0)


def turbulence_loss_dB(
    elevation: float,
    Cn2: float,
    wavelength: float,
    D_rx: float,
    atmospheric_height_km: float = 8.0,
) -> float:
    """Estimate turbulence attenuation from the Fried parameter."""
    r0 = fried_parameter_m(elevation, Cn2, wavelength, atmospheric_height_m=atmospheric_height_km * 1000.0)
    aperture_ratio = max(1.0, D_rx / max(r0, 1e-9))
    sec_factor = (1.0 / math.sin(math.radians(max(elevation, 1e-6)))) ** 0.25
    loss = 0.7 + 1.4 * (math.log10(aperture_ratio) ** 1.2) * sec_factor
    return min(12.0, max(0.0, loss))


def expected_qber(
    elevation: float,
    Cn2: float,
    optical_misalignment_error: float = 0.02,
    min_elevation_deg: float = 20.0,
    max_elevation_deg: float = 90.0,
) -> float:
    """Estimate QBER from optical alignment, elevation, and turbulence."""
    span = max_elevation_deg - min_elevation_deg
    low_elevation_fraction = min(1.0, max(0.0, (max_elevation_deg - elevation) / span)) if span > 0 else 0.0
    elevation_penalty = 0.065 * (low_elevation_fraction**1.35)
    turbulence_decades = max(0.0, math.log10(max(Cn2, 1e-30) / 1e-17))
    turbulence_penalty = 0.010 * (turbulence_decades**1.5)
    return min(0.5, optical_misalignment_error + 0.005 + elevation_penalty + turbulence_penalty)


def total_loss_dB(elevation: float, params: PassParameters) -> float:
    """Combine all link-budget terms into total positive attenuation."""
    R = slant_range(elevation, params.altitude_km, params.earth_radius_km)
    return (
        geometric_loss_dB(R, params.satellite_aperture_m, params.ground_telescope_m, params.wavelength_m)
        + atmospheric_loss_dB(elevation, params.extinction_coefficient_per_km, params.atmospheric_height_km)
        + turbulence_loss_dB(elevation, params.cn2, params.wavelength_m, params.ground_telescope_m, params.atmospheric_height_km)
        + params.pointing_loss_dB
        + params.calibration_loss_dB
    )


def total_loss_timeseries(
    pass_params: PassParameters | Mapping[str, Any] | None = None,
    elevation_model: str = "sinusoidal",
) -> list[dict[str, float]]:
    """Return one-second samples for a full satellite pass.

    ``elevation_model`` selects the pass geometry: ``"sinusoidal"`` (default),
    ``"linear"``, or ``"gaussian"``. The default is the simplest model; the
    alternatives exist for the orbital sensitivity analysis (see Section 7.5).
    """
    _MODELS = {
        "sinusoidal": elevation_angle,
        "linear": elevation_angle_linear,
        "gaussian": elevation_angle_gaussian,
    }
    if elevation_model not in _MODELS:
        raise ValueError(f"Unknown elevation_model: {elevation_model!r}; choose from {set(_MODELS)}")
    elev_fn = _MODELS[elevation_model]

    if isinstance(pass_params, PassParameters):
        params = pass_params
    elif pass_params is None:
        params = pass_parameters_from_config()
    else:
        params = pass_parameters_from_config(**dict(pass_params))

    samples: list[dict[str, float]] = []
    elements = {
        "pass_duration_sec": params.pass_duration_sec,
        "min_elevation_deg": params.min_elevation_deg,
        "max_elevation_deg": params.max_elevation_deg,
    }
    for second in range(params.pass_duration_sec):
        elevation = elev_fn(second, params.altitude_km, orbital_elements=elements)
        R = slant_range(elevation, params.altitude_km, params.earth_radius_km)
        # Apply time-varying turbulence: effective Cn² depends on elevation
        effective_cn2 = time_varying_turbulence(elevation, params.cn2)
        loss = total_loss_dB(elevation, params)
        qber = expected_qber(elevation, effective_cn2, params.optical_misalignment_error, params.min_elevation_deg, params.max_elevation_deg)
        samples.append(
            {
                "pass_id": float(params.pass_id),
                "timestamp": params.start_timestamp + second,
                "time_sec": float(second),
                "elevation_deg": elevation,
                "slant_range_km": R,
                "loss_dB": loss,
                "expected_QBER": qber,
                "Cn2": effective_cn2,
            }
        )
    return samples


def compare_orbital_models(
    altitude_km: float = 500.0,
    cn2: float = 1e-15,
    models: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare elevation-angle models for the orbital sensitivity analysis.

    Returns one entry per model with the timeseries and summary stats.
    The key result is ``max_token_deviation_pct`` — the largest percentage
    difference in estimated token yield from the baseline sinusoidal model.
    If all three models produce <3% variation in token yield, the sinusoidal
    simplification is considered validated for the operating range of interest.

    Also reports ``max_deviation_from_baseline_deg`` (elevation) and
    ``max_loss_deviation_dB`` for diagnostics.
    """
    if models is None:
        models = ["sinusoidal", "linear", "gaussian"]
    config = pass_parameters_from_config(altitude_km=altitude_km, cn2=cn2)
    full_config = load_config()
    token_size_bits = int(full_config.get("qkd", {}).get("token_size_bits", 256))
    results: list[dict[str, Any]] = []
    baseline: list[dict[str, float]] | None = None
    baseline_tokens: float | None = None

    for name in models:
        series = total_loss_timeseries(pass_params=config, elevation_model=name)
        elevations = [row["elevation_deg"] for row in series]
        max_el = max(elevations) if elevations else 0.0
        estimated_key_bits = sum(
            _estimate_secure_key_bits_per_second(row["loss_dB"], row["expected_QBER"], full_config)
            for row in series
        )
        estimated_tokens = int(estimated_key_bits) // token_size_bits if token_size_bits > 0 else 0
        entry: dict[str, Any] = {
            "model": name,
            "elevations": elevations,
            "max_elevation_deg": max_el,
            "total_loss_range_dB": (min(r["loss_dB"] for r in series), max(r["loss_dB"] for r in series)),
            "estimated_key_bits": int(round(estimated_key_bits)),
            "estimated_tokens": estimated_tokens,
        }
        if name == "sinusoidal":
            baseline = series
            baseline_tokens = estimated_tokens
            entry["max_deviation_from_baseline_deg"] = 0.0
            entry["max_loss_deviation_dB"] = 0.0
            entry["max_token_deviation_pct"] = 0.0
        elif baseline is not None and baseline_tokens is not None and baseline_tokens > 0:
            max_dev = max(
                abs(r["elevation_deg"] - b["elevation_deg"])
                for r, b in zip(series, baseline)
            )
            max_loss_dev = max(
                abs(r["loss_dB"] - b["loss_dB"])
                for r, b in zip(series, baseline)
            )
            token_dev_pct = 100.0 * abs(estimated_tokens - baseline_tokens) / baseline_tokens
            entry["max_deviation_from_baseline_deg"] = max_dev
            entry["max_loss_deviation_dB"] = max_loss_dev
            entry["max_token_deviation_pct"] = round(token_dev_pct, 3)
        results.append(entry)
    return results


def keplerian_elevation_profile(
    altitude_km: float = 500.0,
    inclination_deg: float = 45.0,
    raan_deg: float = 0.0,
    argument_of_perigee_deg: float = 0.0,
    mean_anomaly_start_deg: float = 0.0,
    eccentricity: float = 0.001,
    min_elevation_deg: float = 20.0,
    max_elevation_deg: float = 90.0,
    pass_duration_sec: int = 600,
) -> tuple[list[float], list[float]]:
    """Compute elevation vs time using simplified Kepler propagation.

    Uses mean anomaly → true anomaly via eccentric anomaly (Newton iteration)
    to produce a physically motivated elevation profile. The simplified geometry
    maps the ground-track angle to elevation via a cosine projection.

    This model is used for orbital sensitivity analysis alongside the sinusoidal,
    linear, and Gaussian models. Output is comparable to ``elevation_angle()``.

    Args:
        altitude_km: Satellite altitude in km.
        inclination_deg: Orbital inclination.
        raan_deg: Right ascension of ascending node.
        argument_of_perigee_deg: Argument of perigee.
        mean_anomaly_start_deg: Initial mean anomaly.
        eccentricity: Orbital eccentricity.
        min_elevation_deg: Minimum elevation.
        max_elevation_deg: Maximum elevation.
        pass_duration_sec: Duration of the pass in seconds.

    Returns:
        Tuple of (times, elevations) where times are in seconds and elevations
        are in degrees.
    """
    mu_earth = 3.986e14  # gravitational parameter (m^3/s^2)
    r_earth = 6371.0e3   # Earth radius (m)
    r_orbit = (altitude_km * 1000.0) + r_earth
    # Orbital period (seconds)
    orbital_period = 2.0 * math.pi * math.sqrt(r_orbit**3 / mu_earth)
    # Mean motion (rad/s)
    n = 2.0 * math.pi / orbital_period

    times: list[float] = []
    elevations: list[float] = []
    el_span = max_elevation_deg - min_elevation_deg

    for sec in range(pass_duration_sec):
        M = math.radians(mean_anomaly_start_deg) + n * sec
        # Solve M = E - e*sin(E) for E via Newton
        E = M
        for _ in range(10):
            dE = (M - (E - eccentricity * math.sin(E))) / (1.0 - eccentricity * math.cos(E))
            E += dE
            if abs(dE) < 1e-12:
                break
        # True anomaly
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + eccentricity) * math.sin(E / 2.0),
            math.sqrt(1.0 - eccentricity) * math.cos(E / 2.0),
        )
        # Elevation proxy: cosine-modulation of true anomaly to [min, max] range
        phase = (1.0 + math.cos(nu)) / 2.0  # 0 at perigee, 1 at apogee → smooth transition
        elev = min_elevation_deg + el_span * (0.5 + 0.5 * math.sin(nu - math.pi / 2.0))
        elev = max(min_elevation_deg, min(max_elevation_deg, elev))

        times.append(float(sec))
        elevations.append(elev)

    return times, elevations


def time_varying_turbulence(
    elevation_deg: float,
    cn2_zenith: float,
) -> float:
    """Compute effective Cn² that varies with elevation angle.

    Models the longer atmospheric path at low elevations, which concentrates
    turbulence effects. The scaling factor is sqrt(sec(ζ)) where ζ = 90° - θ
    is the zenith angle.

    Args:
        elevation_deg: Elevation angle in degrees.
        cn2_zenith: Turbulence strength at zenith (Cn² value at 90° elevation).

    Returns:
        Effective Cn² value accounting for elevation-dependent path lengthening.
    """
    if elevation_deg >= 89.9:
        return cn2_zenith
    zenith_angle = math.radians(max(0.1, 90.0 - elevation_deg))
    sec_zeta = 1.0 / max(math.cos(zenith_angle), 1e-9)
    # Cn² scales as sqrt(sec(ζ))
    scaling = math.sqrt(sec_zeta)
    return cn2_zenith * scaling

