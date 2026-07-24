"""Statistical validation helpers for the simulation pipeline.

Provides bootstrap confidence intervals, effect-size measures, power analysis,
and distribution-fit tests — all using only the Python standard library so the
simulation remains dependency-free.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Sequence


# ── Bootstrap ────────────────────────────────────────────────────────


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Return bootstrap confidence interval for a sample statistic.

    ``statistic`` can be ``"mean"``, ``"median"``, or a callable.
    Returns ``{"low": p_low, "high": p_high, "stat": observed_value,
    "n_resamples": n, "ci": ci}``.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"low": 0.0, "high": 0.0, "stat": 0.0, "n": 0, "n_resamples": 0, "ci": ci}
    if statistic == "mean":
        obs = sum(vals) / n
        _stat = lambda sample: sum(sample) / len(sample)  # noqa: E731
    elif statistic == "median":
        ordered = sorted(vals)
        obs = ordered[n // 2]
        _stat = lambda sample: sorted(sample)[len(sample) // 2]  # noqa: E731
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}")

    rng = random.Random(seed)
    bootstraps: list[float] = []
    for _ in range(n_resamples):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        bootstraps.append(_stat(resample))
    bootstraps.sort()
    alpha = 1.0 - ci
    p_low = alpha / 2.0
    p_high = 1.0 - alpha / 2.0
    low_index = max(0, int(p_low * n_resamples))
    high_index = min(n_resamples - 1, int(p_high * n_resamples))
    return {
        "low": bootstraps[low_index],
        "high": bootstraps[high_index],
        "stat": obs,
        "n": n,
        "n_resamples": n_resamples,
        "ci": ci,
    }


# ── Effect size (Cohen's d) ─────────────────────────────────────────


def cohens_d(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> dict[str, float]:
    """Compute Cohen's d for two independent groups.

    Returns ``{"d": d, "interpretation": "negligible"|"small"|"medium"|"large",
    "n_a": n, "n_b": n}``.
    """
    a, b = list(group_a), list(group_b)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {"d": 0.0, "interpretation": "negligible", "n_a": n_a, "n_b": n_b}
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return {"d": 0.0, "interpretation": "negligible", "n_a": n_a, "n_b": n_b}
    d_val = abs(mean_a - mean_b) / pooled
    if d_val < 0.2:
        interp = "negligible"
    elif d_val < 0.5:
        interp = "small"
    elif d_val < 0.8:
        interp = "medium"
    else:
        interp = "large"
    return {"d": round(d_val, 4), "interpretation": interp, "n_a": n_a, "n_b": n_b}


# ── Power analysis ──────────────────────────────────────────────────


def minimum_detectable_effect(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    """Approximate minimum detectable Cohen's d for a two-sample t-test.

    Uses the z-test approximation: d ≈ (z_{1-alpha/2} + z_{power}) * sqrt(2/n).
    Valid for n > 10 per group. For smaller n, the t-distribution correction
    is needed but the approximation is conservative (overestimates required n).
    """
    if n_per_group < 2:
        return {"d": float("inf"), "n_per_group": n_per_group, "alpha": alpha, "power": power}
    z_alpha = 1.96 if alpha == 0.05 else _normal_ppf(1.0 - alpha / 2.0)
    z_power = 0.84 if power == 0.80 else _normal_ppf(power)
    d_min = (z_alpha + z_power) * math.sqrt(2.0 / n_per_group)
    return {"d": round(d_min, 4), "n_per_group": n_per_group, "alpha": alpha, "power": power}


def _normal_ppf(p: float) -> float:
    """Approximate standard normal quantile (Abramowitz & Stegun 26.2.23)."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    t = math.sqrt(-2.0 * math.log(1.0 - p if p > 0.5 else p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    return -z if p <= 0.5 else z


# ── KS goodness-of-fit test ─────────────────────────────────────────


def ks_test_normal(values: Sequence[float]) -> dict[str, float]:
    """Lilliefors-corrected Kolmogorov-Smirnov test for normality.

    Returns the KS statistic and an approximate p-value. The critical values
    for the Lilliefors correction are approximated by interpolation from
    Dallal & Wilkinson (1986). When p < 0.05, the null hypothesis of
    normality is rejected at the 5% level.
    """
    vals = sorted(values)
    n = len(vals)
    if n < 4:
        return {"statistic": 0.0, "p_value": 1.0, "n": n, "is_normal": True}
    mean = sum(vals) / n
    variance = sum((x - mean) ** 2 for x in vals) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 1.0
    # Empirical CDF vs fitted normal CDF
    d_plus = 0.0
    d_minus = 0.0
    for i, x in enumerate(vals):
        z = (x - mean) / std
        ecdf = (i + 1) / n
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        d_plus = max(d_plus, ecdf - cdf)
        d_minus = max(d_minus, cdf - (i / n))
    d_stat = max(d_plus, d_minus)
    # Lilliefors critical approximation (Dallal & Wilkinson 1986)
    # For alpha=0.05: critical ≈ 0.886 / sqrt(n)
    # For alpha=0.01: critical ≈ 1.031 / sqrt(n)
    critical_05 = 0.886 / math.sqrt(n)
    critical_01 = 1.031 / math.sqrt(n)
    if d_stat > critical_01:
        p_value = 0.005
    elif d_stat > critical_05:
        p_value = 0.025
    else:
        p_value = 0.50
    return {
        "statistic": round(d_stat, 4),
        "p_value": p_value,
        "n": n,
        "is_normal": p_value >= 0.05,
    }


# ── Summary report ──────────────────────────────────────────────────


def validation_report(values: Sequence[float], label: str = "sample") -> dict[str, Any]:
    """Produce a combined validation report for a sample.

    Returns bootstrap CI, normality test, and descriptive statistics.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"label": label, "n": 0, "error": "empty sample"}
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / max(1, n - 1)) if n > 1 else 0.0
    ordered = sorted(vals)
    median = ordered[n // 2]
    boot = bootstrap_ci(vals)
    norm = ks_test_normal(vals)
    return {
        "label": label,
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(median, 4),
        "ci_95_low": boot["low"],
        "ci_95_high": boot["high"],
        "normality_p": norm["p_value"],
        "is_normal": norm["is_normal"],
        "statistical_power_80": minimum_detectable_effect(n),
    }
