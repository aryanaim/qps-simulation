"""Random helpers that avoid non-standard dependencies."""

from __future__ import annotations

import math
import random


def poisson_count(rate: float, rng: random.Random) -> int:
    """Draw a Poisson count with a normal approximation for large rates."""
    if rate <= 0:
        return 0
    if rate < 30:
        threshold = math.exp(-rate)
        k = 0
        product = 1.0
        while product > threshold:
            k += 1
            product *= rng.random()
        return k - 1
    return max(0, int(round(rng.gauss(rate, math.sqrt(rate)))))


def noisy_count(mean: float, rng: random.Random, relative_sigma: float = 0.03) -> int:
    """Return a non-negative integer count with mild Gaussian variation."""
    if mean <= 0:
        return 0
    sigma = max(1.0, abs(mean) * relative_sigma)
    return max(0, int(round(rng.gauss(mean, sigma))))


def percentile(values: list[float], pct: float) -> float:
    """Return a percentile from a sorted copy of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    weight = idx - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

