"""Satellite pass orchestration for WS1."""

from __future__ import annotations

import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from common.io import write_json
from .bb84_satellite import simulate_bb84_over_timeseries
from .orbital_dynamics import PassParameters, load_config, pass_parameters_from_config, total_loss_timeseries


def simulate_pass(
    pass_params: PassParameters | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int | None = None,
    pulses_per_interval: int | None = None,
) -> dict[str, Any]:
    """Run one full satellite pass and aggregate its key yield.

    Uses the NetSquid BB84 backend for photon-level simulation.
    """
    cfg = dict(config or load_config())
    if isinstance(pass_params, PassParameters):
        params = pass_params
    elif pass_params is None:
        params = pass_parameters_from_config(cfg)
    else:
        params = pass_parameters_from_config(cfg, **dict(pass_params))

    orbital = total_loss_timeseries(params)
    intervals = simulate_bb84_over_timeseries(
        orbital,
        config=cfg,
        seed=seed,
        pulses_per_interval=pulses_per_interval,
    )
    rows = [item.to_dict() for item in intervals]
    total_secure = sum(item.secure_bits for item in intervals)
    token_size = int(cfg.get("qkd", {}).get("token_size_bits", 256))
    qbers = [item.measured_QBER for item in intervals]
    non_aborted = [item for item in intervals if not item.aborted]
    duration_above_threshold = len(non_aborted)
    # avg_QBER excludes aborted intervals (QBER jumps to 0.5–1.0 during abort,
    # which would inflate the pass average and misrepresent channel quality).
    non_aborted_qbers = [item.measured_QBER for item in non_aborted]

    return {
        "pass_id": params.pass_id,
        "parameters": asdict(params),
        "summary": {
            "timestamp": params.start_timestamp,
            "duration_sec": params.pass_duration_sec,
            "total_secure_key_bits": total_secure,
            "tokens_issued": total_secure // token_size,
            "avg_QBER": statistics.fmean(non_aborted_qbers) if non_aborted_qbers else 0.0,
            "peak_QBER": max(qbers) if qbers else 0.0,
            "pass_duration_above_threshold_sec": duration_above_threshold,
            "token_size_bits": token_size,
            "backend": rows[0]["backend"] if rows else "netsquid",
            "sampled_pulses_per_interval": rows[0]["sampled_pulses"] if rows else 0,
        },
        "timeseries": rows,
        "contract_A": {
            "pass_id": params.pass_id,
            "timestamp": params.start_timestamp,
            "tokens_issued": total_secure // token_size,
            "avg_QBER": statistics.fmean(non_aborted_qbers) if non_aborted_qbers else 0.0,
            "duration_sec": params.pass_duration_sec,
            "total_secure_key_bits": total_secure,
        },
    }


def write_pass_output(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write a pass JSON file containing Contract A and per-second samples."""
    return write_json(path, result)
