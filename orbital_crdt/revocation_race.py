"""Satellite-specific revocation race demonstration."""

from __future__ import annotations

from typing import Any

from satellite_qkd.orbital_dynamics import load_config
from .node import GSetNode
from .scheduler import SatelliteVisibilityScheduler


def run_revocation_race(
    token_id: str = "T-race",
    next_pass_min: float = 47.0,
    presentation_before_min: float = 30.0,
    presentation_after_min: float = 50.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the seven-step revocation race demonstration."""
    cfg = config or load_config()
    crdt_cfg = cfg.get("crdt", {})
    terrestrial_latency_sec = float(crdt_cfg.get("terrestrial_latency_max_ms", 200.0)) / 1000.0
    satellite_latency_sec = float(crdt_cfg.get("leo_satellite_latency_ms", 600.0)) / 1000.0
    scheduler = SatelliteVisibilityScheduler.periodic(
        orbital_period_min=94.6,
        first_window_start_min=next_pass_min,
        satellite_latency_sec=satellite_latency_sec,
        horizon_min=180.0,
    )
    gs_a, gs_b, gs_c = GSetNode("GS-A"), GSetNode("GS-B"), GSetNode("GS-C")
    events: list[dict[str, Any]] = []

    gs_a.revoke(token_id)
    events.append({"timestamp": 0.0, "timestamp_min": 0.0, "event": f"GS-A revokes {token_id}"})

    gs_b.import_state(gs_a.export_state())
    events.append(
        {
            "timestamp": terrestrial_latency_sec,
            "timestamp_min": terrestrial_latency_sec / 60.0,
            "event": "GS-B receives revocation over terrestrial link",
        }
    )

    before_time = presentation_before_min * 60.0
    before_accept = gs_c.accepts(token_id)
    events.append(
        {
            "timestamp": before_time,
            "timestamp_min": presentation_before_min,
            "event": f"Holder presents at GS-C before satellite pass: {'false-accept' if before_accept else 'reject'}",
        }
    )

    merge_time = scheduler.delivery_time(0.0)
    gs_c.import_state(gs_a.export_state())
    events.append(
        {
            "timestamp": merge_time,
            "timestamp_min": merge_time / 60.0,
            "event": "GS-C receives revocation during satellite merge",
        }
    )

    after_time = presentation_after_min * 60.0
    after_accept = gs_c.accepts(token_id)
    events.append(
        {
            "timestamp": after_time,
            "timestamp_min": presentation_after_min,
            "event": f"Holder presents at GS-C after satellite pass: {'false-accept' if after_accept else 'reject'}",
        }
    )

    return {
        "token_id": token_id,
        "false_accept_window_sec": merge_time,
        "false_accept_window_min": merge_time / 60.0,
        "before_accept": before_accept,
        "after_accept": after_accept,
        "events": sorted(events, key=lambda item: item["timestamp"]),
    }

