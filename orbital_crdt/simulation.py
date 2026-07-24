"""Async-shaped CRDT simulations for the three-ground-station topology."""

from __future__ import annotations

import asyncio
from typing import Any

from satellite_qkd.orbital_dynamics import load_config
from .node import GSetNode
from .scheduler import SatelliteVisibilityScheduler


def simulate_three_node_convergence(
    token_id: str = "T-demo",
    origin: str = "GS-A",
    revocation_time: float = 0.0,
    orbital_period_min: float = 94.6,
    next_pass_offset_min: float | None = None,
    config: dict[str, Any] | None = None,
    fiber_backup: bool = False,
) -> dict[str, Any]:
    """Deterministically simulate CRDT convergence after one revocation.

    Args:
        token_id: Token identifier for the revocation.
        origin: Node that initiates the revocation.
        revocation_time: Simulation timestamp of revocation.
        orbital_period_min: Orbital period in minutes.
        next_pass_offset_min: Minutes until the first satellite pass window
            (defaults to orbital_period_min if not given).
        config: Full configuration dict.
        fiber_backup: If True, GS-C is connected via terrestrial fiber (always
            available) instead of satellite relay, eliminating orbital delays.

    Returns:
        Dict with convergence metrics and event log.
    """
    cfg = config or load_config()
    crdt_cfg = cfg.get("crdt", {})
    terrestrial_latency_sec = float(crdt_cfg.get("terrestrial_latency_max_ms", 200.0)) / 1000.0
    satellite_latency_sec = float(crdt_cfg.get("leo_satellite_latency_ms", 600.0)) / 1000.0
    offset = orbital_period_min if next_pass_offset_min is None else next_pass_offset_min
    scheduler = SatelliteVisibilityScheduler.periodic(
        orbital_period_min=orbital_period_min,
        first_window_start_min=offset,
        satellite_latency_sec=satellite_latency_sec,
        horizon_min=orbital_period_min * 3,
    )

    nodes = {node_id: GSetNode(node_id) for node_id in ["GS-A", "GS-B", "GS-C"]}
    nodes[origin].revoke(token_id)
    arrival_times = {origin: revocation_time}
    events = [
        {
            "timestamp": revocation_time,
            "timestamp_min": revocation_time / 60.0,
            "event": f"{origin} revokes {token_id}",
        }
    ]

    if origin in {"GS-A", "GS-B"}:
        peer = "GS-B" if origin == "GS-A" else "GS-A"
        arrival_times[peer] = revocation_time + terrestrial_latency_sec
        events.append({"timestamp": arrival_times[peer], "timestamp_min": arrival_times[peer] / 60.0, "event": f"{peer} receives revocation over terrestrial link"})
        if fiber_backup:
            # GS-C has fiber backup — deliver over terrestrial link immediately
            arrival_times["GS-C"] = revocation_time + terrestrial_latency_sec
            events.append({"timestamp": arrival_times["GS-C"], "timestamp_min": arrival_times["GS-C"] / 60.0, "event": "GS-C receives revocation over fiber backup link"})
        else:
            # GS-C only reachable via satellite relay
            arrival_times["GS-C"] = scheduler.delivery_time(revocation_time)
            events.append({"timestamp": arrival_times["GS-C"], "timestamp_min": arrival_times["GS-C"] / 60.0, "event": "GS-C receives revocation at satellite merge"})
    else:
        # origin is GS-C
        if fiber_backup:
            # GS-C delivers revocation to GS-A, GS-B over fiber directly
            arrival_times["GS-A"] = revocation_time + terrestrial_latency_sec
            events.append({"timestamp": arrival_times["GS-A"], "timestamp_min": arrival_times["GS-A"] / 60.0, "event": "GS-A receives GS-C revocation over fiber backup link"})
        else:
            arrival_times["GS-A"] = scheduler.delivery_time(revocation_time)
            events.append({"timestamp": arrival_times["GS-A"], "timestamp_min": arrival_times["GS-A"] / 60.0, "event": "GS-A receives GS-C revocation at satellite merge"})
        arrival_times["GS-B"] = arrival_times["GS-A"] + terrestrial_latency_sec
        events.append({"timestamp": arrival_times["GS-B"], "timestamp_min": arrival_times["GS-B"] / 60.0, "event": "GS-B receives revocation from GS-A terrestrial propagation"})

    for node_id, arrival in arrival_times.items():
        if arrival >= revocation_time:
            nodes[node_id].revoke(token_id)

    max_arrival = max(arrival_times.values())
    return {
        "token_id": token_id,
        "origin": origin,
        "revocation_time": revocation_time,
        "arrival_times": arrival_times,
        "max_convergence_time_sec": max_arrival - revocation_time,
        "false_accept_window_sec": max(0.0, arrival_times.get("GS-C", revocation_time) - revocation_time),
        "events": sorted(events, key=lambda item: item["timestamp"]),
        "all_converged": all(not node.accepts(token_id) for node in nodes.values()),
    }


async def run_async_convergence(**kwargs: Any) -> dict[str, Any]:
    """Async wrapper matching the intended orchestrator shape."""
    await asyncio.sleep(0)
    return simulate_three_node_convergence(**kwargs)


def simulate_byzantine_scenario(
    token_id: str = "T-byzantine",
    origin: str = "GS-A",
    revocation_time: float = 0.0,
    byzantine_node: str = "GS-C",
    drop_revocations: bool = True,
    target_token_ids: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a Byzantine adversary scenario where one node misbehaves.

    In the non-responding mode (``drop_revocations=True``), the Byzantine node
    does not propagate revocations. In the selective-drop mode, it drops
    revocations for specific token IDs only.

    GS-A and GS-B detect the divergence via periodic consistency checks
    (heartbeat protocol). After N missed merges, they may blacklist the
    Byzantine node.

    Args:
        token_id: Token identifier for the revocation.
        origin: Node that initiates the revocation.
        revocation_time: Simulation timestamp of revocation.
        byzantine_node: Node identifier that behaves byzantinely.
        drop_revocations: If True, the node drops all revocations.
        target_token_ids: If set, only drop revocations for these tokens.
        config: Full configuration dict.

    Returns:
        Dict with detection_time_sec, convergence_status, and event log.
    """
    cfg = config or load_config()
    crdt_cfg = cfg.get("crdt", {})
    terrestrial_latency_sec = float(crdt_cfg.get("terrestrial_latency_max_ms", 200.0)) / 1000.0
    heartbeat_interval = 2.0  # seconds between consistency checks
    max_missed_heartbeats = 3

    nodes = {node_id: GSetNode(node_id) for node_id in ["GS-A", "GS-B", "GS-C"]}
    nodes[origin].revoke(token_id)

    events: list[dict[str, Any]] = [
        {"timestamp": revocation_time, "event": f"{origin} revokes {token_id}"},
    ]

    # Simulate revocation propagation with a Byzantine node
    honest_nodes = [n for n in ["GS-A", "GS-B", "GS-C"] if n != byzantine_node]
    arrival_times: dict[str, float] = {origin: revocation_time}

    # Propagate to peers
    for node_id in honest_nodes:
        if node_id == origin:
            continue
        delay = terrestrial_latency_sec if node_id != "GS-C" else crdt_cfg.get("leo_satellite_latency_ms", 600.0) / 1000.0
        arrival = revocation_time + delay
        arrival_times[node_id] = arrival
        events.append({"timestamp": arrival, "event": f"{node_id} receives revocation (delay={delay:.3f}s)"})

    # Byzantine node behavior: determine if it drops the revocation
    should_drop = drop_revocations
    if target_token_ids is not None and token_id not in target_token_ids:
        should_drop = False

    if should_drop:
        # Byzantine node does not revoke
        events.append({"timestamp": revocation_time + 0.01, "event": f"{byzantine_node} (Byzantine) drops revocation"})
        # Simulate heartbeat detection
        detection_time = revocation_time + heartbeat_interval * max_missed_heartbeats
        events.append({"timestamp": detection_time, "event": f"Heartbeat timeout: {byzantine_node} missed {max_missed_heartbeats} checks"})
        events.append({"timestamp": detection_time + terrestrial_latency_sec, "event": f"System blacklists {byzantine_node} after divergence detected"})
        return {
            "token_id": token_id,
            "byzantine_node": byzantine_node,
            "scenario": "drop_all" if target_token_ids is None else "selective_drop",
            "arrival_times": arrival_times,
            "converged_honest_nodes": all(not nodes[n].accepts(token_id) for n in honest_nodes),
            "byzantine_converged": nodes[byzantine_node].accepts(token_id),
            "detection_time_sec": detection_time - revocation_time,
            "missed_heartbeats": max_missed_heartbeats,
            "events": sorted(events, key=lambda e: e["timestamp"]),
        }
    else:
        # Byzantine node behaves honestly for this token
        delay = crdt_cfg.get("leo_satellite_latency_ms", 600.0) / 1000.0 if byzantine_node == "GS-C" else terrestrial_latency_sec
        arrival = revocation_time + delay
        arrival_times[byzantine_node] = arrival
        events.append({"timestamp": arrival, "event": f"{byzantine_node} receives revocation (honest behavior)"})
        for n in arrival_times:
            nodes[n].revoke(token_id)
        return {
            "token_id": token_id,
            "byzantine_node": byzantine_node,
            "scenario": "honest",
            "arrival_times": arrival_times,
            "converged_honest_nodes": True,
            "byzantine_converged": True,
            "detection_time_sec": 0.0,
            "missed_heartbeats": 0,
            "events": sorted(events, key=lambda e: e["timestamp"]),
        }

