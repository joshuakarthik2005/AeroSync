"""Metrics engine: turns a :class:`SimResult` into comparable numbers.

All metrics are computed from what actually happened in the run (actual
on/off-block times, actual vehicle trips), never from the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aerosync.config import BUFFER_MIN
from aerosync.sim.airport import Airport, build_airport

if TYPE_CHECKING:
    from aerosync.sim.environment import SimResult

METRIC_LABELS: dict[str, str] = {
    "avg_walk_m": "Avg passenger walk (m)",
    "avg_dep_delay_min": "Avg departure delay (min)",
    "gate_utilisation_pct": "Contact-gate utilisation (%)",
    "remote_ratio_pct": "Remote-stand ratio (%)",
    "schedule_conflicts": "Schedule conflicts",
    "reassignments": "Reassignments",
    "avg_vehicle_trip_min": "Avg ground-vehicle trip (min)",
    "compute_ms": "Decision compute time (ms)",
}
"""Headline metrics in display order."""

LOWER_IS_BETTER: dict[str, bool] = {
    "avg_walk_m": True,
    "avg_dep_delay_min": True,
    "gate_utilisation_pct": False,
    "remote_ratio_pct": True,
    "schedule_conflicts": True,
    "reassignments": True,
    "avg_vehicle_trip_min": True,
    "compute_ms": True,
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def hard_violations(flights: list[dict[str, Any]], airport: Airport,
                    closures: list[dict[str, Any]] | None = None) -> int:
    """Count violated hard constraints in the *executed* schedule."""
    n = 0
    by_stand: dict[str, list[tuple[int, int]]] = {}
    for f in flights:
        if f["on_block"] is None or f["stand"] is None:
            continue
        if f["towed"]:
            continue  # its stand changed mid-turn; occupancy split across two stands
        st = airport.stands[f["stand"]]
        if not st.accepts(f["size"]):
            n += 1
        end = f["off_block"] if f["off_block"] is not None else f["on_block"] + 1
        by_stand.setdefault(f["stand"], []).append((f["on_block"], end))
        for c in closures or []:
            if c["stand"] == f["stand"] and c["start"] <= f["on_block"] < c["end"]:
                n += 1
    for items in by_stand.values():
        items.sort()
        for (_s1, e1), (s2, _e2) in zip(items, items[1:], strict=False):
            if s2 < e1 + BUFFER_MIN:
                n += 1
    return n


def compute_metrics(result: SimResult, airport: Airport | None = None) -> dict[str, float]:
    """Compute every metric for one run."""
    airport = airport or build_airport()
    flights = result.flights
    blocked = [f for f in flights if f["on_block"] is not None]
    finished = [f for f in flights if f["off_block"] is not None]
    pax_total = sum(f["pax"] for f in blocked)
    avg_walk = (sum(f["walk_m"] * f["pax"] for f in blocked) / pax_total) if pax_total else 0.0
    delays = [f["dep_delay"] for f in finished]
    window = result.window_min
    gates = airport.contact_gates
    occupied = 0
    for f in blocked:
        if f["stand"] in gates:
            end = f["off_block"] if f["off_block"] is not None else result.end_t
            occupied += max(0, min(end, window) - max(f["on_block"], 0))
    util = 100.0 * occupied / (len(gates) * window) if window else 0.0
    remote = sum(1 for f in blocked if f["remote"])
    trips = [j["trip_min"] for j in result.jobs]
    holds = [f["hold"] for f in blocked]
    return {
        "flights": len(flights),
        "flights_completed": len(finished),
        "avg_walk_m": round(avg_walk, 1),
        "avg_dep_delay_min": round(_mean(delays), 2),
        "gate_utilisation_pct": round(util, 1),
        "remote_ratio_pct": round(100.0 * remote / len(blocked), 1) if blocked else 0.0,
        "schedule_conflicts": int(result.counters.get("conflicts", 0)),
        "reassignments": int(sum(f["reassignments"] for f in flights)),
        "avg_vehicle_trip_min": round(_mean(trips), 2),
        "compute_ms": round(result.compute_ms, 1),
        "avg_hold_min": round(_mean(holds), 2),
        "on_time_pct": round(100.0 * sum(1 for d in delays if d <= 15) / len(delays), 1)
        if delays else 0.0,
        "terminal_mismatch_pct": round(
            100.0 * sum(1 for f in blocked if f["terminal_mismatch"]) / len(blocked), 1)
        if blocked else 0.0,
        "towing_movements": int(result.counters.get("towing", 0)),
        "hard_violations": hard_violations(flights, airport, result.closures),
    }
