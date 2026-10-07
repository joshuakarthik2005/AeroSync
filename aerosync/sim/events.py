"""Fault / disruption events injected into a simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EVENT_KINDS = ("gate_closure", "flight_delay", "vehicle_breakdown", "comms_loss", "weather")


@dataclass
class Event:
    """A disruption that becomes known to the agents at step ``at``.

    ``kind``-specific parameters:

    * ``gate_closure``: ``gate``, ``duration`` (minutes, ``None`` = until end of day)
    * ``flight_delay``: ``flight``, ``minutes``
    * ``vehicle_breakdown``: ``vehicle``, ``duration``
    * ``comms_loss``: ``gate``, ``duration``
    * ``weather``: ``factor`` (edge-cost multiplier), ``duration``
    """

    kind: str
    at: int  # simulation step
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}; expected one of {EVENT_KINDS}")

    def describe(self) -> str:
        p = self.params
        if self.kind == "gate_closure":
            dur = p.get("duration")
            return f"gate {p['gate']} closed" + (f" for {dur} min" if dur else " until end of day")
        if self.kind == "flight_delay":
            return f"flight {p['flight']} delayed by {p['minutes']} min"
        if self.kind == "vehicle_breakdown":
            return f"vehicle {p['vehicle']} broke down for {p.get('duration', 60)} min"
        if self.kind == "comms_loss":
            return f"gate {p['gate']} lost comms for {p.get('duration', 30)} min"
        return f"weather: apron edge costs x{p.get('factor', 1.4)} for {p.get('duration', 60)} min"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Event:
        d = dict(d)
        kind = d.pop("kind")
        at = int(d.pop("at"))
        params = d.pop("params", {})
        params.update(d)
        return Event(kind=kind, at=at, params=params)


def make_event(kind: str, at: int, **params: Any) -> Event:
    """Convenience constructor used by the CLI and web app."""
    return Event(kind=kind, at=at, params={k: v for k, v in params.items() if v is not None})
