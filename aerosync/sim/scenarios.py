"""Scenario loading. Scenarios are JSON files in the top-level ``scenarios/`` folder."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aerosync.config import STEP_MIN
from aerosync.sim.events import Event
from aerosync.sim.flights import Flight, generate_flights

SCENARIO_DIR = Path(os.environ.get("AEROSYNC_SCENARIOS",
                                   Path(__file__).resolve().parents[2] / "scenarios"))
SCENARIO_NAMES = ["normal_day", "rush_hour", "storm_disruption", "gate_failure"]


@dataclass
class Scenario:
    """A fully instantiated (seeded) scenario."""

    name: str
    description: str
    start_hour: int
    steps: int
    seed: int
    flights: list[Flight]
    events: list[Event] = field(default_factory=list)
    weather_factor: float = 1.0
    fleet: dict[str, int] = field(default_factory=lambda: {"tug": 4, "fuel": 4, "bus": 3})
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def window_min(self) -> int:
        return self.steps * STEP_MIN


def scenario_path(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".json" and p.exists():
        return p
    return SCENARIO_DIR / f"{name}.json"


def list_scenarios() -> list[dict[str, Any]]:
    """Summaries of every scenario JSON in :data:`SCENARIO_DIR`."""
    out = []
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({
            "name": d["name"],
            "description": d["description"],
            "window": f"{d['start_hour']:02d}:00 + {d['steps'] * STEP_MIN // 60} h",
            "steps": d["steps"],
            "flights": d.get("generator", {}).get("n_flights", len(d.get("flights", []))),
            "events": len(d.get("events", [])),
        })
    order = {n: i for i, n in enumerate(SCENARIO_NAMES)}
    out.sort(key=lambda s: (order.get(s["name"], 99), s["name"]))
    return out


def scenario_from_dict(d: dict[str, Any], seed: int) -> Scenario:
    """Instantiate a scenario dict with a seed (deterministic)."""
    if "flights" in d:
        flights = [Flight.from_dict(f) for f in d["flights"]]
    else:
        flights = generate_flights(d["generator"], seed, d.get("featured_flights"))
    events = [Event.from_dict(e) for e in d.get("events", [])]
    return Scenario(
        name=d["name"],
        description=d.get("description", ""),
        start_hour=int(d.get("start_hour", 6)),
        steps=int(d["steps"]),
        seed=seed,
        flights=flights,
        events=events,
        weather_factor=float(d.get("weather_factor", 1.0)),
        fleet=dict(d.get("fleet", {"tug": 4, "fuel": 4, "bus": 3})),
        raw=d,
    )


def load_scenario(name: str, seed: int = 42) -> Scenario:
    """Load ``scenarios/<name>.json`` (or a path) and instantiate it with ``seed``."""
    path = scenario_path(name)
    if not path.exists():
        names = ", ".join(s["name"] for s in list_scenarios())
        raise FileNotFoundError(f"unknown scenario {name!r}; available: {names}")
    return scenario_from_dict(json.loads(path.read_text(encoding="utf-8")), seed)
