"""Seeded runs are reproducible."""

from __future__ import annotations

import pytest

from aerosync.experiments import run_single
from aerosync.sim.scenarios import load_scenario


def _strip(m):
    return {k: v for k, v in m.items() if k != "compute_ms"}


@pytest.mark.parametrize("controller", ["fcfs", "greedy", "aerosync"])
def test_same_seed_same_result(controller):
    a = run_single("storm_disruption", controller, 7)
    b = run_single("storm_disruption", controller, 7)
    assert a.flights == b.flights
    assert _strip(a.metrics) == _strip(b.metrics)
    assert [m["summary"] for m in a.messages] == [m["summary"] for m in b.messages]


def test_different_seed_different_schedule():
    a = [f.to_dict() for f in load_scenario("rush_hour", 1).flights]
    b = [f.to_dict() for f in load_scenario("rush_hour", 2).flights]
    assert a != b
    assert any(f["id"] == "AI302" for f in a)
