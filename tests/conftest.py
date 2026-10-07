"""Shared fixtures."""

from __future__ import annotations

import pytest

from aerosync.ai.model import Task, static_cost_fn
from aerosync.sim.airport import build_airport
from aerosync.sim.scenarios import load_scenario


@pytest.fixture(scope="session")
def airport():
    return build_airport()


@pytest.fixture(scope="session")
def cost_fn(airport):
    return static_cost_fn(airport)


def scenario_tasks(name: str, seed: int = 42) -> list[Task]:
    sc = load_scenario(name, seed)
    return [Task(id=f.id, size=f.size, eta=f.sched_arr, sched_dep=f.sched_dep,
                 min_turn=f.min_turn, pref_terminal=f.pref_terminal, pax=f.pax)
            for f in sc.flights]
