"""Metrics on a hand-built mini scenario (numbers worked out by hand)."""

from __future__ import annotations

import pytest

from aerosync.controllers import make_controller
from aerosync.experiments import run_single
from aerosync.sim.environment import Simulation
from aerosync.sim.scenarios import SCENARIO_NAMES, scenario_from_dict

MINI = {
    "name": "mini",
    "description": "three flights, hand-checked",
    "start_hour": 8,
    "steps": 36,
    "fleet": {"tug": 3, "fuel": 3, "bus": 1},
    "flights": [
        {"id": "F1", "airline": "6E", "size": "N", "sched_arr": 0, "sched_dep": 60, "pax": 150,
         "pref_terminal": "T1"},
        {"id": "F2", "airline": "6E", "size": "N", "sched_arr": 0, "sched_dep": 60, "pax": 100,
         "pref_terminal": "T1"},
        {"id": "F3", "airline": "EK", "size": "W", "sched_arr": 30, "sched_dep": 150, "pax": 300,
         "pref_terminal": "T2"},
    ],
}


@pytest.fixture(scope="module")
def result():
    sc = scenario_from_dict(MINI, seed=0)
    return Simulation(sc, make_controller("fcfs", 0)).run()


def test_fcfs_assignment_is_first_feasible(result):
    stands = {f["id"]: f["stand"] for f in result.flights}
    assert stands == {"F1": "G1", "F2": "G2", "F3": "G3"}


def test_walking_distance_is_passenger_weighted(result, airport):
    # G1 -> T1 exit: 80 + |60-360| = 380 m; G2: 80 + |180-360| = 260 m;
    # G3 (T1 pier) -> T2 exit: 80 + |300-360| + 250 + |1160-360| = 1190 m.
    assert airport.walk_distance("G1", "T1") == 380
    assert airport.walk_distance("G2", "T1") == 260
    assert airport.walk_distance("G3", "T2") == 1190
    expected = (380 * 150 + 260 * 100 + 1190 * 300) / 550  # = 800.0
    assert result.metrics["avg_walk_m"] == pytest.approx(expected)


def test_delay_utilisation_and_counts(result):
    m = result.metrics
    assert [f["dep_delay"] for f in result.flights] == [0, 0, 0]
    assert m["avg_dep_delay_min"] == 0
    # Occupied contact-gate minutes: 60 + 60 + 120 = 240 over 12 gates x 180 min.
    assert m["gate_utilisation_pct"] == pytest.approx(round(100 * 240 / (12 * 180), 1))
    assert m["remote_ratio_pct"] == 0
    assert m["schedule_conflicts"] == 0 and m["reassignments"] == 0
    assert m["terminal_mismatch_pct"] == pytest.approx(round(100 / 3, 1))
    assert m["hard_violations"] == 0
    trips = [j["trip_min"] for j in result.jobs]
    assert m["avg_vehicle_trip_min"] == pytest.approx(round(sum(trips) / len(trips), 2))


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_every_controller_respects_hard_constraints(scenario):
    for ctl in ("fcfs", "greedy", "aerosync"):
        res = run_single(scenario, ctl, 42)
        assert res.metrics["hard_violations"] == 0, ctl
        assert res.metrics["flights_completed"] == res.metrics["flights"], ctl
