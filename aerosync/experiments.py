"""Experiment runner: single runs, controller comparisons and batch reports."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aerosync.config import DEFAULT_WEIGHTS, STEP_MIN
from aerosync.controllers import CONTROLLER_NAMES, make_controller
from aerosync.metrics.engine import METRIC_LABELS
from aerosync.sim.environment import SimResult, Simulation
from aerosync.sim.events import Event
from aerosync.sim.scenarios import SCENARIO_NAMES, load_scenario

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = Path("results")


def run_single(scenario: str, controller: str, seed: int = 42,
               extra_events: list[Event] | None = None, steps: int | None = None) -> SimResult:
    """Run one scenario under one controller and return the full result."""
    sc = load_scenario(scenario, seed)
    sim = Simulation(sc, make_controller(controller, seed), extra_events=extra_events)
    return sim.run(steps=steps)


def compare(scenario: str, seed: int = 42, extra_events: list[Event] | None = None,
            controllers: list[str] | None = None) -> dict[str, SimResult]:
    """Run every controller on the identical seeded scenario."""
    return {c: run_single(scenario, c, seed, extra_events) for c in controllers or CONTROLLER_NAMES}


def run_report(scenarios: list[str] | None = None, seeds: list[int] | None = None,
               out: Path = RESULTS_DIR, progress: Callable[[str], None] | None = None,
               command: str = "") -> dict[str, Any]:
    """Full batch experiment: all scenarios x all controllers x all seeds.

    Writes ``runs/*.json`` (one per run, without the message log), ``runs.csv``,
    ``summary.json``/``summary.csv`` (mean and std over seeds) and returns the summary.
    """
    scenarios = scenarios or SCENARIO_NAMES
    seeds = seeds or DEFAULT_SEEDS
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        for seed in seeds:
            for ctl in CONTROLLER_NAMES:
                if progress:
                    progress(f"{sc} / {ctl} / seed {seed}")
                res = run_single(sc, ctl, seed)
                path = out / "runs" / f"{sc}_{ctl}_seed{seed}.json"
                d = res.to_dict(include_messages=False)
                path.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
                rows.append({"scenario": sc, "controller": ctl, "seed": seed, **res.metrics})
    _write_csv(out / "runs.csv", rows)
    summary = summarise(rows, seeds, command)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat = []
    for sc, ctls in summary["results"].items():
        for ctl, ms in ctls.items():
            row: dict[str, Any] = {"scenario": sc, "controller": ctl}
            for k, v in ms.items():
                row[f"{k}_mean"] = v["mean"]
                row[f"{k}_std"] = v["std"]
            flat.append(row)
    _write_csv(out / "summary.csv", flat)
    return summary


def summarise(rows: list[dict[str, Any]], seeds: list[int], command: str = "") -> dict[str, Any]:
    """Mean/std of every metric per (scenario, controller) over seeds."""
    metric_keys = [k for k in rows[0] if k not in ("scenario", "controller", "seed")]
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for r in rows:
        results.setdefault(r["scenario"], {}).setdefault(r["controller"], {})
    for sc, ctls in results.items():
        for ctl in ctls:
            sel = [r for r in rows if r["scenario"] == sc and r["controller"] == ctl]
            for k in metric_keys:
                xs = [float(r[k]) for r in sel]
                ctls[ctl][k] = {
                    "mean": round(statistics.mean(xs), 2),
                    "std": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
                }
    return {
        "seeds": seeds,
        "command": command,
        "timestep_min": STEP_MIN,
        "weights": DEFAULT_WEIGHTS.as_dict(),
        "headline_metrics": list(METRIC_LABELS),
        "results": results,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_result(res: SimResult, out: Path) -> Path:
    """Persist one run as JSON and its flight table as CSV."""
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{res.scenario}_{res.controller}_seed{res.seed}"
    jp = out / f"{stem}.json"
    jp.write_text(json.dumps(res.to_dict(include_messages=False), separators=(",", ":")),
                  encoding="utf-8")
    flights = [{k: v for k, v in f.items() if k != "history"} for f in res.flights]
    _write_csv(out / f"{stem}_flights.csv", flights)
    return jp
