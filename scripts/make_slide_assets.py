"""Build everything the slide deck needs in ``docs/slide_assets/`` from real runs.

Prerequisite: ``aerosync report --scenario all --out results/`` (writes results/summary.json).
Usage: ``python scripts/make_slide_assets.py``
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

from rich.console import Console

from aerosync import config
from aerosync.ai.csp import GateCSP
from aerosync.ai.model import Task, static_cost_fn
from aerosync.cli import main as cli
from aerosync.controllers import make_controller
from aerosync.experiments import run_single
from aerosync.metrics.engine import LOWER_IS_BETTER, METRIC_LABELS
from aerosync.sim.airport import build_airport
from aerosync.sim.environment import Simulation
from aerosync.sim.events import make_event
from aerosync.sim.scenarios import SCENARIO_NAMES, load_scenario
from aerosync.viz import charts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "slide_assets"
TERM_WIDTH = 150

# The disruption used for the before/after Gantt (same as the demo `disrupt` command).
DISRUPT = {"scenario": "normal_day", "gate": "G4", "at": 30, "seed": 42}


def metrics_summary(summary: dict) -> None:
    out = {
        "source": "results/summary.json",
        "command": summary["command"],
        "seeds": summary["seeds"],
        "aggregation": "mean and sample std over seeds",
        "metric_labels": METRIC_LABELS,
        "lower_is_better": LOWER_IS_BETTER,
        "results": summary["results"],
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (OUT / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "controller", "metric", "label", "mean", "std", "seeds",
                    "command"])
        for sc in SCENARIO_NAMES:
            for ctl in ("fcfs", "greedy", "aerosync"):
                for k, label in METRIC_LABELS.items():
                    v = summary["results"][sc][ctl][k]
                    w.writerow([sc, ctl, k, label, v["mean"], v["std"],
                                " ".join(map(str, summary["seeds"])), summary["command"]])


def bar_charts(summary: dict) -> None:
    for dest in (OUT / "charts", RESULTS / "charts"):
        for k in METRIC_LABELS:
            charts.metric_bars(summary, k, dest / f"bars_{k}.png")
        charts.overview_grid(summary, dest / "overview.png")


def gantt_charts() -> None:
    d = DISRUPT
    ev = make_event("gate_closure", d["at"], gate=d["gate"])
    t_ev = d["at"] * config.STEP_MIN
    before = run_single(d["scenario"], "aerosync", d["seed"])
    after = run_single(d["scenario"], "aerosync", d["seed"], extra_events=[ev])
    greedy = run_single(d["scenario"], "greedy", d["seed"], extra_events=[ev])
    moved = {row["flight"] for r in after.repairs
             if r["t"] == t_ev and "gate_closure" in r["trigger"] for row in r["changed"]}
    win = {"t0": max(0, t_ev - 90), "t1": t_ev + 420, "marker_t": t_ev}
    clock = config.fmt_clock(t_ev, before.start_hour)
    c = OUT / "charts"
    charts.gantt_chart(before, c / "gantt_before_disruption.png",
                       f"Before: {d['scenario']} (seed {d['seed']}), aerosync, no fault", **win)
    charts.gantt_chart(after, c / "gantt_after_disruption_aerosync.png",
                       f"After: {d['gate']} closes at {clock} - min-conflicts moves {len(moved)} "
                       f"flights; {after.metrics['schedule_conflicts']} conflicts at arrival",
                       highlight=moved, **win)
    charts.gantt_chart(greedy, c / "gantt_after_disruption_greedy.png",
                       f"Same fault under greedy (no repair): "
                       f"{greedy.metrics['schedule_conflicts']} conflicts at arrival", **win)


def diagrams() -> dict:
    dg = OUT / "diagrams"
    sc = load_scenario("rush_hour", 42)
    sim = Simulation(sc, make_controller("aerosync", 42))
    sim.run(steps=20, drain=False)
    veh = sim.vehicles["TUG1"]
    route = veh.plan_astar(sim.airport, "G7", sim.t, sim.edge_multiplier, start="HANGAR")
    charts.topology_diagram(
        dg / "airport_topology.png", route=route.path, airport=sim.airport,
        title="Apron topology with an A* route (HANGAR -> G7)",
        subtitle=(f"Tug route at {sim.clock()} in rush_hour (seed 42): "
                  f"{' -> '.join(route.path)}; {route.minutes:.2f} min, "
                  f"{route.expanded} nodes expanded."))
    charts.topology_diagram(dg / "airport_topology_plain.png", airport=sim.airport,
                            title="Airport apron topology: 12 gates (T1, T2), 4 remote stands")
    charts.architecture_diagram(dg / "architecture.png")
    ap = build_airport()
    stands = {s: ap.stands[s] for s in ("G3", "G5", "G9", "G11")}
    tasks = [Task("A", "N", 0, 60, 50, "T1", 150), Task("B", "N", 10, 75, 50, "T1", 160),
             Task("C", "W", 20, 140, 100, "T2", 320), Task("D", "N", 80, 140, 50, "T2", 140),
             Task("E", "W", 40, 160, 100, "T2", 300)]
    csp = GateCSP(tasks, stands, static_cost_fn(ap), max_hold=30, trace=True)
    initial = {t.id: len(csp.domains[t.id]) for t in tasks}
    res = csp.solve()
    charts.csp_example_diagram(res.trace, [t.__dict__ for t in tasks], res.assignment,
                               dg / "csp_worked_example.png", initial)
    return {"route": route, "csp": res, "route_clock": sim.clock()}


def recording_console() -> Console:
    return Console(record=True, width=TERM_WIDTH, force_terminal=True, color_system="truecolor",
                   file=io.StringIO())


def terminal_captures() -> list[tuple[str, str]]:
    shots: list[tuple[str, str]] = []
    jobs = [
        ("simulate", "aerosync simulate --scenario rush_hour --controller aerosync --seed 42",
         lambda: cli.simulate(scenario="rush_hour", controller="aerosync", seed=42, steps=None,
                              live=False, delay=0.0, save=None)),
        ("disrupt", "aerosync disrupt --scenario normal_day --event gate_closure --gate G4 "
                    "--at 30",
         lambda: cli.disrupt(scenario="normal_day", event="gate_closure", at=30, gate="G4",
                             flight=None, vehicle=None, minutes=45, duration=None,
                             controller="aerosync", seed=42)),
        ("explain", "aerosync explain --flight AI302",
         lambda: cli.explain(flight="AI302", scenario="rush_hour", controller="aerosync",
                             seed=42)),
        ("route", "aerosync route --from HANGAR --to G7 --seed 42",
         lambda: cli.route(src="HANGAR", dst="G7", seed=42, scenario="rush_hour", kind="tug",
                           at=20)),
        ("compare", "aerosync compare --scenario storm_disruption --seed 42",
         lambda: cli.compare_cmd(scenario="storm_disruption", seed=42,
                                 out=Path("results/compare"))),
        ("gantt", "aerosync gantt --scenario rush_hour --controller aerosync",
         lambda: cli.gantt_cmd(scenario="rush_hour", controller="aerosync", seed=42,
                               resolution=None)),
    ]
    for name, cmd, fn in jobs:
        con = recording_console()
        cli.console = con
        con.print(f"[bold #1baf7a]$[/] [bold]{cmd}[/]")
        fn()
        text = con.export_text(clear=False)
        (OUT / "screenshots" / f"terminal_{name}.txt").write_text(f"$ {cmd}\n" + text.split(
            "\n", 1)[1], encoding="utf-8")
        svg = con.export_svg(title=cmd)
        shots.append((name, svg))
        if name == "explain":
            (OUT / "explain_example.txt").write_text(text, encoding="utf-8")
    return shots


def svgs_to_png(shots: list[tuple[str, str]]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1800, "height": 1000}, device_scale_factor=1)
        for name, svg in shots:
            page.set_content(f"<html><body style='margin:0;background:#fff'>{svg}</body></html>")
            el = page.locator("svg").first
            box = el.bounding_box()
            assert box is not None
            scale = 1700 / box["width"]
            page.evaluate(f"""() => {{ const s = document.querySelector('svg');
                s.setAttribute('width', {box['width'] * scale});
                s.setAttribute('height', {box['height'] * scale}); }}""")
            el.screenshot(path=str(OUT / "screenshots" / f"terminal_{name}.png"))
        browser.close()


def results_table(summary: dict) -> str:
    keys = ["avg_walk_m", "avg_dep_delay_min", "gate_utilisation_pct", "remote_ratio_pct",
            "schedule_conflicts", "reassignments", "avg_vehicle_trip_min", "compute_ms"]
    short = ["Walk (m)", "Dep. delay (min)", "Gate util. (%)", "Remote (%)", "Conflicts",
             "Reassign.", "Vehicle trip (min)", "Compute (ms)"]
    lines = ["| Scenario | Controller | " + " | ".join(short) + " |",
             "|---|---|" + "---:|" * len(keys)]
    for sc in SCENARIO_NAMES:
        for ctl in ("fcfs", "greedy", "aerosync"):
            r = summary["results"][sc][ctl]
            cells = []
            for k in keys:
                vals = [summary["results"][sc][c][k]["mean"] for c in ("fcfs", "greedy",
                                                                       "aerosync")]
                best = min(vals) if LOWER_IS_BETTER[k] else max(vals)
                v = r[k]["mean"]
                s = f"{v:,.1f}"
                cells.append(f"**{s}**" if v == best else s)
            name = f"**{ctl}**" if ctl == "aerosync" else ctl
            lines.append(f"| {sc} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(a: float, b: float) -> str:
    return f"{100 * (b - a) / a:+.0f}%"


def facts(summary: dict, extra: dict) -> str:
    w = config.DEFAULT_WEIGHTS
    R = summary["results"]

    def m(sc: str, c: str, k: str) -> float:
        return R[sc][c][k]["mean"]

    walk_lines = [f"  - {sc}: aerosync {m(sc, 'aerosync', 'avg_walk_m'):.0f} m vs greedy "
                  f"{m(sc, 'greedy', 'avg_walk_m'):.0f} m "
                  f"({pct(m(sc, 'greedy', 'avg_walk_m'), m(sc, 'aerosync', 'avg_walk_m'))}) "
                  f"and fcfs {m(sc, 'fcfs', 'avg_walk_m'):.0f} m" for sc in SCENARIO_NAMES]
    delay_lines = [f"  - {sc}: aerosync {m(sc, 'aerosync', 'avg_dep_delay_min'):.1f} min vs "
                   f"greedy {m(sc, 'greedy', 'avg_dep_delay_min'):.1f} / fcfs "
                   f"{m(sc, 'fcfs', 'avg_dep_delay_min'):.1f} min" for sc in SCENARIO_NAMES]
    conf_lines = [f"  - {sc}: aerosync {m(sc, 'aerosync', 'schedule_conflicts'):.1f} vs greedy "
                  f"{m(sc, 'greedy', 'schedule_conflicts'):.1f} / fcfs "
                  f"{m(sc, 'fcfs', 'schedule_conflicts'):.1f}" for sc in SCENARIO_NAMES]
    trade_lines = [f"  - {sc}: reassignments aerosync {m(sc, 'aerosync', 'reassignments'):.1f}"
                   f" vs greedy {m(sc, 'greedy', 'reassignments'):.1f}; compute "
                   f"{m(sc, 'aerosync', 'compute_ms'):,.0f} ms vs "
                   f"{m(sc, 'greedy', 'compute_ms'):,.0f} ms" for sc in SCENARIO_NAMES]
    route = extra["route"]
    csp = extra["csp"]
    return f"""# AeroSync - facts for the slides

Every number below is produced by code in this repository. Benchmark numbers come from
`results/summary.json`, generated by:

    {summary['command']}

Seeds: {summary['seeds']} (mean over seeds; std in `metrics_summary.csv`).

## Airport and time model

- 12 contact gates G1-G12 in two terminals (T1: G1-G6, T2: G7-G12) + 4 remote stands R1-R4.
- Wide-body capable: G3, G4, G9, G10 and all remote stands (8 stands in total).
- Apron/taxi graph: {len(build_airport().nodes)} nodes, {len(build_airport().edges())} edges
  (gates, apron road, service taxiway, remote road, DEPOT, HANGAR).
- Timestep: {config.STEP_MIN} min. Buffer between flights at a stand: {config.BUFFER_MIN} min.
- Look-ahead (gate request / announcement): {config.LOOKAHEAD_MIN} min before ETA.
- Max hold considered by planners: {config.MAX_HOLD_MIN} min.
- Minimum turnaround: {config.MIN_TURN_FRACTION:.0%} of scheduled ground time.
- Remote-stand walking equivalent: {config.REMOTE_WALK_M:.0f} m.
- Vehicle speeds (m/min): {config.VEHICLE_SPEED_M_PER_MIN}; service times (min):
  {config.SERVICE_MIN}.
- Congestion: edge cost x (weather factor) x (1 + {config.MOVEMENT_CONGESTION} x aircraft
  movements within +/-{config.MOVEMENT_WINDOW_MIN} min at an adjacent stand).

## Gate-agent bid (Contract Net)

    bid = w1*walk_m*(pax/150) + w2*terminal_mismatch + w3*remote + w4*buffer_risk
        + w5*(alpha * neighbour_load) + w6*hold_min + w7*(reassignment + displaced)

- w1 = {w.walk}, w2 = {w.mismatch}, w3 = {w.remote}, w4 = {w.buffer_risk}, w5 = {w.neighbour},
  w6 = {w.delay}, w7 = {w.reassign}
- alpha (coordination weight) = {config.ALPHA}
- neighbour_load = mean share of [start-60, end+60] occupied at 1-hop neighbour stands
  (known only through their STATE_UPDATE messages).
- buffer_risk = max(0, (30 - slack_min) / 30), slack measured beyond the 15-min buffer.
- Award rule: minimum bid, ties broken by stand id. Infeasible stands REFUSE.

## Algorithms

- CSP: variables = flights, values = (stand, hold); backtracking + MRV + degree heuristic +
  forward checking + least-cost value ordering. Node limit {config.CSP_NODE_LIMIT:,}.
- Worked example (diagrams/csp_worked_example.png): 5 flights, 4 stands, solved in
  {csp.nodes} nodes with {csp.backtracks} backtracks and {csp.pruned} values pruned.
- Min-conflicts repair: max {config.MIN_CONFLICTS_MAX_STEPS} steps, patience 150,
  random-walk noise 0.2; announced flights are protected (picked last, 3x move penalty);
  falls back to a CSP re-plan of the movable flights if local search stalls.
- A*: edge cost = length / speed x congestion; h(n) = euclid(n, goal) / speed (admissible).
  Example: tug HANGAR -> G7 at {extra['route_clock']} (rush_hour, seed 42):
  {' -> '.join(route.path)}, {route.minutes:.2f} min, {route.expanded} nodes expanded.

## Metric definitions

- Walk: passenger-weighted mean of walk(stand, preferred terminal exit).
- Departure delay: mean of (actual off-block - scheduled departure) in minutes.
- Gate utilisation: occupied minutes on the 12 contact gates / (12 x window).
- Schedule conflicts: flights whose planned stand was not free when they were ready to
  block on.
- Reassignments: stand changes after the gate was announced (T-60); an ETA slip beyond the
  look-ahead withdraws the announcement (same rule for every controller).
- Decision compute time: total wall-clock ms spent inside the controller per run
  (machine-dependent, not reproducible to the millisecond).

## Headline findings (mean over seeds {summary['seeds']})

- Passenger walking is lowest with aerosync in every scenario:
{chr(10).join(walk_lines)}
- Departure delay: aerosync is on par in calm scenarios and clearly better under disruption:
{chr(10).join(delay_lines)}
- Conflicts at arrival (planned stand not free) nearly disappear with repair:
{chr(10).join(conf_lines)}
- Trade-offs (honest): aerosync changes more announced gates and needs more compute:
{chr(10).join(trade_lines)}
"""


def main() -> None:
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        raise SystemExit("results/summary.json missing: run `aerosync report` first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for sub in ("charts", "diagrams", "screenshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    metrics_summary(summary)
    bar_charts(summary)
    gantt_charts()
    extra = diagrams()
    shots = terminal_captures()
    svgs_to_png(shots)
    (OUT / "facts.md").write_text(facts(summary, extra), encoding="utf-8")
    (OUT / "results_table.md").write_text(results_table(summary) + "\n", encoding="utf-8")
    shutil.copy(OUT / "results_table.md", RESULTS / "summary_table.md")
    print("slide assets written to", OUT)


if __name__ == "__main__":
    main()
