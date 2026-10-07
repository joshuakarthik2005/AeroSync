"""AeroSync terminal CLI (``aerosync ...`` or ``python -m aerosync ...``)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from aerosync.config import (
    ALPHA,
    BUFFER_MIN,
    DEFAULT_WEIGHTS,
    STEP_MIN,
    VEHICLE_SPEED_M_PER_MIN,
    fmt_clock,
)
from aerosync.controllers import CONTROLLER_NAMES, make_controller
from aerosync.experiments import DEFAULT_SEEDS, compare, run_report, run_single, save_result
from aerosync.sim.environment import Simulation
from aerosync.sim.events import EVENT_KINDS, make_event
from aerosync.sim.scenarios import SCENARIO_NAMES, list_scenarios, load_scenario
from aerosync.viz import terminal as tv

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="[bold]AeroSync[/] - multi-agent airport gate assignment and ground "
                       "operations coordinator.")
console = Console()

ScenarioOpt = Annotated[str, typer.Option("--scenario", "-s", help="Scenario name.")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Random seed (deterministic).")]
ControllerOpt = Annotated[str, typer.Option("--controller", "-c",
                                            help="fcfs | greedy | aerosync")]


def _check_controller(name: str) -> None:
    if name not in CONTROLLER_NAMES:
        raise typer.BadParameter(f"controller must be one of {CONTROLLER_NAMES}")


def _header(title: str) -> None:
    console.rule(f"[bold #2a78d6]{title}")


@app.command()
def scenarios() -> None:
    """List the available scenarios."""
    _header("Scenarios")
    t = Table(header_style="bold white on #1d2733")
    for c in ("Name", "Window", "Steps", "Flights", "Scripted faults", "Description"):
        t.add_column(c, overflow="fold")
    for s in list_scenarios():
        t.add_row(f"[bold]{s['name']}[/]", s["window"], str(s["steps"]), str(s["flights"]),
                  str(s["events"]), s["description"])
    console.print(t)
    console.print(f"Timestep {STEP_MIN} min, buffer {BUFFER_MIN} min, alpha {ALPHA}. "
                  "All scenarios are deterministic for a given --seed.")


def _live_table(sim: Simulation) -> Group:
    t = Table(title=f"{sim.scenario.name} / {sim.controller.name} - {sim.clock()} "
                    f"(step {sim.step_index}/{sim.scenario.steps})",
              header_style="bold white on #1d2733", expand=False)
    for c in ("Stand", "Status", "Flight", "Next"):
        t.add_column(c)
    for sid in sim.airport.stands:
        occ = sim.occupant[sid]
        nxt = sorted((fs.planned_start(), fs.id) for fs in sim.states.values()
                     if fs.stand == sid and fs.on_block is None)
        status = ("[red]CLOSED[/]" if sim.is_closed(sid, sim.t)
                  else "[green]occupied[/]" if occ else "free")
        t.add_row(sid, status, occ or "",
                  f"{nxt[0][1]} @ {sim.clock(nxt[0][0])}" if nxt else "")
    msgs = Text()
    for m in sim.bus.log[-8:]:
        msgs.append(f"{sim.clock(m.t)} {m.type.value:<12} {m.sender:>8} -> {m.receiver:<10} "
                    f"{m.summary()[:70]}\n", style="#52514e")
    return Group(t, Panel(msgs, title="Agent message log (latest)", border_style="#2a78d6"))


@app.command()
def simulate(scenario: ScenarioOpt = "rush_hour", controller: ControllerOpt = "aerosync",
             seed: SeedOpt = 42,
             steps: Annotated[int | None, typer.Option(help="Steps to run (default: whole "
                                                            "window + drain).")] = None,
             live: Annotated[bool, typer.Option("--live", help="Animate the run.")] = False,
             delay: Annotated[float, typer.Option(help="Seconds per step with --live.")] = 0.04,
             save: Annotated[Path | None, typer.Option(help="Write JSON/CSV here.")] = None
             ) -> None:
    """Run one scenario under one controller and print the metrics."""
    _check_controller(controller)
    _header(f"simulate {scenario} with {controller} (seed {seed})")
    sc = load_scenario(scenario, seed)
    sim = Simulation(sc, make_controller(controller, seed))
    t0 = time.perf_counter()
    if live:
        sim.setup()
        n = steps or sc.steps
        with Live(_live_table(sim), console=console, refresh_per_second=20) as lv:
            for _ in range(n):
                sim.step()
                lv.update(_live_table(sim))
                time.sleep(delay)
        res = sim.run(steps=0, drain=False) if steps else _drain(sim)
    else:
        res = sim.run(steps=steps)
    wall = time.perf_counter() - t0
    console.print(tv.run_summary(res))
    console.print(tv.metrics_table({controller: res.metrics}, title="Metrics"))
    extra = res.metrics
    console.print(f"Avg hold {extra['avg_hold_min']} min | on-time (<=15 min) "
                  f"{extra['on_time_pct']}% | terminal mismatch {extra['terminal_mismatch_pct']}%"
                  f" | wall time {wall:.2f} s")
    if save:
        p = save_result(res, save)
        console.print(f"Saved {p}")


def _drain(sim: Simulation):  # type: ignore[no-untyped-def]
    while not sim.all_done() and sim.step_index < sim.scenario.steps + 96:
        sim.step()
    return sim.result()


@app.command(name="compare")
def compare_cmd(scenario: ScenarioOpt = "storm_disruption", seed: SeedOpt = 42,
                out: Annotated[Path, typer.Option(help="Output folder.")] = Path("results")
                ) -> None:
    """Run fcfs, greedy and aerosync on the same seeded scenario and compare."""
    from aerosync.viz.charts import compare_chart

    _header(f"compare controllers on {scenario} (seed {seed})")
    with console.status("Running 3 controllers..."):
        results = compare(scenario, seed)
    table = {c: r.metrics for c, r in results.items()}
    console.print(tv.metrics_table(table, title=f"{scenario} - seed {seed}"))
    for r in results.values():
        save_result(r, out)
    chart = compare_chart(table, f"{scenario} (seed {seed}): fcfs vs greedy vs aerosync",
                          out / "charts" / f"compare_{scenario}_seed{seed}.png")
    a, g = table["aerosync"], table["greedy"]
    console.print(Panel(
        f"AeroSync vs greedy: walk {a['avg_walk_m']:.0f} vs {g['avg_walk_m']:.0f} m, "
        f"departure delay {a['avg_dep_delay_min']:.1f} vs {g['avg_dep_delay_min']:.1f} min, "
        f"conflicts {a['schedule_conflicts']} vs {g['schedule_conflicts']}, reassignments "
        f"{a['reassignments']} vs {g['reassignments']}.", title="Headline", border_style="#1aa37a"))
    console.print(f"Saved per-run JSON/CSV to {out}/ and chart {chart}")


@app.command()
def disrupt(scenario: ScenarioOpt = "normal_day",
            event: Annotated[str, typer.Option(help=f"One of {', '.join(EVENT_KINDS)}")]
            = "gate_closure",
            at: Annotated[int, typer.Option(help="Step at which the fault happens.")] = 30,
            gate: Annotated[str | None, typer.Option(help="Gate for closure/comms loss.")] = None,
            flight: Annotated[str | None, typer.Option(help="Flight for flight_delay.")] = None,
            vehicle: Annotated[str | None, typer.Option(help="Vehicle for breakdown.")] = None,
            minutes: Annotated[int, typer.Option(help="Delay minutes.")] = 45,
            duration: Annotated[int | None, typer.Option(help="Fault duration (min).")] = None,
            controller: ControllerOpt = "aerosync", seed: SeedOpt = 42) -> None:
    """Inject a fault and show how the agents repair the schedule."""
    _check_controller(controller)
    params: dict[str, object] = {}
    if event == "gate_closure":
        params = {"gate": gate or "G4", "duration": duration}
    elif event == "flight_delay":
        params = {"flight": flight or "AI302", "minutes": minutes}
    elif event == "vehicle_breakdown":
        params = {"vehicle": vehicle or "FUEL1", "duration": duration or 60}
    elif event == "comms_loss":
        params = {"gate": gate or "G4", "duration": duration or 30}
    elif event == "weather":
        params = {"factor": 1.5, "duration": duration or 90}
    else:
        raise typer.BadParameter(f"event must be one of {EVENT_KINDS}")
    ev = make_event(event, at, **params)
    _header(f"disrupt {scenario}: {ev.describe()} at step {at}")
    t_ev = at * STEP_MIN
    with console.status("Running baseline day and disrupted day..."):
        base = run_single(scenario, controller, seed)
        hit = run_single(scenario, controller, seed, extra_events=[ev])
        others = {c: run_single(scenario, c, seed, extra_events=[ev])
                  for c in CONTROLLER_NAMES if c != controller}
    console.print(f"Fault time: [bold]{fmt_clock(t_ev, hit.start_hour)}[/] - {ev.describe()}")
    console.print(tv.repair_view(hit, since=t_ev))
    moved = {row["flight"] for r in hit.repairs if r["t"] >= t_ev for row in r["changed"]}
    lo = max(0, t_ev - 60)
    hi = t_ev + 240
    console.print(tv.gantt(base, t0=lo, t1=hi, resolution=5))
    console.print(Text("  ^ BEFORE: same seed, no fault", style="bold"))
    console.print(tv.gantt(hit, t0=lo, t1=hi, resolution=5, highlight=moved))
    console.print(Text("  ^ AFTER: fault injected (yellow = moved by repair, x = closed)",
                       style="bold"))
    table = {f"{controller} (no fault)": base.metrics, controller: hit.metrics}
    table.update({c: r.metrics for c, r in others.items()})
    console.print(tv.metrics_table(table, title="Impact of the fault"))


@app.command()
def explain(flight: Annotated[str, typer.Option("--flight", "-f", help="Flight id.")] = "AI302",
            scenario: ScenarioOpt = "rush_hour", controller: ControllerOpt = "aerosync",
            seed: SeedOpt = 42) -> None:
    """Show the bids, the winner and the reasoning behind one assignment."""
    _check_controller(controller)
    _header(f"explain {flight} ({scenario}, {controller}, seed {seed})")
    res = run_single(scenario, controller, seed)
    console.print(tv.explain_view(res, flight))
    w = DEFAULT_WEIGHTS
    console.print(Text(
        f"bid = {w.walk}*walk_m*pax/150 + {w.mismatch}*terminal_mismatch + {w.remote}*remote + "
        f"{w.buffer_risk}*buffer_risk + {w.neighbour}*(alpha={ALPHA} x neighbour_load) + "
        f"{w.delay}*hold_min + {w.reassign}*(reassignment + displaced)", style="#52514e"))


@app.command()
def route(src: Annotated[str, typer.Option("--from", help="Start node.")] = "HANGAR",
          dst: Annotated[str, typer.Option("--to", help="Goal node.")] = "G7",
          seed: SeedOpt = 42, scenario: ScenarioOpt = "rush_hour",
          kind: Annotated[str, typer.Option(help="tug | fuel | bus")] = "tug",
          at: Annotated[int, typer.Option(help="Step whose traffic is used.")] = 24) -> None:
    """Plan an A* route on the live apron graph and show its cost breakdown."""
    from aerosync.agents.vehicle_agent import GroundVehicleAgent
    from aerosync.ai.astar import dijkstra

    _header(f"A* route {src} -> {dst} ({kind}, traffic of {scenario} at step {at})")
    sc = load_scenario(scenario, seed)
    sim = Simulation(sc, make_controller("aerosync", seed))
    sim.run(steps=at, drain=False)
    t = sim.t
    veh = GroundVehicleAgent("ROUTE", kind, src)
    for n in (src, dst):
        if n not in sim.airport.nodes:
            raise typer.BadParameter(f"unknown node {n}; try one of "
                                     f"{', '.join(sorted(sim.airport.nodes))}")
    r = veh.plan_astar(sim.airport, dst, t, sim.edge_multiplier)
    speed = VEHICLE_SPEED_M_PER_MIN[kind]

    def cost(u: str, v: str, length: float) -> float | None:
        m = sim.edge_multiplier(u, v, t)
        return None if m is None else length / speed * m

    dj = dijkstra(sim.airport.adjacency, src, dst, cost_fn=cost)
    static = veh.plan_static(sim.airport, dst, t, sim.edge_multiplier, sim.edge_cleared)
    tbl = Table(title=f"A* path at {sim.clock(t)}", header_style="bold white on #1d2733")
    for c in ("#", "Edge", "Length (m)", "Free-flow (min)", "Congestion x", "Cost (min)"):
        tbl.add_column(c, justify="right" if c != "Edge" else "left")
    for i, e in enumerate(r.edges, 1):
        ff = float(e["length_m"]) / speed
        tbl.add_row(str(i), f"{e['u']} -> {e['v']}", f"{e['length_m']:.0f}", f"{ff:.2f}",
                    f"{e['congestion']:.2f}", f"{e['minutes']:.2f}")
    console.print(tbl)
    console.print(Panel(
        f"Path: [bold]{' -> '.join(r.path)}[/]\n"
        f"A* cost {r.minutes:.2f} min over {r.length_m:.0f} m, nodes expanded "
        f"[bold]{r.expanded}[/]\n"
        f"Dijkstra (h=0) cost {dj.cost:.2f} min, nodes expanded {dj.expanded} "
        f"(same optimum: {abs(dj.cost - r.minutes) < 1e-9})\n"
        f"Static shortest-distance path (baselines): {' -> '.join(static.path)} = "
        f"{static.minutes:.2f} min\n"
        f"Heuristic h(n) = euclid(n, {dst}) / {speed:.0f} m/min (admissible: congestion >= 1)",
        title="A* summary", border_style="#1aa37a"))


@app.command(name="gantt")
def gantt_cmd(scenario: ScenarioOpt = "rush_hour", controller: ControllerOpt = "aerosync",
              seed: SeedOpt = 42,
              resolution: Annotated[int | None, typer.Option(help="Minutes per char.")] = None
              ) -> None:
    """Gate-occupancy timeline in the terminal."""
    _check_controller(controller)
    res = run_single(scenario, controller, seed)
    console.print(tv.gantt(res, resolution=resolution))
    console.print(f"Red = flight that hit a conflict at arrival, x = stand closed. Conflicts: "
                  f"{res.metrics['schedule_conflicts']}, hard violations: "
                  f"{res.metrics['hard_violations']}")


@app.command()
def report(scenario: Annotated[str, typer.Option("--scenario", "-s",
                                                 help="Scenario or 'all'.")] = "all",
           out: Annotated[Path, typer.Option(help="Output folder.")] = Path("results"),
           seeds: Annotated[str, typer.Option(help="Comma-separated seeds.")] =
           ",".join(map(str, DEFAULT_SEEDS))) -> None:
    """Full batch experiment: writes CSV/JSON and charts."""
    from aerosync.metrics.engine import METRIC_LABELS
    from aerosync.viz.charts import metric_bars, overview_grid

    scs = SCENARIO_NAMES if scenario == "all" else [scenario]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    _header(f"report: {len(scs)} scenario(s) x 3 controllers x {len(seed_list)} seed(s)")
    total = len(scs) * len(seed_list) * len(CONTROLLER_NAMES)
    cmd = f"aerosync report --scenario {scenario} --out {out.as_posix()} --seeds {seeds}"
    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/"
                  "{task.total}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("runs", total=total)

        def tick(label: str) -> None:
            prog.update(task, advance=1, description=label)

        summary = run_report(scs, seed_list, out, progress=tick, command=cmd)
    charts = out / "charts"
    for m in METRIC_LABELS:
        metric_bars(summary, m, charts / f"bars_{m}.png")
    overview_grid(summary, charts / "overview.png")
    for sc, ctls in summary["results"].items():
        console.print(tv.metrics_table({c: {k: v["mean"] for k, v in ms.items()}
                                        for c, ms in ctls.items()},
                                       title=f"{sc} - mean over seeds {seed_list}"))
    console.print(f"Wrote {out}/runs.csv, {out}/summary.json, {out}/summary.csv and "
                  f"{len(METRIC_LABELS) + 1} charts in {charts}/")


@app.command()
def serve(port: Annotated[int, typer.Option(help="Port.")] = 8501,
          headless: Annotated[bool, typer.Option(help="Do not open a browser.")] = False) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).resolve().parents[1] / "webapp" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
           "--browser.gatherUsageStats", "false", "--theme.base", "light"]
    if headless:
        cmd += ["--server.headless", "true"]
    console.print(f"Starting dashboard on http://localhost:{port} (Ctrl+C to stop)")
    raise typer.Exit(subprocess.call(cmd))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
