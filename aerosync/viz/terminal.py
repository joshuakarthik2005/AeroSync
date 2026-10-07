"""Rich renderables for the terminal CLI (tables, Gantt, explanations)."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aerosync.config import fmt_clock
from aerosync.metrics.engine import LOWER_IS_BETTER, METRIC_LABELS
from aerosync.sim.environment import SimResult

PALETTE = ["#2a78d6", "#1aa37a", "#d98a00", "#8b5bd6", "#d6452a", "#0f9bb3", "#b5487f",
           "#6a8f1f", "#c26f2e", "#4a5fc1"]
CTL_STYLE = {"fcfs": "#8a8f98", "greedy": "#d98a00", "aerosync": "#2a78d6"}


def metrics_table(results: dict[str, dict[str, float]], title: str = "Metrics") -> Table:
    """Controllers as columns, headline metrics as rows; best value highlighted."""
    t = Table(title=title, header_style="bold white on #1d2733", title_style="bold")
    t.add_column("Metric", style="bold")
    for c in results:
        t.add_column(c, justify="right", style=CTL_STYLE.get(c, ""))
    for key, label in METRIC_LABELS.items():
        vals = {c: results[c].get(key, 0.0) for c in results}
        best = (min if LOWER_IS_BETTER[key] else max)(vals.values())
        cells = []
        for c in results:
            v = vals[c]
            s = f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"
            cells.append(Text(s, style="bold green") if v == best and len(results) > 1
                         else Text(s))
        t.add_row(label, *cells)
    return t


def run_summary(res: SimResult) -> Panel:
    m = res.metrics
    mc = res.message_counts
    lines = [
        f"[bold]{res.scenario}[/] under [bold {CTL_STYLE.get(res.controller, '')}]"
        f"{res.controller}[/] (seed {res.seed})",
        f"Window {fmt_clock(0, res.start_hour)}-{fmt_clock(res.window_min, res.start_hour)}, "
        f"{m['flights']} flights, {m['flights_completed']} departed by "
        f"{fmt_clock(res.end_t, res.start_hour)}",
        "Messages: " + ", ".join(f"{k} {v}" for k, v in mc.items()),
        f"Auctions {res.counters.get('auctions', 0)}, repairs {res.counters.get('repairs', 0)}, "
        f"vehicle dispatches {res.counters.get('dispatches', 0)}, towing "
        f"{res.counters.get('towing', 0)}, hard violations {m['hard_violations']}",
    ]
    csp = next((d for d in res.decisions if d["kind"] == "csp_plan"), None)
    if csp:
        lines.append("CSP: " + csp["reasoning"])
    return Panel("\n".join(lines), title="Run summary", border_style="#2a78d6")


def gantt(res: SimResult, resolution: int | None = None, t0: int = 0,
          t1: int | None = None, highlight: set[str] | None = None) -> Table:
    """ASCII/Rich gate-occupancy timeline (actual on/off-block)."""
    t1 = t1 if t1 is not None else max(res.window_min, max(
        (f["off_block"] or 0) for f in res.flights))
    span = t1 - t0
    res_min = resolution or max(5, int(-(-span // 90 // 5) * 5))
    ncols = -(-span // res_min)
    table = Table(title=f"Gate occupancy - {res.scenario} / {res.controller} "
                        f"(1 char = {res_min} min)",
                  show_lines=False, header_style="bold", pad_edge=False, box=None)
    table.add_column("Stand", style="bold", no_wrap=True)
    header = Text()
    for i in range(ncols):
        tm = t0 + i * res_min
        header.append("|" if tm % 60 == 0 else " ")
    table.add_column(header, no_wrap=True)
    colour = {f["id"]: PALETTE[i % len(PALETTE)] for i, f in enumerate(
        sorted(res.flights, key=lambda f: (f["sched_arr"], f["id"])))}
    highlight = highlight or set()
    for stand in res.stands:
        row = Text()
        cells: list[tuple[str, str]] = [(" ", "")] * ncols
        for c in res.closures:
            if c["stand"] != stand:
                continue
            for i in range(ncols):
                tm = t0 + i * res_min
                if c["start"] <= tm < c["end"]:
                    cells[i] = ("x", "bold red")
        for f in res.flights:
            if f["stand"] != stand or f["on_block"] is None:
                continue
            end = f["off_block"] if f["off_block"] is not None else t1
            style = colour[f["id"]]
            if f["conflict"]:
                style = "bold white on red"
            if f["id"] in highlight:
                style = "bold black on yellow"
            label = f["id"]
            k = 0
            for i in range(ncols):
                tm = t0 + i * res_min
                if f["on_block"] <= tm + res_min - 1 and tm < end:
                    ch = label[k] if k < len(label) else "="
                    cells[i] = (ch, style)
                    k += 1
        for ch, st in cells:
            row.append(ch if ch != " " else "·", style=st or "grey35")
        table.add_row(stand, row)
    hours = Text()
    for i in range(ncols):
        tm = t0 + i * res_min
        if tm % 120 == 0 and i + 5 <= ncols:
            hours.append(fmt_clock(tm, res.start_hour))
        elif len(hours) <= i:
            hours.append(" ")
    table.add_row("", hours)
    return table


def explain_view(res: SimResult, fid: str) -> Group | Text:
    """Every Contract Net round for one flight, with bid breakdowns."""
    f = res.flight(fid)
    if f is None:
        return Text(f"Flight {fid} not found in {res.scenario}.", style="red")
    items: list[Any] = []
    size = "wide-body" if f["size"] == "W" else "narrow-body"
    hdr = (f"[bold]{fid}[/] {f['airline']} {size}, {f['pax']} pax, prefers {f['pref_terminal']}"
           f" | STA {res.clock(f['sched_arr'])} STD {res.clock(f['sched_dep'])}"
           f" | arrival delay {f['arr_delay']} min\n"
           f"Final stand [bold]{f['stand']}[/], on-block {res.clock(f['on_block'])}, off-block "
           f"{res.clock(f['off_block'])}, hold {f['hold']} min, departure delay "
           f"{f['dep_delay']} min, walk {f['walk_m']:.0f} m\n"
           "History: " + " -> ".join(f"{res.clock(h[0])} {h[1]} ({h[2]})" for h in f["history"]))
    items.append(Panel(hdr, title="Flight", border_style="#2a78d6"))
    auctions = res.auctions.get(fid, [])
    if not auctions:
        rows = [d for d in res.decisions if d.get("flight") == fid]
        txt = "\n".join(d["reasoning"] for d in rows) or "No recorded decisions."
        items.append(Panel(txt, title=f"{res.controller} decision (no negotiation)"))
        return Group(*items)
    for a in auctions:
        t = Table(title=f"Contract Net round at {res.clock(a['t'])} ({a['context']})",
                  header_style="bold white on #1d2733")
        cols = ["Stand", "Bid", "Walk", "Term.", "Remote", "Buffer", "Neighb.", "Hold",
                "Reassign", "Displace", "On-block"]
        for c in cols:
            t.add_column(c, justify="right" if c != "Stand" else "left")
        win = a["winner"]["bidder"] if a["winner"] else None
        for b in a["bids"]:
            bd = b["breakdown"]
            style = "bold green" if b["bidder"] == win else ""
            t.add_row(b["bidder"] + (" *" if b["bidder"] == win else ""), f"{b['cost']:.1f}",
                      f"{bd['walk']:.1f}", f"{bd['terminal_mismatch']:.0f}",
                      f"{bd['remote']:.0f}", f"{bd['buffer_risk']:.1f}",
                      f"{bd['neighbour_pressure']:.1f}", f"{bd['delay']:.0f}",
                      f"{bd['reassignment']:.0f}", f"{bd.get('displacement', 0):.0f}",
                      res.clock(b["start"]), style=style)
        for r in a["refusals"]:
            t.add_row(r["bidder"], "REFUSE", *[""] * 8, r["reason"], style="grey50")
        items.append(t)
        items.append(Panel(a["reasoning"], title="Reasoning", border_style="#1aa37a"))
    return Group(*items)


def repair_view(res: SimResult, since: int = 0, limit: int = 3) -> Group:
    """Repairs triggered at or after ``since`` (minutes)."""
    items: list[Any] = []
    reps = [r for r in res.repairs if r["t"] >= since]
    for r in reps[:limit]:
        t = Table(title=f"{r.get('method', 'min-conflicts').capitalize()} repair at "
                        f"{res.clock(r['t'])} - trigger: {r['trigger']}",
                  header_style="bold white on #1d2733")
        for c in ("Flight", "Before", "After", "Old on-block", "New on-block"):
            t.add_column(c)
        for row in r["changed"]:
            t.add_row(row["flight"], row["from"], row["to"], res.clock(row["old_start"]),
                      res.clock(row["new_start"]))
        items.append(t)
        items.append(Text(f"  conflicts {r['conflicts_before']} -> {r['conflicts_after']} in "
                          f"{r['steps']} step(s), {r['time_ms']:.1f} ms; agents involved: "
                          + ", ".join(r["agents"]), style="#1aa37a"))
    if len(reps) > limit:
        items.append(Text(f"  ... and {len(reps) - limit} smaller follow-up repair(s) "
                          "(displacements / drift) later in the day.", style="#52514e"))
    if not items:
        items.append(Text("No min-conflicts repair was needed after this event.", style="yellow"))
    return Group(*items)
