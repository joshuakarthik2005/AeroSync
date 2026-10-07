"""Matplotlib charts and diagrams (PNG, white background, >= 1600 px wide)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from aerosync.config import fmt_clock  # noqa: E402
from aerosync.metrics.engine import LOWER_IS_BETTER, METRIC_LABELS  # noqa: E402
from aerosync.sim.airport import Airport, build_airport  # noqa: E402
from aerosync.sim.environment import SimResult  # noqa: E402

# Categorical slots 1-3 of the validated reference palette; colour follows the
# controller (entity), never its rank.
CTL_COLORS = {"aerosync": "#2a78d6", "greedy": "#eb6834", "fcfs": "#1baf7a"}
CTL_ORDER = ["fcfs", "greedy", "aerosync"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
CRITICAL = "#d03b3b"
SCENARIO_LABELS = {"normal_day": "Normal day", "rush_hour": "Rush hour",
                   "storm_disruption": "Storm disruption", "gate_failure": "Gate failure"}
DPI = 120


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def metric_bars(summary: dict[str, Any], metric: str, path: Path) -> Path:
    """Grouped bars: scenarios on x, one bar per controller, mean +/- std over seeds."""
    res = summary["results"]
    scenarios = [s for s in SCENARIO_LABELS if s in res]
    fig, ax = plt.subplots(figsize=(14, 7.2))
    _style(ax)
    width = 0.26
    for i, ctl in enumerate(CTL_ORDER):
        xs = [j + (i - 1) * (width + 0.02) for j in range(len(scenarios))]
        means = [res[s][ctl][metric]["mean"] for s in scenarios]
        stds = [res[s][ctl][metric]["std"] for s in scenarios]
        bars = ax.bar(xs, means, width, color=CTL_COLORS[ctl], label=ctl, zorder=3,
                      edgecolor=SURFACE, linewidth=2)
        if any(stds):
            ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=TEXT2, elinewidth=1,
                        capsize=3, zorder=4)
        for b, m in zip(bars, means, strict=False):
            ax.annotate(f"{m:,.1f}", (b.get_x() + b.get_width() / 2, m), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10,
                        color=TEXT)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=12, color=TEXT)
    better = "lower is better" if LOWER_IS_BETTER[metric] else "higher is better"
    ax.set_title(f"{METRIC_LABELS[metric]}  ({better})", loc="left", fontsize=16,
                 color=TEXT, pad=14, fontweight="bold")
    seeds = summary.get("seeds", [])
    ax.text(0, 1.01, f"Mean over seeds {seeds}; whiskers = 1 std", transform=ax.transAxes,
            fontsize=10, color=TEXT2)
    ax.legend(frameon=False, fontsize=12, ncols=3, loc="upper right",
              bbox_to_anchor=(1, 1.09))
    return _save(fig, path)


def overview_grid(summary: dict[str, Any], path: Path) -> Path:
    """All eight headline metrics as small multiples."""
    res = summary["results"]
    scenarios = [s for s in SCENARIO_LABELS if s in res]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    width = 0.26
    for ax, metric in zip(axes.flat, METRIC_LABELS, strict=False):
        _style(ax)
        for i, ctl in enumerate(CTL_ORDER):
            xs = [j + (i - 1) * (width + 0.02) for j in range(len(scenarios))]
            ax.bar(xs, [res[s][ctl][metric]["mean"] for s in scenarios], width,
                   color=CTL_COLORS[ctl], label=ctl, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([SCENARIO_LABELS[s].replace(" ", "\n") for s in scenarios],
                           fontsize=9, color=TEXT2)
        arrow = "lower better" if LOWER_IS_BETTER[metric] else "higher better"
        ax.set_title(f"{METRIC_LABELS[metric]}\n({arrow})", fontsize=11, color=TEXT, loc="left")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, fontsize=13)
    fig.suptitle("AeroSync vs baselines - all headline metrics", x=0.01, ha="left",
                 fontsize=17, color=TEXT, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def compare_chart(results: dict[str, dict[str, float]], title: str, path: Path) -> Path:
    """Single-scenario comparison (one seed): small multiples, one per metric."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    ctls = [c for c in CTL_ORDER if c in results]
    for ax, metric in zip(axes.flat, METRIC_LABELS, strict=False):
        _style(ax)
        vals = [results[c][metric] for c in ctls]
        bars = ax.bar(ctls, vals, color=[CTL_COLORS[c] for c in ctls], zorder=3, width=0.6,
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals, strict=False):
            ax.annotate(f"{v:,.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=10, color=TEXT)
        arrow = "lower better" if LOWER_IS_BETTER[metric] else "higher better"
        ax.set_title(f"{METRIC_LABELS[metric]} ({arrow})", fontsize=11, color=TEXT, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=16, color=TEXT, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def gantt_chart(res: SimResult, path: Path, title: str, highlight: set[str] | None = None,
                marker_t: int | None = None) -> Path:
    """Gate-occupancy Gantt from actual on/off-block times."""
    highlight = highlight or set()
    stands = res.stands
    fig, ax = plt.subplots(figsize=(18, 8))
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ypos = {s: len(stands) - 1 - i for i, s in enumerate(stands)}
    t_end = max([res.window_min] + [f["off_block"] or 0 for f in res.flights])
    for c in res.closures:
        ax.add_patch(Rectangle((c["start"], ypos[c["stand"]] - 0.42), c["end"] - c["start"], 0.84,
                               facecolor="#f3d6d6", edgecolor=CRITICAL, hatch="//",
                               linewidth=0.8, zorder=2))
    for f in res.flights:
        if f["on_block"] is None or f["stand"] is None:
            continue
        end = f["off_block"] or t_end
        color = "#9ec5f0"
        edge = "#2a78d6"
        if f["id"] in highlight:
            color, edge = "#f6b38f", "#eb6834"
        if f["conflict"]:
            color, edge = "#f2a9a9", CRITICAL
        ax.add_patch(Rectangle((f["on_block"], ypos[f["stand"]] - 0.36), end - f["on_block"],
                               0.72, facecolor=color, edgecolor=edge, linewidth=1, zorder=3))
        if end - f["on_block"] >= 35:
            ax.text((f["on_block"] + end) / 2, ypos[f["stand"]], f["id"], ha="center",
                    va="center", fontsize=8, color=TEXT, zorder=4)
    if marker_t is not None:
        ax.axvline(marker_t, color=CRITICAL, linewidth=1.5, linestyle="--", zorder=5)
        ax.text(marker_t, len(stands) - 0.4, f" disruption {fmt_clock(marker_t, res.start_hour)}",
                color=CRITICAL, fontsize=10, va="bottom")
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()), fontsize=10, color=TEXT)
    ax.set_ylim(-0.6, len(stands) - 0.2)
    ax.set_xlim(0, t_end)
    ticks = list(range(0, t_end + 1, 60))
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt_clock(t, res.start_hour) for t in ticks], fontsize=10)
    ax.set_title(title, loc="left", fontsize=15, color=TEXT, fontweight="bold", pad=12)
    from matplotlib.patches import Patch

    legend = [Patch(facecolor="#9ec5f0", edgecolor="#2a78d6", label="turnaround"),
              Patch(facecolor="#f6b38f", edgecolor="#eb6834", label="moved by repair"),
              Patch(facecolor="#f2a9a9", edgecolor=CRITICAL, label="hit a conflict"),
              Patch(facecolor="#f3d6d6", edgecolor=CRITICAL, hatch="//", label="stand closed")]
    ax.legend(handles=legend, frameon=False, ncols=4, loc="upper right",
              bbox_to_anchor=(1, 1.07), fontsize=10)
    return _save(fig, path)


# ------------------------------------------------------------------ diagrams
def topology_diagram(path: Path, route: list[str] | None = None, airport: Airport | None = None,
                     title: str = "Airport apron topology", subtitle: str = "") -> Path:
    """Stands, taxi/apron graph and an optional highlighted A* route."""
    ap = airport or build_airport()
    fig, ax = plt.subplots(figsize=(18, 9.5))
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    for u, v, _l in ap.edges():
        (x1, y1), (x2, y2) = ap.nodes[u], ap.nodes[v]
        ax.plot([x1, x2], [-y1, -y2], color="#c9c7c0", linewidth=2, zorder=1)
    if route and len(route) > 1:
        xs = [ap.nodes[n][0] for n in route]
        ys = [-ap.nodes[n][1] for n in route]
        ax.plot(xs, ys, color="#eb6834", linewidth=5, zorder=2, solid_capstyle="round",
                label="A* route")
    for name, (x, _y) in ap.exits.items():
        ax.add_patch(FancyBboxPatch((x - 330, 20), 660, 45, boxstyle="round,pad=4",
                                    facecolor="#1d2733", edgecolor="none", zorder=1))
        ax.text(x, 42, f"Terminal {name} (exit)", color="white", ha="center", va="center",
                fontsize=12, fontweight="bold")
    for sid, st in ap.stands.items():
        x, y = ap.nodes[sid]
        color = "#2a78d6" if st.kind == "gate" else "#4a3aa7"
        size = 34 if st.wide else 24
        ax.add_patch(Rectangle((x - size, -y - size * 0.6), 2 * size, 1.2 * size,
                               facecolor=color, edgecolor="white", linewidth=2, zorder=4))
        ax.text(x, -y, sid, color="white", ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=5)
    for n, (x, y) in ap.nodes.items():
        if n in ap.stands:
            continue
        special = n in ("DEPOT", "HANGAR")
        ax.scatter([x], [-y], s=160 if special else 40, color="#1baf7a" if special else "#8a8f98",
                   zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(x + 12, -y + 14, n, fontsize=8 if not special else 11, color=TEXT2,
                fontweight="bold" if special else "normal", zorder=5)
    ax.set_title(title, loc="left", fontsize=17, color=TEXT, fontweight="bold")
    note = ("Blue = contact gates (large = wide-body capable), violet = remote stands, "
            "grey dots = apron/taxi nodes.")
    ax.text(0.0, -0.02, (subtitle + "\n" if subtitle else "") + note, transform=ax.transAxes,
            fontsize=11, color=TEXT2, va="top")
    ax.set_xlim(-330, 1680)
    ax.set_ylim(-790, 90)
    ax.set_aspect("equal")
    if route:
        ax.legend(frameon=False, loc="lower right", fontsize=12)
    return _save(fig, path)


def architecture_diagram(path: Path) -> Path:
    """Layered system architecture (UI / AI agents / algorithms / environment)."""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, tc: str = "white",
            fs: int = 12) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=fc, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold", wrap=True)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                     color=TEXT2, linewidth=1.6))
        if label:
            ax.text((x1 + x2) / 2 + 0.8, (y1 + y2) / 2, label, fontsize=10, color=TEXT2)

    ax.text(0, 97, "AeroSync architecture", fontsize=18, fontweight="bold", color=TEXT)
    box(2, 82, 30, 9, "Terminal CLI (Typer + Rich)", "#1d2733")
    box(35, 82, 30, 9, "Streamlit dashboard (6 tabs)", "#1d2733")
    box(68, 82, 30, 9, "Experiments + metrics engine", "#1d2733")
    ax.text(2, 77, "AI layer - independent of the UI", fontsize=12, color=TEXT2)
    box(2, 58, 22, 16, "Flight agents\n(request / re-request)", "#2a78d6")
    box(27, 58, 22, 16, "Coordinator agent\n(auctioneer, planner,\nrepairer)", "#4a3aa7")
    box(52, 58, 22, 16, "Gate agents x16\n(bid / refuse,\n1-hop view)", "#2a78d6")
    box(77, 58, 21, 16, "Ground vehicle agents\n(tug, fuel, bus)", "#2a78d6")
    box(2, 36, 22, 13, "CSP\nMRV + degree +\nforward checking", "#1baf7a", TEXT)
    box(27, 36, 22, 13, "Contract Net\nCFP -> BID -> AWARD", "#1baf7a", TEXT)
    box(52, 36, 22, 13, "Min-conflicts\nrepair", "#1baf7a", TEXT)
    box(77, 36, 21, 13, "A* search\n(congestion-aware)", "#1baf7a", TEXT)
    box(2, 14, 46, 13, "Typed message bus\nCFP | BID | REFUSE | AWARD | STATE_UPDATE | FAULT",
        "#eda100", TEXT)
    box(52, 14, 46, 13, "Airport environment (5-min steps)\nstands, taxi graph, flights, faults",
        "#eda100", TEXT)
    box(2, 1, 96, 8, "Scenarios (JSON, seeded): normal_day | rush_hour | storm_disruption | "
        "gate_failure", "#e4e3df", TEXT)
    for x in (17, 50, 83):
        arrow(x, 82, x, 74.5)
    arrow(38, 58, 13, 49.5)
    arrow(38, 58, 38, 49.5)
    arrow(38, 58, 63, 49.5)
    arrow(87, 58, 87, 49.5)
    arrow(25, 36, 25, 27.5, "messages")
    arrow(75, 36, 75, 27.5, "actions")
    arrow(48, 20.5, 52, 20.5)
    return _save(fig, path)


def csp_example_diagram(trace: list[dict[str, Any]], tasks: list[dict[str, Any]],
                        assignment: dict[str, tuple[str, int]], path: Path,
                        initial_domains: dict[str, int]) -> Path:
    """Worked CSP example: variables, domain sizes after each assignment, final schedule."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.axis("off")
    ax1.set_title("Backtracking with MRV + forward checking (real solver trace)", loc="left",
                  fontsize=14, color=TEXT, fontweight="bold")
    names = [t["id"] for t in tasks]
    header = ["Step", "MRV picks", "Value (stand, hold)", "Cost", "Pruned"] + [
        f"|D({n})|" for n in names]
    rows = [["0", "-", "initial domains", "", ""] + [str(initial_domains[n]) for n in names]]
    for st in trace:
        rows.append([str(st["step"]), st["var"], f"{st['value'][0]}, {st['value'][1]} min",
                     f"{st['cost']:.1f}", str(st["pruned"])]
                    + [str(st["domain_sizes"].get(n, "=")) for n in names])
    tbl = ax1.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.0)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#1d2733")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
    lines = [f"{t['id']}: {'wide' if t['size'] == 'W' else 'narrow'}-body, ETA "
             f"{t['eta']}, STD {t['sched_dep']}, prefers {t['pref_terminal']}" for t in tasks]
    ax1.text(0, 0.02, "Variables:\n" + "\n".join(lines) + "\n'=' : already assigned",
             transform=ax1.transAxes, fontsize=10.5, color=TEXT2, va="bottom")
    _style(ax2)
    ax2.yaxis.grid(False)
    stands = sorted({v[0] for v in assignment.values()})
    ypos = {s: i for i, s in enumerate(stands)}
    task_by = {t["id"]: t for t in tasks}
    for fid, (s, h) in assignment.items():
        t = task_by[fid]
        start = t["eta"] + h
        end = max(t["sched_dep"], start + t["min_turn"])
        ax2.add_patch(Rectangle((start, ypos[s] - 0.3), end - start, 0.6, facecolor="#9ec5f0",
                                edgecolor="#2a78d6"))
        ax2.add_patch(Rectangle((end, ypos[s] - 0.3), 15, 0.6, facecolor="#f2f1ec",
                                edgecolor=GRID, hatch=".."))
        ax2.text((start + end) / 2, ypos[s], f"{fid}\nhold {h}", ha="center", va="center",
                 fontsize=10, color=TEXT)
    ax2.set_yticks(list(ypos.values()))
    ax2.set_yticklabels(stands, fontsize=11)
    ax2.set_ylim(-0.7, len(stands) - 0.3)
    xmax = max(max(t["sched_dep"] for t in tasks) + 60, 180)
    ax2.set_xlim(0, xmax)
    ax2.set_xlabel("minutes", color=TEXT2)
    ax2.set_title("Resulting schedule (dotted = 15-min buffer)", loc="left", fontsize=14,
                  color=TEXT, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
