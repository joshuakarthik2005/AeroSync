"""AeroSync Streamlit dashboard.

Run with ``aerosync serve`` or ``streamlit run aerosync/webapp/app.py``.
Everything is computed locally and offline from seeded simulations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aerosync.config import (
    ALPHA,
    BUFFER_MIN,
    DEFAULT_WEIGHTS,
    LOOKAHEAD_MIN,
    MAX_HOLD_MIN,
    STEP_MIN,
    fmt_clock,
)
from aerosync.controllers import CONTROLLER_NAMES
from aerosync.experiments import run_report, run_single
from aerosync.metrics.engine import LOWER_IS_BETTER, METRIC_LABELS
from aerosync.sim.airport import build_airport
from aerosync.sim.environment import SimResult
from aerosync.sim.events import make_event
from aerosync.sim.scenarios import SCENARIO_NAMES, list_scenarios, load_scenario

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DIAGRAMS = ROOT / "docs" / "slide_assets" / "diagrams"
CTL_COLORS = {"aerosync": "#2a78d6", "greedy": "#eb6834", "fcfs": "#1baf7a"}
STATUS = {"free": "#d9d7d0", "ok": "#2a78d6", "late": "#eda100", "conflict": "#d03b3b",
          "closed": "#7a1f1f"}
VEH_COLORS = {"tug": "#4a3aa7", "fuel": "#1baf7a", "bus": "#eb6834"}

st.set_page_config(page_title="AeroSync", page_icon="✈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: #f7f7f5; }
    header[data-testid="stHeader"] { background: #1d2733; }
    .block-container { padding-top: 3.2rem; }
    .aero-hero { background: #1d2733; color: #fff; padding: 18px 24px; border-radius: 10px;
                 margin-bottom: 14px; }
    .aero-hero h1 { color: #fff; font-size: 1.8rem; margin: 0; padding: 0; }
    .aero-hero p { color: #c3c2b7; margin: 4px 0 0 0; }
    .card { background: #fff; border: 1px solid #e4e3df; border-radius: 10px; padding: 14px 16px;
            height: 100%; }
    .card h4 { margin: 0 0 6px 0; color: #1d2733; }
    .card p { margin: 0; color: #52514e; font-size: 0.92rem; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid #e4e3df;
                                  border-radius: 10px; padding: 8px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ cached runs
@st.cache_data(show_spinner=False, max_entries=48)
def cached_run(scenario: str, controller: str, seed: int, events_json: str = "[]"
               ) -> dict[str, Any]:
    events = [make_event(e["kind"], e["at"], **e["params"]) for e in json.loads(events_json)]
    return run_single(scenario, controller, seed, extra_events=events).to_dict()


def get_run(scenario: str, controller: str, seed: int, events: list[dict] | None = None
            ) -> SimResult:
    return SimResult.from_dict(cached_run(scenario, controller, seed,
                                          json.dumps(events or [], sort_keys=True)))


@st.cache_resource
def airport():
    return build_airport()


# --------------------------------------------------------------------- helpers
def vehicle_positions(res: SimResult, t: float) -> list[dict[str, Any]]:
    """Interpolate every ground vehicle's position at minute ``t`` from its trips."""
    ap = airport()
    last: dict[str, dict[str, Any]] = {}
    for j in sorted(res.jobs, key=lambda j: j["depart"]):
        vid = j["vehicle"]
        if j["depart"] > t:
            continue
        path = j["path"] or [j["stand"]]
        if t >= j["arrive"] or len(path) < 2:
            x, y = ap.nodes[path[-1]]
            # Park beside the stand (towards the apron road) so the stand stays visible.
            x += {"tug": -14, "fuel": 0, "bus": 14}[j["kind"]]
            y += 45 if y < 400 else -45
            moving = False
        else:
            frac = (t - j["depart"]) / max(1e-6, j["arrive"] - j["depart"])
            seg = [ap.euclid(a, b) for a, b in zip(path, path[1:], strict=False)]
            target = frac * sum(seg)
            acc = 0.0
            x, y = ap.nodes[path[0]]
            for (a, b), d in zip(zip(path, path[1:], strict=False), seg, strict=False):
                if acc + d >= target:
                    r = (target - acc) / d if d else 0
                    (x1, y1), (x2, y2) = ap.nodes[a], ap.nodes[b]
                    x, y = x1 + r * (x2 - x1), y1 + r * (y2 - y1)
                    break
                acc += d
            moving = True
        last[vid] = {"id": vid, "kind": j["kind"], "x": x, "y": y, "moving": moving,
                     "path": path if moving else [], "flight": j["flight"]}
    out = []
    for vid in sorted({j["vehicle"] for j in res.jobs}):
        if vid in last:
            out.append(last[vid])
        else:
            x, y = ap.nodes["DEPOT"]
            kind = vid.rstrip("0123456789").lower()
            out.append({"id": vid, "kind": kind, "x": x, "y": y, "moving": False, "path": [],
                        "flight": None})
    return out


def stand_status(res: SimResult, t: int) -> dict[str, tuple[str, str]]:
    out = {s: ("free", "") for s in res.stands}
    for f in res.flights:
        if f["stand"] and f["on_block"] is not None and f["on_block"] <= t < (
                f["off_block"] or 10**9):
            state = "conflict" if f["conflict"] else ("late" if (f["dep_delay"] or 0) > 15
                                                       else "ok")
            out[f["stand"]] = (state, f["id"])
    for c in res.closures:
        if c["start"] <= t < c["end"]:
            out[c["stand"]] = ("closed", out[c["stand"]][1])
    return out


def apron_figure(res: SimResult, t: int) -> go.Figure:
    ap = airport()
    fig = go.Figure()
    ex, ey = [], []
    for u, v, _ in ap.edges():
        (x1, y1), (x2, y2) = ap.nodes[u], ap.nodes[v]
        ex += [x1, x2, None]
        ey += [-y1, -y2, None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line={"color": "#c9c7c0", "width": 2},
                             hoverinfo="skip", showlegend=False))
    vehicles = vehicle_positions(res, t)
    for v in vehicles:
        if v["path"]:
            xs = [ap.nodes[n][0] for n in v["path"]]
            ys = [-ap.nodes[n][1] for n in v["path"]]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line={"color": VEH_COLORS[v["kind"]], "width": 3,
                                           "dash": "dot"}, showlegend=False, opacity=0.6))
    nx = [p[0] for n, p in ap.nodes.items() if n not in ap.stands]
    ny = [-p[1] for n, p in ap.nodes.items() if n not in ap.stands]
    names = [n for n in ap.nodes if n not in ap.stands]
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers", marker={"size": 6, "color": "#8a8f98"},
                             text=names, hovertemplate="%{text}<extra></extra>",
                             showlegend=False))
    status = stand_status(res, t)
    for key, label in [("free", "free"), ("ok", "occupied"), ("late", "occupied, late"),
                       ("conflict", "hit a conflict"), ("closed", "closed")]:
        ids = [s for s, (st_, _) in status.items() if st_ == key]
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[ap.nodes[s][0] for s in ids], y=[-ap.nodes[s][1] for s in ids], mode="markers+text",
            marker={"symbol": "square", "size": [30 if ap.stands[s].wide else 24 for s in ids],
                    "color": STATUS[key], "line": {"color": "white", "width": 2}},
            text=[f"{s}<br>{status[s][1]}" if status[s][1] else s for s in ids],
            textposition="top center", textfont={"size": 10, "color": "#0b0b0b"},
            name=label, hovertemplate="%{text}<extra></extra>"))
    for kind in ("tug", "fuel", "bus"):
        vs = [v for v in vehicles if v["kind"] == kind]
        if not vs:
            continue
        fig.add_trace(go.Scatter(
            x=[v["x"] for v in vs], y=[-v["y"] for v in vs], mode="markers",
            marker={"size": 13, "color": VEH_COLORS[kind], "symbol": "diamond",
                    "line": {"color": "white", "width": 1.5}},
            name=f"{kind} ({len(vs)})", text=[f"{v['id']} → {v['flight'] or 'idle'}" for v in vs],
            hovertemplate="%{text}<extra></extra>"))
    for name, (x, _y) in ap.exits.items():
        fig.add_annotation(x=x, y=50, text=f"<b>Terminal {name}</b>", showarrow=False,
                           font={"color": "white", "size": 12}, bgcolor="#1d2733",
                           borderpad=6)
    fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      xaxis={"visible": False, "range": [-320, 1680]},
                      yaxis={"visible": False, "range": [-770, 80], "scaleanchor": "x"},
                      legend={"orientation": "h", "y": -0.02, "x": 0})
    return fig


def gantt_figure(res: SimResult, highlight: set[str] | None = None, marker: int | None = None,
                 height: int = 520) -> go.Figure:
    highlight = highlight or set()
    fig = go.Figure()
    groups: dict[str, list[dict]] = {"turnaround": [], "moved by repair": [],
                                     "hit a conflict": []}
    for f in res.flights:
        if f["on_block"] is None or f["stand"] is None:
            continue
        key = ("hit a conflict" if f["conflict"] else "moved by repair" if f["id"] in highlight
               else "turnaround")
        groups[key].append(f)
    colors = {"turnaround": "#2a78d6", "moved by repair": "#eb6834", "hit a conflict": "#d03b3b"}
    for key, fl in groups.items():
        if not fl:
            continue
        fig.add_trace(go.Bar(
            y=[f["stand"] for f in fl],
            x=[(f["off_block"] or res.end_t) - f["on_block"] for f in fl],
            base=[f["on_block"] for f in fl], orientation="h", name=key,
            marker={"color": colors[key], "line": {"color": "white", "width": 1}},
            text=[f["id"] for f in fl], textposition="inside", insidetextanchor="middle",
            customdata=[[f["id"], res.clock(f["on_block"]), res.clock(f["off_block"]),
                         f["dep_delay"], f["hold"]] for f in fl],
            hovertemplate="%{customdata[0]}<br>on-block %{customdata[1]} → off-block "
                          "%{customdata[2]}<br>hold %{customdata[4]} min, dep. delay "
                          "%{customdata[3]} min<extra></extra>"))
    for c in res.closures:
        fig.add_trace(go.Bar(y=[c["stand"]], x=[c["end"] - c["start"]], base=[c["start"]],
                             orientation="h", name="stand closed", showlegend=True,
                             marker={"color": "rgba(208,59,59,0.18)", "pattern": {"shape": "/"},
                                     "line": {"color": "#d03b3b", "width": 1}},
                             hovertemplate=f"{c['stand']} closed<extra></extra>"))
    end = max([res.window_min] + [f["off_block"] or 0 for f in res.flights])
    ticks = list(range(0, end + 1, 60))
    if marker is not None:
        fig.add_vline(x=marker, line={"color": "#d03b3b", "dash": "dash", "width": 2})
    fig.update_layout(barmode="overlay", height=height, bargap=0.25,
                      margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      yaxis={"categoryorder": "array", "categoryarray": list(reversed(res.stands)),
                             "gridcolor": "#efeee9"},
                      xaxis={"tickvals": ticks, "ticktext": [res.clock(x) for x in ticks],
                             "gridcolor": "#e4e3df", "range": [0, end]},
                      legend={"orientation": "h", "y": 1.08, "x": 0}, uniformtext_minsize=7,
                      uniformtext_mode="hide")
    return fig


def metric_row(m: dict[str, float], ref: dict[str, float] | None = None) -> None:
    cols = st.columns(4)
    keys = ["avg_walk_m", "avg_dep_delay_min", "schedule_conflicts", "reassignments"]
    for col, k in zip(cols, keys, strict=False):
        delta = None
        if ref is not None:
            d = m[k] - ref[k]
            delta = f"{d:+.1f}"
        col.metric(METRIC_LABELS[k], f"{m[k]:,.1f}" if isinstance(m[k], float) else m[k],
                   delta=delta, delta_color="inverse" if LOWER_IS_BETTER[k] else "normal")


# ---------------------------------------------------------------------- layout
st.markdown(
    '<div class="aero-hero"><h1>✈ AeroSync</h1><p>Multi-agent airport gate assignment and '
    "ground operations coordinator - flight, gate and ground-vehicle agents negotiate stands "
    "and apron routes, and repair the schedule when delays, closures or weather hit.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run settings")
    scen_info = {s["name"]: s for s in list_scenarios()}
    scenario = st.selectbox("Scenario", SCENARIO_NAMES, index=1,
                            format_func=lambda s: s.replace("_", " "))
    st.caption(scen_info[scenario]["description"])
    seed = int(st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1))
    controller = st.radio("Controller", CONTROLLER_NAMES, index=2, horizontal=True)
    st.divider()
    st.caption(f"Timestep {STEP_MIN} min · buffer {BUFFER_MIN} min · alpha {ALPHA} · "
               "runs fully offline and deterministic per seed.")

tabs = st.tabs(["Live Simulation", "Gate Timeline", "Disruption Lab", "Explainability",
                "Benchmark", "About / PEAS"])

# ------------------------------------------------------------- live simulation
with tabs[0]:
    with st.spinner("Simulating..."):
        res = get_run(scenario, controller, seed)
    total_steps = res.end_t // STEP_MIN
    ss = st.session_state
    run_key = f"{scenario}|{controller}|{seed}"
    if ss.get("run_key") != run_key:
        ss.run_key, ss.step, ss.playing = run_key, 0, False
    def _toggle() -> None:
        ss.playing = not ss.get("playing", False)

    def _step() -> None:
        ss.playing = False
        ss.step = min(total_steps, ss.step + 1)

    def _reset() -> None:
        ss.playing, ss.step = False, 0

    def _jump() -> None:
        ss.playing = False
        ss.step = ss.jump

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.6, 4])
    c1.button("⏸ Pause" if ss.get("playing") else "▶ Play", on_click=_toggle, width="stretch")
    c2.button("⏭ Step", on_click=_step, width="stretch")
    c3.button("⟲ Reset", on_click=_reset, width="stretch")
    speed = c4.select_slider("Speed", options=[1, 2, 4, 8, 16], value=4,
                             format_func=lambda v: f"{v} steps/s")
    ss.jump = ss.step
    c5.slider("Time", 0, total_steps, format="step %d", key="jump", on_change=_jump)

    @st.fragment(run_every=(1.0 / speed) if ss.get("playing") else None)
    def live_view() -> None:
        if ss.get("playing"):
            ss.step = ss.step + 1
            if ss.step >= total_steps:
                ss.step, ss.playing = total_steps, False
        t = ss.step * STEP_MIN
        fl = res.flights
        at_gate = sum(1 for f in fl if f["on_block"] is not None and f["on_block"] <= t
                      < (f["off_block"] or 10**9))
        holding = sum(1 for f in fl if f["sched_arr"] + f["arr_delay"] <= t
                      and (f["on_block"] is None or f["on_block"] > t))
        departed = sum(1 for f in fl if f["off_block"] is not None and f["off_block"] <= t)
        inbound = sum(1 for f in fl if t < f["sched_arr"] + f["arr_delay"] <= t + LOOKAHEAD_MIN)
        k = st.columns(5)
        k[0].metric("Clock", res.clock(t), f"step {ss.step}/{total_steps}", delta_color="off")
        k[1].metric("At stand", at_gate)
        k[2].metric("Holding for a stand", holding)
        k[3].metric("Inbound (< 60 min)", inbound)
        k[4].metric("Departed", departed)
        left, right = st.columns([2.2, 1])
        left.plotly_chart(apron_figure(res, t), width="stretch",
                          config={"displayModeBar": False}, key=f"apron_{ss.step}")
        right.markdown("**Agent message log**")
        types = right.multiselect("Message types", ["CFP", "BID", "REFUSE", "AWARD",
                                                    "STATE_UPDATE", "FAULT"],
                                  default=["CFP", "BID", "REFUSE", "AWARD", "FAULT"],
                                  key="msg_types", label_visibility="collapsed")
        msgs = [m for m in res.messages if m["t"] <= t and m["type"] in types][-60:]
        if msgs:
            df = pd.DataFrame([{"time": res.clock(m["t"]), "type": m["type"],
                                "from": m["sender"], "to": m["receiver"],
                                "message": m["summary"]} for m in reversed(msgs)])
            right.dataframe(df, height=470, hide_index=True, width="stretch")
        else:
            right.info("No messages yet.")

    live_view()

# ---------------------------------------------------------------- gate timeline
with tabs[1]:
    ctl_t = st.radio("Controller", CONTROLLER_NAMES, index=CONTROLLER_NAMES.index(controller),
                     horizontal=True, key="tl_ctl")
    r_tl = get_run(scenario, ctl_t, seed)
    metric_row(r_tl.metrics)
    st.plotly_chart(gantt_figure(r_tl), width="stretch")
    st.caption("Bars are actual on-block → off-block times. Red bars hit a conflict on arrival "
               "(their planned stand was not free); hatched = stand closed.")

# --------------------------------------------------------------- disruption lab
with tabs[2]:
    st.subheader("Inject a fault and watch the agents repair the schedule")
    sc_obj = load_scenario(scenario, seed)
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Fault", ["gate_closure", "flight_delay", "vehicle_breakdown",
                                  "comms_loss"],
                        format_func=lambda k: k.replace("_", " "))
    at = c2.slider("At step", 0, sc_obj.steps - 1, min(30, sc_obj.steps - 1))
    params: dict[str, Any] = {}
    if kind in ("gate_closure", "comms_loss"):
        params["gate"] = c3.selectbox("Stand", list(airport().stands), index=3)
        params["duration"] = st.select_slider(
            "Duration (min)", [30, 60, 120, 180, 240, 0], value=0 if kind == "gate_closure"
            else 60, format_func=lambda v: "rest of day" if v == 0 else str(v)) or None
    elif kind == "flight_delay":
        ids = sorted(f.id for f in sc_obj.flights)
        params["flight"] = c3.selectbox("Flight", ids, index=ids.index("AI302")
                                        if "AI302" in ids else 0)
        params["minutes"] = st.slider("Delay (min)", 5, 120, 45, 5)
    else:
        vids = [f"{k.upper()}{i}" for k in ("tug", "fuel", "bus")
                for i in range(1, sc_obj.fleet.get(k, 0) + 1)]
        params["vehicle"] = c3.selectbox("Vehicle", vids, index=vids.index("FUEL1"))
        params["duration"] = st.slider("Duration (min)", 15, 180, 60, 15)
    ev = make_event(kind, at, **params)
    st.markdown(f"**Fault:** {ev.describe()} at **{fmt_clock(at * STEP_MIN, sc_obj.start_hour)}**"
                f" (controller: `{controller}`)")
    if st.button("Inject fault and run", type="primary"):
        st.session_state.lab = {"event": {"kind": ev.kind, "at": ev.at,
                                          "params": ev.params}, "controller": controller,
                                "scenario": scenario, "seed": seed}
    lab = st.session_state.get("lab")
    if lab and lab["scenario"] == scenario and lab["seed"] == seed:
        evd = lab["event"]
        t_ev = evd["at"] * STEP_MIN
        with st.spinner("Running before / after..."):
            before = get_run(scenario, lab["controller"], seed)
            after = get_run(scenario, lab["controller"], seed, [evd])
        reps = [r for r in after.repairs if r["t"] >= t_ev]
        moved = {row["flight"] for r in reps if r["t"] == t_ev for row in r["changed"]}
        st.markdown(f"#### Impact on `{lab['controller']}` (Δ vs. same day without the fault)")
        metric_row(after.metrics, before.metrics)
        b1, b2 = st.columns(2)
        b1.markdown("**Before** - no fault")
        b1.plotly_chart(gantt_figure(before, marker=t_ev, height=460), width="stretch",
                        key="g_before")
        b2.markdown("**After** - fault injected (orange = moved by the repair at the fault time)")
        b2.plotly_chart(gantt_figure(after, highlight=moved, marker=t_ev, height=460),
                        width="stretch", key="g_after")
        if reps:
            st.markdown("#### Repairs and the agents involved")
            rows = []
            for r in reps:
                for row in r["changed"]:
                    rows.append({"time": after.clock(r["t"]), "trigger": r["trigger"],
                                 "method": r.get("method", "min-conflicts"),
                                 "flight": row["flight"], "from": row["from"], "to": row["to"],
                                 "old on-block": after.clock(row["old_start"]),
                                 "new on-block": after.clock(row["new_start"]),
                                 "agents": ", ".join(r["agents"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        elif lab["controller"] != "aerosync":
            st.info(f"`{lab['controller']}` has no repair mechanism: affected flights simply wait"
                    " (or are pushed to another stand when they arrive at a closed one).")
        else:
            st.info("No plan conflicts were created by this fault.")
        st.markdown("#### All controllers under the same fault")
        comp = {c: get_run(scenario, c, seed, [evd]).metrics for c in CONTROLLER_NAMES}
        st.dataframe(pd.DataFrame({c: {METRIC_LABELS[k]: comp[c][k] for k in METRIC_LABELS}
                                   for c in CONTROLLER_NAMES}), width="stretch")

# --------------------------------------------------------------- explainability
with tabs[3]:
    r_ex = get_run(scenario, controller, seed)
    ids = sorted(f["id"] for f in r_ex.flights)
    fid = st.selectbox("Flight", ids, index=ids.index("AI302") if "AI302" in ids else 0)
    f = r_ex.flight(fid)
    assert f is not None
    k = st.columns(5)
    k[0].metric("Final stand", f["stand"])
    k[1].metric("On-block", r_ex.clock(f["on_block"]), f"hold {f['hold']} min",
                delta_color="off")
    k[2].metric("Departure delay", f"{f['dep_delay']} min")
    k[3].metric("Walking distance", f"{f['walk_m']:.0f} m")
    k[4].metric("Gate changes after announcement", f["reassignments"])
    st.markdown("**Assignment history:** " + " → ".join(
        f"`{r_ex.clock(h[0])} {h[1]}` ({h[2]})" for h in f["history"]))
    auctions = r_ex.auctions.get(fid, [])
    if not auctions:
        dec = [d for d in r_ex.decisions if d.get("flight") == fid]
        st.info(f"`{controller}` does not negotiate. Decision: "
                + ("; ".join(d["reasoning"] for d in dec) or "none recorded"))
    for i, a in enumerate(auctions):
        st.markdown(f"#### Contract Net round {i + 1} - {r_ex.clock(a['t'])} ({a['context']})")
        st.success(a["reasoning"])
        rows = []
        for b in a["bids"]:
            rows.append({"stand": b["bidder"], "bid": b["cost"], **b["breakdown"],
                         "on-block": r_ex.clock(b["start"]), "hold (min)": b["hold"],
                         "winner": "★" if a["winner"] and b["bidder"] == a["winner"]["bidder"]
                         else ""})
        if rows:
            df = pd.DataFrame(rows)
            cc1, cc2 = st.columns([1.3, 1])
            cc1.dataframe(df, hide_index=True, width="stretch")
            comp_cols = [c for c in ["walk", "terminal_mismatch", "remote", "buffer_risk",
                                     "neighbour_pressure", "delay", "reassignment",
                                     "displacement"] if c in df]
            figb = go.Figure()
            palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                       "#4a3aa7", "#e34948"]
            for col, colr in zip(comp_cols, palette, strict=False):
                figb.add_trace(go.Bar(y=df["stand"], x=df[col], name=col, orientation="h",
                                      marker={"color": colr, "line": {"color": "white",
                                                                      "width": 1}}))
            figb.update_layout(barmode="stack", height=420, margin={"l": 10, "r": 10,
                                                                    "t": 10, "b": 10},
                               yaxis={"autorange": "reversed"}, plot_bgcolor="#fff",
                               legend={"orientation": "h", "y": -0.15}, xaxis_title="bid cost")
            cc2.plotly_chart(figb, width="stretch", key=f"bidbar_{i}")
        if a["refusals"]:
            st.caption("Refused: " + ", ".join(f"{r['bidder']} ({r['reason']})"
                                               for r in a["refusals"]))
    w = DEFAULT_WEIGHTS
    st.code(f"bid = {w.walk}*walk_m*pax/150 + {w.mismatch}*terminal_mismatch + {w.remote}*remote"
            f" + {w.buffer_risk}*buffer_risk\n    + {w.neighbour}*(alpha={ALPHA} * "
            f"neighbour_load) + {w.delay}*hold_min + {w.reassign}*(reassignment + displaced)",
            language="text")

# -------------------------------------------------------------------- benchmark
with tabs[4]:
    summ_path = RESULTS / "summary.json"
    c1, c2 = st.columns([3, 1])
    seeds_txt = c2.text_input("Seeds for a new run", "42,43,44,45,46")
    if c2.button("Run benchmark now", type="primary"):
        seeds = [int(x) for x in seeds_txt.split(",") if x.strip()]
        bar = c2.progress(0.0)
        total = len(SCENARIO_NAMES) * len(seeds) * 3
        done = {"n": 0}

        def tick(label: str) -> None:
            done["n"] += 1
            bar.progress(done["n"] / total, text=label)

        run_report(SCENARIO_NAMES, seeds, RESULTS, progress=tick,
                   command=f"dashboard: run benchmark (seeds {seeds_txt})")
        from aerosync.viz.charts import metric_bars, overview_grid

        for mk in METRIC_LABELS:
            metric_bars(json.loads(summ_path.read_text()), mk,
                        RESULTS / "charts" / f"bars_{mk}.png")
        overview_grid(json.loads(summ_path.read_text()), RESULTS / "charts" / "overview.png")
        st.toast("Benchmark finished")
    if not summ_path.exists():
        c1.warning("No results yet - run `aerosync report` or press the button.")
    else:
        summ = json.loads(summ_path.read_text())
        c1.markdown(f"Real runs from `results/summary.json` - mean over seeds "
                    f"**{summ['seeds']}**. Command: `{summ.get('command', '')}`")
        metric = st.selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
        res_s = summ["results"]
        scs = [s for s in SCENARIO_NAMES if s in res_s]
        fig = go.Figure()
        for ctl in ["fcfs", "greedy", "aerosync"]:
            fig.add_trace(go.Bar(
                x=[s.replace("_", " ") for s in scs],
                y=[res_s[s][ctl][metric]["mean"] for s in scs],
                error_y={"type": "data", "array": [res_s[s][ctl][metric]["std"] for s in scs],
                         "color": "#52514e"},
                name=ctl, marker={"color": CTL_COLORS[ctl], "line": {"color": "white",
                                                                     "width": 2}},
                text=[f"{res_s[s][ctl][metric]['mean']:,.1f}" for s in scs],
                textposition="inside", insidetextanchor="end",
                textfont={"color": "white"}))
        fig.update_layout(barmode="group", height=460, plot_bgcolor="#fff",
                          margin={"l": 10, "r": 10, "t": 80, "b": 10},
                          title=METRIC_LABELS[metric] + (" (lower is better)"
                                                         if LOWER_IS_BETTER[metric]
                                                         else " (higher is better)"),
                          legend={"orientation": "h", "y": 1.02, "x": 1,
                                  "xanchor": "right", "yanchor": "bottom"})
        st.plotly_chart(fig, width="stretch")
        table = []
        for s in scs:
            for ctl in ["fcfs", "greedy", "aerosync"]:
                table.append({"scenario": s, "controller": ctl,
                              **{METRIC_LABELS[k]: res_s[s][ctl][k]["mean"]
                                 for k in METRIC_LABELS}})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        ov = RESULTS / "charts" / "overview.png"
        if ov.exists():
            st.image(str(ov), caption="All headline metrics (results/charts/overview.png)")

# ------------------------------------------------------------------ about/peas
with tabs[5]:
    st.subheader("PEAS - Gate Agent")
    peas = pd.DataFrame([
        {"": "Performance", "Gate Agent": "Low passenger walking, few remote / terminal-mismatch "
         "parkings, no buffer violations, low hold delay, few gate changes, balanced load with "
         "neighbouring stands."},
        {"": "Environment", "Gate Agent": "Its own stand schedule, the 1-hop neighbouring stands "
         "(via STATE_UPDATE only), closures, CFPs from the coordinator; dynamic apron with "
         "delays and faults."},
        {"": "Actuators", "Gate Agent": "BID / REFUSE replies, STATE_UPDATE broadcasts, accepting "
         "AWARDs (commitments)."},
        {"": "Sensors", "Gate Agent": "CFP, AWARD, FAULT and neighbour STATE_UPDATE messages."},
    ])
    st.table(peas.set_index(""))
    st.subheader("Environment characteristics")
    cards = [
        ("Partially observable", "Arrival delays stay hidden until revealed 20-60 min out; gate "
         "agents see only their own and neighbours' schedules."),
        ("Stochastic", "Seeded random delays, plus closures, breakdowns and comms loss."),
        ("Sequential", "Each award constrains later flights at that stand and its neighbours."),
        ("Dynamic", "The world advances every 5 minutes while agents deliberate."),
        ("Multi-agent", "Flight, gate (x16), ground-vehicle and coordinator agents."),
        ("Cooperative", "All agents minimise one shared airport-wide cost."),
    ]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, (title, body) in zip(cols, row, strict=False):
            col.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                         unsafe_allow_html=True)
        st.write("")
    st.subheader("Architecture")
    arch = DIAGRAMS / "architecture.png"
    if not arch.exists():
        from aerosync.viz.charts import architecture_diagram

        architecture_diagram(arch)
    st.image(str(arch), width="stretch")
    st.subheader("Key parameters")
    st.markdown(
        f"- Timestep **{STEP_MIN} min**, separation buffer **{BUFFER_MIN} min**, look-ahead "
        f"**{LOOKAHEAD_MIN} min**, max hold **{MAX_HOLD_MIN} min**\n"
        f"- Coordination weight **alpha = {ALPHA}**; bid weights `{DEFAULT_WEIGHTS.as_dict()}`\n"
        "- 12 contact gates (G3, G4, G9, G10 wide-body capable) in T1/T2 + 4 remote stands")
