#!/usr/bin/env bash
# AeroSync - scripted ~3 minute terminal demo.
#   bash demo.sh                 # full demo, ends by launching the dashboard
#   PAUSE=0 DEMO_SERVE=0 bash demo.sh   # fast, non-interactive run (CI / verification)
set -euo pipefail

PAUSE="${PAUSE:-4}"
DEMO_SERVE="${DEMO_SERVE:-1}"
if command -v aerosync >/dev/null 2>&1; then
  AS=(aerosync)
else
  AS=(python -m aerosync)
fi
export PYTHONIOENCODING=utf-8

step() {
  echo
  printf '\033[1;44;97m  %s  \033[0m\n' "$1"
  printf '\033[2m$ aerosync %s\033[0m\n' "${*:2}"
  sleep "$PAUSE"
  "${AS[@]}" "${@:2}"
  sleep "$PAUSE"
}

step "1/7  The four seeded scenarios" scenarios
step "2/7  Simulate the rush hour with AeroSync (CSP + Contract Net + repair + A*)" \
  simulate --scenario rush_hour --controller aerosync --seed 42
step "3/7  Inject a gate closure and watch min-conflicts repair it" \
  disrupt --scenario normal_day --event gate_closure --gate G4 --at 30
step "4/7  Explain one assignment: every bid, the winner and why" \
  explain --flight AI302
step "5/7  A* ground-vehicle routing on the live apron graph" \
  route --from HANGAR --to G7 --seed 42
step "6/7  Same seed, three controllers: fcfs vs greedy vs aerosync" \
  compare --scenario storm_disruption --seed 42

if [ "$DEMO_SERVE" = "1" ]; then
  step "7/7  Launch the web dashboard (Ctrl+C to stop)" serve
else
  echo
  echo "7/7  Dashboard step skipped (DEMO_SERVE=0). Start it with: aerosync serve"
fi
