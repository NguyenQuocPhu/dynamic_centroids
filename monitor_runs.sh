#!/bin/bash
# Live monitor for whichever attack batch is running.
#
# Auto-detects the active logs (any *.log under runs/ ($DC_RUNS) touched in
# the last 3 min) instead of a hardcoded directory -- a hardcoded /tmp/base read
# 0 req/s for the whole beam/mcts batch, which writes to /tmp/conf.
#
# Reports BOTH signals, because they can disagree:
#   req/s  - counted from httpx "200 OK" lines; only present once a run's branch
#            enables httpx INFO logging, so it reads 0 for older runs that
#            predate that (the run is still fine -- check iter/s).
#   iter/s - iterations advanced across all active runs; works on every branch
#            and is the ground truth for "are the runs actually progressing".
#
#   ./monitor_runs.sh        # refresh every 5s until Ctrl-C
#   ./monitor_runs.sh 3      # every 3s
export LC_NUMERIC=C
INT=${1:-5}
active() { find "${DC_RUNS:-$(dirname "$(readlink -f "$0")")/runs}" -name '*.log' -mmin -3 2>/dev/null; }
count_ok() { local fs; fs=$(active); [ -z "$fs" ] && { echo 0; return; }; echo "$fs" | xargs grep -h "proxy.*200 OK" 2>/dev/null | wc -l; }
count_iter() { local fs t=0 n; fs=$(active); [ -z "$fs" ] && { echo 0; return; }
  for f in $fs; do n=$(grep -oE '[Ii]teration:? ?[0-9]+' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+'); t=$((t+${n:-0})); done; echo "$t"; }

po=$(count_ok); pi=$(count_iter); pt=$(date +%s)
printf "%-9s %8s %8s %5s %7s %6s  %s\n" "giờ" "req/s" "iter/s" "run" "429/to" "load" "batch"
while :; do
  sleep "$INT"
  co=$(count_ok); ci=$(count_iter); t=$(date +%s); dt=$((t-pt)); [ "$dt" -le 0 ] && dt=1
  rate=$(echo "scale=1;($co-$po)/$dt" | bc); irate=$(echo "scale=2;($ci-$pi)/$dt" | bc)
  runs=$(ps -eo args | grep '[r]ainbowplus' | grep -o 'log_dir [^ ]*/runs/[a-z]*/[^ ]*' | sort -u | wc -l)
  which=$(ps -eo args | grep '[r]ainbowplus' | grep -o 'log_dir [^ ]*/runs/[a-z]*/[a-z]*_' | sed 's|.*/||;s|_$||' | sort -u | tr '\n' ',' | sed 's/,$//')
  fs=$(active)
  e4=0; eto=0
  if [ -n "$fs" ]; then
    e4=$(echo "$fs" | xargs grep -hc 'Rate limit exceeded' 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    eto=$(echo "$fs" | xargs grep -hc 'Request timed out' 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  fi
  ld=$(uptime | grep -oE 'average: [0-9]+' | grep -oE '[0-9]+')
  printf "%-9s %8s %8s %5s %7s %6s  %s\n" "$(date +%H:%M:%S)" "$rate" "$irate" "$runs" "${e4:-0}/${eto:-0}" "$ld" "${which:-none}"
  po=$co; pi=$ci; pt=$t
done
