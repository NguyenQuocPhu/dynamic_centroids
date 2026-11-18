#!/usr/bin/env bash
# Whole campaign, unattended. Run inside tmux (survives ssh disconnects):
#
#   tmux new -s paper 'bash paper/run_background.sh 2>&1 | tee -a runs/paper_driver.log'
#   # detach: Ctrl-b d      re-attach: tmux attach -t paper
#   # or:  nohup bash paper/run_background.sh >> runs/paper_driver.log 2>&1 &
#
# Phases (each resumable -- re-running skips DONE runs):
#   A  E1 for every target, one target at a time   (main table first: most important)
#   B  E2 / E3 / E4 / E4b                          (llama31_8b + qwen25_7b only)
#   C  eval: collect -> judges -> analyze
#
# Knobs (env): JOBS (parallel runs, default 6), PHASES (default "A B C"),
#   TARGETS_A (order of phase A), PY (python), SKIP_JUDGES ("sr_glm sr_kimi" ...)
# Rate cap of the shared Qwen proxy: echo 10 > ~/.llm_max_rps (live, no restart)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-6}
PHASES=${PHASES:-"A B C"}
TARGETS_A=${TARGETS_A:-"llama31_8b qwen25_7b gemma2_9b ministral_8b"}
TARGETS_B=${TARGETS_B:-"llama31_8b qwen25_7b"}
SKIP_JUDGES=${SKIP_JUDGES:-""}
mkdir -p runs
[ -f ~/.llm_max_rps ] || echo 10 > ~/.llm_max_rps

stamp() { date '+%F %T'; }
run_exp() {  # $1 = target, rest = experiments
  local t=$1; shift
  for pass in 1 2; do   # pass 2 retries transient failures once
    echo "[$(stamp)] ===== $* on $t (pass $pass)"
    $PY paper/run_matrix.py run --exp "$@" --targets "$t" --jobs "$JOBS" $([ $pass = 2 ] && echo --retry_failed)
  done
  $PY paper/backup_runs.py || true
}

if [[ " $PHASES " == *" A "* ]]; then
  for t in $TARGETS_A; do run_exp "$t" E1; done
fi
if [[ " $PHASES " == *" B "* ]]; then
  for t in $TARGETS_B; do run_exp "$t" E2 E3 E4 E4b; done
fi
if [[ " $PHASES " == *" C "* ]]; then
  echo "[$(stamp)] ===== eval"
  $PY paper/eval/judge_panel.py collect
  for j in string_match harmbench_cls llamaguard3 sr_glm sr_kimi; do
    [[ " $SKIP_JUDGES " == *" $j "* ]] && continue
    $PY paper/eval/judge_panel.py judge --judge "$j" || echo "judge $j failed (fix judges.yaml / endpoint, then re-run phase C)"
  done
  $PY paper/eval/analyze.py > /dev/null && echo "report: runs/paper/_eval/report/summary.md"
  $PY paper/backup_runs.py || true
fi
$PY paper/run_matrix.py status
echo "[$(stamp)] all requested phases finished"
