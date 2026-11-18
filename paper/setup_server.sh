#!/usr/bin/env bash
# One-time server setup for the paper runs. Idempotent: safe to re-run.
#
#   git clone <repo> && cd <repo> && git checkout exp/paper
#   bash paper/setup_server.sh
#
# Creates: .wt/beam (exp/paper-beam), .wt/mcts (exp/paper-mcts), .venv, NLTK data.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-$(git remote | head -1)}

echo "== git: fetch $REMOTE and create worktrees"
git fetch "$REMOTE" --prune
for pair in "beam:exp/paper-beam" "mcts:exp/paper-mcts"; do
  name=${pair%%:*}; br=${pair#*:}
  if [ -d ".wt/$name" ]; then
    echo "   .wt/$name exists -> pulling $br"
    git -C ".wt/$name" pull --ff-only "$REMOTE" "$br" || echo "   (pull failed; check manually)"
  else
    if git show-ref --verify --quiet "refs/heads/$br"; then
      git worktree add ".wt/$name" "$br"
    else
      git worktree add -b "$br" ".wt/$name" "$REMOTE/$br"
    fi
  fi
done
git worktree list

echo "== python venv (.venv)"
PYBIN=${PYBIN:-python3}
[ -d .venv ] || "$PYBIN" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
if ! python -c "import torch" 2>/dev/null; then
  pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q -r paper/requirements-paper.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab')]"

echo "== sanity"
python - <<'EOF'
import importlib
for m in ["openai", "easyjailbreak", "fastchat", "nltk", "sentence_transformers", "pandas", "scipy", "modal", "yaml"]:
    importlib.import_module(m)
print("imports ok")
EOF
[ -f .env ] || { echo "!! .env missing: cp .env.example .env and fill it (see paper/SERVER_HANDOFF.md)"; exit 1; }
python paper/run_matrix.py plan | tail -3
echo
echo "next:"
echo "  1) modal setup / modal secret create ... / modal deploy   (SERVER_HANDOFF.md §2)"
echo "  2) python paper/run_matrix.py check-endpoints"
echo "  3) python paper/run_matrix.py run --exp SMOKE --jobs 3"
echo "  4) tmux new -s paper 'bash paper/run_background.sh'"
