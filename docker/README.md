# Dev container

CUDA 13 (Ubuntu 24.04) container with a passwordless-sudo `dev` user and the
project's 3 compared branches mounted side by side:

| Path in container | Host path | Branch |
|---|---|---|
| `/workspace/main` | `/home/phunq/5080_cuda13/Research/dynamic_centroids` | `main` |
| `/workspace/beam` | `/tmp/wt-beam3` | `algo/beam-search` |
| `/workspace/mcts` | `/tmp/wt-mcts3` | `algo/mcts` |

These are git worktrees of the same repo, so commits/branch switches inside
the container are visible on the host and vice versa (they share one
`.git`). No agent/LLM CLI is baked into the image — install what you want to
use once you're inside (e.g. `curl -fsSL https://claude.ai/install.sh | bash`
for Claude Code, or `pipx install aider-chat`, etc.) — `sudo` is available
with no password.

## Build & run

```bash
cd docker
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d
docker compose exec dev bash
```

`HOST_UID`/`HOST_GID` make the in-container `dev` user match your host user,
so files created in the mounted worktrees keep normal ownership on the host.

GPU passthrough is on by default (`gpus: all`); check it works with:

```bash
docker compose exec dev nvidia-smi
```

## Stop / remove

```bash
docker compose down
```

## Notes

- `/tmp/wt-beam3` and `/tmp/wt-mcts3` are worktrees under `/tmp` — they don't
  survive a host reboot on systems that clear `/tmp`. Recreate them with
  `git worktree add /tmp/wt-beam3 algo/beam-search` (and
  `git worktree add /tmp/wt-mcts3 algo/mcts`) if they're gone before running
  `docker compose up`.
- The `.env` file with `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL_ID`
  lives in each worktree already (not baked into the image); it's visible at
  e.g. `/workspace/main/.env` once mounted.
