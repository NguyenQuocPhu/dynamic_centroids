#!/usr/bin/env python3
"""
Experiment driver for the paper (baseline vs beam vs MCTS).

  python paper/run_matrix.py plan   --exp E1                 # list runs, nothing executed
  python paper/run_matrix.py run    --exp E1 --jobs 6        # run (resumable; DONE runs skipped)
  python paper/run_matrix.py run    --exp E1 --targets llama31_8b   # one target at a time (GPU cost)
  python paper/run_matrix.py status [--exp E1]               # progress table
  python paper/run_matrix.py check-endpoints                 # ping proxy + every target endpoint

Every run gets its own directory
    <runs_root>/<method>[-<variant>]/<target>/<dataset>/b<budget>_s<seed>/
containing cmd.txt, stdout.log, logs/ (the method's own JSON logs) and either
DONE.json (success + query accounting) or FAILED.json. A run is identified by
that path, so experiments that share a configuration share the run.

Environment (.env at repo root, parsed here and passed to every child):
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL_ID   attacker + in-loop judge (Qwen3.5-35B)
  MODAL_TARGET_API_KEY                                 vLLM --api-key of the Modal targets
  TARGET_URL_<KEY>  or  MODAL_WORKSPACE[/MODAL_ENVIRONMENT]  target endpoint urls
  PAPER_RUNS                                           overrides runs_root
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
EXP_FILE = REPO / "paper" / "experiments.yaml"
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


# ───────────────────────────── config / env ──────────────────────────────────

def load_dotenv_file(path):
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        env[k] = v
    return env


def base_env():
    env = dict(os.environ)
    for k, v in load_dotenv_file(REPO / ".env").items():
        env.setdefault(k, v)   # real environment wins over .env
    return env


def load_spec():
    with open(EXP_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def runs_root(spec, env):
    root = env.get("PAPER_RUNS") or spec["runs_root"]
    p = Path(root)
    return p if p.is_absolute() else REPO / p


def target_url(spec, key, env):
    t = spec["targets"][key]
    if env.get(t["url_env"]):
        return env[t["url_env"]].rstrip("/")
    ws = env.get("MODAL_WORKSPACE")
    if not ws:
        return None
    menv = env.get("MODAL_ENVIRONMENT")
    prefix = f"{ws}-{menv}" if menv and menv != "main" else ws
    label = f"paper-targets-{t['modal_fn']}".replace("_", "-")
    return f"https://{prefix}--{label}.modal.run/v1"


# ───────────────────────────── expansion ─────────────────────────────────────

def expand(spec, exp_names, only_targets=None, only_methods=None, only_datasets=None):
    runs, seen = [], set()
    for name in exp_names:
        e = spec["experiments"][name]
        targets = list(spec["targets"]) if e["targets"] == "all" else e["targets"]
        datasets = spec["datasets"] if e["datasets"] == "all" else e["datasets"]
        variants = e.get("variants") or {"": {}}
        for method in e["methods"]:
            for vname, override in variants.items():
                for tgt in targets:
                    for ds in datasets:
                        for budget in e["budgets"]:
                            for seed in e["seeds"]:
                                if only_targets and tgt not in only_targets:
                                    continue
                                if only_methods and method not in only_methods:
                                    continue
                                if only_datasets and ds not in only_datasets:
                                    continue
                                arm = f"{method}-{vname}" if vname else method
                                rel = f"{arm}/{tgt}/{ds}/b{budget}_s{seed}"
                                if name == "SMOKE":
                                    rel = "SMOKE/" + rel
                                if rel in seen:
                                    continue
                                seen.add(rel)
                                runs.append(dict(exp=name, method=method, variant=vname, override=override,
                                                 target=tgt, dataset=ds, budget=budget, seed=seed, rel=rel,
                                                 extra_args=e.get("extra_args", [])))
    # cluster by target so one Modal container serves a contiguous block of runs
    order = {t: i for i, t in enumerate(spec["targets"])}
    runs.sort(key=lambda r: (order.get(r["target"], 99), r["exp"], r["dataset"], r["seed"], r["method"]))
    return runs


def build_cmd(spec, run, run_dir, python):
    m = spec["methods"][run["method"]]
    args = list(m["args"])
    # apply variant overrides (--flag value) onto the method's default args
    for flag, val in (run["override"] or {}).items():
        if flag in args:
            args[args.index(flag) + 1] = str(val)
        else:
            args += [flag, str(val)]
    ds_path = m["dataset_path"].format(ds=run["dataset"])
    cmd = [python, "-m", "rainbowplus.rainbowplus",
           "--config_file", str(REPO / m["config"]),
           "--dataset", ds_path,
           "--seed", str(run["seed"]),
           "--target_query_budget", str(run["budget"]),
           "--log_dir", str(run_dir / "logs")] + args + list(run["extra_args"])
    return cmd


# ───────────────────────────── execution ─────────────────────────────────────

def find_final_log(spec, run, run_dir):
    pat = spec["methods"][run["method"]]["final_log_glob"]
    hits = sorted(glob.glob(str(run_dir / "logs" / pat), recursive=True), key=os.path.getmtime)
    return hits[-1] if hits else None


def query_totals(log_path):
    try:
        with open(log_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    calls = d.get("llm_call_logs", {}) or {}
    tot = lambda r: (calls.get(r, {}) or {}).get("total_queries")
    return {"target": tot("target_llm"), "attack": tot("attack_llm"), "eval": tot("eval_llm")}


def wait_ready(url, api_key, timeout_s=1200):
    """Poll <url>/models until the (possibly cold) endpoint answers."""
    t0, delay = time.time(), 10
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(url + "/models", headers={"Authorization": f"Bearer {api_key}",
                                                                   "User-Agent": "paper-run-matrix"})
            with urllib.request.urlopen(req, timeout=60) as r:
                ids = [m["id"] for m in json.load(r).get("data", [])]
                return True, ids
        except Exception as e:
            last = e
        time.sleep(delay)
        delay = min(delay * 1.5, 60)
    return False, str(last)


def run_one(spec, run, root, env0, python, debug_target, stop_flag):
    if stop_flag.is_set():
        return run["rel"], "skipped"
    run_dir = root / run["rel"]
    if (run_dir / "DONE.json").exists():
        return run["rel"], "already-done"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "FAILED.json").unlink(missing_ok=True)

    env = dict(env0)
    if debug_target:
        env["TARGET_BASE_URL"], env["TARGET_MODEL_ID"] = debug_target
        env["TARGET_API_KEY"] = env.get("DEBUG_TARGET_API_KEY") or env.get("OPENAI_API_KEY", "")
    else:
        env["TARGET_BASE_URL"] = target_url(spec, run["target"], env0)
        env["TARGET_MODEL_ID"] = spec["targets"][run["target"]]["model_id"]
        env["TARGET_API_KEY"] = env0.get("MODAL_TARGET_API_KEY", "EMPTY")
    wt = REPO / spec["worktrees"][run["method"]]
    env["PYTHONPATH"] = str(wt) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LLM_RATE_LOCKFILE", str(Path.home() / ".llm_rate_limit.lock"))

    cmd = build_cmd(spec, run, run_dir, python)
    meta = {k: run[k] for k in ("exp", "method", "variant", "target", "dataset", "budget", "seed")}
    meta.update(target_model=env["TARGET_MODEL_ID"], target_url=env["TARGET_BASE_URL"],
                attacker_model=env.get("OPENAI_MODEL_ID"), worktree=str(wt),
                git_commit=_git_head(wt), cmd=cmd)
    (run_dir / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n", encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t0 = time.time()
    with open(run_dir / "stdout.log", "ab") as out:
        out.write(f"\n===== start {time.ctime()} =====\n".encode())
        out.flush()
        rc = subprocess.call(cmd, cwd=str(wt), env=env, stdout=out, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    final = find_final_log(spec, run, run_dir)
    totals = query_totals(final) if final else {}
    result = dict(meta, returncode=rc, wall_s=round(wall, 1), final_log=final, queries=totals,
                  finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    tq = totals.get("target") or 0
    if rc == 0 and final and tq > 0:
        result["budget_reached"] = tq >= 0.9 * run["budget"]
        (run_dir / "DONE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return run["rel"], f"OK  target={tq} attack={totals.get('attack')} eval={totals.get('eval')} {wall/60:.1f}min"
    result["reason"] = ("nonzero exit" if rc != 0 else "no final log" if not final else "zero target queries")
    (run_dir / "FAILED.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return run["rel"], f"FAIL ({result['reason']}, rc={rc}) -> {run_dir / 'stdout.log'}"


def _git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# ───────────────────────────── commands ──────────────────────────────────────

def cmd_plan(spec, runs, root):
    by = {}
    for r in runs:
        st = "done" if (root / r["rel"] / "DONE.json").exists() else \
             "failed" if (root / r["rel"] / "FAILED.json").exists() else "todo"
        by.setdefault((r["exp"], r["method"] + (f"-{r['variant']}" if r["variant"] else ""), r["target"]), {}) \
          .setdefault(st, 0)
        by[(r["exp"], r["method"] + (f"-{r['variant']}" if r["variant"] else ""), r["target"])][st] += 1
    print(f"{'exp':6} {'arm':14} {'target':14} {'todo':>5} {'done':>5} {'failed':>6}")
    for (e, a, t), c in by.items():
        print(f"{e:6} {a:14} {t:14} {c.get('todo', 0):5} {c.get('done', 0):5} {c.get('failed', 0):6}")
    print(f"\n{len(runs)} runs total under {root}")


def cmd_check(spec, env):
    ok = True
    pairs = [("proxy (attacker/judge)", env.get("OPENAI_BASE_URL", "").rstrip("/"), env.get("OPENAI_API_KEY", ""))]
    for k in spec["targets"]:
        pairs.append((f"target {k}", target_url(spec, k, env), env.get("MODAL_TARGET_API_KEY", "EMPTY")))
    for name, url, key in pairs:
        if not url:
            print(f"  ✗ {name}: no url (set {name.split()[-1].upper()} url env or MODAL_WORKSPACE)")
            ok = False
            continue
        good, info = wait_ready(url, key, timeout_s=900 if name.startswith("target") else 60)
        print(f"  {'✓' if good else '✗'} {name}: {url}  -> {info if good else 'ERROR ' + str(info)}")
        ok &= good
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "status", "check-endpoints"])
    ap.add_argument("--exp", nargs="+", default=None, help="experiment names (default: all except SMOKE)")
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--jobs", type=int, default=6, help="parallel runs (proxy rate cap is shared)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--retry_failed", action="store_true", help="also re-run runs that have FAILED.json")
    ap.add_argument("--max_runs", type=int, default=0, help="stop after launching this many (0 = all)")
    ap.add_argument("--debug_target", nargs=2, metavar=("BASE_URL", "MODEL_ID"),
                    help="TESTING ONLY: send every target query to this endpoint/model instead of Modal")
    a = ap.parse_args()

    spec = load_spec()
    env = base_env()
    root = runs_root(spec, env)
    exps = a.exp or [e for e in spec["experiments"] if e != "SMOKE"]
    runs = expand(spec, exps, a.targets, a.methods, a.datasets)

    if a.action in ("plan", "status"):
        cmd_plan(spec, runs, root)
        return
    if a.action == "check-endpoints":
        sys.exit(0 if cmd_check(spec, env) else 1)

    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_ID") if not env.get(k)]
    if missing:
        sys.exit(f"missing {missing} in env/.env (attacker + judge endpoint)")
    if not a.retry_failed:
        runs = [r for r in runs if not (root / r["rel"] / "FAILED.json").exists()]
    runs = [r for r in runs if not (root / r["rel"] / "DONE.json").exists()]
    if a.max_runs:
        runs = runs[: a.max_runs]
    if not runs:
        log("nothing to run")
        return

    # warm + verify each target endpoint that this batch needs
    if not a.debug_target:
        for tgt in dict.fromkeys(r["target"] for r in runs):
            url = target_url(spec, tgt, env)
            if not url:
                sys.exit(f"no endpoint for target {tgt}: set {spec['targets'][tgt]['url_env']} or MODAL_WORKSPACE")
            log(f"waiting for {tgt} @ {url} ...")
            good, info = wait_ready(url, env.get("MODAL_TARGET_API_KEY", "EMPTY"))
            if not good:
                sys.exit(f"target {tgt} not reachable: {info}")
            want = spec["targets"][tgt]["model_id"]
            if want not in info:
                sys.exit(f"target {tgt} serves {info}, expected {want}")
            log(f"  ready: {info}")

    log(f"launching {len(runs)} runs with {a.jobs} workers -> {root}")
    stop = threading.Event()
    done = fail = 0
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(run_one, spec, r, root, env, a.python, a.debug_target, stop): r for r in runs}
        try:
            for fut in cf.as_completed(futs):
                rel, msg = fut.result()
                done += msg.startswith("OK")
                fail += msg.startswith("FAIL")
                log(f"[{done + fail}/{len(runs)}] {rel}: {msg}")
                if fail >= 5 and done == 0:
                    log("5 failures and no success -- stopping (check stdout.log of a failed run)")
                    stop.set()
        except KeyboardInterrupt:
            stop.set()
            log("interrupted: running jobs finish, queued jobs skipped")
    log(f"finished: {done} OK, {fail} failed")


if __name__ == "__main__":
    main()
