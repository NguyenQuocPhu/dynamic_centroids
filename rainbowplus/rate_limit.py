"""Cross-process request throttle for the shared LLM endpoint.

Every run is a separate process, so a per-process limit would multiply by the
number of concurrent runs. This gates on a lock file instead, so ALL processes
together stay under one global rate no matter how many run in parallel.

Two independent gates exist:

  * the SHARED gate (default) -- the attacker/judge proxy (Qwen3.5-35B). Rate
    from LLM_RATE_FILE (default ~/.llm_max_rps, live-tunable) or LLM_MAX_RPS,
    falling back to 10 rps. Lock file: LLM_RATE_LOCKFILE.
  * the TARGET gate -- used when the caller's base_url equals TARGET_BASE_URL
    (the per-role target override, e.g. a Modal-hosted vLLM). Target traffic
    must not eat the proxy's budget, and vice versa. Rate from
    TARGET_RATE_FILE / TARGET_MAX_RPS; default 0 = unthrottled (a self-hosted
    vLLM has no shared quota). Lock file: <LLM_RATE_LOCKFILE>.target.

Only the *initiation* of requests is serialised -- once a caller passes the
gate its request proceeds concurrently with the others.
"""
import os
import time

try:
    import fcntl
except ImportError:  # non-POSIX; throttling unavailable
    fcntl = None

_LOCKFILE = os.environ.get("LLM_RATE_LOCKFILE", "/tmp/.llm_rate_limit.lock")


def _norm(url):
    return (url or "").strip().rstrip("/")


def _is_target_url(base_url):
    target = _norm(os.environ.get("TARGET_BASE_URL"))
    return bool(target) and _norm(base_url) == target


def _read_rate(file_var, file_default, env_var, env_default):
    ratefile = os.environ.get(file_var, file_default) if (file_var or file_default) else None
    if ratefile:
        try:
            with open(ratefile) as rf:
                return float(rf.read().strip() or 0)
        except (OSError, ValueError):
            pass
    try:
        return float(os.environ.get(env_var, env_default) or env_default)
    except ValueError:
        return float(env_default)


def throttle(base_url=None):
    """Block until this process is allowed to issue its next request.

    base_url: the endpoint the caller is about to hit. When it is the target
    override endpoint (TARGET_BASE_URL) the separate target gate is used;
    anything else (or None, for old call sites) uses the shared gate.
    """
    if _is_target_url(base_url):
        rps = _read_rate("TARGET_RATE_FILE", "", "TARGET_MAX_RPS", "0")
        lockfile = _LOCKFILE + ".target"
    else:
        # Rate is read fresh on every call, preferring a live-tunable file
        # (LLM_RATE_FILE, default ~/.llm_max_rps -- NOT /tmp, a reboot wiped it
        # once) over the env var, so the cap can be changed for all running
        # processes without restarting. If both are missing the cap is 10.
        rps = _read_rate("LLM_RATE_FILE", os.path.expanduser("~/.llm_max_rps"), "LLM_MAX_RPS", "10")
        lockfile = _LOCKFILE
    if rps <= 0 or fcntl is None:
        return
    min_interval = 1.0 / rps
    try:
        with open(lockfile, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                last = float(raw) if raw else 0.0
                now = time.time()
                wait = last + min_interval - now
                if wait > 0:
                    time.sleep(wait)
                    now = last + min_interval
                f.seek(0)
                f.truncate()
                f.write(repr(now))
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        # Never let a throttling problem take down a multi-hour run.
        return
