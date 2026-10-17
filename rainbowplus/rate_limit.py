"""Cross-process request throttle for the shared LLM endpoint.

Every run is a separate process, so a per-process limit would multiply by the
number of concurrent runs. This gates on a lock file instead, so ALL processes
together stay under one global rate no matter how many run in parallel.

Enable with the env var LLM_MAX_RPS (requests/second, float). Unset or <=0
disables throttling entirely, so existing behaviour is unchanged by default.
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


def throttle():
    """Block until this process is allowed to issue its next request."""
    # Rate is read fresh on every call, preferring a live-tunable file
    # (LLM_RATE_FILE, default ~/.llm_max_rps -- NOT /tmp, a reboot wiped it once)
    # over the env var, so the cap can be changed for all running processes
    # without restarting -- just write a new number to that file. Env LLM_MAX_RPS
    # is the fallback; if both are missing the cap is 10 (the user's chosen
    # value), never "unthrottled".
    rps = 0.0
    ratefile = os.environ.get("LLM_RATE_FILE", os.path.expanduser("~/.llm_max_rps"))
    try:
        with open(ratefile) as rf:
            rps = float(rf.read().strip() or 0)
    except (OSError, ValueError):
        try:
            rps = float(os.environ.get("LLM_MAX_RPS", "10") or 10)
        except ValueError:
            return
    if rps <= 0 or fcntl is None:
        return
    min_interval = 1.0 / rps
    try:
        with open(_LOCKFILE, "a+") as f:
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
