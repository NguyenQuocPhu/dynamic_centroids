#!/usr/bin/env python3
"""
Snapshot everything needed to rebuild the paper results into ONE tar.gz:
per run {DONE.json, run_meta.json, cmd.txt, final log} + the whole _eval/ dir.
(The periodic per-iteration logs are skipped -- they are large and redundant.)

  python paper/backup_runs.py                 # -> backups/paper_runs_<ts>.tar.gz (keeps newest 5)
  BACKUP_CMD='rclone copy {file} gdrive:paper' python paper/backup_runs.py   # also ship it off-box

Lesson from 2026-10-16: a reboot wiped /tmp and every run log with it. Keep
runs/ on a persistent disk AND copy the snapshot somewhere else.
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix as RM  # noqa: E402

KEEP = 5


def main():
    spec, env = RM.load_spec(), RM.base_env()
    root = RM.runs_root(spec, env)
    if not root.exists():
        print("no runs yet")
        return
    out_dir = RM.REPO / "backups"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"paper_runs_{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for done in root.glob("**/DONE.json"):
            d = done.parent
            for name in ("DONE.json", "run_meta.json", "cmd.txt"):
                if (d / name).exists():
                    tar.add(d / name, arcname=str((d / name).relative_to(root.parent)))
            final = json.loads(done.read_text(encoding="utf-8")).get("final_log")
            if final and Path(final).exists():
                tar.add(final, arcname=str(Path(final).resolve().relative_to(root.parent.resolve())))
            n += 1
        if (root / "_eval").exists():
            tar.add(root / "_eval", arcname=str((root / "_eval").relative_to(root.parent)))
    print(f"backup: {n} runs -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for old in sorted(out_dir.glob("paper_runs_*.tar.gz"))[:-KEEP]:
        old.unlink()
    cmd = env.get("BACKUP_CMD")
    if cmd:
        subprocess.call(cmd.format(file=str(out)), shell=True)


if __name__ == "__main__":
    main()
