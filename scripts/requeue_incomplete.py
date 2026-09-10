#!/usr/bin/env python3
"""Put a suite's queue back in agreement with what is actually on disk.

    python3 scripts/requeue_incomplete.py SUITE [--apply]

A queue entry is a *claim* about a run; `metrics.jsonl` is the run. When the two
disagree the queue is wrong, and it is wrong in the direction that hides work:

  * `status: failed` -- the job died. It stays failed forever, so a relaunch sees
    `pending 0` and exits reporting the old failures as though they were fresh.
    G5's family probe hit exactly this: the recipe lock was repaired, the stage
    launched cleanly, and still ran nothing.
  * `status: done` with no `done` record on disk -- the run directory was moved,
    quarantined or deleted after the fact. The queue keeps vouching for data that
    is not there, which is the same hazard as a check that could not run
    reporting the same result as one that ran and passed.

Both go back to `pending`. A job whose metrics DO carry a `done` record is left
alone whatever the queue says -- the artifact wins over the claim, never the
other way round.

`pending` and `held` are not claims about a run and are never touched. A held
job is one `--hold` parked while the workers drain; it has no `done` record
because it has not run, and releasing it here would start the very jobs the
operator held.

Dry-run by default; `--apply` writes. No GPU, no network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def has_done_record(run_dir: Path) -> bool:
    """True only if metrics.jsonl carries an actual `done` event."""
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return False
    for line in m.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            if json.loads(line).get("event") == "done":
                return True
        except json.JSONDecodeError:
            continue
    return False


def reconcile(suite: str, apply: bool) -> int:
    root = ROOT / "nanolab" / "out" / suite
    qpath = root / "queue.json"
    if not qpath.exists():
        raise SystemExit(f"no queue at {qpath}")
    q = json.loads(qpath.read_text(encoding="utf-8"))
    jobs = q if isinstance(q, list) else q.get("jobs")
    if not isinstance(jobs, list):
        raise SystemExit(f"unrecognised queue shape in {qpath}")

    changed = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        jid = job.get("id")
        status = job.get("status")
        # Only `done`, `failed` and `running` assert that a run happened; those
        # are the claims disk can contradict. `pending` and `held` assert
        # nothing. `held` especially: `--hold` parks pending jobs so the workers
        # can drain, so a held job has no `done` record BY CONSTRUCTION, and
        # requeueing one releases a hold that only `--unhold` should release --
        # this tool's own defect inverted, a queue claiming work that must not
        # run rather than work that is not there.
        if not jid or status in ("pending", "held"):
            continue
        on_disk = has_done_record(root / jid)
        if on_disk:
            continue                      # the artifact wins; leave it alone
        why = ("failed, no run on disk" if status == "failed"
               else f"status {status!r} but no done record on disk")
        changed.append((jid, status, why))
        job["status"] = "pending"
        for k in ("worker", "started", "finished", "detail"):
            job.pop(k, None)

    print(f"=== {suite}: {len(jobs)} entries, {len(changed)} to requeue ===")
    for jid, was, why in changed:
        print(f"  {jid}\n      was {was!r} -- {why}")
    if not changed:
        print("  queue already agrees with disk; nothing to do")
        return 0
    if not apply:
        print("\n  DRY RUN -- re-run with --apply to write")
        return 0
    qpath.write_text(json.dumps(q, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {qpath}: {len(changed)} job(s) back to pending")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("--apply", action="store_true", help="write the change")
    a = ap.parse_args()
    raise SystemExit(reconcile(a.suite, a.apply))


if __name__ == "__main__":
    main()
