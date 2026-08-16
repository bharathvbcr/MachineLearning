"""Resumable equal-data native optimizer funnel and champion handoff.

The runner executes one job at a time on the local Metal engine, records the
exact argv and scalar result, and only materializes the next stage after every
job in the current stage has a finite final EMA validation BPB.  It never
substitutes an unsupported optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .optimizer_funnel import BASE_LRS, MUON_FAMILY, NATIVE_BLOCKERS, write_native_plan


ROOT = Path(__file__).resolve().parents[1]
METAL_DIR = ROOT / "Rust_MLKit/arch_02_value_resid/metal-native"
TRAIN_BIN = METAL_DIR / "target/release/train"
GATE_BIN = METAL_DIR / "target/release/exact_gate"
DEFAULT_PLAN = ROOT / "research/native-optimizer-funnel.json"
DEFAULT_STUDY = ROOT / "research/optimizer-study.json"
DEFAULT_CHAMPION = ROOT / "research/champion-run.json"
DEFAULT_DATA = ROOT / "parameter-golf/data/datasets/fineweb10B_sp1024"
DEFAULT_TOKEN_BYTES = ROOT / "Rust_MLKit/arch_02_value_resid/burn-port/token_bytes.json"
STAGE_ORDER = (
    "lr_sweep_16m",
    "stable_500",
    "advance_1000",
    "exact_128m_500",
    "exact_128m_1000",
)


def _atomic_json(path: Path, value: Any) -> None:
    def sanitize(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: sanitize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lr_flags(candidate: str, lr: float) -> list[str]:
    flags = ["--matrix-lr", f"{lr:.9g}"]
    if candidate not in MUON_FAMILY:
        flags += ["--embed-lr", f"{lr:.9g}", "--scalar-lr", f"{lr:.9g}"]
    return flags


def job_argv(job: dict[str, Any], data: Path, token_bytes: Path) -> list[str]:
    preset = "arch02-128m" if job["stage"].startswith("exact_128m") else "16m"
    out = ROOT / job["output"]
    return [
        str(TRAIN_BIN),
        "--preset", preset,
        "--optimizer", job["candidate"],
        "--total-steps", str(job["steps"]),
        "--seed", str(job["seed"]),
        "--eval-every", str(job["steps"]),
        "--log-every", "50",
        "--research-manifest", str(DEFAULT_STUDY),
        "--data-dir", str(data),
        "--token-bytes", str(token_bytes),
        "--no-final-weight-save",
        "--out", str(out),
        *_lr_flags(job["candidate"], float(job["lr"])),
    ]


def _read_result(output: Path) -> dict[str, Any]:
    rows = []
    metrics = output / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    step_rows = [row for row in rows if "step" in row]
    final = next((row["final_ema_sliding_bpb"] for row in reversed(rows)
                  if "final_ema_sliding_bpb" in row), None)
    if final is not None and not math.isfinite(float(final)):
        final = None
    nonfinite = sum(
        int((row.get("research") or {}).get("nonfinite_values", 0)) for row in step_rows
    )
    required_step_keys = ("loss", "grad_norm_global", "step_ms", "current_physical_mb")
    missing_scalar = any(row.get(key) is None for row in step_rows for key in required_step_keys)
    scalars = [
        float(value)
        for row in step_rows
        for key in required_step_keys
        if (value := row.get(key)) is not None
    ]
    finite = (not missing_scalar and final is not None and math.isfinite(float(final))
              and all(map(math.isfinite, scalars)))
    profiled_optim = [float(row["optimizer_ms"]) for row in step_rows
                      if float(row.get("optimizer_ms", 0.0)) > 0.0]
    step_ms = [float(row["step_ms"]) for row in step_rows]
    failure_reason = None
    run_log = output / "run.log"
    if run_log.exists():
        for line in reversed(run_log.read_text(encoding="utf-8", errors="replace").splitlines()):
            if "Error:" in line or "numerical failure" in line:
                failure_reason = line.strip()
                break
    return {
        "finite": finite and nonfinite == 0,
        "validation_bpb": final,
        "mean_logged_step_ms": sum(step_ms) / len(step_ms) if step_ms else None,
        "mean_profiled_optimizer_ms": (
            sum(profiled_optim) / len(profiled_optim) if profiled_optim else None
        ),
        "max_current_physical_mb": max(
            (float(row.get("current_physical_mb", 0.0)) for row in step_rows), default=None
        ),
        "max_dispatches": max((int(row.get("dispatches", 0)) for row in step_rows), default=None),
        "nonfinite_values": nonfinite,
        "logged_steps": len(step_rows),
        "failure_reason": failure_reason,
        "training_loss_trace": [
            {"step": int(row["step"]), "loss": float(row["loss"])}
            for row in step_rows if row.get("loss") is not None
        ],
    }


def _stage_jobs(plan: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return [job for job in plan["jobs"] if job["stage"] == stage]


def _completed_score(job: dict[str, Any]) -> float | None:
    result = job.get("result") or {}
    value = result.get("validation_bpb")
    if job.get("status") != "completed" or not result.get("finite") or value is None:
        return None
    return float(value)


def _current_stage(plan: dict[str, Any]) -> str:
    present = {job["stage"] for job in plan["jobs"]}
    return max((stage for stage in STAGE_ORDER if stage in present), key=STAGE_ORDER.index)


def _tuned_lrs(plan: dict[str, Any]) -> dict[str, float]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in _stage_jobs(plan, "lr_sweep_16m"):
        if _completed_score(job) is not None:
            by_candidate[job["candidate"]].append(job)
    return {
        candidate: float(min(jobs, key=lambda job: _completed_score(job) or math.inf)["lr"])
        for candidate, jobs in by_candidate.items()
    }


def _rank_candidates(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if _completed_score(job) is not None:
            grouped[job["candidate"]].append(job)
    rankings = []
    final_losses = [
        job["result"]["training_loss_trace"][-1]["loss"]
        for candidate_jobs in grouped.values()
        for job in candidate_jobs
        if job["result"].get("training_loss_trace")
    ]
    common_loss_target = max(final_losses) if final_losses else math.inf
    for candidate, candidate_jobs in grouped.items():
        scores = [_completed_score(job) for job in candidate_jobs]
        values = [float(score) for score in scores if score is not None]
        mean = statistics.fmean(values)
        ci95 = (1.96 * statistics.stdev(values) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        step_times = [float(job["result"]["mean_logged_step_ms"]) for job in candidate_jobs
                      if job["result"].get("mean_logged_step_ms") is not None]
        footprints = [float(job["result"]["max_current_physical_mb"]) for job in candidate_jobs
                      if job["result"].get("max_current_physical_mb") is not None]
        times_to_loss = []
        for job in candidate_jobs:
            trace = job["result"].get("training_loss_trace") or []
            crossed = next((row["step"] + 1 for row in trace
                            if row["loss"] <= common_loss_target), None)
            mean_step_ms = job["result"].get("mean_logged_step_ms")
            if crossed is not None and mean_step_ms is not None:
                times_to_loss.append(crossed * float(mean_step_ms))
        rankings.append({
            "candidate": candidate,
            "mean_validation_bpb": mean,
            "ci95": ci95,
            "seeds": [job["seed"] for job in candidate_jobs],
            "time_to_common_training_loss_ms": (
                statistics.fmean(times_to_loss) if times_to_loss else math.inf
            ),
            "common_training_loss_target": common_loss_target,
            "max_current_physical_mb": max(footprints) if footprints else math.inf,
            "mean_logged_step_ms": statistics.fmean(step_times) if step_times else math.inf,
        })
    if not rankings:
        return []
    rankings.sort(key=lambda item: item["mean_validation_bpb"])
    best = rankings[0]
    best_lo = best["mean_validation_bpb"] - best["ci95"]
    best_hi = best["mean_validation_bpb"] + best["ci95"]
    for item in rankings:
        lo = item["mean_validation_bpb"] - item["ci95"]
        hi = item["mean_validation_bpb"] + item["ci95"]
        item["confidence_interval_overlaps_best"] = lo <= best_hi and best_lo <= hi
    # Only arms statistically tied with the best invoke the declared systems
    # tie breakers. Non-overlapping arms remain ordered by mean validation BPB.
    tied = [item for item in rankings if item["confidence_interval_overlaps_best"]]
    untied = [item for item in rankings if not item["confidence_interval_overlaps_best"]]
    tied.sort(key=lambda item: (
        item["time_to_common_training_loss_ms"],
        item["max_current_physical_mb"],
        item["mean_logged_step_ms"],
        item["mean_validation_bpb"],
    ))
    untied.sort(key=lambda item: item["mean_validation_bpb"])
    return tied + untied


def _new_job(stage: str, candidate: str, lr: float, steps: int, seed: int) -> dict[str, Any]:
    preset = "128m" if stage.startswith("exact_128m") else "16m"
    return {
        "id": f"{stage}__{candidate}__seed{seed}",
        "stage": stage,
        "candidate": candidate,
        "seed": seed,
        "steps": steps,
        "lr": lr,
        "status": "pending",
        "output": f"out/funnel/{stage}/{candidate}_seed{seed}_{preset}",
    }


def advance(plan_path: Path, champion_path: Path) -> str:
    plan = _load(plan_path)
    stage = _current_stage(plan)
    current = _stage_jobs(plan, stage)
    unfinished = [job["id"] for job in current if job.get("status") not in {"completed", "failed"}]
    if unfinished:
        raise RuntimeError(f"{stage} still has {len(unfinished)} unfinished jobs")
    lrs = _tuned_lrs(plan)
    eligible_current = current
    excluded = []
    if stage != "lr_sweep_16m":
        expected_per_candidate = 2 if stage in {"advance_1000", "exact_128m_1000"} else 1
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for job in current:
            grouped[job["candidate"]].append(job)
        eligible_candidates = set()
        for candidate, candidate_jobs in grouped.items():
            finite_jobs = [job for job in candidate_jobs if _completed_score(job) is not None]
            if len(finite_jobs) == expected_per_candidate:
                eligible_candidates.add(candidate)
            else:
                excluded.append({
                    "candidate": candidate,
                    "stage": stage,
                    "required_finite_jobs": expected_per_candidate,
                    "finite_jobs": len(finite_jobs),
                    "failed_jobs": [
                        {
                            "id": job["id"],
                            "status": job.get("status"),
                            "failure_reason": (job.get("result") or {}).get("failure_reason"),
                        }
                        for job in candidate_jobs if _completed_score(job) is None
                    ],
                })
        eligible_current = [job for job in current if job["candidate"] in eligible_candidates]
        minimum = {
            "stable_500": 6,
            "advance_1000": 4,
            "exact_128m_500": 2,
            "exact_128m_1000": 1,
        }[stage]
        if len(eligible_candidates) < minimum:
            raise RuntimeError(
                f"{stage} has only {len(eligible_candidates)} fully finite candidates; "
                f"need at least {minimum}; exclusions={excluded}"
            )
    if stage == "lr_sweep_16m":
        candidates = sorted(lrs)
        if not candidates:
            raise RuntimeError("no candidate survived the LR sweep")
        jobs = [_new_job("stable_500", c, lrs[c], 500, 1337) for c in candidates]
        next_stage = "stable_500"
    elif stage == "stable_500":
        ranked = _rank_candidates(eligible_current)
        candidates = [item["candidate"] for item in ranked[:6]]
        for anchor in ("adamw", "muon_ns5_adamw"):
            if anchor not in {job["candidate"] for job in eligible_current}:
                raise RuntimeError(f"mandatory anchor {anchor} failed stable_500")
            if anchor not in candidates:
                candidates.append(anchor)
        jobs = [_new_job("advance_1000", c, lrs[c], 1000, seed)
                for c in candidates for seed in (42, 2026)]
        next_stage = "advance_1000"
    elif stage == "advance_1000":
        ranked = _rank_candidates(eligible_current)[:4]
        jobs = [_new_job("exact_128m_500", item["candidate"], lrs[item["candidate"]], 500, 1337)
                for item in ranked]
        next_stage = "exact_128m_500"
    elif stage == "exact_128m_500":
        ranked = _rank_candidates(eligible_current)[:2]
        jobs = [_new_job("exact_128m_1000", item["candidate"], lrs[item["candidate"]], 1000, seed)
                for item in ranked for seed in (42, 2026)]
        next_stage = "exact_128m_1000"
    else:
        ranked = _rank_candidates(eligible_current)
        if not ranked:
            raise RuntimeError("no exact-scale candidate survived")
        winner = ranked[0]["candidate"]
        mean_bpb = ranked[0]["mean_validation_bpb"]
        write_champion(champion_path, winner, lrs[winner], mean_bpb, locked=True)
        plan["champion"] = {
            "candidate": winner,
            "lr": lrs[winner],
            "mean_validation_bpb": mean_bpb,
            "winner_exact_gate": "pending",
        }
        _atomic_json(plan_path, plan)
        return (f"champion selected: {winner} (mean BPB {mean_bpb:.6f}); "
                "manifest remains locked pending winner-specific exact gate")

    plan["jobs"].extend(jobs)
    plan.setdefault("history", []).append({
        "advanced_from": stage, "advanced_to": next_stage, "timestamp": time.time(),
        "ranking": _rank_candidates(eligible_current) if stage != "lr_sweep_16m" else None,
        "excluded_candidates": excluded,
    })
    _atomic_json(plan_path, plan)
    return f"materialized {len(jobs)} {next_stage} jobs"


def champion_argv(candidate: str, lr: float, data: Path, token_bytes: Path) -> list[str]:
    return [
        str(TRAIN_BIN),
        "--preset", "arch02-128m",
        "--optimizer", candidate,
        "--total-steps", "2000",
        "--batch", "16",
        "--seq-len", "256",
        "--seed", "1337",
        "--final-warmdown", "350",
        "--eval-every", "250",
        "--checkpoint-every", "250",
        "--log-every", "50",
        "--ema-decay", "0.997",
        "--research-manifest", str(DEFAULT_STUDY),
        "--data-dir", str(data),
        "--token-bytes", str(token_bytes),
        "--out", str(ROOT / "out/champion_128m_seed1337"),
        *_lr_flags(candidate, lr),
    ]


def write_champion(
    path: Path,
    candidate: str | None,
    lr: float | None,
    mean_bpb: float | None,
    *,
    locked: bool,
    data: Path = DEFAULT_DATA,
    token_bytes: Path = DEFAULT_TOKEN_BYTES,
) -> None:
    command = (champion_argv(candidate, lr, data, token_bytes)
               if candidate and lr is not None and not locked else None)
    value = {
        "schema_version": 1,
        "locked": locked,
        "lock_reason": (
            "winner-specific exact gate must pass"
            if locked and candidate
            else "optimizer funnel and winner-specific exact gate must pass"
            if locked
            else None
        ),
        "candidate": candidate,
        "tuned_lr": lr,
        "selection_mean_validation_bpb": mean_bpb,
        "tokens": 8_192_000,
        "required_gates": [
            "native deterministic parity", "finite toy screen", "equal-data funnel complete",
            "exact checkpoint resume", "current physical footprint below 52 GiB",
            "zero swap pressure", "dispatch budget below 10000", "no silent fallback",
        ],
        "preflight_command": ([str(GATE_BIN), "--optimizer", candidate] if candidate else None),
        "command_argv": command,
    }
    _atomic_json(path, value)


def unlock_from_gate(champion_path: Path, gate_path: Path, data: Path, token_bytes: Path) -> str:
    champion = _load(champion_path)
    gate = _load(gate_path)
    candidate = champion.get("candidate")
    lr = champion.get("tuned_lr")
    if not candidate or lr is None:
        raise RuntimeError("champion selection is incomplete")
    failures = []
    if not gate.get("passed"):
        failures.append("gate report is not passing")
    if gate.get("optimizer") != candidate:
        failures.append(f"gate optimizer {gate.get('optimizer')} != champion {candidate}")
    if gate.get("parameter_count") != 128_367_988:
        failures.append("gate did not use exact 128M preset")
    if float(gate.get("current_physical_mb", math.inf)) >= 52.0 * 1024.0:
        failures.append("physical footprint gate failed")
    if int(gate.get("dispatches", 10_000)) >= 10_000:
        failures.append("dispatch budget gate failed")
    swap_before = float(gate.get("swap_before_mb", math.inf))
    swap_after = float(gate.get("swap_after_mb", math.inf))
    if not math.isfinite(swap_before) or not math.isfinite(swap_after):
        failures.append("swap pressure was not recorded")
    elif swap_after - swap_before > 1e-6:
        failures.append(
            f"swap pressure increased during exact gate "
            f"({swap_before:.6g} -> {swap_after:.6g} MiB)"
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    write_champion(
        champion_path,
        candidate,
        float(lr),
        champion.get("selection_mean_validation_bpb"),
        locked=False,
        data=data,
        token_bytes=token_bytes,
    )
    value = _load(champion_path)
    value["winner_exact_gate_artifact"] = str(gate_path.resolve())
    _atomic_json(champion_path, value)
    return f"unlocked champion manifest for {candidate}"


def run_next(plan_path: Path, data: Path, token_bytes: Path, dry_run: bool) -> str:
    plan = _load(plan_path)
    job = next((item for item in plan["jobs"] if item.get("status") == "pending"), None)
    if job is None:
        return "no pending jobs; use --advance"
    argv = job_argv(job, data, token_bytes)
    if dry_run:
        return json.dumps({"job": job["id"], "argv": argv}, indent=2)
    # Cargo's incremental no-op is cheap and guarantees the ledger never runs a
    # stale release binary after a kernel or Rust source edit.
    subprocess.run(["cargo", "build", "--release", "--bins"], cwd=METAL_DIR, check=True)
    if not data.is_dir() or not token_bytes.is_file():
        raise RuntimeError(f"FineWeb/token-byte inputs missing: {data} / {token_bytes}")
    output = ROOT / job["output"]
    if output.exists() and any(output.iterdir()):
        archived = output.with_name(f"{output.name}.attempt-{int(time.time())}")
        os.replace(output, archived)
        job.setdefault("attempt_history", []).append({
            "archived_output": str(archived), "archived_at": time.time(),
        })
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    job["status"] = "running"
    job["command_argv"] = argv
    job["started_at"] = time.time()
    _atomic_json(plan_path, plan)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(argv, cwd=METAL_DIR, stdout=log, stderr=subprocess.STDOUT)
    plan = _load(plan_path)
    saved = next(item for item in plan["jobs"] if item["id"] == job["id"])
    saved["returncode"] = process.returncode
    saved["finished_at"] = time.time()
    saved["result"] = _read_result(output)
    saved["status"] = "completed" if process.returncode == 0 and saved["result"]["finite"] else "failed"
    _atomic_json(plan_path, plan)
    return f"{saved['id']}: {saved['status']} ({log_path})"


def status(plan_path: Path) -> str:
    plan = _load(plan_path)
    lines = []
    for stage in STAGE_ORDER:
        jobs = _stage_jobs(plan, stage)
        if not jobs:
            continue
        counts: dict[str, int] = defaultdict(int)
        for job in jobs:
            counts[job.get("status", "pending")] += 1
        lines.append(f"{stage}: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    for candidate, reason in NATIVE_BLOCKERS.items():
        lines.append(f"blocked {candidate}: {reason}")
    return "\n".join(lines)


def requeue(plan_path: Path, job_id: str) -> str:
    plan = _load(plan_path)
    job = next((item for item in plan["jobs"] if item["id"] == job_id), None)
    if job is None:
        raise RuntimeError(f"unknown job id {job_id}")
    if job.get("status") == "completed":
        raise RuntimeError("completed jobs are immutable; use a new plan for a deliberate rerun")
    job["status"] = "pending"
    job.setdefault("attempt_history", []).append({
        "requeued_at": time.time(),
        "previous_returncode": job.get("returncode"),
        "previous_result": job.get("result"),
    })
    _atomic_json(plan_path, plan)
    return f"requeued {job_id}"


def collect_existing(plan_path: Path, job_id: str) -> str:
    plan = _load(plan_path)
    job = next((item for item in plan["jobs"] if item["id"] == job_id), None)
    if job is None:
        raise RuntimeError(f"unknown job id {job_id}")
    result = _read_result(ROOT / job["output"])
    job["result"] = result
    job["status"] = "completed" if result["finite"] else "failed"
    job["collected_at"] = time.time()
    _atomic_json(plan_path, plan)
    return f"{job_id}: {job['status']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--token-bytes", type=Path, default=DEFAULT_TOKEN_BYTES)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init", action="store_true")
    action.add_argument("--run-next", action="store_true")
    action.add_argument("--dry-run-next", action="store_true")
    action.add_argument("--advance", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--write-champion-template", action="store_true")
    action.add_argument("--requeue", metavar="JOB_ID")
    action.add_argument("--collect", metavar="JOB_ID")
    action.add_argument("--unlock-from-gate", type=Path, metavar="GATE_JSON")
    args = parser.parse_args()
    if args.init:
        write_native_plan(args.plan)
        write_champion(args.champion, None, None, None, locked=True, data=args.data,
                       token_bytes=args.token_bytes)
    elif args.run_next or args.dry_run_next:
        print(run_next(args.plan, args.data, args.token_bytes, args.dry_run_next))
    elif args.advance:
        print(advance(args.plan, args.champion))
    elif args.status:
        print(status(args.plan))
    elif args.requeue:
        print(requeue(args.plan, args.requeue))
    elif args.collect:
        print(collect_existing(args.plan, args.collect))
    elif args.unlock_from_gate:
        print(unlock_from_gate(args.champion, args.unlock_from_gate, args.data, args.token_bytes))
    else:
        write_champion(args.champion, None, None, None, locked=True, data=args.data,
                       token_bytes=args.token_bytes)
        print(args.champion)


if __name__ == "__main__":
    main()
