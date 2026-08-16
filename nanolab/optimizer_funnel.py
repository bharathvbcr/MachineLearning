"""Reproducible, gated optimizer-study manifest and toy-stage runner.

The expensive phases are emitted, not launched implicitly.  A candidate cannot
enter a native phase unless ``native_ready`` is true; this prevents a registered
paper arm from silently falling back to Muon.

Examples:
  python -m nanolab.optimizer_funnel --write-manifest
  python -m nanolab.optimizer_funnel --run-toy
  python -m nanolab.optimizer_funnel --print-native-commands
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import OPTIMIZERS, build_config
from .train import train


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    oracle_ready: bool
    native_ready: bool
    source: str
    native_block_reason: str | None


SOURCES = {
    "muon_ns5_adamw": "https://github.com/KellerJordan/Muon",
    "muon_ns3_adamw": "https://github.com/KellerJordan/Muon",
    "muon_polar_adamw": "https://arxiv.org/abs/2505.16932",
    "normuon_adamw": "https://github.com/zichongli5/NorMuon",
    "muown_adamw": "https://arxiv.org/abs/2605.10797",
    "mona_adamw": "https://arxiv.org/abs/2605.26842",
    "mimuon_adamw": "https://arxiv.org/abs/2605.19619",
    "adamw": "https://arxiv.org/abs/1711.05101",
    "lion": "https://arxiv.org/abs/2302.06675",
    "cautious_adamw": "https://github.com/kyleliang919/C-Optim",
    "cautious_lion": "https://github.com/kyleliang919/C-Optim",
    "sgd_momentum": "https://pytorch.org/docs/stable/generated/torch.optim.SGD.html",
    "sophia": "https://arxiv.org/abs/2305.14342",
    "schedule_free_adamw": "https://github.com/facebookresearch/schedule_free",
    "prodigy": "https://github.com/konstmish/prodigy",
    "soap_adamw": "https://github.com/nikhilvyas/SOAP",
}

MUON_FAMILY = {
    "muon_ns5_adamw", "muon_ns3_adamw", "muon_polar_adamw", "normuon_adamw",
    "muown_adamw", "mona_adamw", "mimuon_adamw",
}
NATIVE_READY = {
    "muon_ns5_adamw", "muon_ns3_adamw", "muon_polar_adamw",
    "normuon_adamw", "muown_adamw", "mona_adamw",
    "adamw", "lion", "cautious_adamw", "cautious_lion", "sgd_momentum",
    "sophia", "schedule_free_adamw",
    "prodigy",
}

NATIVE_BLOCKERS = {
    "mimuon_adamw": (
        "Exact singular-gap routing requires per-matrix SVD; Metal/MPS exposes no GPU "
        "SVD/eigensolver, and an Accelerate fallback would force a host synchronization."
    ),
    "soap_adamw": (
        "Exact SOAP basis refresh requires symmetric eigendecomposition; Metal/MPS exposes "
        "no GPU eigensolver, and a CPU LAPACK fallback violates the no-forced-sync gate."
    ),
}

BASE_LRS = {
    "lion": 1.2e-4,
    "cautious_lion": 1.2e-4,
    "sgd_momentum": 0.1,
    "prodigy": 1.0,
}

CANDIDATES = tuple(
    Candidate(
        name=name,
        family="muon_hybrid" if name in MUON_FAMILY else "full_parameter",
        oracle_ready=True,
        native_ready=name in NATIVE_READY,
        source=SOURCES[name],
        native_block_reason=NATIVE_BLOCKERS.get(name),
    )
    for name in OPTIMIZERS
)

STAGES = (
    dict(name="toy_numerics", model="toy", steps=20, seeds=[1337], candidates="all"),
    dict(name="lr_sweep_16m", model="arch02-16m", steps=100, seeds=[1337],
         lr_multipliers=[0.25, 0.5, 1.0, 2.0, 4.0], candidates="native_ready"),
    dict(name="stable_500", model="arch02-16m", steps=500, seeds=[1337],
         candidates="stable_after_lr_sweep"),
    dict(name="advance_1000", model="arch02-16m", steps=1000, seeds=[42, 2026],
         candidates="best_validation_plus_adamw_and_muon"),
    dict(name="exact_128m_500", model="arch02-128m", steps=500, seeds=[1337],
         candidates="top_four"),
    dict(name="exact_128m_1000", model="arch02-128m", steps=1000, seeds=[42, 2026],
         candidates="top_two"),
    dict(name="champion_128m_2000", model="arch02-128m", steps=2000, seeds=[1337],
         candidates="winner", final_warmdown=350, eval_every=250, checkpoint_every=250),
)


def manifest():
    return {
        "schema_version": 1,
        "selection_metric": "mean_validation_bpb_at_equal_tokens",
        "tie_breakers": ["time_to_loss", "current_footprint", "step_time"],
        "hard_gates": {
            "exact_resume": True,
            "max_current_footprint_gib": 52,
            "swap_pressure": False,
            "silent_fallback": False,
            "soft_split_clip": {"muon_scale": "sqrt(c)", "adaptive_scale": "c",
                                "global_threshold": 0.3},
        },
        "required_metrics": [
            "optimizer_ms", "step_ms", "validation_bpb", "gradient_norm_by_role",
            "update_norm_by_role", "orthogonality_error", "row_drift",
            "spectral_drift", "current_footprint", "numerical_failures",
        ],
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "stages": list(STAGES),
    }


def write_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(), indent=2) + "\n", encoding="utf-8")
    print(path)


def run_toy(out_dir: Path):
    """Execute the mandatory equal-order 20-step numerical screen."""
    results = []
    for candidate in CANDIDATES:
        cfg = build_config("cpu_smoke", {
            "optimizer": candidate.name,
            "seed": 1337,
            "max_steps": 20,
            "eval_interval": 20,
            "eval_iters": 8,
            "ckpt_interval": 20,
            "out_dir": str(out_dir),
            "run_name": f"funnel_toy_{candidate.name}",
        })
        try:
            val_loss = train(cfg)
            results.append({"candidate": candidate.name, "status": "stable",
                            "validation_loss": val_loss,
                            "validation_bpb": val_loss / math.log(2)})
        except Exception as error:
            results.append({"candidate": candidate.name, "status": "failed",
                            "error": repr(error)})
    result_path = out_dir / "toy-results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(result_path)
    return results


def native_commands():
    """Commands for currently parity-qualified arms; later arms stay blocked."""
    base = "cargo run --release --bin train --"
    data = "--data-dir <FINEWEB_DIR> --token-bytes <TOKEN_BYTES_JSON>"
    for candidate in CANDIDATES:
        if not candidate.native_ready:
            continue
        base_lr = BASE_LRS.get(candidate.name, 0.025 if candidate.name in MUON_FAMILY else 6e-4)
        for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
            lr = base_lr * multiplier
            lr_flags = (f"--matrix-lr {lr:.8g}" if candidate.name in MUON_FAMILY
                        else f"--matrix-lr {lr:.8g} --embed-lr {lr:.8g} --scalar-lr {lr:.8g}")
            print(f"{base} --preset 16m --optimizer {candidate.name} --total-steps 100 "
                  f"{lr_flags} --seed 1337 --research-telemetry {data} "
                  f"--out out/funnel/lr16m/{candidate.name}_lr{lr:.8g}")


def write_native_plan(path: Path):
    jobs = []
    for candidate in CANDIDATES:
        if not candidate.native_ready:
            continue
        base_lr = BASE_LRS.get(candidate.name, 0.025 if candidate.name in MUON_FAMILY else 6e-4)
        for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
            lr = base_lr * multiplier
            jobs.append({
                "id": f"lr16m__{candidate.name}__{lr:.8g}",
                "stage": "lr_sweep_16m",
                "candidate": candidate.name,
                "seed": 1337,
                "steps": 100,
                "lr": lr,
                "status": "pending",
                "output": f"out/funnel/lr16m/{candidate.name}_lr{lr:.8g}",
            })
    plan = {
        "schema_version": 1,
        "generated_from": "research/optimizer-study.json",
        "blocked_candidates": NATIVE_BLOCKERS,
        "jobs": jobs,
        "advancement": {
            "lr_sweep_16m": "lowest finite validation_bpb per candidate",
            "stable_500": "all finite LR winners; seed 1337",
            "advance_1000": "best validation candidates plus AdamW and Muon anchors; seeds 42,2026",
            "exact_128m_500": "top four by mean equal-token validation BPB",
            "exact_128m_1000": "top two; seeds 42,2026",
            "champion": "lowest mean BPB; time-to-loss, footprint, then step time break CI ties",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--run-toy", action="store_true")
    parser.add_argument("--print-native-commands", action="store_true")
    parser.add_argument("--write-native-plan", action="store_true")
    parser.add_argument("--manifest-path", type=Path,
                        default=Path("research/optimizer-study.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("nanolab/out/funnel"))
    parser.add_argument("--native-plan-path", type=Path,
                        default=Path("research/native-optimizer-funnel.json"))
    args = parser.parse_args()
    if args.write_manifest or not (args.run_toy or args.print_native_commands):
        write_manifest(args.manifest_path)
    if args.run_toy:
        run_toy(args.out_dir)
    if args.print_native_commands:
        native_commands()
    if args.write_native_plan:
        write_native_plan(args.native_plan_path)


if __name__ == "__main__":
    main()
