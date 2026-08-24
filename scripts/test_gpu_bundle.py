#!/usr/bin/env python3
"""Adversarial suite for scripts/gpu_bundle.py.

This runner spends rented GPU time and produces the numbers PAPER section 8.4
reads against a pre-registered decision rule. Between a wrong result and a wrong
conclusion there is nothing but this file, so the runner gets attacked first.

No GPU, no trainer and no network are touched: every case is a synthetic run
directory, and the end-to-end cases drive the real ``main()`` with a fake trainer
substituted for ``run_one``.

    python3 scripts/test_gpu_bundle.py     # exit 0 = clean

Run from the repo root.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("gb", "scripts/gpu_bundle.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILS: list[str] = []
PASSES: list[str] = []


def check(name, cond, detail=""):
    (PASSES if cond else FAILS).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_run(d: pathlib.Path, *, starts=1, dones=1, best=3.5, evals=((1000, 4.0), (2000, 3.5)),
              tokens=2000, elapsed=12.0, cfg: dict | None = None):
    """Synthesize a run directory. `starts`/`dones` let a test build the exact
    multi-segment shape that made run128m_20k unusable."""
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for _ in range(starts):
        lines.append({"event": "start", "params": 1, "params_non_embed": 1})
    for step, (tok, val) in enumerate(evals):
        lines.append({"event": "eval", "step": step, "val_loss": val, "tokens": tok})
    for _ in range(dones):
        rec = {"event": "done", "best_val": best, "tokens": tokens}
        if elapsed is not None:
            rec["elapsed_s"] = elapsed
            rec["mean_tok_s"] = tokens / elapsed
        lines.append(rec)
    (d / "metrics.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    if cfg is not None:
        (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def sandbox(tmp: pathlib.Path):
    """Point every module-level path at a temp tree. Returns the old values."""
    old = (m.OUT_ROOT, m.SMOKE_ROOT, m.ARCHIVE_ROOT, m.TRANSFER)
    m.OUT_ROOT = tmp / "gpu_bundle"
    m.SMOKE_ROOT = m.OUT_ROOT / "_smoke"
    m.ARCHIVE_ROOT = m.OUT_ROOT / "_archived"
    m.TRANSFER = m.OUT_ROOT / "transfer.json"
    return old


def unsandbox(old):
    m.OUT_ROOT, m.SMOKE_ROOT, m.ARCHIVE_ROOT, m.TRANSFER = old


class FakeTrainer:
    """Stands in for run_one. Writes what a real run writes, on demand badly."""

    def __init__(self, mode="ok", loss=None):
        self.mode = mode           # ok | die_midrun | nonzero | wrong_config | nometrics
        self.loss = loss           # optional f(job) -> val, to shape a real sweep
        self.calls: list[tuple[str, str | None]] = []
        self.lock = threading.Lock()

    def __call__(self, job, root, smoke, device):
        with self.lock:
            self.calls.append((job["id"], device))
        d = root / job["id"]
        d.mkdir(parents=True, exist_ok=True)
        cfg = m.build_config(m.PRESET, job["overrides"]).to_dict()
        if self.mode == "nometrics":
            (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            return 0, 0.1
        if self.mode == "die_midrun":
            write_run(d, starts=1, dones=0, evals=((1000, 4.0),), cfg=cfg)
            return 1, 0.1
        if self.mode == "wrong_config":
            cfg["seed"] = 999999
        code = 3 if self.mode == "nonzero" else 0
        v = self.loss(job) if self.loss else 3.5
        write_run(d, starts=1, dones=1, best=v - 0.1,
                  evals=((1000, v + 0.4), (2000, v)), cfg=cfg)
        return code, 0.1


def run_main(argv, trainer=None, gpus=0):
    """Drive the real main() with a fake trainer and no GPU probe."""
    old_run, old_det, old_argv = m.run_one, m.detect_gpus, sys.argv
    if trainer is not None:
        m.run_one = trainer
    m.detect_gpus = lambda: gpus
    sys.argv = ["gpu_bundle.py"] + argv
    try:
        return m.main()
    finally:
        m.run_one, m.detect_gpus, sys.argv = old_run, old_det, old_argv


# ===========================================================================
print("\n=== A. SMOKE ISOLATION: --smoke must never mark a real job done ===")
# The documented procedure is "--smoke first on the rented box". The previous
# runner wrote 40-step smoke runs into the SAME directory the matrix resumes
# from, so following the docs silently substituted a 40-step best_val for one
# real job per suite -- including e1_mup, whose readouts are pre-registered.
check("A1 smoke root is a distinct subtree", m.SMOKE_ROOT != m.OUT_ROOT
      and str(m.SMOKE_ROOT).startswith(str(m.OUT_ROOT)))

with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    old = sandbox(tmp)
    jid = "e1_proxy_attention_mlr0.0016_s1337"
    write_run(m.SMOKE_ROOT / jid, best=9.99)      # a 40-step smoke result
    check("A2 a smoke result does not mark the matrix job done",
          m.inspect_run(m.OUT_ROOT / jid)["status"] == "missing",
          m.inspect_run(m.OUT_ROOT / jid)["status"])
    check("A3 the smoke result IS recorded in the smoke tree",
          m.inspect_run(m.SMOKE_ROOT / jid)["status"] == "done")
    check("A4 smoke and matrix ledgers are different files",
          m.ledger_path(m.SMOKE_ROOT) != m.ledger_path(m.OUT_ROOT))
    unsandbox(old)

src = pathlib.Path("scripts/gpu_bundle.py").read_text()
check("A5 out_dir is chosen at run time, not baked into the matrix",
      'ov["out_dir"] = str(root)' in src and 'out_dir=str(OUT_ROOT)' not in src)


print("\n=== B. LEDGER: merge, never clobber (gap D8, by a second mechanism) ===")
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    old = sandbox(tmp)
    r = m.OUT_ROOT
    a = [{"id": "e1_proxy_x", "suite": "e1_proxy", "status": "done", "elapsed_s": 3600}]
    b = [{"id": "e1_mup_y", "suite": "e1_mup", "status": "done", "elapsed_s": 1800}]
    m.write_ledger(r, a, "t0", {})
    m.write_ledger(r, b, "t0", {})
    led = json.loads(m.ledger_path(r).read_text())
    ids = {j["id"] for j in led["jobs"]}
    check("B1 a second --only run keeps the first run's jobs", ids == {"e1_proxy_x", "e1_mup_y"}, ids)
    check("B2 jobs_total counts the union", led["jobs_total"] == 2, led["jobs_total"])
    check("B3 gpu_hours_measured sums both", abs(led["gpu_hours_measured"] - 1.5) < 1e-9,
          led["gpu_hours_measured"])
    check("B4 by_suite is per suite", led["by_suite"] == {"e1_proxy": {"done": 1},
                                                          "e1_mup": {"done": 1}}, led["by_suite"])
    # updating an existing id replaces it rather than duplicating
    m.write_ledger(r, [{"id": "e1_proxy_x", "suite": "e1_proxy", "status": "failed"}], "t0", {})
    led = json.loads(m.ledger_path(r).read_text())
    check("B5 re-writing an id updates in place", len(led["jobs"]) == 2
          and [j for j in led["jobs"] if j["id"] == "e1_proxy_x"][0]["status"] == "failed")
    # a corrupt prior ledger must not block the current write
    m.ledger_path(r).write_text("{not json", encoding="utf-8")
    m.write_ledger(r, b, "t0", {})
    check("B6 a corrupt prior ledger does not block the write",
          json.loads(m.ledger_path(r).read_text())["jobs_total"] == 1)
    check("B7 no .tmp file survives the atomic replace",
          not list(r.glob("*.tmp")), list(r.glob("*.tmp")))

    # concurrent writers must not lose records
    m.ledger_path(r).unlink()
    def w(i):
        m.write_ledger(r, [{"id": f"j{i}", "suite": "e1_proxy", "status": "done"}], "t0", {})
    ts = [threading.Thread(target=w, args=(i,)) for i in range(40)]
    [t.start() for t in ts]; [t.join() for t in ts]
    led = json.loads(m.ledger_path(r).read_text())
    check("B8 40 concurrent writers lose nothing", led["jobs_total"] == 40, led["jobs_total"])
    unsandbox(old)


print("\n=== C. RUN INSPECTION: a partial run must never look like a finished one ===")
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    check("C1 missing dir -> missing", m.inspect_run(tmp / "nope")["status"] == "missing")

    d = tmp / "clean"; write_run(d)
    st = m.inspect_run(d)
    check("C2 one start + one done -> done", st["status"] == "done", st["status"])
    check("C3 curve is extracted in order", st["curve"] == [[1000, 4.0], [2000, 3.5]], st["curve"])
    check("C4 final_val is the LAST eval, not the minimum", st["final_val"] == 3.5)
    check("C5 elapsed_s is carried", st["elapsed_s"] == 12.0)

    d = tmp / "partial"; write_run(d, starts=1, dones=0)
    check("C6 started, never finished -> partial", m.inspect_run(d)["status"] == "partial")

    # the exact run128m_20k shape: many segments in one appended file
    d = tmp / "segments"; write_run(d, starts=9, dones=8)
    st = m.inspect_run(d)
    check("C7 9 starts / 8 dones -> suspect (the run128m_20k shape)",
          st["status"] == "suspect", st["status"])
    d = tmp / "restarted"; write_run(d, starts=2, dones=1)
    check("C8 a silent restart -> suspect, not done",
          m.inspect_run(d)["status"] == "suspect", m.inspect_run(d)["status"])

    d = tmp / "noval"; write_run(d, best=None)
    check("C9 done with best_val null is not done",
          m.inspect_run(d)["status"] != "done", m.inspect_run(d)["status"])

    d = tmp / "torn"; d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"event": "start"}\n{"event": "eval", "step": 0, "val_loss": 4.0, "tokens": 10}\n'
        '{"event": "done", "best_v\n', encoding="utf-8")   # killed mid-write
    st = m.inspect_run(d)
    check("C10 a torn final line is skipped, not fatal", st["status"] == "partial", st["status"])
    check("C11 and the completed evals still parse", st["curve"] == [[10, 4.0]])

    d = tmp / "noevaltok"; d.mkdir()
    (d / "metrics.jsonl").write_text(
        '{"event": "start"}\n{"event": "eval", "step": 0, "val_loss": 4.0}\n', encoding="utf-8")
    check("C12 an eval without a token count is not put on the curve",
          m.inspect_run(d)["curve"] == [])


print("\n=== D. FINGERPRINT: the written config must be the requested config ===")
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    job = m.build_matrix()[0]
    good = m.build_config(m.PRESET, job["overrides"]).to_dict()

    d = tmp / "ok"; d.mkdir(); (d / "config.json").write_text(json.dumps(good))
    check("D1 the matching config passes", m.verify_fingerprint(job, d) is None,
          m.verify_fingerprint(job, d))

    for field, bad_val in (("seed", 999), ("max_steps", 7), ("batch_size", 1),
                           ("mup", False), ("matrix_lr", 0.9), ("d_model", 512),
                           ("n_head", 12), ("compile", True), ("eval_iters", 4)):
        c = dict(good); c[field] = bad_val
        d = tmp / f"bad_{field}"; d.mkdir(); (d / "config.json").write_text(json.dumps(c))
        why = m.verify_fingerprint(job, d)
        check(f"D2.{field} a wrong {field} fails and the reason names it",
              why is not None and field in why, why)

    d = tmp / "nocfg"; d.mkdir()
    check("D3 a missing config.json fails", m.verify_fingerprint(job, d) is not None)
    d = tmp / "badjson"; d.mkdir(); (d / "config.json").write_text("{oops")
    check("D4 an unreadable config.json fails", m.verify_fingerprint(job, d) is not None)

    # float vs int must not produce a spurious mismatch
    emb = [j for j in m.build_matrix() if j["suite"] == "e1_embed_lr"][0]
    c = m.build_config(m.PRESET, emb["overrides"]).to_dict()
    c["embed_lr_mult"] = 3           # int, not 3.0
    d = tmp / "intfloat"; d.mkdir(); (d / "config.json").write_text(json.dumps(c))
    check("D5 3 and 3.0 are the same embed_lr_mult",
          m.verify_fingerprint(emb, d) is None, m.verify_fingerprint(emb, d))


print("\n=== E. CLASSIFICATION: only a complete, verified run reads as measured ===")
done = {"status": "done", "best_val": 3.5, "starts": 1, "dones": 1}
check("E1 clean run -> done", m.classify_result(0, done, None)[0] == "done")
check("E2 nonzero exit -> failed", m.classify_result(3, done, None)[0] == "failed")
st, why = m.classify_result(0, {"status": "partial", "best_val": None, "starts": 1, "dones": 0}, None)
check("E3 exit 0 with no done record -> failed", st == "failed")
check("E4 and the reason says so", "partial" in why or "best_val absent" in why, why)
st, why = m.classify_result(0, done, "seed: want 42 got 999")
check("E5 a fingerprint mismatch fails a job that otherwise looks perfect", st == "failed")
check("E6 and the reason carries the field", "seed" in why, why)
check("E7 suspect never reads as done",
      m.classify_result(0, {"status": "suspect", "best_val": 3.5, "starts": 2, "dones": 2},
                        None)[0] == "failed")


print("\n=== F. PROXY SWEEP: an edge is not an optimum ===")
def rec(arm, lr, val, status="done"):
    return {"suite": "e1_proxy", "arm": arm, "tag": f"mlr{lr:g}",
            "status": status, "final_val": val, "best_val": val - 0.5}

GRID = m.PROXY_MATRIX_LRS
interior = [rec("mingru", lr, v) for lr, v in zip(GRID, [4.0, 3.7, 3.5, 3.6, 3.8, 4.1])]
left_edge = [rec("attention", lr, v) for lr, v in zip(GRID, [3.4, 3.5, 3.6, 3.7, 3.8, 3.9])]
right_edge = [rec("attention", lr, v) for lr, v in zip(GRID, [3.9, 3.8, 3.7, 3.6, 3.5, 3.4])]

a = m.analyse_proxy(interior)["arms"]["mingru"]
check("F1 an interior minimum is bracketed", a["bracketed"] and a["matrix_lr"] == GRID[2],
      f"{a['bracketed']} @ {a['matrix_lr']}")
check("F2 a minimum at the LOW edge is not bracketed",
      not m.analyse_proxy(left_edge)["arms"]["attention"]["bracketed"])
check("F3 a minimum at the HIGH edge is not bracketed",
      not m.analyse_proxy(right_edge)["arms"]["attention"]["bracketed"])
check("F4 an unfinished cell is ignored, not counted",
      m.analyse_proxy(interior[:-1] + [rec("mingru", GRID[-1], 0.0, "failed")])
      ["arms"]["mingru"]["points"] == len(GRID) - 1)
# PAPER 3.2: minimum-over-evaluations is not a ranking. Every proxy cell stops at
# the same token budget, so the paired snapshot is the LAST eval.
curves = m.proxy_curves([{"suite": "e1_proxy", "arm": "x", "tag": "mlr0.0125",
                          "status": "done", "final_val": 3.9, "best_val": 1.0}])
check("F5 the sweep ranks on final_val, not best_val", curves["x"] == [(0.0125, 3.9)], curves)

with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    m.save_transfer(m.analyse_proxy(interior + right_edge))
    t = json.loads(m.TRANSFER.read_text())
    check("F6 only the bracketed arm is published", set(t["arms"]) == {"mingru"}, set(t["arms"]))
    check("F7 and the excluded arm says why", "attention" in t["excluded"])
    m.TRANSFER.unlink()
    m.save_transfer(m.analyse_proxy(interior[:-1]))       # incomplete sweep
    check("F8 an incomplete sweep publishes nothing", not m.TRANSFER.exists())
    unsandbox(old)


print("\n=== G. MATRIX: shape, blocking, and the hardware control ===")
jobs = m.build_matrix()
by = {}
for j in jobs:
    by.setdefault(j["suite"], []).append(j)
check("G1 job ids are unique", len({j["id"] for j in jobs}) == len(jobs))
check("G2 e1_sp_rerun is present by default", "e1_sp_rerun" in by,
      "the 2x2's SP cells must be measured on the same box as its muP cells")
check("G3 --sp-cells suite24 drops them (GH200 only)",
      "e1_sp_rerun" not in {j["suite"] for j in m.build_matrix(sp_cells="suite24")})
check("G4 dropping them removes exactly 10 jobs",
      len(jobs) - len(m.build_matrix(sp_cells="suite24")) == 10)

# The proxy must be a NARROW model, not the target with a thinner residual stream.
prox = by["e1_proxy"][0]["overrides"]
tgt = by["e1_mup"][0]["overrides"]
check("G5 the proxy's attention inner width equals its own width",
      prox["n_head"] * prox["head_dim"] == prox["d_model"] == m.BASE_WIDTH,
      f"{prox['n_head']}x{prox['head_dim']} vs d_model {prox['d_model']}")
check("G6 the target's attention inner width equals its own width",
      tgt["n_head"] * tgt["head_dim"] == tgt["d_model"] == m.TARGET_WIDTH)
check("G7 the proxy is genuinely narrower than the target",
      prox["n_head"] < tgt["n_head"] and prox["d_model"] < tgt["d_model"])
check("G8 head_dim is held fixed across the transfer",
      prox["head_dim"] == tgt["head_dim"] == m.HEAD_DIM)

# The 2x2's two rows must differ ONLY in the parametrization.
sp = {j["id"].replace("e1_sp_rerun_", ""): j["overrides"] for j in by["e1_sp_rerun"]}
mu = {j["id"].replace("e1_mup_", ""): j["overrides"] for j in by["e1_mup"]}
check("G9 the 2x2's SP and muP cells cover the same arms and seeds",
      set(sp) == set(mu), set(sp) ^ set(mu))
k = sorted(sp)[0]
diff = {f for f in set(sp[k]) | set(mu[k])
        if sp[k].get(f, "<->") != mu[k].get(f, "<->")} - {"run_name"}
check("G10 SP and muP cells differ only in parametrization fields",
      diff <= {"mup", "mup_base_width", "matrix_lr"}, diff)

bad11 = bad11b = None
for j in jobs:
    o = j["overrides"]
    tps = o["batch_size"] * o["grad_accum"] * o["block_size"]
    got = tps * o["max_steps"]
    # max_steps floors the budget; the recorded token_budget must be what RUNS.
    if got != j["token_budget"]:
        bad11 = bad11 or f"{j['id']}: recorded {j['token_budget']} but runs {got}"
    short = j["token_budget_requested"] - got
    if not (0 <= short < tps):
        bad11b = bad11b or f"{j['id']}: short by {short}, one step is {tps}"
check("G11 the recorded token budget is the one that runs", bad11 is None, bad11)
check("G11b and it is under the request by less than one step", bad11b is None, bad11b)
check("G11c the 50M suites realize 49,987,584 tokens, as suites 22-26 report",
      {j["token_budget"] for j in by["e2_matched32_50m"]} == {49_987_584},
      {j["token_budget"] for j in by["e2_matched32_50m"]})

s24 = [j for j in jobs if j["suite"] == "e1_sp_rerun"][0]["overrides"]
check("G12 the 20M stop keeps the 50M cosine horizon (suite 24, not suite 23)",
      s24["lr_max_steps"] > s24["max_steps"],
      f"lr_max_steps {s24['lr_max_steps']} vs max_steps {s24['max_steps']}")

d10 = {j["tag"]: j["overrides"] for j in by["d10_horizon"]}
d10diff = {f for f in d10["10k"] if d10["10k"][f] != d10["20k"].get(f)} - {"run_name"}
check("G13 the d10 pair differs only in horizon",
      d10diff == {"max_steps", "lr_max_steps", "eval_interval", "ckpt_interval",
                  "log_interval"} or d10diff <= {"max_steps", "lr_max_steps",
                                                 "eval_interval", "ckpt_interval",
                                                 "log_interval"}, d10diff)
check("G14 both d10 arms share one learning rate",
      "lr" not in d10diff and "matrix_lr" not in d10diff)

check("G15 e1_mup is blocked with no transfer",
      all(j.get("blocked_on") for j in by["e1_mup"]))
tj = m.build_matrix(transfer={"arms": {"attention": {"matrix_lr": 0.00625, "bracketed": True},
                                       "mingru": {"matrix_lr": 0.0125, "bracketed": True}}})
mups = [j for j in tj if j["suite"] == "e1_mup"]
check("G16 a transfer unblocks it", not any(j.get("blocked_on") for j in mups))
check("G17 and the transferred matrix_lr reaches the config",
      {j["overrides"]["matrix_lr"] for j in mups} == {0.00625, 0.0125},
      {j["overrides"]["matrix_lr"] for j in mups})
half = m.build_matrix(transfer={"arms": {"attention": {"matrix_lr": 0.00625}}})
check("G18 a half-finished transfer blocks only the untuned arm",
      {j["arm"] for j in half if j["suite"] == "e1_mup" and j.get("blocked_on")} == {"mingru"})


print("\n=== H. END TO END: the real main(), with a fake trainer ===")
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    old = sandbox(tmp)

    ft = FakeTrainer()
    rc = run_main(["--only", "e1_proxy", "--skip-preflight", "--workers", "4"], ft)
    led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
    check("H1 a suite runs to completion", rc == 0 and led["jobs_done"] == 12,
          f"rc={rc} done={led['jobs_done']}")
    check("H2 each job ran exactly once", len(ft.calls) == 12
          and len({c[0] for c in ft.calls}) == 12, len(ft.calls))

    ft2 = FakeTrainer()
    rc = run_main(["--only", "e1_proxy", "--skip-preflight"], ft2)
    check("H3 a second launch is a no-op (resume)", rc == 0 and not ft2.calls, ft2.calls)

    # THE REGRESSION: running a second suite must not erase the first
    ft3 = FakeTrainer()
    run_main(["--only", "e1_embed_lr", "--skip-preflight", "--workers", "2"], ft3)
    led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
    ids = {j["id"] for j in led["jobs"]}
    check("H4 --only twice keeps BOTH suites in the ledger",
          sum(1 for i in ids if i.startswith("e1_proxy")) == 12
          and sum(1 for i in ids if i.startswith("e1_embed_lr")) == 10, len(ids))
    check("H5 gpu-hours accumulate across both", led["gpu_hours_measured"] > 0)

    # FakeTrainer returned a constant loss, so every arm's minimum is the first
    # grid point -- an edge. Nothing may be transferred from that.
    check("H6 a flat sweep publishes no transfer at all", not m.TRANSFER.exists(),
          m.TRANSFER.read_text() if m.TRANSFER.exists() else "")
    unsandbox(old)

# ... and the same sweep with a real interior minimum DOES publish one, end to end.
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    peak = {"attention": 0.00625, "mingru": 0.0125}
    def parabola(job):
        import math
        lr = float(job["tag"].replace("mlr", ""))
        return 3.0 + (math.log(lr) - math.log(peak[job["arm"]])) ** 2 * 0.1
    rc = run_main(["--only", "e1_proxy", "--skip-preflight", "--workers", "3"],
                  FakeTrainer(loss=parabola))
    check("H6b a sweep with an interior minimum publishes a transfer",
          rc == 0 and m.TRANSFER.exists(), rc)
    t = json.loads(m.TRANSFER.read_text())
    check("H6c and it transfers each arm's OWN optimum",
          {a: v["matrix_lr"] for a, v in t["arms"].items()} == peak,
          {a: v["matrix_lr"] for a, v in t["arms"].items()})
    check("H6d marked bracketed", all(v["bracketed"] for v in t["arms"].values()))
    # the published transfer must now unblock the muP cells for a later launch
    mups = [j for j in m.build_matrix(transfer=t) if j["suite"] == "e1_mup"]
    check("H6e the published file unblocks e1_mup with the right per-arm LR",
          not any(j.get("blocked_on") for j in mups)
          and all(j["overrides"]["matrix_lr"] == peak[j["arm"]] for j in mups))
    unsandbox(old)

with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td)
    old = sandbox(tmp)
    rc = run_main(["--only", "e1_mup", "--skip-preflight"], FakeTrainer())
    check("H8 blocked jobs refuse to launch", rc == 1, rc)
    check("H9 and nothing was written for them",
          not (m.OUT_ROOT / "e1_mup_attention_s1337" / "metrics.jsonl").exists())

    m.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    m.TRANSFER.write_text(json.dumps({
        "base_width": 256, "target_width": 768,
        "arms": {"attention": {"matrix_lr": 0.00625, "bracketed": True},
                 "mingru": {"matrix_lr": 0.0125, "bracketed": True}}}), encoding="utf-8")
    ft = FakeTrainer()
    rc = run_main(["--only", "e1_mup", "--skip-preflight", "--workers", "5"], ft)
    check("H10 a published transfer unblocks the muP cells", rc == 0 and len(ft.calls) == 10,
          f"rc={rc} calls={len(ft.calls)}")
    led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
    lrs = {j.get("matrix_lr") for j in led["jobs"]}
    check("H11 the transferred matrix_lr is recorded per job", lrs == {0.00625, 0.0125}, lrs)
    check("H12 the ledger records where it came from",
          all(j.get("transfer", {}).get("source") for j in led["jobs"]))
    unsandbox(old)


print("\n=== I. FAILURE PATHS: nothing broken may read as measured ===")
for mode, label in (("nonzero", "a nonzero exit"),
                    ("nometrics", "an exit 0 that wrote no metrics"),
                    ("wrong_config", "a run whose config is not the one requested")):
    with tempfile.TemporaryDirectory() as td:
        old = sandbox(pathlib.Path(td))
        rc = run_main(["--only", "e1_embed_lr", "--skip-preflight"], FakeTrainer(mode))
        led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
        bad = [j for j in led["jobs"] if j["status"] == "failed"]
        check(f"I1.{mode} {label} -> every job failed, exit 1",
              rc == 1 and len(bad) == 10, f"rc={rc} failed={len(bad)}")
        check(f"I2.{mode} and the ledger carries a reason",
              all(j.get("failure_reason") for j in bad))
        unsandbox(old)

with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    run_main(["--only", "e1_embed_lr", "--skip-preflight"], FakeTrainer("die_midrun"))
    jid = "e1_embed_lr_attention_s1337"
    check("I3 a killed run is left `partial`",
          m.inspect_run(m.OUT_ROOT / jid)["status"] == "partial")
    ft = FakeTrainer()
    rc = run_main(["--only", "e1_embed_lr", "--skip-preflight"], ft)
    check("I4 a partial run is NOT silently re-run into the same file", not ft.calls, ft.calls)
    n_before = len((m.OUT_ROOT / jid / "metrics.jsonl").read_text().splitlines())
    run_main(["--only", "e1_embed_lr", "--reset-partial"], None)
    check("I5 --reset-partial archives it", not (m.OUT_ROOT / jid).exists()
          and any(m.ARCHIVE_ROOT.rglob(jid)))
    ft = FakeTrainer()
    rc = run_main(["--only", "e1_embed_lr", "--skip-preflight"], ft)
    check("I6 and then it re-runs clean", rc == 0 and len(ft.calls) == 10, len(ft.calls))
    st = m.inspect_run(m.OUT_ROOT / jid)
    check("I7 the re-run is a single segment, not appended to the old one",
          st["status"] == "done" and st["starts"] == 1, st)
    check("I8 (guard) the archived file still holds the old partial record", n_before > 0)
    unsandbox(old)


print("\n=== J. SCHEDULING: workers, devices, and no lost jobs ===")
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    ft = FakeTrainer()
    run_main(["--only", "e1_proxy", "--skip-preflight", "--workers", "4",
              "--gpus", "0,1,2,3"], ft)
    devs = {c[1] for c in ft.calls}
    check("J1 jobs are spread across every named device", devs == {"0", "1", "2", "3"}, devs)
    check("J2 every job ran exactly once under concurrency",
          len(ft.calls) == 12 and len({c[0] for c in ft.calls}) == 12, len(ft.calls))
    led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
    check("J3 no job is left `running` in the ledger",
          not [j for j in led["jobs"] if j["status"] == "running"])
    check("J4 the device is recorded per job",
          all(j.get("device") in {"0", "1", "2", "3"} for j in led["jobs"]))
    unsandbox(old)

with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    ft = FakeTrainer()
    rc = run_main(["--only", "e1_proxy", "--skip-preflight", "--workers", "8",
                   "--gpus", "0,1"], ft)
    check("J4b more workers than GPUs is refused, not silently co-located",
          rc == 1 and not ft.calls, f"rc={rc} calls={len(ft.calls)}")
    rc = run_main(["--only", "e1_proxy", "--skip-preflight", "--workers", "4",
                   "--gpus", "0,1", "--oversubscribe"], ft)
    check("J4c --oversubscribe allows it explicitly", rc == 0 and len(ft.calls) == 12,
          f"rc={rc} calls={len(ft.calls)}")
    unsandbox(old)

# --reset-partial must act on the tree the invocation points at, not always the matrix
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    jid = "e1_proxy_attention_mlr0.0016_s1337"
    write_run(m.OUT_ROOT / jid, starts=1, dones=0)
    write_run(m.SMOKE_ROOT / jid, starts=1, dones=0)
    run_main(["--only", "e1_proxy", "--smoke", "--reset-partial"], None)
    check("J4d --smoke --reset-partial archives the SMOKE dir, not the matrix dir",
          (m.OUT_ROOT / jid).exists() and not (m.SMOKE_ROOT / jid).exists(),
          f"matrix={{(m.OUT_ROOT/jid).exists()}} smoke={{(m.SMOKE_ROOT/jid).exists()}}")
    unsandbox(old)

# --report reads a ledger and never invents one
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    check("J4e --report on an empty tree exits non-zero rather than printing zeros",
          run_main(["--report"], None) == 1)
    run_main(["--only", "e1_embed_lr", "--skip-preflight"], FakeTrainer())
    check("J4f --report on a real ledger exits 0", run_main(["--report"], None) == 0)
    unsandbox(old)

check("J5 --workers is actually wired to a pool",
      "ThreadPoolExecutor" in src and "args.workers" in src,
      "the previous version declared --workers and ran serially")
check("J6 the child gets CUDA_VISIBLE_DEVICES", 'env["CUDA_VISIBLE_DEVICES"] = device' in src)


print("\n=== K. PREFLIGHT: gate the box before it bills ===")
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    jobs = m.build_matrix()
    m.detect_gpus, det = (lambda: 4), m.detect_gpus
    # a corpus smaller than the largest budget must fail closed
    rc = m.preflight(jobs, allow_repeat=False)
    check("K1 preflight fails when a job would re-read the corpus", rc == 1, rc)
    rc = m.preflight(jobs, allow_repeat=True)
    check("K2 --allow-data-repeat downgrades it to a warning", rc == 0, rc)
    d = m.OUT_ROOT / jobs[0]["id"]
    write_run(d, starts=1, dones=0)
    check("K3 a partial run dir fails preflight",
          m.preflight(jobs, allow_repeat=True) == 1)
    m.detect_gpus = det
    unsandbox(old)

check("K4 the launch path runs preflight unless told not to",
      "if not args.skip_preflight and not args.smoke:" in src)
check("K5 data_epochs is recorded per job so a repeat cannot go unreported",
      '"data_epochs"' in src)


print("\n=== L. CLI SURFACE ===")
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--plan"],
                   capture_output=True, text=True, timeout=300)
check("L1 --plan exits 0", r.returncode == 0, r.stderr[-300:])
check("L2 --plan totals 64 jobs", "TOTAL" in r.stdout and " 64" in r.stdout, r.stdout[-200:])
check("L3 --plan names the blocked suite", "blocked" in r.stdout.lower())
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--dry-run"],
                   capture_output=True, text=True, timeout=300)
check("L4 --dry-run marks blocked jobs", r.returncode == 0 and "[BLOCKED]" in r.stdout)
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--only", "nope"],
                   capture_output=True, text=True, timeout=300)
check("L5 an unknown suite is rejected with the known list",
      r.returncode != 0 and "e1_proxy" in (r.stderr + r.stdout))
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--help"],
                   capture_output=True, text=True, timeout=300)
for flag in ("--workers", "--gpus", "--sp-cells", "--preflight", "--reset-partial",
             "--report", "--allow-data-repeat"):
    check(f"L6{flag} is documented", flag in r.stdout)


print("\n=== M. PARAMETRIZATION: the overrides must reach the optimizer ===")
# The point of E1 is that muP transfers a tuned hidden-layer LR. If the override
# does not change the LR the optimizer actually uses, the arm measures nothing.
try:
    import torch
    torch.set_num_threads(1)
    from nanolab.model import build_model
    from nanolab.optim import build_optimizers
except ImportError as e:
    check("M0 torch available for the parametrization checks", False, str(e))
else:
    TRANSFER = {"arms": {"attention": {"matrix_lr": 0.00625, "bracketed": True},
                         "mingru": {"matrix_lr": 0.0125, "bracketed": True}}}
    SMALL = dict(n_layer=1, vocab_size=128, block_size=16, batch_size=2)

    def lrs(job):
        cfg = m.build_config(m.PRESET, dict(job["overrides"], **SMALL))
        model = build_model(cfg)
        opts = build_optimizers(model, cfg)
        return (opts[0].param_groups[0]["lr"],      # Muon: 2-D hidden matrices
                opts[1].param_groups[0]["lr"],      # AdamW: embeddings + head
                opts[1].param_groups[1]["lr"],      # AdamW: scalars
                cfg)

    jobs = {(j["suite"], j["arm"]): j for j in m.build_matrix(transfer=TRANSFER)}
    base_mlr = m.build_config(m.PRESET, {"run_name": "p"}).matrix_lr
    width = m.TARGET_WIDTH / m.BASE_WIDTH

    mlr, emb, sca, cfg = lrs(jobs[("e1_sp_rerun", "attention")])
    check("M1 the SP cell uses the suite's inherited matrix_lr unchanged",
          mlr == base_mlr and cfg.mup is False, f"{mlr} vs {base_mlr}")

    for arm in ("attention", "mingru"):
        want = TRANSFER["arms"][arm]["matrix_lr"] / width
        mlr, emb, sca, cfg = lrs(jobs[("e1_mup", arm)])
        check(f"M2.{arm} the muP cell gets transferred_lr / (768/256)",
              abs(mlr - want) < 1e-12, f"{mlr} != {want}")
        check(f"M3.{arm} and its embedding LR stays width-constant (the muP rule)",
              emb == cfg.lr == 6e-4, f"{emb}")
    check("M4 the two muP arms get DIFFERENT hidden LRs (per-arm tuning)",
          lrs(jobs[("e1_mup", "attention")])[0] != lrs(jobs[("e1_mup", "mingru")])[0])

    mlr, emb, sca, cfg = lrs(jobs[("e1_proxy", "attention")])
    check("M5 at the base width muP is a no-op on the LR (width_mult = 1)",
          abs(mlr - cfg.matrix_lr) < 1e-12 and cfg.d_model == m.BASE_WIDTH,
          f"{mlr} vs {cfg.matrix_lr}")

    import math
    mlr, emb, sca, cfg = lrs(jobs[("e1_perlayer_sp", "attention")])
    check("M6 per-layer SP scales the hidden LR by 1/sqrt(width ratio)",
          abs(mlr - base_mlr / math.sqrt(width)) < 1e-9, mlr)
    check("M7 and leaves the embedding LR alone (Everett et al. Table 1)", emb == cfg.lr)

    mlr, emb, sca, cfg = lrs(jobs[("e1_embed_lr", "attention")])
    check("M8 the embedding-LR probe raises ONLY the embedding LR, by the width ratio",
          abs(emb - cfg.lr * width) < 1e-12 and mlr == base_mlr and sca == cfg.lr,
          f"emb {emb} muon {mlr} scalar {sca}")

    # muP and per-layer SP are different parametrizations and must not compose
    try:
        cfg = m.build_config(m.PRESET, dict(run_name="x", mup=True, per_layer_sp=True, **SMALL))
        build_optimizers(build_model(cfg), cfg)
        ok = False
    except ValueError:
        ok = True
    check("M9 mup + per_layer_sp together is refused, not silently combined", ok)

    # every arm must build and produce finite gradients
    bad = []
    for key, job in sorted(jobs.items()):
        cfg = m.build_config(m.PRESET, dict(job["overrides"], **SMALL))
        model = build_model(cfg)
        x = torch.randint(0, 128, (2, 16))
        _, loss = model(x, x)
        loss.backward()
        if not math.isfinite(loss.item()):
            bad.append(f"{key} loss {loss.item()}")
        for n, prm in model.named_parameters():
            if prm.grad is not None and not torch.isfinite(prm.grad).all():
                bad.append(f"{key} {n}")
    check("M10 every arm builds and back-propagates finite gradients", not bad, bad[:3])


print(f"\n{'=' * 70}\nPASS {len(PASSES)}   FAIL {len(FAILS)}")
if FAILS:
    print("\nFAILING:")
    for f in FAILS:
        print("  -", f)
sys.exit(1 if FAILS else 0)
