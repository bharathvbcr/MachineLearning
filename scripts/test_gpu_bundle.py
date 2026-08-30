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

# Counts are DERIVED from the matrix, never typed: this file exists to catch a
# runner that silently drops jobs, and a hand-typed total is the one assertion
# that gets updated to match the bug.
N_PROXY = sum(1 for j in m.build_matrix() if j["suite"] == "e1_proxy")
N_ALL = len(m.build_matrix())

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
    old = (m.OUT_ROOT, m.SMOKE_ROOT, m.ARCHIVE_ROOT, m.TRANSFER, m.CROSSINGS,
           m.ANCHOR)
    m.OUT_ROOT = tmp / "gpu_bundle"
    m.SMOKE_ROOT = m.OUT_ROOT / "_smoke"
    m.ARCHIVE_ROOT = m.OUT_ROOT / "_archived"
    m.TRANSFER = m.OUT_ROOT / "transfer.json"
    m.CROSSINGS = m.OUT_ROOT / "crossings.json"
    m.ANCHOR = m.OUT_ROOT / "anchor.json"
    return old


def unsandbox(old):
    (m.OUT_ROOT, m.SMOKE_ROOT, m.ARCHIVE_ROOT, m.TRANSFER, m.CROSSINGS,
     m.ANCHOR) = old


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


def run_main(argv, trainer=None, gpus=0, transfer=None, vram_gib=0.0):
    """Drive the real main() with a fake trainer and no GPU probe.

    `transfer` publishes a transfer.json first, the way a finished e1_proxy would,
    so the muP suites unblock. Passing a different one on a later call is how a
    re-anchored learning rate is simulated.
    """
    old_run, old_det, old_argv = m.run_one, m.detect_gpus, sys.argv
    old_vram = m.device_total_vram_gib
    if trainer is not None:
        m.run_one = trainer
    m.detect_gpus = lambda: gpus
    # Stubbed for the same reason detect_gpus is: otherwise these cases pass on a
    # laptop with no CUDA and fail on the box with a real 98 GiB card, which is a
    # test reporting the environment rather than the code. 0.0 means "no cap".
    m.device_total_vram_gib = lambda: vram_gib
    sys.argv = ["gpu_bundle.py"] + argv
    if transfer is not None:
        m.TRANSFER.parent.mkdir(parents=True, exist_ok=True)
        m.TRANSFER.write_text(json.dumps(transfer))
    try:
        return m.main()
    finally:
        m.run_one, m.detect_gpus, sys.argv = old_run, old_det, old_argv
        m.device_total_vram_gib = old_vram


# ===========================================================================
print("\n=== A. SMOKE ISOLATION: --smoke must never mark a real job done ===")
# The documented procedure is "--smoke first on the rented box". The previous
# runner wrote 40-step smoke runs into the SAME directory the matrix resumes
# from, so following the docs silently substituted a 40-step best_val for one
# real job per suite -- including e1_mup, whose readouts are pre-registered.
check("A1 smoke root is a distinct subtree", m.SMOKE_ROOT != m.OUT_ROOT
      and str(m.SMOKE_ROOT).startswith(str(m.OUT_ROOT)))
# ANCHOR was added to the module and not to sandbox(), so a test was one line from
# writing anchor.json into the real repository. Every module-level path under
# OUT_ROOT must be redirected, and this notices the next one that is not.
_PATHS = [n for n in dir(m) if n.isupper()
          and isinstance(getattr(m, n), pathlib.Path)
          and str(getattr(m, n)).startswith(str(m.ROOT / "nanolab/out"))]
with tempfile.TemporaryDirectory() as _td:
    _old = sandbox(pathlib.Path(_td))
    _leaked = [n for n in _PATHS if not str(getattr(m, n)).startswith(_td)]
    unsandbox(_old)
check("A1b sandbox() redirects EVERY module path under nanolab/out",
      not _leaked, f"not redirected: {_leaked}")

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


print("\n=== F. PROXY SWEEP: an edge is not an optimum, and one seed is not a sweep ===")
def rec(arm, lr, val, status="done", seed=1337):
    return {"suite": "e1_proxy", "arm": arm, "tag": f"mlr{lr:g}", "seed": seed,
            "status": status, "final_val": val, "best_val": val - 0.5}


def sweep(arm, vals, seeds=m.PROXY_SEEDS, jitter=0.0):
    """One record per (lr, seed). `jitter` displaces the LAST seed's curve so the
    argmin wins on the mean while losing on a seed."""
    out = []
    for lr, v in zip(m.PROXY_MATRIX_LRS, vals):
        for i, sd in enumerate(seeds):
            d = jitter if (i == len(seeds) - 1 and lr == m.PROXY_MATRIX_LRS[2]) else 0.0
            out.append(rec(arm, lr, v + d, seed=sd))
    return out


GRID = m.PROXY_MATRIX_LRS
# Curves are generated to the grid's length, not typed to it: the grid is extended
# whenever a minimum lands on an edge (it already has been once), and a fixture with
# a hard-coded point count silently shortens the sweep it is meant to be testing.
def vshape(n, low):
    """A curve with its single minimum at index `low`."""
    return [4.0 + 0.1 * abs(i - low) for i in range(n)]


interior = sweep("mingru", vshape(len(GRID), 2))
left_edge = sweep("attention", vshape(len(GRID), 0))
right_edge = sweep("attention", vshape(len(GRID), len(GRID) - 1))

a = m.analyse_proxy(interior)["arms"]["mingru"]
check("F1 an interior minimum is bracketed", a["bracketed"] and a["matrix_lr"] == GRID[2],
      f"{a['bracketed']} @ {a['matrix_lr']}")
check("F2 a minimum at the LOW edge is not bracketed",
      not m.analyse_proxy(left_edge)["arms"]["attention"]["bracketed"])
check("F3 a minimum at the HIGH edge is not bracketed",
      not m.analyse_proxy(right_edge)["arms"]["attention"]["bracketed"])
check("F4 an unfinished cell is ignored, not counted",
      m.analyse_proxy([r for r in interior if r["tag"] != f"mlr{GRID[-1]:g}"])
      ["arms"]["mingru"]["points"] == len(GRID) - 1)

# PAPER 3.2: minimum-over-evaluations is not a ranking. Every proxy cell stops at
# the same token budget, so the paired snapshot is the LAST eval.
pts = m.proxy_points([{"suite": "e1_proxy", "arm": "x", "tag": "mlr0.0125", "seed": 42,
                       "status": "done", "final_val": 3.9, "best_val": 1.0}])
check("F5 the sweep ranks on final_val, not best_val", pts["x"] == {0.0125: {42: 3.9}}, pts)

# The sweep decides the LR that every muP cell in the bundle inherits. An argmin
# that wins on the mean but loses on a seed is inside the seed spread -- the n=2
# result PAPER 8.3 had overturned three times -- and must not be transferred.
check("F6 the sweep aggregates seeds rather than overwriting them",
      a["n_seeds"] == len(m.PROXY_SEEDS), a["n_seeds"])
check("F7 an interior minimum that wins on every seed is sign-consistent",
      a["sign_consistent"] and all(nb["seeds_won"] == nb["seeds_paired"]
                                   for nb in a["beats_neighbours"]))
flaky = m.analyse_proxy(sweep("mingru", vshape(len(GRID), 2), jitter=0.5))["arms"]["mingru"]
check("F8 a minimum that loses to a neighbour on one seed is NOT sign-consistent",
      flaky["bracketed"] and not flaky["sign_consistent"],
      f"bracketed={flaky['bracketed']} consistent={flaky['sign_consistent']}")
check("F9 every cell carries a Student-t interval and its per-seed values",
      all(c["lo"] <= c["mean"] <= c["hi"] and len(c["per_seed"]) == c["n"]
          for c in a["curve"]))

with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    m.save_transfer(m.analyse_proxy(interior + right_edge))
    t = json.loads(m.TRANSFER.read_text())
    check("F10 only the bracketed arm is published", set(t["arms"]) == {"mingru"}, set(t["arms"]))
    check("F11 and the excluded arm says why", "attention" in t["excluded"])
    m.TRANSFER.unlink()
    m.save_transfer(m.analyse_proxy([r for r in interior if r["tag"] != f"mlr{GRID[-1]:g}"]))
    check("F12 an incomplete sweep publishes nothing", not m.TRANSFER.exists())
    m.save_transfer(m.analyse_proxy(sweep("mingru", vshape(len(GRID), 2), jitter=0.5)))
    check("F13 a sign-inconsistent minimum publishes nothing", not m.TRANSFER.exists())
    m.save_transfer(m.analyse_proxy(interior))
    check("F14 and the published rule names the unvalidated Muon divisor",
          "Muon" in json.loads(m.TRANSFER.read_text())["rule"])
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
check("G4 dropping them removes every SP cell, at all three recipes",
      {j["suite"] for j in jobs} - {j["suite"] for j in m.build_matrix(sp_cells="suite24")}
      == {"e1_sp_rerun", "e1_sp_sched20", "e1_sp_bs8"})

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

# d10 is opt-in: it is not on section 8.4's critical path and at one seed it was
# not claimable. It must be absent by default and n=3 when asked for.
check("G13 d10_horizon is absent from the default matrix", "d10_horizon" not in by)
d10jobs = [j for j in m.build_matrix(with_d10=True) if j["suite"] == "d10_horizon"]
check("G13b --with-d10 adds it at 3 seeds per arm, not 1",
      len(d10jobs) == 2 * len(m.BASIN_SEEDS)
      and len({j["seed"] for j in d10jobs}) == len(m.BASIN_SEEDS),
      f"{len(d10jobs)} jobs, {len({j['seed'] for j in d10jobs})} seeds")
d10 = {}
for j in d10jobs:
    d10.setdefault(j["tag"], {})[j["seed"]] = j["overrides"]
sd = sorted(d10["10k"])[0]
d10diff = {f for f in d10["10k"][sd] if d10["10k"][sd][f] != d10["20k"][sd].get(f)} - {"run_name"}
check("G13c the d10 pair differs only in horizon",
      d10diff <= {"max_steps", "lr_max_steps", "eval_interval", "ckpt_interval",
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
    check("H1 a suite runs to completion", rc == 0 and led["jobs_done"] == N_PROXY,
          f"rc={rc} done={led['jobs_done']}")
    check("H2 each job ran exactly once", len(ft.calls) == N_PROXY
          and len({c[0] for c in ft.calls}) == N_PROXY, len(ft.calls))

    ft2 = FakeTrainer()
    rc = run_main(["--only", "e1_proxy", "--skip-preflight"], ft2)
    check("H3 a second launch is a no-op (resume)", rc == 0 and not ft2.calls, ft2.calls)

    # THE REGRESSION: running a second suite must not erase the first
    ft3 = FakeTrainer()
    run_main(["--only", "e1_embed_lr", "--skip-preflight", "--workers", "2"], ft3)
    led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
    ids = {j["id"] for j in led["jobs"]}
    check("H4 --only twice keeps BOTH suites in the ledger",
          sum(1 for i in ids if i.startswith("e1_proxy")) == N_PROXY
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
          len(ft.calls) == N_PROXY and len({c[0] for c in ft.calls}) == N_PROXY,
          len(ft.calls))
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
    check("J4c --oversubscribe allows it explicitly", rc == 0 and len(ft.calls) == N_PROXY,
          f"rc={rc} calls={len(ft.calls)}")
    unsandbox(old)

# --oversubscribe says "co-locate them", not "fill the card". Four 23.3 GiB minGRU
# jobs on a 97.9 GiB device is 97.6% full, and that is how it happened here: the
# worker count was sized against attention's 17 GiB.
_sched = [j for j in m.build_matrix() if j["suite"] == "e1_mup_sched20"]
heavy = max(m.job_vram_gib(j) for j in _sched)
check("J5 the per-job VRAM figure is measured, and minGRU is the heavy arm",
      heavy == m.JOB_VRAM_DEFAULT_GIB == 23.3
      and m.job_vram_gib([j for j in _sched if j["arm"] == "attention"][0]) < heavy,
      heavy)
n_safe, per = m.vram_safe_workers(_sched, 97.9)
check("J6 the cap is sized on the HEAVIEST queued job, not the average",
      per == heavy and n_safe == int(97.9 * m.VRAM_HEADROOM / heavy) == 3,
      f"{n_safe} workers at {per} GiB")
check("J7 and it would have refused the 4 that nearly OOMed", n_safe < 4)
check("J8 an unknown shape is sized pessimistically, not optimistically",
      m.job_vram_gib({"overrides": {"mixer": "novel", "d_model": 9999,
                                    "batch_size": 7}}) == m.JOB_VRAM_DEFAULT_GIB)
check("J9 --ignore-vram is a SEPARATE escape hatch from --oversubscribe",
      "--ignore-vram" in src and "args.ignore_vram" in src)
check("J10 a device with no CUDA yields no cap rather than a bogus one",
      m.vram_safe_workers(_sched, 0.0) == (0, 0.0))

VR_XFER = {"arms": {a: {"matrix_lr": 0.00625, "bracketed": True} for a in m.ARMS}}
with tempfile.TemporaryDirectory() as td:
    old_s = sandbox(pathlib.Path(td))
    ft = FakeTrainer()
    (m.OUT_ROOT).mkdir(parents=True, exist_ok=True)
    m.ANCHOR.write_text(json.dumps(
        {"arms": {a: {"mult": 2.0} for a in m.ARMS}}))
    rc = run_main(["--only", "e1_mup_sched20", "--skip-preflight", "--workers", "4",
                   "--oversubscribe"], ft, transfer=VR_XFER, vram_gib=97.9)
    check("J11 --workers 4 on a 97.9 GiB card is refused end to end",
          rc == 1 and not ft.calls, f"rc={rc} calls={len(ft.calls)}")
    rc = run_main(["--only", "e1_mup_sched20", "--skip-preflight", "--workers", "4",
                   "--oversubscribe", "--ignore-vram"], ft, transfer=VR_XFER,
                  vram_gib=97.9)
    check("J12 --ignore-vram overrides it explicitly", rc == 0 and ft.calls,
          f"rc={rc} calls={len(ft.calls)}")
    # batch 8 is obviously lighter than batch 32, but it has not been MEASURED, so
    # it inherits the worst known figure and is capped with everything else. That is
    # the intended behaviour: an unmeasured shape must not be sized by a plausible
    # guess. Measuring it is what relaxes the cap, not arguing about it.
    ft2 = FakeTrainer()
    rc = run_main(["--only", "e1_mup_bs8", "--skip-preflight", "--workers", "4",
                   "--oversubscribe"], ft2, transfer=VR_XFER, vram_gib=97.9)
    check("J13 an unmeasured shape is capped at the worst known figure, not guessed",
          rc == 1 and not ft2.calls
          and ("mingru", 768, 8) not in m.JOB_VRAM_GIB, f"rc={rc}")
    rc = run_main(["--only", "e1_mup_bs8", "--skip-preflight", "--workers", "3",
                   "--oversubscribe"], ft2, transfer=VR_XFER, vram_gib=97.9)
    check("J14 and the cap it computes is actually runnable", rc == 0 and ft2.calls,
          f"rc={rc} calls={len(ft2.calls)}")
    unsandbox(old_s)


# --reset-partial must act on the tree the invocation points at, not always the matrix
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    # derived, not typed: the LR grid gets extended when a minimum lands on an edge,
    # which moves which job `--smoke` picks first
    jid = [j for j in m.build_matrix() if j["suite"] == "e1_proxy"][0]["id"]
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



# A comparator that moves the two arms in OPPOSITE directions is the worst case,
# not an acceptable one: it manufactures an ordering out of the parametrization.
# The old rule required "worse on every arm" too, so helping one arm bought a
# pass no matter how wide the spread.
def _health(dattn, dmingru):
    recs = []
    for arm, d in (("attention", dattn), ("mingru", dmingru)):
        for sd in m.SEEDS:
            recs.append({"suite": "e1_sp_rerun", "arm": arm, "seed": sd,
                         "status": "done", "final_val": 4.0})
            recs.append({"suite": "e1_mup", "arm": arm, "seed": sd,
                         "status": "done", "final_val": 4.0 + d})
    return m.parametrization_health(recs)["e1_mup"]

check("H20 an even-handed parametrization is usable however much it costs",
      _health(-0.30, -0.29)["usable_as_comparator"], _health(-0.30, -0.29))
check("H21 opposite-direction damage is refused even though one arm improved",
      not _health(-0.0977, +0.0991)["usable_as_comparator"],
      _health(-0.0977, +0.0991))
check("H22 and uniform damage with a wide spread is still refused",
      not _health(+0.5446, +0.0810)["usable_as_comparator"],
      _health(+0.5446, +0.0810))

print("\n=== K. PREFLIGHT: gate the box before it bills ===")
with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    det, dev, dsk = m.detect_gpus, m.device_name, m.free_gib
    m.detect_gpus = lambda: 4
    m.device_name = lambda: "NVIDIA GH200 480GB"
    # The box these gate has terabytes; the laptop running the tests may not.
    # Disk pressure here is the developer's, not the matrix's.
    m.free_gib = lambda: 100_000.0

    # d10's 20k arm asks for 655M tokens of a 497.5M corpus -- 1.32 epochs, while
    # its 10k partner sits at 0.66. That is a second variable moving alongside the
    # one the pair exists to isolate, so it fails closed.
    d10jobs = m.build_matrix(with_d10=True)
    check("K1 preflight fails when a job would re-read the corpus",
          m.preflight(d10jobs, allow_repeat=False) == 1)
    check("K2 --allow-data-repeat downgrades it to a warning",
          m.preflight(d10jobs, allow_repeat=True) == 0)
    jobs = m.build_matrix()
    check("K2b the default matrix stays inside the corpus with no override",
          m.preflight(jobs, allow_repeat=False) == 0)

    d = m.OUT_ROOT / jobs[0]["id"]
    write_run(d, starts=1, dones=0)
    check("K3 a partial run dir fails preflight",
          m.preflight(jobs, allow_repeat=True) == 1)
    shutil.rmtree(d)

    # The hardware control, enforced. e2's cells are merged into suite 26's GH200
    # board; measured on another GPU they carry the cross-hardware confound PAPER
    # 7.1 refuses, which leaves that board worse than it found it.
    m.device_name = lambda: "NVIDIA A100-SXM4-40GB"
    check("K6 the GH200-gated suites fail closed on another GPU",
          m.preflight(jobs, allow_repeat=True) == 1)
    check("K6b --allow-cross-hardware-board proceeds with the caveat",
          m.preflight(jobs, allow_repeat=True, allow_cross_hardware=True) == 0)
    check("K6c dropping the gated suite needs no override",
          m.preflight([j for j in jobs if j["suite"] not in m.GH200_REQUIRED],
                      allow_repeat=True) == 0)
    check("K7 --sp-cells suite24 is refused off a GH200",
          m.preflight([j for j in jobs if j["suite"] not in m.GH200_REQUIRED],
                      allow_repeat=True, sp_cells="suite24") == 1)
    m.device_name = lambda: "NVIDIA GH200 480GB"
    check("K7b and accepted on one",
          m.preflight([j for j in m.build_matrix(sp_cells="suite24")
                       if j["suite"] not in m.GH200_REQUIRED],
                      allow_repeat=True, sp_cells="suite24") == 0)

    m.detect_gpus, m.device_name, m.free_gib = det, dev, dsk
    unsandbox(old)

check("K4 the launch path runs preflight unless told not to",
      "if not args.skip_preflight and not args.smoke:" in src)
check("K5 data_epochs is recorded per job so a repeat cannot go unreported",
      '"data_epochs"' in src)
check("K9 the corpus size is checked against the one suites 22-26 trained on",
      "REFERENCE_CORPUS_TOKENS" in src
      and m.REFERENCE_CORPUS_TOKENS == 497_500_000)
check("K8 the device model is recorded in the ledger, not just checked",
      '"device_name"' in src and '"is_gh200"' in src)


print("\n=== L. CLI SURFACE ===")
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--plan"],
                   capture_output=True, text=True, timeout=300)
check("L1 --plan exits 0", r.returncode == 0, r.stderr[-300:])
check(f"L2 --plan totals {N_ALL} jobs",
      "TOTAL" in r.stdout and f" {N_ALL}" in r.stdout, r.stdout[-200:])
check("L3 --plan reports a blocked column for every suite",
      "blocked" in r.stdout.lower()
      and all(su in r.stdout for su in m.SUITE_ORDER if su != "d10_horizon"))
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--dry-run"],
                   capture_output=True, text=True, timeout=300)
# These two drive the real CLI against the REAL repo, where a transfer.json may or
# may not exist depending on how far a live run has got. Asserting "[BLOCKED]
# appears" passes on a fresh checkout and fails on a box mid-experiment -- a test
# reporting the environment rather than the code. Compare against the matrix the
# same state produces instead, which is exact either way; blocking itself is
# covered in-process and sandboxed by G15/G18 and N5.
# built exactly as main() builds it -- transfer AND anchor -- or this compares the
# CLI against a different matrix than the CLI is printing
LIVE = dict(transfer=m.load_transfer(), anchor=m.load_anchor())
n_blocked = sum(1 for j in m.build_matrix(**LIVE) if j.get("blocked_on"))
check("L4 --dry-run marks exactly the blocked jobs, and no others",
      r.returncode == 0 and r.stdout.count("[BLOCKED]") == n_blocked,
      f"printed {r.stdout.count('[BLOCKED]')}, matrix has {n_blocked}")
check("L4b --dry-run prints every job with its overrides",
      r.stdout.count("\n") >= len(m.build_matrix(**LIVE)))
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--only", "nope"],
                   capture_output=True, text=True, timeout=300)
check("L5 an unknown suite is rejected with the known list",
      r.returncode != 0 and "e1_proxy" in (r.stderr + r.stdout))
r = subprocess.run([sys.executable, "scripts/gpu_bundle.py", "--help"],
                   capture_output=True, text=True, timeout=300)
for flag in ("--workers", "--gpus", "--sp-cells", "--preflight", "--reset-partial",
             "--report", "--allow-data-repeat"):
    check(f"L6{flag} is documented", flag in r.stdout)


print("\n=== N. RECIPE COVERAGE: the matrix must be able to answer its own readouts ===")
# The defect this section exists to catch: every E1 cell ran at the s24 recipe,
# while three of PAPER 8.4's four pre-registered readouts are statements about the
# s23 (20M cosine) and s25 (batch 8) recipes. The run would have produced data and
# then had to report "not measured" against its own decision rule.
suites_default = {j["suite"] for j in m.build_matrix()}
missing = {}
for r in m.readouts({"cells": {}}):
    absent = [n for n in r["needs"] if n not in suites_default]
    if absent:
        missing[r["row"]] = absent
check("N1 every pre-registered readout has its cells in the default matrix",
      not missing, missing)

# and each of those cells must actually be the published recipe it claims to be
def one(suite):
    return [j for j in m.build_matrix() if j["suite"] == suite][0]["overrides"]

s24o, s23o, s25o = one("e1_sp_rerun"), one("e1_sp_sched20"), one("e1_sp_bs8")
check("N2 s23 differs from s24 in the cosine horizon and nothing else",
      {f for f in s24o if s24o[f] != s23o.get(f)} - {"run_name"} == {"lr_max_steps"},
      {f for f in s24o if s24o[f] != s23o.get(f)} - {"run_name"})
check("N2b s23's cosine ends where the run does; s24's runs past it",
      s23o["lr_max_steps"] == s23o["max_steps"] and s24o["lr_max_steps"] > s24o["max_steps"],
      f"s23 {s23o['lr_max_steps']}/{s23o['max_steps']}  "
      f"s24 {s24o['lr_max_steps']}/{s24o['max_steps']}")
check("N3 s25 is batch 8 at suite 14's token budget",
      s25o["batch_size"] == 8
      and [j for j in m.build_matrix() if j["suite"] == "e1_sp_bs8"][0]
      ["token_budget_requested"] == m.SUITE14_TOKEN_BUDGET, s25o["batch_size"])
check("N3b and its horizon covers 7.4M tokens, the readout's question",
      [j for j in m.build_matrix() if j["suite"] == "e1_sp_bs8"][0]["token_budget"]
      >= m.BS8_HORIZON_TOKENS)

# every recipe carries BOTH parametrizations, on the same box, at matched seeds
for sp_s, mup_s in (("e1_sp_rerun", "e1_mup"), ("e1_sp_sched20", "e1_mup_sched20"),
                    ("e1_sp_bs8", "e1_mup_bs8")):
    got = {}
    for j in m.build_matrix():
        if j["suite"] in (sp_s, mup_s):
            got.setdefault(j["suite"], set()).add((j["arm"], j["seed"]))
    check(f"N4.{mup_s} SP and muP cover the same arms and seeds",
          got.get(sp_s) == got.get(mup_s) and len(got.get(sp_s, ())) == 2 * len(m.SEEDS),
          f"{len(got.get(sp_s, ()))} vs {len(got.get(mup_s, ()))}")

check("N5 every muP suite is blocked until the sweep publishes a transfer",
      {j["suite"] for j in m.build_matrix() if j.get("blocked_on")}
      == set(m.MUP_TRANSFER_SUITES),
      {j["suite"] for j in m.build_matrix() if j.get("blocked_on")})
TR = {"arms": {"attention": {"matrix_lr": 0.00625, "bracketed": True},
               "mingru": {"matrix_lr": 0.0125, "bracketed": True}}}
check("N5b a transfer alone unblocks only the un-anchored muP suites",
      {j["suite"] for j in m.build_matrix(transfer=TR) if j.get("blocked_on")}
      == set(m.MUP_ANCHOR_SUITES),
      {j["suite"] for j in m.build_matrix(transfer=TR) if j.get("blocked_on")})
AN = {"arms": {"attention": {"mult": 4.0}, "mingru": {"mult": 2.0}}}
check("N5c and a transfer plus an anchor unblocks all of them",
      not any(j.get("blocked_on")
              for j in m.build_matrix(transfer=TR, anchor=AN)))
# expectations derived from the fixtures, not typed: transferred x anchor multiplier
want_tuned = {(a, TR["arms"][a]["matrix_lr"] * AN["arms"][a]["mult"]) for a in m.ARMS}
got_tuned = {(j["arm"], j["overrides"]["matrix_lr"])
             for j in m.build_matrix(transfer=TR, anchor=AN)
             if j["suite"] == "e1_mup_tuned"}
check("N5d the anchored suites run at the MEASURED optimum, per arm",
      got_tuned == want_tuned, f"{got_tuned} != {want_tuned}")
want_xfer = {TR["arms"][a]["matrix_lr"] for a in m.ARMS}
got_xfer = {j["overrides"]["matrix_lr"] for j in m.build_matrix(transfer=TR, anchor=AN)
            if j["suite"] == "e1_mup"}
check("N5e while e1_mup stays at the TRANSFERRED value -- the failed transfer is kept",
      got_xfer == want_xfer, f"{got_xfer} != {want_xfer}")
check("N5g and the two are genuinely different learning rates",
      got_tuned != {(a, lr) for a in m.ARMS for lr in want_xfer if True} and
      any(TR["arms"][a]["matrix_lr"] * AN["arms"][a]["mult"]
          != TR["arms"][a]["matrix_lr"] for a in m.ARMS))
check("N5f the readouts read the TUNED cell, never the failed-transfer one",
      all("e1_mup_tuned" in r["needs"] or "e1_mup" not in r["needs"]
          for r in m.readouts({"cells": {}})))


print("\n=== P. BASIN: a single learning rate is not a measurement of a basin ===")
# PAPER 8.3's mechanism is offset flat basins: two arms each measured at ONE
# learning rate can sit on opposite sides of their own optima, and the ordering
# that falls out is an artifact of where they were measured. The muP arm was
# designed with exactly one point per arm.
bas = [j for j in m.build_matrix(transfer=TR) if j["suite"] == "e1_mup_basin"]
mup = [j for j in m.build_matrix(transfer=TR) if j["suite"] == "e1_mup"]
per_arm = {}
for j in bas + mup:
    per_arm.setdefault(j["arm"], set()).add(j["overrides"]["matrix_lr"])
check("P1 each arm gets a multi-point LR curve at the TARGET width",
      all(len(v) == len(m.BASIN_MULTS) + 1 for v in per_arm.values()),
      {k: len(v) for k, v in per_arm.items()})
check("P2 centred on the transferred value, which stays interior",
      all(min(v) < TR["arms"][a]["matrix_lr"] < max(v) for a, v in per_arm.items()))
check("P3 spanning at least 16x, enough to bracket a 3x transfer-rule error",
      all(max(v) / min(v) >= 16 for v in per_arm.values()),
      {k: max(v) / min(v) for k, v in per_arm.items()})
check("P4 the basin points carry the multiplier that produced them",
      all(j.get("transfer", {}).get("mult") in m.BASIN_MULTS for j in bas))
check("P5 and the applied LR is the transferred value times that multiplier",
      all(abs(j["overrides"]["matrix_lr"]
              - TR["arms"][j["arm"]]["matrix_lr"] * j["transfer"]["mult"]) < 1e-12
          for j in bas))
check("P6 basin cells run at 3 seeds, not 1",
      len({j["seed"] for j in bas}) == len(m.BASIN_SEEDS) >= 3,
      len({j["seed"] for j in bas}))
check("P7 a basin cell differs from its 1x cell in matrix_lr alone",
      {f for f in mup[0]["overrides"]
       if mup[0]["overrides"][f] != bas[0]["overrides"].get(f)}
      - {"run_name", "seed"} == {"matrix_lr"},
      {f for f in mup[0]["overrides"]
       if mup[0]["overrides"][f] != bas[0]["overrides"].get(f)} - {"run_name", "seed"})


# --smoke must be runnable BEFORE the sweep -- it is the gate that checks the box
# before it bills, and the muP code path is the part most worth checking.
with tempfile.TemporaryDirectory() as td:
    old_s = sandbox(pathlib.Path(td))
    ft = FakeTrainer()
    rc = run_main(["--smoke", "--workers", "1"], ft)
    ran = {i.rsplit("_s", 1)[0] for i, _ in ft.calls}
    check("P8 --smoke covers every suite with no transfer published",
          rc == 0 and len(ft.calls) == len({j["suite"] for j in m.build_matrix()}),
          f"rc={rc} ran {len(ft.calls)}")
    check("P9 and it touched the muP suites, not just the unblocked ones",
          any("e1_mup" in i for i, _ in ft.calls), sorted(ran))
    check("P10 smoke output stays in the isolated subtree",
          m.SMOKE_ROOT.exists() and not (m.OUT_ROOT / "ledger.json").exists())
    check("P11 a REAL run still refuses the muP suites without a transfer",
          run_main(["--only", "e1_mup", "--skip-preflight"], FakeTrainer()) == 1)
    unsandbox(old_s)


# The basin's own readout: did the transfer actually land at the target width?
def bcell(suite, arm, mult, val, seeds=m.BASIN_SEEDS):
    out = []
    for i, sd in enumerate(seeds):
        r = {"suite": suite, "arm": arm, "seed": sd, "tag": f"x{mult:g}" if mult != 1 else "",
             "status": "done", "final_val": val + 0.001 * i, "curve": []}
        if mult != 1.0:
            r["transfer"] = {"mult": mult}
        out.append(r)
    return out


# a curve minimised at the transferred value -> transfer verified
good = []
for mult, v in zip((0.25, 0.5, 1.0, 2.0, 4.0), (4.4, 4.2, 4.0, 4.1, 4.3)):
    good += bcell("e1_mup" if mult == 1.0 else "e1_mup_basin", "attention", mult, v)
ab = m.analyse_basin(good)["arms"]["attention"]
check("P12 a minimum at the transferred value reads as a verified transfer",
      ab["transfer_verified"] and ab["best_mult"] == 1.0, ab["best_mult"])

# a curve minimised at 2x -- what the competing no-divisor Muon rule predicts
missed = []
for mult, v in zip((0.25, 0.5, 1.0, 2.0, 4.0), (4.4, 4.3, 4.1, 4.0, 4.2)):
    missed += bcell("e1_mup" if mult == 1.0 else "e1_mup_basin", "attention", mult, v)
am = m.analyse_basin(missed)["arms"]["attention"]
check("P13 a minimum elsewhere is a MISSED transfer, not a muP measurement",
      am["bracketed"] and not am["transfer_verified"] and am["best_mult"] == 2.0,
      am)
# a boundary minimum is neither
edge = []
for mult, v in zip((0.25, 0.5, 1.0, 2.0, 4.0), (4.4, 4.3, 4.2, 4.1, 4.0)):
    edge += bcell("e1_mup" if mult == 1.0 else "e1_mup_basin", "attention", mult, v)
ae = m.analyse_basin(edge)["arms"]["attention"]
check("P14 a boundary minimum is reported as unbracketed, not as a transfer verdict",
      not ae["bracketed"] and not ae["transfer_verified"], ae)
# "Verified" must not be declarable on a mean difference smaller than the seed
# spread -- the 1x cell has five seeds and the basin points three, and locating a
# minimum by unequal-n means is the PAPER 8.3 error.
def bshaky(arm, mult, vals, seeds=m.BASIN_SEEDS):
    out = []
    for sd, v in zip(seeds, vals):
        r = {"suite": "e1_mup" if mult == 1.0 else "e1_mup_basin", "arm": arm,
             "seed": sd, "tag": "" if mult == 1.0 else f"x{mult:g}",
             "status": "done", "final_val": v, "curve": []}
        if mult != 1.0:
            r["transfer"] = {"mult": mult}
        out.append(r)
    return out

# 1x wins on the mean but loses to 2x on the last shared seed
shaky = (bshaky("attention", 0.25, [4.40, 4.41, 4.42])
         + bshaky("attention", 0.5, [4.30, 4.31, 4.32])
         + bshaky("attention", 1.0, [3.95, 3.96, 4.08])
         + bshaky("attention", 2.0, [4.05, 4.06, 4.07])
         + bshaky("attention", 4.0, [4.20, 4.21, 4.22]))
sk = m.analyse_basin(shaky)["arms"]["attention"]
check("P16 a minimum inside the seed spread is NOT a verified transfer",
      sk["best_mult"] == 1.0 and sk["bracketed"]
      and not sk["sign_consistent"] and not sk["transfer_verified"], sk)
firm = (bshaky("attention", 0.25, [4.40, 4.41, 4.42])
        + bshaky("attention", 0.5, [4.30, 4.31, 4.32])
        + bshaky("attention", 1.0, [4.00, 4.01, 4.02])
        + bshaky("attention", 2.0, [4.05, 4.06, 4.07])
        + bshaky("attention", 4.0, [4.20, 4.21, 4.22]))
fm = m.analyse_basin(firm)["arms"]["attention"]
check("P17 one that wins on every shared seed is",
      fm["transfer_verified"] and all(nb["seeds_won"] == nb["seeds_paired"]
                                      for nb in fm["beats_neighbours"]), fm)
check("P18 the basin keeps per-seed values, not just means",
      all("per_seed" in c for c in fm["curve"]))

# With the optimum located, the two candidate width rules become a measurement.
# A rule off in OPPOSITE directions on the two arms is within grid resolution; one
# off in the SAME direction on both is biased, and that is the finding.
two = (bshaky("attention", 0.25, [5.7, 5.71, 5.72]) + bshaky("attention", 0.5, [5.5, 5.51, 5.52])
       + bshaky("attention", 1.0, [5.3, 5.31, 5.32]) + bshaky("attention", 2.0, [5.13, 5.14, 5.15])
       + bshaky("attention", 4.0, [5.10, 5.11, 5.12]) + bshaky("attention", 8.0, [5.17, 5.18, 5.19])
       + bshaky("mingru", 0.25, [5.15, 5.16, 5.17]) + bshaky("mingru", 0.5, [5.09, 5.10, 5.11])
       + bshaky("mingru", 1.0, [5.02, 5.03, 5.04]) + bshaky("mingru", 2.0, [4.99, 5.00, 5.01])
       + bshaky("mingru", 4.0, [5.00, 5.01, 5.02]) + bshaky("mingru", 8.0, [5.05, 5.06, 5.07]))
ab2 = m.analyse_basin(two)
check("P19 a located optimum prices BOTH candidate width rules",
      ab2["arms"]["attention"]["rules"]["muon_no_divisor"]["predicts_mult"]
      == m.TARGET_WIDTH / m.BASE_WIDTH
      and ab2["arms"]["attention"]["rules"]["optim_py_divisor"]["predicts_mult"] == 1.0)
v = ab2["rule_verdict"]
check("P20 the divisor rule is flagged as wrong in the same direction on both arms",
      v["optim_py_divisor"]["same_direction_on_every_arm"] is True
      and v["optim_py_divisor"]["worst_factor_off"] >= 4, v["optim_py_divisor"])
check("P21 the no-divisor rule's errors straddle 1.0 -- within grid resolution",
      v["muon_no_divisor"]["same_direction_on_every_arm"] is False
      and v["muon_no_divisor"]["worst_factor_off"] < 2, v["muon_no_divisor"])
check("P22 an unlocated optimum prices no rule at all",
      m.analyse_basin(shaky)["arms"]["attention"]["rules"] is None
      and "rule_verdict" not in m.analyse_basin(shaky))

check("P15 the analysis names the multiplier the competing Muon rule predicts",
      m.analyse_basin(good)["competing_rule_predicts_mult"]
      == m.TARGET_WIDTH / m.BASE_WIDTH)


print("\n=== O. CROSSINGS: the readouts must be derived, and never faked when absent ===")
def pair(suite, seed, xing, tag="", n=8, last=20e6):
    """Two records whose gap changes sign at `xing` tokens. attention starts high."""
    toks = [last * (i + 1) / n for i in range(n)]
    out = []
    for arm in ("attention", "mingru"):
        curve = [[int(t), 4.0 + (0.1 if arm == "attention" else -0.1) * (1 if t < xing else -1)]
                 for t in toks]
        out.append({"suite": suite, "arm": arm, "seed": seed, "tag": tag,
                    "status": "done", "curve": curve, "final_val": curve[-1][1]})
    return out


recs = []
for i, sd in enumerate(m.SEEDS):
    recs += pair("e1_mup_tuned", sd, 12.0e6 + i * 1e5)
cx = m.crossing_tokens(recs)
cell = cx["cells"]["e1_mup_tuned"]
check("O1 a crossing is computed per seed and then aggregated",
      cell["n_seeds"] == len(m.SEEDS) and len(cell["per_seed"]) == len(m.SEEDS))
check("O2 the interval brackets the per-seed spread",
      cell["last"]["lo"] <= cell["last"]["per_seed_min"]
      and cell["last"]["per_seed_max"] <= cell["last"]["hi"], cell["last"])
check("O3 a cell with no finished partner produces no crossing at all",
      "e1_mup_bs8" not in m.crossing_tokens(
          [r for r in recs if r["arm"] == "attention"])["cells"])

# The readout table must never report a row it could not measure.
rows = {r["row"]: r for r in m.readouts(cx)}
check("O4 rows whose cells did not run read `unanswered`, not pass or fail",
      rows[2]["verdict"] is None and rows[3]["verdict"] is None
      and rows[4]["verdict"] is None,
      {k: v["verdict"] for k, v in rows.items()})
check("O5 row 1 is answerable from the tuned s24 cell alone",
      rows[1]["verdict"] is not None)

# row 2: the 20M cosine must still cross LATER than the 50M cosine
late = list(recs)
for i, sd in enumerate(m.SEEDS):
    late += pair("e1_mup_sched20", sd, 14.5e6 + i * 1e5)
r2 = {r["row"]: r for r in m.readouts(m.crossing_tokens(late))}
check("O6 row 2 sees the ordering of recipes preserved", r2[2]["verdict"] is True)
check("O7 and row 3 does not call two separated intervals a collapse",
      r2[3]["verdict"] is False)
same = list(recs)
for i, sd in enumerate(m.SEEDS):
    same += pair("e1_mup_sched20", sd, 12.0e6 + i * 1e5)
r3 = {r["row"]: r for r in m.readouts(m.crossing_tokens(same))}
check("O8 identical crossings under both schedules read as a collapse",
      r3[3]["verdict"] is True and r3[2]["verdict"] is False)

# row 4: does the batch-8 arm develop a crossing inside 7.4M tokens?
bs8_xing = list(recs)
for sd in m.SEEDS:
    bs8_xing += pair("e1_mup_bs8", sd, 6.0e6, last=8.19e6)
r4 = {r["row"]: r for r in m.readouts(m.crossing_tokens(bs8_xing))}
check("O9 row 4 sees a crossing inside the 7.4M horizon", r4[4]["verdict"] is True)
flat = list(recs)
for sd in m.SEEDS:
    for arm, off in (("attention", 0.1), ("mingru", -0.1)):
        flat.append({"suite": "e1_mup_bs8", "arm": arm, "seed": sd, "tag": "",
                     "status": "done", "final_val": 4.0 + off,
                     "curve": [[int(8.19e6 * (i + 1) / 8), 4.0 + off] for i in range(8)]})
r5 = {r["row"]: r for r in m.readouts(m.crossing_tokens(flat))}
check("O10 and reports `no` -- not `unanswered` -- when the arm never crosses",
      r5[4]["verdict"] is False)

# Gap D1, one module over: a quantity measured once has no interval. `mean_ci`
# returns (mu, mu, mu) at n=1, and reporting that as an interval is infinite
# precision from one sample -- the defect `native_funnel` was corrected for.
one = pair("e1_mup_tuned", m.SEEDS[0], 12.0e6)
c1 = m.crossing_tokens(one)["cells"]["e1_mup_tuned"]["last"]
check("O11a a single-seed crossing reports no interval, not a zero-width one",
      c1["ci_informative"] is False and c1["lo"] is None and c1["hi"] is None, c1)
check("O11b a multi-seed crossing does report one",
      m.crossing_tokens(recs)["cells"]["e1_mup_tuned"]["last"]["ci_informative"] is True)
solo = one + pair("e1_mup_sched20", m.SEEDS[0], 12.0e6)
r6 = {r["row"]: r for r in m.readouts(m.crossing_tokens(solo))}
check("O11c and the overlap readout refuses a verdict from it",
      r6[3]["verdict"] is None, r6[3]["verdict"])

# The bands are PER-SEED bands, so a mean over the seeds that happened to cross is
# a different statement from all of them crossing.
partial = [r for r in recs if not (r["seed"] == m.SEEDS[0])]
for arm, off in (("attention", 0.1), ("mingru", -0.1)):
    partial.append({"suite": "e1_mup_tuned", "arm": arm, "seed": m.SEEDS[0], "tag": "",
                    "status": "done", "final_val": 4.0 + off,
                    "curve": [[int(20e6 * (i + 1) / 8), 4.0 + off] for i in range(8)]})
cp = m.crossing_tokens(partial)["cells"]["e1_mup_tuned"]
check("O11d a cell records how many seeds crossed, not just the mean of those that did",
      cp["seeds_with_a_crossing"] == len(m.SEEDS) - 1 and cp["n_seeds"] == len(m.SEEDS),
      f"{cp['seeds_with_a_crossing']}/{cp['n_seeds']}")
check("O11e and row 1 does not pass when a seed never crossed",
      {r["row"]: r for r in m.readouts(m.crossing_tokens(partial))}[1]["verdict"] is False)

# "not measured" and "measured, and the arms never crossed" are different findings.
flat_only = []
for sd in m.SEEDS:
    for arm, off in (("attention", 0.1), ("mingru", -0.1)):
        flat_only.append({"suite": "e1_mup_tuned", "arm": arm, "seed": sd, "tag": "",
                          "status": "done", "final_val": 4.0 + off,
                          "curve": [[int(20e6 * (i + 1) / 8), 4.0 + off]
                                    for i in range(8)]})
fc = m.crossing_tokens(flat_only)["cells"]["e1_mup_tuned"]
check("O14 a cell that ran but never crossed is PRESENT with zero crossings",
      fc["n_seeds"] == len(m.SEEDS) and fc["seeds_with_a_crossing"] == 0
      and "last" not in fc, fc)
check("O15 and the readout distinguishes that from a missing cell",
      "NEVER CROSSED" in src and "not measured:" in src)

# A parametrization uniformly worse than its own SP control cannot adjudicate an
# ordering: any ordering read off it is a property of the mis-specification.
def fin(suite, arm, v, seeds=m.SEEDS):
    return [{"suite": suite, "arm": arm, "seed": sd, "tag": "", "status": "done",
             "final_val": v, "curve": []} for sd in seeds]

sick = (fin("e1_sp_rerun", "attention", 4.75) + fin("e1_sp_rerun", "mingru", 4.94)
        + fin("e1_mup_tuned", "attention", 5.10) + fin("e1_mup_tuned", "mingru", 4.99))
h = m.parametrization_health(sick)["e1_mup_tuned"]
check("O16 a parametrization worse on BOTH arms is flagged unusable",
      h["worse_on_every_arm"] and not h["usable_as_comparator"], h)
check("O16b and the arm-dependent damage is quantified, not just noted",
      abs(h["damage_spread_between_arms"] - 0.30) < 1e-6,
      h["damage_spread_between_arms"])
well = (fin("e1_sp_rerun", "attention", 4.75) + fin("e1_sp_rerun", "mingru", 4.94)
        + fin("e1_perlayer_sp", "attention", 4.67) + fin("e1_perlayer_sp", "mingru", 4.87))
check("O17 one that matches or beats its control is not flagged",
      m.parametrization_health(well)["e1_perlayer_sp"]["usable_as_comparator"])
even = (fin("e1_sp_rerun", "attention", 4.75) + fin("e1_sp_rerun", "mingru", 4.94)
        + fin("e1_mup_tuned", "attention", 4.79) + fin("e1_mup_tuned", "mingru", 4.98))
check("O18 uniformly worse but EVENLY so is still usable -- it preserves the gap",
      m.parametrization_health(even)["e1_mup_tuned"]["usable_as_comparator"])

check("O11 the bands the readouts compare against are PAPER 4.2's, in code",
      m.SUITE24_LATE_BAND == (11.93e6, 12.58e6)
      and m.SUITE24_EARLY_BAND == (1.03e6, 1.09e6))

with tempfile.TemporaryDirectory() as td:
    old = sandbox(pathlib.Path(td))
    m.report_crossings(bs8_xing)
    out = json.loads(m.CROSSINGS.read_text())
    check("O12 crossings.json is generated from the ledger, and says so",
          "gpu_bundle.py" in out["generated_by"] and out["cells"] and out["readouts"])
    check("O13 and carries every per-seed crossing, not just the mean",
          all("per_seed" in c for c in out["cells"].values()))
    unsandbox(old)


print("\n=== T. DOC DRIFT: the documents must agree with the matrix ===")
# This repository has been bitten twice by a document and the code disagreeing --
# gap D3, and the SP-cells question where GPU_BUNDLE.md and ISSUES_AND_GAPS said
# opposite things and the runner followed the wrong one. Both grids here have since
# been extended mid-run, which moves every total in both documents.
check("T1 the shipped documents agree with the shipped matrix",
      m.check_docs(m.build_matrix(transfer=m.load_transfer(),
                                  anchor=m.load_anchor())) == 0)
check("T2 and the check actually fails when they do not",
      m.check_docs(m.build_matrix()[:5]) == 1)
check("T3 it names both documents", len(m.DOCS) == 2
      and all(d.exists() for d in m.DOCS), [str(d) for d in m.DOCS])


print("\n=== S. SP TARGET-WIDTH SWEEP: the other half of the tuning control ===")
# Comparing muP at its own optimum against SP at an inherited value would measure
# tuning quality and report it as parametrization -- the error this arm exists to
# remove, committed on the other side of the table. PAPER 8.1 also names the
# inherited global LR as the paper's largest uncontrolled factor, and 8.2 says seed
# agreement cannot bound it.
INHERITED = m.build_config(m.PRESET, {"run_name": "probe"}).matrix_lr
spb = [j for j in m.build_matrix() if j["suite"] == "e1_sp_basin"]
spr = [j for j in m.build_matrix() if j["suite"] == "e1_sp_rerun"]
check("S1 the SP sweep spans the inherited value on both sides",
      min(j["overrides"]["matrix_lr"] for j in spb) < INHERITED
      < max(j["overrides"]["matrix_lr"] for j in spb), INHERITED)
check("S2 it does not re-measure the inherited cell -- e1_sp_rerun already has it",
      INHERITED not in {j["overrides"]["matrix_lr"] for j in spb}
      and INHERITED == spr[0]["overrides"].get("matrix_lr", INHERITED))
check("S3 an SP sweep cell differs from e1_sp_rerun in matrix_lr alone",
      {f for f in spr[0]["overrides"] if spr[0]["overrides"][f]
       != spb[0]["overrides"].get(f)} - {"run_name", "seed"} <= {"matrix_lr"},
      {f for f in spr[0]["overrides"] if spr[0]["overrides"][f]
       != spb[0]["overrides"].get(f)} - {"run_name", "seed"})
check("S4 it needs no transfer -- SP has nothing to transfer",
      not any(j.get("blocked_on") for j in spb))
check("S5 it runs at its own declared seed count (5, not the muP basin's 3)",
      len({j["seed"] for j in spb}) == len(m.SP_BASIN_SEEDS) == 5)

# e1_sp_rerun's five seeds must land on the curve as the inherited point
def sprec(suite, arm, lr, vals, seeds):
    return [{"suite": suite, "arm": arm, "seed": sd, "tag": f"mlr{lr:g}",
             "status": "done", "final_val": v, "curve": [], "matrix_lr": lr}
            for sd, v in zip(seeds, vals)]

# generated to the grid's length, not typed to it -- this grid has been extended
# once already and a fixture with a fixed point count silently shortens the sweep
SP_LOW = len(m.SP_TARGET_LRS) // 2          # index of the intended minimum
SP_VALS = [4.70 + 0.05 * abs(i - SP_LOW) for i in range(len(m.SP_TARGET_LRS))]
SP_BEST = m.SP_TARGET_LRS[SP_LOW]
curve = []
for lr, v in zip(m.SP_TARGET_LRS, SP_VALS):
    if lr == INHERITED:
        curve += sprec("e1_sp_rerun", "attention", lr,
                       [v, v + .01, v + .02, v + .03, v + .04], m.SEEDS)
    else:
        curve += sprec("e1_sp_basin", "attention", lr,
                       [v + .01 * i for i in range(len(m.SP_BASIN_SEEDS))],
                       m.SP_BASIN_SEEDS)
pts = m.sp_basin_points(curve)["attention"]
check("S6 e1_sp_rerun's cells are folded into the curve at the inherited LR",
      len(pts[INHERITED]) == len(m.SEEDS), len(pts.get(INHERITED, {})))
check("S7 and every swept LR appears exactly once",
      len(pts) == len(m.SP_TARGET_LRS), sorted(pts))

sp = m.analyse_sp_basin(curve)["arms"]["attention"]
check("S8 the sweep locates SP's own optimum, bracketed and sign-consistent",
      sp["matrix_lr"] == SP_BEST and sp["bracketed"] and sp["sign_consistent"],
      f"{sp['matrix_lr']} != {SP_BEST}")
pen = sp["inherited_penalty"]
check("S9 and prices the inherited LR against it, paired, with an interval",
      pen["factor_off"] == round(INHERITED / SP_BEST, 2) and pen["mean"] > 0
      and pen["n"] == len(m.SP_BASIN_SEEDS) and pen["lo"] is not None, pen)

# A partial sweep makes the inherited point its own argmin. Reporting "1x its own
# optimum, costs +0.000000" reads like a completed check that found no penalty.
partial_sp = sprec("e1_sp_rerun", "attention", INHERITED, [4.7, 4.71, 4.72, 4.73, 4.74],
                   m.SEEDS)
psp = m.analyse_sp_basin(partial_sp)["arms"]["attention"]
# A flat basin defeats "which LR is best" without defeating "what does the
# inherited LR cost". minGRU's real 2026-08-27 curve is that case: 0.003125 and
# 0.00625 sit 0.0024 apart on a 4/5 sign test -- no seed count fixes a basin
# that flat -- yet both price the inherited 0.025 at ~+0.125. Refusing outright
# discarded a number robust to the entire ambiguity.
_INH_I = list(m.SP_TARGET_LRS).index(INHERITED)
TIE_LO, TIE_HI = m.SP_TARGET_LRS[_INH_I - 3], m.SP_TARGET_LRS[_INH_I - 2]
assert INHERITED not in (TIE_LO, TIE_HI)

def tie_curve(lo_vals, hi_vals, inh_vals):
    """argmin TIE_LO, tied with TIE_HI (TIE_LO loses one seed to it)."""
    rows = []
    for lr in m.SP_TARGET_LRS:
        if lr == INHERITED:
            rows += sprec("e1_sp_rerun", "mingru", lr, inh_vals, m.SEEDS)
        elif lr == TIE_LO:
            rows += sprec("e1_sp_basin", "mingru", lr, lo_vals, m.SP_BASIN_SEEDS)
        elif lr == TIE_HI:
            rows += sprec("e1_sp_basin", "mingru", lr, hi_vals, m.SP_BASIN_SEEDS)
        else:
            rows += sprec("e1_sp_basin", "mingru", lr, [5.40] * 5, m.SP_BASIN_SEEDS)
    return rows

FLAT_LO = [4.700, 4.703, 4.700, 4.701, 4.702]
FLAT_HI = [4.701, 4.700, 4.702, 4.702, 4.703]   # beats TIE_LO on seed 1 -> tied
ap = m.analyse_sp_basin(tie_curve(FLAT_LO, FLAT_HI,
                                  [4.830, 4.833, 4.830, 4.831, 4.832])
                        )["arms"]["mingru"]["inherited_penalty"]
check("S16 an unresolved argmin still prices the inherited LR when the tied "
      "optima agree",
      ap["mean"] is not None and ap.get("argmin_unresolved")
      and set(ap["candidates"]) == {TIE_LO, TIE_HI}, ap)
check("S17 and reports the SMALLEST of the tied prices, never the flattering one",
      abs(ap["mean"] - 0.1296) < 0.001 and ap["candidate_spread"] < 0.01, ap)

# Tied on the sign test but NOT on the price: a high-variance tie where the two
# candidates value the inherited LR 0.61 vs 0.50 nats apart. Which optimum you
# pick now changes the answer, so the unresolved argmin IS the answer.
dp = m.analyse_sp_basin(tie_curve([4.50, 4.95, 4.50, 4.50, 4.50], [4.70] * 5,
                                  [5.20] * 5))["arms"]["mingru"]["inherited_penalty"]
check("S18 but refuses when the tied optima disagree about the price",
      dp["mean"] is None and "not located" in dp["unavailable"], dp)

# A candidate the inherited value BEATS on some seed is not a price at all.
bp = m.analyse_sp_basin(tie_curve([4.50, 4.95, 4.50, 4.50, 4.50], [4.70] * 5,
                                  [4.83] * 5))["arms"]["mingru"]["inherited_penalty"]
check("S19 and refuses when a tied optimum does not beat the inherited value "
      "on every seed", bp["mean"] is None, bp)

check("S11 a one-point sweep does not price the inherited LR against itself",
      psp["inherited_penalty"]["mean"] is None
      and "not located" in psp["inherited_penalty"]["unavailable"],
      psp["inherited_penalty"])
edge_sp = list(partial_sp)
_top = [lr for lr in m.SP_TARGET_LRS if lr > INHERITED]
for lr, v in zip(_top, [4.7 - 0.1 * (i + 1) for i in range(len(_top))]):
    edge_sp += sprec("e1_sp_basin", "attention", lr, [v, v + .01, v + .02], m.BASIN_SEEDS)
esp = m.analyse_sp_basin(edge_sp)["arms"]["attention"]
check("S12 nor does an unbracketed one",
      esp["inherited_penalty"]["mean"] is None
      and "bracket" in esp["inherited_penalty"]["unavailable"], esp["inherited_penalty"])
check("S13 a bracketed, sign-consistent sweep still prices it",
      m.analyse_sp_basin(curve)["arms"]["attention"]["inherited_penalty"]["mean"] > 0)

# all three sweeps must run through ONE analyser, or the gates drift apart
for name, arms in (("proxy", m.analyse_proxy(interior)["arms"]),
                   ("sp_basin", m.analyse_sp_basin(curve)["arms"]),
                   ("mup_basin", m.analyse_basin(firm)["arms"])):
    check(f"S10.{name} carries both gates from the shared analyser",
          all({"bracketed", "sign_consistent", "beats_neighbours", "curve"} <= set(v)
              for v in arms.values()), name)


print("\n=== R. RESUME: a finished run is only finished FOR THE RECIPE THAT ASKED ===")
# The job id does not encode matrix_lr, so re-anchoring the transferred learning
# rate and re-running would otherwise skip every muP cell as "done" and publish
# numbers trained at the OLD rate under the new rate's label.
TR_A = {"arms": {a: {"matrix_lr": 0.0016, "bracketed": True} for a in m.ARMS}}
TR_B = {"arms": {a: {"matrix_lr": 0.0160, "bracketed": True} for a in m.ARMS}}
with tempfile.TemporaryDirectory() as td:
    old_s = sandbox(pathlib.Path(td))
    ft = FakeTrainer()
    rc = run_main(["--only", "e1_mup", "--skip-preflight", "--workers", "2"], ft,
                  transfer=TR_A)
    check("R1 the muP cells run under the first transfer", rc == 0 and len(ft.calls) == 10,
          f"rc={rc} calls={len(ft.calls)}")

    # same ids, different recipe: must NOT be reused
    ft2 = FakeTrainer()
    rc = run_main(["--only", "e1_mup", "--skip-preflight", "--workers", "2"], ft2,
                  transfer=TR_B)
    led = json.loads(m.ledger_path(m.OUT_ROOT).read_text())
    stale = [r for r in led["jobs"] if r["status"] == "stale"]
    check("R2 a re-anchored LR marks the old runs stale, it does not reuse them",
          len(stale) == 10 and not ft2.calls,
          f"{len(stale)} stale, {len(ft2.calls)} re-run")
    check("R3 and the reason names the field that changed",
          all("matrix_lr" in (r.get("failure_reason") or "") for r in stale),
          stale[0].get("failure_reason") if stale else None)
    check("R4 a stale run never counts as done",
          all(r["status"] != "done" for r in stale))

    # preflight must refuse rather than let the launch discover it
    check("R5 preflight fails closed on a stale run dir",
          m.preflight([j for j in m.build_matrix(transfer=TR_B)
                       if j["suite"] == "e1_mup"], allow_repeat=True) == 1)

    # --reset-partial archives them, and then the re-run happens for real
    m.reset_partial([j for j in m.build_matrix(transfer=TR_B)
                     if j["suite"] == "e1_mup"], m.OUT_ROOT)
    ft3 = FakeTrainer()
    rc = run_main(["--only", "e1_mup", "--skip-preflight", "--workers", "2"], ft3,
                  transfer=TR_B)
    check("R6 --reset-partial archives them so the re-run actually runs",
          rc == 0 and len(ft3.calls) == 10, f"rc={rc} calls={len(ft3.calls)}")

    # and an UNCHANGED recipe still resumes, or resume is useless
    ft4 = FakeTrainer()
    rc = run_main(["--only", "e1_mup", "--skip-preflight", "--workers", "2"], ft4,
                  transfer=TR_B)
    check("R7 an unchanged recipe still resumes rather than re-running",
          rc == 0 and not ft4.calls, f"{len(ft4.calls)} re-run")
    unsandbox(old_s)


print("\n=== Q. RECIPE IDENTITY: each recipe must BE the published suite it names ===")
# The readouts compare a muP cell here against an SP crossing published in section 4.
# That only means anything if the recipe is the same recipe, so it is checked against
# the committed config.json of the actual runs rather than against this file's belief
# about them. Two fields are excused: per_layer_sp and embed_lr_mult did not exist
# when those suites ran, and their defaults (False, 1.0) are what "absent" meant.
PUBLISHED = [
    ("e1_sp_rerun", "nanolab/out/crossover20m_matched_lr", "suite 24 (s24)"),
    ("e1_sp_sched20", "nanolab/out/crossover20m_locked", "suite 23 (s23)"),
    ("e1_sp_bs8", "nanolab/out/crossover8m_bs8", "suite 25 (s25)"),
]
RECIPE_ID_FIELDS = ["mixer", "n_layer", "d_model", "n_head", "head_dim", "batch_size",
                    "grad_accum", "block_size", "max_steps", "warmup_steps",
                    "eval_interval", "eval_iters", "optimizer", "schedule", "lr",
                    "matrix_lr", "mup", "compile", "hf_dataset"]
NEW_FIELD_DEFAULTS = {"per_layer_sp": False, "embed_lr_mult": 1.0}

for suite, out_dir, label in PUBLISHED:
    cands = sorted(pathlib.Path(out_dir).glob("*attention*s1337")) if \
        pathlib.Path(out_dir).exists() else []
    cfgp = (cands[0] / "config.json") if cands else None
    if not cfgp or not cfgp.exists():
        check(f"Q.{suite} published config is available to compare against",
              False, f"no run dir under {out_dir}")
        continue
    pub = json.loads(cfgp.read_text())
    job = [j for j in m.build_matrix() if j["suite"] == suite
           and j["arm"] == "attention" and j["seed"] == 1337][0]
    mine = m.build_config(m.PRESET, job["overrides"]).to_dict()
    diff = {f: (pub.get(f), mine.get(f)) for f in RECIPE_ID_FIELDS
            if pub.get(f) != mine.get(f)}
    check(f"Q.{suite} reproduces {label} field for field", not diff, diff)
    # the cosine horizon, which is the whole difference between s23 and s24
    pub_total = pub.get("lr_max_steps") or pub.get("max_steps")
    mine_total = mine.get("lr_max_steps") or mine.get("max_steps")
    check(f"Q.{suite} matches its cosine horizon "
          "(lr_max_steps=0 means max_steps, schedules.py)",
          pub_total == mine_total, f"published {pub_total} vs bundle {mine_total}")
    check(f"Q.{suite} the two fields added since are at their absent-value defaults",
          all(mine.get(f) == d for f, d in NEW_FIELD_DEFAULTS.items())
          and all(pub.get(f) in (None, d) for f, d in NEW_FIELD_DEFAULTS.items()))

# and the three must be genuinely different recipes, or the readouts compare nothing
tot = {}
for suite, _, _ in PUBLISHED:
    o = [j for j in m.build_matrix() if j["suite"] == suite][0]["overrides"]
    tot[suite] = (o["batch_size"], o["max_steps"], o["lr_max_steps"])
check("Q4 s24, s23 and s25 are three distinct recipes", len(set(tot.values())) == 3, tot)
check("Q5 s23 and s24 differ ONLY in the cosine horizon",
      tot["e1_sp_sched20"][:2] == tot["e1_sp_rerun"][:2]
      and tot["e1_sp_sched20"][2] != tot["e1_sp_rerun"][2], tot)


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
