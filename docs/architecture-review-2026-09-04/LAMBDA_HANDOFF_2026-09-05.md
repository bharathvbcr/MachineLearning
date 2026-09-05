# Lambda GH200 handoff: the September 2026 experiment program

Written 2026-09-05 06:20 UTC for a fresh agent with no memory of the discussion
that produced it. Everything marked **verified** was read from the repo, the run
records, or the box itself at that time; **estimate** means arithmetic on measured
rates; **decision** means Bharath has not decided yet. Read this whole file before
touching the box. It bills $2.29 per hour whether or not the GPU is busy.

---

## 1. What this program is, in one screen

The repo's paper (`PAPER_2026-08_Recipe_Dependent_Rankings.md`) shows that
attention-vs-recurrent rankings at 124M parameters and 50M tokens move with the
training recipe. Two reviews on 2026-09-04 (`docs/architecture-review-2026-09-04/`)
re-read all of that evidence and found:

1. **A hybrid that beats both pure arms along the whole token axis.** `hybrid_mingru8_attn4`
   (8 minGRU layers, then 4 attention) beats `attention` and `mingru` at every saved
   evaluation from 1.65M to 50M tokens, paired by seed, 5/5 (`README.md` in that folder,
   "What the hybrid curves establish"). Caveat found the same day: it has **19% more
   parameters** than attention because nanolab's minGRU uses expansion 2. The parity test
   has not been run.
2. **Two code defects that change what existing boards mean** (`inputs/04-…md` §9.1;
   probes in `evidence/`):
   - `nanolab/mixers.py` implements the gated delta rule with the correction computed from
     the **undecayed** state, `S <- aS + b(v - Sk)k^T`. The published Gated DeltaNet
     (arXiv 2412.06464, eq. 8) uses the decayed state, `S <- aS + b(v - aSk)k^T`. Every GDN
     number in the repo is about the repo's variant.
   - `nanolab/model.py` `MoE`: renormalising the top-k weights makes the top-1 weight
     exactly 1, so the router gets **no task gradient** (norm 3e-18); it trains on the
     balancing loss alone. Every `moe_e4k1` / `moe_e8k1` run measured a balance-only router.
3. **A deployment axis nobody had priced** (`inputs/04-…md` §5): on the GH200 in
   PyTorch every linear mixer trains slower than attention at ctx 512 (8+4 at 0.85x; GDN
   at 0.11x at ctx 2048); on Apple silicon the order flips; decode at this size is
   dispatch-bound. Those items run on the M5 Pro, not here.

The experiments below are items from `inputs/04-efficiency-reconciliation-memo.md` §7
(numbers kept so results can be filed against them). The order was chosen so that the
two defect ablations run **before** anything that depends on GDN or MoE numbers.

---

## 2. The box, verified 2026-09-05 06:15 UTC

| fact | value |
|---|---|
| ssh alias | `lambda-gpu` in `~/.ssh/config` (HostName 192.222.59.136, User ubuntu, key set there). The old address 192.222.51.171 in `scripts/e22_queue.sh` is stale. If Lambda recycles the instance, repoint the alias; do **not** accept a changed host key just to make a script work. |
| GPU | NVIDIA GH200 480GB, 97,871 MiB. Python 3.10.12, torch 2.7.0, CUDA 12.8, aarch64. |
| repo on the box | `~/MLSystemsLab` is **not a git checkout**. Code arrives by rsync (`scripts/swaboard_launch.sh` step 1 is the pattern). `git` commands there fail. `rg` is not installed; use `grep`. |
| PATH | scripts export `PATH=$HOME/.local/bin:$PATH` (pip-installed `ninja`, `pybind11`). |
| corpus | `nanolab/data/HuggingFaceFW_fineweb-edu/train.bin`, must be exactly 497,500,000 tokens (`swaboard_launch.sh` step 2 asserts it). It is part of the recipe: the batcher samples with replacement. Never re-tokenize; rsync the reference copy. |
| disk | 3.9T, 849G used. Checkpoints (`*.pt`) are never pulled and never committed. |
| GPU state | idle: 0% util, 138 MiB used, no compute processes, no tmux server. |
| code drift | `nanolab/crossover_replicate.py` differs between laptop and box. The laptop has an **uncommitted 59-line change** (a VRAM guard that refuses over-subscribed launches, plus `launch` now exits 1 when any job failed). Everything else in `nanolab/` matches by md5. |

### 2.1 What is on the box and not yet in git

Compare with `ls nanolab/out` locally; pull with `bash scripts/pull_artifacts.sh <suite>`
from the laptop (it uses the `lambda-gpu` alias; ~16 MB of `metrics.jsonl` + `config.json`,
never `.pt`). As of 06:15 UTC local and box agree on `crossover50m_moe32c` (20/20 done)
and `gpu_bundle` (476 run dirs each). The two `…1536` suites exist locally but are
behind the box; pull them first.

### 2.2 The board that is currently broken: E27, width 1536 (verified)

`scripts/e27_ladder_w1536.sh` ran overnight. Its state:

- **Probe** (`nanolab/out/crossover_ladder_probe1536`, 10M tokens, seed 1337, 2 workers):
  4 done, **2 failed** (`w1536_mingru_lr40`, `w1536_mingru_lr80`: CUDA OOM, two minGRU jobs
  at 45.9 + 48.4 GiB co-resident). The script then picked the minGRU argmin from the
  survivors: `w1536_mingru_lr20`, an **edge** value from an incomplete sweep (the script
  itself printed `EDGE:w1536_mingru_lr20`). Attention's pick, `w1536_attention_lr80`, is a
  real interior argmin (lr40 5.1800, lr80 5.1584, lr160 5.2317).
- **Stage 2** (`nanolab/out/crossover_ladder1536`, 50M, 2 workers): 2 `attention_lr80` jobs
  were running when every worker stopped at **06:05:29 UTC** with no error in the logs
  (last train records at step 1525/1535 of ~3050; `ckpt.pt` written 06:05). 8 jobs
  pending, 5 of them the edge-picked minGRU arm. Nothing launched them under tmux, so the
  likely cause is the launching shell going away; treat it as a hangup, not a crash.
- The queue runner only claims jobs whose status is `pending` (`claim_job`); the two
  orphaned jobs are stuck at `running` and will **not** be picked up by a relaunch until
  their status is reset. A worker that finds `ckpt.pt` sets `RESUME=1` and continues.

Repair, in this order (step 0 below runs it): reset the two orphans to `pending`; hold the
five `w1536_mingru_lr20` jobs; relaunch the attention half under tmux at `--workers 2`;
rerun the two OOM'd probe cells at `--workers 1` in a **new** probe dir (the old dir's
`recipe.json` records `workers: 2`, and `lock_recipe` refuses a different tenancy in the
same dir); pick minGRU's argmin from all four cells; run that arm's five seeds at
`--workers 1` in its own dir. Tenancy only changes throughput, never the loss curve, so
the attention/minGRU comparison across those two dirs is still valid; say so in the writeup.

---

## 3. Rules that every board in this repo obeys

These were each earned by a specific failure recorded in `scripts/*.sh` comments and
`docs/ISSUES_AND_GAPS_2026-08-22.md`. Break one and the result is unpublishable.

1. **One recipe per output directory.** `lock_recipe` compares batch, block, eval_iters,
   token budget, LR horizon, tenancy (`workers`), device, `budget_by_arm`, and each arm's
   overrides. Only the arm list may **grow**. A changed override (window, width, LR, rule)
   must be a new arm name, and a changed recipe a new directory.
2. **Tenancy is a recipe field.** `--workers N` is jobs per GPU. Throughput measured at one
   tenancy does not transfer. Wall-clock boards (`budget_by_arm`) must run at the tenancy
   their budgets were sized at (E20: 3).
3. **New experiment identity for changed code.** The GDN and MoE fixes go behind config
   flags with new arm names and new output dirs. Never edit an existing run's identity or
   pool pre-fix and post-fix runs.
4. **Chain on log markers, never `pgrep`.** Every stage script ends with
   `echo "eNN exit=$rc $(date -u +%FT%TZ)"`; the next stage waits on that line.
   `pgrep -f` matches the tmux/ssh command line that carries the pattern and once idled
   the GPU for 79 minutes.
5. **Launch under tmux.** `tmux new-session -d -s <name> 'cd ~/MLSystemsLab && bash scripts/<stage>.sh 2>&1 | tee nanolab/out/<stage>.log'`.
   Attach with `tmux attach -t <name>`. Then `exit` the ssh session; never leave a board as a
   child of an ssh shell (see §2.2).
6. **Pull artifacts during the board, not after** (`scripts/pull_artifacts.sh`; the E21
   phase-2 records were lost once). Commit them: `git add nanolab/out/<suite> && git commit -m 'artifacts: <suite>'`.
7. **`final_val`, never `best_val`.** `best_val` is a minimum over noisy evaluations.
   Boards report the last evaluation; `wcboard` refuses to substitute.
8. **Smoke every new arm before the board**: `python3 -m nanolab.crossover_replicate smoke --arms <list>`
   (40 steps, isolated subtree). Run `python3 -m nanolab.tests` on the laptop after any code
   change (CPU, under a minute; SKIP is not PASS).
9. **`compile=False` on the GH200** (Inductor stall on aarch64); the runner forces it.
10. **Five seeds** (1337, 42, 100, 2026, 777), batch 32, ctx 512, `eval_iters` 20, 50M
    cosine, Muon lr 6e-4 / matrix_lr 0.025, unless the board is explicitly about one of those.
11. **Report paired by seed**, with the sign count and a Student-t interval (4 dof); five
    of five is p = 0.0625 on a sign test and is not the claim. Label every number
    verified / inferred / estimate. Recall boards report solved-seed **rates** with Wilson
    intervals, never mean recall.
12. **No new dependencies without asking.** In particular do not install
    `flash-linear-attention` to speed up GDN; that is a numerics change and Bharath's call.
13. **Never commit `.env*`, keys, or `.pt` files.** `.gitignore` publishes only
    `metrics.jsonl`, `config.json`, `queue.json`, `recipe.json`, logs.

---

## 4. Step 0: triage, sync, and code changes (laptop first)

### 4.0 Sync what exists

```bash
bash scripts/pull_artifacts.sh crossover_ladder_probe1536
bash scripts/pull_artifacts.sh crossover_ladder1536
git add nanolab/out/crossover_ladder_probe1536 nanolab/out/crossover_ladder1536
git commit -m 'artifacts: E27 width-1536 probe (2 OOM cells) and the interrupted stage 2'
```

### 4.1 Repair E27 on the box

```bash
ssh lambda-gpu
cd ~/MLSystemsLab && export PATH=$HOME/.local/bin:$PATH
python3 - <<'EOF'
from pathlib import Path
from nanolab.crossover_replicate import _lock_load, _lock_save
q = Path("nanolab/out/crossover_ladder1536/queue.json")
fh, state = _lock_load(q)
for j in state["jobs"]:
    if j["status"] == "running":            # orphaned 06:05Z; ckpt.pt resumes them
        j["status"] = "pending"
    if j["arm"] == "w1536_mingru_lr20":     # picked from an incomplete probe
        j["status"] = "held"
_lock_save(fh, state)
print({j["id"]: j["status"] for j in state["jobs"]})
EOF
tmux new-session -d -s e27b 'cd ~/MLSystemsLab && export PATH=$HOME/.local/bin:$PATH && \
  CROSSOVER_ARMS=w1536_attention_lr80,w1536_mingru_lr20 CROSSOVER_JOB_PREFIX=cx32lad1536 \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=50000000 \
  python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_ladder1536 --workers 2 \
  2>&1 | tee -a nanolab/out/e27.log; echo "e27b exit=$? $(date -u +%FT%TZ)" | tee -a nanolab/out/e27.log'
```

Then, in a second tmux session, the two missing minGRU probe cells at tenancy 1 (new dir,
same recipe otherwise; cost: two 10M jobs, about 8 minutes each):

```bash
tmux new-session -d -s e27p 'cd ~/MLSystemsLab && export PATH=$HOME/.local/bin:$PATH && \
  CROSSOVER_ARMS=w1536_mingru_lr40,w1536_mingru_lr80 CROSSOVER_JOB_PREFIX=cx32lad1536pb \
  CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=10000000 \
  python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover_ladder_probe1536b --workers 1 --seed 1337 \
  2>&1 | tee nanolab/out/e27p.log; echo "e27p exit=$? $(date -u +%FT%TZ)" | tee -a nanolab/out/e27p.log'
```

Do not run the probe and the stage-2 pair at the same time if the VRAM would exceed the
device: one w1536 minGRU job is ~48 GiB and one w1536 attention job is smaller; check
`nvidia-smi` before starting the second session and wait if needed. When the probe is in,
pick minGRU's argmin on `final_val` over lr20 (4.9556 in the first probe; `status` shows 4.9815 because it reads the 9.85M-token marker, not the terminal evaluation) / lr40 / lr80. If the argmin
is still lr20, unhold the five held jobs (`launch … --unhold`) at `--workers 1` **in a new
dir** `crossover_ladder1536_mingru` (prefix `cx32lad1536m`), because 2 minGRU jobs OOM
together. Readout is in `scripts/e27_ladder_w1536.sh` (margin vs prior widths: 0.147 /
0.159 / 0.152).

### 4.2 The uncommitted runner change

`git diff nanolab/crossover_replicate.py` on the laptop is Bharath's VRAM guard (`check_plan`
from `nanolab/vram.py`) and failure propagation in `launch`. It exists because of exactly
the E27 failure above. Run `python3 -m nanolab.tests`; if green, ask Bharath to commit it
(or commit it with his OK), then rsync. Until it is on the box, `launch` returns 0 with
failed jobs in the queue.

### 4.3 The code the boards need: implemented 2026-09-05 (uncommitted at the time of writing)

Everything in this subsection now exists in the checkout and `python3 -m nanolab.tests`
passes 172/172 (3 SKIP for no CUDA). Verify it is committed and on the box before
launching; the descriptions below say what each piece is so a reviewer can check it.
Each flag is a new `Config` field with the old behaviour as default, so every existing
run and its `config.json` stay reproducible, and the recipe fingerprint
(`arm_overrides`) records which variant a job used.

**(a) `gdn_rule: str = "repo"`** (`"repo"` | `"published"`), `nanolab/config.py`.
- `GatedDeltaNet.__init__` reads it; `forward` passes it to `gdn_chunked(..., rule=...)`;
  `_sequential` honours it too.
- In `gdn_chunked`, the published rule changes two lines and nothing else. Derivation:
  with `u_t = b_t (v_t - a_t S_{t-1} k_t)` and `A_t = prod_{i<=t} a_i`, unrolling the chunk
  gives `S_{t-1} k_t = A_{t-1} S_in k_t + sum_{j<t} (A_{t-1}/A_j) u_j (k_j . k_t)`, so
  `u_t = b_t v_t - b_t A_t (S_in k_t) - b_t sum_{j<t} (A_t/A_j)(k_t . k_j) u_j`.
  Therefore: `logM` uses `cum` (A_t) instead of `cum_prev` (A_{t-1}), and `R` uses `A_t`
  instead of `A_prev`. `Wy`, `dS`, `A_C` are unchanged. Treat this as the expected diff and
  let the parity test confirm it.
- In `_sequential`: `delta = (vt - at * pred) * bt` for the published rule (`at` is the
  per-step alpha, broadcast to `pred`'s shape).
- Tests (`nanolab/tests.py`): extend `gdn_chunked_matches_sequential` to loop over both
  rules; add a two-token probe: unit `q=k=v`, zero state, `alpha=beta=0.5` gives
  outputs `[0.5, 0.5]` under `repo` and `[0.5, 0.625]` under `published`; at `alpha=1` or
  `beta=0` both rules agree. `docs/architecture-review-2026-09-04/evidence/gdn_rule_probe.py`
  is the reference implementation of that check.
- Arms (`nanolab/crossover_replicate.py`, appended to `ARMS`):
  `gdn_pub` (mixer `gdn`, overrides `(("gdn_rule","published"),)`) and
  `hybrid_gdn_periodic_pub` (`gdn*3,attention,gdn*3,attention,gdn*3,attention`, same
  override). `nanolab/mqar_suite.py` resolves arms from the same `ARMS`, so the recall grid
  sees them without further wiring.
- Do **not** change the default. Which rule the repo should carry is Bharath's decision
  (§8); the docstring of `GatedDeltaNet`, `MASTER_ARCHITECTURAL_KB.md` and
  `learning-notes/21-…` all state the repo variant and should say so explicitly.

**(b) `moe_router_weight: str = "renorm"`** (`"renorm"` | `"raw"`), `nanolab/config.py`.
- `MoE.forward` (`nanolab/model.py`, the line `weights = weights / weights.sum(-1, keepdim=True)`):
  apply only under `renorm`. Under `raw` the selected expert's output is scaled by its raw
  softmax probability, which is the Switch Transformer formulation and is what gives the
  router a task gradient at top-1.
- Tests: keep `moe_router_gets_gradient` (top-2). Add `moe_top1_router_task_gradient`:
  with `moe_top_k=1`, gate task-gradient norm is `< 1e-12` under `renorm` (documents the
  defect) and `> 0` under `raw`; the balancing-loss gradient is nonzero under both.
  `evidence/router_gradient_probe.py` is the reference check.
- Arms: `moe_e1k1_raw`, `moe_e4k1_raw`, `moe_e8k1_raw` (copy the existing three, add the
  override). The one-expert control is a dense SwiGLU either way (softmax over one expert
  is exactly 1), so it must land inside `attention`'s interval; if it does not, the board
  is unreadable, exactly as E19b's script says.

**(c) `mingru_expand: int = 2`**, `nanolab/config.py`; `MinGRU.__init__` uses it instead
of the hard-coded `self.expand = 2`.
- Arms: `hybrid_mingru8_attn4_x1`, `hybrid_mingru_periodic_x1`, `mingru_x1` with
  `(("mingru_expand", 1),)`. Measured at the board shape (verified, CPU build):

  | arm | parameters | vs attention |
  |---|---:|---:|
  | attention | 123,699,612 | 0 |
  | hybrid_mingru8_attn4 (x2) | 147,217,732 | +19.0% |
  | hybrid_mingru8_attn4_x1 | 128,343,364 | +3.8% |
  | hybrid_mingru_periodic (x2) | 150,157,497 | +21.4% |
  | hybrid_mingru_periodic_x1 | 128,923,833 | +4.2% |
  | mingru_x1 | 130,665,240 | +5.6% |
  | attention_untied | 162,333,084 | +31.2% |
  | attn6_w512 / attn6_w576 | 45,055,286 / 52,902,396 | |

  Expansion 1 is **near** parity, not parity: a minGRU layer keeps its two value-residual
  projections (2·d·kv), so 8+4 lands 3.8% above attention. Exact parity would mean removing
  those projections, which is a different arm. Write "+3.8%" in the E30 readout.

**(d) Copy-loss probe** (for item 2; can wait until the boards above are queued).
`nanolab/train.py`'s eval site (`log.eval(step, val_loss=val, ...)`) gains, behind a new
`Config.copy_probe: bool = False`, a `copy_loss` field: a fixed-seed batch of sequences of
length `block_size` filled with uniform random tokens, containing one random span of 32
tokens that is repeated later in the sequence; loss and accuracy on the second occurrence
only. Same probe batches for every arm and every evaluation. Add `copy_probe` to
`current_recipe()` so `lock_recipe` records it. Cost: four extra forward batches per eval.

After the edits: `python3 -m nanolab.tests`, commit, then

```bash
rsync -az -e ssh --exclude '__pycache__' --exclude 'out' --exclude 'data' nanolab scripts docs experiment-notes paper lambda-gpu:MLSystemsLab/
ssh lambda-gpu 'cd MLSystemsLab && export PATH=$HOME/.local/bin:$PATH && python3 -m nanolab.tests && python3 -m nanolab.crossover_replicate smoke --arms gdn_pub,hybrid_gdn_periodic_pub,moe_e1k1_raw,moe_e4k1_raw,hybrid_mingru8_attn4_x1,attention_untied,attention_novr,attention_untied_novr,attn6_w512,attn6_w576'
```

(the `attention_*` and `attn6_*` arms are registered in §5). The repo's `smoke` at the
board shape is far too slow on a CPU (16 tok/s); a one-step CPU pre-flight of every new arm
through the real trainer was run instead on 2026-09-05, and the box smoke above is still
required because it exercises the CUDA paths.

**Also written on 2026-09-05:** `scripts/paired_board.py` (paired-by-seed gaps at shared
markers and `final_val`, 95/99/99.9% intervals, sign counts, mean-curve crossings; refuses a
cross-recipe pairing) and the stage scripts `scripts/e28_gdn_rule.sh` … `scripts/e35_w384_ladder.sh`,
which source `scripts/_stage_common.sh` (shared 50M recipe env, optional log-marker chaining
via `PREV=eNN`). Each stage script carries its pre-registered readout in its header comment
and prints the readout at the end. The board tuples `GDN_RULE_ARMS`, `MOE_RAW_ARMS`,
`PARITY_ARMS`, `TIE_ARMS`, `SHAPE_ARMS`, `COPY_ARMS`, `W384X_ARMS` in
`nanolab/crossover_replicate.py` name each board's full arm list. E34 needs
`CROSSOVER_COPY_PROBE=1`, which the script sets and the recipe records.

---

## 5. The boards, in order

The stage scripts exist (`scripts/e28_gdn_rule.sh` … `scripts/e35_w384_ladder.sh`); launch each
under tmux, optionally chained with `PREV=e28 bash scripts/e29_moe_raw.sh`. E27 is the ladder. Per-job GPU-minutes below are measured
elapsed at the stated tenancy divided by the tenancy (`docs/architecture-review-2026-09-04/evidence/evidence-tables.md`
has the elapsed column); dollars are at $2.29/h and are estimates.

| # | board (memo item) | arms | new dir / prefix | tenancy | jobs | GPU-h est. | $ est. |
|---|---|---|---|---|---|---|---|
| E28 | GDN rule ablation on recall (14) | `gdn`, `gdn_pub`, `hybrid_gdn_periodic`, `hybrid_gdn_periodic_pub` | `nanolab/out/mqar_e8` (same ledger; run names carry pairs/batch/steps/arm/seed) | 4 workers | 4 cells × 2 new arms × 15 seeds = 120 runs (+ existing arms are skipped off the ledger) | ~2 | ~5 |
| E28b | same, the hard cell | `gdn_pub` at 64 pairs / seq 255, batch 64 (E16's calibrated cell) | `nanolab/out/mqar_e16_board` | 4 workers | 15 | ~0.5 | ~1 |
| E29 | MoE with a task-trained router (15) | `attention`, `moe_e1k1_raw`, `moe_e4k1_raw`, `moe_e8k1_raw` | `crossover50m_moe32d` / `cx32moed` | 2 | 20 | ~3.5 | ~8 |
| E30 | closers on the hybrid result (1, 9) | into `crossover50m_ratioplace32`: `attention`; new dir for `hybrid_mingru8_attn4_x1`, `hybrid_mingru_periodic_x1`, `attention` | ratioplace32 grows by one arm (`cx32p`); `crossover50m_parity32` / `cx32par` | 3 | 5 + 15 | ~3.5 | ~8 |
| E31 | tied/untied × value residual (3) | `attention`, `attention_untied`, `attention_novr`, `attention_untied_novr` | `crossover50m_tie32` / `cx32tie` | 3 | 20 | ~3 | ~7 |
| E32 | 8+4 on the recall grid (4) | `hybrid_mingru8_attn4` at p=4,8 × 3k,9k | `nanolab/out/mqar_e8` | 4 workers | 60 | ~1 | ~2.5 |
| E33 | depth for width at ~21M non-embedding (11) | `attention`, `attn6_w512`, `attn6_w576`, `w384_attention_lr10`; read `attn3` from `crossover50m_loop32` (same recipe and tenancy) | `crossover50m_shape32` / `cx32shape` | 3 | 20 | ~2.5 | ~6 |
| E34 | copy-loss probe (2) | `attention`, `mingru`, `hybrid_mingru8_attn4` with `copy_probe` | `crossover50m_copy32` / `cx32copy` | 3 | 15 | ~2.5 | ~6 |
| E35 | token ladder at width 384 (6), rung 1 | `w384_attention_lr80`, `w384_mingru_lr40`, plus the 8+4 hybrid at w384 at lr40 and lr80 (register `w384_hybrid_mingru8_attn4_lr40/lr80` with the ladder's overrides) | `crossover200m_w384` / `cx32w384x4`, `CROSSOVER_TOKEN_BUDGET=200000000` | 2 | 20 | ~6.5 | ~15 |
| E35b | rung 2, 800M tokens | same arms | `crossover800m_w384`, budget 800000000 | 2 | 20 | ~25 | ~57 |

Total for E28–E34: about 19 GPU-hours, about $45. E35 rung 2 revisits the corpus (800M
over 497.5M tokens = 1.6 epochs); record `data_epochs` in the writeup and treat it as a
different distribution from rung 1, which is 0.4 epochs.

Arms to register for E31/E33 (all `mixer="attention"`):
`attention_untied` `(("tie_embeddings", False),)`; `attention_novr` `(("value_residual", False),)`;
`attention_untied_novr` both; `attn6_w512` `(("n_layer", 6), ("d_model", 512), ("n_head", 8), ("head_dim", 64))`;
`attn6_w576` `(("n_layer", 6), ("d_model", 576), ("n_head", 9), ("head_dim", 64))`.
`w384_attention_lr10` already exists (ladder arm at LR multiplier 1.0 = the shared base
LR). Non-embedding parameters: attn3 21.3M, 12L×384 21.2M, 6L×512 18.9M, 6L×576 23.9M
(head_dim is pinned at 64, so 544 is not reachable; the two shapes bracket 21.3M). Untied
embeddings add 38,633,472 parameters at width 768; report both counts.

### 5.1 Exact launch lines

E28 (recall, GDN rule). Run the four cells as four invocations so each cell's batch is its
own (E8's calibration: batch 256, LR rule `sqrt`):

```bash
for STEPS in 3000 9000; do for PAIRS in 4 8; do
python3 -u -m nanolab.mqar_suite --out nanolab/out/mqar_e8 --device cuda \
  --arms gdn,gdn_pub,hybrid_gdn_periodic,hybrid_gdn_periodic_pub \
  --pairs $PAIRS --steps $STEPS --batch 256 --seeds 15 --lr-rule sqrt --workers 4 --gpus 1
done; done
python3 -u -m nanolab.mqar_suite --out nanolab/out/mqar_e16_board --device cuda \
  --cells 64 --batch 64 --arms attention,gdn,gdn_pub --seeds 15 --steps 3000 --lr-rule sqrt --workers 4 --gpus 1
```

Existing arms on the ledger are skipped; they are listed so the report prints the
comparison. `--report-only` re-prints a cell.

E29 (MoE): copy `scripts/e19b_moe_rerun.sh`, change `CROSSOVER_ARMS=attention,moe_e1k1_raw,moe_e4k1_raw,moe_e8k1_raw`,
`CROSSOVER_JOB_PREFIX=cx32moed`, `--out nanolab/out/crossover50m_moe32d --workers 2`. Keep
its control check verbatim: the board is readable only if `moe_e1k1_raw` overlaps `attention`.

E30a (attention into ratioplace32; the recipe's arm list grows, tenancy must stay 3):

```bash
export CROSSOVER_ARMS=hybrid_mingru11_attn1,hybrid_mingru_periodic,hybrid_mingru_bookend,hybrid_mingru8_attn4,hybrid_mingru10_attn2,attention
export CROSSOVER_JOB_PREFIX=cx32p CROSSOVER_BATCH=32 CROSSOVER_EVAL_ITERS=20 CROSSOVER_TOKEN_BUDGET=50000000
python3 -u -m nanolab.crossover_replicate launch --out nanolab/out/crossover50m_ratioplace32 --workers 3
```

E30b, E31, E33, E34: same shape as `scripts/e18_loop.sh` with the arms, prefix and dir
from the table, `--workers 3`. E35: `--workers 2` and the token budget in the env.

### 5.2 Pre-registered readouts (write these into the stage script before launching)

- **E28.** If `gdn_pub` solves at least 4/15 more seeds than `gdn` at p=8/9k, or any seed at
  seq 255, the variant was the defect and every downstream GDN claim is re-based on the
  published rule. If not, the hard-recall failure is capacity or optimisation, and the
  fast-weight line proceeds on the published rule anyway (it is the operator the field
  compares against). Either outcome is a result; file it in the review folder's
  corrections table.
- **E29.** Control first. Then `moe_e8k1_raw` minus `attention`, paired: at or below
  +0.016 nats means parameters are still not the bottleneck at 0.4 tokens/param; better
  than attention by a disjoint interval means E19's conclusion was a router artifact.
- **E30.** `attention` inside ratioplace32 within ±0.005 of suite 22's attention per seed
  at 50M makes §2 of the memo within-suite. The x1 hybrids (+3.8% parameters, not parity):
  paired 5/5 against attention at 50M **and** no re-crossing on the mean curves keeps the
  no-regret claim at near parity; otherwise the margin was parameters.
- **E31.** VR gain (tied) minus VR gain (untied): if untying shrinks the VR gain by more than
  half, the two interact; unchanged means independent. Report the parameter delta beside it.
- **E32.** Rates within attention's Wilson interval at every cell: no-regret on both metrics.
- **E33.** 6L within 0.05 of 12L at 50M makes 6 layers the latency shape; 3L within 0.05
  makes it 3L. Shared LR is a known limitation (E21 measured different optima per shape);
  say so rather than tuning.
- **E34.** On each seed, the token at which `copy_loss` drops sharply versus the token at
  which the attention curve overtakes minGRU's (from `crossover50m`, per seed, `verify_curves.py`
  in the review folder shows how to read paired curves). Within ±1M on 4/5 seeds is
  predictive; otherwise the induction-head explanation is withdrawn.
- **E35.** Whether the 50M crossing token scales with budget, and the ordering at 5 and 20
  tokens per parameter.

### 5.3 Analysis

`python3 -m nanolab.crossover_replicate table --out <dir>` prints the token-grid table;
`wcboard` is for wall-clock suites only. `python3 scripts/paired_board.py ARM@SUITE … --ref ARM@SUITE`
is the paired reading (validated 2026-09-05: it reproduces the 8+4 result, −0.0176
[−0.0222, −0.0130] at the last marker, 5/5, crossing at 0.94M only). It refuses pairs whose
recipes differ. Report 95% and 99.9% intervals when a result was chosen after scanning many arms and
markers. `paper/derive_figures.py --check` must stay green if any manuscript figure is
touched.

### 5.4 What to write back

For each board: a dated doc under `docs/` in the style of `docs/LADDER_BOARD_2026-09-04.md`
(question, recipe, table with intervals, readout against the pre-registration, what is
still unverified), an entry in `experiment-notes/00-INDEX.md`, a line in
`docs/EXPERIMENT_BACKLOG_2026-08-26.md`, and, for E28/E29, a new row in the corrections
table of `docs/architecture-review-2026-09-04/README.md`. Commit artifacts per suite
(`artifacts: <suite>`), then the doc. Commit messages in this repo state the finding, not
the action (see `git log`).

---

## 6. Not for this box

- Items 10, 12, 13 (decode benchmark, iso-serving-cost board, Metal training-speed
  board) run on the M5 Pro with the Metal trainer, which also still lacks its corpus.
- Item 16 (damage-aware write policy) is a CPU diagnostic first;
  `MEMORY_UPDATE_PROTOCOL.md` specifies it.
- Item 17 (value-residual ladder with a long-stage control) is the sprint trainer on the
  RTX 3070 Ti.
- Item 8 (looped transformer with a learned halt) needs the controller implemented
  before it can be priced.

---

## 7. Things that will bite

- `launch --workers N` is per GPU; `gpu_bundle.py` uses a different convention and refuses
  two of its jobs per device.
- A relaunch into an existing dir with a **subset** of its arms is refused; always pass the
  full recorded arm list plus any new arm.
- `--seed` on `launch` filters the job list; the probe scripts use it for single-seed LR picks.
- `mqar_suite` at seq 511 needs `--workers 2` (GDN peaks near 29 GiB per job); the seq-255 board ran at 4 (`scripts/overnight.sh`, `scripts/longcell.sh`).
- The GH200 memory total the runner sees is 94.5 GiB, not 97,871 MiB.
- `smoke` writes to an isolated subtree; it is safe to run while a board is live, but it
  competes for the GPU, so run it before launching, not during.
- E22's log says `exit=1` because two `gpu_bundle` stages were refused for tenancy; the MoE
  stage in it completed. Do not "fix" that log.

---

## 8. Open decisions for Bharath (do not decide these yourself)

1. Whether `gdn_rule` should default to `published` after E28, which changes every future
   GDN run and the docs that describe the mixer.
2. Whether `moe_router_weight` should default to `raw`.
3. `docs/MUON_MUP_RULE_2026-09-05.md` recommends a `mup_muon_no_divisor` flag (default off)
   and a landing check at n=5. It touches `nanolab/optim.py:750` and every µP run; it is
   unrelated to the boards above and should not be bundled with them.
4. Whether to spend E35 rung 2 ($57 estimate, 1.6 epochs over the corpus).
5. Committing the uncommitted VRAM guard (§4.2).

---

## 9. Where everything is

| what | path |
|---|---|
| consolidated verdict and corrections | `docs/architecture-review-2026-09-04/README.md` |
| the memo whose §7 numbers the items above (version 3.1) | `docs/architecture-review-2026-09-04/inputs/04-efficiency-reconciliation-memo.md` |
| evidence review with the GDN and MoE findings | `docs/architecture-review-2026-09-04/EVIDENCE_REVIEW.md` |
| reproducible probes | `docs/architecture-review-2026-09-04/evidence/*.py` |
| suite runner, arm registry, recipe lock | `nanolab/crossover_replicate.py` |
| recall suite | `nanolab/mqar_suite.py`, `nanolab/mqar.py` |
| launch/chain patterns | `scripts/e18_loop.sh`, `scripts/e19b_moe_rerun.sh`, `scripts/e20_wcloop.sh`, `scripts/e27_ladder_w1536.sh`, `scripts/overnight.sh` |
| box deploy and artifact pull | `scripts/swaboard_launch.sh`, `scripts/pull_artifacts.sh` |
| corpus and hardware rules | `docs/GPU_BUNDLE.md` ("The corpus is part of the recipe"), `docs/ISSUES_AND_GAPS_2026-08-22.md` §3.4 |
| existing boards to compare against | `docs/LADDER_BOARD_2026-09-04.md`, `docs/MOE_BOARD_2026-09-04.md`, `docs/SWA_BOARD_2026-08-31.md`, `docs/EXPERIMENT_BACKLOG_2026-08-26.md` |
